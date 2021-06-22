import argparse
import math
import random
import os
from Model.model_irse import IR_50
from Model.model_lightcnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from Model.resnet50_ft_dims_2048 import resnet50_ft
from util.InputSize_Select import TrainingSize_Select
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from util.LoadPretrained import LoadPretrained
import datetime
import shutil
import numpy as np
import cv2
from PIL import Image

try:
    import wandb

except ImportError:
    wandb = None

from model import Generator, Discriminator
from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

loss_criterion_L1 = nn.L1Loss(reduction='mean').cuda()

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def feature_loss(f_in, f_tgt):
    epsilon = 1e-9

    f_in = torch.div(f_in, (f_in.norm(2, dim=1, keepdim=True) + epsilon))
    f_tgt = torch.div(f_tgt, (f_tgt.norm(2, dim=1, keepdim=True) + epsilon))

    loss = (1 - torch.mul(f_in, f_tgt).sum(1)).sum()

    return loss

def L1Loss(input, target):

    Loss = loss_criterion_L1(input, target)  # L1Loss(input, target)

    return Loss

def SymLoss(input):

    Loss = loss_criterion_L1(input, torch.flip(input, [3]))  # L1Loss(input, flipped)

    return Loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, loader_src, loader_norm, generator, discriminator, ExpertModel, g_optim, d_optim, g_ema, device):

    # Save Path
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ImgSavePath = 'sample/{}'.format(date)
    CheckpointSavePath = 'checkpoint/{}'.format(date)
    if not os.path.exists(ImgSavePath): os.makedirs(ImgSavePath)
    if not os.path.exists(CheckpointSavePath): os.makedirs(CheckpointSavePath)
    shutil.copy('./train.py', './{}/train.py'.format(CheckpointSavePath))
    shutil.copy('./model.py', './{}/model.py'.format(CheckpointSavePath))

    # Reference Constant Map
    ConstantMap = Image.open('./ReferenceMap/{}.png'.format(args.mean_face))
    ConstantMap = transforms.ToTensor()(ConstantMap).unsqueeze_(0).to(device).repeat(args.batch, 1, 1, 1)


    loader_src = sample_data(loader_src)
    loader_norm = sample_data(loader_norm)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0
    r1_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))


    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader_src) # source set
        tgt_img = next(loader_norm) # normal set
        real_img = real_img.to(device)
        tgt_img = tgt_img.to(device)

        #################################### Train discrimiantor ####################################
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        Profile_Fea, _ = ExpertModel(TrainingSize_Select(real_img, args), args)
        Profile_Syn_Img, _ = generator(Profile_Fea, ConstantMap)
        Front_Fea, _ = ExpertModel(TrainingSize_Select(tgt_img, args), args)
        Front_Syn_Img, _ = generator(Front_Fea, ConstantMap)

        Profile_Syn_Pred = discriminator(Profile_Syn_Img)
        Front_Syn_Pred = discriminator(Front_Syn_Img)
        Real_Pred = discriminator(tgt_img)
        d_loss = (d_logistic_loss(Real_Pred, Profile_Syn_Pred) + d_logistic_loss(Real_Pred, Front_Syn_Pred)) / 2

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = Real_Pred.mean()
        loss_dict["profile_fake_score"] = Profile_Syn_Pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
            d_optim.step()

        loss_dict["r1"] = r1_loss

        #################################### Train generator ####################################
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        Front_Fea, Front_Map = ExpertModel(TrainingSize_Select(tgt_img, args), args)
        Front_Syn_Img, _ = generator(Front_Fea, ConstantMap)
        Front_Syn_Pred = discriminator(Front_Syn_Img)
        Front_Syn_Fea, _ = ExpertModel(TrainingSize_Select(Front_Syn_Img, args), args)

        Profile_Fea, _ = ExpertModel(TrainingSize_Select(real_img, args), args)
        Profile_Syn_Img, _ = generator(Profile_Fea, ConstantMap)
        Profile_Syn_Pred = discriminator(Profile_Syn_Img)
        Profile_Syn_Fea, _ = ExpertModel(TrainingSize_Select(Profile_Syn_Img, args), args)

        adv_g_loss = (g_nonsaturating_loss(Profile_Syn_Pred) + g_nonsaturating_loss(Front_Syn_Pred)) / 2
        fea_loss = (feature_loss(Profile_Syn_Fea[0], Profile_Fea[0]) + feature_loss(Front_Syn_Fea[0], Front_Fea[0])) / 2
        sym_loss = (SymLoss(Front_Syn_Img) + SymLoss(Profile_Syn_Img)) / 2
        L1_loss = L1Loss(Front_Syn_Img, tgt_img)
        g_loss = args.lambda_adv * adv_g_loss + args.lambda_fea * fea_loss + args.lambda_sym * sym_loss + args.lambda_l1 * L1_loss

        loss_dict["g"] = g_loss
        loss_dict["adv_g_loss"] = args.lambda_adv * adv_g_loss
        loss_dict["fea_loss"] = args.lambda_fea * fea_loss
        loss_dict["symmetry_loss"] = args.lambda_sym * sym_loss
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            noise, _ = ExpertModel(TrainingSize_Select(real_img, args), args)
            fake_img, latents = generator(noise, ConstantMap, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        fea_loss_val = loss_reduced["fea_loss"].mean().item()
        sym_loss_val = loss_reduced["symmetry_loss"].mean().item()

        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        profile_fake_score_val = loss_reduced["profile_fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()


        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g_total: {g_loss_val:.4f}; fea: {fea_loss_val:.4f}; sym: {sym_loss_val:.4f}; r1: {r1_val:.4f};"
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Profile Score": profile_fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    pro_fea, _ = ExpertModel(TrainingSize_Select(real_img, args), args)
                    pro_syn, _ = g_ema(pro_fea, ConstantMap)
                    tgt_fea, _ = ExpertModel(TrainingSize_Select(tgt_img, args), args)
                    tgt_syn, _ = g_ema(tgt_fea, ConstantMap)

                    result = torch.cat([real_img, pro_syn, tgt_img, tgt_syn], 2)
                    utils.save_image(
                        result,
                        f"{ImgSavePath}/{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 2000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                    },
                    f"{CheckpointSavePath}/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    parser.add_argument("--path_src", type=str, default='../../Database/LMDB/Test_CASIA_LMDB')
    parser.add_argument("--path_norm", type=str, default='../../Database/LMDB/MPIE_FOCropped_051deg_512_512_168')
    parser.add_argument("--mean_face", type=str, default='MeanFace_Frontal_Gray')
    parser.add_argument("--iter", type=int, default=300000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--n_sample", type=int, default=64)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing", type=float, default=0.9)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--model_select", type=str, default='VGGFace2', help='Model Select')
    # Hyperparameter Settings
    parser.add_argument("--lambda_adv", type=float, default=1)
    parser.add_argument("--lambda_fea", type=float, default=3)
    parser.add_argument("--lambda_l1", type=float, default=0.1)
    parser.add_argument("--lambda_sym", type=float, default=0.01)
    # Architecture Settings
    parser.add_argument("--translayer", type=int, default=4)
    parser.add_argument("--mapsize", type=int, default=4)

    args = parser.parse_args()


    BACKBONE_DICT = {'IR-50': IR_50(112),
                 'Light_CNN_9': LightCNN_9Layers(),
                 'Light_CNN_29': LightCNN_29Layers(),
                 'Light_CNN_29_v2': LightCNN_29Layers_v2(),
                 'VGGFace2': resnet50_ft(weights_path='Pretrained/VGGFace2/resnet50_ft_dims_2048.pth')
    }
    BACKBONE = BACKBONE_DICT[args.model_select]
    ExpertModel = LoadPretrained(BACKBONE, args)


    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    if args.model_select.startswith('IR-50'): args.latent = 512
    elif args.model_select.startswith('VGGFace2'): args.latent = 2048
    args.n_mlp = 8

    args.start_iter = 0

    generator = Generator(args.size, args.latent, args, channel_multiplier=args.channel_multiplier).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)
    g_ema = Generator(args.size, args.latent, args, channel_multiplier=args.channel_multiplier).to(device)
    ExpertModel = ExpertModel.to(device)
    g_ema.eval()
    ExpertModel.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset_src = MultiResolutionDataset(args.path_src, transform, args.size)
    loader_src = data.DataLoader(
        dataset_src,
        batch_size=args.batch,
        sampler=data_sampler(dataset_src, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    dataset_norm = MultiResolutionDataset(args.path_norm, transform, args.size)
    loader_norm = data.DataLoader(
        dataset_norm,
        batch_size=args.batch,
        sampler=data_sampler(dataset_norm, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    train(args, loader_src, loader_norm, generator, discriminator, ExpertModel, g_optim, d_optim, g_ema, device)
