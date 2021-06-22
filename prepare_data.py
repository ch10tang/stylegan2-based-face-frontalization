import argparse
import os
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn
from torch.utils.data.dataset import Dataset  # For custom datasets
import pandas as pd
import numpy as np


class CustomDatasetFromImages(Dataset):
    def __init__(self, dataset_path=None, csv_path=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.dataset_path = dataset_path
        self.data_info = pd.read_csv(csv_path, header=0)
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])


    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        img_as_img = os.path.join(self.dataset_path, *single_image_name.split('/'))

        return img_as_img


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format='jpeg', quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(img_file, sizes, resample):
    i, file = img_file
    img = Image.open(file)
    img = img.convert('RGB')
    out = resize_multiple(img, sizes=sizes, resample=resample)

    return i, out, file


def resize_worker_path(img_file, sizes, resample):
    i, file, label = img_file
    img = Image.open(file)
    img = img.convert('RGB')
    out = resize_multiple(img, sizes=sizes, resample=resample)

    return i, out, file, label

def prepare_path(env, files, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS):
    resize_fn = partial(resize_worker_path, sizes=sizes, resample=resample)

    files = [(i, file, label) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs, label in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):

                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')
                with env.begin(write=True) as txn:
                    txn.put(key, img)
                
                label_key = f'{size}-label-{str(i).zfill(5)}'.encode('utf-8')
                with env.begin(write=True) as txn:
                    txn.put(label_key, str(label).encode('utf-8'))

            total += 1

        with env.begin(write=True) as txn:
            txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))

def prepare_csv(env, files, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)

    files = [(i, file) for i, file in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs, name in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):

                # image
                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')
                with env.begin(write=True) as txn:
                    txn.put(key, img)
                name_key = f'{size}-name-{str(i).zfill(5)}'.encode('utf-8')
                with env.begin(write=True) as txn:
                    txn.put(name_key, str(name).encode('utf-8'))
                
                # # label
                # label_key = f'{size}-label-{str(i).zfill(5)}'.encode('utf-8')
                # with env.begin(write=True) as txn:
                #     txn.put(label_key, str(label).encode('utf-8'))
                #
                #
                # label2_key = f'{size}-label2-{str(i).zfill(5)}'.encode('utf-8')
                # with env.begin(write=True) as txn:
                #     txn.put(label2_key, str(label2).encode('utf-8'))

            total += 1

        with env.begin(write=True) as txn:
            txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='../../Database/FFHQ_FOCropped_512_512_168')
    parser.add_argument('--size', type=str, default='256')
    parser.add_argument('--n_worker', type=int, default=6)
    parser.add_argument('--resample', type=str, default='lanczos')
    parser.add_argument('--csv', type=str, default='./FFHQ_FOCropped_512_512_168.csv')
    parser.add_argument('--path', type=str, default='../../Database')
    parser.add_argument('--G', type=float, default=5)

    args = parser.parse_args()
    
    resample_map = {'lanczos': Image.LANCZOS, 'bilinear': Image.BILINEAR}
    resample = resample_map[args.resample]
    
    sizes = [int(s.strip()) for s in args.size.split(',')]

    print(f'Make dataset of image sizes:', ', '.join(str(s) for s in sizes))

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    if args.csv:
        imgset = CustomDatasetFromImages(args.path, args.csv)
        files = sorted([x for x in imgset], key=lambda x: x[0])
        with lmdb.open(args.out, map_size=(1024 ** 3) * args.G, readahead=False, writemap=True) as env:
            prepare_csv(env, files, args.n_worker, sizes=sizes, resample=resample)
    else:
        imgset = datasets.ImageFolder(args.path)
        files = sorted(imgset.imgs, key=lambda x: x[0])
        with lmdb.open(args.out, map_size=(1024 ** 3) * args.G, readahead=False, writemap=True) as env:
            prepare_path(env, files, args.n_worker, sizes=sizes, resample=resample)


