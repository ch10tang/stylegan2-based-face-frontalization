
# Face Normalization Model
A PyTorch implementation of [Unsupervised Face Normalization with Extreme Pose and Expression in the Wild](http://openaccess.thecvf.com/content_CVPR_2019/papers/Qian_Unsupervised_Face_Normalization_With_Extreme_Pose_and_Expression_in_the_CVPR_2019_paper.pdf) from the paper by Qian, Yichen and Deng, Weihong and Hu, Jiani.

Here are some examples made by fnm.pytorch.
![Alt text](./imgs/Samples.png)



Pre-requisites
-- 
- python3
- CUDA 9.0 or higher
- Install [Pytorch](https://pytorch.org/?utm_source=Google&utm_medium=PaidSearch&utm_campaign=%2A%2ALP+-+TM+-+General+-+HV+-+TW&utm_adgroup=Install+PyTorch&utm_keyword=%2Binstall%20%2Bpytorch&utm_offering=AI&utm_Product=PyTorch&gclid=Cj0KCQjw1Iv0BRDaARIsAGTWD1uxAZX565HEO1i5eJJ9OE_mshYp7PJ6JBaVNUqZUln93a37cKlhSjUaAppiEALw_wcB) following the website. (or install w/ pip install torch torchvision)
- numpy 
- pillow 
- matplotlib 
- tensorboardX 
- pandas 
- scipy

Datasets
--
- Download face dataset such as CAISA-WebFace, VGGFace2, and MS-Celeb-1M as source set, and you can use any constrained (in-the-house) dataset as normal set.
- All face images are normalized to 250x250 according to landmarks. According to the five facial points, please follow the align protocol in [LightCNN](https://github.com/AlfredXiangWu/LightCNN). We also provide the crop code (MTCNN) as shown below.

Training and Inference 
--
1. Colone the Repository to preserve Directory Strcuture. 
2. Download the face expert model from [VGGFace2 Github](), and put the models in **/Pretrained/VGGFace2/** directory. 
3. Change the directrory to **/FaceAlignment/** (*cd FaceAlignment*), and crop and align the input face images by running:

    ```python face_align.py```
4. Train the face normalization model by running:

    ```python main.py -front-list {} -profile-list {}```
5. We also provide a simple test code, which can help to generate the normalized face and extract the features:

    ```python main.py -generate -gen-list {} -snapshot {your trained model}```

Note that, you need to define the csv files of source/normal/generate data roots during training/testing.


To-do list
--
- [x] Released the training code. 
- [x] Released the evaluation code.



I have made the following modification:




# StyleGAN 2 in PyTorch

Implementation of Analyzing and Improving the Image Quality of StyleGAN (https://arxiv.org/abs/1912.04958) in PyTorch

## Notice

I have tried to match official implementation as close as possible, but maybe there are some details I missed. So please use this implementation with care.

## Requirements

I have tested on:

* PyTorch 1.3.1
* CUDA 10.1/10.2

## Usage

First create lmdb datasets:

> python prepare_data.py --out LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... DATASET_PATH

This will convert images to jpeg and pre-resizes it. This implementation does not use progressive growing, but you can create multiple resolution datasets using size arguments with comma separated lists, for the cases that you want to try another resolutions later.

Then you can train model in distributed settings

> python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT train.py --batch BATCH_SIZE LMDB_PATH

train.py supports Weights & Biases logging. If you want to use it, add --wandb arguments to the script.

### Convert weight from official checkpoints

You need to clone official repositories, (https://github.com/NVlabs/stylegan2) as it is requires for load official checkpoints.

For example, if you cloned repositories in ~/stylegan2 and downloaded stylegan2-ffhq-config-f.pkl, You can convert it like this:

> python convert_weight.py --repo ~/stylegan2 stylegan2-ffhq-config-f.pkl

This will create converted stylegan2-ffhq-config-f.pt file.  

### Generate samples

> python generate.py --sample N_FACES --pics N_PICS --ckpt PATH_CHECKPOINT  

You should change your size (--size 256 for example) if you train with another dimension.   

### Project images to latent spaces

> python projector.py --ckpt [CHECKPOINT] --size [GENERATOR_OUTPUT_SIZE] FILE1 FILE2 ...

## Pretrained Checkpoints

[Link](https://drive.google.com/open?id=1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO)

I have trained the 256px model on FFHQ 550k iterations. I got FID about 4.5. Maybe data preprocessing, resolution, training loop could made this difference, but currently I don't know the exact reason of FID differences.

## Samples

![Sample with truncation](doc/sample.png)

At 110,000 iterations. (trained on 3.52M images)

### Samples from converted weights

![Sample from FFHQ](doc/stylegan2-ffhq-config-f.png)

Sample from FFHQ (1024px)

![Sample from LSUN Church](doc/stylegan2-church-config-f.png)

Sample from LSUN Church (256px)

## License

Model details and custom CUDA kernel codes are from official repostiories: https://github.com/NVlabs/stylegan2

Codes for Learned Perceptual Image Patch Similarity, LPIPS came from https://github.com/richzhang/PerceptualSimilarity

To match FID scores more closely to tensorflow official implementations, I have used FID Inception V3 implementations in https://github.com/mseitzer/pytorch-fid
# stylegan2-based-face-frontalization
