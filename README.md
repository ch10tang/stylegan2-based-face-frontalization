
# StyleGAN2-based Face Frontalization Model

Here are some examples made by our face frontalization model.
![Alt text](misc/Samples.png)



Pre-requisites
-- 
- python3
- CUDA 10.0 or higher
- Install the [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch) environment following the instruction.

Datasets
--
- Download the in-the-wild face dataset such as CAISA-WebFace, VGGFace2, and MS-Celeb-1M as the source set, and use any constrained (in-the-house) dataset as the normal set.
- All face images are normalized to 250x250 according to the five point landmarks, please refer to the align protocol in [LightCNN](https://github.com/AlfredXiangWu/LightCNN) for more details. 

Training and Inference 
--

**Preprocessing**

1. Change the directrory to **/FaceAlignment/** (*cd FaceAlignment*), and crop and align the input face images by running:
2. Pack the cropped data to the lmdb file by running:

    ```python prepare_data.py --out {LMDB file} --csv {the csv file}```

**Data Preparation**
1. Clone the Repository to preserve Directory Strcuture. 
2. Download the [face expert model](https://drive.google.com/drive/folders/1V7oMdPm2gmoBXKLsHrlzD0Gx2yAyk8qZ?usp=sharing), and put the model in **/Pretrained/VGGFace2/** directory. 
3. Train the face normalization model by running:

    ```python train.py --path_src {the source set} --path_norm {the target set}```


To-do list
--
- [x] Release the training code. 
- [ ] Release the evaluation code.
- [ ] Release the pre-trained model.

Acknowledgement
- The structures of the generator and the discriminator are borrowed from [StyleGAN2](https://arxiv.org/abs/1912.04958).
- The idea of MultiPIE-like face frontalization is borrowed by [FNM](https://openaccess.thecvf.com/content_CVPR_2019/papers/Qian_Unsupervised_Face_Normalization_With_Extreme_Pose_and_Expression_in_the_CVPR_2019_paper.pdf).

