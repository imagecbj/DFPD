""" Basic configuration and settings for training the model"""
import torch
import os
## HiNet set
# Super parameters
clamp = 2.0
channels_in = 3
log10_lr = -4.5
lr = 10 ** log10_lr
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 5
lamda_guide = 1
lamda_low_frequency = 1
device_ids = [0]






# specify protection model architecture [resnet_9blocks, resnet_6blocks, resnet_2blocks, resnet_1blocks, unet_32, unet_64, unet_128, unet_256]
net_noise = 'unet_64'

# of gen filters in the last conv layer
ngf = 64

# instance normalization or batch normalization [instance | batch | none]
norm = 'instance'

# network initialization [normal | xavier | kaiming | orthogonal]
init_type = 'normal'

# scaling factor for normal, xavier and orthogonal.
init_gain = 0.02

no_dropout = False

input_nc = 3
output_nc = 3

# running device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


conv_dim = 64
c_dim = 5
repeat_num = 6
mid_layer = 8



# attrs
attrs = [
        "Bald",
        "Bangs",
        "Black_Hair",
        "Blond_Hair",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Eyeglasses",
        "Male",
        "Mouth_Slightly_Open",
        "Mustache",
        "No_Beard",
        "Pale_Skin",
        "Young"
    ]


## HiNet Path
Hinet_path = './checkpoints/hinet.pt'
## simswap path
BASE_DIR = './checkpoints/'

# Pretrained Checkpoints for manipulation models
simswap_ckpt = os.path.join(BASE_DIR,  'SimSwap', 'G_simswap.pth') # 256
simswap_arcface_ckpt = os.path.join(BASE_DIR,  'SimSwap', 'arcface.pth')
perturb_wt = 50
loss_type = 'l2'

## fs path
fs_path = './checkpoints/G_latest.pth'
arcface_path = './checkpoints/model_ir_se50.pth'

## stargan
starganPath = "./checkpoints/stargan.ckpt"

## fgan
fgan_path = "./checkpoints/200000-G.ckpt"

## dataset
dataset_path = '/home/lab/workspace/works/hyt/img_test_256/'