

import cv2
import torch
import fractions
import numpy as np
from PIL import Image
from torchvision import transforms
from insightface_func.face_detect_crop_single import Face_detect_crop
import os

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer_Arcface = transforms.Compose([
        # transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)
if __name__ == '__main__':
    # CelebA-HQ images path
    celeba_path = '/home/lab/workspace/dataset/CelebAMask-HQ/CelebA-HQ-img/'
    crop_save_path = './crop_256_/'
    ori_save_path = './ori_256/'

    all_files = os.listdir(celeba_path)


    start_epoch, epoch_iter = 1, 0
    crop_size = 256
    torch.nn.Module.dump_patches = True
    mode = 'None'


    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)
    j = 0
    # for i in range(1000):
    for filename in all_files:
        img_path = os.path.join(celeba_path, filename)

        ori_img = Image.open(img_path)
        img_a_whole = cv2.imread(img_path)

        if (app.get(img_a_whole, crop_size) == None):
            continue
        img_a_align_crop, _ = app.get(img_a_whole, crop_size)
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB))
        save_path = crop_save_path + '{}.png'.format(j)
        ori_path = ori_save_path + '{}.png'.format(j)
        j += 1
        img_a_align_crop_pil.save(save_path)
        ## 保存对应的原图
        ori_img.save(ori_path)
        
        print("save " + str(j) + " images")


