# DFPD: Dual-Forgery Proactive Defense against Both Deepfakes and Traditional Image Manipulations

![Python 3.](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)

![PyTorch 1.10.0](https://img.shields.io/badge/pytorch-1.10.0-green.svg?style=plastic)

> **Abstract:** _Proactive defense against face forgery seeks to disrupt the output of forgery models by embedding imperceptible adversarial perturbations into face images to be protected. However, existing methods predominantly focus on deepfakes, often neglecting traditional image manipulations. It limits their practical applicability, as attackers may resort to traditional manipulations when deepfake attempts fail. To bridge this gap, a Dual-Forgery Proactive Defense (DFPD) method is proposed for combating both deepfakes and traditional image manipulations. For deepfake resistance, the DFPD designs a gradient-based ensemble adversarial attack that effectively disrupts outputs from multiple deepfake models. To defeat traditional manipulations, it also designs a fragile watermarking algorithm based on Invertible Neural Network (INN), enabling accurate localization of tampered regions. Furthermore, to mitigate the mutual interference between perturbation injection and watermark embedding, on the one hand, the DFPD adopts a serial pipeline starting with watermark embedding and then perturbation injection, which ensures that the injected perturbations are not displaced into residual image during INN-based embedding. On the other hand, a morphological post-processing module is introduced to eliminate adversarial noise in the tampering localization results. Extensive experiments validate the effectiveness of DFPD, demonstrating a 20.25% improvement in deepfake disruption over the best baseline in terms of PSNR and a 9.67% increase in traditional tampering localization in terms of ACC, while preserving high perceptual quality (32.75 dB PSNR)._
>

## Datasets
+ Considering that the SimSwap model requires face detection and alignment of faces, here the [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset is preprocessed in advance to obtain a cropped face dataset of size 256x256，and the preprocessed dataset can be downloaded by clicking [here](https://drive.google.com/file/d/1y4NQId6RvrjuUoWzGDbEFIVvN8LFo7pP/view?usp=sharing).
+ You can use the face detection and alignment methods from [insightface](https://github.com/deepinsight/insightface) for image preprocessing. Please download the relative files and unzip them to `./insightface_func/models` from [this link](https://onedrive.live.com/?authkey=%21ADJ0aAOSsc90neY&cid=4A83B6B633B029CC&id=4A83B6B633B029CC%215837&parId=4A83B6B633B029CC%215834&action=locate). Then change the path to the dataset to be preprocessed for the operation on the`preprocess.py` file and run this file again.

## Preparation
+ We apply Arcface to extract the identity of the target face for face swapping, please click [here](https://github.com/TreB1eN/InsightFace_Pytorch) to download the project to the current project directory.
+ Click on [checkpoints](https://drive.google.com/file/d/1f7Au2MpkvI5CyuMi_fTsHIOsztnf0ekv/view?usp=sharing) to download the weights of the 4 forgery models, the weights of HiNet, and the weights of the arcface model, and unzip them into the `./checkpoints/` directory.

```xml
checkpoints/
├── SimSwap/
│   |── arcface.pth
│   └── G_simswap.pth
├── hinet.pt
├── model_ir_se50.pth
├── 200000-G.ckpt
├── stargan.ckpt
└── G_latest.pth
```

## Train
Execute the following commands for training

```python
  python main.py
```

The results of the experiment are stored in the `./results/` directory.

## References
This work is based on [simswap](https://github.com/neuralchen/SimSwap) and [HiNet](https://github.com/TomTomTommi/HiNet).
