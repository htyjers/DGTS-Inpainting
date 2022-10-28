# DGTS

This repository is the official code for the paper "Delving Globally into Texture and Structure for Image Inpainting" by Haipeng Liu, Yang Wang (corresponding author: yangwang@hfut.edu.cn), Meng Wang, Yong Rui. *ACM Multimedia 2022, Lisbon, Portugal*
#
## Introduction
In this paper, we delve globally into texture and structure information to well capture the semantics for image inpainting. Unlike the current decoder-only transformer within the pixel level for image inpainting, our model adopts the transformer pipeline paired with both encoder and decoder. On one hand, the encoder captures the texture semantic correlations of all patches across image via self-attention module. On the other hand, an adaptive patch vocabulary is dynamically established in the decoder for the filled patches over the masked regions. Building on this,  a structure-texture matching attention module (**_Eq.5 and 6_**) anchored on the known regions comes up to marry the best of these two worlds for progressive inpainting via a probabilistic diffusion process (**_Eq.8_**). Our model is orthogonal to the fashionable arts, such as Convolutional Neural Networks (CNNs), Attention and Transformer model, from the perspective of texture and structure information for image inpainting.

![](https://github.com/htyjers/DGTS-Inpainting/blob/DGTS/images/model.png)
<p align="center">Figure 1. Illustration of the proposed transformer pipeline.</p>

In summary, our contributions are summarized below:
- We propose a transformer pipeline paired with both encoder and decoder, where the encoder module aims at capturing the semantic correlations of the whole images within texture references, leading to a *global* texture reference set; we design a coarse filled attention module to exploit all the known image patches to fill in the masked regions, yielding a *global* structure information.
- To endow the decoder with the capacity of marring the best of the two worlds, *i.e.*, global texture reference and structure information. we equip the decoder with a structure-texture matching attention module via an intuitive attention transition manner, where  an adaptive patch vocabulary is dynamically established for the filled patches over the masked regions via a probabilistic diffusion process.
- To ease the computational burden, we disclose several training tricks to overcome memory overhead for GPUs.

![](https://github.com/htyjers/DGTS-Inpainting/blob/DGTS/images/bridge.png)
<p align="center">Figure 2.  Intuition of the bridge module.</p>



#
## Run 
0. Requirements
```
Python >= 3.6
PyTorch >= 1.0
NVIDIA GPU + CUDA cuDNN
scikit-image
scipy
opencv-python
matplotlib
```


1. To train the proposed model described in the paper. Prepare training datasets and put them in ```./data/places2/train```, then run the following command:
```
Run "Python3 /DGTS/code/train/run_train.py"
```


2. To inpaint the masked images, Prepare testing datasets and put them in ```./data/places2/test```, then run the following command:
```
Run "Python3 /DGTS/code/test/run_train.py"
```
* *The code and pre-trained models of upsampling network borrows heavily from [here](https://github.com/yingchen001/BAT-Fill), we apprecite the authors for sharing their codes.*


3. Please download the pre-trained model of [Places2](https://www.dropbox.com/s/jipius8hwcr3795/places.pth?dl=0). 
* *You can set the path of our pre-trained model in [here](https://github.com/htyjers/DGTS-Inpainting/blob/DGTS/code/test/test.py#L106) and the path of upsampling pre-trained model in [here](https://github.com/htyjers/DGTS-Inpainting/blob/DGTS/code/test/test.py#L100).*

4. Following the previous work, the input images randomly match masks which adopted from the widely used [irregular mask dataset](https://nv-adlr.github.io/publication/partialconv-inpainting) to generate the masked images.



#
## Example Results

- Visual comparison between our method and the competitors.

![](https://github.com/htyjers/DGTS-Inpainting/blob/DGTS/images/compare.png)

- Attention maps of exemplar texture references.

![](https://github.com/htyjers/DGTS-Inpainting/blob/DGTS/images/correct.png)


#
## Citation

If any part of our paper and repository is helpful to your work, please generously cite with:

```
@inproceedings{liu2022delving,
  title={Delving Globally into Texture and Structure for Image Inpainting},
  author={Liu, Haipeng and Wang, Yang and Wang, Meng and Rui, Yong},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={1270--1278},
  year={2022}
}
```
