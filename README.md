# Stylegan-implementation-using-pytorch
This repository contains code for training and generating fashion images using StyleGAN on the FashionMNIST dataset
## Installation
1.Clone the repository
```
git clone https://github.com/TmohamedashrafT/Stylegan-implementation-using-pytorch.git
```
2.Train
```
%cd Stylegan-implementation-using-pytorch
python Train.py --dataset_path 'dataset_path' --weights_path 'weights_path'
```

## StyleGan overview
The progressive GAN journey continues with advancements in StyleGAN, showcasing improvements in the Generator architecture. These enhancements provide higher diversity, fidelity, and increased control over image synthesis. The new architecture empowers users to generate high-resolution images with greater precision and customization. Additionally, it introduces two new metrics, namely Perceptual Path Length and Linear Separability, aimed at evaluating the disentanglement and quality of the generated images

### StyleGan architecture
In the StyleGAN architecture, the discriminator component is fundamentally similar to the discriminator used in the Progressive GAN (PGAN) model, but In the generator, There are a lot of modifications in the architecture.

![image](https://github.com/TmohamedashrafT/Stylegan-implementation-using-pytorch/blob/main/images/gen_architecture.png)

The first change in the generators is the Mapping network, which consists of 8 MLPs (Multi-layer Perceptrons) responsible for generating a new intermediate latent space. The main idea of this step is to achieve better disentanglement. The authors compared the traditional 'z' with the new intermediate latent space using different numbers of MLPs and found that the 8-layer configuration yields the best FID.

![image](https://github.com/TmohamedashrafT/Stylegan-implementation-using-pytorch/blob/main/images/Comparison.png)

Afterward, the new intermediate latent space is directed through an additional fully connected layer intended for a learned affine transformation (A), generating scale and bias parameters for injection into each layer of the synthesis network

Unlike ProGAN, StyleGAN employs Adaptive Instance Normalization (AdaIN) instead of pixel-wise normalization at each convolution. AdaIN normalizes individual channels, and the outcome of this normalization for each channel is multiplied by the 'A' scale and added to the 'A' bias obtained from the affine transformation

![image](https://github.com/TmohamedashrafT/Stylegan-implementation-using-pytorch/blob/main/images/AdaIN.png)


