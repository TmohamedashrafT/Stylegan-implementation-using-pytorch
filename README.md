# Stylegan-implementation-using-pytorch
This repository contains code for training and generating Anime faces using StyleGAN on the Anime GAN Lite dataset
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
## Results
Scale 1 (4x4)

![image](https://github.com/TmohamedashrafT/Stylegan-implementation-using-pytorch/blob/main/images/Samples%20from%20scale%201.png)

Scale 2 (8x8)

![image](https://github.com/TmohamedashrafT/Stylegan-implementation-using-pytorch/blob/main/images/Samples%20from%20scale%202.png)

Scale 3 (16x16)

![image](https://github.com/TmohamedashrafT/Stylegan-implementation-using-pytorch/blob/main/images/Samples%20from%20scale%203.png)

Scale 4 (32x32)

![image](https://github.com/TmohamedashrafT/Stylegan-implementation-using-pytorch/blob/main/images/Samples%20from%20scale%204.png)

Scale 5 (64x64)

![image](https://github.com/TmohamedashrafT/Stylegan-implementation-using-pytorch/blob/main/images/Samples%20from%20scale%205.png)

Scale 6 (128x128)

![image](https://github.com/TmohamedashrafT/Stylegan-implementation-using-pytorch/blob/main/images/Samples%20from%20scale%206.png)
## StyleGan overview
The progressive GAN journey continues with advancements in StyleGAN, showcasing improvements in the Generator architecture. These enhancements provide higher diversity, fidelity, and increased control over image synthesis. The new architecture empowers users to generate high-resolution images with greater precision and customization. Additionally, it introduces two new metrics, namely Perceptual Path Length and Linear Separability, aimed at evaluating the disentanglement and quality of the generated images

### StyleGan architecture
In the StyleGAN architecture, the discriminator component is fundamentally similar to the discriminator used in the Progressive GAN (PGAN) model, but In the generator, There are a lot of modifications in the architecture.

![image](https://github.com/TmohamedashrafT/Stylegan-implementation-using-pytorch/blob/main/images/gen_architecture.png)

The first change in the generators is the Mapping network, which consists of 8 MLPs (Multi-layer Perceptrons) responsible for generating a new intermediate latent space. The main idea of this step is to achieve better disentanglement. The authors compared the traditional 'z' with the new intermediate latent space using different numbers of MLPs and found that the 8-layer configuration yields the best FID.

![image](https://github.com/TmohamedashrafT/Stylegan-implementation-using-pytorch/blob/main/images/Comparison.png)

Afterward, the new intermediate latent space is directed through an additional fully connected layer intended for a learned affine transformation (A), generating scale and bias parameters for injection into each layer of the synthesis network
#### AdaIN
Unlike ProGAN, StyleGAN employs Adaptive Instance Normalization (AdaIN) instead of pixel-wise normalization at each convolution. AdaIN normalizes individual channels, and the outcome of this normalization for each channel is multiplied by the 'A' scale and added to the 'A' bias obtained from the affine transformation

![image](https://github.com/TmohamedashrafT/Stylegan-implementation-using-pytorch/blob/main/images/AdaIN.png)

For stochastic variation, noise is injected with learnable weights after every convolution. The aim of this noise is to enhance diversity and refine minor details in the image, ultimately creating more realistic images

#### StyleMixing 
Style mixing in StyleGAN refers to a technique used to manipulate the visual features of generated images by combining different styles from different images during the generation process. This process involves blending or mixing the learned style vectors (latent codes) from two or more source images at various layers within the neural network architecture. By mixing styles, it becomes possible to create new images that possess characteristics or features from multiple input images, allowing for creative and diverse image synthesis in StyleGAN-based models.

Besides its role in regularization, this approach enables the manipulation of features across various stages during inference. The image below illustrates how style mixing impacts the application of the second latent space at different numbers of blocks. Adding the second latent space in the initial layers (Coarse style) induces significant changes, including pose, overall hairstyle, face shape, and eyeglasses. In intermediate layers (Middle style), changes are less pronounced, affecting facial features, hairstyle nuances, and eye openness/closure. However, in later layers (Fine style), alterations are minimal, primarily influencing the background and hair color.

![image](https://github.com/TmohamedashrafT/Stylegan-implementation-using-pytorch/blob/main/images/style%20mixing.webp)
