
# some codes copied from https://github.com/nashory/pggan-pytorch

import torch
import torch.nn as nn


# for equaliaeed-learning rate.
class EqualizedConv2d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride, pad):
        super(EqualizedConv2d, self).__init__()
        conv = nn.Conv2d(c_in, c_out, k_size, stride, pad)

        conv.weight.data.normal_()
        conv.bias.data.zero_()

        self.conv = equal_lr(conv)

    def forward(self, x):
        return self.conv(x)


class EqualizedLinear(nn.Module):
    def __init__(self, c_in, c_out):
        super(EqualizedLinear, self).__init__()
        linear = nn.Linear(c_in, c_out)

        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, x):
        return self.linear(x)
def pixel_norm(z):
  return z / (torch.mean(z**2,dim = 1, keepdim = True) + 1e-7) ** 0.5
class style_map(nn.Module):
  def __init__(self, in_style = 512, MLP_num = 8):
    super().__init__()
    layers = []
    for i in range(MLP_num):
      layers.append(EqualizedLinear(in_style, in_style))
      layers.append(nn.LeakyReLU(0.2))
    self.style_map = nn.Sequential(*layers)

  def forward(self, z):
    z = pixel_norm(z)
    return self.style_map(z)

class AdaIN(nn.Module):
  def __init__(self,in_style, in_channels):
    super().__init__()
    self.normalize = nn.InstanceNorm2d(in_channels)
    self.linear    = EqualizedLinear(in_style, in_channels * 2)
  def forward(self, x, style_map):
    mu, sig = self.linear(style_map).chunk(2, dim=1) ## mu,sig shape : [batch_size, in_channels]
    x   = self.normalize(x) # x shape : [batch_size, in_channels, height, width]
    x   = x * (mu[:,:,None,None] + 1) + sig[:,:,None,None]
    return x

def minibatch_stddev_layer(x, group_size = 4, num_new_features = 1):
  group_size = min(group_size, x.shape[0])
  y = x.reshape(group_size, -1, num_new_features, x.shape[1] //num_new_features, x.shape[2], x.shape[3])
  y = torch.sqrt(torch.mean((y - torch.mean(y, dim = 0, keepdim = True)) ** 2, dim = 0) + 1e-8) ## calc std
  y = torch.mean(y, dim =[2,3,4], keepdim = True)
  y = torch.squeeze(y, dim=2)
  y = y.repeat(group_size, 1, x.shape[2], x.shape[3])
  return torch.cat([x,y],dim = 1)
class inject_noise(nn.Module):
  def __init__(self,noise_channels):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(1, noise_channels, 1, 1)) ## Learnable weight

  def forward(self, x, noise):
    return x + self.weight * noise

class genSynthesis_block(nn.Module):
  def __init__(self, in_style, in_block, block_channels, first_block = False):
    super().__init__()
    self.first_block = first_block
    if first_block:
      self.const = nn.Parameter(torch.randn(1, 512, 4, 4))
    else:
      self.first_conv = EqualizedConv2d(in_block, block_channels, 3 , 1, 1)

    self.up_sample   = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners=False)
    self.inj_noise_1 = inject_noise(block_channels)
    self.leaky_relu_1= nn.LeakyReLU(0.2)
    self.AdaIN_1     = AdaIN(in_style, block_channels)
    self.conv        = EqualizedConv2d(block_channels, block_channels, 3, 1, 1)
    self.inj_noise_2 = inject_noise(block_channels)
    self.leaky_relu_2= nn.LeakyReLU(0.2)
    self.AdaIN_2     = AdaIN(in_style, block_channels) 
    self.to_rgb      = EqualizedConv2d(block_channels, 1, 1,1,0)

  def forward(self, x, w):
    if self.first_block:
      x = self.const.repeat(w.shape[0], 1, 1, 1) ## x shape: [batch_size, 512, 4, 4]
    else:
      x = self.up_sample(x)
      x = self.first_conv(x)
        
    noise =  torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) ## [batch_size, channels, height, width]
    x = self.inj_noise_1(x, noise)
    x = self.leaky_relu_1(x)
    x = self.AdaIN_1(x, w)

    x = self.conv(x)

    x = self.inj_noise_2(x, noise)
    x = self.leaky_relu_2(x)
    x = self.AdaIN_2(x, w)

    return x
class Dis_block(nn.Module):
  def __init__(self,in_block, block_channels, last_block = False):
    super().__init__()
    self.down_sample = nn.Upsample(scale_factor = 0.5, mode = 'bilinear', align_corners=False)
    self.last_block  = last_block
    if last_block:
      self.block = nn.Sequential(
                       EqualizedConv2d(in_block + 1, block_channels, 3, 1, 1),
                       nn.LeakyReLU(0.2),
                       EqualizedConv2d(block_channels, block_channels, 4, 1, 0),
                       nn.LeakyReLU(0.2),
                       nn.Flatten(),
                       EqualizedLinear(block_channels, 1)
                       )
    else:
      self.block = nn.Sequential(
                   EqualizedConv2d(in_block, block_channels, 3, 1, 1),
                   nn.LeakyReLU(0.2),
                   EqualizedConv2d(block_channels, block_channels, 3, 1, 1),
                   nn.LeakyReLU(0.2),
                    )
    self.from_rgb = EqualizedConv2d(1, in_block, 1, 1, 0)
    
  def forward(self, x):
    if self.last_block:
      x = minibatch_stddev_layer(x)
    x = self.block(x)
    if not self.last_block:
      x = self.down_sample(x)
    return x






class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module
