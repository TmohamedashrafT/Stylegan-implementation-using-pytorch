import torch
import torch.nn as nn
from custom_layers import genSynthesis_block, Dis_block, style_map
class Generator(nn.Module):
  def __init__(self, MLP_num, in_style, channels, in_channels):
    super().__init__()
    self.in_style   = in_style
    self.style_map  = style_map(in_style, MLP_num)
    self.gen_blocks = nn.ModuleList([genSynthesis_block(in_style, channels[0], channels[1], first_block = True)])
    self.channels   = channels
    self.num_of_blocks = 1
  def forward(self, z, alpha = 1, style_mixing = None, z2 = None):
    gen = None
    assert style_mixing == None or z2 != None,'assign values to z2'
    style_mixing = style_mixing if style_mixing else self.num_of_blocks
    w   = self.style_map(z)
    if z2 is not None:
      w2 = self.style_map(z2)
    w = [w for _ in range(style_mixing)] + [w2 for _ in range(abs(style_mixing - self.num_of_blocks))]

    if self.num_of_blocks > 1:
      for block_i in range(self.num_of_blocks - 1):
         gen = self.gen_blocks[block_i](gen, w[block_i])
      up_sampled = self.gen_blocks[-2].to_rgb(self.gen_blocks[-2].up_sample(gen))
      gen  = alpha * self.gen_blocks[-1].to_rgb(self.gen_blocks[-1](gen, w[-1])) + (1 - alpha) * up_sampled
    else:
      gen = self.gen_blocks[-1].to_rgb(self.gen_blocks[-1](gen,w[-1]))

    return gen
  def grow(self,):
    self.gen_blocks.append(genSynthesis_block(self.in_style, self.channels[self.num_of_blocks], self.channels[self.num_of_blocks + 1]))
    self.num_of_blocks += 1

    
 class Discriminator(nn.Module):
  def __init__(self, channels, in_channels):
    super().__init__()
    self.channels   = channels
    self.dis_blocks = nn.ModuleList([Dis_block(channels[1], channels[0], last_block = True)])
    self.num_of_blocks = 1
  def forward(self, x, alpha = 0.2):
    input = x
    x = self.dis_blocks[-1].from_rgb(x)
    x = self.dis_blocks[-1](x)
    if self.num_of_blocks > 1:
      down_sampled_rgb = self.dis_blocks[-1].down_sample(input)
      down_sampled = self.dis_blocks[-2].from_rgb(down_sampled_rgb)
      x = alpha * x + (1 - alpha) * down_sampled
      for block_i in reversed(range(0, self.num_of_blocks - 1)):
        x = self.dis_blocks[block_i](x)
    return x
  def grow(self,):
    self.dis_blocks.append(Dis_block(self.channels[self.num_of_blocks + 1], self.channels[self.num_of_blocks]))
    self.num_of_blocks += 1