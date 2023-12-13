import torchvision.utils as vutils
import numpy as np
import random
import torch
import yaml
import matplotlib.pyplot as plt
def visualize_output(fake_imgs, scale):
  plt.figure(figsize=(10,10))
  plt.imshow(np.transpose(vutils.make_grid(fake_imgs.detach().cpu(), normalize=True),(1,2,0)))
  plt.savefig(f"Samples from scale {scale}.png") 

def get_style_mixing(batch_size, in_style, device, num_layers = 1, prop = 0.8):
  style_mixing, z = (None, None) if random.random() > prop else (num_layers, torch.randn(batch_size, in_style, device = device))
  return style_mixing, z

def parse_yaml(path):
    with open(path, "r") as ya:
        try:
            config = yaml.safe_load(ya)
            return config
        except yaml.YAMLError as exc:
            print(exc)
def grow_test(gen, dis, loader):
    gen.grow()
    dis.grow()
    gen.cuda()
    dis.cuda()
    train_loader = loader.grow()
    
def load_ckpts_test(ckpt_path, gen, dis, loader):
    ckpt    = torch.load(self.ckpt_path + 'last.pt', map_location = 'cpu')
    for i in range(ckpt['grow_rank']):
        gen, dis, train_loader = grow(gen, dis, loader)
    gen.load_state_dict(ckpt['generator'])
    dis.load_state_dict(ckpt['discriminator'])
    return gen, dis, train_loader
