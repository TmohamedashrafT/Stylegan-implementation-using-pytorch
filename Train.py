from tqdm import tqdm
import torch.nn as nn
import warnings
import os
import torch
import argparse
from Data_loader import get_loader
from models import Discriminator, Generator
from utils import visualize_output, get_style_mixing, parse_yaml
warnings.filterwarnings("ignore")
class Training:
  def __init__(self,
               dataset_path,
               batch_size,
               MLP_num,
               in_style,
               channels,
               epochs,
               lr,
               ckpt_path,
               device,
               pretrained,
               max_scale,
               output_size = 8):
    self.device = device
    self.in_style = in_style
    self.loader = get_loader(dataset_path, batch_size)
    self.train_loader = self.loader.grow()
    self.dis    = Discriminator(channels).to(device)
    self.gen    = Generator(MLP_num,in_style , channels).to(device)
    self.lambda_gp   = 10
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.9,0.999))
    self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(0.9,0.999))
    self.grow_rank = 0
    self.ckpt_path = ckpt_path if ckpt_path[-3:] == '.pt' else os.path.join(ckpt_path, 'last.pt') 
    self.pretrained= pretrained
    self.lr = lr
    self.epochs  = epochs
    self.output_size = output_size
    self.max_scale = max_scale
    self.alpha = 0
    if self.pretrained:
      self.load_ckpts_train()
  def train_loop(self):
    trick = 1
    alpha_p = int(0.5 * self.epochs[self.grow_rank])
    for epoch in range(self.epochs[self.grow_rank]):
        dis_total_loss = 0
        gen_total_loss = 0
        for i,(real_imgs,_) in enumerate(tqdm(self.train_loader, ascii = True, desc ="Training")):
            self.dis_opt.zero_grad()
            images_len = len(real_imgs)
            real_imgs  = real_imgs.to(self.device)
            self.alpha = min(1,trick  / alpha_p * len(self.train_loader)) if self.grow_rank > 0 else 1
            z = torch.randn(images_len, self.in_style, device = self.device)
            style_mixing, z2 = get_style_mixing(images_len, self.in_style, self.device)

            fake_imgs = self.gen(z, self.alpha , style_mixing, z2)
            real_pred   = self.dis(real_imgs, self.alpha)
            fake_pred   = self.dis(fake_imgs.detach(),self.alpha)
            gradient_penalty = self.calculate_gradient_penalty(real_imgs, fake_imgs.detach())
            fake_pred_loss     = nn.functional.softplus(fake_pred).mean()  
            fake_pred_loss.backward()
            real_pred_loss     = nn.functional.softplus(- real_pred).mean() 
            real_pred_loss.backward()

            gradient_penalty_l = self.lambda_gp * gradient_penalty
            gradient_penalty_l.backward()
            self.dis_opt.step()

            dis_total_loss += fake_pred_loss  +  real_pred_loss + gradient_penalty_l
            self.dis_opt.step()
            self.dis_opt.zero_grad()
            z = torch.randn(images_len, self.in_style, device = self.device)
            style_mixing, z2 = get_style_mixing(images_len, self.in_style, self.device)
            fake_imgs = self.gen(z, self.alpha , style_mixing, z2)
            fake_pred   = self.dis(fake_imgs, self.alpha)
            gen_loss    =  nn.functional.softplus(-fake_pred).mean()
            gen_total_loss += gen_loss
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            self.gen_opt.zero_grad()
            trick +=1
        print('gpu_mem = ',round(torch.cuda.memory_reserved()/1E9,3),
              ' Discriminator loss = ',  round(dis_total_loss.item(),3),
              ' Generator loss = ',        round(gen_total_loss.item(),3),
              )
        self.save_model()


  def train(self):
    #self.grow()
    
    while True:
      2
      self.train_loop()
      print(f'The final samples from scale {self.grow_rank+1}')
      z = torch.randn(self.output_size , self.in_style, device = self.device)
      style_mixing, z2 = get_style_mixing(self.output_size , self.in_style, self.device)
      fake_imgs = self.gen(z, self.alpha , style_mixing, z2)
      visualize_output(fake_imgs, self.grow_rank)
      continue_training = self.grow()
      if self.grow_rank == self.max_scale:
         print('The maximum scale has been reached')
         break
      


  def calculate_gradient_penalty(self, real_images, fake_images):

    epsilon = torch.rand(len(real_images), 1, 1, 1, device = self.device, requires_grad=True)
    interpolated = epsilon * real_images + ((1 - epsilon) * fake_images)
    pred_interpolated = self.dis(interpolated,self.alpha)
    gradients = torch.autograd.grad(outputs = pred_interpolated, inputs = interpolated,
                               grad_outputs = torch.ones_like(pred_interpolated),
                               create_graph = True, retain_graph = True)[0]
    gradients = gradients.view(gradients.shape[0], -1)
    grad_penalty = ((gradients.norm(2, dim=1).mean() - 1) ** 2)
    return grad_penalty

  def load_ckpts_train(self,):
    ckpt    = torch.load(self.ckpt_path, map_location = self.device)
    for i in range(ckpt['grow_rank']):
      self.grow()
    self.gen.load_state_dict(ckpt['generator'])
    self.dis.load_state_dict(ckpt['discriminator'])
    self.gen_opt.load_state_dict(ckpt['generator_opt'])
    self.dis_opt.load_state_dict(ckpt['discriminator_opt'])
    self.grow_rank = ckpt['grow_rank']
    del ckpt
  def grow(self):
    torch.cuda.empty_cache()
    self.gen.grow()
    self.dis.grow()
    self.gen.cuda()
    self.dis.cuda()
    self.train_loader = self.loader.grow()
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr = self.lr, betas=(0.9,0.999))
    self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr = self.lr, betas=(0.9,0.999))
    self.grow_rank  += 1
  def save_model(self):
      ckpt = {'generator':self.gen.state_dict(),
            'discriminator':self.dis.state_dict(),
            'generator_opt' :self.gen_opt.state_dict(),
            'discriminator_opt':self.dis_opt.state_dict(),
            'grow_rank' : self.grow_rank,
            }
      torch.save(ckpt, self.ckpt_path)
      del ckpt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type = str, default = "config/style_config.yaml", help = "cfg.yaml")  
    parser.add_argument("--pretrained", type = bool, default = False, help = "Resume trainin")  
    parser.add_argument("--weights", type = str, default = None, help = "weights path")  
    parser.add_argument("--max-scale", type = int, default = 5, help = "max-scale")  

    opt = parser.parse_args()
    cfg = parse_yaml(opt.cfg)
    device   ='cuda' if torch.cuda.is_available() else 'cpu'
    cfg['ckpt_path'] = opt.weights if opt.weights else cfg['ckpt_path']
    training = Training(
              dataset_path =  cfg['dataset_path'],
              batch_size   =  cfg['batch_size'],
              MLP_num      =  cfg['MLP_num'],
              in_style     = cfg['in_style'],
              channels     =  cfg['channels'],
              epochs       =  cfg['epochs'],
              lr           = cfg['lr'],
              device       = device,
              ckpt_path    =  cfg['ckpt_path'],
              pretrained   =  opt.pretrained,
              max_scale    =  opt.max_scale)

    training.train()
    
    
    
    
    
    
    
