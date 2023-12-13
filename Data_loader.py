from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
class get_loader:
  def __init__(self, dataset_path, batch_size, in_channels = 3):
    super().__init__()
    self.dataset_path = dataset_path
    self.img_size     = 4
    self.batch_size   = batch_size
    self.dataset      = 0
    self.in_channels  = in_channels

  def __len__(self):
    return len(self.dataset)
  def grow(self,):
    dataset_transforms= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*self.in_channels, (0.5,)*self.in_channels),
            ])
    
    self.dataset = ImageFolder(root = self.dataset_path, transform = dataset_transforms)
    
    train_loader = DataLoader(
    self.dataset, batch_size = self.batch_size[self.img_size], shuffle=True, drop_last=True, num_workers=0,
                             )
    ## drop_last = True, becouse of minibatch_stddev_layer the input batch size should be Divisible

    self.img_size *= 2 ## 4 8 16 32 64 128 256
    return train_loader
