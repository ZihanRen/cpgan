#%%
from torch import nn
import torch
from torchinfo import summary
import math

def downsample_size_cal(input_dim,stride=1,padding=1,dilation=1,kernel_size=3):
  return (input_dim + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1

def downsample(input_dim):
  index = 0
  while input_dim > 10:
    input_dim = downsample_size_cal(input_dim)
    print(input_dim)
    index+=1
    if index == 20:
      break
    
# downsample(128)


#%%
# build discriminator model
class Discriminator(nn.Module):
  def __init__(self,img_channel=1,img_size=128,c_dim=2):
    super(Discriminator,self).__init__()

    self.init_filter = img_channel
    self.fmap_size = img_size
    self.c_dim = c_dim
    self.img_channel = img_channel

    # layers initialization
    self.base = nn.ModuleList()
    # add 3 conv layers as base layer for reprensetation extraction
    for i in range(3):
      self.base.append(
          nn.Conv3d(self.init_filter,self.init_filter*4,kernel_size=3,padding=1)
          )
      self.base.append(nn.BatchNorm3d(num_features=self.init_filter*4) )
      self.base.append(nn.LeakyReLU(0.2))
      self.base.append(nn.MaxPool3d(kernel_size=4,stride=2,padding=1))
      # update input fileter num
      self.init_filter *= 4

    # define Q network to extract latent features
    self.cnnq1 = nn.Sequential(
      nn.Conv3d(self.init_filter,self.init_filter,kernel_size=4,stride=2,padding=1), #16-8
      nn.BatchNorm3d(num_features=self.init_filter),
      nn.LeakyReLU(0.2)
    )

    self.cnnq2 = nn.Sequential(
      nn.Conv3d(self.init_filter,self.init_filter,kernel_size=4,stride=2,padding=1), #8-4
      nn.BatchNorm3d(num_features=self.init_filter),
      nn.LeakyReLU(0.2)
    )

    self.cnnq3 = nn.Sequential(
      nn.Conv3d(self.init_filter,self.init_filter,kernel_size=4,stride=2,padding=1), #4-2
      nn.BatchNorm3d(num_features=self.init_filter),
      nn.LeakyReLU(0.2)
  )

    self.cnnq4 = nn.Sequential(
      nn.Conv3d(self.init_filter,self.c_dim*2,kernel_size=4,stride=2,padding=1)
      ) # 2-1

    # define adversarial learning network
    self.cnnd1 = nn.Sequential(
      nn.Conv3d(self.init_filter,self.init_filter,kernel_size=4,stride=2,padding=1), #16-8
      nn.BatchNorm3d(num_features=self.init_filter),
      nn.LeakyReLU(0.2)
    )

    self.cnnd2 = nn.Sequential(
      nn.Conv3d(self.init_filter,self.init_filter,kernel_size=4,stride=2,padding=1), #8-4
      nn.BatchNorm3d(num_features=self.init_filter),
      nn.LeakyReLU(0.2)
    )

    self.cnnd3 = nn.Sequential(
      nn.Conv3d(self.init_filter,self.init_filter,kernel_size=4,stride=2,padding=1), #4-2
      nn.BatchNorm3d(num_features=self.init_filter),
      nn.LeakyReLU(0.2)
  )

    self.cnnd4 = nn.Sequential(
      nn.Conv3d(self.init_filter,self.img_channel,kernel_size=4,stride=2,padding=1)
      ) # 2-1


  def forward(self,input_img):

    for i in range(len(self.base)):
      input_img = self.base[i] (input_img)
    
    # auxillary network Q for latent code prediction
    q_x = self.cnnq1(input_img)
    q_x = self.cnnq2(q_x)
    q_x = self.cnnq3(q_x)
    q_x = self.cnnq4(q_x)


    # discriminator netowork D for adversarial loss
    d_x = self.cnnd1(input_img)
    d_x = self.cnnd2(d_x)
    d_x = self.cnnd3(d_x)
    d_x = self.cnnd4(d_x)

    return d_x.view(len(d_x),-1), q_x.view(len(q_x),-1)

#%%
if __name__ == "__main__":
    dis = Discriminator(c_dim=2)
    c= dis(torch.rand(10,1,128,128,128))
    batch_size = 5
    
    print( 'The discriminator architecture is'+'\n{}'.format(
       summary(dis,(batch_size,1,128,128,128)) 
       ))

# %%
