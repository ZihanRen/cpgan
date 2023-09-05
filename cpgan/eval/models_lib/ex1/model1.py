from torch import nn
import torch
from torchinfo import summary
import math

class Generator(nn.Module):
  def __init__(self,z_dim=100,init_size=8,init_channel=256,img_channel=1,f_dim=1,embed_size=6):
    super(Generator,self).__init__()
    self.init_size = init_size
    self.init_channel = init_channel

    # layers between features input and z
    self.f1 = nn.Linear(f_dim,embed_size)
    self.f1batchnorm = nn.BatchNorm1d(embed_size)

    # layers between z and inital feature maps
    self.linear = nn.Linear(z_dim+embed_size,(init_size**3)*init_channel)
    self.batchnorm1d = nn.BatchNorm1d((init_size**3)*init_channel)
    self.lrelu = nn.LeakyReLU(0.2)

    # more feature maps to get final image output
    self.cnnt1 = self.make_gen_block(init_channel,init_channel//2) # 32
    self.cnnt2 = self.make_gen_block(init_channel//2,init_channel//4) # 64
    self.cnnt3 = self.make_gen_block(init_channel//4,init_channel//8) # 128
    self.cnnt4 = self.make_gen_block(init_channel//8,img_channel,final_layer=True) # 128
    
  def make_gen_block(self,input_channel,output_channel,kernel_size=4,padding=1,stride=2,final_layer=False):
    if not final_layer:
      return nn.Sequential(
          nn.ConvTranspose3d(input_channel,output_channel,kernel_size,stride=stride,padding=padding),
          nn.BatchNorm3d(output_channel),
          nn.LeakyReLU(0.2)
      )

    else:

      return nn.Sequential(
          nn.ConvTranspose3d(input_channel,output_channel,kernel_size,stride=stride,padding=padding),
          nn.Tanh()
      )

  def forward(self,z,f):

    assert f.shape[0] == z.shape[0]
    feature = self.f1(f.float())
    feature = self.f1batchnorm(feature)

    x = torch.cat((z,feature),dim=1)

    x = self.linear(x)
    x = self.batchnorm1d(x)
    x = self.lrelu(x)

    x = x.view(-1,self.init_channel,self.init_size,self.init_size,self.init_size)
    x = self.cnnt1(x)
    x = self.cnnt2(x)
    x = self.cnnt3(x)
    x = self.cnnt4(x)

    return x

# build discriminator model
class Discriminator(nn.Module):
  def __init__(self,img_channel=1,f_dim=1,img_size=128):
    super(Discriminator,self).__init__()

    self.init_filter = 1 + img_channel
    self.img_channel = img_channel
    self.fmap_size = img_size
    self.img_size = img_size

    self.f1 = nn.Linear(f_dim,img_size**3)
    self.f1batchnorm = nn.BatchNorm1d(img_size**3)  

    # layers initialization
    self.layers = nn.ModuleList()
    while self.fmap_size > 10:
      self.layers.append(nn.Conv3d(self.init_filter,self.init_filter*2,6,padding='same'))
      self.layers.append( nn.BatchNorm3d(num_features=self.init_filter*2) )
      self.layers.append(nn.LeakyReLU(0.2))
      self.layers.append(nn.AvgPool3d(6,stride=2))

      # recalculate the output image size
      self.fmap_size = self.img_size_cal(self.fmap_size,6,2)
      # update input fileter num
      self.init_filter *= 2

    
    flatsize = self.init_filter * (self.fmap_size**3)
    # add end flatten and linear layer
    self.layers.append(nn.Flatten(start_dim=1))
    self.layers.append(nn.Linear(flatsize,1))


  def img_size_cal(self,i_size,kernel_size,stride):
    o_size = (i_size - kernel_size)/stride + 1
    return math.floor(o_size)


  def forward(self,fake,f):

    assert f.shape[0] == fake.shape[0]
    feature = self.f1(f.float())
    feature = self.f1batchnorm(feature)
    feature = feature.view(-1,1,self.img_size,self.img_size,self.img_size)
    x = torch.cat((fake,feature),dim=1)

    n_layer = len(self.layers)
    for i in range(n_layer):
      x = self.layers[i] (x)

    return x
