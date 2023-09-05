from torch import nn
import torch
from torchinfo import summary
import math

#%% shape check for upsample layer design
def upsample_size_cal(input_dim,stride=2,padding=1,dilation=1,kernel_size=4,output_padding=0):
  return (input_dim-1)*stride - 2*padding + dilation * (kernel_size-1) + output_padding+1

def upsample(input_dim):
  while input_dim < 128:
    input_dim = upsample_size_cal(input_dim)
    print(input_dim)

# upsample(1)

#%%
class Generator(nn.Module):
  def __init__(self,z_dim=128,img_chan=1,hidden_dim=2):
    super(Generator,self).__init__()
    self.img_chan = img_chan
    self.hidden_dim = hidden_dim
    self.z_dim = z_dim

    # more feature maps to get final image output
    self.cnnt1 = self.make_gen_block(self.z_dim,self.hidden_dim*64) # 1-2
    self.cnnt2 = self.make_gen_block(self.hidden_dim*64,self.hidden_dim*32) # 2-4
    self.cnnt3 = self.make_gen_block(self.hidden_dim*32,self.hidden_dim*16) # 4-8
    self.cnnt4 = self.make_gen_block(self.hidden_dim*16,self.hidden_dim*8) # 8-16
    self.cnnt5 = self.make_gen_block(self.hidden_dim*8,self.hidden_dim*4) # 16-32
    self.cnnt6 = self.make_gen_block(self.hidden_dim*4,self.hidden_dim) # 32-64
    self.cnnt7 = self.make_gen_block(self.hidden_dim,self.img_chan) # 64-128
    
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

  def forward(self,z):
    z = z.view(len(z),self.z_dim,1,1,1)
    x = self.cnnt1(z)
    x = self.cnnt2(x)
    x = self.cnnt3(x)
    x = self.cnnt4(x)
    x = self.cnnt5(x)
    x = self.cnnt6(x)
    x = self.cnnt7(x)

    return x


if __name__ == "__main__":
    # check the structure of generator
    z_dim = 20
    gen = Generator(z_dim=z_dim,hidden_dim=8)
    c = gen(torch.rand(10,20))
    print(c.shape)
    batch_size = 5
    print( 'The generator architecture is'+'\n{}'.format(
       summary(gen,(batch_size,z_dim)) 
       ))
