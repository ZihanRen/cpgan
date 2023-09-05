from torch import nn
import torch
from torchinfo import summary

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
  def __init__(self,z_dim=202,init_size=8,init_channel=256):
    super(Generator,self).__init__()
    self.init_size = init_size
    self.init_channel = init_channel

    # layers between z and inital feature maps
    self.linear = nn.Linear(z_dim,(init_size**3)*init_channel)
    self.batchnorm1d = nn.BatchNorm1d((init_size**3)*init_channel)
    self.lrelu = nn.LeakyReLU(0.2)

    # more feature maps to get final image output
    self.cnnt1 = self.make_gen_block(init_channel,init_channel//2) # 32
    self.cnnt2 = self.make_gen_block(init_channel//2,init_channel//4) # 64
    self.cnnt3 = self.make_gen_block(init_channel//4,init_channel//8) # 128
    self.cnnt4 = self.make_gen_block(init_channel//8,1,final_layer=True) # 128 final image channel is 1
    
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
    x = self.linear(z)
    x = self.batchnorm1d(x)
    x = self.lrelu(x)

    x = x.view(-1,self.init_channel,self.init_size,self.init_size,self.init_size)
    x = self.cnnt1(x)
    x = self.cnnt2(x)
    x = self.cnnt3(x)
    x = self.cnnt4(x)

    return x


if __name__ == "__main__":
    # check the structure of generator
    z_dim = 20
    gen = Generator(z_dim=z_dim)
    c = gen(torch.rand(10,20))
    print(c.shape)
    batch_size = 5
    print( 'The generator architecture is'+'\n{}'.format(
       summary(gen,(batch_size,z_dim)) 
       ))
