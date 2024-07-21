#%%
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class Dataset3D(Dataset):
  def __init__(self,PATH,names,transform=None):
    # image PATH
    self.PATH = PATH
    self.transform = transform
    self.names = names

  def __len__(self):
    return len(self.names)

  def __getitem__(self,index):
    
    # load images
    img_path = os.path.join(self.PATH,self.names[index])
    image = np.load(img_path)
    # convert images and features into tensor
    image = np.repeat(image[np.newaxis,:,:],1,axis=0)
    image_t = torch.from_numpy(image).float()


    if self.transform is not None:
      image_t = self.transform(image_t)
   
    return image_t

  def img_name(self):
    return (os.listdir(self.PATH))


#%%
if __name__ == "__main__":
  # test dataset loading process
  import os
  from cpgan import init_yaml

  img_path = init_yaml.yaml_f['img_path']['img_chunk']
  names_img = os.listdir(img_path)

  batch_size = 3
  dataGAN = Dataset3D(img_path,names_img)
  # print total number of dataset
  print(f'Total number of dataset is {len(dataGAN)}')
  train_data_loader = DataLoader(dataGAN,batch_size=batch_size,shuffle=True)

  for i, img in enumerate(train_data_loader):
    if i == 3:
      break
    print(f'Batch number:{i}')
    print(f'input img shape is {img.size()}, dtype is {img.dtype}')
    print(2*'\n')



# %%
