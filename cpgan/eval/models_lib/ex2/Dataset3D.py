#%%
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class Dataset3D(Dataset):
  def __init__(self,PATH,PATH_f,names,transform=None):
    # image PATH
    self.PATH = PATH

    # features PATH
    self.PATH_f = PATH_f
    self.transform = transform
    self.names = names

  def __len__(self):
    return len(self.names)

  def feature_reshape_disc(self,f):
      features_d = np.zeros((128,128,128))
      increment = int(128/len(f))
      start_idx = 0
      for i in range(len(f)):
        features_d[::,start_idx:start_idx+increment] = f[i]
        start_idx += increment
      return features_d

  def __getitem__(self,index):
    
    # load images
    img_path = os.path.join(self.PATH,self.names[index])
    image = np.load(img_path)
    # convert images and features into tensor
    image = np.repeat(image[np.newaxis,:,:],1,axis=0)
    image_t = torch.from_numpy(image).float()

    # load spatial features
    df = pd.read_csv(self.PATH_f)
    features = df[self.names[index]].values

    # features_g as input to generator
    features_g = np.repeat(features,32)
    features_g = torch.from_numpy(features_g)

    # features_d as input to discriminator
    features_d = self.feature_reshape_disc(features)
    features_d = torch.from_numpy(features_d)
    features_d = torch.unsqueeze(features_d,dim=0)

    if self.transform is not None:
      image_t = self.transform(image_t)
   
    return (image_t, features_g, features_d)

  def img_name(self):
    return (os.listdir(self.PATH))



if __name__ == "__main__":
  # test dataset loading process
  import os
  from cpgan import init_yaml

  img_path = init_yaml.yaml_f['img_path']['img_chunk']
  feature_path = os.path.join(init_yaml.yaml_f['feature_path'],'phi-component.csv')
  names_img = os.listdir(img_path)

  batch_size = 3
  dataGAN = Dataset3D(img_path,feature_path,names_img)
  train_data_loader = DataLoader(dataGAN,batch_size=batch_size,shuffle=True)



  for i, (img,features_g,features_d) in enumerate(train_data_loader):
    if i == 3:
      break
    print(f'Batch number:{i}')
    print(f'input img shape is {img.size()}, dtype is {img.dtype}')
    print(f'input gen feature shape is {features_g.size()}, dtype is {features_g.dtype}')
    print(
      f'input disc feature shape is {features_d.size()}, dtype is {features_d.dtype}'
      )
    print(2*'\n')



# %%
