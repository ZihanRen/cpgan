#%%
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle

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

  def __getitem__(self,index):
      
    # load images
    img_path = os.path.join(self.PATH,self.names[index])
    image = np.load(img_path)
    # convert images and features into tensor
    image = np.repeat(image[np.newaxis,:,:],1,axis=0)
    image_t = torch.from_numpy(image).float()

    # load features
    with open(self.PATH_f, "rb") as tf:
        df = pickle.load(tf)

    df = df[self.names[index]]
    features = [df['porosity']]
    feature_t = torch.FloatTensor(features)

    if self.transform is not None:
      image_t = self.transform(image_t)
   
    return (image_t, feature_t)

  def img_name(self):
    return (os.listdir(self.PATH))



if __name__ == "__main__":
  # test dataset loading process
  import os
  # PATH = '/akshat/s0/zur74/data/ibm-11/Berea-sub'
  # PATH_f = '/akshat/s0/zur74/OneDrive/phd22/gan/3d/features-3d-1.pkl'
  PATH = r'D:\data\3d\Berea-sub'
  PATH_f = r'C:\Users\rtopa\OneDrive\phd22\gan\3d\features-3d-1.pkl'

  names_img = os.listdir(PATH)

  batch_size = 1
  dataGAN = Dataset3D(PATH,PATH_f,names_img)
  train_data_loader = DataLoader(dataGAN,batch_size=batch_size,shuffle=True)



  for i, (img,features) in enumerate(train_data_loader):
    if i == 3:
      break
    print(f'Batch number:{i}')
    print(f'input img shape is {img.size()}, dtype is {img.dtype}')
    print(f'input feature shape is {features.size()}, dtype is {features.dtype}')
    print(2*'\n')



# %%
