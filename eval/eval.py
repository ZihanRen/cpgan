#%% models load up
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F
# from model1 import Generator
import random
import pandas as pd
import porespy as ps
import phi
import eul
import psdtsd
import math
import pnm as pnm
from modelslib.ex1.model1 import Generator
import kr_model


def tensor_process(image_tensor):

    image_unflat = image_tensor.detach().cpu()
    image_numpy = image_unflat.numpy()

    return image_numpy

def img_filter(im):
  return im>0.5

def plt_hist(feature,gen_feature,name,name_seed):
  f = plt.figure()
  min_x = -999
  max_x = 999
  fname = ''

  if name == 'Porosity':
    min_x = 0
    max_x = 1
    fname = PATH+'histogram-'+name+'{:.2f}.png'.format(name_seed)

  if name=='Euler Characteristics':
    min_x = -300
    max_x = 100
    fname = PATH+'histogram-'+name+'{}.png'.format(name_seed)

  if name=='Mean Pore Size':
    min_x = 0.5e-05
    max_x = 2e-05
    fname = PATH+'histogram-'+name+'{}.png'.format(name_seed)

  plt.hist(gen_feature,facecolor='g',alpha=0.75,density=True,stacked=True,bins=20,label='generation')
  plt.hist(feature.cpu().detach().numpy(),range=[min_x,max_x],facecolor='r',alpha=0.75,density=True,stacked=True,bins=20,label='features input')
  plt.xlabel(name)
  plt.ylabel('Freqency')
  plt.legend()
  f_name_hist.append(fname)
  f.savefig(fname)

def plt_hist_sf(gen_feature,name,name_seed):
  f = plt.figure()
  min_x = min(gen_feature)
  max_x = max(gen_feature)
  fname = ''

  if name == 'Porosity':
    min_x = 0
    max_x = 1
    fname = PATH+'histogramsf-'+name+'{:.2f}.png'.format(name_seed)

  if name=='Euler Characteristics':
    min_x = -300
    max_x = 100
    fname = PATH+'histogramsf-'+name+'{}.png'.format(name_seed)

  if name=='Mean Pore Size':
    min_x = 0.5e-05
    max_x = 3e-05
    fname = PATH+'histogramsf-'+name+'{}.png'.format(name_seed)

  plt.hist(gen_feature,range=[min_x,max_x],facecolor='g',alpha=0.75,density=True,stacked=True,bins=20,label='generation')
  plt.xlabel(name)
  plt.ylabel('Freqency')
  plt.legend()
  f_name_histsf.append(fname)
  f.savefig(fname)

# load the model
torch.manual_seed(0)
ex = 1

PATH = 'result/ex{}/fig20/'.format(ex)
PATH_model = "modelslib/ex{}/cgan20.pth".format(ex)
PATH_f = 'features-3d-1.pkl'

os.makedirs(PATH, exist_ok=True)

gen = Generator(embed_size=20)
gen.load_state_dict(torch.load(PATH_model,map_location=torch.device('cpu')))
gen.eval()

# store parameters in a list
par_gen = list(gen.parameters())
# print name and parameters shape
for name, parameter in gen.named_parameters():
    print(name)

import pickle
with open(PATH_f, "rb") as tf:
    df = pickle.load(tf)

img_num = 20


# %% Reconstructed statistics comparision
feature_name = 'Porosity'
feature_name_o = 'Euler Characteristics'
image_num = 20

f_name_hist = []
f_name_histsf = []

phi_range = (0.2,0.3)
eul_range = (10,20)
psd_range = (1.2e-05,1.4e-05)

# for i in np.arange(0,60,10):
# for i in np.arange(1.2e-05,1.8e-05,0.1e-05):
for i in np.arange(0.15,0.4,0.05):

  # psd_range = (0+i,0.1e-05+i)
  phi_range = (0+i,i+0.05)
  # eul_range = (0+i,10+i)

  # images generation based on latent vector and features
  z = torch.randn(image_num,100)
  phi_f = (phi_range[1] - phi_range[0] ) * torch.rand(image_num,1) + phi_range[0]
  eul_f = torch.randint(eul_range[0],eul_range[1],(image_num,1))
  psd_f = (psd_range[1] - psd_range[0] ) * torch.rand(image_num,1) + psd_range[0]

  features = phi_f

  with torch.no_grad():
    fake = gen(z,features)
    img = tensor_process(fake)
    img = img_filter(img)

  # features generation and images visualization

  phi_g = []
  eul_g = []
  psd_g = []


  for i in range(image_num):
    phi_g.append(phi.phi(img[i,0,:,:,:]))
    eul_g.append(eul.eul(img[i,0,:,:,:]))
    # psd_g.append( psdtsd.psdtsd(im_list_g[i])[0] )

  # fake images visualization
  # for i in range(len(im_list_g)):
  #   im_fake = im_list_g[i]
  #   plt.imshow(im_fake)
  #   plt.show()

  plt_hist(phi_f,phi_g,feature_name,phi_range[0])
  plt_hist_sf(eul_g,feature_name_o,phi_range[0])
  # scatter_plot(eul_f,eul_g,feature_name,eul_range[0])
  # plt_hist(eul_f,eul_g,feature_name,eul_range[0])
  # scatter_plot(psd_f,psd_g,feature_name,psd_range[0])
  # plt_hist(psd_f,psd_g,feature_name,psd_range[0])

#%% gif visualization
from PIL import Image 
# Create the frames
frames = []
frames_o = []
for i,j in zip(f_name_hist,f_name_histsf):
    new_frame = Image.open(i)
    new_frame_o = Image.open(j)
    frames.append(new_frame)
    frames_o.append(new_frame_o)

# Save into a GIF file that loops forever
frames[0].save(PATH+f'hist_{feature_name}.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=600, loop=0)

# Save into a GIF file that loops forever
frames_o[0].save(PATH+f'hist_{feature_name_o}.gif', format='GIF',
               append_images=frames_o[1:],
               save_all=True,
               duration=600, loop=0)



#%% transport properties visualization
df_kr = {}
image_num = 3
for i in np.arange(0.25,0.4,0.05):
  phi_range = (0+i,i+0.05)
  kr_tmp = []

  # images generation based on latent vector and features
  z = torch.randn(image_num,100)
  features = (phi_range[1] - phi_range[0] ) * torch.rand(image_num,1) + phi_range[0]

  with torch.no_grad():
    fake = gen(z,features)
    img = tensor_process(fake)
    img = img_filter(img)
  
  for i in range(image_num):
    df_krtmp = pnm.pnm(img[i,0,:,:,:])
    try:
      data_model = kr_model.main(df_krtmp)
      if (
        data_model['data.model']['krnw'].max() - data_model['data.model']['krnw'].min()
        ) > 0.1:
        kr_tmp.append(data_model)
    except Exception as e:
      print("image num {} at porosity range {} is not available".format(i,phi_range))
      
  
  df_kr["{:.2f}".format(phi_range[0])] = kr_tmp



for porosity in df_kr.keys():
  f = plt.figure()
  for i in range(len( df_kr[porosity] )):
    kr_model = df_kr[porosity][i]['data.model']
    plt.plot(kr_model['snw_tnw'],kr_model['krnw'],c='r')
  plt.xlabel(r'$S_{nw}$')
  plt.ylabel(r'$kr_{air}$')
  plt.title(porosity)
  plt.xlim([0,1])
  plt.ylim([0,1])
  f.savefig(PATH+'krfake' + porosity + '.png')

#%%from PIL import Image
# Create the frames
f_name_list = ['krfake0.30.png','krfake0.35.png','krfake0.40.png']
f_name_list = [PATH+x for x in f_name_list]
frames = []
for i in f_name_list:
    new_frame = Image.open(i)
    frames.append(new_frame)

feature_name = 'porosity'
# Save into a GIF file that loops forever
frames[0].save(PATH+f'kr_{feature_name}.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=600, loop=0)


#%%






#%% transport properties comparing with training image transport properties

# constructed kr properties between 0.15 and 0.2
def filter_phi(df,low,high):
  for keys in df.keys():
    if (df[keys]['porosity'] >= low) and (df[keys]['porosity'] <= high):
      return df[keys]['kr_air']

random_seed = 0
torch.manual_seed(random_seed)
image_num = 5

for i in np.arange(0.15,0.4,0.05):
  phi_range = (i,i+0.05)
  kr_fake = []
  # images generation based on latent vector and features
  z = torch.randn(image_num,100)
  features = (phi_range[1] - phi_range[0] ) * torch.rand(image_num,1) + phi_range[0]

  with torch.no_grad():
    fake = gen(z,features)
    img = tensor_process(fake)
    img = img_filter(img)

  for i in range(image_num):
    df_krtmp = pnm.pnm(img[i,0,:,:,:])
    kr_fake.append(df_krtmp['kr_air'])


  # plot training images kr values
  f = plt.figure()
  snw = np.arange(0,1.1,0.1)
  for i in range(image_num):
    plt.plot(snw,kr_fake[i],c='r')
    
    low_phi = phi_range[0] + i*0.01
    kr_real = filter_phi(df,low_phi,low_phi+0.01)
    plt.plot(snw,kr_real,c='b')

    plt.xlabel(r'$S_{nw}$')
    plt.ylabel(r'$kr_{air}$')

  f.savefig(PATH+'kr-fakereal-{:.2f}{}.png'.format(phi_range[0],random_seed))