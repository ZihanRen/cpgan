#%% models load up
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F
import random
import pandas as pd
import math
from cpgan.ooppnm import wrap_openpnm as pnm_sim
from cpgan.eval.models_lib.model1 import Generator
from os import path

def tensor_process(image_tensor):

    image_unflat = image_tensor.detach().cpu()
    image_numpy = image_unflat.numpy()

    return image_numpy

def img_filter(im):
  return im>0.5

# load the model
torch.manual_seed(0)
ex = 1


PATH = 'result/ex{}/fig50/'.format(ex)
PATH_model = "modelslib/ex{}/cgan20.pth".format(ex)
PATH_f = 'features-3d-1.pkl'
PATH_tr_img = "/journel/s0/zur74/data/ibm-11/Berea-sub/"
os.makedirs(PATH, exist_ok=True)

gen = Generator(embed_size=20)
gen.load_state_dict(torch.load(PATH_model,map_location=torch.device('cpu')))
gen.eval()

import pickle
with open(PATH_f, "rb") as tf:
    df = pickle.load(tf)


#%% fake transport properties visualization
def kr_simulation(img_input):
    data_pnm = pnm_sim.Wrap_pnm(im=img_input)
    data_pnm.network_extract()
    data_pnm.add_boundary_pn()
    data_pnm.init_physics()
    data_pnm.invasion_percolation()
    df_kr = data_pnm.kr_simulation()
    data_pnm.close_ws()
    return df_kr

kr_phi = {}
phi_list = [0.2,0.3,0.4]
image_num = 3

for phi in phi_list:

  # initialize inputs for GAN
  z = torch.randn(image_num,100)
  features = torch.full((image_num,1),phi)
  kr_store_i = []


  with torch.no_grad():
    fake = gen(z,features)
    img = tensor_process(fake)
    img = img_filter(img)
  
  for i in range(image_num):
    
    # perform PNM
    # input should be 3D numpy array
    df_kr = kr_simulation(img[i,0,::])
    kr_store_i.append(df_kr)    
      
  
  kr_phi["{:.2f}".format(phi)] = kr_store_i


#average df
kr_avg = {}
for porosity in kr_phi.keys():
  kr_avg[porosity] = pd.concat(kr_phi[porosity]).groupby(level=0).mean()



f = plt.figure(figsize=[8,8])
c = ['b*','r*','g*']
for phi,i in zip(kr_avg.keys(),c):
  df = kr_avg[phi]
  plt.plot(df['snwp'], df['krnw'], i, label=phi)
  plt.plot(df['snwp'], df['krw'], i)
plt.xlabel('Snwp')
plt.xlim([0,1])
plt.ylabel('Kr')
plt.legend()




# for porosity in df_kr.keys():
#   f = plt.figure()
#   for i in range(len( df_kr[porosity] )):
#     kr_model_data = df_kr[porosity][i]['data.model']
#     plt.plot(kr_model_data['snw_tnw'],kr_model_data['krnw'],c='r')
#   plt.xlabel(r'$S_{nw}$')
#   plt.ylabel(r'$kr_{air}$')
#   plt.title(porosity)
#   plt.xlim([0,1])
#   plt.ylim([0,1])
#   f.savefig(PATH+'krfake' + porosity + '.png')

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

#%% transport properties comparing with training image transport properties
def filter_phi(df,low,high):
  for keys in df.keys():
    if (df[keys]['porosity'] >= low) and (df[keys]['porosity'] <= high):
      return keys

random_seed = 1
torch.manual_seed(random_seed)
image_num = 5

phi_range = (0.2,0.2+0.05)
kr_fake = []
kr_real = []
# images generation based on latent vector and features
z = torch.randn(image_num,100)
features = (phi_range[1] - phi_range[0] ) * torch.rand(image_num,1) + phi_range[0]

with torch.no_grad():
    fake = gen(z,features)
    img = tensor_process(fake)
    img = img_filter(img)

for i in range(image_num):
    # plot kr of fake images
    df_krtmp = pnm.pnm(img[i,0,:,:,:])
    if df_krtmp==None:
      continue
    try: 
        data_model = kr_model.main(df_krtmp)
        if (
        data_model['data.model']['krnw'].max() - data_model['data.model']['krnw'].min()
        ) > 0.1:
            kr_fake.append(data_model)
    except Exception as e:
        print("image num {} at porosity range {} is not available".format(i,phi_range))

# plot kr of training images
f = plt.figure()
for i in range(image_num):

    low_phi = phi_range[0] + i*0.01
    filter_key = filter_phi(df,low_phi,low_phi+0.01)

    # load target training image
    img = np.load( path.join(PATH_tr_img,filter_key) )
    df_krtmp = pnm.pnm(img)

    try:
        data_model = kr_model.main(df_krtmp)
        if (
        data_model['data.model']['krnw'].max() - data_model['data.model']['krnw'].min()
        ) > 0.1:
            kr_real.append(data_model)
    except Exception as e:
        print("image num {} at porosity range {} is not available".format(i,phi_range))

for i in range(min(len(kr_real),len(kr_fake))):
    real_data = kr_real[i]['data.model']
    fake_data = kr_fake[i]['data.model']

    plt.plot(real_data['snw_tnw'],real_data['krnw'],c='r',label='real')
    plt.plot(fake_data['snw_tnw'],fake_data['krnw'],c='b',label='fake')

plt.xlabel(r'$S_{nw}$')
plt.ylabel(r'$kr_{air}$')
plt.legend()
f.savefig(PATH+'kr-fakereal-{:.2f}{}.png'.format(phi_range[0],random_seed))
# %% kr table
f = plt.figure()
real_data = kr_real[0]['data.model']
fake_data = kr_fake[0]['data.model']

plt.plot(real_data['snw_tnw'],real_data['krnw'],c='r',label='real')
plt.plot(real_data['snw_tw'],real_data['krw'],c='r',label='real')

plt.plot(fake_data['snw_tnw'],fake_data['krnw'],c='b',label='fake')
plt.plot(fake_data['snw_tw'],fake_data['krw'],c='b',label='fake')

plt.xlabel(r'$S_{nw}$')
plt.ylabel(r'$kr_{air}$')
plt.legend()
f.savefig(PATH+'krtable0-fakereal-{:.2f}{}.png'.format(phi_range[0],random_seed))

# %%
