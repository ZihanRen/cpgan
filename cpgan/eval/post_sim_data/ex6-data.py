
'''
    if you want to do minmax normalization:
    # min max image into range of [0,1]
    img_size = image_tensor.shape[-1]
    batch_size = image_tensor.size(0)
    image_tensor = image_tensor.view(image_tensor.size(0), -1)
    image_tensor -= image_tensor.min(1, keepdim=True)[0]
    image_tensor /= image_tensor.max(1, keepdim=True)[0]
    image_tensor = image_tensor.view(batch_size, img_size, img_size, img_size)
    https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-3-otsu-thresholding/
    https://learnopencv.com/otsu-thresholding-with-opencv/
'''

#%%
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from cpgan import init_yaml
from cpgan.eval.models_lib.ex6 import generator
import pandas as pd
from cpgan.ooppnm import img_process
from cpgan.ooppnm import pnm_sim
import random
import porespy as ps
from cpgan.ooppnm import img_process
from cpgan.ooppnm import pnm_sim_old
import pickle

img_prc = img_process.Image_process()

# make sure the result is reproducible
random.seed(10)
torch.manual_seed(0)
np.random.seed(0)

# def kabs_sim(img):
#     data_pnm = pnm_sim.Pnm_sim(im=img)
#     psd, tsd = data_pnm.network_extract()
#     kabs = data_pnm.cal_abs_perm()
#     data_pnm.close_ws()
#     return psd,tsd,kabs

def kabs_sim(img):
    data_pnm = pnm_sim_old.Pnm_sim(im=img)
    data_pnm.network_extract()
    if data_pnm.error == 1:
        return None
    data_pnm.init_physics()
    kabs = data_pnm.get_absolute_perm()
    data_pnm.close_ws()
    return kabs, data_pnm.data_tmp

def save_pickle(PATH,data):
    with open(PATH,'wb') as f:
        pickle.dump(data ,f)


ex=6
epoch = 15

f_yaml = init_yaml.yaml_f
gen_path = os.path.join(f_yaml['model'],"ex{}/cganex{}-{}.pth".format(ex,ex,epoch))
feature_path = os.path.join(f_yaml['feature_path'],'features.csv')
img_path = f_yaml['img_path']['img_chunk']

# load features
df = pd.read_csv(feature_path)
gen = generator.Generator(z_dim=200)
gen.load_state_dict(torch.load(gen_path,map_location=torch.device('cpu')))
gen.eval()

# %% training and fake images generation
sample_num = 300
df_subset = df.sample(n=sample_num)
img_name = df_subset['name'].to_list()

noise = torch.randn(sample_num,200)
fake_img = gen(noise)
fake_img = img_prc.clean_img(fake_img)
# %% simulate kabs on real images

kabs_t = []
phi_t = []
eul_t = []
spec_t = []
psd_t = []
tsd_t = []
coord_t = []
covar_t = []


for i in range(sample_num):
    img_t = np.load(os.path.join(img_path,img_name[i]))
    kabs_tmp, df_tmp = kabs_sim(img_t)
    if kabs_tmp == None:
        continue
    psd_t.append(df_tmp['pore.diameter'])
    tsd_t.append(df_tmp['throat.diameter'])
    kabs_t.append(kabs_tmp)
    phi_t.append(img_prc.phi(img_t))
    eul_t.append(img_prc.eul(img_t))
    spec_t.append(img_prc.spec_suf_area(img_t))
    coord_t.append(df_tmp['coordination'])
    covar_t.append(img_prc.two_point_corr(img_t))

save_pickle('real_covar.pkl',coord_t)

# %% simulate on fake iamge
kabs_f = []
phi_f = []
eul_f = []
spec_f = []
psd_f = []
tsd_f = []
coord_f = []
covar_f = []

for i in range(sample_num):
    img_tmp = fake_img[i]
    kabs_tmp, df_tmp = kabs_sim(img_tmp)
    if kabs_tmp == None:
        continue
    psd_f.append(df_tmp['pore.diameter'])
    tsd_f.append(df_tmp['throat.diameter'])
    kabs_f.append(kabs_tmp)
    phi_f.append(img_prc.phi(img_tmp))
    eul_f.append(img_prc.eul(img_tmp))
    spec_f.append(img_prc.spec_suf_area(img_tmp))
    coord_f.append(df_tmp['coordination'])
    covar_f.append(img_prc.two_point_corr(img_tmp))

save_pickle('synthetic_covar.pkl',coord_f)


# %% save data to csv
df_fake = {
    'kabs':kabs_f,
    'phi':phi_f,
    'eul':eul_f,
    'spec_area':spec_f,
    'psd':psd_f,
    'tsd':tsd_f,
    'coord':coord_f
    }

df_real = {
    'name':df_subset['name'].to_list(),
    'kabs':kabs_t,
    'phi':phi_t,
    'eul':eul_t,
    'spec_area':spec_t,
    'psd':psd_t,
    'tsd':tsd_t,
    'coord':coord_f
    }

# %%

df_fake = pd.DataFrame(df_fake)
df_real = pd.DataFrame(df_real)

df_fake.to_csv(f"df_fakeold_{ex}.csv",index=False)
df_real.to_csv(f"df_realold_{ex}.csv",index=False)




# %%
