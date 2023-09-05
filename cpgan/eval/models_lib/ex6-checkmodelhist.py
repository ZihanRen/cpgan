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
random.seed(10)
np.random.seed(10)

# save the matrix of images that are not available for simulation
bad_im = []

def kr_simulation(img_input,sim_num=100):
    data_pnm = pnm_sim.Pnm_sim(im=img_input)
    data_pnm.network_extract()
    data_pnm.add_boundary_pn()
    data_pnm.init_physics()
    data_pnm.invasion_percolation()
    df_kr = data_pnm.kr_simulation(Snwp_num=sim_num)
    data_pnm.close_ws()
    return df_kr

def kabs_sim(img):
    data_pnm = pnm_sim.Pnm_sim(im=img)
    psd, tsd = data_pnm.network_extract()
    data_pnm.add_boundary_pn()
    kabs = data_pnm.cal_abs_perm()
    data_pnm.close_ws()
    return psd,tsd,kabs

def imgshow(im,sample_idx,z_idx):
    f = plt.figure()
    plt.imshow(im[sample_idx,0,z_idx,::])
    plt.show()

def plt_hist(fake,real,range_input,xlabel):
    # range input should be list
    f = plt.figure()
    plt.hist(real,range=range_input,density=True,bins=20,alpha=0.4,edgecolor='black',label='real')
    plt.hist(fake,range=range_input,density=True,bins=20,alpha=0.4,edgecolor='black',label='fake')
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()

def plt_scatter(fake,real,range_input,xlabel):
    # range input should be list
    f = plt.figure()
    plt.scatter(real,fake)
    plt.xlim(range_input)
    plt.ylim(range_input)
    plt.xlabel("Real "+xlabel)
    plt.ylabel("Fake "+xlabel)
    plt.show()

def box_plot(fake,real,ylabel):
    phi_box = [fake, real]
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(phi_box)
    ax.set_xticks([1,2],['fake','real'])
    ax.set_ylabel(ylabel)
    plt.show()


ex = 6
epoch = 20

# load model and features
f_yaml = init_yaml.yaml_f
gen_path = os.path.join(f_yaml['model'],"ex{}/cganex{}-{}.pth".format(ex,ex,epoch))
feature_path = os.path.join(f_yaml['feature_path'],'features.csv')
img_path = f_yaml['img_path']['img_chunk']

df = pd.read_csv(feature_path)
gen = generator.Generator(z_dim=200)
gen.load_state_dict(torch.load(gen_path,map_location=torch.device('cpu')))
gen.eval()

real_data = pd.read_csv('data/df_real_{}.csv'.format(ex))
img_prc = img_prc = img_process.Image_process()

sample_num = 40
fake_img = img_prc.clean_img( gen(torch.randn(sample_num,200)) )
#%% models load up
# kabs_f = []
phi_f = []
eul_f = []
spec_f = []
# psd_f = []
# tsd_f = []

for i in range(sample_num):
    img_tmp = fake_img[i]
    # psd_f_tmp, tsd_f_tmp, kabs_tmp = kabs_sim(img_tmp)
    # psd_f.append(psd_f_tmp)
    # tsd_f.append(tsd_f_tmp)
    # kabs_f.append(kabs_tmp)        
    phi_f.append(img_prc.phi(img_tmp))
    eul_f.append(img_prc.eul(img_tmp))
    spec_f.append(img_prc.spec_suf_area(img_tmp))

df_fake = {
    'phi':phi_f,
    'eul':eul_f,
    'spec_area':spec_f
    }

df_fake = pd.DataFrame(df_fake)
df_real = real_data[['phi','eul','spec_area']]
# %% another model
ex = 6
epoch = 15
gen_path = os.path.join(f_yaml['model'],"ex{}/cganex{}-{}.pth".format(ex,ex,epoch))
gen = generator.Generator(z_dim=200)
gen.load_state_dict(torch.load(gen_path,map_location=torch.device('cpu')))
gen.eval()

sample_num = 40
fake_img = img_prc.clean_img( gen(torch.randn(sample_num,200)) )

phi_f = []
eul_f = []
spec_f = []
# psd_f = []
# tsd_f = []

for i in range(sample_num):
    img_tmp = fake_img[i]
    # psd_f_tmp, tsd_f_tmp, kabs_tmp = kabs_sim(img_tmp)
    # psd_f.append(psd_f_tmp)
    # tsd_f.append(tsd_f_tmp)
    # kabs_f.append(kabs_tmp)        
    phi_f.append(img_prc.phi(img_tmp))
    eul_f.append(img_prc.eul(img_tmp))
    spec_f.append(img_prc.spec_suf_area(img_tmp))

df_fake1 = {
    'phi':phi_f,
    'eul':eul_f,
    'spec_area':spec_f
    }

df_fake1 = pd.DataFrame(df_fake1)


# %% clearly the epoch 15 is cloest to the real image correlation

