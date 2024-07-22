import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from cpgan import init_yaml
from cpgan.eval.models_lib.ex6 import generator
import pandas as pd
from cpgan.ooppnm import img_process
from cpgan.ooppnm import pnm_sim
import porespy as ps
from cpgan.eval.util import z_perturb
import pickle
from cpgan.ooppnm import pnm_sim_boundary
from cpgan.ooppnm import pnm_sim_old
import sys
import io
import random
import time

class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

f_yaml = init_yaml.yaml_f
img_prc = img_process.Image_process()

def load_gan(ex,epoch):
    f_yaml = init_yaml.yaml_f
    gen_path = os.path.join(f_yaml['model'],"ex{}/cganex{}-{}.pth".format(ex,ex,epoch))
    gen = generator.Generator(z_dim=200)
    gen.load_state_dict(torch.load(gen_path,map_location=torch.device('cpu')))
    gen.eval()
    return gen

def z_to_img(z,gen):
    img = img_prc.clean_img_filt( gen(z) )[0]
    return img

ex = 6
epoch_gan = 25
num_samples = 100
gen = load_gan(ex,epoch_gan)


def perturb_gen(gen,phys_func,target_list,threshold):
    start_time = time.time()
    num_sample = 100
    
    gen_list = []
    epoch_list = []
    target_list_output = []
    for i in range(num_sample):
        target_val = target_list[i]
        opt = z_perturb.Z_perturb(gen,phys_func,target_val)
        z,_,phys_gen,epoch = opt.optimize(100,threshold,eta=0.5)
        if not epoch:
            continue
        # img = z_to_img(z,gen) > 0.5
        # phi_gen = img_prc.phi(img)
        gen_list.append(phys_gen)
        epoch_list.append(epoch)
        target_list_output.append(target_val)

    end_time = time.time()
    execution_time = end_time - start_time
    data_dict = {'time': execution_time, 'generate_list':gen_list, 'target_list': target_list_output, 'epoch_list': epoch_list}
    return data_dict
    


#%% porosity condition


# phi_list = [random.uniform(0.1, 0.3) for _ in range(num_samples)]
# porosity_dict = perturb_gen(gen,img_prc.phi,phi_list,0.01)

# # save this dict to pickle
# with open('cond_result/porosity_dict.pkl', 'wb') as f:
#     pickle.dump(porosity_dict, f)


#%% kabs condition
# kabs_list = [random.uniform(100, 300) for _ in range(num_samples)]
# def kabs_sim(img):
#     with SuppressPrint():   
#         data_pnm = pnm_sim_old.Pnm_sim(im=img)
#         data_pnm.network_extract()
#         if data_pnm.error == 1:
#             print('Error in network extraction')
#             return None
#         data_pnm.init_physics()
#         data_pnm.get_absolute_perm()
#         kabs = data_pnm.data_tmp['kabs']
#         data_pnm.close_ws()
#     return kabs
# kabs_dict = perturb_gen(gen,kabs_sim,kabs_list,17)

# with open('cond_result/kabs_dict.pkl', 'wb') as f:
#     pickle.dump(kabs_dict, f)






#%% psd control
def psd_sim(img):
    data_pnm = pnm_sim.Pnm_sim(im=img)
    psd, tsd = data_pnm.network_extract()
    return psd

psd_list = [random.uniform(0.95e-05, 1.1e-05) for _ in range(num_samples)]
psd_dict = perturb_gen(gen,psd_sim,psd_list,1e-07)

# save this dict to pickle
with open('cond_result/psd_dict.pkl', 'wb') as f:
    pickle.dump(psd_dict, f)


#%% tsd control
def tsd_sim(img):
    data_pnm = pnm_sim.Pnm_sim(im=img)
    psd, tsd = data_pnm.network_extract()
    return tsd


tsd_list = [random.uniform(3.6e-06, 4.2e-06) for _ in range(num_samples)]
tsd_dict = perturb_gen(gen,tsd_sim,tsd_list,5e-08)

# save this dict to pickle
with open('cond_result/tsd_dict.pkl', 'wb') as f:
    pickle.dump(tsd_dict, f)