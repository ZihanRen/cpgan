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
from cpgan.eval.util import z_perturb
import pickle
from cpgan.ooppnm import pnm_sim_boundary
from cpgan.ooppnm import pnm_sim_old
import sys
import io

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
epoch_gan = 15

# load model and features
gen = load_gan(ex,epoch_gan)
phys_func = img_prc.phi

target_phi = 0.2

opt = z_perturb.Z_perturb(gen,phys_func,target_phi)
z,_,_,epoch = opt.optimize(100,0.01,eta=0.5)
print("epoch: ",epoch)
img = z_to_img(z,gen) > 0.5
print("phi: ",img_prc.phi(img), "target phi: ",target_phi)


data_pnm = pnm_sim.Pnm_sim(im=img)
psd, tsd = data_pnm.network_extract()
kabs = data_pnm.cal_abs_perm()
data_pnm.close_ws()

print("kabs: ",kabs)
print("tsd: ",tsd)
print("psd: ",psd)
# %%
