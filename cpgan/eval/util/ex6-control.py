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

def kabs_sim(img):
    data_pnm = pnm_sim.Pnm_sim(im=img)
    _, _ = data_pnm.network_extract()
    data_pnm.add_boundary_pn()
    kabs = data_pnm.cal_abs_perm()
    data_pnm.close_ws()
    return kabs

def tensor_process(image_tensor):

    image_unflat = image_tensor.detach().cpu()
    image_numpy = image_unflat.numpy()

    return image_numpy>0.5

def img_phy_qual(im):
    eul_im = img_prc.eul(im)
    if abs(eul_im)<500:
        return True
    else:
        return False

def imgshow(im,sample_idx,z_idx):
    f = plt.figure()
    plt.imshow(im[sample_idx,0,z_idx,::])
    plt.show()

def grad_cal(err,z1,z2,t):
    err = torch.Tensor([err])
    grad = err * (-z1*torch.sin(t)+z2*torch.cos(t))
    return torch.mean(grad)

def grad_update(grad,t,eta):
    t -= eta*grad
    return t

def compound_vec(z1,z2,t):
    z_n = z1*torch.cos(t) + z2*torch.sin(t)
    return z_n

def forward_img(z_n,gen):
    gen.eval()
    img = gen(z_n)
    return img[0,0,::]

torch.manual_seed(0)
ex = 6
epoch = 15

# load model and features
f_yaml = init_yaml.yaml_f
gen_path = os.path.join(f_yaml['model'],"ex{}/cganex{}-{}.pth".format(ex,ex,epoch))
gen = generator.Generator(z_dim=200)
gen.load_state_dict(torch.load(gen_path,map_location=torch.device('cpu')))
gen.eval()

# initialize image processor
img_prc = img_process.Image_process()

# %% control on phi
err_list = []

z_n = torch.randn(1,200)
t = torch.Tensor([0.5])
epoch = 100
phi_target = 0.35
eta = 0.1

for i in range(epoch):
    z2 = torch.randn(1,200)
    z_n = compound_vec(z_n,z2,t)
    img = forward_img(z_n,gen)
    img = tensor_process(img)
    if img_phy_qual(img):
        phi_pred = img_prc.phi(img)
        err = phi_pred - phi_target
        err_list.append(abs(err))
        print(f'Epoch {i}, error: {err}')
        grad_t = grad_cal(err,z_n,z2,t)
        t = grad_update(grad_t,t,eta)
        z_n = compound_vec(z_n,z2,t)
        if abs(err) < 0.01:
            break
    else:
        print(f'At epoch {i} the generated image is not reasonable')
        continue

# %% control on kabs
z_n = torch.randn(1,200)
t = torch.Tensor([0.5])
epoch = 100
k_target = 150
eta = 0.1
err_list = []


for i in range(epoch):
    z2 = torch.randn(1,200)
    z_n = compound_vec(z_n,z2,t)
    img = forward_img(z_n,gen)
    img = tensor_process(img)
    if img_phy_qual(img):
        k_pred = kabs_sim(img)
        err = k_pred - k_target
        print(f'Epoch {i}, error: {err}')
        grad_t = grad_cal(err,z_n,z2,t)
        t = grad_update(grad_t,t,eta)
        z_n = compound_vec(z_n,z2,t)
        err_list.append(abs(err))
        if abs(err) < 5:
            break
    else:
        print(f'At epoch {i+1} the generated image is not reasonable')
        continue

# %% multivariate optimization
err_list = []

z_n = torch.randn(1,200)
t = torch.Tensor([0.5])
epoch = 100
phi_target = 0.17
k_target = 160
err_list_phi = []
err_list_k = []
eta = 0.1

for i in range(epoch):
    z2 = torch.randn(1,200)
    z_n = compound_vec(z_n,z2,t)
    img = forward_img(z_n,gen)
    img = tensor_process(img)
    if img_phy_qual(img):
        phi_pred = img_prc.phi(img)
        k_pred = kabs_sim(img)
        err_phi = phi_pred - phi_target
        err_k = k_pred - k_target
        err_list_phi.append(abs(err_phi))
        err_list_k.append(abs(err_k))
        print(f'Epoch {i}, error k: {err_k}')
        print(f'Epoch {i}, error phi: {err_phi}')

        grad_t_phi = grad_cal(err_phi,z_n,z2,t)
        grad_t_k = grad_cal(err_k/80,z_n,z2,t)

        err_check = abs(err_phi) + abs(err_k/80)

        # if err_check < 0.2:
        #     eta = 0.1
        # elif  err_check < 0.15:
        #     eta = 0.05
        # elif  err_check < 0.1:
        #     eta = 0.01
        # else:
        #     eta = 0.5

        t = grad_update(grad_t_phi + grad_t_k,t,eta)
        z_n = compound_vec(z_n,z2,t)

        if err_check < 0.1:
            break
    else:
        print(f'At epoch {i} the generated image is not reasonable')
        continue
# %%
