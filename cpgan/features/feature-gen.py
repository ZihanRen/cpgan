#%%
import numpy as np
import openpnm as op
import porespy as ps
import matplotlib.pyplot as plt
import pandas as pd
import os
from openpnm.models import physics as mods
from scipy import stats
from skimage.measure import euler_number

np.random.seed(10)
np.set_printoptions(precision=4)
resolution = 2.32e-06
im_shape = 128
snwp = np.arange(0, 1.1, 0.1)

def n2_cal(pore_index,pore_occ):
    # cal of redundant loop
    return (pore_occ[pore_index[0]] & 
            pore_occ[pore_index[1]])

def n3_cal(pore_index,pore_occ):
    # cal of isolated pore
    return not (pore_occ[pore_index[0]] | 
            pore_occ[pore_index[1]])


def phi_cal(geo,size,resolution):
    v_tot = (size*resolution)**3
    vol_pores = geo['pore.volume'].sum()
    vol_throats = geo['throat.volume'].sum()
    phi = (vol_pores+vol_throats)/v_tot
    return phi


def phase_conn(ip,air,sat):
    chi_arr = []
    chi_max = len(ip['pore.all'])
    chi_min = chi_max-len(ip['throat.all'])

    for s in sat:
        occupancies=ip.results(Snwp=s)
        air.update(ip.results(Snwp=s))
        n1 = np.sum(air['pore.occupancy']*1)
        fill_t = np.where(air['throat.occupancy']==True)[0]
        n2 = 0
        n3 = 0
        
        for index in fill_t:
            conn_pore = pn['throat.conns'][index]
            if n2_cal(conn_pore,air['pore.occupancy']):
                n2+=1
            if n3_cal(conn_pore,air['pore.occupancy']):
                n3+=1
        chi_arr.append(n1-n2+n3)


    norm_chi = [(x-chi_max)/(chi_min-chi_max) for x in chi_arr]
    return np.array(norm_chi)

def invasion(phase,direction):
    ip = op.algorithms.InvasionPercolation(network=pn)
    ip.setup(phase=phase)
    in_pores=pn.pores(direction)
    ip.set_inlets(pores=in_pores)
    ip.run()
    return ip
    
def stokes_flow(phase,inlet,outlet):
    st = op.algorithms.StokesFlow(network=pn)
    st.setup(phase=phase)
    # in boundary front water saturation is 1
    # Boundary pores with constant condition
    # left is the inlet of flow
    st.set_value_BC(pores=pn.pores(inlet), values=0)
    # outlet of flow
    st.set_value_BC(pores=pn.pores(outlet), values=1) # this part needs some validation
    st.run()
    return st

def cal_absolute_perm(image_size,resolution,outlet,st):
    A = (image_size*image_size) *resolution**2 # m^2
    L = image_size * resolution # m
    mu = water['pore.viscosity'].max() # N s/m^2 or Pa s
    Pressure = 1 # pa
    delta_P = Pressure - 0

    Q = st.rate(pores=pn.pores(outlet), mode='group')
    K = Q[0] * L * mu / (A * delta_P)

    return K/0.98e-12*1000 # unit md

def cal_eff_perm(image_size,resolution,outlet,st,phase):
    A = (image_size*image_size) *resolution**2
    L = image_size * resolution
    mu = phase['pore.viscosity'].max() # N s/m^2 or Pa s
    Pressure = 1 # pa
    delta_P = Pressure - 0

    Q = st.rate(pores=pn.pores(outlet), mode='group')
    K = Q[0] * L * mu / (A * delta_P)

    return K/0.98e-12*1000

def network_extract(im,resolution):

    snow = ps.networks.snow(
    im=im,
    voxel_size=resolution)

    proj = op.io.PoreSpy.import_data(snow)

    return proj

def get_physics(gen_physics):
    gen_physics.add_model(propname='throat.hydraulic_conductance',
                model=mods.hydraulic_conductance.classic_hagen_poiseuille)
    gen_physics.add_model(propname='throat.entry_pressure',
                model=mods.capillary_pressure.washburn)
    return gen_physics

def load_nparray(fname,name):
    # fname: image index
    # name: rock type
    # PATH: current working directory
    load_PATH = name+'-sub/'+ fname
    im1 = np.load(load_PATH)
    return im1


#%%
# load numpy array
PATH = '/akshat/s0/zur74/data/ibm-11'
# PATH = os.path.join("d:\\",'data',"3d")
os.chdir(PATH)
name_arr = ['Berea','BanderaBrown','BB','BUG','Kirby','Parker','BanderaGray','Bentheimer','BSG','CastleGate','Leopard']
name = name_arr[0] 
f_name_list = os.listdir(name+'-sub/')
data = {}

for fname in f_name_list:
    im = load_nparray(fname,name)
    
    print( '\n'+fname )


    ws = op.Workspace()
    resolution = 2.25e-6 
    snow = ps.networks.snow(
    im=im,
    voxel_size=resolution)

    proj = op.io.PoreSpy.import_data(snow)
    pn,geo = proj[0],proj[1]




    # os.chdir('/akshat/s0/zur74/data')
    # os.chdir('/Users/zihanren/OneDrive/tmp')
    data_tmp = {}

    data_tmp['coordination'] = np.mean(pn.num_neighbors(pn.Ps))
    data_tmp['porosity'] = ps.metrics.porosity(im)
    data_tmp['pore.diameter'] = np.mean(geo['pore.diameter'])
    data_tmp['throat.diameter'] = np.mean(geo['throat.diameter'])
    data_tmp['euler'] = euler_number(im,connectivity=3)

    health = pn.check_network_health()
    op.topotools.trim(network=pn, pores=health['trim_pores'])

    if ( (len(pn.pores('top')) > 0) & (len(pn.pores('bottom')) > 0) ):
        inlet = 'top'
        outlet = 'bottom'

    elif ( (len(pn.pores('left')) > 0) & (len(pn.pores('right')) > 0) ):
        inlet = 'left'
        outlet = 'right'

    elif ( (len(pn.pores('front')) > 0) & (len(pn.pores('back')) > 0) ):
        inlet = 'front'
        outlet = 'back'

    else:
        continue

    # define phase and physics
    air = op.phases.Air(network=pn)
    water = op.phases.Water(network=pn)
    water['pore.contact_angle'] = 0
    air['pore.contact_angle'] = 180
    # water['pore.surface_tension'] = 0.064
    # air['pore.surface_tension'] = 0.064

    phys_air = op.physics.Standard(network=pn, phase=air, geometry=geo)
    phys_water = op.physics.Standard(network=pn, phase=water, geometry=geo)

    # perform invasion flow simulation
    
    ip=invasion(air,inlet)
    data_tmp['chi'] = phase_conn(ip,air,snwp)

    st = stokes_flow(water,inlet,outlet)
    Q_abs_water = st.rate(pores=pn.pores(outlet))
    kabs_tmp = cal_absolute_perm(im_shape,resolution,outlet,st)
    val = 0
    phys_water.regenerate_models() 
    kw_tmp = []

    for s in snwp:  
        air.update(ip.results(Snwp=s))
        phys_water['throat.hydraulic_conductance'][air['throat.occupancy']] = val
        st.run()
        kw_tmp.append(cal_eff_perm(im_shape,resolution,outlet,st,water))
        phys_water.regenerate_models()

    # calculate permeability/eff perm of air
    phys_water.regenerate_models()
    phys_air.regenerate_models()
    st_a = stokes_flow(air,inlet,outlet)
    Q_abs_air = st_a.rate(pores=pn.pores(outlet))
    knw_tmp = []

    for s in snwp:  
        air.update(ip.results(Snwp=s))
        phys_air['throat.hydraulic_conductance'][~air['throat.occupancy']] = val
        st_a.run()
        knw_tmp.append(cal_eff_perm(im_shape,resolution,outlet,st_a,air))
        phys_air.regenerate_models()

    # calcualte kr
    krw_tmp = np.array(kw_tmp)/kabs_tmp
    krnw_tmp = np.array(knw_tmp)/kabs_tmp

    data_tmp['kr_water'] = krw_tmp
    data_tmp['kr_air'] = krnw_tmp
    data_tmp['k_water'] = np.array(kw_tmp)
    data_tmp['k_air'] = np.array(krnw_tmp)
    data_tmp['kabs'] = kabs_tmp

    data[fname] = data_tmp
    ws.close_project(proj)


#%% save dictionary file
# os.chdir(r'C:\Users\rtopa\OneDrive\phd22\gan\3d\data-gen')
os.chdir('/akshat/s0/zur74/OneDrive/phd22/gan/3d/data-gen')
import pickle
with open("features-3d-1.pkl", "wb") as tf:
    pickle.dump(data,tf)

# load dictionary file
# import pickle
# with open("test.pkl", "rb") as tf:
#     new_dict = pickle.load(tf)


