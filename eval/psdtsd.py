import porespy as ps
import openpnm as op
import numpy as np


import sys, os
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__



def psdtsd(im,voxel_size=2.32e-06):
    blockPrint()
    snow = ps.networks.snow(
    im=im,
    voxel_size=voxel_size)
    

    proj = op.io.PoreSpy.import_data(snow)
    pn = proj[0]
    geo = proj[1]

    mean_psd = np.mean(geo['pore.diameter'])
    mean_tsd = np.mean(geo['throat.diameter'])
    enablePrint()
    return mean_psd,mean_tsd