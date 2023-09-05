#%%
import os
import numpy as np
import openpnm as op
import porespy as ps

np.set_printoptions(precision=4)
np.random.seed(10)

name_arr = ['Berea','BanderaBrown','BB','BUG','Kirby','Parker','BanderaGray','Bentheimer','BSG','CastleGate','Leopard']
name = name_arr[0] # read Berea
os.mkdir(name+'-sub')

raw_file = np.fromfile('raw/'+name+'_2d25um_binary.raw', dtype=np.uint8)
# raw_file = np.fromfile(name+'_2d25um_binary.raw', dtype=np.uint8)

im = (raw_file.reshape(1000,1000,1000))
im = im==0
print( 'The porosity of {} is {}'.format( name,ps.metrics.porosity(im) ) )



#%% find the optimal image interval for sub-sampling
import math

im_size = 1e03
crop_s = 128
img_interval = 30

def output_img_num(im_size,crop_s,img_interval):
    img_num_one_side = ( ((im_size - 1)  - crop_s) / img_interval )
    img_num_one_side = math.floor(img_num_one_side)
    return img_num_one_side,img_num_one_side**3

print( output_img_num(im_size,crop_s,img_interval)[1] )

img_one,_ = output_img_num(im_size,crop_s,img_interval)


# %% beginning cropping images: 128*128*128
# save cropped images into dict
index = 0
img_samples = {}
for i in range(img_one):
    for j in range(img_one):
        for k in range(img_one):
            index_ib = img_interval*i
            index_ie = img_interval*i + crop_s
            index_jb = img_interval*j
            index_je = img_interval*j + crop_s
            index_kb = img_interval*k
            index_ke = img_interval*k + crop_s
            
            img_sample = im[
            index_ib:index_ie,
            index_jb:index_je,
            index_kb:index_ke
            ]
            
            img_samples[str(index)] = img_sample
            index += 1


# %% test generated image
import matplotlib.pyplot as plt
test = img_samples['200']
print(test.shape)
print( 'The porosity of test image is {}'.format( ps.metrics.porosity(test) ) )
plt.imshow(test[:,:,100])


#%% filter the library
def filter(img,phi):
    img_surface = []
    filt_matrix = []
    for index in [0,-1]:
        img_sx = img[index,:,:]
        img_sy = img[:,index,:]
        img_sz = img[:,:,index]

        img_surface.append(img_sx)
        img_surface.append(img_sy)
        img_surface.append(img_sz)

    for img_t in img_surface:
        phi_temp = ps.metrics.porosity(img_t)
        filt_matrix.append(phi_temp>phi)

    if False in filt_matrix:
        return False
    else:
        return True

img_list = []
filter_phi = 0.07

index = 0
for img in img_samples.values():

    if filter(img,filter_phi):
        f_name = f'{index}.npy'
        save_PATH = name+'-sub/'+ f_name
        np.save(save_PATH,img)
        index += 1


# %% test image load
test = np.load(save_PATH)
print(test.shape)
print('Unique elements are {}'.format(np.unique(test)))


resolution = 2.25e-6
snow = ps.networks.snow(
im=test,
voxel_size=resolution)

proj = op.io.PoreSpy.import_data(snow)

pn,geo = proj[0],proj[1]

# %%
