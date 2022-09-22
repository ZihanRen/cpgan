## package requirement:

# porespy: 2.0.1
# python: 3.8+

#%%
import porespy as ps
print(ps.__version__)

import porespy as ps
import matplotlib.pyplot as plt
import numpy as np

#%% visualization

im = ps.generators.blobs(shape=[128, 128], porosity=0.4, blobiness=1)
plt.imshow(im,cmap='binary')
phi = len(im[im==True])/(len(im)**2)
from PIL import Image
test = Image.fromarray(im)
f = plt.figure()
test
plt.show()

# im = np.zeros([128, 128])
# im = ps.generators.RSA(im, r=5)
# plt.imshow(im,cmap='binary')
# from PIL import Image
# test = Image.fromarray(im)
# f = plt.figure()
# test
# plt.show()

# %% data export
import os
# PATH = "/akshat/s0/zur74/data/2d"
PATH = os.path.join("d:\\","data","2d","blob")

os.chdir(PATH)

# circle generation
im = np.zeros([128, 128])
from IPython.display import display

radius = [x for x in range(6,8)]
n_max = [x for x in range(10,60)]

print(f'Total image generation is {len(radius)*len(n_max)*90}')

# for r in radius:
#   for n in n_max:
#     for i in range(90):

#       im_tmp = ps.generators.RSA(im, r=r,n_max=n)
#       im_tmp = Image.fromarray(im_tmp)
#       img_name = f"circle/im{int(i)}" + f"-n{int(n)}"  '-'+f'r{int(r)}.png'
#       im_tmp.save(img_name)

# blobs generation
phi = np.arange(0.2,0.5,0.001)
print(f'Total image generation is {len(phi)*30}')

for phi_tmp in phi:
  for i in range(30):
    im_tmp = ps.generators.blobs(shape=im.shape, porosity=phi_tmp, blobiness=1)
    im_tmp = Image.fromarray(im_tmp)
    img_name = f"im{int(i)}" + f"-phi{phi_tmp: .3f}" + ".png"
    im_tmp.save(img_name)


# check the total number of images 
print('The number of blob images are {}'.format(len(os.listdir())))
# %%
