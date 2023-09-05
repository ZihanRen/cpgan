#%%
from cpgan.ooppnm import img_process
from cpgan import init_yaml
import os
import numpy as np
import pandas as pd

f_yaml = init_yaml.yaml_f
PATH_demo = os.path.join(f_yaml['img_path']['img_demo'],'1.npy')
img_prc = img_process.Image_process(PATH = PATH_demo)
img = img_prc.im

# %% divide the porous media image into different components
img_segment = []
img_size = 128
segment_num = 4
segment_size = int( img_size/segment_num )

start_index = 0
end_index = 0+segment_size

for i in range(segment_num):
    img_segment.append( img[:,:,start_index:end_index] )
    start_index += segment_size
    end_index += segment_size

# %% compute porosity for each block
phi_component = []
for i in img_segment:
    phi_component.append(img_prc.phi(i))
y = ['0-32','32-64','64-96','96-128']
from matplotlib import pyplot as plt
plt.plot(phi_component,y,'o-')
plt.ylabel('Porosity')
plt.xlabel('Image Section Index')
#%% or simply use two lines of code
img_prc = img_process.Image_process(PATH = PATH_demo)
phi_component = img_prc.vertical_phi()




# %% compute section porosity for whole image DB
def img_sg_phi(img,segment_num):
    # input an image and return a list of phi
    img_segment = []
    img_size = 128
    segment_size = int( img_size/segment_num )

    start_index = 0
    end_index = 0+segment_size

    for i in range(segment_num):
        img_segment.append( img[:,:,start_index:end_index] )
        start_index += segment_size
        end_index += segment_size

    phi_component = []
    for i in img_segment:
        phi_component.append(img_prc.phi(i))
    return phi_component
    

# %% compute porosity for each block
df = {}
f_names = os.listdir(f_yaml['img_path']['img_chunk'])
img_prc = img_process.Image_process()
for f_name in f_names:
    img_tmp = np.load(os.path.join(f_yaml['img_path']['img_chunk'],f_name))
    df[f_name] = img_sg_phi(img_tmp,segment_num=4)
df = pd.DataFrame(df)

# %%
df.to_csv('phi-spatial.csv',index=False)
# %%
