#%%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# file_dir = "/akshat/s0/zur74/OneDrive/phd22/gan/3d/features-3d-1.pkl"
# PATH = "/akshat/s0/zur74/OneDrive/phd22/gan/3d/data-gen/fig"

file_dir = r'C:\Users\rtopa\OneDrive\phd22\gan\3d\features-3d-1.pkl'
PATH = r'C:\Users\rtopa\OneDrive\phd22\gan\3d\data-gen\fig'
with open(file_dir, "rb") as tf:
    df = pickle.load(tf)
names_img = [x for x in df.keys()]
random.shuffle(names_img)
names_img = names_img[0:9000] # sample certain portion of images for training
os.chdir(PATH)


# %% construct dataframe
phi = []
kabs = []
coord = []
psd = []
euler = []


for filename in names_img:
    phi.append( df[filename]['porosity'] )
    kabs.append( df[filename]['kabs'] )
    coord.append( df[filename]['coordination'] )
    psd.append( df[filename]['pore.diameter'] )
    euler.append( df[filename]['euler'] )

features = {}
features['phi'] = np.array(phi)
features['kabs'] = np.array(kabs)
features['coord'] = np.array(coord)
features['psd'] = np.array(psd)
features['euler'] = np.array(euler)


features = pd.DataFrame(features)

#%% correlation map
corr = features.corr()
corr.to_csv('corr.csv')
#%% phi vs kabs
f = plt.figure()
plt.scatter(features['phi'],features['kabs'])
plt.xlabel(r'$\phi$')
plt.ylabel(r'$k_{abs} (md)$')
f.savefig('phivskabs.png')

# %% phi vs mean pore size
f = plt.figure()
plt.scatter(features['phi'],features['psd'])
plt.xlabel(r'$\phi$')
plt.ylabel('Mean Pore Size (macrons)')
f.savefig('phivspsd.png')

# %% phi vs connectivity
f = plt.figure()
plt.scatter(features['phi'],features['coord'])
plt.xlabel(r'$\phi$')
plt.ylabel('coordination number')
f.savefig('phivscoord.png')

# %% kr simulation
snw = np.arange(0,1.1,0.1)
f = plt.figure()
for filename in df.keys():
    plt.plot(snw,df[filename]['kr_air'],c='r')
plt.xlabel(r'$S_{nw}$')
plt.ylabel(r'$kr_{air}$')
f.savefig('kr-multiple.png')

# %%
f = plt.figure()
for filename in df.keys():
    plt.plot(snw,df[filename]['chi'],c='r')
plt.xlabel(r'$S_{nw}$')
plt.ylabel(r'$\chi_{air}$')
f.savefig('chi-multiple.png')
# %%
f = plt.figure()
plt.scatter(features['phi'],features['euler'])
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\chi$')
f.savefig('phivseul.png')
# %%
