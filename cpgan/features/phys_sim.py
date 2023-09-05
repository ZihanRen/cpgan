
import os
import numpy as np
from cpgan import init_yaml
from cpgan.ooppnm import img_process
from cpgan.ooppnm import pnm_sim
import pandas as pd

img_prc = img_process.Image_process()

def kabs_sim(img):
    data_pnm = pnm_sim.Pnm_sim(im=img)
    psd, tsd = data_pnm.network_extract()
    data_pnm.add_boundary_pn()
    kabs = data_pnm.cal_abs_perm()
    data_pnm.close_ws()
    return psd,tsd,kabs


f_yaml = init_yaml.yaml_f
img_path = f_yaml['img_path']['img_chunk']

df_feature = {}
df_feature['name'] = []
df_feature['phi'] = []
df_feature['k'] = []
df_feature['spec_area'] = []
df_feature['eul'] = []
df_feature['psd'] = []
df_feature['tsd'] = []

for f_name in os.listdir(img_path):
    im = np.load(os.path.join(img_path,f_name))
    psd_t,tsd_t,k_t = kabs_sim(im)
    df_feature['name'].append(f_name)
    df_feature['phi'].append(img_prc.phi(im))
    df_feature['k'].append(k_t)
    df_feature['spec_area'].append(img_prc.spec_suf_area(im))
    df_feature['eul'].append(img_prc.eul(im))
    df_feature['psd'].append(psd_t)
    df_feature['tsd'].append(tsd_t)

df = pd.DataFrame(df_feature)
df.to_csv(os.path.join(f_yaml['feature_path'],'features.csv'),index=False)
