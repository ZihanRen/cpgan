#%%
from cpgan import init_yaml
import os
import numpy as np
import numpy as np
from cpgan import init_yaml
import porespy as ps
import pickle
from cpgan.ooppnm import pnm_sim_op
from cpgan.ooppnm import pnm_sim_old

f_yaml = init_yaml.yaml_f
f_names = os.listdir(f_yaml['img_path']['img_large'])
f_large = os.path.join(
    f_yaml['img_path']['img_large'],'BanderaGray_2d25um_binary.raw'
    )

f_large_parker = os.path.join(
    f_yaml['img_path']['img_large'],'Parker_2d25um_binary.raw'
    )

f_large_berea = os.path.join(
    f_yaml['img_path']['img_large'],'Berea_2d25um_binary.raw'
    )
#%%
raw_file = np.fromfile(f_large, dtype=np.uint8)
im = (raw_file.reshape(1000,1000,1000))
im = im==0

raw_file = np.fromfile(f_large_parker, dtype=np.uint8)
im1 = (raw_file.reshape(1000,1000,1000))
im1 = im1 ==0

raw_file = np.fromfile(f_large_berea, dtype=np.uint8)
im2 = (raw_file.reshape(1000,1000,1000))
im2 = im2 ==0

print('simulation begin')

# def kr_simulation(img_input,sim_num=100,trapping=False):
#     data_pnm = pnm_sim_boundary.Pnm_sim(im=img_input)
#     data_pnm.network_extract()
#     data_pnm.init_physics()
#     ip_snw,ip_pc = data_pnm.invasion_percolation(trapping=trapping)
#     df_kr = data_pnm.kr_simulation(Snwp_num=sim_num)
#     data_pnm.close_ws()
#     return [ip_snw,ip_pc],df_kr

# def kr_simulation(img_input,sim_num=30):
#     data_pnm = pnm_sim_op.Pnm_sim(im=img_input)
#     data_pnm.network_extract()
#     data_pnm.init_physics()
#     pc_data = data_pnm.ordinary_percolation(num_data=sim_num)
#     df_kr = data_pnm.kr_simulation()
#     data_pnm.close_ws()
#     return pc_data,df_kr

def kr_simulation(img_input,num_points=50):
    data_pnm = pnm_sim_old.Pnm_sim(im=img_input)
    data_pnm.network_extract()
    if data_pnm.error == 1:
        return None
    data_pnm.init_physics()
    data_pnm.invasion_percolation(num_points=num_points)
    df_kr = data_pnm.kr_simulation()
    data_pnm.close_ws()
    return df_kr

# pnm_obj_berea = pnm_extract(im2)
# k1 = kabs_sim(im)
# k2 = kabs_sim(im1)
# k3 = kabs_sim(im2)

# with open('save_obj/ibm11/kabs.pkl','wb') as f:
#     pickle.dump([k1,k2,k3] ,f)

df = kr_simulation(im)
df1 = kr_simulation(im1)
df2 = kr_simulation(im2)



with open(f'save_obj/ibm11/kr-bd-bg.pkl','wb') as f:
    pickle.dump(df ,f)


with open(f'save_obj/ibm11/kr-bd-pk.pkl','wb') as f:
    pickle.dump(df1 ,f)


with open(f'save_obj/ibm11/kr-bd-berea.pkl','wb') as f:
    pickle.dump(df2 ,f)