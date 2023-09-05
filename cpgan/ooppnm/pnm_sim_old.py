'''
old version of openpnm and porespy
increase variability of results
'''

import numpy as np
import porespy as ps
import openpnm as op
import matplotlib.pyplot as plt
import random
import pandas as pd
from skimage.measure import euler_number

class Pnm_sim:
    def __init__(self,PATH=None,im=None,voxel_size=2.25e-06,inlet='left',outlet='right',img_size=128):
        np.random.seed(0)
        random.seed(0)

        # read numpy image PATH 
        if im is not None:
            self.im = im
        else:
            self.im = np.load(PATH)
        if ( (PATH==None) & (im.size==0) ):
            raise Exception("Sorry you need to provide both image PATH or image array")

        self.ws = op.Workspace()
        self.voxel_size = voxel_size
        self.img_size = img_size
        
        # initialize OpenPNM objects
        self.proj = None
        self.geo = None
        self.pn = None
        self.air = None
        self.water = None
        self.ip = None
        self.data_tmp = {}
        self.inlets = inlet
        self.outlets = outlet
        self.phys_air = None
        self.phys_water = None
        self.OP_1 = None
        self.error = None

    def network_extract(self):
        snow = ps.networks.snow(
            im=self.im,
            voxel_size=self.voxel_size)

        self.proj = op.io.PoreSpy.import_data(snow)  
        self.pn,self.geo = self.proj[0],self.proj[1]

        self.data_tmp['coordination'] = np.mean(self.pn.num_neighbors(self.pn.Ps))
        self.data_tmp['porosity'] = ps.metrics.porosity(self.im)
        self.data_tmp['pore.diameter'] = np.mean(self.geo['pore.diameter'])
        self.data_tmp['throat.diameter'] = np.mean(self.geo['throat.diameter'])
        self.data_tmp['euler'] = euler_number(self.im,connectivity=3)
        health = self.pn.check_network_health()
        op.topotools.trim(network=self.pn, pores=health['trim_pores'])

        if ( (len(self.pn.pores(self.inlets)) == 0) | (len(self.pn.pores(self.outlets)) == 0) ):
            print('Condition not satisfied')
            self.error = 1
    
    def __cal_abs_perm(self,Q):
        length = self.im.shape[0]
        width = self.im.shape[1]
        height = self.im.shape[2]
        A = (width * height)*self.voxel_size**2
        L = length*self.voxel_size
        mu = self.water['pore.viscosity'].max()
        delta_p = 1
        K = Q * L * mu / (A * delta_p)
        return K/0.98e-12*1000

    def __cal_eff_perm(self,Q,phase):
        length = self.im.shape[0]
        width = self.im.shape[1]
        height = self.im.shape[2]
        A = (width * height)*self.voxel_size**2
        L = length*self.voxel_size
        mu = phase['pore.viscosity'].max()
        delta_p = 1
        K = Q * L * mu / (A * delta_p)
        return K/0.98e-12*1000

    def init_physics(self):
        self.air = op.phases.Air(network=self.pn)
        self.water = op.phases.Water(network=self.pn)
        self.water['pore.contact_angle'] = 0
        self.air['pore.contact_angle'] = 180
        self.phys_air = op.physics.Standard(network=self.pn, phase=self.air, geometry=self.geo)
        self.phys_water = op.physics.Standard(network=self.pn, phase=self.water, geometry=self.geo)

    def invasion_percolation(self,num_points=50):
        self.OP_1 = op.algorithms.OrdinaryPercolation(network=self.pn,phase=self.air)
        self.OP_1.set_inlets(pores=self.pn.pores(self.inlets))
        self.OP_1.setup(phase=self.air, pore_volume='pore.volume', throat_volume='throat.volume')
        self.OP_1.run(points=num_points)

    def __update_phase_and_phys_air(self,results):
        val = np.amin(self.phys_water['throat.hydraulic_conductance'])/1000
        results['pore.occupancy'] = results['pore.occupancy'].astype(int) > 0
        results['throat.occupancy'] = results['throat.occupancy'].astype(int) > 0
        self.air.update(results)
        self.phys_air['throat.hydraulic_conductance'][~self.air['throat.occupancy']] = val


    def __update_phase_and_phys_water(self,results):
        results['pore.occupancy'] = results['pore.occupancy'].astype(int) > 0
        results['throat.occupancy'] = results['throat.occupancy'].astype(int) > 0
        val = np.amin(self.phys_water['throat.hydraulic_conductance'])/1000
        self.air.update(results)
        self.phys_water['throat.hydraulic_conductance'][self.air['throat.occupancy']] = val


    def get_absolute_perm(self):
        '''
        you need to firstly init physics before running this function
        '''
        self.phys_water.regenerate_models()
        st = op.algorithms.StokesFlow(network=self.pn)
        st.setup(phase=self.water)
        st.set_value_BC(pores=self.pn.pores(self.inlets), values=0)
        st.set_value_BC(pores=self.pn.pores(self.outlets), values=1)
        st.run()
        Q_abs_water = st.rate(pores=self.pn.pores(self.outlets),mode='group')
        kabs_tmp = self.__cal_abs_perm(Q_abs_water)
        self.phys_water.regenerate_models()
        return kabs_tmp[0]

    def kr_simulation(self):

        self.phys_water.regenerate_models()
        st = op.algorithms.StokesFlow(network=self.pn)
        st.setup(phase=self.water)
        st.set_value_BC(pores=self.pn.pores(self.inlets), values=0)
        st.set_value_BC(pores=self.pn.pores(self.outlets), values=1)
        st.run()
        Q_abs_water = st.rate(pores=self.pn.pores(self.outlets),mode='group')
        kabs_tmp = self.__cal_abs_perm(Q_abs_water)
        self.phys_water.regenerate_models()

        data =  self.OP_1.get_intrusion_data()
        kw_tmp = []
        self.phys_water.regenerate_models()

        for Pc in data.Pcap:
            self.__update_phase_and_phys_water(self.OP_1.results(Pc=Pc))
            st = op.algorithms.StokesFlow(network=self.pn)
            st.setup(phase=self.water)
            st.set_value_BC(pores=self.pn.pores(self.inlets), values=0)
            st.set_value_BC(self.pn.pores(self.outlets), values=1)
            st.run()
            Q = st.rate(self.pn.pores(self.outlets),mode='group')[0]
            keff = self.__cal_eff_perm(Q,self.water)
            kw_tmp.append(keff)
            self.proj.purge_object(obj=st)
            self.phys_water.regenerate_models()

        # simulation of air
        knw_tmp = []
        self.phys_air.regenerate_models()
        for Pc in data.Pcap:
            self.__update_phase_and_phys_air(self.OP_1.results(Pc=Pc))
            st = op.algorithms.StokesFlow(network=self.pn)
            st.setup(phase=self.air)
            st.set_value_BC(self.pn.pores(self.inlets), values=0)
            st.set_value_BC(self.pn.pores(self.outlets), values=1)
            st.run()
            Q = st.rate(pores=self.pn.pores(self.outlets),mode='group')[0]
            keff = self.__cal_eff_perm(Q,self.air)
            knw_tmp.append(keff)
            self.proj.purge_object(obj=st)
            self.phys_air.regenerate_models()

        krw_tmp = np.array(kw_tmp)/kabs_tmp
        krnw_tmp = np.array(knw_tmp)/kabs_tmp
        self.data_tmp['kr_water'] = krw_tmp
        self.data_tmp['kr_air'] = krnw_tmp
        self.data_tmp['k_water'] = np.array(kw_tmp)
        self.data_tmp['k_air'] = np.array(knw_tmp)
        self.data_tmp['kabs'] = kabs_tmp
        sw = [1-x for x in data.Snwp]
        snw = data.Snwp
        self.data_tmp['snw'] = np.array(snw)
        self.data_tmp['sw'] = np.array(sw)
        
        return self.data_tmp

    def close_ws(self):
        self.ws.close_project(self.pn.project)

    def crop_kr(self,df):
        df_crop = df[ (df['krw']>0.03) ]
        return df_crop

    def kr_visualize(self,df):
        f = plt.figure(figsize=[8,8])
        plt.plot(df['snwp'], df['krnw'], '*-', label='Kr_nw')
        plt.plot(df['snwp'], df['krw'], 'o-', label='Kr_w')
        plt.xlabel('Snwp')
        plt.xlim([0,1])
        plt.ylabel('Kr')
        plt.title('Relative Permeability drainage curve')
        plt.legend()




        
    

