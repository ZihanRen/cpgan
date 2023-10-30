import numpy as np
import porespy as ps
import openpnm as op
import matplotlib.pyplot as plt
import random
import pandas as pd

class Pnm_sim:
    def __init__(self,PATH=None,im=None,voxel_size=2.25e-06,inlet='left',outlet='right'):
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
        self.inlet_dir = inlet
        self.outlet_dir = outlet

        # get the dimension of the image
        self.img_size = im.shape[0]

        # initialize OpenPNM objects
        self.pn = None
        self.air = None
        self.water = None
        self.drn = None


    def network_extract(self):
        
        snow = ps.networks.snow2(
            phases=self.im,
            phase_alias={True:"void",False:"solid"},
            voxel_size=self.voxel_size)
        
        self.pn = op.io.network_from_porespy(snow.network)
        self.pn = ps.networks.label_boundaries(self.pn)
        self.pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
        self.pn.regenerate_models()
        mean_psd = np.mean(self.pn['pore.diameter'])
        mean_tsd = np.mean(self.pn['throat.diameter'])

        return mean_psd,mean_tsd

    def visualize_pn(self,inlet,outlet):
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Inlet Pores')
        green_patch = mpatches.Patch(color='green', label='Outlet Pores')

        fig, ax = plt.subplots(figsize=(7, 7))
        op.visualization.plot_connections(network=self.pn, ax=ax)
        op.visualization.plot_coordinates(network=self.pn, pores=self.pn.pores('left'), 
                                            markersize=50, ax=ax, c='r')
        op.visualization.plot_coordinates(network=self.pn, pores=self.pn.pores('right'), 
                                            markersize=50, ax=ax, c='g')
        # plt.axis('off')
        plt.legend(handles=[red_patch,green_patch])
        plt.show()


    def cal_abs_perm(self,change_model=False):
        
        phase = op.phase.Phase(network=self.pn)
        phase['pore.viscosity']=1.0
        phase.add_model_collection(op.models.collections.physics.basic)
        if change_model:
            phase.add_model(
                model=op.models.physics.hydraulic_conductance.hagen_poiseuille, 
                propname='throat.conduit_hydraulic_conductance')
        phase.regenerate_models()

        inlet = self.pn.pores('left')
        outlet = self.pn.pores('right')
        flow = op.algorithms.StokesFlow(network=self.pn, phase=phase)
        flow.set_value_BC(pores=inlet, values=1,mode='overwrite')
        flow.set_value_BC(pores=outlet, values=0,mode='overwrite')
        flow.run()
        phase.update(flow.soln)

        Q = flow.rate(pores=inlet, mode='group')[0]
        A =  (self.img_size*self.voxel_size)**2
        L = self.img_size * self.voxel_size
        K = Q * L / A
        return K/0.98e-12*1000

    
    def init_physics(self,sf_tension=0.072,change_model=False):
        # should return self.air and physics
        self.air = op.phase.Air(network=self.pn,name='self.air')

        self.air['pore.surface_tension'] = sf_tension
        self.air['pore.contact_angle'] = 180.0
        self.air.add_model_collection(op.models.collections.phase.air)
        self.air.add_model_collection(op.models.collections.physics.basic)
        if change_model:
            self.air.add_model(
                model=op.models.physics.hydraulic_conductance.hagen_poiseuille, 
                propname='throat.conduit_hydraulic_conductance'
                )
        self.air.regenerate_models()

        self.water = op.phase.Water(network=self.pn,name='water')
        self.water.add_model_collection(op.models.collections.phase.water)
        self.water.add_model_collection(op.models.collections.physics.basic)
        if change_model:
            self.water.add_model(
                model=op.models.physics.hydraulic_conductance.hagen_poiseuille, 
                propname='throat.conduit_hydraulic_conductance'
                )
        self.water.regenerate_models()

    def ordinary_percolation(self,num_data=30):
        self.drn = op.algorithms.Drainage(network=self.pn, phase=self.air)
        self.drn.set_inlet_BC(self.pn.pores('left'))
        self.drn.run(num_data)
        pc,sat = self.drn.pc_curve()
        return [pc,sat]

    def __sat_update(self,network, nwp):
        pore_mask = nwp["pore.occupancy"] == 1
        throat_mask = nwp["throat.occupancy"] == 1
        sat_p = np.sum(network['pore.volume'][pore_mask])
        sat_t = np.sum(network['throat.volume'][throat_mask])
        sat1 = sat_p + sat_t
        bulk = network['pore.volume'].sum() + network['throat.volume'].sum()
        sat = sat1/bulk
        return sat

    def __Rate_calc(self,network, phase, inlet, outlet, conductance):
        phase.regenerate_models()
        St_p = op.algorithms.StokesFlow(network=network, phase=phase)
        St_p.settings._update({'conductance' : conductance})
        St_p.set_value_BC(pores=inlet, values=1,mode="overwrite")
        St_p.set_value_BC(pores=outlet, values=0,mode="overwrite")
        St_p.run()
        val = np.abs(St_p.rate(pores=inlet, mode='group'))
        return val

    def kr_simulation(self):
    
        diff_in = self.pn.pores('left')
        diff_out = self.pn.pores('right')

        model_mp_cond = op.models.physics.multiphase.conduit_conductance
        self.air.add_model(model=model_mp_cond, propname='throat.conduit_hydraulic_conductance',
                    throat_conductance='throat.hydraulic_conductance', mode='medium', regen_mode='deferred')
        self.water.add_model(model=model_mp_cond, propname='throat.conduit_hydraulic_conductance',
                    throat_conductance='throat.hydraulic_conductance', mode='medium', regen_mode='deferred')
        
        Snwparr = []
        relperm_nwp = []
        relperm_wp = []
        for Pc in np.unique(self.drn['pore.invasion_pressure']):
            self.air['pore.occupancy'] = self.drn['pore.invasion_pressure'] < Pc
            self.air['throat.occupancy'] = self.drn['throat.invasion_pressure'] < Pc
            self.water['pore.occupancy'] = 1 - self.air['pore.occupancy']
            self.water['throat.occupancy'] = 1 - self.air['throat.occupancy']
            self.air.regenerate_models()
            self.water.regenerate_models()
            sat_point = self.__sat_update(self.pn, self.air)
            Snwparr.append(sat_point)

            Rate_single_nwp = self.__Rate_calc(self.pn, self.air, diff_in, diff_out, conductance = 'throat.hydraulic_conductance')
            Rate_single_wp = self.__Rate_calc(self.pn, self.water, diff_in, diff_out, conductance = 'throat.hydraulic_conductance')
            Rate_mult_nwp = self.__Rate_calc(self.pn, self.air, diff_in, diff_out, conductance = 'throat.conduit_hydraulic_conductance')
            Rate_mult_wp = self.__Rate_calc(self.pn, self.water, diff_in, diff_out, conductance = 'throat.conduit_hydraulic_conductance')
            relperm_nwp.append(Rate_mult_nwp/Rate_single_nwp)
            relperm_wp.append(Rate_mult_wp/Rate_single_wp)

        # output pandas dataframe
        kr_data = {
            'snwp':Snwparr,
            'krnw':np.array(relperm_nwp).flatten(),
            'krw':np.array(relperm_wp).flatten()
            }

        kr_data_df = pd.DataFrame(kr_data)

        return kr_data_df

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




        
    

