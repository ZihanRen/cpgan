import pandas as pd
from scipy.optimize import curve_fit
import numpy as np

def main(data):
    
    data_output = {}

    def corey_func_nw(x,n):
        return krnw_o * (x**n)

    def corey_func_w(x,n):
        return krw_o * ( x**n )

    df = pd.DataFrame()
    data_model = pd.DataFrame()

    df['snw'] = data['snw']
    df['kr_air'] = data['kr_air']
    df['kr_water'] = data['kr_water']

    # if kr_water is smaller than 1e-02, the water is irreducible
    df_new = df[ ~( (df['kr_water']<1e-02) ) ]
    # analyze the end point relative permeabililty and irreducible water saturation
    krw_o = 1
    krnw_o = df_new['kr_air'].iloc[-1]
    snwr = 0
    swir = 1-df_new['snw'].iloc[-1]
    
    # model relative permeability of air 
    y = df_new['kr_air'].values
    x = (df_new['snw'].values)/(1-swir)
    popt_nw, _ = curve_fit(corey_func_nw, x, y)

    # xx: normalized saturation
    # to transform back snw: simply
    # snw = xx*(1-swir)
    xx = np.arange(0,1.05,0.05) 
    snw_tnw = xx*(1-swir)
    data_model['snw_tnw'] = snw_tnw
    data_model['krnw'] = corey_func_nw(xx,*popt_nw)

    # model relative permeability of water
    y = df_new['kr_water'].values
    x = (1-df_new['snw'].values-swir)/(1-swir)
    popt_w, _ = curve_fit(corey_func_w, x, y)

    # visualize data - kr-air
    # xx: normalized saturation
    #  to transform back snw: simply
    # snw = xx*(1-swir)+
    xx = np.arange(0,1.05,0.05)
    snw_tw = 1-swir-xx*(1-swir)
    data_model['snw_tw'] = snw_tw
    data_model['krw'] = corey_func_w(xx,*popt_w)

    data_output['data.model'] = data_model
    data_output['swir'] = swir
    data_output['snwr'] = snwr
    data_output['krnw_o'] = krnw_o
    data_output['krw_o'] = krw_o

    return data_output
