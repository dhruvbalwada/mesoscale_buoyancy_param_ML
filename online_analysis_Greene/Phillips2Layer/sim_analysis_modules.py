import xarray as xr
import numpy as np

class analyze_sims:
    def __init__(self, file_path):
        self.path = file_path 
        
        self.OS = xr.open_dataset(self.path+'ocean.stats.nc')
        self.prog = xr.open_dataset(self.path + 'prog.nc', decode_times=False)
        self.cont = xr.open_dataset(self.path + 'cont.nc', decode_times=False)
        
        self.m3_to_Sv = 1e-6
        try:
            self.calc_overturning()
        except: 
            print('no cont file.')
            
    def __repr__(self):
        return f"Data for sim at: {self.path}"
    

    ## Overturning 
    def calc_overturn_circ_resolved(self): 
        self.Vbar_resolved = self.cont.vh.sum('xh').isel(zl=0) * self.m3_to_Sv

    def calc_overturn_circ_param(self): 
        self.Vbar_param = (self.cont.vhGM.sum('xh')/self.cont.zl).isel(zl=0) * self.m3_to_Sv

    def calc_overturn_circ_total(self): 
        self.Vbar_total = (self.cont.vh.sum('xh') + self.cont.vhGM.sum('xh')/self.cont.zl).isel(zl=0) * self.m3_to_Sv


    def calc_overturning(self): 
        self.calc_overturn_circ_resolved()
        self.calc_overturn_circ_param()
        self.calc_overturn_circ_total()






def create_leaf(dic, keys, value):
    """
    Recursively create leaf nodes in the dictionary if they don't exist.
    """
    if len(keys) == 1:
        dic[keys[0]] = value
    else:
        key = keys[0]
        if key not in dic:
            dic[key] = {}
        create_leaf(dic[key], keys[1:], value)