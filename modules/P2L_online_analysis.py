import xarray as xr 
import xrft
import matplotlib.pyplot as plot

class analyze_sims:
    def __init__(self, file_path):
        self.path = file_path 
        
        self.OS = xr.open_dataset(self.path+'ocean.stats.nc')
        self.prog = xr.open_dataset(self.path + 'prog.nc', decode_times=False)
        self.cont = xr.open_dataset(self.path + 'cont.nc', decode_times=False)
        
        self.m3_to_Sv = 1e-6
        try:
            self.calc_overturn_circ_resolved()
            self.calc_overturn_circ_param()
            self.calc_overturn_circ_total()
        except: 
            print('no cont file.')
            
    def __repr__(self):
        return f"Data for sim at: {self.path}"
    
        
    def calc_overturn_circ_resolved(self): 
        self.Vbar_resolved = self.cont.vh.sum('xh').isel(zl=0) * self.m3_to_Sv

    def calc_overturn_circ_param(self): 
        self.Vbar_param = (self.cont.vhGM.sum('xh')/self.cont.zl).isel(zl=0) * self.m3_to_Sv

    def calc_overturn_circ_total(self): 
        self.Vbar_total = (self.cont.vh.sum('xh') + self.cont.vhGM.sum('xh')/self.cont.zl).isel(zl=0) * self.m3_to_Sv
