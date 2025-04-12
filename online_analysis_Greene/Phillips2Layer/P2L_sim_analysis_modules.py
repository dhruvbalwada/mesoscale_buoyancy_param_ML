import xarray as xr
import numpy as np
from xgcm import Grid
from datatree import open_datatree, DataTree 
import gcm_filters as gcmf



def filter_dataset(ds, Lfilter): 

    ## Create grid area element 
    coords = {'X': {'center': 'xh', 'outer': 'xq'},
                'Y': {'center': 'yh', 'outer': 'yq'},
                'Z': {'center': 'zl', 'outer': 'zi'} }
    
    grid = Grid(ds, coords=coords, periodic=['X'])


    #dx = ds.xh.diff('xh').values[0] * 1e3
    #dy = ds.yh.diff('yh').values[0] * 1e3
    
    #area_t = dx*dy

    dx = 1e3*grid.diff(ds.xq,'X', boundary='extend') #* np.cos(ds.yh*np.pi/180)
    dy = 1e3*grid.diff(ds.yq,'Y', boundary='extend')
    area_t = dy*dx


    ## parameters
    h_min = 20 #m
    dx_min = 1
    filter_scale = Lfilter # lat/lon points

    ## Make masks
    
    # make wet mask for layer thickness
    wet_mask_h = (ds.h>h_min).astype('float32').rename('wet_mask')
    # Add edges to domain
    
    wet_mask_h = wet_mask_h.where(ds.yh <1595., other=0)
    #wet_mask_h = wet_mask_h.where(ds.xh>=0.03, other=0)

    # Make wet mask for interface points, by taking the mask for the layer below.
    wet_mask_e = wet_mask_h.copy()
    wet_mask_e = wet_mask_e.rename({'zl': 'zi'}).drop_vars('zi')
    #zeros_array = xr.DataArray(np.zeros((wet_mask_e.sizes['Time'], 1, wet_mask_e.sizes['yh'], wet_mask_e.sizes['xh'])),
    #                           dims=('Time', 'zi', 'yh', 'xh'))
    zeros_array = xr.DataArray(np.zeros((wet_mask_e.sizes['Time'], 1, wet_mask_e.sizes['yh'], wet_mask_e.sizes['xh'])),
                               dims=('Time','zi', 'yh', 'xh'))
    wet_mask_e = xr.concat([wet_mask_e, zeros_array], dim='zi').chunk({'zi':-1, 'xh':-1,'yh':-1})

    # make wet masks for interface but on zl points
    wet_mask_e_up = xr.DataArray(wet_mask_e.isel(zi=slice(0, -1)).data, 
                         dims=['Time','zl','yh','xh'])
    wet_mask_e_down = xr.DataArray(wet_mask_e.isel(zi=slice(1, None)).data, 
                         dims=['Time','zl','yh','xh'])


    filter_C_up = gcmf.Filter(filter_scale= filter_scale,  
                     dx_min = dx_min, 
                     filter_shape=gcmf.FilterShape.GAUSSIAN,
                     grid_type=gcmf.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED,
                     grid_vars = {'area':area_t,'wet_mask': wet_mask_e_up})

    filter_C_down = gcmf.Filter(filter_scale= filter_scale,  
                     dx_min = dx_min, 
                     filter_shape=gcmf.FilterShape.GAUSSIAN,
                     grid_type=gcmf.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED,
                     grid_vars = {'area':area_t,'wet_mask': wet_mask_e_down})

    filter_C_e = gcmf.Filter(filter_scale= filter_scale,  
                     dx_min = dx_min, 
                     filter_shape=gcmf.FilterShape.GAUSSIAN,
                     grid_type=gcmf.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED,
                     grid_vars = {'area':area_t,'wet_mask': wet_mask_e})

    ## Filter

    # Center velocities
    u_c = grid.interp(ds.u.fillna(0), 'X', boundary='extend')
    v_c = grid.interp(ds.v.fillna(0), 'Y', boundary='extend')

    # Bring interface values to zl, so they can be operated on with velocities
    e_up = xr.DataArray(ds.e.isel(zi=slice(0, -1)).data, 
                         dims=['Time','zl','yh','xh'])
    e_down = xr.DataArray(ds.e.isel(zi=slice(1, None)).data, 
                         dims=['Time','zl','yh','xh'])

    ebar_up = filter_C_up.apply(e_up, dims=['yh','xh'])
    ebar_down = filter_C_down.apply(e_down, dims=['yh','xh'])
    
    hbar = ebar_up - ebar_down 

    ebar = filter_C_e.apply(ds.e, dims=['yh','xh'])
    
    # u component
    ue_bar_up = filter_C_up.apply(u_c * e_up, dims=['yh','xh'])
    ue_bar_down = filter_C_down.apply(u_c * e_down, dims=['yh','xh'])

    ubar_up = filter_C_up.apply(u_c, dims=['yh','xh'])
    ubar_down = filter_C_down.apply(u_c, dims=['yh','xh'])
    ubar = ubar_up # mask for upper interface is same as that for a layer (if layer vanishes then interface is grounded).
    
    ubar_up_ebar_up =   ubar_up * ebar_up
    ubar_down_ebar_down =   ubar_down * ebar_down

    uh_bar = ue_bar_up - ue_bar_down
    ubar_hbar = ubar_up_ebar_up - ubar_down_ebar_down 

    uphp = uh_bar - ubar_hbar

    # v component
    ve_bar_up = filter_C_up.apply(v_c * e_up, dims=['yh','xh'])
    ve_bar_down = filter_C_down.apply(v_c * e_down, dims=['yh','xh'])

    vbar_up = filter_C_up.apply(v_c, dims=['yh','xh'])
    vbar_down = filter_C_down.apply(v_c, dims=['yh','xh'])
    vbar = vbar_up
    
    vbar_up_ebar_up = vbar_up * ebar_up
    vbar_down_ebar_down = vbar_down * ebar_down

    vh_bar = ve_bar_up - ve_bar_down
    vbar_hbar = vbar_up_ebar_up - vbar_down_ebar_down 

    vphp = vh_bar - vbar_hbar

    # Add ubar, vbar, hbar, ebar, uphp, vphp to a dataset
    ds_filt = xr.Dataset(coords=ds.coords)
    ds_filt['ubar'] = ubar
    ds_filt['vbar'] = vbar
    ds_filt['hbar'] = hbar
    ds_filt['ebar'] = ebar
    ds_filt['uphp'] = uphp
    ds_filt['vphp'] = vphp

    ds_filt = add_gradients(ds_filt, grid,dx,dy)
    
    return ds_filt


def add_gradients(ds_filt, grid,dx,dy): 
    ds_filt['dudx'] = grid.interp(grid.diff(ds_filt.ubar, 'X'), 'X')/dx
    ds_filt['dvdx'] = grid.interp(grid.diff(ds_filt.vbar, 'X'), 'X')/dx
    ds_filt['dudy'] = grid.interp(grid.diff(ds_filt.ubar, 'Y'), 'Y')/dy
    ds_filt['dvdy'] = grid.interp(grid.diff(ds_filt.vbar, 'Y'), 'Y')/dy
    
    ds_filt['dedx'] = grid.interp(grid.diff(ds_filt.ebar, 'X'), 'X')/dx
    ds_filt['dedy'] = grid.interp(grid.diff(ds_filt.ebar, 'Y'), 'Y')/dy

    ds_filt['dhdx'] = grid.interp(grid.diff(ds_filt.hbar, 'X'), 'X')/dx
    ds_filt['dhdy'] = grid.interp(grid.diff(ds_filt.hbar, 'Y'), 'Y')/dy

    return ds_filt


def coarsen_dataset(ds, coarsen_points): 
    print('coarsening')
    return ds.coarsen(xh=coarsen_points, yh=coarsen_points, boundary='trim').mean()



def calc_SG_forcing(ds):

    if 'RV' in ds:
        grid = Grid(ds, coords={'X': {'center': 'xh', 'right': 'xq'},
                                'Y': {'center': 'yh', 'right': 'yq'},
                                'Z': {'inner': 'zl', 'outer': 'zi'} }, periodic='X')
        
        dx = ds.xh.diff('xh').values[0] * 1e3
        dy = ds.yh.diff('yh').values[0] * 1e3
        dA = dx*dy
        
        ds['SG_forc'] = -1/dA* (grid.diff(ds.uhTrANN, 'X') + grid.diff(ds.vhTrANN, 'Y', boundary='fill'))

    else:
        pass

    return ds


#def add_PE_reduction_rate(ds): 


def add_PE_reduction_rate(ds): 
    if 'RV' in ds: 
        ds = ds.copy()

        dx = ds.xh.diff('xh').values[0] * 1e3
        dy = ds.yh.diff('yh').values[0] * 1e3
        dA = dx*dy
        #print(dA)

        ds_Tsel = ds.isel(Time=slice(72, None)) 
        
        ds['APE_reduce_rate'] = dA* 1031. * 1.96e-02* (ds_Tsel.Fx * ds_Tsel.dhdx + ds_Tsel.Fy * ds_Tsel.dhdy).isel(zl=1).mean('Time').sum(['xh', 'yh'])

        ds['APE_reduce_rate_mean'] = dA* 1031. * 1.96e-02* (ds_Tsel.Fx.mean('Time') * ds_Tsel.dhdx.mean('Time') + 
                                                           ds_Tsel.Fy.mean('Time') * ds_Tsel.dhdy.mean('Time')
                                                          ).isel(zl=1).sum(['xh', 'yh'])
        
        ds['APE_reduce_rate_eddy'] = ds['APE_reduce_rate'] - ds['APE_reduce_rate_mean']
    else: 
        pass
    return ds


def APE(interface):
    '''
    Returns APE in units of kinetic energy per unit mass, i.e.
    m^3/s^2
    '''
    #interface_rest = xr.DataArray([    0.,   -25.,   -75.,  -175.,  -300.,  -450.,  -625.,  -825.,
    #   -1050., -1300., -1600., -1950., -2350., -2850., -3400., -4000.], dims='zi')
    #g = xr.DataArray([1.0e+01, 2.1e-03, 3.9e-03, 5.4e-03, 5.8e-03, 5.8e-03, 5.7e-03,
    #   5.3e-03, 4.8e-03, 4.2e-03, 3.7e-03, 3.1e-03, 2.4e-03, 1.7e-03,
    #   1.1e-03, 0.0e+00], dims='zi')
    interface_rest = xr.DataArray([0. , -500., -2000.], dims='zi')
    g = xr.DataArray([9.8, 1.96e-02, 9.8e+01], dims='zi')
    
    coordinate_of_bottom = interface.isel(zi=-1).drop_vars(['zi'])
    
    hint = interface - interface_rest
    
    # Where bottom is upper than the rest interface
    hbot = np.maximum(coordinate_of_bottom - interface_rest,0)
    
    APE_instant = (0.5* g * (hint**2)) 
    APE_constant = (0.5* g * (hbot**2)) 
    
    return (APE_instant - APE_constant).sum('zi')


def add_energy_metrics_filt_coarse(ds): 
    dx = ds.xh.diff('xh').values[0] * 1e3
    dy = ds.yh.diff('yh').values[0] * 1e3
    dA = dx*dy

    ds['KE'] = (0.5* ds.hbar *ds.zl *(ds.ubar**2 + ds.vbar**2) *dA).sum('zl').sum(['xh', 'yh'])
    
    ds_mean = ds.isel(Time=slice(72, None)).mean('Time')
    
    ds['MKE'] = (0.5* ds_mean.hbar * ds_mean.zl *(ds_mean.ubar**2 + ds_mean.vbar**2) *dA).sum('zl').sum(['xh', 'yh'])

    ds['EKE'] = ds['KE'] - ds['MKE']

    ds['APE'] = (APE(ds.ebar)* 1031.*dA).sum(['xh', 'yh'])
    
    ds['MAPE'] = (APE(ds_mean.ebar)* 1031.*dA).sum(['xh', 'yh'])

    ds['EAPE'] = ds['APE'] - ds['MAPE']
    
    return ds


def add_energy_metrics(ds): 

    if 'RV' in ds: 
        ds = ds.copy()

        if len(ds.xq) > len(ds.xh):
            ds = ds.isel(xq=slice(1,None), yq=slice(1,None))
        
        grid = Grid(ds, coords={'X': {'center': 'xh', 'right': 'xq'},
                        'Y': {'center': 'yh', 'right': 'yq'},
                        'Z': {'inner': 'zl', 'outer': 'zi'} }, periodic='X')

        dx = ds.xh.diff('xh').values[0] * 1e3
        dy = ds.yh.diff('yh').values[0] * 1e3
        dA = dx*dy
        #print(dx)
        
        ds['KE'] = (0.5 * ( grid.interp(ds.u, 'X', boundary='extend')**2 +  
                            grid.interp(ds.v, 'Y')**2 ) * 
                      ds.h * dA * ds.zl).sum('zl').sum(['xh','yh'])
        
        ds_mean = ds.isel(Time=slice(72, None)).mean('Time')
        
        ds['MKE'] = (0.5 * ( grid.interp(ds_mean.u, 'X', boundary='extend')**2 +  
                            grid.interp(ds_mean.v, 'Y')**2 ) 
                      * ds_mean.h * dA * ds.zl).sum('zl').sum(['xh','yh'])

        ds['EKE'] = ds['KE'] - ds['MKE']

        ds['APE'] = (APE(ds.e)* 1031.*dA).sum(['xh', 'yh'])
        ds['MAPE'] = (APE(ds_mean.e)* 1031.*dA).sum(['xh', 'yh'])
        
        ds['EAPE'] = ds['APE'] - ds['MAPE']
        
        
    else:
        pass

    return ds
    
def add_transport_metrics(ds): 
    if 'Time_bnds' in ds: 
        ds = ds.copy()

        Tsel = slice(72, None) 

        m3_to_Sv = 1e-6

        ds['Vbar_resolved'] = ds.vh.isel(Time=Tsel).mean('Time').sum('xh') * m3_to_Sv
        ds['Vbar_param'] = ds.vhTrANN.isel(Time=Tsel).mean('Time').sum('xh') * m3_to_Sv
        ds['Vbar_total'] = ds['Vbar_resolved'] + ds['Vbar_param'] 
    else:
        pass
    return ds

def add_zonal_eddy_mean_transport_metrics(ds): 
    if 'Time_bnds' in ds: 
        ds = ds.copy()

        Tsel = slice(72, None) 

        m3_to_Sv = 1e-6

        ds = ds.isel(xq=slice(1,None), yq=slice(1,None))
        
        grid = Grid(ds, coords={'X': {'center': 'xh', 'right': 'xq'},
                    'Y': {'center': 'yh', 'right': 'yq'},
                    'Z': {'inner': 'zl', 'outer': 'zi'} }, periodic='X')

        dx = ds.xh.diff('xh').values[0] * 1e3
        dy = ds.yh.diff('yh').values[0] * 1e3
        
        dA = dx*dy
        
        ds['Vbar_mean'] =  ( ds.v.isel(Time=Tsel).mean(['xh','Time']) * grid.interp(ds.h.isel(Time=Tsel).mean(['xh','Time']), 'Y') ) * dx * m3_to_Sv
        ds['Vbar_eddy'] = ds['Vbar_resolved'] - ds['Vbar_mean']
    else :
        pass

    return ds

def add_HR_transport(ds): 
    
    ds = ds.isel(xq=slice(1,None), yq=slice(1,None))
        
    grid = Grid(ds, coords={'X': {'center': 'xh', 'right': 'xq'},
                    'Y': {'center': 'yh', 'right': 'yq'},
                    'Z': {'inner': 'zl', 'outer': 'zi'} }, periodic='X')
    
    dx = ds.xh.diff('xh').values[0] * 1e3
    dy = ds.yh.diff('yh').values[0] * 1e3
    
    dA = dx*dy
    
    vh = ds.h * grid.interp(ds.v, 'Y')
    
    Tsel = slice(72, None) 
    
    m3_to_Sv = 1e-6
    
    ds['Vbar_resolved'] = (vh * dx).isel(Time=Tsel).mean('Time').sum('xh') * m3_to_Sv

    return ds

def add_filt_coarse_transport(ds):
    
    dx = ds.xh.diff('xh').values[0] * 1e3
    dy = ds.yh.diff('yh').values[0] * 1e3
    
    print(dx)
    
       
    Tsel = slice(72, None) 
    
    m3_to_Sv = 1e-6

    ds['Vbar_resolved'] = (ds.vbar * ds.hbar * dx).isel(Time=Tsel).mean('Time').sum('xh') * m3_to_Sv

    ds['Vbar_SG'] = (ds.vphp * dx).isel(Time=Tsel).mean('Time').sum('xh') * m3_to_Sv

    return ds



def load_sims(exp_dir, model_types, res, C_ANN, C_GM):
    model_type_dic = {}
    for ANN_type in model_types: 
        
        res_dic = {}
        for r in res: 
            r = str(r)
            #exp_dic[ANN_type][r] = {}
            if ANN_type == 'ANN':
                C = C_ANN
            elif ANN_type == 'GM1000':
                C = C_GM
    
             
            coeff_dic = {}
            for coeff in C: 
                coeff = str(coeff)
                #exp_dic[ANN_type][r][coeff] = {}
                exp_name = 'res_' + str(r) + 'km_' + str(ANN_type) + '_' + str(coeff)
                print('Reading :' + exp_name)
                run_dic = {}
                
                run_dic['prog']     = xr.open_mfdataset(exp_dir + 'runs/' + exp_name + '/OUTPUT/prog_*.nc', decode_times=False)
                run_dic['ave_prog'] = xr.open_mfdataset(exp_dir + 'runs/' + exp_name + '/OUTPUT/ave_prog_*.nc', decode_times=False)
                run_dic['oce_stats'] = xr.open_dataset(exp_dir + 'runs/' + exp_name + '/OUTPUT/ocean.stats.nc', decode_times=False)
                
                coeff_dic[coeff] = DataTree.from_dict(run_dic)
                
            res_dic[r] = DataTree.from_dict(coeff_dic)
    
        model_type_dic[ANN_type] = DataTree.from_dict(res_dic)
    
    exp_tree = DataTree.from_dict(model_type_dic) 
    return exp_tree

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