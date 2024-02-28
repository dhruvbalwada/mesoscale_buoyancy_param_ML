import gcm_filters as gcmf
from xgcm import Grid
import xarray as xr
import numpy as np 
import xrft


def div_uh(ds): 
    xgrid = Grid(ds, coords={'X': {'center': 'xh', 'right': 'xq'},
                         'Y': {'center': 'yh', 'right': 'yq'},
                         'Z': {'center': 'zl', 'outer': 'zi'} },
             periodic=['X'])
    
    dx = ds.xh.diff('xh')[0].values*1e3
    
    uh = ds.u * xgrid.interp(ds.h, 'X', boundary='periodic')
    vh = ds.v * xgrid.interp(ds.h, 'Y', boundary='extend')

    uh_x = xgrid.diff(uh, 'X', boundary='periodic')/dx
    vh_y = xgrid.diff(vh, 'Y', boundary='extend')/dx

    div_uh = uh_x + vh_y

    return div_uh, uh, vh

def GM(ds, kappa=1000.): 
    xgrid = Grid(ds, coords={'X': {'center': 'xh', 'right': 'xq'},
                         'Y': {'center': 'yh', 'right': 'yq'},
                         'Z': {'center': 'zl', 'outer': 'zi'} },
             periodic=['X'])

    dx = ds.xh.diff('xh')[0].values*1e3

    eta_x = xgrid.diff(ds.e.isel(zi=1), 'X', boundary='periodic')/dx
    eta_y = xgrid.diff(ds.e.isel(zi=1), 'Y', boundary='extend')/dx

    uphp_gm = np.zeros_like(ds.u)
    vphp_gm = np.zeros_like(ds.v)

    uphp_gm[:,1,:,:] = - kappa* eta_x
    vphp_gm[:,1,:,:] = - kappa* eta_y

    uphp_gm[:,0,:,:] =  kappa* eta_x
    vphp_gm[:,0,:,:] =  kappa* eta_y

    # The above formulae only work for 2 layers. With more layers
    # we need to first define stream function and then go through that.

    uphp_gm = xr.DataArray(uphp_gm, dims=ds.u.dims)
    vphp_gm = xr.DataArray(vphp_gm, dims=ds.v.dims)

    uphp_gm_x = xgrid.diff(uphp_gm, 'X', boundary='periodic')/dx
    vphp_gm_y = xgrid.diff(vphp_gm, 'Y', boundary='extend')/dx
   
    div_uphp_gm = uphp_gm_x + vphp_gm_y

    return div_uphp_gm, uphp_gm, vphp_gm

def NGM(ds, C=1.): 
    xgrid = Grid(ds, coords={'X': {'center': 'xh', 'right': 'xq'},
                         'Y': {'center': 'yh', 'right': 'yq'},
                         'Z': {'center': 'zl', 'outer': 'zi'} },
             periodic=['X'])

    dx = ds.xh.diff('xh')[0].values*1e3

    eta_x = xgrid.diff(ds.e.isel(zi=1), 'X', boundary='periodic')/dx
    eta_y = xgrid.diff(ds.e.isel(zi=1), 'Y', boundary='extend')/dx

    u_x = xgrid.diff(ds.u.isel(zi=1), 'X', boundary='periodic')/dx
    v_x = xgrid.diff(ds.v.isel(zi=1), 'X', boundary='periodic')/dx
    u_y = xgrid.diff(ds.u.isel(zi=1), 'Y', boundary='extend')/dx
    v_y = xgrid.diff(ds.v.isel(zi=1), 'Y', boundary='extend')/dx
    ## Need to get all to proper grids. 
    
    uphp_gm = np.zeros_like(ds.u)
    vphp_gm = np.zeros_like(ds.v)

    uphp_gm[:,1,:,:] = - kappa* eta_x
    vphp_gm[:,1,:,:] = - kappa* eta_y

    uphp_gm[:,0,:,:] =  kappa* eta_x
    vphp_gm[:,0,:,:] =  kappa* eta_y

    # The above formulae only work for 2 layers. With more layers
    # we need to first define stream function and then go through that.

    uphp_gm = xr.DataArray(uphp_gm, dims=ds.u.dims)
    vphp_gm = xr.DataArray(vphp_gm, dims=ds.v.dims)

    uphp_gm_x = xgrid.diff(uphp_gm, 'X', boundary='periodic')/dx
    vphp_gm_y = xgrid.diff(vphp_gm, 'Y', boundary='extend')/dx
   
    div_uphp_gm = uphp_gm_x + vphp_gm_y

    return div_uphp_gm
    

def filter_dataset(ds, Lfilter): 
    
    dx = ds.xh.diff('xh')[0].values*1e3
    wet_mask_C= (ds.h.isel(Time=0, zl=0)*0. + 1.).rename('wet_mask')
    wet_mask_x= (ds.u.isel(Time=0, zl=0)*0. + 1.).rename('wet_mask')
    wet_mask_y= (ds.v.isel(Time=0, zl=0)*0. + 1.).rename('wet_mask')
    
    
    filter_C = gcmf.Filter(filter_scale= Lfilter,  
                     dx_min = dx, 
                     filter_shape=gcmf.FilterShape.GAUSSIAN,
                     grid_type=gcmf.GridType.REGULAR)
                     #grid_type=gcmf.GridType.REGULAR_WITH_LAND,
                     #grid_vars = {'wet_mask': wet_mask_C})
    filter_x = gcmf.Filter(filter_scale= Lfilter,  
                     dx_min = dx, 
                     filter_shape=gcmf.FilterShape.GAUSSIAN,
                     grid_type=gcmf.GridType.REGULAR)
                     #grid_type=gcmf.GridType.REGULAR_WITH_LAND,
                     #grid_vars = {'wet_mask': wet_mask_x})

    filter_y = gcmf.Filter(filter_scale= Lfilter,  
                     dx_min = dx, 
                     filter_shape=gcmf.FilterShape.GAUSSIAN,
                     grid_type=gcmf.GridType.REGULAR)
                     #grid_type=gcmf.GridType.REGULAR_WITH_LAND,
                     #grid_vars = {'wet_mask': wet_mask_y})
    
    
    xgrid = Grid(ds, coords={'X': {'center': 'xh', 'outer': 'xq'},
                         'Y': {'center': 'yh', 'outer': 'yq'},
                         'Z': {'center': 'zl', 'outer': 'zi'} },
             periodic=['X'])
    
    ds_filt = xr.Dataset() # For storing the filtered data 

    #print('Started')
    ds_filt['h'] = filter_C.apply(ds.h, dims=['yh','xh']).rename('h')
    ds_filt['e'] = filter_C.apply(ds.e, dims=['yh','xh']).rename('e')

    ds_filt['filt_mask'] = wet_mask_C.where( (ds.yh > ds.yh.min() + Lfilter/1e3) & (ds.yh < ds.yh.max() - Lfilter/1e3))
    
    # u_c = xgrid.interp(ds.u.fillna(0), 'X')
    # v_c = xgrid.interp(ds.v.fillna(0), 'Y')
    #ds_filt['u'] = filter_C.apply(u_c, dims=['yh','xh']).rename('u')
    #ds_filt['v'] = filter_C.apply(v_c, dims=['yh','xh']).rename('v')
    ds_filt['u'] = filter_x.apply(ds.u, dims=['yh','xq']).rename('u')
    ds_filt['v'] = filter_y.apply(ds.v, dims=['yq','xh']).rename('v')

    
    ds_filt['div_uhbar'] = filter_C.apply(ds.div_uh, dims=['yh','xh'])
    ds_filt['uhbar'] = filter_x.apply(ds.uh, dims=['yh','xq'])
    ds_filt['vhbar'] = filter_y.apply(ds.vh, dims=['yq','xh'])
    
    ds_filt['div_ubarhbar'], ds_filt['ubarhbar'], ds_filt['vbarhbar'] = div_uh(ds_filt)
    
    ds_filt['div_uphp'] = ds_filt['div_uhbar'] - ds_filt['div_ubarhbar']
    ds_filt['uphp'] = ds_filt['uhbar'] - ds_filt['ubarhbar']
    ds_filt['vphp'] = ds_filt['vhbar'] - ds_filt['vbarhbar']

    ds_filt['div_uphp_gm'], ds_filt['uphp_gm'], ds_filt['vphp_gm'] = GM(ds_filt)
    # print('Filtered fields computed')
    
    # ds_filt['uh'] = ds.h*u_c
    # ds_filt['vh'] = ds.h*v_c
    
    # ds_filt['uu'] = u_c*u_c
    # ds_filt['vv'] = v_c*v_c
    
    # ds_filt['uh_bar'] = filter_C.apply(ds_filt['uh'], dims=['yh','xh']).rename('uh_bar')
    # ds_filt['vh_bar'] = filter_C.apply(ds_filt['vh'], dims=['yh','xh']).rename('vh_bar')
    
    # ds_filt['uu_bar'] = filter_C.apply(ds_filt['uu'], dims=['yh','xh']).rename('uu_bar')
    # ds_filt['vv_bar'] = filter_C.apply(ds_filt['vv'], dims=['yh','xh']).rename('vv_bar')
    
    # ds_filt['ubar_hbar'] = (ds_filt.h*ds_filt.u)
    # ds_filt['vbar_hbar'] = (ds_filt.h*ds_filt.v)
    
    # ds_filt['ubar_ubar'] = (ds_filt.u*ds_filt.u)
    # ds_filt['vbar_vbar'] = (ds_filt.v*ds_filt.v)
    
    # ds_filt['uh_sg'] = ds_filt['uh_bar'] - ds_filt['ubar_hbar']
    # ds_filt['vh_sg'] = ds_filt['vh_bar'] - ds_filt['vbar_hbar']
    
    # ds_filt['uu_sg'] = ds_filt['uu_bar'] - ds_filt['ubar_ubar']
    # ds_filt['vv_sg'] = ds_filt['vv_bar'] - ds_filt['vbar_vbar']
    
    # print('Fluxes computed')
    
    # ds_filt['dudx'] = xgrid.interp(xgrid.diff(ds_filt.u, 'X')/dx, 'X')
    # ds_filt['dvdx'] = xgrid.interp(xgrid.diff(ds_filt.v, 'X')/dx, 'X')
    # ds_filt['dudy'] = xgrid.interp(xgrid.diff(ds_filt.u, 'Y')/dx, 'Y')
    # ds_filt['dvdy'] = xgrid.interp(xgrid.diff(ds_filt.v, 'Y')/dx, 'Y')
    # ds_filt['slope_x'] = xgrid.interp(xgrid.diff(ds_filt.e, 'X')/dx, 'X')
    # ds_filt['slope_y'] = xgrid.interp(xgrid.diff(ds_filt.e, 'Y')/dx, 'Y')
    
    # print('Gradients computed')
    
    return ds_filt


def PE(ds): 
    gr = np.zeros(3)
    gr[0] = 9.81
    gr[1] = gr[0] * (ds.zl[1] - ds.zl[0])/ds.zl[0]
    gr[2] = 0.

    ds['gr'] = xr.DataArray(gr, dims={'zi'})

    PEi = 0.5 * ds.gr * (ds.e**2)

    PE = PEi.sum('zi')

    return PE, PEi

def eta_tend(ds, var = 'div_uh'): 

    eta_tend = np.zeros_like(ds.e)

    for i in range(len(ds.zl)):
        eta_tend[:,i,:,:] = - ds[var].isel(zl=slice(i,None)).sum('zl')

    eta_tend = xr.DataArray(eta_tend, dims=ds.e.dims)

    return eta_tend

def PE_tend(ds, var = 'dt_eta'): 
    gr = np.zeros(3)
    gr[0] = 9.81
    gr[1] = gr[0] * (ds.zl[1] - ds.zl[0])/ds.zl[0]
    gr[2] = 0.

    ds['gr'] = xr.DataArray(gr, dims={'zi'})

    PE_tend_i = (ds.gr * ds.e * ds[var])
    PE_tend = (ds.gr * ds.e * ds[var]).sum('zi')
    

    return PE_tend, PE_tend_i

def PE_tend_spectral(ds, var = 'dt_eta'): 
    gr = np.zeros(3)
    gr[0] = 9.81
    gr[1] = gr[0] * (ds.zl[1] - ds.zl[0])/ds.zl[0]
    gr[2] = 0.

    ds['gr'] = xr.DataArray(gr, dims={'zi'})

    #e_dft = xrft.dft(ds.e, dim='xh', true_phase=True, true_amplitude=True)
    #var_dft = xrft.dft(ds[var], dim='xh', true_phase=True, true_amplitude=True)

    e_dft = xrft.dft(ds.e, dim='xh', true_phase=True, true_amplitude=True)
    var_dft = xrft.dft(ds[var], dim='xh', true_phase=True, true_amplitude=True)
    
    PE_tend_i = (ds.gr * e_dft * np.conjugate(var_dft)).real
    PE_tend = PE_tend_i.sum('zi')
    
    return PE_tend, PE_tend_i

