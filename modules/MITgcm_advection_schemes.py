import xarray as xr
import numpy as np
import xgcm 

def flux2transport(ds, input_var=['U', 'V','W'], output_var=['Utrans', 'Vtrans', 'Wtrans']): 
    '''
    Compute transport from velocities on c-grid vel points.
    This involves multiplying by the appropriate areas.
    
    Parameters
    ----------
    ds: xarray dataset
    
    Return
    ------
    Transport as xarray dataset
    
    '''
    
    ds_transport = xr.Dataset() 
    
    ds_transport[output_var[0]] = ds[input_var[0]]*ds.dyG*ds.drF*ds.hFacW
    ds_transport[output_var[1]] = ds[input_var[1]]*ds.dxG*ds.drF*ds.hFacS
    ds_transport[output_var[2]] = ds[input_var[2]]*ds.rA
    
    return ds_transport

def transport2flux(ds, input_var=['Fx', 'Fy', 'Fz'], tracer='T'):

    ds_flux = xr.Dataset()
    ds_flux = ds_flux.assign_coords(ds.coords)
    
    ds_flux['u'+tracer] = ds[input_var[0]]/(ds.dyG*ds.drF)*ds.hFacW
    ds_flux['v'+tracer] = ds[input_var[1]]/(ds.dxG*ds.drF)*ds.hFacS
    ds_flux['w'+tracer] = ds[input_var[2]]/ ds.rA
    
    return ds_flux
    
def flux_div(ds, flux_vars = ['uT', 'vT', 'wT'], return_components=False):
    '''
    Compute divergence of the flux.
    
    Parameters
    ----------
    ds: xarray dataset
        Dataset with flux components.
    flux_vars: List of strings
        List of names of for flux variables in order of x, y, z component. 
    Return
    ------
    Xarry dataarray with flux divergence (advective tendency). OR
    xarray dataset with gradient components of flux div and flux div.
    
    '''
    
    ds_flux_div = flux_div_components(ds, flux_vars)
    flux_div = ds_flux_div['F_dx'] + ds_flux_div['F_dy'] + ds_flux_div['F_dz']
    
    if return_components==False:
        return flux_div
    else:
        ds_flux_div['Fdiv'] = flux_div
        return ds_flux_div

def flux_div_components(ds, flux_vars = ['uT', 'vT','wT'], prop_trans_vars = ['Fx','Fy','Fz'] ):
    '''
    Compute components of the flux divergence. 
    
    Parameters
    ----------
    ds: xarray dataset
        Dataset with flux components.
    flux_vars: List of strings
        List of names of for flux variables in order of x, y, z component. 
    
    Return
    ------
    Xarray dataset with three gradients of fluxes in the three directions.
    We divide the flux by the cell volume, which implies that this is the contribution
    to the point tendency.
    (We return Gadv https://mitgcm.readthedocs.io/en/latest/algorithm/adv-schemes.html)
    
    '''
    
    grid = xgcm.Grid(ds, periodic='X')
    dV = ds.rA * ds.drF * ds.hFacC # Volume of each cell
    
    ds_prop_transport = flux2transport(ds, input_var=flux_vars, output_var=prop_trans_vars)
    
    flux_div_components = xr.Dataset()
    flux_div_components = flux_div_components.assign_coords(ds.coords)
    
    # compute tendency due to each gradient (dividing by dV)
    flux_div_components['F_dx'] = grid.diff(ds_prop_transport[prop_trans_vars[0]], 'X')/dV 
    flux_div_components['F_dy'] = grid.diff(ds_prop_transport[prop_trans_vars[1]],'Y', boundary='extend')/dV 
    flux_div_components['F_dz'] = - grid.diff(ds_prop_transport[prop_trans_vars[2]], 'Z', boundary='extend')/dV

    return flux_div_components

### MITgcm advection schemes 
def advection_scheme2(ds, tracer='T'):
    '''
    2nd order advection scheme.
        Copied from MITgcm code base
        
    Parameters
    ----------
    ds: xarray dataset
        Contains the 3 velocity components and a tracer.
    return_flux: boolean
        By default this function returns area integrated flux, but this can
        be changed to flux by setting this as True.
    
    Return
    ------
    Area integrated fluxes of the tracer in three direction. 
    '''
    
    grid = xgcm.Grid(ds, periodic='X')

    ds_transport = flux2transport(ds)
        
    ds_prop_transport = xr.Dataset()
    
    ds_prop_transport = ds_prop_transport.assign_coords(ds.coords)
    
    ds_prop_transport['Fx'] = ds_transport.Utrans * grid.interp(ds[tracer], 'X')
    ds_prop_transport['Fy'] = ds_transport.Vtrans * grid.interp(ds[tracer], 'Y', boundary='extend')
    ds_prop_transport['Fz'] = ds_transport.Wtrans * grid.interp(ds[tracer], 'Z', boundary='extend')
    
    #if return_flux==True:
    #    ds_flux = transport2flux(ds_prop_transport, var=['Fx','Fy','Fz'])
        #dV = ds.rA * ds.drF * ds.hFacC # Volume of each cell
        #ds_flux['uT'] = ds_prop_transport['Fx']/dV
        #ds_flux['vT'] = ds_prop_transport['Fy']/dV
        #ds_flux['wT'] = ds_prop_transport['Fz']/dV    
    
    #    return ds_flux
    #else:
    return ds_prop_transport

def advection_scheme3(ds, return_flux=False):
    '''
    3rd order advection scheme.
        Copied from MITgcm code base
        
    Parameters
    ----------
    ds: xarray dataset
        Contains the 3 velocity components and a tracer.
    return_flux: boolean
        By default this function returns area integrated flux, but this can
        be changed to flux by setting this as True.
    
    Return
    ------
    Area integrated fluxes of the tracer in three direction. 
    '''
    grid = xgcm.Grid(ds, periodic='X')
    ds_transport = vel2transport(ds)
    ds_flux = xr.Dataset()
    ds_flux = ds_flux.assign_coords(ds.coords)
    
    deliiT = grid.diff(grid.diff(ds['T'], 'X'), 'X')

    term1 = grid.interp(ds['T'] - 1/6*deliiT, 'X')
    term2 = grid.diff(1/6*deliiT, 'X')

    ds_flux['Fx'] = ds_transport.Utrans*term1 + 0.5*np.abs(ds_transport.Utrans)*term2

    deljjT = grid.diff(grid.diff(ds['T'], 'Y', boundary='extend'), 'Y', boundary='extend')
    term1 = grid.interp(ds['T'] - 1/6*deljjT, 'Y', boundary='extend')
    term2 = grid.diff(1/6*deljjT, 'Y', boundary='extend')

    ds_flux['Fy'] = ds_transport.Vtrans*term1 + 0.5*np.abs(ds_transport.Vtrans)*term2


    delkkT = grid.diff(grid.diff(ds['T'], 'Z', boundary='extend'), 'Z', boundary='extend')
    term1 = grid.interp(ds['T'] - 1/6*delkkT, 'Z', boundary='extend')
    term2 = grid.diff(1/6*delkkT, 'Z', boundary='extend')

    ds_flux['Fz'] = ds_transport.Wtrans*term1 + 0.5*np.abs(ds_transport.Wtrans)*term2
    
    if return_flux==True:
        dV = ds.rA * ds.drF * ds.hFacC # Volume of each cell
        ds_flux['Fx'] = ds_flux['Fx']/dV
        ds_flux['Fy'] = ds_flux['Fy']/dV
        ds_flux['Fz'] = ds_flux['Fz']/dV  
        
    return ds_flux

####