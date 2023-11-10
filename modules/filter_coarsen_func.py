import xarray as xr
import xgcm
from scipy.ndimage import gaussian_filter
import numpy as np 
import MITgcm_advection_schemes as MITadv
import gcm_filters as gcmf

########################
########################
###### Filtering #######
########################
########################


##############################
##### Filtering functions ####
##############################
def gaussian_filter_wrap(data, sigma):
    '''
    Wrapper to pass additional arguments to scipy
    
    Parameters
    ----------
    data : array_like
        data on which filter will be applied.
    sigma: scalar
        Filter scale in terms of standard deviation of a gaussian
        
    Return
    ------
    Filtered array.
    '''
    return gaussian_filter(data, sigma=sigma, mode=['constant', 'wrap'])



def apply_gaussian_filter(da, Lfilter, dx, dims=['YC','XC']):
    '''
    Apply gaussian filter along specified dimension of xarray data array
    
    Parameters
    ----------
    da :  xarray dataarray
        Data on which filter will be applied. 
    sigma : scalar
        Filter scale in terms of standard deviation of a gaussian
    dims : list
        list of dimension on which to apply filter
    
    Return
    ------
    Filtered xarray data array.
    '''
    
    sigma = Lfilter/dx/np.sqrt(12)
    
    
    return xr.apply_ufunc(gaussian_filter_wrap, 
               da,
               sigma,
               input_core_dims=[dims,[]], 
               output_core_dims=[dims],
               dask='parallelized',
               vectorize=True)



def apply_gcm_filter(da, Lfilter, dx, wet_mask, dims=['YC','XC']):
    '''
    Apply gcm filter along specified dimension of xarray data array
    gcm-filters.readthedocs.io
    
    Parameters
    ----------
    da :  xarray dataarray
        Data on which filter will be applied. 
    sigma : scalar
        Filter scale in terms of standard deviation of a gaussian
    dims : list
        list of dimension on which to apply filter
        
    Return
    ------
    Filtered xarray data array.
    '''
    filter = gcmf.Filter(filter_scale= Lfilter,  
                     dx_min = dx, 
                     filter_shape=gcmf.FilterShape.GAUSSIAN,
                     grid_type=gcmf.GridType.REGULAR_WITH_LAND,
                     grid_vars = {'wet_mask':wet_mask})
    
    return filter.apply(da, dims=dims)
    
    
#############################################    
#### Functions to apply filters to MITgcm    
#############################################

def filter_MITgcm_dataset(ds, Lfilter, dx, filter_type='gauss'):
    '''
    Apply a filter on all variables (U,V,T,W) in a MITgcm dataset.
    
    Parameters
    ----------
    ds: xarray dataset
    
    Lfilter: scalar
        Filter scale in actual length [m]
    dx: scalar
        Grid size. [m]
    filter_type: string
        Type of filter.
    
    Return
    ------
    A dataset where all MITgcm variables have been filtered.
    
    '''
    # 
    ds_filt = xr.Dataset()

    #ds_filt = ds_filt.assign_coords(ds.coords)

    if filter_type == 'gauss':
        ds_filt['T'] = apply_gaussian_filter(ds['T'], Lfilter, dx)
        ds_filt['U'] = apply_gaussian_filter(ds['U'], Lfilter, dx,['YC', 'XG'])
        ds_filt['V'] = apply_gaussian_filter(ds['V'], Lfilter, dx, ['YG', 'XC'])
        ds_filt['W'] = apply_gaussian_filter(ds['W'], Lfilter, dx)
    else:
        xr.set_options(keep_attrs=True)
        ds_filt['T'] = apply_gcm_filter(ds['T'], Lfilter, dx, ds.maskC.astype('float'))
        ds_filt['U'] = apply_gcm_filter(ds['U'], Lfilter, dx, ds.maskW.astype('float'), ['YC', 'XG'])
        ds_filt['V'] = apply_gcm_filter(ds['V'], Lfilter, dx, ds.maskS.astype('float'), ['YG', 'XC'])
        ds_filt['W'] = apply_gcm_filter(ds['W'], Lfilter, dx, ds.maskC.isel(Z=0).astype('float'))
    
    return ds_filt

    
def filter_MITgcm_flux_dataset(ds, Lfilter, dx, filter_type='gauss', var=['uT','vT','wT']):
    '''
    Apply a filter on all variables in a MITgcm flux dataset.
    
    Parameters
    ----------
    ds: xarray dataset
    
    Lfilter: scalar
        Filter scale in actual length [m]
    dx: scalar
        Grid size. [m]
    filter_type: string
        Which filter to use.
    
    Return
    ------
    A dataset where all MITgcm variables have been filtered.
    
    '''
    # 

    ds_filt = xr.Dataset()
    
    if filter_type == 'gauss':
        ds_filt[var[0]] = apply_gaussian_filter(ds[var[0]], Lfilter, ['YC', 'XG'])
        ds_filt[var[1]] = apply_gaussian_filter(ds[var[1]], Lfilter, ['YG', 'XC'])
        ds_filt[var[2]] = apply_gaussian_filter(ds[var[2]], Lfilter)
        
    else:
        ds_filt[var[0]] = apply_gcm_filter(ds[var[0]], Lfilter, dx, ds.maskW.astype('float'), ['YC', 'XG'])
        ds_filt[var[1]] = apply_gcm_filter(ds[var[1]], Lfilter, dx, ds.maskS.astype('float'), ['YG', 'XC'])
        ds_filt[var[2]] = apply_gcm_filter(ds[var[2]], Lfilter, dx, ds.maskC.isel(Z=0).astype('float'))
    
    return ds_filt
    
    
### Functions to compute fluxes using LES definitions
def ds_LES_fluxes_filtered(ds, ds_filt, Lfilter, dx):
    '''
    Compute the full flux, large scale flux, sub-grid scale flux using a filter.
    
    Parameters
    ----------
    ds: xarray dataset
        contains the u,v,w,T variables
    ds_filt: xarray dataset
        contains filtered variables
    Lfilter: scalar
    dx: scalar 
    
    Return
    ------
    3 xarray dataset with all the flux components 
    
    '''
    
    # Full flux [filt(u,theta)]
    ds_flux = MITadv.transport2flux(MITadv.advection_scheme2(ds)).chunk({'XC':-1, 'YC':-1, 'XG':-1, 'YG':-1})
    ds_full_flux = filter_MITgcm_flux_dataset(ds_flux, Lfilter, dx, filter_type='gcm_filter')
    ds_full_flux = ds_full_flux.assign_coords(ds.coords) # need to do this because coordinates get lost.
    
    # Large scale flux [filt(u)filt(theta)]
    #ds_filt = filter_MITgcm_dataset(ds, Lfilter, dx)
    ds_large_scale_flux = MITadv.transport2flux(MITadv.advection_scheme2(ds_filt)).chunk({'XC':-1, 'YC':-1, 'XG':-1, 'YG':-1})
    ds_large_scale_flux = ds_large_scale_flux.assign_coords(ds.coords)
    
    # Small scale flux
    ds_small_scale_flux = ds_full_flux - ds_large_scale_flux
    
    return ds_full_flux, ds_large_scale_flux, ds_small_scale_flux
    
def ds_LES_flux_div_filtered(ds_full_flux, ds_large_scale_flux, ds_small_scale_flux): 
    '''
    Compute the divergence of the full, large scale, small scale fluxes
    
    Parameters
    ----------
    ds: xarray dataset
        contains the u,v,w,T variables
    Lfilter: scalar
    dx: scalar
    
    Return
    ------
    3 xarray datasets with the flux divergences and the components. 
    Returns the tendency, in units like [C/T]
    '''
    
    # Compute the different fluxes
    #ds_full_flux, ds_large_scale_flux, ds_small_scale_flux = ds_fluxes_filtered(ds, Lfilter, dx)
    
    # Compute the different flux divergences
    #ds_full_flux_div = MITadv.flux_div_components(ds_full_flux)
    #ds_full_flux_div['F_div'] = MITadv.flux_div(ds_full_flux)
    ds_full_flux_div = MITadv.flux_div(ds_full_flux, return_components=True)
    
    ds_large_scale_flux_div = MITadv.flux_div(ds_large_scale_flux, return_components=True)
    
    ds_small_scale_flux_div = MITadv.flux_div(ds_small_scale_flux, return_components=True)
    
    return ds_full_flux_div, ds_large_scale_flux_div, ds_small_scale_flux_div
   
    
########################
########################
###### COARSENING ######
########################
########################

def coarsen_MITgcm_datarray(da, coarsen_points, dims=['YC', 'XC']):
    '''
    Apply coarsening to MITgcm data arrays on C-grid. 
    
    Mixed coarsning is applied, which means that the strategy is different
    is the point is on cell-center vs cell-edge. 
    
    Parameters
    ----------
    da : xarray data array
    
    coarsen_points: scalar
        Number of points to coarsen over
        
    dims: list
        The horizontal dimensions, which determine if cell-center or cell-edge
        
    Return
    ------
    A dataarray that has been appropriately coarsened.
    
    '''
    if dims == ['YC', 'XC']:
        da_coarse = da.coarsen(XC=coarsen_points, YC=coarsen_points).mean()
        da_coarse['rA'] = da_coarse['rA']*coarsen_points**2
        #da_coarse['hFacC'] = da['
    elif dims == ['YG', 'XC']:
        da_coarse = da.isel(YG=slice(0,-1,coarsen_points)).coarsen(XC=coarsen_points).mean()
        da_coarse['dxG'] = coarsen_points*da_coarse['dxG'] 
        da_coarse['dyC'] = coarsen_points*da_coarse['dyC'] 
    elif dims == ['YC', 'XG']:
        da_coarse = da.isel(XG=slice(0,-1,coarsen_points)).coarsen(YC=coarsen_points).mean()
        da_coarse['dxC'] = coarsen_points*da_coarse['dxC'] 
        da_coarse['dyG'] = coarsen_points*da_coarse['dyG'] 
        
    return da_coarse

    
def coarsen_MITgcm_dataset(ds, coarsen_points): 
    '''
    Apply coarsening on a MITgcm dataset 
    (could be filtered or not)
    (only applies to U,V,T, W)
    
    Parameters
    ----------
    ds: xarray dataset
    
    coarsen_points: scalar
        Numer of points to coarse over
    
    Return
    ------
    A dataset where all MITgcm variables have been coarsened.
    '''
    
    ds_coarse = xr.Dataset()
    
    ds_coarse['T'] = coarsen_MITgcm_datarray(ds['T'], coarsen_points)
    ds_coarse['U'] = coarsen_MITgcm_datarray(ds['U'], coarsen_points, ['YC', 'XG'])
    ds_coarse['V'] = coarsen_MITgcm_datarray(ds['V'], coarsen_points, ['YG', 'XC'])
    ds_coarse['W'] = coarsen_MITgcm_datarray(ds['W'], coarsen_points)

    return ds_coarse


def coarsen_MITgcm_flux_dataset(ds, coarsen_points,var=['uT','vT','wT']):
    '''
    Apply coarsen on all variables in a MITgcm flux dataset.
    
    Parameters
    ----------
    ds: xarray dataset
        Flux dataset
    
    coarsen_points: scalar
    
    Return
    ------
    A dataset where all MITgcm flux variables have been coarsened.
    
    '''
    ds_coarse = xr.Dataset()
    ds_coarse[var[0]] = coarsen_MITgcm_datarray(ds[var[0]], coarsen_points, ['YC', 'XG'])#*coarsen_points
    ds_coarse[var[1]] = coarsen_MITgcm_datarray(ds[var[1]], coarsen_points, ['YG', 'XC'])#*coarsen_points
    ds_coarse[var[2]] = coarsen_MITgcm_datarray(ds[var[2]], coarsen_points)#*coarsen_points**2
    ds_coarse['hFacC'] = coarsen_MITgcm_datarray(ds['hFacC'], coarsen_points)
    
    return ds_coarse


## Next two functions are to-do, as we don't use this case at the moment. 
def ds_fluxes_coarsened(ds, coarsen_points): 
    return
    
def ds_flux_div_coarsened(ds, coarsen_points): 
    return
    
### Combining filtering and coarsening.
def ds_LES_flux_filtered_coarsened(ds, ds_filt_coarse, Lfilter, dx, coarsen_factor):
    '''
    Compute the full flux, large scale flux, sub-grid scale flux using a filter+coarsening.
    
    Parameters
    ----------
    ds: xarray dataset
        contains the u,v,w,T variables at high resolution
    ds_filt_coarse: xarray dataset
        contains the u,v,w,T variables at that have been filtered and coarsened
    Lfilter: scalar
    dx: scalar 
    coarsen_factor: integer
    
    Return
    ------
    3 xarray dataset with all the flux components
    '''
    coarsen_points = int(Lfilter/coarsen_factor/dx)
    
    # Full flux (<uT>)
    ds_flux      = MITadv.transport2flux(MITadv.advection_scheme2(ds)).chunk({'XC':-1, 'YC':-1, 'XG':-1, 'YG':-1})
    ds_full_flux = filter_MITgcm_flux_dataset(ds_flux, Lfilter, dx, filter_type='gcm_filter')
    ds_full_flux = ds_full_flux.assign_coords(ds_flux.coords)
    ds_full_flux = coarsen_MITgcm_flux_dataset(ds_full_flux, coarsen_points)
    
    # Large scale flux  (<u><T>)
    ds_large_scale_flux = MITadv.transport2flux(MITadv.advection_scheme2(ds_filt_coarse)).chunk({'XC':-1, 'YC':-1, 'XG':-1, 'YG':-1})
    
    # Small scale flux (<uT> - <u><T>)
    ds_small_scale_flux = ds_full_flux - ds_large_scale_flux
    
    return ds_full_flux, ds_large_scale_flux, ds_small_scale_flux
    

def ds_LES_flux_div_filtered_coarsened(ds_full_flux, 
                                   ds_large_scale_flux, 
                                   ds_small_scale_flux,
                                   return_components=False):
    '''
    Compute the divergence of the full, large scale, small scale fluxes
    
    Parameters
    ----------
    ds_full_flux, ds_large_scale_flux, ds_small_scale_flux: xarray datasets
        contain the LES style fluxes
    Lfilter: scalar
    dx: scalar
    
    Return
    ------
    3 xarray datasets with the flux divergences and the components. 
    Returns the tendency, in units like [C/T]
    '''
    
     # Compute the different flux divergences

    ds_full_flux_div = MITadv.flux_div(ds_full_flux, return_components=return_components)
    
    ds_large_scale_flux_div = MITadv.flux_div(ds_large_scale_flux, return_components=return_components)
    
    ds_small_scale_flux_div = MITadv.flux_div(ds_small_scale_flux, return_components=return_components)
    
    return ds_full_flux_div, ds_large_scale_flux_div, ds_small_scale_flux_div
    
    
