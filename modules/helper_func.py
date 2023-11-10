import xarray as xr
import numpy as np
import xgcm

def center_data(ds):
    grid = xgcm.Grid(ds, periodic='X')

    ds_centered = ds.copy()

    ds_variables = list(ds.keys())

    for var in ds_variables:
        dims = ds_centered[var].dims
        if 'XG' in dims:
            ds_centered[var] = grid.interp(ds_centered[var], 'X')

        if 'YG' in dims: 
            ds_centered[var] = grid.interp(ds_centered[var], 'Y', boundary='extend')

        if 'Zl' in dims: 
            ds_centered[var] = grid.interp(ds_centered[var], 'Z', boundary='extend')
            
    return ds_centered


def compute_std(ds_C, mask, T_slice, dims=['XC','YC','Z','time'], tol = 0.999999): 
    
    ds_sel = ds_C.isel(time=T_slice)
        
    ds_std = ds_sel.where(mask>tol).std(dims)
    
    return ds_std


def normalize_da_std(da, std):
    """" Normalize by standard deviation """
    da_normalized = da/std
    return da_normalized

def denormalize_da_std(da, std):
    """" De-Normalize by standard deviation """
    da_denormalized = da*std
    return da_denormalized

def normalize_ds(ds, ds_std, method=normalize_da_std):
    ''' Normalized all variables in a dataset using a specific method '''
    ds_normalized = xr.Dataset()

    for key in ds.var():
        ds_normalized[key] = method(ds[key], ds_std[key])
        
    return ds_normalized
    
    
def masked_input_output(ds, mask, input_channels, output_channels, tol=0.99999):
    return ds[input_channels].where(mask>tol), ds[output_channels].where(mask>tol)
# helps drop unnecessary variables

def split_train_test(ds, train_slice = slice(0, 300), test_slice = slice(450, 495)):
    ds_train = ds.isel(time=train_slice)
    ds_test  = ds.isel(time=test_slice)
    
    return ds_train, ds_test

def R2(true, pred, var, dims = ['time','XC','YC','Z']):
    RSS = ((pred[var]  - true[var])**2).mean(dims)
    TSS = ((true[var])**2).mean(dims)
    R2 = 1 - RSS/TSS
    
    return R2