import xarray as xr
from xmitgcm import open_mdsdataset
import xgcm 

def heat_budget_terms(ds):
    # Send ds to function and it will return heat budget terms
    # Some notes http://mitgcm.org/download/daily_snapshot/MITgcm/doc/Heat_Salt_Budget_MITgcm.pdf
    
    # To-dos's
    # - Add attributes to variables
    # - Budgets don't account for internal sources (needs more diagnostics vars)
    # - why budgets don't close in bottom cells? 
    
    Cp   = 3994.
    rho0 = 999.8
    dV = ds.rA * ds.drF * ds.hFacC # Volume of each cell
    
    grid = xgcm.Grid(ds, periodic='X')
    
    Surf_Trestore_tend = ds.TRELAX/(rho0 * Cp * ds.drF.isel(Z=0) * ds.hFacC.isel(Z=0))
    Surf_cor_tend = -ds.WTHMASS/(ds.drF.isel(Z=0) * ds.hFacC.isel(Z=0))
    
    Adv_tend = - (grid.diff(ds.ADVx_TH, 'X') + 
                  grid.diff(ds.ADVy_TH,'Y', boundary='extend') - 
                  grid.diff(ds.ADVr_TH, 'Z', boundary='extend')
                 )/dV
    Diff_tend = - (grid.diff(ds.DFxE_TH, 'X') + 
                  grid.diff(ds.DFyE_TH,'Y', boundary='extend') - 
                  grid.diff(ds.DFrE_TH, 'Z', boundary='extend')- 
                  grid.diff(ds.DFrI_TH, 'Z', boundary='extend')
                 )/dV

    Kpp_tend = - (- grid.diff(ds.KPPg_TH,'Z', boundary='extend'))/dV
    
    time_tend = ds.TOTTTEND/86400 # Units are per day in TOTTTEND
    
    
    ds_budgets = xr.Dataset() 
    
    ds_budgets['Surf_Trestore_tend'] = Surf_Trestore_tend
    ds_budgets['Surf_cor_tend'] = Surf_cor_tend
    ds_budgets['Adv_tend'] = Adv_tend
    ds_budgets['Diff_tend'] = Diff_tend
    ds_budgets['Kpp_tend'] = Kpp_tend
    ds_budgets['time_tend'] = time_tend
    
    ds_budgets['interior_tend'] = (Adv_tend + Diff_tend + Kpp_tend) #.isel(Z=slice(1,len(ds.Z)))
    ds_budgets['res_interior'] = ds_budgets['time_tend'] -  ds_budgets['interior_tend']# .isel(Z=slice(1,len(ds.Z))) -
   
    
    ds_budgets['surface_tend'] = (Adv_tend + Diff_tend + Kpp_tend + Surf_Trestore_tend + Surf_cor_tend).isel(Z=0)
    ds_budgets['res_surface']= ds_budgets['time_tend'].isel(Z=0) - ds_budgets['surface_tend']
    
    
    
    return ds_budgets
    
    
               
               