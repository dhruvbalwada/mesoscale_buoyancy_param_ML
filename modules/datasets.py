import xarray as xr
from datatree import open_datatree
import xgcm
import helper_func as hf
import xbatcher
import numpy as np


class base_transformer: 
    def __init__(self, file_path, L, input_channels= ['U_x', 'U_y', 
                                                      'V_x', 'V_y', 
                                                      'Sx', 'Sy', 'Lfilt']):
        self.file_path = file_path
        self.L = L
        
        self.output_channels = ['Sfnx','Sfny'] 
        self.input_channels = input_channels
        self.dataset = None
        self.ML_dataset = None 
        self.mask = None
               
        
    def read_dataset(self):
        try:
            self.dataset = xr.open_zarr(self.file_path)
            print(f"Dataset loaded from {self.file_path}")
        except Exception as e:
            print(f"Error reading dataset: {e}")
            
            
class MITgcm_transformer(base_transformer): 
    
    def _center_dataset(self): 
        
        ds = self.dataset
        
        grid = xgcm.Grid(self.dataset, periodic='X')
        
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

        print('Centered')  
        return ds_centered
    
    
    def _get_MITgcm_mask(self):
        
        MITgcm_bucket ='gs://leap-persistent/dhruvbalwada/m2lines_transfer'
        ds_mask_tree = open_datatree(f'{MITgcm_bucket}/ML_data/ds_ML_masks', engine='zarr')
        Lstr = str(self.L)
        
        self.mask = ds_mask_tree[Lstr].to_dataset().maskC
        
        
    def transform_vars(self): 
        
        ds_centered = self._center_dataset()
        
        ds_centered['T_z'] = (ds_centered['T_z'].where(ds_centered['T_z']>=5e-5, 5e-5)) 

        # Add variables that are actually used in the ML model.
        ds_centered['Sx'] = -ds_centered['T_x']/ds_centered['T_z']
        ds_centered['Sy'] = -ds_centered['T_y']/ds_centered['T_z']

        ds_centered['Sfnx'] =  - ds_centered['uT']/ds_centered['T_z']
        ds_centered['Sfny'] =  - ds_centered['vT']/ds_centered['T_z']

        ds_centered['Lfilt'] = (float(self.L) + 0*ds_centered.T)

        self.ML_dataset = xr.merge([ds_centered[self.output_channels], 
                                    ds_centered[self.input_channels]])
    
    
    def remove_boundary(self, largest_remove=True): 
        
        self._get_MITgcm_mask()
        
        if largest_remove:
            self.ML_dataset = self.ML_dataset.where(self.mask>0.9999999, drop=True
                                 ).sel(Z=slice(-200, -2700), YC=slice(400*1e3,(2000 - 400)*1e3))
        else:
            self.ML_dataset = self.ML_dataset.where(self.mask>0.9999999, drop=True
                                 ).sel(Z=slice(-200, -2700), YC=slice(self.L*1e3,(2000 - self.L)*1e3))
        
        
        
    def subsample(self):
        sub_samp_fac = int( 400/ int(self.L) )
        
        self.ML_dataset = self.ML_dataset.isel( XC=slice(0,None,sub_samp_fac), 
                                                YC=slice(0,None,sub_samp_fac) )
    
    
    def load_norm_factors(self):
        path = '~/mesoscale_buoyancy_param_ML/normalize_factors/'
        
        try:
            self.norm_factors = xr.open_zarr(path+'STD_'+str(self.L)+'_km.nc')
            print('Loaded from saved norm nc')
            
        except: 
            self.norm_factors = hf.compute_std(self.ML_dataset, self.mask, slice(0,100))
            self.norm_factors.compute()
            self.norm_factors.to_zarr(path+'STD_'+str(self.L)+'_km.nc')
            print('Computed and saved norm nc')
        
        
    def normalize(self): 
        self.ML_dataset_norm = hf.normalize_ds(self.ML_dataset, self.norm_factors) 
        print('Normalized data')
        
    
    def convert_subsampled_normed(self):
        self.read_dataset()
        self.transform_vars()
        self.remove_boundary(largest_remove=True)
        self.subsample()
        self.load_norm_factors()
        self.normalize()
    
    def convert_normed(self):
        self.read_dataset()
        self.transform_vars()
        self.remove_boundary(largest_remove=True)
        #self.subsample()
        self.load_norm_factors()
        self.normalize()
        
    def generate_test_train_batches(self):
        
        self.ds_train, self.ds_test = hf.split_train_test(self.ML_dataset_norm)
        
        print("loading")
        self.ds_train.load();
        self.ds_test.load();
        
        print('stacking, droping nans, randomizing')
        self.ds_train = self.ds_train.stack(points=('XC','YC','Z','time'))
        self.ds_test = self.ds_test.stack(points=('XC','YC','Z','time'))
        
        self.ds_train = self.ds_train.dropna('points', subset=['Sfnx'])
        self.ds_test  = self.ds_test.dropna('points', subset=['Sfnx'])
        
        npoints_train = len(self.ds_train['Sfnx'])
        npoints_test = len(self.ds_test['Sfnx'])
        
        self.ds_train = self.ds_train.isel(points=np.random.choice(npoints_train, size=npoints_train, replace=False))
        self.ds_test  = self.ds_test.isel(points=np.random.choice(npoints_test, size=npoints_test, replace=False))
        
        
        self.bgen_train = xbatcher.BatchGenerator(ds = self.ds_train, 
                               input_dims={},
                               batch_dims={'points': int(75600)}   )

        self.bgen_test = xbatcher.BatchGenerator(ds = self.ds_test, 
                               input_dims={},
                               batch_dims={'points': int(75600)}   )
        
        print('Test and train batches split. Number of batches: ' + str(len(self.bgen_train)) + '-' + str(len(self.bgen_test)) )
        
#class MOM6_transformer(base_transformer):



     