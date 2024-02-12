import xarray as xr
from datatree import open_datatree
import xgcm
import helper_func as hf
import xbatcher
import numpy as np
from datatree import DataTree
from datatree import open_datatree


class base_transformer: 
    def __init__(self, 
                 file_path: str, 
                 L: str, 
                 input_channels = ['U_x', 'U_y', 
                                    'V_x', 'V_y', 
                                    'Sx', 'Sy', 'Lfilt']) -> None:
        
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
                                 ).sel(Z=slice(-200, -2700), YC=slice(int(self.L)*1e3,(2000 - int(self.L))*1e3))
        
        
    def subsample(self, attr='ML_dataset', sub_samp_fac=None):
        if sub_samp_fac==None:
            sub_samp_fac = int( 400/ int(self.L) )

        ds_attr = getattr(self, attr)

        setattr(self, attr, ds_attr.isel( XC=slice(0,None,sub_samp_fac), 
                                                YC=slice(0,None,sub_samp_fac) )
               )
        #self['var'] = self['var'].isel( XC=slice(0,None,sub_samp_fac), 
        #                                        YC=slice(0,None,sub_samp_fac) )
    
    
    def load_norm_factors(self, ML_name='single'):
        path = '~/mesoscale_buoyancy_param_ML/normalize_factors/'
        
        if ML_name == 'single': 
            try:
                self.norm_factors = xr.open_dataset(path+'MITgcm_STD_'+str(self.L)+'_km.nc')
                print('Loaded from saved norm nc')

            except: 
                self.norm_factors = hf.compute_std(self.ML_dataset, self.mask, slice(0,100))
                self.norm_factors.compute()
                self.norm_factors.to_netcdf(path+'MITgcm_STD_'+str(self.L)+'_km.nc')
                print('Computed and saved norm nc for '+ str(self.L))
                
        elif ML_name == 'all': 
            try:
                self.norm_factors = xr.open_dataset(path+'MITgcm_STD_'+str('all')+'_km.nc')
                print('Loaded from saved norm nc for all scales file.')

            except: 
                #self.norm_factors = hf.compute_std(self.ML_dataset, self.mask, slice(0,100))
                #self.norm_factors.compute()
                #self.norm_factors.open_dataset(path+'MITgcm_STD_'+str('all')+'_km.nc')
                print('Normalize factors for all are not saved. Run the ML training once for all to save these.')
        
        
    def normalize(self): 
        self.ML_dataset_norm = hf.normalize_ds(self.ML_dataset, self.norm_factors) 
        print('Normalized data')
        
    def convert_subsampled_normed(self, ML_name='single'):
        self.read_dataset()
        self.transform_vars()
        self.remove_boundary(largest_remove=True)
        self.subsample()
        self.load_norm_factors(ML_name)
        self.normalize()
    
    def convert_normed(self, ML_name='single', norm_factors=None):
        self.read_dataset()
        self.transform_vars()
        self.remove_boundary(largest_remove=False)
        #self.subsample()
        if norm_factors is None: 
            self.load_norm_factors(ML_name) # to be used when training
        else: 
            self.norm_factors = norm_factors # to be used when evaluating
        self.normalize()

    # The method below captures the behavior of the above two, slowly change out.
    def convert_any(self, window_size, ML_name='single', norm_factors=None, sub_sample=False, largest_remove=False): 
        self.window_size = window_size
        self.read_dataset()
        self.transform_vars()
        self.remove_boundary(largest_remove=largest_remove)
        #self.subsample()
        if window_size>1:
            self.ML_dataset = self.ML_dataset.rolling({'XC': window_size, 'YC': window_size},
                                                                     min_periods=1, 
                                                                     center=True).construct(XC='Xn',YC='Yn')
            
        if sub_sample:
            self.subsample()
            
        if norm_factors is None: 
            self.load_norm_factors(ML_name) # to be used when training
        else: 
            self.norm_factors = norm_factors # to be used when evaluating
        self.normalize()

        
    def generate_test_train_batches(self, input_dims={}):
        '''
        If windowing then use input_dims={'Xn':window_size,'Yn':window_size}
        '''
        
        self.ds_train, self.ds_test = hf.split_train_test(self.ML_dataset_norm)
        
        print("loading")
        self.ds_train.load();
        self.ds_test.load();
        
        print('stacking, droping nans, randomizing')
        self.ds_train = self.ds_train.stack(points=('XC','YC','Z','time'), create_index=False)
        self.ds_test = self.ds_test.stack(points=('XC','YC','Z','time'), create_index=False)
        
        self.ds_train = self.ds_train.dropna('points', subset=['Sfnx'])
        self.ds_test  = self.ds_test.dropna('points', subset=['Sfnx'])
        
        npoints_train = len(self.ds_train['Sfnx'].points)
        npoints_test = len(self.ds_test['Sfnx'].points)
        
        self.ds_train = self.ds_train.isel(points=np.random.choice(npoints_train, size=npoints_train, replace=False))
        self.ds_test  = self.ds_test.isel(points=np.random.choice(npoints_test, size=npoints_test, replace=False))
        
        
        self.bgen_train = xbatcher.BatchGenerator(ds = self.ds_train, 
                               input_dims=input_dims,
                               batch_dims={'points': int(75600)}   )

        self.bgen_test = xbatcher.BatchGenerator(ds = self.ds_test, 
                               input_dims=input_dims,
                               batch_dims={'points': int(75600)}   )
        
        print('Test and train batches split. Number of batches: ' + str(len(self.bgen_train)) + '-' + str(len(self.bgen_test)) )


    # This function can also be discarded in favor of the one above, if used appropriately.
    def generate_test_train_batches_windowed(self):
         
        self.ds_train, self.ds_test = hf.split_train_test(self.ML_dataset_norm)
        
        print("loading")
        self.ds_train.load();
        self.ds_test.load();
        
        print('stacking, droping nans, randomizing')
        self.ds_train = self.ds_train.stack(points=('XC','YC','Z','time'), create_index=False)
        self.ds_test = self.ds_test.stack(points=('XC','YC','Z','time'), create_index=False)
        
        self.ds_train = self.ds_train.dropna('points', subset=['Sfnx'])
        self.ds_test  = self.ds_test.dropna('points', subset=['Sfnx'])
        
        npoints_train = len(self.ds_train['Sfnx'].points)
        npoints_test = len(self.ds_test['Sfnx'].points)
        
        self.ds_train = self.ds_train.isel(points=np.random.choice(npoints_train, size=npoints_train, replace=False))
        self.ds_test  = self.ds_test.isel(points=np.random.choice(npoints_test, size=npoints_test, replace=False))
        
        
        self.bgen_train = xbatcher.BatchGenerator(ds = self.ds_train, 
                               input_dims={'Xn':self.window_size,'Yn':self.window_size},
                               batch_dims={'points': int(75600)}   )

        self.bgen_test = xbatcher.BatchGenerator(ds = self.ds_test, 
                               input_dims={'Xn':self.window_size,'Yn':self.window_size},
                               batch_dims={'points': int(75600)}   )
        
        print('Test and train batches split. Number of batches: ' + str(len(self.bgen_train)) + '-' + str(len(self.bgen_test)) )
        
        
class MITgcm_all_transformer(MITgcm_transformer):    
    
    def read_datatree(self, M2LINES_bucket): 
             
        self.Lkeys = ['50','100','200','400']
        dtree = {}
        for L in self.Lkeys:
            self.L = L
            self.file_path = f'{M2LINES_bucket}/ML_data/ds_ML_'+L+'km_3D'
            self.read_dataset()
            self.transform_vars()
            self.remove_boundary(largest_remove=True)
            self.subsample()
            
            dtree[L] = self.ML_dataset.copy()
        
        self.datatree = DataTree.from_dict(dtree)
        self.masktree = open_datatree(f'{M2LINES_bucket}/ML_data/ds_ML_masks', engine='zarr')

    def generate_test_train_batches(self): 
        self.ds_train, self.ds_test = hf.split_train_test(self.datatree)
        
        self.ds_train = self.ds_train.stack(points=('XC','YC','Z','time'))
        self.ds_test  = self.ds_test.stack(points=('XC','YC','Z','time'))
        
        self.ds_train = self._concat_scales(self.ds_train)
        self.ds_test = self._concat_scales(self.ds_test)
        
        self.ds_train = self.ds_train.dropna('points', subset=['Sfnx'])
        self.ds_test  = self.ds_test.dropna('points', subset=['Sfnx'])
        
        npoints_train = len(self.ds_train['Sfnx'])
        npoints_test = len(self.ds_test['Sfnx'])
        
        self.ds_train.load();
        self.ds_test.load();
        
        self.ds_train = self.ds_train.isel(points=np.random.choice(npoints_train, size=npoints_train, replace=False))
        self.ds_test  = self.ds_test.isel(points=np.random.choice(npoints_test, size=npoints_test, replace=False))
        
        self.load_norm_factors()
        
        self.normalize()
        
        self.bgen_train = xbatcher.BatchGenerator(ds = self.ds_train, 
                               input_dims={},
                               batch_dims={'points': int(756000/2)}   )

        self.bgen_test = xbatcher.BatchGenerator(ds = self.ds_test, 
                               input_dims={},
                               batch_dims={'points': int(756000/2)}   )
        
        print('Test and train batches split. Number of batches: ' + str(len(self.bgen_train)) + '-' + str(len(self.bgen_test)) )
        
     
    def normalize(self):
        self.ds_train = self.ds_train/ self.norm_factors
        self.ds_test = self.ds_test/ self.norm_factors
        
    def load_norm_factors(self): 
        path = '~/mesoscale_buoyancy_param_ML/normalize_factors/'
        
        try:
            self.norm_factors = xr.open_dataset(path+'MITgcm_STD_'+str('all')+'_km.nc')
            print('Loaded from saved norm nc for all.')
            
        except: 
            self.norm_factors = self.ds_train.std()
            self.norm_factors.compute()
            self.norm_factors.to_netcdf(path+'MITgcm_STD_'+str('all')+'_km.nc')
            print('Computed and saved norm nc for all.')
        
    @staticmethod    
    def _concat_scales(ds):
    
        return xr.concat([ds['50'].to_dataset(),
                          ds['100'].to_dataset(),
                          ds['200'].to_dataset(),
                          ds['400'].to_dataset()], 
                                   dim='points')
        
    
class MOM6_transformer(base_transformer):
    def transform_vars(self, choice=1):
        
        ds_temp = self.dataset.copy()
        
        ds_temp['Sx'] = ds_temp.slope_x.isel(zi=1)
        ds_temp['Sy'] = ds_temp.slope_y.isel(zi=1)
        #ds_test['Lfilt'] = ds_L.h.isel(zl=0)*0. + L

        # For the gradients we have some choices to make 

        #choice = 0 # 0 thickness weighted, 1 bottom layer, 2 top layer
        # Choose 0, since that is what we are currently using in online sims
        if choice == 0:
            ds_temp['U_x'] = (ds_temp.dudx * ds_temp.h).sum('zl') / ds_temp.h.sum('zl')
            ds_temp['U_y'] = (ds_temp.dudy * ds_temp.h).sum('zl') / ds_temp.h.sum('zl')
            ds_temp['V_x'] = (ds_temp.dvdx * ds_temp.h).sum('zl') / ds_temp.h.sum('zl')
            ds_temp['V_y'] = (ds_temp.dvdy * ds_temp.h).sum('zl') / ds_temp.h.sum('zl')
        elif choice ==1: 
            ds_temp['U_x'] = ds_temp.dudx.isel(zl=1)
            ds_temp['U_y'] = ds_temp.dudy.isel(zl=1)
            ds_temp['V_x'] = ds_temp.dvdx.isel(zl=1)
            ds_temp['V_y'] = ds_temp.dvdy.isel(zl=1)

        ds_temp['Sfnx'] = ds_temp.uh_sg.isel(zl=1)
        ds_temp['Sfny'] = ds_temp.vh_sg.isel(zl=1)
        
        ds_temp['Lfilt'] = (float(self.L) + 0*ds_temp['Sx'])
        
        
        self.ML_dataset = xr.merge([ds_temp[self.output_channels], 
                                    ds_temp[self.input_channels]])
        
    def remove_boundary(self, largest_remove=True, large_filt = 400): 
        
        Ymin = self.ML_dataset.yh.min().values
        Ymax = self.ML_dataset.yh.max().values
        
        if largest_remove:
            self.ML_dataset = self.ML_dataset.sel(yh=slice((Ymin + large_filt),(Ymax - large_filt)))
        else:
            #self.ML_dataset = self.ML_dataset.sel(yh=slice( (Ymin + int(self.L)),(Ymax - int(self.L))))
            self.ML_dataset = self.ML_dataset.sel(yh=slice((Ymin + large_filt),(Ymax - large_filt)))
            
    def mask_domain(self, H_mask=0): 
        mask = self.dataset.h.isel(Time=0, zl=1)>=H_mask
        self.ML_dataset = self.ML_dataset.where(mask)
        
    def subsample(self): 
        sub_samp_fac = int(400/ int(self.L))
        print('Subsampling')
        self.ML_dataset = self.ML_dataset.isel( xh=slice(0, None, sub_samp_fac), 
                                      yh=slice(0, None, sub_samp_fac) )
       
    def load_norm_factors(self, exp_name, ML_name='single'): 
        path = '~/mesoscale_buoyancy_param_ML/normalize_factors/'
        
        if ML_name == 'single': 
            try:
                self.norm_factors = xr.open_dataset(path+exp_name+'_STD_'+str(self.L)+'_km.nc')
                print('Loaded from saved norm nc for single scale.')
            except: 
                self.norm_factors = self.ML_dataset.isel(Time=slice(100, 200)).std()
                self.norm_factors.compute()
                self.norm_factors.to_netcdf(path+exp_name+'_STD_'+str(self.L)+'_km.nc')
                print('Computed and saved norm nc')
                
        elif ML_name == 'all': 
            try:
                self.norm_factors = xr.open_dataset(path+exp_name+'_STD_'+str('all')+'_km.nc')
                print('Loaded from saved norm nc for all')
            except: 
                print('Normalize factors for all are not saved. Run the ML training once for all to save these.')

    def normalize(self):
        self.ML_dataset_norm = hf.normalize_ds(self.ML_dataset, self.norm_factors) 
        print('Normalized data')
        
    def convert_subsampled_normed(self, exp_name):
        self.read_dataset()
        self.transform_vars()
        self.remove_boundary(largest_remove=True)
        self.subsample()
        self.load_norm_factors(exp_name)
        self.normalize()
    
    def convert_normed(self, exp_name='P2L', ML_name='single', norm_factors=None, large_filt=400, mask_wall=False, H_mask=0):
        self.read_dataset()
        self.transform_vars()
        self.remove_boundary(largest_remove=False, large_filt=large_filt)
        if norm_factors is None: 
            self.load_norm_factors(exp_name, ML_name) # to be used when training
        else: 
            self.norm_factors = norm_factors # to be used when evaluating
        if mask_wall:
            self.mask_domain(H_mask=H_mask)
        self.normalize()
        
    def generate_test_train_batches(self): 
        
        nTime = len(self.ML_dataset_norm.Time)
        
        fac = .9
        
        self.ds_train = self.ML_dataset_norm.isel( Time=slice(0, int(fac*nTime)) ).stack(points=('Time','xh','yh'))
        self.ds_test = self.ML_dataset_norm.isel( Time=slice(int(fac*nTime), None) ).stack(points=('Time','xh','yh'))
        
        print("loading")
        self.ds_train.load();
        self.ds_test.load();
        
        npoints_train = len(self.ds_train['Sfnx'])
        npoints_test = len(self.ds_test['Sfnx'])
        
        self.ds_train = self.ds_train.isel(points=np.random.choice(npoints_train, size=npoints_train, replace=False))
        
        self.ds_test = self.ds_test.isel(points=np.random.choice(npoints_test, size=npoints_test, replace=False))
        
        self.bgen_train = xbatcher.BatchGenerator(ds = self.ds_train, 
                               input_dims={},
                               batch_dims={'points': int(npoints_train/37)}   )

        self.bgen_test = xbatcher.BatchGenerator(ds = self.ds_test, 
                               input_dims={},
                               batch_dims={'points': int(npoints_test/5)}   )
        
        print('Test and train batches split. Number of batches: ' + str(len(self.bgen_train)) + '-' + str(len(self.bgen_test)) )
        
        
class MOM6_all_transformer(MOM6_transformer):    
    def read_datatree(self, MOM6_bucket, file_names='res4km_sponge10day_long_ml_data_', largest_remove=True, H_mask=0, large_filt=400): 
             
        self.Lkeys = ['50', '100','200','400']
        dtree = {}
        for L in self.Lkeys:
            self.L = L
            self.file_path = f'{MOM6_bucket}{file_names}'+L+'km.zarr'
            self.read_dataset()
            self.transform_vars()
            self.mask_domain(H_mask)
            self.remove_boundary(largest_remove=largest_remove, large_filt=large_filt)
            self.subsample()
            
            dtree[L] = self.ML_dataset.copy()
        
        self.datatree = DataTree.from_dict(dtree)
        
    def generate_test_train_batches(self, exp_name='P2L'): 
        nTime = len(self.datatree['100'].Time)
        
        fac = .9
        
        self.ds_train = self.datatree.isel( Time=slice(0, int(fac*nTime)) )
        self.ds_test = self.datatree.isel( Time=slice(int(fac*nTime), None) )
        
        if exp_name == 'P2L':
            self.ds_train['50'] = self.ds_train['50'].isel(Time=slice(0, int(nTime*.6)))
            self.ds_test['50'] = self.ds_test['50'].isel(Time=slice(0, int(nTime*.6)))
        
        self.ds_train = self.ds_train.stack(points=('Time','xh','yh'))
        self.ds_test = self.ds_test.stack(points=('Time','xh','yh'))
        
        self.ds_train = self._concat_scales(self.ds_train)
        self.ds_test = self._concat_scales(self.ds_test)
        
        self.ds_train = self.ds_train.dropna('points', subset=['Sfnx'])
        self.ds_test  = self.ds_test.dropna('points', subset=['Sfnx'])
        
        npoints_train = len(self.ds_train['Sfnx'])
        npoints_test = len(self.ds_test['Sfnx'])
        
        self.ds_train.load();
        self.ds_test.load();
        
        self.ds_train = self.ds_train.isel(points=np.random.choice(npoints_train, size=npoints_train, replace=False))
        self.ds_test  = self.ds_test.isel(points=np.random.choice(npoints_test, size=npoints_test, replace=False))
        
        self.load_norm_factors(exp_name)
        
        self.normalize()
        
        self.bgen_train = xbatcher.BatchGenerator(ds = self.ds_train, 
                               input_dims={},
                               batch_dims={'points': int(npoints_train/37)}   )

        self.bgen_test = xbatcher.BatchGenerator(ds = self.ds_test, 
                               input_dims={},
                               batch_dims={'points': int(npoints_test/5)}   )
        
        print('Test and train batches split. Number of batches: ' + str(len(self.bgen_train)) + '-' + str(len(self.bgen_test)) )
        
    def normalize(self):
        self.ds_train = self.ds_train/ self.norm_factors
        self.ds_test = self.ds_test/ self.norm_factors
        
    def load_norm_factors(self, exp_name): 
        path = '~/mesoscale_buoyancy_param_ML/normalize_factors/'
        
        try:
            self.norm_factors = xr.open_dataset(path+exp_name+'_STD_'+str('all')+'_km.nc')
            print('Loaded from saved norm nc for all.')

        except: 
            self.norm_factors = self.ds_train.std()
            self.norm_factors.compute()
            self.norm_factors.to_netcdf(path+exp_name+'_STD_'+str('all')+'_km.nc')
            print('Computed and saved norm nc for all.')
      
        
    @staticmethod    
    def _concat_scales(ds):
    
        return xr.concat([ds['50'].to_dataset(),
                          ds['100'].to_dataset(),
                          ds['200'].to_dataset(),
                          ds['400'].to_dataset()], 
                                   dim='points')
        