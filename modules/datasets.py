import xarray as xr
from datatree import open_datatree, DataTree
import xgcm
import helper_func as hf
import xbatcher
import numpy as np
from datatree import DataTree
from datatree import open_datatree

### --- General functions for reading xarray dataarrays and opening as datatrees

def read_filtered_dataset(exp_name='DG', scale='100', assign_attrs=True): 
    '''
    Read data prepared for ml using filtering
    from zarr store and return xarray dataset object.
    
    Parameters:
    - exp_name: Name of the experiment ('DG' or 'P2L').
    - scale: Resolution scale to load (e.g., '100').
    - assign_attrs: Boolean to determine whether to assign attributes to the dataset.
    
    Returns:
    - ds: xarray dataset object with optional attributes assigned.
    '''
    
    MOM6_bucket = 'gs://leap-persistent/dhruvbalwada/MOM6/'
    
    if exp_name == 'DG': 
        ml_start = 'Double_Gyre/res5km/ml_data_'
        ml_end   = 'km_8_Aug_24.zarr'
    elif exp_name == 'P2L': 
        ml_start = 'Phillips2Layer/res4km_sponge10day_long_ml_data_'
        ml_end   = 'km_8_Aug_24.zarr'
        
    fname = f'{MOM6_bucket}{ml_start}{scale}{ml_end}'
    ds = xr.open_zarr(fname)
    
    if assign_attrs:
        # Example attributes to assign
        ds.attrs['simulation_name'] = exp_name
        ds.attrs['filter_scale'] = scale
        ds.attrs['source'] = fname
        ds.attrs['description'] = f"Dataset for {exp_name} experiment at {scale} km resolution"
    
    return ds
    
# def read_filtered_dataset(exp_name='DG', scale='100'): 
#     '''
#     Read data prepared for ml using filtering
#     from zarr store and return xarray dataset object.
#     '''
#     MOM6_bucket = 'gs://leap-persistent/dhruvbalwada/MOM6/'
    
#     if exp_name == 'DG': 
#         ml_start = 'Double_Gyre/res5km/ml_data_'
#         ml_end   = 'km_8_Aug_24.zarr'
#     elif exp_name == 'P2L': 
#         ml_start = 'Phillips2Layer/res4km_sponge10day_long_ml_data_'
#         ml_end   = 'km_8_Aug_24.zarr'
        
#     fname = f'{MOM6_bucket}{ml_start}{scale}{ml_end}'
#     ds = xr.open_zarr(fname)

#     return ds

def read_filtered_datatree(exp_name=['DG'], scales = ['50','100','200','400']):
    '''
    Read data from multiple scales and experiments (or other properties) at once 
    and expose using datatree.
    Inputs:
        exp_name : str or list of experiment names. 
                    Probably best to use lists, so the experiment name is carried around. 
        scales: list of experiment names
    Note:
        - Adding more layers will require a different algorithm for handling. 
    '''
    dtree_dict = {}

    if isinstance(exp_name, str):
        for L in scales:
            dtree_dict[L] = read_filtered_dataset(exp_name, L)
            
    elif isinstance(exp_name, list):
        dtree_exp = {}
        for exp in exp_name: 
            
            for L in scales:
                dtree_exp[L] = read_filtered_dataset(exp, L) 
                
            dtree_dict[exp] = DataTree.from_dict(dtree_exp)
                
    
    return DataTree.from_dict(dtree_dict)


### ------------ functions for manipulation of regular data 
def calculate_magnitudes(dtree):
    '''
    Calculate various magnitudes and add them to each dataset in the DataTree.
    '''

    dtree = dtree.map_over_subtree(lambda ds: ds.assign(magGradU = (ds.dudx**2 + ds.dudy**2 + ds.dvdx**2 + ds.dvdy**2)**0.5))
    dtree = dtree.map_over_subtree(lambda ds: ds.assign(magGradH = (ds.dhdx**2 + ds.dhdy**2)**0.5))
    dtree = dtree.map_over_subtree(lambda ds: ds.assign(magGradE = (ds.dedx**2 + ds.dedy**2)**0.5))
    dtree = dtree.map_over_subtree(lambda ds: ds.assign(magHFlux = (ds.uphp**2 + ds.vphp**2)**0.5))
    
    # The map_over_subtree magically replaces the loop below.
    
    # for exp_name in list(dtree.children):
    #     exp_node = dtree[exp_name]
    #     for scale_name in list(exp_node.children):
    #         scale_node = exp_node[scale_name]
    #         ds = scale_node
            
    #         # Compute magnitudes
    #         magGradU = (ds.dudx**2 + ds.dudy**2 + ds.dvdx**2 + ds.dvdy**2)**0.5
    #         magGradH = (ds.dhdx**2 + ds.dhdy**2)**0.5
    #         magGradE = (ds.dedx**2 + ds.dedy**2)**0.5
    #         magHFlux = (ds.uphp**2 + ds.vphp**2)**0.5
            
    #         # Add computed fields to the dataset
    #         scale_node['magGradU'] = magGradU
    #         scale_node['magGradH'] = magGradH
    #         scale_node['magGradE'] = magGradE
    #         scale_node['magHFlux'] = magHFlux

    return dtree


### ----- Classes and methods for ML related things Regular data -> training ready ML data 

class MLDataset():
    """
    A class to extract a machine learning dataset from a set of simulation datasets. 

    Attributes: 

    Methods:
    
    """
    def __init__(self,
                 simulation_names = ['DG','P2L'],
                 filter_scales    = ['50','100','200','400'], 
                 input_variables  = ['dudx','dvdx','dudy','dvdy','dhdx','dhdy'],
                 output_variables = ['uphp','vphp'],
                 use_mask         = True):

        self.simulation_names = simulation_names
        self.filter_scales = filter_scales
        
        self.LARGEST_FILTER_SCALE = int(self.filter_scales[-1])

        self.input_variables = input_variables
        self.output_variables = output_variables
        self.use_mask = use_mask
        # reserve input_channels for what actually goes into ML model,
        # there can be cases where variables and channels have slight differences 
        # e.g when stencil is wider (there might be other use cases too).

        # Do some method calls in init, which we think will be common across.
        self.load_simulation_data()
        #self.choose_ml_variables()
        

    def load_simulation_data(self):
        '''
        Open up simulaiton data as a DataTree
        '''
        try: 
            self.simulation_data = read_filtered_datatree(self.simulation_names,
                                                  self.filter_scales)
            
        except:
            print("Error reading simulation data")
            
        if self.use_mask:
                self.generate_h_mask()
                self.input_variables.append('h_mask')
                self.output_variables.append('h_mask')


    # Pre-processing methods
    ## These apply to individual datasets, which sit at end nodes of datatree. 

    def generate_h_mask(self, thin_limit = 5, thickness_variable='hbar'): 
        '''
        Often we need a thickness based mask :
            Don't consider points where the thickness is too small.
        '''
        self.simulation_data = self.simulation_data.map_over_subtree(lambda n: n.assign(h_mask = (n[thickness_variable]>= thin_limit) ))
                                                                     
        
    def choose_ml_variables(self):
        '''
        Select input and output variables, to be operated on going forward. 
        '''
        
        self.ml_input_dataset = self.simulation_data.map_over_subtree(lambda n: n[self.input_variables])
        self.ml_output_dataset = self.simulation_data.map_over_subtree(lambda n: n[self.output_variables])
        # This process of choosing subset from datatree is based on the discussion from this issue
        # https://github.com/xarray-contrib/datatree/issues/79 
        # Eventually if a subset function is introduced, the map_over_subtree can be removed.

    def add_ml_variables(self, add_filter_scale=True, add_mask=True):
        '''
        To add variables that don't already exist in the dataset.
        Examples can be length scales, or some grid related variable.
        '''
        # Add length scales
        if add_filter_scale:
            self.ml_input_dataset = self.ml_input_dataset.map_over_subtree(lambda n: n.assign(filter_scale = float(n.attrs['filter_scale'])*1e3 + 0.*n.dudx))

        # if add_mask:
        #     self.ml_input_dataset = self.ml_input_dataset.map_over_subtree(lambda n: n.assign(h_mask = 
            
        
    
    def create_wider_stencil(self, variables_to_widen=['dudx','dvdx','dudy','dvdy','dhdx','dhdy'], window_size=3, not_replace=True):
        '''
        Add a stencil of specified size around the prediction point for selected variables in the dataset.
    
        This method applies a rolling window operation to a subset of variables in each dataset node, creating a stencil 
        (e.g., 3x3, 5x5) around the prediction point. The method allows the user to either replace the original variables 
        with their widened versions or retain the original variables while adding the widened versions with modified names.
    
        Parameters:
        -----------
        variables_to_widen : list of str, optional
            A list of variable names to which the stencil operation will be applied. These variables will be 
            rolled over the specified window size. The default list includes 'dudx', 'dvdx', 'dudy', 'dvdy', 
            'dhdx', and 'dhdy'.
        
        window_size : int, optional
            The size of the rolling window to apply around the prediction point. The default value is 3, 
            which creates a 3x3 stencil.
    
        not_replace : bool, optional
            If True (default), the widened variables will be renamed with the suffix '_widened' to avoid 
            overwriting the original variables. If False, the original variables will be replaced with 
            their widened versions.
        '''
        
        def widen_stencil(n): 
            # Apply rolling and construct on the variables_to_widen
            widened = n[variables_to_widen].rolling(xh=window_size, yh=window_size, 
                                                    min_periods=1, center=True).construct(xh='Xn', yh='Yn')

            if not_replace:                
            # Rename the widened variables to avoid overwriting
                widened = widened.rename({var: f"{var}_widened" for var in variables_to_widen})
            
            # Merge the widened variables with the original dataset
            combined = xr.merge([widened, n], compat='override')
            #combined= widened

            return combined

        self.ml_input_dataset = self.ml_input_dataset.map_over_subtree(widen_stencil)

    
    def subsample_horizontally(self): 
        '''
        To maintain uniformity in data size we need to sub-sample the simulations with finer filter scales.
        '''
        def subsample_by_filter_scale(n):
            
            subsample_factor = int(self.LARGEST_FILTER_SCALE/float(n.attrs['filter_scale']))

            return n.isel(xh=slice(0, None, subsample_factor), 
                          yh=slice(0, None, subsample_factor))
                               
        self.ml_input_dataset  = self.ml_input_dataset.map_over_subtree(subsample_by_filter_scale)
        self.ml_output_dataset = self.ml_output_dataset.map_over_subtree(subsample_by_filter_scale)
                                                                       

    def h_mask_ml_variables(self):
        '''
        Mask as ml_variables using thickness masks.
        '''
        def only_h_mask_data_variables(n):
            n_masked = n.copy()
            
            for var in n.data_vars:
                if var != 'h_mask':
                    n_masked[var] = n[var].where(n['h_mask'])
            
            return n_masked
            
        if self.use_mask:
            self.ml_input_dataset = self.ml_input_dataset.map_over_subtree(only_h_mask_data_variables)
            self.ml_output_dataset = self.ml_output_dataset.map_over_subtree(only_h_mask_data_variables)
        else:
            raise ValueError("use_mask flag is not set to true.")
        
    def rotate_frame(self):
        '''
        Rotate variables from x-y coordinates to a flow dependent coordinate. 
        '''
        
        def rotate_vector():
            pass
        def rotate_tensor():
            pass
            
        
    def nondimensionalize():
        '''
        Non-dimensionalize input and output variables
        '''
        pass

    
    def scale_normalize():
        '''
        Do some scale normalization using fixed constants, to make everything order 1. 
        '''

        pass


    # Pre-processing pipeline methods
    ## Depending on the model the pre processing pipeline may be slightly different, the function below handles this.
    ## Also apply to datasets
    
    def default_preprocess_pipeline(self, window_size=1): 
        
        '''
        Some set of default operations read in as a list and done on data to pre-process. 
        Takes in simulation_data -> ml_data
        Note:
            This function can be customized for different models. 
        '''
        # The pipeline that is emerging is:
        ## 
        
        self.choose_ml_variables()
        self.add_ml_variables()
        self.create_wider_stencil(window_size=window_size, not_replace=True)
        self.subsample_horizontally()
        self.h_mask_ml_variables()
        


    
    # ML dataset setup pipelines
    ## These will work with datatree.
    def split_train_test_data(self):
        '''
        Split data into training and testing sets
        '''

        pass

    def generate_batches(self):
        '''
        Generate batches with some prescribed size. 
        '''
        pass
        
    def stack_physical_dimensions(self):
        pass 

    
    def concat_datatree_nodes(self):

        pass
        


        

### --- Older data classes --- 

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
            
    
class MOM6_transformer(base_transformer):
    def transform_vars(self, choice=1, keep_filt_scale=False, para_perp_out=False, eta_bottom=False):
        
        ds_temp = self.dataset.copy()
        
        ds_temp['Sx'] = ds_temp.slope_x.isel(zi=1)
        ds_temp['Sy'] = ds_temp.slope_y.isel(zi=1)

        ds_temp['hx'] = ds_temp.slope_x.isel(zi=2)
        ds_temp['hy'] = ds_temp.slope_y.isel(zi=2)
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

        if para_perp_out:
            print('Out para perp')
            
            ds_temp['Sfnx'] = ds_temp.uh_sg.isel(zl=1)
            ds_temp['Sfny'] = ds_temp.vh_sg.isel(zl=1)
            if eta_bottom==True:
                ds_temp['Sfnx'] = ds_temp.u2e1_sg
                ds_temp['Sfny'] = ds_temp.v2e1_sg
            
            S_mag = (ds_temp.Sx * ds_temp.Sx + ds_temp.Sy * ds_temp.Sy)**0.5
        
            # Unit vector components in S direction
            Shatx = ds_temp.Sx/S_mag
            Shaty = ds_temp.Sy/S_mag
        
            # Unit vector components perp to S direction
            Nhatx = - ds_temp.Sy/S_mag
            Nhaty = ds_temp.Sx/S_mag

            #ds_centered['Sfn_perp_scalar'] = (ds.Sfnx * Shatx + ds.Sfny * Shaty)
            #ds_centered['Sfn_para_scalar'] = (ds.Sfnx * Nhatx + ds.Sfny * Nhaty)

            # Being lazy I have rename things here such that now x represents component in direction of S 
            ds_temp['Sfnx'] = (ds_temp.Sfnx * Shatx + ds_temp.Sfny * Shaty)
            ds_temp['Sfny'] = (ds_temp.Sfnx * Nhatx + ds_temp.Sfny * Nhaty)
        else:
            ds_temp['Sfnx'] = ds_temp.uh_sg.isel(zl=1)
            ds_temp['Sfny'] = ds_temp.vh_sg.isel(zl=1)
            if eta_bottom==True:
                ds_temp['Sfnx'] = ds_temp.u2e1_sg
                ds_temp['Sfny'] = ds_temp.v2e1_sg
        
        ds_temp['Lfilt'] = (float(self.L) + 0*ds_temp['Sx'])

        if keep_filt_scale==False: 
            self.ML_dataset = xr.merge([ds_temp[self.output_channels], 
                                        ds_temp[self.input_channels]])
        else:
            self.ML_dataset = xr.merge([ds_temp[self.output_channels], 
                                        ds_temp[self.input_channels], 
                                        ds_temp['Lfilt']])
        
        
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

    
    def read_datatree(self, MOM6_bucket, file_names='res4km_sponge10day_long_ml_data_', 
                      largest_remove=True, H_mask=0, large_filt=400, keep_filt_scale=False, 
                      sub_sample=True, Lkeys = ['50','100','200','400'], window_size=1, para_perp_out=False,
                     eta_bottom=False): 

        self.window_size = window_size
        self.Lkeys = Lkeys
        dtree = {}
        
        for L in self.Lkeys:
            self.L = L
            self.file_path = f'{MOM6_bucket}{file_names}'+L+'km.zarr'
            self.read_dataset()
            self.transform_vars(keep_filt_scale=keep_filt_scale, para_perp_out=para_perp_out, eta_bottom=eta_bottom)
            self.mask_domain(H_mask)
            self.remove_boundary(largest_remove=largest_remove, large_filt=large_filt)

            if self.window_size>1: 
                self.ML_dataset = self.ML_dataset.rolling({'xh': window_size, 'yh': window_size},
                                                                     min_periods=1, 
                                                                     center=True).construct(xh='Xn',yh='Yn')
            
            if sub_sample:
                self.subsample()
            
            dtree[L] = self.ML_dataset.copy()
        
        self.datatree = DataTree.from_dict(dtree)
        
    def generate_test_train_batches(self, exp_name='P2L', normalize=True, input_dims={}): 
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
        
        npoints_train = len(self.ds_train['Sfnx'].points)
        npoints_test = len(self.ds_test['Sfnx'].points)
        
        self.ds_train.load();
        self.ds_test.load();
        
        self.ds_train = self.ds_train.isel(points=np.random.choice(npoints_train, size=npoints_train, replace=False))
        self.ds_test  = self.ds_test.isel(points=np.random.choice(npoints_test, size=npoints_test, replace=False))

        if normalize == True:
            self.load_norm_factors(exp_name)
            self.normalize()
        
        self.bgen_train = xbatcher.BatchGenerator(ds = self.ds_train, 
                               input_dims=input_dims,
                               batch_dims={'points': int(npoints_train/37)}   )

        self.bgen_test = xbatcher.BatchGenerator(ds = self.ds_test, 
                               input_dims=input_dims,
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



#### ---- Old code blocks
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
        
        
    def transform_vars(self, keep_filt_scale=False, para_perp_out=False): 
        
        ds_centered = self._center_dataset()
        
        ds_centered['T_z'] = (ds_centered['T_z'].where(ds_centered['T_z']>=5e-5, 5e-5)) 

        # Add variables that are actually used in the ML model.
        ds_centered['Sx'] = -ds_centered['T_x']/ds_centered['T_z']
        ds_centered['Sy'] = -ds_centered['T_y']/ds_centered['T_z']

        if para_perp_out: 
            print('Out para perp')
            ds_centered['Sfnx'] =  - ds_centered['uT']/ds_centered['T_z']
            ds_centered['Sfny'] =  - ds_centered['vT']/ds_centered['T_z']
            
            S_mag = (ds_centered.Sx * ds_centered.Sx + ds_centered.Sy * ds_centered.Sy)**0.5
        
            # Unit vector components in S direction
            Shatx = ds_centered.Sx/S_mag
            Shaty = ds_centered.Sy/S_mag
        
            # Unit vector components perp to S direction
            Nhatx = - ds_centered.Sy/S_mag
            Nhaty = ds_centered.Sx/S_mag

            #ds_centered['Sfn_perp_scalar'] = (ds.Sfnx * Shatx + ds.Sfny * Shaty)
            #ds_centered['Sfn_para_scalar'] = (ds.Sfnx * Nhatx + ds.Sfny * Nhaty)

            # Being lazy I have rename things here such that now x represents component in direction of S 
            ds_centered['Sfnx'] = (ds_centered.Sfnx * Shatx + ds_centered.Sfny * Shaty)
            ds_centered['Sfny'] = (ds_centered.Sfnx * Nhatx + ds_centered.Sfny * Nhaty)
            
        else:
            ds_centered['Sfnx'] =  - ds_centered['uT']/ds_centered['T_z']
            ds_centered['Sfny'] =  - ds_centered['vT']/ds_centered['T_z']

        ds_centered['Lfilt'] = (float(self.L) + 0*ds_centered.T)

        if keep_filt_scale==False: 
            self.ML_dataset = xr.merge([ds_centered[self.output_channels], 
                                        ds_centered[self.input_channels]])
        else:
            self.ML_dataset = xr.merge([ds_centered[self.output_channels], 
                                        ds_centered[self.input_channels], 
                                        ds_centered['Lfilt']])
    
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
    
    def read_datatree(self, M2LINES_bucket, keep_filt_scale=False, 
                      sub_sample=True, largest_remove=True, 
                      Lkeys = ['50','100','200','400'],
                      window_size=1, para_perp_out=False): 
        
        self.window_size = window_size
        self.Lkeys = Lkeys
        dtree = {}
        for L in self.Lkeys:
            self.L = L
            self.file_path = f'{M2LINES_bucket}/ML_data/ds_ML_'+L+'km_3D'
            self.read_dataset()
            self.transform_vars(keep_filt_scale=keep_filt_scale, para_perp_out=para_perp_out)
            self.remove_boundary(largest_remove=largest_remove)

            if self.window_size>1:
                self.ML_dataset = self.ML_dataset.rolling({'XC': window_size, 'YC': window_size},
                                                                     min_periods=1, 
                                                                     center=True).construct(XC='Xn',YC='Yn')
            
            if sub_sample:
                self.subsample()
            
            dtree[L] = self.ML_dataset.copy()
        
        self.datatree = DataTree.from_dict(dtree)
        self.masktree = open_datatree(f'{M2LINES_bucket}/ML_data/ds_ML_masks', engine='zarr')

    def generate_test_train_batches(self, normalize=True, input_dims={}): 
        self.ds_train, self.ds_test = hf.split_train_test(self.datatree)
        
        self.ds_train = self.ds_train.stack(points=('XC','YC','Z','time'))
        self.ds_test  = self.ds_test.stack(points=('XC','YC','Z','time'))
        
        self.ds_train = self._concat_scales(self.ds_train)
        self.ds_test = self._concat_scales(self.ds_test)
        
        self.ds_train = self.ds_train.dropna('points', subset=['Sfnx'])
        self.ds_test  = self.ds_test.dropna('points', subset=['Sfnx'])
        
        npoints_train = len(self.ds_train['points'])
        npoints_test = len(self.ds_test['points'])
        
        self.ds_train.load();
        self.ds_test.load();
        
        self.ds_train = self.ds_train.isel(points=np.random.choice(npoints_train, size=npoints_train, replace=False))
        self.ds_test  = self.ds_test.isel(points=np.random.choice(npoints_test, size=npoints_test, replace=False))

        if normalize == True:
            self.load_norm_factors()
            self.normalize()
        
        self.bgen_train = xbatcher.BatchGenerator(ds = self.ds_train, 
                               input_dims=input_dims,
                               batch_dims={'points': int(756000/2)}   )

        self.bgen_test = xbatcher.BatchGenerator(ds = self.ds_test, 
                               input_dims=input_dims,
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
        