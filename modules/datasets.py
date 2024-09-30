import xarray as xr
from datatree import open_datatree, DataTree
import xgcm
import helper_func as hf
import xbatcher
import random
import numpy as np
from datatree import DataTree
from datatree import open_datatree
import time
import jax.numpy as jnp

seed = 42

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
    


def read_filtered_datatree(exp_name=['DG'], scales = ['50','100','200','400']):
    '''
    Read data from multiple scales and experiments (or other properties) at once 
    and expose using datatree.
    Inputs:
        exp_name : str or list of experiment names. 
                    Probably best to use lists, so the experiment name is carried around. 
        scales: list of experiment names
    Note:
        - Adding more data form (e.g. FGR etc) will require a different algorithm for handling. 
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

class SimulationData:
    """
    A class to read in simulation datasets, and apply a set of operations to pre-process the data for ML.
    This includes adding variables, creating wider stencils, rotating frames, and non-dimensionalizing variables.

    This method reads in simulation data as a xarray DataTree and returns simulation_data as a xarray DataTree. 
    This simulation_data can then be parsed to get the variables needed for ML.

    Attributes: 
        - simulation_names: List of simulation names to read in
        - filter_scales: List of filter scales to read in
        - simulation_data: xarray DataTree object with simulation data

    Methods:
        - load_simulation_data: Open up simulation data as a DataTree
        - generate_h_mask: Generate a thickness based mask
        - add_variables: Add variables that don't already exist in the dataset
        - create_wider_stencil: Add a stencil of specified size around the prediction point for selected variables in the dataset
        - rotate_frame: Rotate variables from x-y coordinates to a flow dependent coordinate
        - nondimensionalize: Non-dimensionalize input and output variables
        - preprocess_simulation_data: Apply a set of default operations to pre-process the data.
            (in future, this function can be customized for different models).
    """
    def __init__(self,
                 simulation_names = ['DG','P2L'],
                 filter_scales    = ['50','100','200','400'] ,
                 window_size = 1,
                 preprocess = True):

        self.simulation_names = simulation_names
        self.filter_scales = filter_scales
        self.window_size = window_size
        
        self.load_simulation_data()

        if preprocess:
            self.preprocess_simulation_data()
        

    def load_simulation_data(self):
        '''
        Open up simulaiton data as a DataTree
        '''
        try: 
            self.simulation_data = read_filtered_datatree(self.simulation_names,
                                                  self.filter_scales)
        except:
            print("Error reading simulation data")
            
 
    # Pre-processing methods
    ## These apply to individual datasets, which sit at end nodes of datatree. 

    def generate_h_mask(self, thin_limit = 5, thickness_variable='hbar'): 
        '''
        Often we need a thickness based mask :
            Don't consider points where the thickness is too small.
        '''
        self.simulation_data = self.simulation_data.map_over_subtree(lambda n: n.assign(h_mask = (n[thickness_variable]>= thin_limit) ))
                                                                     

    def add_variables(self, add_filter_scale=True, add_mask=True):
        '''
        To add variables that don't already exist in the dataset.
        Examples can be length scales, or some grid related variable.
        '''
        # Add length scales
        if add_mask:
            self.generate_h_mask()
            
        if add_filter_scale:
            #self.ml_input_dataset = self.ml_input_dataset.map_over_subtree(lambda n: n.assign(filter_scale = float(n.attrs['filter_scale'])*1e3 + 0.*n.dudx))
            self.simulation_data = self.simulation_data.map_over_subtree(lambda n: n.assign(filter_scale = float(n.attrs['filter_scale'])*1e3 + 0.*n.dudx))

        # if add_mask:
        #     self.ml_input_dataset = self.ml_input_dataset.map_over_subtree(lambda n: n.assign(h_mask = 
            
        
    
    def create_wider_stencil(self, variables_to_widen=['dudx','dvdx','dudy','dvdy','dhdx','dhdy'], not_replace=True):
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
            widened = n[variables_to_widen].rolling(xh=self.window_size, yh=self.window_size, 
                                                    min_periods=1, center=True).construct(xh='Xn', yh='Yn')

            if not_replace:                
            # Rename the widened variables to avoid overwriting
                widened = widened.rename({var: f"{var}_widened" for var in variables_to_widen})
            
            # Merge the widened variables with the original dataset
            combined = xr.merge([widened, n], compat='override')
            #combined= widened

            return combined

        #self.ml_input_dataset = self.ml_input_dataset.map_over_subtree(widen_stencil)
        self.simulation_data = self.simulation_data.map_over_subtree(widen_stencil)

       
    def rotate_frame(self, frame_vector_vars = ['dhdx','dhdy'], ):
        '''
        Rotate variables from x-y coordinates to a flow dependent coordinate. 

        frame_vector_vars: Names of the vectors that will be used to rotate the frame. 
        
        '''
        
        
        def rotate_vector(R_11, R_12, R_21, R_22, F_1, F_2): 
            vec_That = R_11 * F_1 + R_21 * F_2
            vec_Nhat = R_12 * F_1 + R_22 * F_2
        
            return vec_That, vec_Nhat

        def two_by_two_matrix_multiplication(A_11, A_12, A_21, A_22, B_11, B_12, B_21, B_22): 
            '''
            A simple matrix multiplication of two 2X2 matrices.
            '''
            C_11 = A_11*B_11 + A_12*B_21
            C_12 = A_11*B_12 + A_12*B_22
            C_21 = A_21*B_11 + A_22*B_21
            C_22 = A_21*B_12 + A_22*B_22
            return C_11, C_12, C_21, C_22


        def rotate_tensor(R_11, R_12, R_21, R_22, T_11, T_12, T_21, T_22): 
            '''
            Rotate tensor = R_transpose (T) R
            '''
            
            # C = R_transpose (T)
            C_11, C_12, C_21, C_22 = two_by_two_matrix_multiplication(R_11, R_21, R_12, R_22, T_11, T_12, T_21, T_22)
        
            # Tp = C R, where Tp is the rotated tensor
            Tp_11, Tp_12, Tp_21, Tp_22 = two_by_two_matrix_multiplication(C_11, C_12, C_21, C_22, R_11, R_12, R_21, R_22)
        
            return Tp_11, Tp_12, Tp_21, Tp_22

        def apply_rotation_to_dataset(ds):
            ds=ds.copy()
            
            mag_frame_vector = (ds[frame_vector_vars[0]]**2 + ds[frame_vector_vars[1]]**2)**0.5
            
            T_hat_i = ds[frame_vector_vars[0]]/ mag_frame_vector
            T_hat_j = ds[frame_vector_vars[1]]/ mag_frame_vector

            N_hat_i = - T_hat_j
            N_hat_j =   T_hat_i

            R_11 = T_hat_i
            R_12 = N_hat_i
            R_21 = T_hat_j
            R_22 = N_hat_j

            
            ds['dudx_widened_rotated'], ds['dudy_widened_rotated'], ds['dvdx_widened_rotated'], ds['dvdy_widened_rotated'] = rotate_tensor(
                                                                                                R_11, R_12, R_21, R_22,
                                                                                               ds['dudx_widened'], ds['dudy_widened'],
                                                                                               ds['dvdx_widened'], ds['dvdy_widened'])
            
            ds['dhdx_widened_rotated'], ds['dhdy_widened_rotated'] = rotate_vector(R_11, R_12, R_21, R_22, ds['dhdx_widened'], ds['dhdy_widened'])
                
            
            ds['uphp_rotated'], ds['vphp_rotated'] = rotate_vector(R_11, R_12, R_21, R_22, ds['uphp'], ds['vphp'])

            return ds

        self.simulation_data = self.simulation_data.map_over_subtree(apply_rotation_to_dataset)
        
           
    def nondimensionalize(self):
        '''
        Non-dimensionalize input and output variables
        '''
        def calc_magnitudes(ds):
            ds = ds.copy()

            ds['mag_nabla_u_widened'] = ((ds.dudx_widened**2 + ds.dudy_widened**2 + ds.dvdx_widened**2 + ds.dvdy_widened**2).sum(['Xn','Yn']))**0.5
            ds['mag_nabla_h_widened'] = ((ds.dhdx_widened**2 + ds.dhdy_widened**2).sum(['Xn','Yn']))**0.5

            return ds

        def nondimensionalize_variables(ds):
            ds = ds.copy()

            ds = calc_magnitudes(ds)

            # Normalize tensor variables
            tensor_vars = ['dudx_widened_rotated', 'dudy_widened_rotated', 'dvdx_widened_rotated', 'dvdy_widened_rotated']
            for var_name in tensor_vars:
                ds[var_name+'_nondim'] = ds[var_name]/ds['mag_nabla_u_widened']
            
            # Normalize grad h 
            vector_vars = ['dhdx_widened_rotated', 'dhdy_widened_rotated']
            for var_name in vector_vars:
                ds[var_name+'_nondim'] = ds[var_name]/ds['mag_nabla_h_widened']

            # Normalize fluxes
            flux_vars = ['uphp_rotated', 'vphp_rotated']
            for var_name in flux_vars:
                ds[var_name+'_nondim'] = ds[var_name]/ds['mag_nabla_u_widened']/(ds['filter_scale']**2)

            return ds

        #self.simulation_data = self.simulation_data.map_over_subtree(calc_magnitudes)
        self.simulation_data = self.simulation_data.map_over_subtree(nondimensionalize_variables)
        

    # Pre-processing pipeline methods
    ## Depending on the model the pre processing pipeline may be slightly different, the function below handles this.
    ## Also apply to datasets
    
    def preprocess_simulation_data(self): 
        
        '''
        Some set of default operations read in as a list and done on data to pre-process. 
        Takes in simulation_data -> creates an expanded set of variables that may be needed for ML.
        Note:
            This function can be customized for different models. 
        '''
        # The pipeline that is emerging is:
        ## 
        
        
        self.add_variables()
        self.create_wider_stencil()
        self.rotate_frame()
        self.nondimensionalize()
        # Maybe splitting default_preprocess_pipeline from create_ML_variables
        # makes
        
        
class MLXarrayDataset:
    '''
    A class to take simulation data and keep only the variables that are needed for ML.

    Attributes:
        - simulation_data: SimulationData object.
        - all_ml_variables: List of all variables that will be needed in the ML (inputs, outputs, masks, coordinates, etc).
        - use_mask: Boolean to determine whether to use a thickness mask.
        - time_range: Slice object to select a time range.
        - ml_variables: List of all variables that will be needed in the ML (inputs, outputs, masks, coordinates, etc).
        - LARGEST_FILTER_SCALE: Largest filter scale in the simulation data.
        - window_size: Size of the stencil window.
        - points_per_node: Number of points per node.
        - total_points: Total number of points.
        - ml_dataset: xarray DataTree object with ML variables.
        - concatenated_ml_dataset: Concatenated xarray dataset with ML variables.
        - ml_batches: BatchGenerator object with ML variables.

    Methods:
        - choose_ml_variables: Select input and output variables, to be operated on going forward.
        - subsample_ml_variables_horizontally: To maintain uniformity in data size we need to sub-sample the simulations with finer filter scales.
        - h_mask_ml_variables: Mask variables using thickness masks.
        - scale_normalize: Do some scale normalization using fixed constants, to make outputs order 1.
        - split_train_test_data: Split data into training and testing sets.
        - stack_physical_dimensions: Stack selected physical dimensions into one.
        - pick_uniform_points: Select a uniform number of points from each dataset.
        - drop_nans: Remove data points.
        - randomize_along_points: Randomize data points.
        - randomize_concatenated_ml_dataset: Randomize concatenated ML dataset.
        - concat_datatree_nodes: Concatenate different node datasets into one.
        - generate_batches: Generate batches with some prescribed size.
        - create_xr_ML_variables: Run the long list of steps that takes simulation data to something that is close to being able to be ingested in the ML model.

    '''
    def __init__(self, simulation_data: SimulationData, 
                 all_ml_variables  = ['dudx','dvdx','dudy','dvdy','dhdx','dhdy','uphp','vphp'],
                 use_mask         = True,
                 time_range= slice(0, 20),
                 default_create = True,
                 num_batches=None):

        self.LARGEST_FILTER_SCALE = int(simulation_data.filter_scales[-1])
        self.simulation_data = simulation_data.simulation_data
        self.window_size = simulation_data.window_size
        # ml_variables should be a list of all variables that will be needed in the ml (inputs, outputs, masks, coordinates, etc).
        self.ml_variables = all_ml_variables
        
        self.time_range = time_range
        self.default_create = default_create
        
        if use_mask: 
            self.use_mask = use_mask
            self.ml_variables.append('h_mask')  
        
        if self.default_create:
            self.create_xr_ML_variables(num_batches)
            self.num_batches = num_batches

    def choose_ml_variables(self):
        '''
        Select variables, to be operated on going forward. 
        '''
        
        self.ml_dataset = self.simulation_data.map_over_subtree(lambda n: n[self.ml_variables])
            
        # This process of choosing subset from datatree is based on the discussion from this issue
        # https://github.com/xarray-contrib/datatree/issues/79 
        # Eventually if a subset function is introduced, the map_over_subtree can be removed.
    
    def subsample_ml_variables_horizontally(self): 
        '''
        To maintain uniformity in data size we need to sub-sample the simulations with finer filter scales.
        '''
        def subsample_by_filter_scale(n):
            
            subsample_factor = int(self.LARGEST_FILTER_SCALE/float(n.attrs['filter_scale']))

            return n.isel(xh=slice(0, None, subsample_factor), 
                          yh=slice(0, None, subsample_factor))
                               
        self.ml_dataset = self.ml_dataset.map_over_subtree(subsample_by_filter_scale)
                                            
    def h_mask_ml_variables(self):
        '''
        Mask variables using thickness masks.
        '''
        def only_h_mask_data_variables(n):
            n_masked = n.copy()
            
            for var in n.data_vars:
                if var != 'h_mask':
                    n_masked[var] = n[var].where(n['h_mask'])

            n_masked = n_masked.drop_vars('h_mask')
            
            return n_masked
            
        if self.use_mask:
            self.ml_dataset = self.ml_dataset.map_over_subtree(only_h_mask_data_variables)
        else:
            raise ValueError("use_mask flag is not set to true.")

    def scale_normalize(self):
        '''
        Do some scale normalization using fixed constants, to make outputs order 1. 
        '''

        pass

    def split_train_test_data(self):
        '''
        Split data into training and testing sets
        Note!! At the moment this takes place somewhere else.
        '''
                
    def stack_physical_dimensions(self, dims_to_stack=('Time','xh','yh','zl')):
        '''
        Stack selected physical dimensions into one. 
        '''
        self.ml_dataset = self.ml_dataset.stack(points=dims_to_stack)

    def pick_uniform_points(self): 
        self.ml_dataset = self.ml_dataset.isel(points=slice(0, self.points_per_node))
        
    def drop_nans(self):
        '''
        Remove data points 
        '''
        self.ml_dataset = self.ml_dataset.dropna('points')
    
    def randomize_along_points(self): 
        random.seed(seed)
        np.random.seed(seed)
        
        min_points = float('inf')
        
        def randomize_dataset(ds): 
            nonlocal min_points 
            npoints = len(ds.points)
            min_points = min(min_points, npoints)
            ds_rand = ds.copy()
            ds_rand = ds_rand.isel(points = np.random.choice(npoints, size=npoints, replace=False))
            
            return ds_rand

        self.ml_dataset = self.ml_dataset.map_over_subtree(randomize_dataset)
        self.points_per_node = min_points

    def randomize_concatenated_ml_dataset(self): 
        npoints = len(self.concatenated_ml_dataset.points)
        self.concatenated_ml_dataset = self.concatenated_ml_dataset.isel(points = np.random.choice(npoints, size=npoints, replace=False))
        self.total_points = npoints
    
    def concat_datatree_nodes(self):

        all_node_ds = [] 
        def append_members(ds):
            all_node_ds.append(ds.copy())
            return ds

        self.ml_dataset.map_over_subtree(append_members) 

        self.concatenated_ml_dataset = xr.concat(all_node_ds, dim='points')

    def generate_batches(self, num_batches):
        '''
        Generate batches with some prescribed size. 
        '''
        if 'Xn' in self.concatenated_ml_dataset.dims:
            self.ml_batches = xbatcher.BatchGenerator(ds = self.concatenated_ml_dataset, 
                               input_dims={'Xn':self.window_size,'Yn':self.window_size},
                               batch_dims={'points': int(self.total_points/num_batches)}   )
        else:
            self.ml_batches = xbatcher.BatchGenerator(ds = self.concatenated_ml_dataset, 
                               input_dims={},
                               batch_dims={'points': int(self.total_points/num_batches)}   )
        
    def create_xr_ML_variables(self, num_batches=100):
        '''
        Run the long list of steps that takes simulation data to something that is close 
        to being able to be ingested in the ML model. Note that this step results in an
        xarray dataset that is loaded into memory.
        '''
        # Choose what variables will be involved in ML
        self.choose_ml_variables()
        
        # Sub-sample in time 
        self.ml_dataset = self.ml_dataset.isel(Time=self.time_range)
    
        # We want to subsample the domain, since the datasets are not uniformly resolved.
        self.subsample_ml_variables_horizontally()
        
        # Mask regions with vanishing thickness
        self.h_mask_ml_variables()
        
        # stack spatial dimensions to points (in each node dataset)
        self.stack_physical_dimensions()
    
        # load data into memory to speed up dataset generation
        start_time = time.time()
        print(f"Will load : {self.ml_dataset.nbytes/1e9} gb into memory.")
        self.ml_dataset.load();
        print(f"load took: {time.time() - start_time:.4f} seconds")
        
        # Drop NaNs
        self.drop_nans()

        # Randomize
        ## if we don't randomize then we will run into problem when we are
        # selecting only a few points from each dataset to make a batch.
        self.randomize_along_points()
        
        # select a uniform number of points from each dataset
        self.pick_uniform_points()
        
        # concatenate different node datasets into one 
        self.concat_datatree_nodes()

        self.randomize_concatenated_ml_dataset()

        # Generate batches 
        self.generate_batches(num_batches)


class MLJAXDataset:
    '''
    A class to take the xarray ML dataset object from MLXarrayDataset 
    (with the meta data and different variables as data arrays)
    and convert it to a jax dataset that can be used directly for training ML models.
    This new dataset is returned through an iterator, which returns the batches 1 by one,
     and that can be used to train the model.

    Attributes:
        - ML_dataset: MLXarrayDataset object.
        - input_channels: List of input channels.
        - output_channels: List of output channels.
        - coeff_channels: List of coefficient channels.
        - ds_norm: Normalization dataset.
        - preprocessed_data: List of preprocessed data.
    Methods:
        - preprocess_batch: Preprocess a batch of data.
        - normalize_ds: Normalize the dataset.
        - get_batches: Generate batches.
    '''
    def __init__(self, ML_dataset, input_channels, output_channels, 
                 coeff_channels=None, ds_norm=None):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.coeff_channels = coeff_channels
        self.ds_norm = ds_norm
        
        # Preprocess the entire dataset
        self.preprocessed_data = []
        for batch in ML_dataset.ml_batches:
            batch_out = self.preprocess_batch(batch)
            self.preprocessed_data.append(batch_out)

    def preprocess_batch(self, batch: xr.Dataset): 
        # Normalize the dataset if normalization is provided
        batch = self.normalize_ds(batch, self.ds_norm)
        
        # Process the input and output channels
        X_xr = batch[self.input_channels].to_stacked_array("input_features", sample_dims=['points'])
        y_xr = batch[self.output_channels].to_array().transpose(..., 'variable')

        X = jnp.asarray(X_xr.data)
        y = jnp.asarray(y_xr.data)
        
        # Xp is set at 1, or the product of the coefficient channels. 
        # So to take the square you would have to pass channel twice.
        if (self.coeff_channels is not None) and (len(self.coeff_channels) > 0):
            Xp_xr = batch[self.coeff_channels[0]].copy()
            for var in self.coeff_channels[1:]:
                Xp_xr = Xp_xr * batch[var]
            Xp = jnp.asarray(Xp_xr.data.reshape(-1, 1))
        else:
            Xp_xr = 0.*y_xr.copy() + 1.
            Xp = jnp.asarray(Xp_xr.data)

        # Prepare the batch output
        # batch_out = {'X': X, 'y': y, 'Xp': Xp, 
        #              'X_xr': X_xr, 'y_xr': y_xr, 'Xp_xr': Xp_xr}
        
        batch_out = {'X': X, 'y': y, 'Xp': Xp}
        return batch_out
    
    def normalize_ds(self, ds, ds_norm):
        if ds_norm is not None:
            return ds / ds_norm
        return ds

    def get_batches(self):
        for batch_out in self.preprocessed_data:
            yield batch_out
    

##############################
### --- Older data classes --- 
### Kept here for backward compatibility.

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
        