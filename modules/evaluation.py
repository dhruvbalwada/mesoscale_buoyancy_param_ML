import ML_classes
import datasets
import xarray as xr
from datatree import DataTree 
import matplotlib.pyplot as plt
import xrft
import PE_module
import numpy as np

class EvalSystem:
    '''
    
    '''
    def __init__(self, 
                 simulation_data, 
                 input_channels,
                 output_channels,
                 coeff_channels,                 
                 eval_time_slice,
                 num_inputs,
                 shape,
                 ckpt_dir, 
                 extra_channels=[],
                 use_norm_factors=True,
                 ds_norm_factors=None,
                 use_coeff_channels=False):

        self.simulation_data = simulation_data 
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.coeff_channels = coeff_channels
        self.extra_channels = extra_channels
        self.use_coeff_channels = use_coeff_channels
        self.use_norm_factors = use_norm_factors
        if self.use_norm_factors and ds_norm_factors is None:
            self.ds_norm_factors = simulation_data.ds_norm
        else:
            print("Not implemented the non-use of norm factors")
        self.eval_time_slice = eval_time_slice
        self.num_inputs = num_inputs
        self.shape = shape
        self.model_ckpt_dir = ckpt_dir

        #self.read_ann_regression_model()

        #self.read_eval_data()

# Methods to setup the evaluation system
    def read_ann_regression_model(self): 
        '''
        Read the ANN model for regression.
        '''
        self.ANN_model = ML_classes.PointwiseANN(num_in=self.num_inputs,
                                                 shape=self.shape)
        self.regress_sys = ML_classes.AnnRegressionSystem(self.ANN_model)
        # This step loads in the model.
        self.regress_sys.read_checkpoint(self.model_ckpt_dir)

    def read_eval_data(self):
        '''
        '''
        self.eval_datatree = datasets.MLXarrayDataset(simulation_data=self.simulation_data,
                                                      all_ml_variables=self.input_channels + self.output_channels + self.coeff_channels + self.extra_channels,
                                                      time_range=self.eval_time_slice,
                                                      default_create=False)
        
        self.eval_datatree.choose_ml_variables()
        self.eval_datatree.subsample_ml_variables_time()

    def predict(self): 
        
        def predict_for_dataset(ds):
            ml_ds = ds.copy()

            input_ds = ml_ds[self.input_channels]
            output_ds = ml_ds[self.output_channels]
            coeff_ds = ml_ds[self.coeff_channels]

            
            # Ensure all DataArrays in input_ds have the sample dimensions
            sample_dims = ['Time', 'zl', 'yh', 'xh']
            for var in input_ds.data_vars:
                for dim in sample_dims:
                    if dim not in input_ds[var].dims:
                        input_ds[var] = input_ds[var].broadcast_like(ml_ds[sample_dims])
            

            for var in coeff_ds.data_vars:
                for dim in sample_dims:
                    if dim not in coeff_ds[var].dims:
                        coeff_ds[var] = coeff_ds[var].broadcast_like(ml_ds[sample_dims])


            input_normed_ds = input_ds/self.ds_norm_factors
            output_normed_ds = output_ds/self.ds_norm_factors
            coeff_normed_ds = coeff_ds/self.ds_norm_factors 

            X_xr = input_normed_ds.to_stacked_array("input_features", sample_dims=sample_dims)
            y_xr = output_normed_ds.to_array().transpose('Time', 'zl', 'yh', 'xh', 'variable')

        
            if (len(self.coeff_channels) > 0) and (self.use_coeff_channels):
                Xp_xr = coeff_normed_ds[self.coeff_channels[0]].copy()
                for var in self.coeff_channels[1:]:
                    Xp_xr = Xp_xr * coeff_normed_ds[var]
                    
                Xp_xr = Xp_xr + (0.* y_xr.copy())

            else:
                Xp_xr = 0.*y_xr.copy() + 1.
           
            y_pred_xr = self.regress_sys.pred(X_xr, Xp_xr)

            pred_xr = y_pred_xr.to_dataset(dim='variable') * self.ds_norm_factors
           

            # Create a dictionary mapping old variable names to new variable names with '_pred' suffix
            rename_dict = {var: f"{var}_pred" for var in pred_xr.data_vars}
            
            # Rename the variables in the dataset
            pred_xr = pred_xr.rename(rename_dict)
            
            ml_ds = ml_ds.update(pred_xr)

            return ml_ds
        self.eval_datatree.ml_dataset = self.eval_datatree.ml_dataset.map_over_subtree(predict_for_dataset)

    def dimensionalize(self): 

        def dimensionalize_for_dataset(ds): 
            ml_ds = ds.copy()

            #for var in ['uphp', 'vphp']: 
            #    ml_ds[var] = ml_ds[var] * self.ds_norm_factors[var]
            
            ml_ds['uphp_rotated_pred'] = ml_ds['uphp_rotated_nondim_pred'] * ml_ds['mag_nabla_u_widened'] * ml_ds['mag_nabla_h_widened'] * ml_ds['filter_scale']**2
            ml_ds['vphp_rotated_pred'] = ml_ds['vphp_rotated_nondim_pred'] * ml_ds['mag_nabla_u_widened'] * ml_ds['mag_nabla_h_widened'] * ml_ds['filter_scale']**2
            
            return ml_ds
        
        self.eval_datatree.ml_dataset = self.eval_datatree.ml_dataset.map_over_subtree(dimensionalize_for_dataset)

    def add_gradient_model_variables(self):

        def gradient_model(ml_ds):
            
            ml_ds = ml_ds.copy()

            sim_ds = self.simulation_data.simulation_data[ml_ds.simulation_name][ml_ds.attrs['filter_scale']]

            mid_point = int(np.floor(self.simulation_data.window_size/2))

            
            ml_ds['uphp_rotated_grad_model'] = sim_ds.filter_scale**2 * (sim_ds['dudx_widened_rotated'].isel(Xn=mid_point, Yn=mid_point) * 
                                                            sim_ds['dhdx_widened_rotated'].isel(Xn=mid_point, Yn=mid_point) + 
                                                            sim_ds['dudy_widened_rotated'].isel(Xn=mid_point, Yn=mid_point) * 
                                                            sim_ds['dhdy_widened_rotated'].isel(Xn=mid_point, Yn=mid_point))
            
            ml_ds['vphp_rotated_grad_model'] = sim_ds.filter_scale**2 * (sim_ds['dvdx_widened_rotated'].isel(Xn=mid_point, Yn=mid_point) * 
                                                            sim_ds['dhdx_widened_rotated'].isel(Xn=mid_point, Yn=mid_point) + 
                                                            sim_ds['dvdy_widened_rotated'].isel(Xn=mid_point, Yn=mid_point) * 
                                                            sim_ds['dhdy_widened_rotated'].isel(Xn=mid_point, Yn=mid_point))
            
            # determine coefficient 
            A = ml_ds['uphp_rotated']
            B = ml_ds['uphp_rotated_grad_model']
            C = ml_ds['vphp_rotated']
            D = ml_ds['vphp_rotated_grad_model']

            c = ((A * B).sum(skipna=True) + (C * D).sum(skipna=True)) / ((B**2).sum(skipna=True) + (D**2).sum(skipna=True))

            ml_ds['uphp_rotated_grad_model'] = c * ml_ds['uphp_rotated_grad_model']
            ml_ds['vphp_rotated_grad_model'] = c * ml_ds['vphp_rotated_grad_model']
            ml_ds['c_grad_model'] = c

            return ml_ds 
        
        self.eval_datatree.ml_dataset = self.eval_datatree.ml_dataset.map_over_subtree(gradient_model)
        
    def add_gent_mcwilliams_variables(self):

        def gent_mcwilliams_model(ml_ds): 
            ml_ds = ml_ds.copy()

            sim_ds = self.simulation_data.simulation_data[ml_ds.simulation_name][ml_ds.attrs['filter_scale']]

            mid_point = int(np.floor(self.simulation_data.window_size/2))

            ml_ds['uphp_rotated_gent_mcwilliams'] = - sim_ds['dhdx_widened_rotated'].isel(Xn=mid_point, Yn=mid_point) 
            ml_ds['vphp_rotated_gent_mcwilliams'] = - sim_ds['dhdy_widened_rotated'].isel(Xn=mid_point, Yn=mid_point)

            # determine coefficient
            A = ml_ds['uphp_rotated']
            B = ml_ds['uphp_rotated_gent_mcwilliams']
            # Only do this calculation for downgradient part of the flow (uphp is the along gradient case, also in the rotated case h_y_rotated=0)
            #C = ml_ds['uphp_rotated']
            #D = ml_ds['uphp_rotated_gent_mcwilliams']

            c = (A * B).sum(skipna=True)  / (B**2).sum(skipna=True) 

            ml_ds['uphp_rotated_gent_mcwilliams'] = c * ml_ds['uphp_rotated_gent_mcwilliams']
            ml_ds['vphp_rotated_gent_mcwilliams'] = c * ml_ds['vphp_rotated_gent_mcwilliams']
            ml_ds['kappa_gent_mcwilliams'] = c

            return ml_ds
        
        self.eval_datatree.ml_dataset = self.eval_datatree.ml_dataset.map_over_subtree(gent_mcwilliams_model)

# Methods to evaluate the model

# Metrics to evaluate the model
    @staticmethod
    def _R2(true, pred, dims=['Time', 'xh', 'yh', 'zl']):
        """
        Calculate the coefficient of determination (R-squared) between true and predicted values for a given variable.

        Parameters:
        - true (xr.DataArray): The true values.
        - pred (xr.DataArray): The predicted values.
        - var (str): The variable to calculate R-squared for.
        - dims (list): List of dimensions to average over (default: ['time', 'XC', 'YC', 'Z']).

        Returns:
        - float: The R-squared value.
        """
        RSS = ((pred - true) ** 2).mean(dims)
        TSS = ((true - true.mean(dims)) ** 2).mean(dims)

        
        R2 = 1 - RSS / TSS

        return R2


    @staticmethod
    def _correlation(true, pred, dims=['Time', 'xh', 'yh', 'zl']):
        """
        Calculate the correlation coefficient between true and predicted values for a given variable.

        Parameters:
        - true (xr.DataArray): The true values.
        - pred (xr.DataArray): The predicted values.
        - var (str): The variable to calculate correlation for.
        - dims (list): List of dimensions to average over (default: ['time', 'XC', 'YC', 'Z']).

        Returns:
        - xr.DataArray: The correlation coefficient.
        """
        correlation_coefficient = xr.corr(true, pred, dim=dims)
        return correlation_coefficient

    @staticmethod
    def _MSE(true, pred, dims=['Time', 'xh', 'yh', 'zl']):
        """
        Calculate the mean squared error between true and predicted values for a given variable.

        Parameters:
        - true (xr.DataArray): The true values.
        - pred (xr.DataArray): The predicted values.
        - var (str): The variable to calculate MSE for.
        - dims (list): List of dimensions to average over (default: ['time', 'XC', 'YC', 'Z']).

        Returns:
        - float: The MSE value.
        """
        MSE = ((pred - true) ** 2).mean(dims)

        return MSE
    
    def calc_time_hor_space_metrics(self, var='uphp', dims=['Time','xh','yh','zl'], 
                                    descriptor='all_space_time',
                                    xh_region=None, 
                                    yh_region=None, 
                                    zl_slice=None,
                                    use_default_subregions=False, 
                                    load=True): 
        
        # xh_region = xh_region
        # yh_region = yh_region
        # print(xh_region)

        def calc_for_dataset(ds): 
            ds = ds.copy()

            simulation_name = ds.attrs['simulation_name']
            if use_default_subregions:
                if simulation_name == 'DG':
                    xh_sel = slice(5, 17)
                    yh_sel = slice(32, 42)
                elif simulation_name == 'P2L':
                    xh_sel = slice(100, 1100)
                    yh_sel = slice(250, 1350)
                else:
                    xh_sel, yh_sel = None, None  # Default case
            else:
                xh_sel, yh_sel = xh_region, yh_region  # Use user-provided regions

            # If regions are None, use full domain
            xh_sel = ds.xh if xh_sel is None else xh_sel
            yh_sel = ds.yh if yh_sel is None else yh_sel
            zl_sel = ds.zl if zl_slice is None else zl_slice

            #print(xh_region)
            true = ds[var].sel(xh=xh_sel, yh=yh_sel, zl=zl_sel)
            #print(true)
            try:
                pred = ds[var+'_pred'].sel(xh=xh_sel, yh=yh_sel, zl=zl_sel)
            except KeyError:
                print('No prediction found for variable: ' + var)

            
            ds[var+'_R2_'+descriptor] = self._R2(true, pred, dims=dims)
            ds[var+'_corr_'+descriptor] = self._correlation(true, pred, dims=dims)
            ds[var+'_mse_'+descriptor] = self._MSE(true, pred, dims=dims)

            if load:
                ds[var+'_R2_'+descriptor].load()
                ds[var+'_corr_'+descriptor].load()
                #ds[var+'_mse_'+descriptor].load()

            return ds 
        
        self.eval_datatree.ml_dataset = self.eval_datatree.ml_dataset.map_over_subtree(calc_for_dataset)


    def calc_PS(self, var='uphp', spec_dims=['xh'], avg_dims=['Time','yh'], descriptor='zonal', 
                xh_region=slice(5,17), yh_region=slice(32, 43), use_default_subregions=False): 
        
        def calc_for_dataset(ds): 
            ds = ds.copy()

            simulation_name = ds.attrs['simulation_name']
            if use_default_subregions:
                if simulation_name == 'DG':
                    xh_sel = slice(5, 17)
                    yh_sel = slice(32, 43)
                elif simulation_name == 'P2L':
                    xh_sel = slice(100, 1100)
                    yh_sel = slice(250, 1350)
                else:
                    xh_sel, yh_sel = None, None  # Default case
            else:
                xh_sel, yh_sel = xh_region, yh_region  # Use user-provided regions

            ds[var+'_ps_'+descriptor] = xrft.power_spectrum(ds[var].sel(xh=xh_sel, yh=yh_sel), dim=spec_dims, window=True, window_correction=True).mean(avg_dims)
            ds[var+'_ps_pred_'+descriptor] = xrft.power_spectrum(ds[var+'_pred'].sel(xh=xh_sel, yh=yh_sel), dim=spec_dims , window=True, window_correction=True).mean(avg_dims)
            ds[var+'_ps_anom_'+descriptor] = xrft.power_spectrum( (ds[var+'_pred'] - ds[var]).sel(xh=xh_sel, yh=yh_sel), dim=spec_dims , window=True, window_correction=True).mean(avg_dims)
            
            return ds 
        
        self.eval_datatree.ml_dataset = self.eval_datatree.ml_dataset.map_over_subtree(calc_for_dataset)

    def calc_time_hor_space_metrics_trad_models(self, var='uphp', dims=['Time','xh','yh','zl'], 
                                    descriptor='all_space_time',
                                    xh_region=None, 
                                    yh_region=None, 
                                    zl_slice=None,
                                    use_default_subregions=False,
                                    load=True): 
        
        # xh_region = xh_region
        # yh_region = yh_region
        # print(xh_region)

        def calc_for_dataset(ds): 
            ds = ds.copy()

            simulation_name = ds.attrs['simulation_name']
            if use_default_subregions:
                if simulation_name == 'DG':
                    xh_sel = slice(5, 17)
                    yh_sel = slice(32, 42)
                elif simulation_name == 'P2L':
                    xh_sel = slice(100, 1100)
                    yh_sel = slice(250, 1350)
                else:
                    xh_sel, yh_sel = None, None  # Default case
            else:
                xh_sel, yh_sel = xh_region, yh_region  # Use user-provided regions

            # If regions are None, use full domain
            xh_sel = ds.xh if xh_sel is None else xh_sel
            yh_sel = ds.yh if yh_sel is None else yh_sel
            zl_sel = ds.zl if zl_slice is None else zl_slice

            #print(xh_region)
            true = ds[var].sel(xh=xh_sel, yh=yh_sel, zl=zl_sel)
            #print(true)
            try:
                pred_grad_model = ds[var+'_grad_model'].sel(xh=xh_sel, yh=yh_sel, zl=zl_sel)
                pred_gent_mcwilliams = ds[var+'_gent_mcwilliams'].sel(xh=xh_sel, yh=yh_sel, zl=zl_sel)
            except KeyError:
                print('No prediction found for variable: ' + var)

            
            ds[var+'_R2_grad_model_'+descriptor] = self._R2(true, pred_grad_model, dims=dims)
            ds[var+'_corr_grad_model_'+descriptor] = self._correlation(true, pred_grad_model, dims=dims)
            ds[var+'_mse_grad_model_'+descriptor] = self._MSE(true, pred_grad_model, dims=dims)

            ds[var+'_R2_gent_mcwilliams_'+descriptor] = self._R2(true, pred_gent_mcwilliams, dims=dims)
            ds[var+'_corr_gent_mcwilliams_'+descriptor] = self._correlation(true, pred_gent_mcwilliams, dims=dims)
            ds[var+'_mse_gent_mcwilliams_'+descriptor] = self._MSE(true, pred_gent_mcwilliams, dims=dims)
            
            if load:
                ds[var+'_R2_grad_model_'+descriptor].load()
                ds[var+'_corr_grad_model_'+descriptor].load()
                #ds[var+'_mse_grad_model_'+descriptor].load()

                ds[var+'_R2_gent_mcwilliams_'+descriptor].load()
                ds[var+'_corr_gent_mcwilliams_'+descriptor].load()
                #ds[var+'_mse_gent_mcwilliams_'+descriptor].load()

            return ds 
        
        self.eval_datatree.ml_dataset = self.eval_datatree.ml_dataset.map_over_subtree(calc_for_dataset)

    def coarsen_time_ML_data(self, coarsen_times=[16, 64, 128, 256]): 
        self.coarsen_times = [str(time) for time in coarsen_times]

        for exp in self.simulation_data.simulation_names:
            for scale in self.simulation_data.filter_scales:
                for coarsen_time in coarsen_times: 
                    self.eval_datatree.ml_dataset[exp][scale][str(coarsen_time)] = DataTree(self.eval_datatree.ml_dataset[exp][scale].ds.coarsen(Time=coarsen_time, boundary='trim').mean())
                      

############################################################
############################################################
############################################################
### Old classes, kept here for backward compatibility. #####
############################################################
############################################################
############################################################

def full_reader(model_nc, data_zarr, L, data_kind, exp_name, ML_name,Tsel=slice(-25, None), Tdim='Time',
               windowed=False, window_size=None, local_norm=False, out_para_perp=False, dims_input = ['time', 'Z', 'YC', 'XC'], diffuse=False):
    '''

    '''
    eval_mod = EvaluationSystem()

    eval_mod.read_model(model_nc, diffuse=diffuse)
    
    if local_norm==False:
        eval_mod.get_model_norm_factors_ds()

    if local_norm & windowed: 
        print('Local normed and windowed')
        eval_mod.read_eval_data_local_normed_windowed(data_zarr, data_kind, Lkey=L,
                                                      window_size=window_size, 
                                                     out_para_perp=out_para_perp)
        eval_mod.eval_ds.datatree = eval_mod.eval_ds.datatree.isel({Tdim:Tsel})
        eval_mod.sel_time(Tsel, Tdim, local_norm=True)
        eval_mod.pred_local_norm_window(L, dims_input = dims_input)
    elif windowed:
        print('Windowed')
        eval_mod.read_eval_data_windowed(data_zarr,L, data_kind, window_size=window_size)
        eval_mod.sel_time(Tsel, Tdim)
        eval_mod.pred_windowed()
    elif local_norm:
        print('Local normed')
        eval_mod.read_eval_data_local_normed(data_zarr, data_kind, Lkey=L, out_para_perp=out_para_perp)
        eval_mod.eval_ds.datatree = eval_mod.eval_ds.datatree.isel({Tdim:Tsel})
        eval_mod.sel_time(Tsel, Tdim, local_norm=True)
        eval_mod.pred_local_norm(L)
    else:
        eval_mod.read_eval_data(data_zarr,L, data_kind)
        eval_mod.sel_time(Tsel, Tdim)
        eval_mod.pred()
    
    return eval_mod

class EvaluationSystem: 
    
    # Reading functions
    def read_eval_data(self, data_fname, scale, data_kind):
        
        if data_kind == 'MITgcm':
            # Note that since we usually do evaluation on a single scale at a time, we load data per scale.
            # this requires us to make sure that the right normalization factors are loaded in. 
            self.eval_ds = datasets.MITgcm_transformer(data_fname, 
                                                   scale, 
                                                   self.input_channels) 
            self.eval_ds.convert_normed(norm_factors=self.norm_ds)
        
        elif data_kind == 'MOM6_P2L':
            self.eval_ds = datasets.MOM6_transformer(data_fname, 
                                                   scale, 
                                                   self.input_channels)
            self.eval_ds.convert_normed(norm_factors=self.norm_ds, large_filt=int(scale))
            
        elif data_kind == 'MOM6_DG':
            self.eval_ds = datasets.MOM6_transformer(data_fname, 
                                                   scale, 
                                                   self.input_channels)
            self.eval_ds.convert_normed(norm_factors=self.norm_ds, large_filt=int(scale)/100,mask_wall=True, H_mask=500)
            #self.eval_ds.mask_domain(mask_wall=True, H_mask=500)
        
        
        
        self.input_ds  = self.eval_ds.ML_dataset[self.input_channels]
        self.output_ds = self.eval_ds.ML_dataset[self.output_channels]
        
        self.input_ds_normed = self.eval_ds.ML_dataset_norm[self.input_channels]

    def read_eval_data_local_normed(self, data_fname, data_kind, Lkey='50', out_para_perp=False): 
        if data_kind == 'MITgcm': 
            self.eval_ds = datasets.MITgcm_all_transformer('_', 
                                                   '-', 
                                                   self.input_channels)
            self.eval_ds.read_datatree(data_fname, keep_filt_scale=True, sub_sample=False, 
                                       largest_remove=False, Lkeys=[Lkey], para_perp_out = out_para_perp)

        if data_kind == 'MOM6_P2L': 
            self.eval_ds = datasets.MOM6_all_transformer('_', 
                                                   '-', 
                                                   self.input_channels)
            self.eval_ds.read_datatree(data_fname, keep_filt_scale=True, sub_sample=False, 
                                       largest_remove=False, large_filt=int(Lkey), Lkeys=[Lkey], 
                                       para_perp_out = out_para_perp)

        if data_kind == 'MOM6_DG': 
            self.eval_ds = datasets.MOM6_all_transformer('_', 
                                                   '-', 
                                                   self.input_channels)
            self.eval_ds.read_datatree(data_fname, file_names='res5km/ml_data_', 
                                       keep_filt_scale=True, sub_sample=False, 
                                       largest_remove=False, large_filt=int(Lkey)/100,
                                       H_mask=150, Lkeys=[Lkey], 
                                       eta_bottom=True, 
                                       para_perp_out = out_para_perp)
            
        self.input_ds = self.eval_ds.datatree[Lkey].ds[self.input_channels]
        self.output_ds = self.eval_ds.datatree[Lkey].ds[self.output_channels]
            

    def read_eval_data_local_normed_windowed(self, data_bucket, data_kind, window_size=3, Lkey='50', out_para_perp=False):
        if data_kind == 'MITgcm':
            self.eval_ds = datasets.MITgcm_all_transformer('-', '-', 
                                      input_channels=self.input_channels)

            self.eval_ds.read_datatree(data_bucket, 
                                       keep_filt_scale=True, 
                                       window_size=window_size, 
                                       sub_sample=False, 
                                       largest_remove=False, 
                                       Lkeys=[Lkey], 
                                       para_perp_out = out_para_perp)
        if data_kind == 'MOM6_P2L': 
            self.eval_ds = datasets.MOM6_all_transformer('_', 
                                                   '-', 
                                                   self.input_channels)
            self.eval_ds.read_datatree(data_bucket, 
                                       keep_filt_scale=True, 
                                       sub_sample=False, 
                                       window_size=window_size,
                                       largest_remove=False, 
                                       large_filt=int(Lkey), 
                                       Lkeys=[Lkey], 
                                       para_perp_out = out_para_perp)
        if data_kind == 'MOM6_DG': 
            self.eval_ds = datasets.MOM6_all_transformer('_', 
                                                   '-', 
                                                   self.input_channels)
            self.eval_ds.read_datatree(data_bucket, 
                                       file_names='res5km/ml_data_', 
                                       keep_filt_scale=True, 
                                       para_perp_out = out_para_perp,
                                       eta_bottom=True,
                                       large_filt=int(Lkey)/100,
                                       H_mask=150,
                                       sub_sample=False, 
                                       window_size=window_size,
                                       largest_remove=False,  Lkeys=[Lkey] )
            
        
        self.input_ds = self.eval_ds.datatree[Lkey].ds[self.input_channels]
        window_mid = int(self.eval_ds.window_size/2)
        self.output_ds = self.eval_ds.datatree[Lkey].ds[self.output_channels].isel(Xn=window_mid, Yn=window_mid)

    
    def read_eval_data_windowed(self, data_fname, scale, data_kind, window_size=3):
        
        if data_kind == 'MITgcm':
            # Note that since we usually do evaluation on a single scale at a time, we load data per scale.
            # this requires us to make sure that the right normalization factors are loaded in. 
            self.eval_ds = datasets.MITgcm_transformer(data_fname, 
                                                   scale, 
                                                   self.input_channels) 
            self.eval_ds.convert_any(norm_factors=self.norm_ds, window_size=window_size)
        
        self.input_ds  = self.eval_ds.ML_dataset[self.input_channels]
        window_mid = int(self.eval_ds.window_size/2)
        self.output_ds = self.eval_ds.ML_dataset[self.output_channels].isel(Xn=window_mid, Yn=window_mid)
        
        self.input_ds_normed = self.eval_ds.ML_dataset_norm[self.input_channels]
    
    
    # Get ML model from weights for evaluation 
    def read_model(self, model_nc_fname, local_norm=False, diffuse=False):
        
        self.model_xr = xr.open_dataset(model_nc_fname) 
        self.local_norm=local_norm
        
        self.input_channels = self.model_xr.attrs['input_channels']
        self.output_channels = self.model_xr.attrs['output_channels']
                                                   
        
        self.ANN_model = ML_classes.ANN(shape = self.model_xr.shape, num_in = self.model_xr.num_in, diffuse=diffuse)
        
        self.regress_sys = ML_classes.RegressionSystem(self.ANN_model, self.local_norm)
        
        self.regress_sys.read_checkpoint(self.model_xr.CKPT_DIR)
        
    def get_model_norm_factors_ds(self):
        # The model normalization factors come with the model, 
        # so during time of evaluation we need to get the norm factors
        # corresponding to the model (rather than the dataset).
        self.norm_ds = xr.Dataset()
        
        for i, var in enumerate(self.model_xr.attrs['input_channels']):
            self.norm_ds[var] = self.model_xr.input_norms[i] 
        for i, var in enumerate(self.model_xr.attrs['output_channels']):
            self.norm_ds[var] = self.model_xr.output_norms[i] 
        

    def sel_time(self, tsel = slice(-25, None), tdim='Time', local_norm=False): 
        self.input_ds = self.input_ds.isel(**{tdim:tsel})
        self.output_ds = self.output_ds.isel(**{tdim:tsel})

        if local_norm==False:
            self.input_ds_normed = self.input_ds_normed.isel(**{tdim:tsel})
        
    
    def pred(self): 
        
        y_pred = self.regress_sys.pred(self.input_ds_normed.to_array().transpose(...,'variable'))
        
        dims = self.output_ds.to_array().transpose(...,'variable').dims
        coords = self.output_ds.to_array().transpose(...,'variable').coords
        
        ds_pred = xr.DataArray(y_pred, dims=dims, coords=coords).to_dataset(dim='variable')
        # convert to real units
        self.output_pred_ds = ds_pred * self.eval_ds.norm_factors

    def pred_windowed(self): 
        
        #y_pred = self.regress_sys.pred(self.input_ds_normed.to_array().transpose(...,'variable'))
        y_pred = self.regress_sys.pred(self.input_ds_normed.to_stacked_array("input_features", sample_dims=['time','Z','YC','XC']).data)
        
        dims = self.output_ds.to_array().transpose(...,'variable').dims
        coords = self.output_ds.to_array().transpose(...,'variable').coords
        
        ds_pred = xr.DataArray(y_pred, dims=dims, coords=coords).to_dataset(dim='variable')
        # convert to real units
        self.output_pred_ds = ds_pred * self.eval_ds.norm_factors

    def pred_local_norm(self, L):
        #y_pred
        #for L in ['50','100','200','400']: 
        #print(L)
        data = self.eval_ds.datatree[L].to_dataset()
        
        y_pred = self.regress_sys.pred_local_normed(data, input_channels=self.input_channels)

        output_pred_ds = xr.Dataset()
        
        output_pred_ds['Sfnx'] = xr.DataArray(y_pred[...,0], dims= self.output_ds['Sfnx'].dims,
                                    coords= self.output_ds['Sfnx'].coords)
        output_pred_ds['Sfny'] = xr.DataArray(y_pred[...,1], dims= self.output_ds['Sfny'].dims,
                                    coords= self.output_ds['Sfny'].coords)

        
        #self.eval_ds.datatree[L].ds = data
        self.output_pred_ds = output_pred_ds

    def pred_local_norm_window(self, L, dims_input = ['time', 'Z', 'YC', 'XC']): 
        '''
        There is a problem right now for periodic boundaries : https://github.com/pydata/xarray/issues/2007 
        Hopefully this only shows up when working with evaluation as the nan's get thrown out during the training phase. 
        '''
        data = self.eval_ds.datatree[L].to_dataset()

        y_pred = self.regress_sys.pred_local_normed_windowed(data, window_size=self.eval_ds.window_size, dims_input=dims_input)

        output_pred_ds = xr.Dataset()
        
        output_pred_ds['Sfnx'] = xr.DataArray(y_pred[...,0], dims= self.output_ds['Sfnx'].dims,
                                    coords= self.output_ds['Sfnx'].coords)
        output_pred_ds['Sfny'] = xr.DataArray(y_pred[...,1], dims= self.output_ds['Sfny'].dims,
                                    coords= self.output_ds['Sfny'].coords)

        self.output_pred_ds = output_pred_ds
            
        
    def horz_snapshot_plot_MITgcm(self, Zlev=5, Tlev = -1, var='Sfny'): 
        
        plt.figure(figsize=(12, 3.5))

        plt.subplot(131)
        self.output_ds[var].isel(time=Tlev, Z=Zlev).plot(robust=True)
        plt.title('Truth')

        plt.subplot(132)
        self.output_pred_ds[var].isel(time=Tlev, Z=Zlev).plot(robust=True)
        plt.title('Prediction')
        
        plt.subplot(133)
        (self.output_ds - self.output_pred_ds)[var].isel(time=Tlev, Z=Zlev).plot(robust=True)
        plt.title('Truth - Prediction')

        plt.tight_layout()
        
    def horz_snapshot_plot_MOM6(self, Tlev = -1, var='Sfny'): 
        
        plt.figure(figsize=(12, 3.5))

        plt.subplot(131)
        self.output_ds[var].isel(Time=Tlev).plot(robust=True)
        plt.title('Truth')

        plt.subplot(132)
        self.output_pred_ds[var].isel(Time=Tlev).plot(robust=True)
        plt.title('Prediction')
        
        plt.subplot(133)
        (self.output_ds - self.output_pred_ds)[var].isel(Time=Tlev).plot(robust=True)
        plt.title('Truth - Prediction')

        plt.tight_layout()
    
    @staticmethod
    def _R2(true, pred, var, dims=['time', 'XC', 'YC', 'Z']):
        """
        Calculate the coefficient of determination (R-squared) between true and predicted values for a given variable.

        Parameters:
        - true (xr.DataArray): The true values.
        - pred (xr.DataArray): The predicted values.
        - var (str): The variable to calculate R-squared for.
        - dims (list): List of dimensions to average over (default: ['time', 'XC', 'YC', 'Z']).

        Returns:
        - float: The R-squared value.
        """
        RSS = ((pred[var] - true[var]) ** 2).mean(dims)
        TSS = ((true[var]) ** 2).mean(dims)

        
        R2 = 1 - RSS / TSS

        return R2


    @staticmethod
    def _correlation(true, pred, var, dims=['time', 'XC', 'YC', 'Z']):
        """
        Calculate the correlation coefficient between true and predicted values for a given variable.

        Parameters:
        - true (xr.DataArray): The true values.
        - pred (xr.DataArray): The predicted values.
        - var (str): The variable to calculate correlation for.
        - dims (list): List of dimensions to average over (default: ['time', 'XC', 'YC', 'Z']).

        Returns:
        - xr.DataArray: The correlation coefficient.
        """
        correlation_coefficient = xr.corr(true[var], pred[var], dim=dims)
        return correlation_coefficient

    
    def calc_R2(self, var='Sfnx', dims=['time','XC','YC','Z']): 
        
        return self._R2(self.output_ds, self.output_pred_ds, var, dims)
    
    def calc_corr(self, var='Sfnx', dims=['time','XC','YC','Z']): 
        
        return self._correlation(self.output_ds, self.output_pred_ds, var, dims)
    
    def zonal_PS_MITgcm(self, var='Sfnx', avg_dims=['time','YC']):
        ps_true = xrft.power_spectrum(self.output_ds[var].chunk({'XC':-1}).drop(['Depth', 'hFacC', 'maskC', 'rA']).sel(XC=slice(400e3, 1600e3)), 
                    'XC')

        ps_pred = xrft.power_spectrum(self.output_pred_ds[var].chunk({'XC':-1}).drop(['Depth', 'hFacC', 'maskC', 'rA']).sel(XC=slice(400e3, 1600e3)), 
                    'XC')
        
        ps_anom = xrft.power_spectrum( (self.output_ds - self.output_pred_ds)[var].chunk({'XC':-1}).drop(['Depth', 'hFacC', 'maskC', 'rA']).sel(XC=slice(400e3, 1600e3)), 
                    'XC')
        
        return ps_true.mean(avg_dims), ps_pred.mean(avg_dims), ps_anom.mean(avg_dims)
    
    def zonal_PS_P2L(self, var='Sfny', avg_dims=['Time','yh'], xh_slice=slice(0, None)):
        ps_true = xrft.power_spectrum(self.output_ds[var].sel(xh=xh_slice), 'xh')

        ps_pred = xrft.power_spectrum(self.output_pred_ds[var].sel(xh=xh_slice),'xh')
        
        ps_anom = xrft.power_spectrum( (self.output_ds - self.output_pred_ds)[var].sel(xh=xh_slice), 'xh')
        
        return ps_true.mean(avg_dims), ps_pred.mean(avg_dims), ps_anom.mean(avg_dims)
    
        
    def zonal_avg_OT(self, var='Sfny', avg_dims=['time','XC']): 
        
        Sfn_true = self.output_ds[var].mean(avg_dims)
        Sfn_pred = self.output_pred_ds[var].mean(avg_dims)
        Sfn_anom = Sfn_true - Sfn_pred
        
        return Sfn_true, Sfn_pred, Sfn_anom


    def evaluate_PE(self, Tsel = slice(-25, None)): 
        ds_filt = self.eval_ds.dataset.isel(Time=Tsel)
        
        ds_filt['e'] = ds_filt.e - ds_filt.e.isel(Time=0).mean(['xh','yh']) # some way to go to APE from PE. 
        
        uh_sg_true = ds_filt.uh_sg.isel(zl=1)
        vh_sg_true = ds_filt.vh_sg.isel(zl=1)
        uh_sg_pred = self.output_pred_ds['Sfnx']
        vh_sg_pred = self.output_pred_ds['Sfny']

        _, _, div_uh_sg_true  = self._div_uphp(uh_sg_true, vh_sg_true)
        _, _, div_uh_sg_pred  = self._div_uphp(uh_sg_pred, vh_sg_pred)

        ds_filt['div_uh_sg_true'] = div_uh_sg_true
        ds_filt['div_uh_sg_pred'] = div_uh_sg_pred

        eta_tend_true = np.zeros_like(ds_filt.e)
        eta_tend_pred = np.zeros_like(ds_filt.e)
        
        eta_tend_true[:,1,:,:] = - ds_filt['div_uh_sg_true'] # Assuming that the eta tend go to 0 back on top.
        eta_tend_pred[:,1,:,:] = - ds_filt['div_uh_sg_pred']
        
        ds_filt['dt_eta_mean_by_eddy_true'] = xr.DataArray(eta_tend_true, dims=ds_filt.e.dims)
        ds_filt['dt_eta_mean_by_eddy_pred'] = xr.DataArray(eta_tend_pred, dims=ds_filt.e.dims)

        ds_filt['MPE_tend_eddy_true'], _ = PE_module.PE_tend(ds_filt,'dt_eta_mean_by_eddy_true')
        ds_filt['MPE_tend_eddy_pred'], _ = PE_module.PE_tend(ds_filt,'dt_eta_mean_by_eddy_pred')

        ds_filt['MPE_tend_eddy_spectral_true'], _ = PE_module.PE_tend_spectral(ds_filt,'dt_eta_mean_by_eddy_true')
        ds_filt['MPE_tend_eddy_spectral_pred'], _ = PE_module.PE_tend_spectral(ds_filt,'dt_eta_mean_by_eddy_pred')

        return ds_filt

    @staticmethod
    def _div_uphp(uh_sg, vh_sg):
        '''
        Estimate the divergence of the uh 
        This is slightly more annoying than trying to use xgcm, 
        because we have made the predictions on
        the center of the grid. 
        '''
        dx = (uh_sg.xh[1] - uh_sg.xh[0]).values * 1e3
        uh_sg_i = uh_sg
        uh_sg_ip1 = uh_sg.roll(xh=-1)
        
        uh_sg_q = 0.5*(uh_sg_ip1 + uh_sg_i)
        
        uh_sg_q_im1 = uh_sg_q.roll(xh=1)
        
        dx_uh_sg = (uh_sg_q - uh_sg_q_im1)/dx 
        
        dy = (uh_sg.yh[1] - uh_sg.yh[0]).values *1e3
        vh_sg_i = vh_sg
        vh_sg_ip1 = vh_sg.roll(yh=-1)
        
        vh_sg_q = 0.5*(vh_sg_ip1 + vh_sg_i)
        
        vh_sg_q_im1 = vh_sg_q.roll(yh=1)
        
        dy_vh_sg = (vh_sg_q - vh_sg_q_im1)/dy
    
        return dx_uh_sg, dy_vh_sg, dx_uh_sg + dy_vh_sg


    def para_perp_2_xy(self): 
        
        self.output_ds['Sfn_perp_x'], self.output_ds['Sfn_perp_y'], self.output_ds['Sfn_para_x'], self.output_ds['Sfn_para_y'] = self._project_slope2xy(self.output_ds.Sfnx, self.output_ds.Sfny, self.input_ds.Sx, self.input_ds.Sy) 

        self.output_pred_ds['Sfn_perp_x'], self.output_pred_ds['Sfn_perp_y'], self.output_pred_ds['Sfn_para_x'], self.output_pred_ds['Sfn_para_y'] = self._project_slope2xy(self.output_pred_ds.Sfnx, self.output_pred_ds.Sfny, self.input_ds.Sx, self.input_ds.Sy) 


    @staticmethod
    def _project_slope2xy(Sfn_perp_scalar, Sfn_para_scalar, Sx, Sy): 
        S_mag = (Sx**2 + Sy**2 )**0.5

        # Unit vector components in S direction
        Shatx = Sx/S_mag
        Shaty = Sy/S_mag
    
        # Unit vector components perp to S direction
        Nhatx = - Sy/S_mag
        Nhaty = Sx/S_mag

        Sfn_perp_x = Sfn_perp_scalar * Shatx
        Sfn_perp_y = Sfn_perp_scalar * Shaty
    
        Sfn_para_x = Sfn_para_scalar * Nhatx
        Sfn_para_y = Sfn_para_scalar * Nhaty

        return Sfn_perp_x, Sfn_perp_y, Sfn_para_x, Sfn_para_y


    def dissipative_prop(self, Sfnx='Sfnx', Sfny='Sfny', name='perp'):
    
        S_mag = self.input_ds.Sx**2 + self.input_ds.Sy**2
        
        self.output_ds['PE_diss_'+name] = self.input_ds.Sx*self.output_ds[Sfnx] + self.input_ds.Sy*self.output_ds[Sfny]
        self.output_pred_ds['PE_diss_'+name] = self.input_ds.Sx*self.output_pred_ds[Sfnx] + self.input_ds.Sy*self.output_pred_ds[Sfny]
        
        #if norm==True:
        #    return diss_true/S_mag#, diss_pred/S_mag
        #else:
        #    return diss_true


