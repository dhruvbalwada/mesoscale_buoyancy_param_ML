import ML_classes
import datasets
import xarray as xr
import matplotlib.pyplot as plt
import xrft

def full_reader(model_nc, data_zarr, L, data_kind, exp_name, ML_name,Tsel=slice(-25, None), Tdim='Time',
               windowed=False, window_size=None):
    '''

    '''
    eval_mod = EvaluationSystem()

    eval_mod.read_model(model_nc)
    eval_mod.get_model_norm_factors_ds()

    if windowed:
        eval_mod.read_eval_data_windowed(data_zarr,L, data_kind, window_size=window_size)
        eval_mod.sel_time(Tsel, Tdim)
        eval_mod.pred_windowed()
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
    def read_model(self, model_nc_fname):
        
        self.model_xr = xr.open_dataset(model_nc_fname) 
        
        self.input_channels = self.model_xr.attrs['input_channels']
        self.output_channels = self.model_xr.attrs['output_channels']
                                                   
                                                   
        self.ANN_model = ML_classes.ANN(shape = self.model_xr.shape, num_in = self.model_xr.num_in)
        
        self.regress_sys = ML_classes.RegressionSystem(self.ANN_model)
        
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
        

    def sel_time(self, tsel = slice(-25, None), tdim='Time'): 
        self.input_ds = self.input_ds.isel(**{tdim:tsel})
        self.output_ds = self.output_ds.isel(**{tdim:tsel})
        
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
        ps_true = xrft.power_spectrum(self.output_ds[var].drop(['Depth', 'hFacC', 'maskC', 'rA']), 
                    'XC')

        ps_pred = xrft.power_spectrum(self.output_pred_ds[var].drop(['Depth', 'hFacC', 'maskC', 'rA']), 
                    'XC')
        
        ps_anom = xrft.power_spectrum( (self.output_ds - self.output_pred_ds)[var].drop(['Depth', 'hFacC', 'maskC', 'rA']), 
                    'XC')
        
        return ps_true.mean(avg_dims), ps_pred.mean(avg_dims), ps_anom.mean(avg_dims)
    
    def zonal_PS_P2L(self, var='Sfny', avg_dims=['Time','yh']):
        ps_true = xrft.power_spectrum(self.output_ds[var], 'xh')

        ps_pred = xrft.power_spectrum(self.output_pred_ds[var],'xh')
        
        ps_anom = xrft.power_spectrum( (self.output_ds - self.output_pred_ds)[var], 'xh')
        
        return ps_true.mean(avg_dims), ps_pred.mean(avg_dims), ps_anom.mean(avg_dims)
    
        
    def zonal_avg_OT(self, var='Sfny', avg_dims=['time','XC']): 
        
        Sfn_true = self.output_ds[var].mean(avg_dims)
        Sfn_pred = self.output_pred_ds[var].mean(avg_dims)
        Sfn_anom = Sfn_true - Sfn_pred
        
        return Sfn_true, Sfn_pred, Sfn_anom


    #def evaluate_PE(self): 
        
        