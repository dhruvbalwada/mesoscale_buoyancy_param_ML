import ML_classes
import datasets
import xarray as xr
import matplotlib.pyplot as plt
import xrft

class EvaluationSystem: 
    
    # def __init__(self):
    #     pass
    
    # Reading functions
    def read_eval_data(self, data_fname, scale, kind='MITgcm'):
        
        if kind == 'MITgcm':
            self.eval_ds = datasets.MITgcm_transformer(data_fname, 
                                                   scale, 
                                                   self.input_channels)
        
#        elif kind == 'MOM6':
        
        self.eval_ds.convert_normed()
        
        self.input_ds = self.eval_ds.ML_dataset[self.input_channels]
        self.output_ds = self.eval_ds.ML_dataset[self.output_channels]
        
        self.input_ds_normed = self.eval_ds.ML_dataset_norm[self.input_channels]
        
        #self.output_ds = self.eval_ds.ML_dataset[self.output_channels]
        
    
    # Get ML model from weights for evaluation 
    def read_model(self, model_nc_fname):
        
        self.model_xr = xr.open_dataset(model_nc_fname) 
        
        self.input_channels = self.model_xr.input_channels
        self.output_channels = self.model_xr.output_channels
                                                   
                                                   
        self.ANN_model = ML_classes.ANN(shape = self.model_xr.shape, num_in = self.model_xr.num_in)
        
        self.regress_sys = ML_classes.RegressionSystem(self.ANN_model)
        
        self.regress_sys.read_checkpoint(self.model_xr.CKPT_DIR)
        
    
    def sel_time(self, tsel = slice(-10, None), tdim='Time'): 
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
        
    def horz_snapshot_plot(self, Zlev=5, Tlev = -1, var='Sfny'): 
        
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
    
    def zonal_PS(self, var='Sfnx', avg_dims=['time','YC']):
        ps_true = xrft.power_spectrum(self.output_ds[var].drop(['Depth', 'hFacC', 'maskC', 'rA']), 
                    'XC')

        ps_pred = xrft.power_spectrum(self.output_pred_ds[var].drop(['Depth', 'hFacC', 'maskC', 'rA']), 
                    'XC')
        
        ps_anom = xrft.power_spectrum( (self.output_ds - self.output_pred_ds)[var].drop(['Depth', 'hFacC', 'maskC', 'rA']), 
                    'XC')
        
        return ps_true.mean(avg_dims), ps_pred.mean(avg_dims), ps_anom.mean(avg_dims)
    
      
    def zonal_avg_OT(self, var='Sfny', avg_dims=['time','XC']): 
        
        Sfn_true = self.output_ds[var].mean(avg_dims)
        Sfn_pred = self.output_pred_ds[var].mean(avg_dims)
        Sfn_anom = Sfn_true - Sfn_pred
        
        return Sfn_true, Sfn_pred, Sfn_anom
        