import ML_classes
import datasets
import xarray as xr

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
        
        self.input_ds_normed = self.eval_ds.ML_dataset[self.input_channels]
        
        #self.output_ds = self.eval_ds.ML_dataset[self.output_channels]
        
    
    # Get ML model from weights for evaluation 
    def read_model(self, model_nc_fname):
        
        self.model_xr = xr.open_dataset(model_nc_fname) 
        
        self.input_channels = self.model_xr.input_channels
        self.output_channels = self.model_xr.output_channels
                                                   
                                                   
        self.ANN_model = ML_classes.ANN(shape = self.model_xr.shape, num_in = self.model_xr.num_in)
        
        self.regress_sys = ML_classes.RegressionSystem(self.ANN_model)
        
        self.regress_sys.read_checkpoint(self.model_xr.CKPT_DIR)
        
    
    def sel_time(self, tsel = slice(-10, 0), tdim='Time'): 
        self.input_ds = self.input_ds.sel(**{tdim:tsel})
        self.output_ds = self.output_ds.sel(**{tdim:tsel})
        
        self.input_ds_normed = self.input_ds_normed.sel(**{tdim:tsel})
        
    
    def pred(self): 
        
        y_pred = self.regress_sys.pred(self.input_ds_normed.to_array().transpose(...,'variable'))
        
        dims = self.output_ds.to_array().transpose(...,'variable').dims
        coords = self.output_ds.to_array().transpose(...,'variable').coords
        
        ds_pred = xr.DataArray(y_pred, dims=dims, coords=coords).to_dataset(dim='variable')
        self.output_pred_ds = ds_pred * self.eval_ds.norm_factors
        
    #def 

    