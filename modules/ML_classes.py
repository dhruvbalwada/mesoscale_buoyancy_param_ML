import jax 
import ml_helper_func as ml_hf
import optax
import flax
import numpy as np
from flax.training import train_state, checkpoints
from flax.training import orbax_utils
flax.config.update('flax_use_orbax_checkpointing', False)
from jax import numpy as jnp
import xarray as xr
import orbax.checkpoint

class ANN:
    
    def __init__(self, shape=[24,24,2], num_in=7, bias=True):
        self.shape = shape
        self.bias = bias
        self.model, self.params = ml_hf.initialize_model(shape, num_in , bias=self.bias)
        
    def count_parameters(self):
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(param_count)
    
    def save_params(self, fpath):
        return
    
    
class RegressionSystem: 
    
    def __init__(self, network, lr=0.01, local_norm=False): 
        
        self.lr = 0.01
        self.network = network
        self.local_norm = local_norm
        
        if local_norm==False:
            self.criterion = jax.value_and_grad(ml_hf.mse)
        else:
            self.criterion = jax.value_and_grad(ml_hf.mse_local_norm)
        
        self.train_loss = np.array([])
        self.test_loss = np.array([])
        self.setup_optimizer()
        
        self.epoch = 0
        
    def setup_optimizer(self):
        
        self.tx = optax.adam(learning_rate=self.lr)
        self.state = train_state.TrainState.create(
                            apply_fn=self.network.model.apply, 
                            params=self.network.params, 
                            tx=self.tx)
        

    def step(self, batch, kind='test'):
        X = jnp.asarray(batch[self.input_channels].to_array().transpose(...,'variable').data)
        y = jnp.asarray(batch[self.output_channels].to_array().transpose(...,'variable').data)
        
        #print(shape(X), shape(y))
        
        loss_val, grads = self.criterion(self.state.params, self.state.apply_fn, X, y)
        
        if kind == 'train':
            self.state = self.state.apply_gradients(grads=grads)
        
        return loss_val
    
    def step_windowed(self, batch, kind='test'): 
        X = jnp.asarray(batch[self.input_channels].to_stacked_array("input_features", sample_dims=['points']).data)
        y = jnp.asarray( batch[self.output_channels].isel(Xn = int(self.window_size/2), Yn = int(self.window_size/2)
                                                                     ).to_stacked_array("output_features", sample_dims=['points']).data)
        
        #print(X.shape, y.shape)
        
        loss_val, grads = self.criterion(self.state.params, self.state.apply_fn, X, y)
        
        if kind == 'train':
            self.state = self.state.apply_gradients(grads=grads)
        
        return loss_val

    def step_local_normed(self, batch, kind='test'):
        
        X_vel = batch[['U_x', 'U_y','V_x', 'V_y']].to_array().transpose(...,'variable')
        X_S = batch[['Sx', 'Sy']].to_array().transpose(...,'variable')
        X_vel_mag = ((X_vel**2).mean('variable'))**0.5 + 1e-10
        X_S_mag = ((X_S**2).mean('variable'))**0.5 + 1e-10
        X_scale_mag = batch['Lfilt']*1e3
        X_vel_normed = X_vel/X_vel_mag
        X_S_normed = X_S/X_S_mag

        psi_mag = jnp.asarray((X_vel_mag*X_S_mag*(X_scale_mag**2)).data.reshape(-1,1))
        
        X = jnp.asarray(xr.concat([X_vel_normed, X_S_normed], dim='variable').data)
        y = jnp.asarray(batch[self.output_channels].to_array().transpose(...,'variable').data)

        loss_val, grads = self.criterion(self.state.params, self.state.apply_fn, X, y, psi_mag)
        
        if kind == 'train':
            self.state = self.state.apply_gradients(grads=grads)
        
        return loss_val

    def step_local_normed_windowed(self, batch, kind='test'):
        
        X_vel = batch[['U_x', 'U_y','V_x', 'V_y']].to_stacked_array("input_features", sample_dims=['points'])
        X_S = batch[['Sx', 'Sy']].to_stacked_array("input_features", sample_dims=['points'])
        X_vel_mag = ((X_vel**2).mean('input_features'))**0.5 + 1e-10
        X_S_mag = ((X_S**2).mean('input_features'))**0.5 + 1e-10
        X_scale_mag = batch['Lfilt'].isel(Xn = int(self.window_size/2), Yn = int(self.window_size/2))*1e3
        X_vel_normed = X_vel/X_vel_mag
        X_S_normed = X_S/X_S_mag

        psi_mag = jnp.asarray((X_vel_mag*X_S_mag*(X_scale_mag**2)).data.reshape(-1,1))
        #print(psi_mag.shape)
        
        X = jnp.asarray(xr.concat([X_vel_normed, X_S_normed], dim='input_features').data)
        #print(X.shape)
        y = jnp.asarray(batch[self.output_channels].isel(Xn = int(self.window_size/2), Yn = int(self.window_size/2)
                                                                     ).to_stacked_array("output_features", sample_dims=['points']).data)
        #print(y.shape)

        loss_val, grads = self.criterion(self.state.params, self.state.apply_fn, X, y, psi_mag)
        
        if kind == 'train':
            self.state = self.state.apply_gradients(grads=grads)
        
        return loss_val
    
    def train_system(self, ML_data, num_epoch, print_freq=20): 
        
        self.ML_data = ML_data
        
        self.input_channels  = ML_data.input_channels
        self.output_channels = ML_data.output_channels
        
        
        for i in range(num_epoch): 
            self.epoch = self.epoch + 1
            
            loss_temp = np.array([])
            for batch in ML_data.bgen_train: 
                if self.local_norm:
                    loss_val = self.step_local_normed(batch, kind='train')
                    #print(str(i) +' = '+str(loss_val))
                else:
                    loss_val = self.step(batch, kind='train')
                
                loss_temp = np.append(loss_temp, loss_val)
            
            self.train_loss = np.append(self.train_loss, np.mean(loss_temp))
            
            loss_temp = np.array([])
            for batch in ML_data.bgen_test: 
                if self.local_norm:
                    loss_val = self.step_local_normed(batch, kind='test')
                else:
                    loss_val = self.step(batch, kind='test')
                loss_temp = np.append(loss_temp, loss_val)
            
            self.test_loss = np.append(self.test_loss, np.mean(loss_temp))
            
            #print(i)
            if i % print_freq  == 0:
                print(f'Train loss step {i}: ', self.train_loss[-1], f'test loss:', self.test_loss[-1])

    def train_system_windowed(self, ML_data, num_epoch, print_freq=20): 
        
        self.ML_data = ML_data
        
        self.input_channels  = ML_data.input_channels
        self.output_channels = ML_data.output_channels
        self.window_size = ML_data.window_size
        
        
        for i in range(num_epoch): 
            self.epoch = self.epoch + 1
            
            loss_temp = np.array([])
            for batch in ML_data.bgen_train: 
                if self.local_norm:
                    loss_val = self.step_local_normed_windowed(batch, kind='train')
                    #print(loss_val)
                else:
                    loss_val = self.step_windowed(batch, kind='train')
                
                loss_temp = np.append(loss_temp, loss_val)
            
            self.train_loss = np.append(self.train_loss, np.mean(loss_temp))
            
            loss_temp = np.array([])
            for batch in ML_data.bgen_test: 
                if self.local_norm:
                    loss_val = self.step_local_normed_windowed(batch, kind='test')
                else:
                    loss_val = self.step_windowed(batch, kind='test')
                loss_temp = np.append(loss_temp, loss_val)
            
            self.test_loss = np.append(self.test_loss, np.mean(loss_temp))
            
            #print(i)
            if i % print_freq  == 0:
                print(f'Train loss step {i}: ', self.train_loss[-1], f'test loss:', self.test_loss[-1])                
    
                
    def save_checkpoint(self, CKPT_DIR): 
        
        # ckpt ={#'epoch': self.epoch, 
        #         'training_state': self.state, 
        #         'train_loss': self.train_loss, 
        #         'test_loss': self.test_loss}
        ckpt = self.state
        self.CKPT_DIR = CKPT_DIR
        # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        # save_args = orbax_utils.save_args_from_target(ckpt)
        # orbax_checkpointer.save(CKPT_DIR, ckpt, save_args=save_args)

        checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, 
                           target=ckpt, step=self.epoch, overwrite=True)
        
    def read_checkpoint(self, CKPT_DIR): 
        
        # empty_state = {#'epoch': self.epoch, 
        #         'training_state': self.state, 
        #         'train_loss': self.train_loss, 
        #         'test_loss': self.test_loss}
        
        self.state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=self.state)
        #orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        #orbax_checkpointer.restore(CKPT_DIR, item=self.state)
        #self.state = ckpt.training_state
        #self.train_loss
        #self.state = 
        #return ckpt
        
    def pred(self, X):
        return self.state.apply_fn(self.state.params, X)

    def pred_local_normed(self, X): 
        X_vel = X[['U_x', 'U_y','V_x', 'V_y']].to_array().transpose(...,'variable')
        X_S = X[['Sx', 'Sy']].to_array().transpose(...,'variable')
        X_vel_mag = ( (X_vel**2).mean('variable') )**0.5 + 1e-10
        X_S_mag = ( (X_S**2).mean('variable') )**0.5 + 1e-10
        X_scale_mag = X['Lfilt']*1e3
        X_vel_normed = X_vel/X_vel_mag
        X_S_normed = X_S/X_S_mag

        psi_mag = jnp.asarray((X_vel_mag*X_S_mag*(X_scale_mag**2)).data[..., jnp.newaxis])
        
        X_input = jnp.asarray(xr.concat([X_vel_normed, X_S_normed], dim='variable').data)

        return self.state.apply_fn(self.state.params, X_input) * psi_mag

    def pred_local_normed_windowed(self, X, window_size=3, dims_input=['time', 'Z', 'YC', 'XC']):
        self.window_size = window_size
        X_vel = X[['U_x', 'U_y','V_x', 'V_y']].to_stacked_array("input_features", sample_dims=dims_input)
        X_S = X[['Sx', 'Sy']].to_stacked_array("input_features", sample_dims=dims_input)
        X_vel_mag = ((X_vel**2).mean('input_features'))**0.5 + 1e-10
        X_S_mag = ((X_S**2).mean('input_features'))**0.5 + 1e-10
        X_scale_mag = X['Lfilt'].isel(Xn = int(self.window_size/2), Yn = int(self.window_size/2))*1e3
        X_vel_normed = X_vel/X_vel_mag
        X_S_normed = X_S/X_S_mag

        psi_mag = jnp.asarray((X_vel_mag*X_S_mag*(X_scale_mag**2)).data[..., jnp.newaxis])
        #print(psi_mag.shape)
        
        X_input = jnp.asarray(xr.concat([X_vel_normed, X_S_normed], dim='input_features').data)

        return self.state.apply_fn(self.state.params, X_input) * psi_mag

    
    
    def save_weights_nc(self, nc_fname): 

        if self.local_norm == False:
            input_norms = np.zeros((len(self.input_channels),))
            output_norms = np.zeros((len(self.output_channels),))
    
            for n, i in enumerate(self.input_channels):
                input_norms[n] = self.ML_data.norm_factors[i].values
    
            for n, i in enumerate(self.output_channels):
                output_norms[n] = self.ML_data.norm_factors[i].values

        ds_layers = xr.Dataset()

        ds_layers['layer_sizes'] = xr.DataArray(np.array([len(self.input_channels),
                                                          self.network.shape[0], 
                                                          self.network.shape[1], len(self.output_channels)]).astype('int32'), 
                                                dims=['num_layers'])

        ds_layers['A0'] = xr.DataArray(np.array(self.state.params['params']['layers_0']['kernel']).astype('float32'), dims=['input', 'layer1'])
        ds_layers['A1'] = xr.DataArray(np.array(self.state.params['params']['layers_1']['kernel']).astype('float32'), dims=['layer1', 'layer2'])
        ds_layers['A2'] = xr.DataArray(np.array(self.state.params['params']['layers_2']['kernel']).astype('float32'), dims=['layer2', 'output'])

        if self.network.bias:
            ds_layers['b0'] = xr.DataArray(np.array(self.state.params['params']['layers_0']['bias']).astype('float32'), dims=['layer1'])
            ds_layers['b1'] = xr.DataArray(np.array(self.state.params['params']['layers_1']['bias']).astype('float32'), dims=['layer2'])
            ds_layers['b2'] = xr.DataArray(np.array(self.state.params['params']['layers_2']['bias']).astype('float32'), dims=['output'])
        else: 
            ds_layers['b0'] = xr.DataArray(np.zeros(self.network.shape[0]).astype('float32'), dims=['layer1'])
            ds_layers['b1'] = xr.DataArray(np.zeros(self.network.shape[1]).astype('float32'), dims=['layer2'])
            ds_layers['b2'] = xr.DataArray(np.zeros(self.network.shape[2]).astype('float32'), dims=['output'])

        if self.local_norm==False:
            ds_layers['input_norms'] = xr.DataArray(input_norms.astype('float32'), dims=['input'])
            ds_layers['output_norms'] = xr.DataArray(output_norms.astype('float32'), dims=['output'])
        
        ds_layers.attrs['CKPT_DIR'] = self.CKPT_DIR
        ds_layers.attrs['shape'] = self.network.shape
        ds_layers.attrs['num_in'] = len(self.input_channels)
        ds_layers.attrs['input_channels'] = self.input_channels
        ds_layers.attrs['output_channels'] = self.output_channels
        

        ds_layers.to_netcdf(nc_fname, mode='w')

    def save_weights_nc_windowed(self, nc_fname): 
        
        input_norms = np.zeros((len(self.input_channels),))
        output_norms = np.zeros((len(self.output_channels),))

        for n, i in enumerate(self.input_channels):
            input_norms[n] = self.ML_data.norm_factors[i].values

        for n, i in enumerate(self.output_channels):
            output_norms[n] = self.ML_data.norm_factors[i].values

        ds_layers = xr.Dataset()

        ds_layers['layer_sizes'] = xr.DataArray(np.array([len(self.input_channels)*self.ML_data.window_size**2,
                                                          self.network.shape[0], 
                                                          self.network.shape[1], len(self.output_channels)]).astype('int32'), 
                                                dims=['num_layers'])

        ds_layers['A0'] = xr.DataArray(np.array(self.state.params['params']['layers_0']['kernel']).astype('float32'), dims=['input', 'layer1'])
        ds_layers['A1'] = xr.DataArray(np.array(self.state.params['params']['layers_1']['kernel']).astype('float32'), dims=['layer1', 'layer2'])
        ds_layers['A2'] = xr.DataArray(np.array(self.state.params['params']['layers_2']['kernel']).astype('float32'), dims=['layer2', 'output'])

        if self.network.bias:
            ds_layers['b0'] = xr.DataArray(np.array(self.state.params['params']['layers_0']['bias']).astype('float32'), dims=['layer1'])
            ds_layers['b1'] = xr.DataArray(np.array(self.state.params['params']['layers_1']['bias']).astype('float32'), dims=['layer2'])
            ds_layers['b2'] = xr.DataArray(np.array(self.state.params['params']['layers_2']['bias']).astype('float32'), dims=['output'])
        else: 
            ds_layers['b0'] = xr.DataArray(np.zeros(self.network.shape[0]).astype('float32'), dims=['layer1'])
            ds_layers['b1'] = xr.DataArray(np.zeros(self.network.shape[1]).astype('float32'), dims=['layer2'])
            ds_layers['b2'] = xr.DataArray(np.zeros(self.network.shape[2]).astype('float32'), dims=['output'])


        ds_layers['input_norms'] = xr.DataArray(input_norms.astype('float32'), dims=['input_channels'])
        ds_layers['output_norms'] = xr.DataArray(output_norms.astype('float32'), dims=['output'])
        
        ds_layers.attrs['CKPT_DIR'] = self.CKPT_DIR
        ds_layers.attrs['shape'] = self.network.shape
        ds_layers.attrs['num_in'] = len(self.input_channels)
        ds_layers.attrs['input_channels'] = self.input_channels
        ds_layers.attrs['output_channels'] = self.output_channels
        

        ds_layers.to_netcdf(nc_fname, mode='w')
        