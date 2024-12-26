import jax 
import ml_helper_func as ml_hf
import optax
import flax
import numpy as np
from typing import Any, Callable, Sequence
from jax import lax, random, numpy as jnp
from flax.training import train_state, checkpoints
from flax.training import orbax_utils
from flax import linen as nn
from jax import numpy as jnp
import xarray as xr
import orbax.checkpoint
import wandb
from functools import partial


######################## New ########################
class ArtificialNeuralNetwork(nn.Module):
    '''
    This class is used to define an ANN model for regression.
    
    Input:
        features: A list containing the number of neurons in each layer.
        bias: A boolean indicating whether bias is to be used.

    Attributes:    
        features: A list containing the number of neurons in each layer.
        bias: A boolean indicating whether bias is to be used.
        layers: A list containing the layers of the ANN model.

    Methods:
        setup: Initializes the layers of the ANN model.
        __call__: Applies the ANN model to the input data.
    '''
    features: Sequence[int]
    bias: True
    
    def setup(self):
        self.layers = [nn.Dense(feat, use_bias=self.bias) for feat in self.features]
        
    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i!=len(self.layers)-1:
                x = nn.relu(x)
        return x
        
class PointwiseANN:
    '''
    This class is used to define an ANN model for regression.

    Input:
        shape: A list containing the number of neurons in each layer.
        num_in: The number of input features.
        bias: A boolean indicating whether bias is to be used.
        random_key: A random key to initialize the model. (Default is 0)
    Attributes:
        shape: The shape of the ANN model.
        bias: A boolean indicating whether bias is to be used.
        num_in: The number of input features.
        random_key: The random key used to initialize the model.
        model: The ANN model.
        params: The parameters of the model.
    Methods:
        initialize_model: Initializes the ANN model.
        count_parameters: Counts the number of parameters in the model.
    ''' 
    def __init__(self, shape=[24,24,2], num_in=7, bias=True, random_key=0): 
        self.shape = shape
        self.bias  = bias
        self.num_in = num_in
        self.random_key = random_key
        
        self.initialize_model()

    def initialize_model(self): 
        self.model = ArtificialNeuralNetwork(features=self.shape, bias = self.bias)
        key1, key2 = random.split(random.PRNGKey(self.random_key))
        x = random.normal(key1, (self.num_in,))
        self.params = self.model.init(key2, x)

    def count_parameters(self):
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(param_count)
  
class AnnRegressionSystem:
    '''
    This class is used to train an ANN model for regression.
    
    Input:
        network: An instance of the PointwiseANN class.
        learning_rate: The learning rate for the optimizer.
        optimizer: The optimizer to be used. Currently only 'adam' is supported.

    Attributes:
        network: The ANN model.
        learning_rate: The learning rate for the optimizer.
        optimizer: The optimizer to be used.
        train_loss: The training loss at each epoch.
        test_loss: The testing loss at each epoch.
        criterion: The loss function.
        state: The state of the optimizer.
        epoch: The current epoch.
    
    Methods:
        setup_optimizer: Sets up the optimizer.
        mse: The mean squared error loss function.
        step: The function that applies the model to the data and computes the loss.
        train_system: Trains the model.
    '''

    def __init__(self, network, learning_rate=0.01, optimizer='adam', loss_type='mse'):

        self.network = network
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_type = loss_type

        self.train_loss = np.array([])
        self.test_loss  = np.array([])
        self.test_R2    = np.array([])
        
        if self.loss_type == 'mse':
            self.criterion = jax.value_and_grad(self.mse, argnums=0)
        elif self.loss_type == 'mae':
            self.criterion = jax.value_and_grad(self.mae, argnums=0)
        
        self.setup_optimizer()
        
        self.epoch = 0 

    def setup_optimizer(self):
        '''
        This function sets up the optimizer.
        '''

        if self.optimizer == 'adam': 
            self.tx = optax.adam(learning_rate=self.learning_rate)
        
        self.state = train_state.TrainState.create(
                            apply_fn=self.network.model.apply, 
                            params=self.network.params, 
                            tx=self.tx)
        
    def mse(self, params, x_batched, y_batched, xp_batched):
        '''
        This is the MSE loss with an extra multiplier that can be sample dependent or 1. 
        When 1, this reverts to regular MSE loss. 
        '''
        # Define squared loss for a single pair (x,y), where y can be a vector (multi-dim output) 
        def squared_error(x,y,xp):
            pred = self.state.apply_fn(params, x) * xp
            return jnp.inner(y-pred, y-pred) / 2.0
    
        return jnp.nanmean(jax.vmap(squared_error)(x_batched, y_batched, xp_batched), axis=0)
    
    def mae(self, params, x_batched, y_batched, xp_batched):
        '''
        This is the MAE loss with an extra multiplier that can be sample dependent or 1. 
        When 1, this reverts to regular MAE loss. 
        '''
        # Define squared loss for a single pair (x,y), where y can be a vector (multi-dim output) 
        def abs_error(x,y,xp):
            pred = self.state.apply_fn(params, x) * xp
            return jnp.mean(jnp.abs(y-pred))
    
        return jnp.nanmean(jax.vmap(abs_error)(x_batched, y_batched, xp_batched), axis=0)

    def R2(self, params, x_batched, y_batched, xp_batched):
        '''
        This function computes the R2 score.
        '''
        def squared_error(x,y,xp):
            pred = self.state.apply_fn(params, x) * xp
            return jnp.inner(y-pred, y-pred) / 2.0
        
        def total_error(y):
            return jnp.inner(y-jnp.mean(y), y-jnp.mean(y)) / 2.0
        
        return 1 - jnp.nanmean(jax.vmap(squared_error)(x_batched, y_batched, xp_batched), axis=0) / jnp.nanmean(jax.vmap(total_error)(y_batched), axis=0)

    #@partial(jax.jit, static_argnums=(0, 2)) # this does not work. Why? 
    def step(self, batch, kind='test'):
        '''
        This functions applies the ML model to the data and computes the loss. 
        If the dataset is the training data, then gradient update is also done. 
        '''
        X = batch['X']
        y = batch['y']
        #Xp = batch['Xp']
        # Xp is received in the following way to ensure that this values is present even when 
        # it is not in the batch.
        Xp = batch.get('Xp', jnp.broadcast_to(1., y.shape))
        
        loss_val, grads = self.criterion(self.state.params, X, y, Xp)
        
        if kind == 'train':
            self.state = self.state.apply_gradients(grads=grads)
            return loss_val
        elif kind == 'test':
            return loss_val, self.R2(self.state.params, X, y, Xp)
        

    def train_system(self, ML_data, num_epoch, print_freq=20, use_wandb=False): 
        '''
        This function trains the ML model using the data in ML_data.
        Input:
            ML_data: A dictionary containing the training and testing data.
            num_epoch: Number of epochs to train the model.
            print_freq: Frequency at which the loss is printed.
        '''
        for i in range(num_epoch): 
            self.epoch = self.epoch + 1

            # training
            loss_temp = np.array([])
            for batch in ML_data['train_data'].get_batches():
                loss = self.step(batch, kind='train')
                loss_temp = np.append(loss_temp, loss)
                
            self.train_loss = np.append(self.train_loss, np.mean(loss_temp))

            #testing
            loss_temp = np.array([])
            R2_temp = np.array([])
            for batch in ML_data['test_data'].get_batches():
                loss, R2 = self.step(batch, kind='test')

                loss_temp = np.append(loss_temp, loss)
                R2_temp = np.append(R2_temp, R2)
                
            self.test_loss = np.append(self.test_loss, np.mean(loss_temp))
            self.test_R2 = np.append(self.test_R2, np.mean(R2_temp))

            if use_wandb:
                wandb.log({'epoch': self.epoch, 'train_loss': self.train_loss[-1], 'test_loss': self.test_loss[-1], 'test_R2': self.test_R2[-1]})

            if i % print_freq  == 0:
                print(f'At epoch {self.epoch}. Train loss : ', self.train_loss[-1], f', Test loss:', self.test_loss[-1], f', Test R2:', self.test_R2[-1])

        return 

    def save_checkpoint(self, checkpoint_dir, overwrite=True):
        '''
        This function saves the checkpoint of the model.
        '''

        save_dic = {'state':self.state, 
                    'train_loss':self.train_loss, 
                    'test_loss':self.test_loss,
                    'epoch':self.epoch}

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(save_dic)
        orbax_checkpointer.save(checkpoint_dir, save_dic, save_args=save_args, force=overwrite)
        # This is the simple way to save the checkpoint.
        # The checkpoint manager can be used for more complex saving.
        # https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html#with-orbax 

    def read_checkpoint(self, checkpoint_dir):
        '''
        This function reads the checkpoint of the model, restoring the state of the object.
        
        Input:
            checkpoint_dir: The directory where the checkpoint is saved.
        '''
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_dic_structure = {'state':self.state, 
                              'train_loss':self.train_loss, 
                              'test_loss':self.test_loss,
                              'epoch':self.epoch}
        
        restored_dic = orbax_checkpointer.restore(checkpoint_dir, item=save_dic_structure)

        self.state = restored_dic['state']
        self.train_loss = restored_dic['train_loss']
        self.test_loss = restored_dic['test_loss']
        self.epoch = restored_dic['epoch']

            
    def pred(self, X, Xp):
        '''
        This function predicts the output of the model for the input X, with multiplier Xp.
        '''
        return self.state.apply_fn(self.state.params, X) * Xp

######################## Old ########################
## Kept here for backward compatibility. ##
    
class RegressionSystem: 
    
    def __init__(self, network, lr=0.01, local_norm=False, diffuse=False): 
        
        self.lr = 0.01
        self.network = network
        self.local_norm = local_norm
        self.diffuse = diffuse

        
        if local_norm==False:
            self.criterion = jax.value_and_grad(ml_hf.mse)
        elif (local_norm == True) & (diffuse==True): 
            print('here')
            self.criterion = jax.value_and_grad(ml_hf.mse_local_norm_diffuse)
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

        if 'Lfilt' in self.input_channels:
                    
            X_L = batch[['Lfilt']].to_array().transpose(...,'variable')/400
            if 'hx' in self.input_channels:
                X_h = batch[['hx','hy']].to_array().transpose(...,'variable')/1e-3
                X = jnp.asarray(xr.concat([X_vel_normed, X_S_normed, X_L, X_h], dim='variable').data)
            else:
                X = jnp.asarray(xr.concat([X_vel_normed, X_S_normed, X_L], dim='variable').data)
        else:
            if 'hx' in self.input_channels:
                X_h = batch[['hx','hy']].to_array().transpose(...,'variable')/1e-3
                X = jnp.asarray(xr.concat([X_vel_normed, X_S_normed, X_h], dim='variable').data)
            else:
                X = jnp.asarray(xr.concat([X_vel_normed, X_S_normed], dim='variable').data)
                
            #X = jnp.asarray(xr.concat([X_vel_normed, X_S_normed], dim='variable').data)

        
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
        if 'Lfilt' in self.input_channels:
            X_L = batch[['Lfilt']].isel(Xn = int(self.window_size/2), Yn = int(self.window_size/2)).to_stacked_array("input_features", sample_dims=['points']).transpose(...,'input_features')/400
            X = jnp.asarray(xr.concat([X_vel_normed.drop_vars(['Xn','Yn']), X_S_normed.drop_vars(['Xn','Yn']), 
           X_L.drop_vars(['input_features'])], dim='input_features').data)
        else:
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
                if self.local_norm & (self.diffuse==False):
                    loss_val = self.step_local_normed(batch, kind='train')
                    #print(str(i) +' = '+str(loss_val))
                elif self.local_norm & self.diffuse: 
                    loss_val = self.step_local_normed_diffuse(batch, kind='train')
                    print(str(i) +' = '+str(loss_val))
                else:
                    loss_val = self.step(batch, kind='train')
                
                loss_temp = np.append(loss_temp, loss_val)
            
            self.train_loss = np.append(self.train_loss, np.mean(loss_temp))
            
            loss_temp = np.array([])
            for batch in ML_data.bgen_test: 
                if self.local_norm  & (self.diffuse==False):
                    loss_val = self.step_local_normed(batch, kind='test')
                elif self.local_norm & self.diffuse: 
                    loss_val = self.step_local_normed_diffuse(batch, kind='train')
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
                    #print('here')
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

    def pred_local_normed(self, X, input_channels=[]): 
        X_vel = X[['U_x', 'U_y','V_x', 'V_y']].to_array().transpose(...,'variable')
        X_S = X[['Sx', 'Sy']].to_array().transpose(...,'variable')
        X_vel_mag = ( (X_vel**2).mean('variable') )**0.5 + 1e-10
        X_S_mag = ( (X_S**2).mean('variable') )**0.5 + 1e-10
        X_scale_mag = X['Lfilt']*1e3
        X_vel_normed = X_vel/X_vel_mag
        X_S_normed = X_S/X_S_mag

        psi_mag = jnp.asarray((X_vel_mag*X_S_mag*(X_scale_mag**2)).data[..., jnp.newaxis])
        
        if 'Lfilt' in input_channels:
            X_L = X[['Lfilt']].to_array().transpose(...,'variable')/400
            if 'hx' in input_channels:
                X_h = X[['hx','hy']].to_array().transpose(...,'variable')/1e-3
                X_input = jnp.asarray(xr.concat([X_vel_normed, X_S_normed, X_L, X_h], dim='variable').data)
            else:
                X_input = jnp.asarray(xr.concat([X_vel_normed, X_S_normed, X_L], dim='variable').data)
        else:
            if 'hx' in input_channels:
                X_h = X[['hx','hy']].to_array().transpose(...,'variable')/1e-3
                X_input = jnp.asarray(xr.concat([X_vel_normed, X_S_normed, X_h], dim='variable').data)
            else:
                X_input = jnp.asarray(xr.concat([X_vel_normed, X_S_normed], dim='variable').data)
                
            #X_input = jnp.asarray(xr.concat([X_vel_normed, X_S_normed], dim='variable').data)

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



######################## Old

class ANN:
    
    def __init__(self, shape=[24,24,2], num_in=7, bias=True, diffuse=False):
        self.shape = shape
        self.bias = bias
        self.model, self.params = ml_hf.initialize_model(shape, num_in , bias=self.bias, diffuse=diffuse)
        
    def count_parameters(self):
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(param_count)