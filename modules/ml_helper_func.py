import jax
from typing import Any, Callable, Sequence
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn
import numpy as np

# Define network
class pointwise_model(nn.Module):
    features: Sequence[int]
    bias: True
    
    def setup(self):
        # self.features = features
        # self.bias = bias
        self.layers = [nn.Dense(feat, use_bias=self.bias) for feat in self.features]
        
    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i!=len(self.layers)-1:
                x = nn.relu(x)
        return x

class pointwise_model_diffuse(nn.Module):
    features: Sequence[int]
    bias: True
    
    def setup(self):
        # self.features = features
        # self.bias = bias
        self.layers = [nn.Dense(feat, use_bias=self.bias) for feat in self.features]
        
    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i!=len(self.layers)-1:
                x = nn.relu(x)
        x = x.at[0].set( -nn.relu(x.at[0].get()) )
        #print(x)
        return x
    
def initialize_model(features, num_inputs, bias=True, random_key=0, diffuse=False):

    if diffuse: 
        model = pointwise_model_diffuse(features = features, bias = bias)
        print('Diffuse')
    else:
        model = pointwise_model(features = features, bias = bias)
    
    key1, key2 = random.split(random.PRNGKey(random_key))
    
    x = random.normal(key1, (num_inputs,) )
    
    params = model.init(key2, x)
    
    return model, params


# loss functions
def mse(params, apply_fn, x_batched, y_batched):
    
    # Define squared loss for a single pair (x,y), where y can be a vector (multi-dim output) 
    def squared_error(x,y):
        pred = apply_fn(params, x)
        return jnp.inner(y-pred, y-pred) / 2.0
    
    return jnp.nanmean(jax.vmap(squared_error)(x_batched, y_batched), axis=0) # 0 is sample axis

def mlse(params, apply_fn, x_batched, y_batched):
    
    # Define squared loss for a single pair (x,y), where y can be a vector (multi-dim output) 
    def log_squared_error(x,y):
        pred = apply_fn(params, x)
        return jnp.log10(jnp.inner(y-pred, y-pred) + 1e-20)
    
    return jnp.nanmean(jax.vmap(log_squared_error)(x_batched, y_batched), axis=0) # 0 is sample axis

def mse_local_norm(params, apply_fn, x_batched, y_batched, Psi_mag):
    
    # Define squared loss for a single pair (x,y), where y can be a vector (multi-dim output) 
    def squared_error(x,y,Psi_mag):
        pred = apply_fn(params, x) * Psi_mag
        return jnp.inner(y-pred, y-pred) / 2.0
    
    return jnp.nanmean(jax.vmap(squared_error)(x_batched, y_batched, Psi_mag), axis=0) # 0 is sample axis

# def mse_local_norm_diffuse(params, apply_fn, x_batched, y_batched, Psi_mag, S_mag):
    
#     # Define squared loss for a single pair (x,y), where y can be a vector (multi-dim output) 
#     def squared_error(x,y,Psi_mag, S_mag):
#         pred = apply_fn(params, x) * Psi_mag
#         #print(pred)
#         print(pred.shape, S_mag.shape)
#         print(jax.device_get(pred))
#         #pred[0] =  - jnp.abs(pred[0])* S_mag
#         #print(- jnp.abs(pred[0])* S_mag)
#         pred = pred.at[...,0].set(- jnp.abs(pred.at[...,0].get())* S_mag)
#         return jnp.inner(y-pred, y-pred) / 2.0
    
#     return jnp.nanmean(jax.vmap(squared_error)(x_batched, y_batched, Psi_mag, S_mag), axis=0) # 0 is sample axis


def drop_nan(ds, var='uT'):
    return ds.where(~np.isnan(ds[var]), drop=True)

def zero_nans(ds, var='uT'):
    return ds.where(~np.isnan(ds[var]), 0.)



@jax.jit
def train_step(state, X_train, y_train):
    #X_train = jnp.asarray(batch[input_channels].to_array().transpose(...,'variable').data)
    #y_train = jnp.asarray(batch[output_channels].to_array().transpose(...,'variable').data)
    #print(X_train)
    loss, grads = loss_grad_fn(state.params, X_train, y_train)
    
    state = state.apply_gradients(grads=grads)
    
    return state, loss


@jax.jit
def eval_step(state, X_test, t_test):
    
    loss = mse(state.params, X_test, y_test)
    
    return loss