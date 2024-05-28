#!/usr/bin/env python
# coding: utf-8

import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

# Set JAX_TRACEBACK_FILTERING to off for detailed traceback
#os.environ['JAX_TRACEBACK_FILTERING'] = 'on'


# @title Imports
import dataclasses
import datetime
import functools
import math
import re
from typing import Optional
from glob import glob
import gc
import datetime

import cartopy.crs as ccrs
#from google.cloud import storage
from . import autoregressive_lam as autoregressive
from . import casting
from . import checkpoint
from . import data_utils
from . import graphcast_lam as graphcast
from . import normalization
from . import rollout
from . import xarray_jax
from . import xarray_tree
from .data_generator import (ZarrDataGenerator,
                             load_wofscast_data, 
                             wofscast_data_generator, 
                             wofscast_batch_generator, 
                             to_static_vars, 
                             add_local_solar_time)

import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray #as xr

from .utils import count_total_parameters, save_model_params, load_model_params 

# For training the weights!
import optax
import jax
import numpy as np
import jax.numpy as jnp

from jax import device_put
from jax import pmap, device_put, local_device_count
# Check available devices
print("Available GPU devices:", jax.devices())
from jax import tree_util

import time 
import wandb


def add_diagnostics(accum_diagnostics, diagnostics, ind): 
    """Append the diagnostics from each epoch to running diagnostic dict"""
    for var, val in diagnostics.items():
        accum_diagnostics[var][ind] = float(val)
  
    return accum_diagnostics

def compute_avg_diagnostics(diag_list, target_vars):
    """Compute the sub-epoch-average diagnostics (due to GPU parallelization)"""
    temp_dict = {v : [] for v in target_vars}
    for diag in diag_list: 
        for v in target_vars:
            temp_dict[v].append(float(diag[v]))
            
    final_dict = {v: np.mean(temp_dict[v]) for v in target_vars}

    return final_dict

def plot_diagnostics(accum_diag, ind):
    fig, ax = plt.subplots(dpi=300, figsize=(6,4))
    for v in accum_diag.keys():
        line, = ax.plot(accum_diag[v], label=v)
        y = accum_diag[v][-1]  # Last value in the series
        x = len(accum_diag[v]) - 1  # Last index
        
        ax.annotate(v, xy=(x, y), xytext=(5,5), textcoords="offset points",
                    color=line.get_color(), fontsize=6)

            
    ax.set(xlabel='Epoch', 
           ylabel='Loss', 
           xlim=[0, x+5], title=f'Diagnostic Phase {ind}')
    ax.grid(alpha=0.5)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(f'diagnostics_{ind}.png')


def construct_wrapped_graphcast(model_config: graphcast.ModelConfig, 
                                task_config: graphcast.TaskConfig,
                                norm_stats: dict
                               ):
    """Constructs and wraps the GraphCast Predictor."""
    # Deeper one-step predictor.
    predictor = graphcast.GraphCast(model_config, task_config)

    # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
    # from/to float32 to/from BFloat16.
    predictor = casting.Bfloat16Cast(predictor)

    # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
    # BFloat16 happens after applying normalization to the inputs/targets.
    predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=norm_stats['diffs_stddev_by_level'],
      mean_by_level=norm_stats['mean_by_level'],
      stddev_by_level=norm_stats['stddev_by_level']
    )

    # Wraps everything so the one-step model can produce trajectories.
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    
    return predictor


# Function for deployment. Used to make predictions on new data and rollout. 
@hk.transform_with_state
def run_forward(model_config, task_config, norm_stats, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config, norm_stats)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)

@hk.transform_with_state
def loss_fn(model_config, task_config, norm_stats, inputs, targets, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config, norm_stats)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn, model_obj, norm_stats):
    return functools.partial(
      fn, model_config=model_obj.model_config, task_config=model_obj.task_config, norm_stats=norm_stats)

'''
def grads_fn(params, state, inputs, targets, forcings, model_config, task_config, norm_stats):
    def compute_loss(params, state, inputs, targets, forcings):
        (loss, diagnostics), next_state = loss_fn.apply(params, state, 
                                                        jax.random.PRNGKey(0), 
                                                        model_config, 
                                                        task_config, norm_stats,
                                                        inputs, targets, forcings)
        
        return loss, (diagnostics, next_state)
    
    # Compute gradients and auxiliary outputs
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(compute_loss, has_aux=True)(params, state, 
                                                                                              inputs, targets, 
                                                                                              forcings)
    
    # Compute the global norm of all gradients
    total_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in tree_util.tree_leaves(grads)))

    # Clip gradients if the total norm exceeds the threshold
    # Use the threshold from Lam et al. (GraphCast) 
    def clip_grads(g, clip_norm=32):
        return jnp.where(total_norm > clip_norm, g * clip_norm / total_norm, g)

    clipped_grads = tree_util.tree_map(clip_grads, grads)

    return loss, diagnostics, next_state, clipped_grads
'''   

def grads_fn_parallel(params, state, inputs, targets, forcings, model_config, task_config, norm_stats):
    def compute_loss(params, state, inputs, targets, forcings):
        (loss, diagnostics), next_state = loss_fn.apply(params, state, 
                                                        jax.random.PRNGKey(0), 
                                                        model_config, 
                                                        task_config, norm_stats, 
                                                        inputs, targets, forcings)
        return loss, (diagnostics, next_state)
    
    # Compute gradients and auxiliary outputs
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(compute_loss, has_aux=True)(params, state, 
                                                                                              inputs, targets, 
                                                                                              forcings)
    
    # Combine the gradient across all devices (by taking their mean).
    grads = jax.lax.pmean(grads, axis_name='devices')

    # Compute the global norm of all gradients
    total_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in tree_util.tree_leaves(grads)))

    # Clip gradients if the total norm exceeds the threshold
    def clip_grads(g, clip_norm=32):
        return jnp.where(total_norm > clip_norm, g * clip_norm / total_norm, g)

    clipped_grads = tree_util.tree_map(clip_grads, grads)

    return loss, diagnostics, next_state, clipped_grads

def modify_path_if_exists(original_path):
    """
    Modifies the given file path by appending a version number if the path already exists.
    Useful for not overwriting existing version of the WoFSCast model parameters. 

    Args:
        original_path (str): The original file path.

    Returns:
        str: A modified file path if the original exists, otherwise returns the original path.
    """
    # Check if the file exists
    if not os.path.exists(original_path):
        return original_path

    # Split the path into directory, basename, and extension
    directory, filename = os.path.split(original_path)
    basename, extension = os.path.splitext(filename)

    # Iteratively modify the filename by appending a version number until an unused name is found
    version = 1
    while True:
        new_filename = f"{basename}_v{version}{extension}"
        new_path = os.path.join(directory, new_filename)
        if not os.path.exists(new_path):
            return new_path
        version += 1

def shard_xarray_dataset(dataset : xarray.Dataset, num_devices : int = None):
    """
    Shards an xarray.Dataset across multiple GPUs.

    Parameters:
    - dataset: xarray.Dataset to be sharded.
    - num_devices: Number of GPUs to shard the dataset across. If None, uses all available GPUs.

    Returns:
    A list of sharded xarray.Dataset, one for each GPU.
    """
    if num_devices is None:
        num_devices = jax.local_device_count()

    if num_devices == 1:
        return dataset 
        
    # Assuming the first dimension of each data variable is the batch dimension
    batch_size = next(iter(dataset.data_vars.values())).shape[0]
    shard_size = batch_size // num_devices

    if batch_size % num_devices != 0:
        raise ValueError(f"Batch size {batch_size} is not evenly divisible by the number of devices {num_devices}.")

    sharded_datasets = []
    for i in range(num_devices):
        start_idx = i * shard_size
        end_idx = start_idx + shard_size
        # Use dataset.isel to select a subset of the batch dimension for each shard
        shard = dataset.isel(indexers={'batch': slice(start_idx, end_idx)})
        sharded_datasets.append(shard)

    return xarray.concat(sharded_datasets, dim='devices')

def replicate_for_devices(params, num_devices=None):
    """Replicate parameters for each device using jax.device_put_replicated."""
    if num_devices is None:
        num_devices = jax.local_device_count()

    if num_devices == 1:
        return params 
        
    return jax.tree_map(lambda x: jnp.array([x] * num_devices), params)
    
# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
    return lambda **kw: fn(**kw)[0]

# Always pass params and state, so the usage below are simpler
def with_params(fn, model_obj):
     return functools.partial(fn, params=model_obj.model_params, state=model_obj.state)
            
class WoFSCastModel:
    """
    A class for training the WoFSCast model, designed to predict weather phenomena using
    the same Graph Neural Network (GNN) approach for GraphCast. The training process is divided into three phases, 
    each with its own number of epochs. The model supports multi-GPU training and periodic 
    checkpointing.

    Parameters:
    - mesh_size (int): The number of mesh subdivisions. The lowest resolution mesh divides the domain into 
                       4 triangles meeting at the center of the domain. Each additional level divides the
                       triangles into subsets. Typically between 4-6. 
    - latent_size (int): The size of the latent vectors in the GNN.
    - gnn_msg_steps (int): The number of message passing steps in the GNN. 
                           Each step is a new MLP or transformer layer.
    - hidden_layers (int): The number of hidden layers in the GNN. Each layer has the same latent_size
    - grid_to_mesh_node_dist (float): Fraction of the maximum distance between mesh nodes 
                                      at the finest resolution. Acts the search distance linking grid points
                                      to mesh nodes for the encoding GNN.  
    - n_epochs_phase1 (int): The number of training epochs for phase 1.
    - n_epochs_phase2 (int): The number of training epochs for phase 2.
    - n_epochs_phase3 (int): The number of training epochs for phase 3.
    - total_timesteps (int): The total number of prediction timesteps.
    - batch_size (int): The batch size used for training.
    - checkpoint (bool): Whether to checkpoint the model during training.
    - norm_stats_path (str): The file path for the normalization statistics.
    - out_path (str): The output path for the trained model and checkpoints.
    - checkpoint_interval (int): The interval (in epochs) between checkpoints.
    - use_multi_gpus (bool): Whether to use multiple GPUs for training.
    - verbose (int): Verbosity level for logging.

    Methods:
    __init__: Initializes the WoFSCastModel instance with the given parameters, ensuring
              not to overwrite existing models, and preparing the model for training.
    """
    
    def __init__(self, 
                 task_config : graphcast.TaskConfig = None, 
                 mesh_size : int = 3, 
                 latent_size : int =32, 
                 gnn_msg_steps : int =4, 
                 hidden_layers: int =1, 
                 grid_to_mesh_node_dist: float = 0.6,
                 use_transformer : bool = False, 
                 k_hop : int = 8, 
                 num_attn_heads : int = 4 , 
                 n_epochs_phase1 : int = 5 , 
                 n_epochs_phase2 : int = 5 ,
                 n_epochs_phase3 : int = 10 ,
                 total_timesteps : int  = 12, 
                 checkpoint : bool =True,
                 norm_stats_path : str = '/work/mflora/wofs-cast-data/normalization_stats', 
                 out_path : str = '/work/mflora/wofs-cast-data/model/wofscast.npz',
                 checkpoint_interval : int = 100,
                 use_multi_gpus=True, 
                 verbose=2, 
                 domain_size=None, 
                 tiling=None, 
                 loss_weights = None,
                 generator_kwargs = {},
                ):

        self.k_hop = k_hop
        self.loss_weights = loss_weights 
        
        self.verbose = verbose
        self.domain_size = domain_size 
        self.tiling = tiling
        
        # Ensure not to overwrite an existing model!
        out_path = modify_path_if_exists(out_path)
        
        # Initialize the Weights & Biases project for logging and tracking 
        # training loss and other metrics. 
        project_name = os.path.basename(out_path).replace('.npz', '')
        wandb.init(project='wofscast',
                   name = project_name,
                  ) 

        # Training Parameters. 
        self.n_epochs_phase1 = n_epochs_phase1
        self.n_epochs_phase2 = n_epochs_phase2
        self.n_epochs_phase3 = n_epochs_phase3
        
        self.epoch_interval_phase3 = n_epochs_phase3 // total_timesteps
        if self.epoch_interval_phase3 < 1:
            self.epoch_interval_phase3 = 1
        
        self.total_timesteps = total_timesteps
        self.checkpoint = checkpoint
        self.checkpoint_interval= checkpoint_interval
        
        # Initialize the GraphCast TaskConfig obj.
        if task_config is not None: 
            self._init_task_config(task_config)
        
            # Initialize the GraphCast ModelConfig obj. 
            self._init_model_config(mesh_size, latent_size, 
                           gnn_msg_steps, hidden_layers, grid_to_mesh_node_dist, 
                           loss_weights, k_hop, use_transformer, num_attn_heads)
        
            self._init_training_loss_diagnostics()
        
        # Load the normalization statistics. 
        self._load_norm_stats(norm_stats_path)
        
        self.out_path = out_path 

        self.generator_kwargs = generator_kwargs 
        self.clip_norm = 32 
        self.use_multi_gpus = use_multi_gpus 
    
    def fit_generator(self, paths, 
                      model_params=None, state={}, opt_state=None):
        """Fit the WoFSCast model using the 3-Phase method outlined in Lam et al.
        using a generator method. 
        
            If model_params, state, and opt_state, then those are initialized.
            Otherwise, it is possible to continue training by passing those
            args in. 
        
        Parameters
        ---------------
            generator: A data generator that returns inputs, targets, forcings. 
        """
        self.num_devices = 1 
        if self.use_multi_gpus: 
            # Assume you have N GPUs
            self.num_devices = jax.local_device_count()
            # Note: Using the GraphCast xarray-based JAX pmap; JAX documentation 
            # says we dont need to jit the function, pmap will handle it. 
            train_step_func = xarray_jax.pmap(with_configs(grads_fn_parallel, self, self.norm_stats), 
                                              dim='devices', axis_name='devices')
      
        else:
            train_step_func = jax.jit(with_configs(grads_fn, self, self.norm_stats))

        # 3-phase learning. 
        total_phases = [1,2,3]
        if self.n_epochs_phase3 < 1:
            total_phases = [1,2]
        
        for phase_num in total_phases:
            if phase_num==1:
                if self.verbose > 0:
                    print('\nStarting Phase 1 learning')
                    print('Training with a linearly increasing learning rate')
                n_timesteps = 1
            elif phase_num==2:
                if self.verbose > 0:
                    print('\nStarting Phase 2 learning...')
                    print('Training with a cosine decaying schedule...')
                n_timesteps = 1 
           
            scheduler = self._init_learning_rate_scheduler(phase_num)
            
            for epoch in range(getattr(self, f'n_epochs_phase{phase_num}')):
                
                # Print the current datetime. 
                print('Current Time: ', datetime.datetime.now())
                
                # Get the current learning rate from the scheduler 
                lr = scheduler(epoch)
                
                # Init the optimiser
                optimizer = self._init_optimizer(lr)
                
                if phase_num == 3:
                    # For the final phase, steadily increase the lead time evaluated. 
                    # Steadily increase the lead time evaluated. 
                    if epoch % self.epoch_interval_phase3== 0: 
                        n_timesteps+=1
                        if n_timesteps > self.total_timesteps:
                            n_timesteps = self.total_timesteps-1
                
                model_params, state, opt_state = self._fit_batch(
                           paths,  
                           train_step_func, 
                           model_params, 
                           state, 
                           opt_state,
                           optimizer, 
                           epoch, 
                           phase_num,
                           )
                
                # Save the model ever so often!
                if epoch % self.checkpoint_interval == 0 and self.checkpoint:
                    if self.verbose > 1:
                        print('Saving model params....')
                    self.save(model_params, state)
        
        # Save the final model params 
        print('Saving the final model...')
        self.save(model_params, state)
        
    def _fit_batch(self, 
                   paths, 
                   train_step_func, 
                   model_params, 
                   state, 
                   opt_state,
                   optimizer, 
                   epoch, 
                   phase_num, 
                   ): 
        
        # Create mini-batches for the current epoch and compute gradients. 
        total_loss = 0. 
        batch_count = 0 
        total_diagnostics = {}
        
        generator = ZarrDataGenerator( self.task_config,
                                  **self.generator_kwargs
                                 )(paths) 
        
        for batch_inputs, batch_targets, batch_forcings in generator: 
            if model_params is None:
                # Initialize the model parameters
                model_params, state = self._init_model_params_and_state(batch_inputs, 
                                                                        batch_targets,
                                                                        batch_forcings, 
                                                                        optimizer
                                                                           )

            if opt_state is None:
                # Initialize the optimizer state only at the beginning
                opt_state = optimizer.init(model_params)
                    
            # Split the inputs, targets, forcings, model_params, state, opt_state 
            # up to send to multiple GPUs.
            batch_inputs_sharded = shard_xarray_dataset(batch_inputs, self.num_devices)
            batch_targets_sharded = shard_xarray_dataset(batch_targets, self.num_devices)
            batch_forcings_sharded = shard_xarray_dataset(batch_forcings, self.num_devices)
   
            model_params_sharded = replicate_for_devices(model_params, self.num_devices)
            state_sharded = replicate_for_devices(state, self.num_devices)

            loss, diagnostics, state, grads = train_step_func(model_params_sharded, 
                                                              state_sharded,            
                                                              batch_inputs_sharded, 
                                                              batch_targets_sharded, 
                                                              batch_forcings_sharded,
                                                             )
                
            if self.verbose > 2:
                print(f'{loss=}')

            # Update parameters
            model_params, opt_state = self.update_step(optimizer, model_params, grads, opt_state)
            
            # Unreplicate model_params 
            # Like the loss, the leaves of params have an extra leading dimension,
            # so we take the params from the first device.
            model_params = jax.device_get(jax.tree_map(lambda x: x[0], model_params))
            
            loss_val = np.mean(np.asarray(loss)).item()
            
            for k, v in diagnostics.items():
                total_diagnostics[k] = total_diagnostics.get(k, 0) + np.mean(np.asarray(v)).item()
                
            if np.isnan(loss_val):
                raise ValueError('Loss includes NaN value. Ending the training...')
            
            total_loss += loss_val 
                
            del loss, diagnostics, grads
            gc.collect() 
            
            if self.verbose > 1:
                print(f'SubEpoch : {batch_count}')
                
            batch_count+=1
        
        avg_diagnostics = {k: v / batch_count for k, v in total_diagnostics.items()}
    
        self.accum_diagnostics[f'Phase {phase_num}'] = add_diagnostics(
            self.accum_diagnostics[f'Phase {phase_num}'], avg_diagnostics, epoch)
        
        self.training_loss[f'Phase {phase_num}'][epoch] = total_loss / batch_count 

        # Log metrics to wandb
        wandb.log({
            f"Phase {phase_num} Loss": total_loss / batch_count,
            **{f"Phase {phase_num} {k}": v / batch_count for k, v in total_diagnostics.items()}
        })
        
        if self.verbose > 0:
            print(f"Phase {phase_num} Epoch: {epoch}.....Loss: {total_loss/batch_count:.5f}")
        
        return model_params, state, opt_state
    
    def _transpose(self, ds, dims):
        return ds.transpose(*dims, missing_dims='ignore')
    
    def get_inputs(self, dataset, lead_times=slice('10min', '120min')): 
        # Add the local solar time variables:
        # TODO: To get the future forcing variables, need 
        # to replace with the extend targets_template.
        dataset = add_local_solar_time(dataset)
        
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                dataset, target_lead_times=lead_times,
                **dataclasses.asdict(self.task_config))

        inputs = inputs.expand_dims(dim='batch')
        targets = targets.expand_dims(dim='batch')
        forcings = forcings.expand_dims(dim='batch')
        
        return inputs, targets, forcings
    
    # Load the model 
    def load_model(self, path, 
                   **additional_model_config # For backwards compat.
                  ):
        
        with open(path, 'rb') as f:
            data = checkpoint.load(f, dict)
    
        #Unravel the task config. 
        _TASK_CONFIG_KEYS = list(vars(graphcast.TaskConfig)['__dataclass_fields__'].keys())
    
        task_config = data['task_config']
    
        task_config_dict = {}
        for key in _TASK_CONFIG_KEYS: 
            if isinstance(task_config.get(key, None), dict):
                # unravel
                if key == 'pressure_levels':
                    task_config_dict[key] = [int(item) for _, item in task_config[key].items()]
                else:
                    task_config_dict[key] = [str(item) for _, item in task_config[key].items()]
            elif key == 'input_duration':
                task_config_dict[key] = str(task_config[key])
            else:
                task_config_dict[key] = task_config.get(key, None)
    
        if self.domain_size is None:
            self.domain_size = 150 #data['domain_size']
            
        data['task_config'] = task_config_dict
   
        self.model_params = data['parameters']
        self.state = {}
        self._init_task_config_run(data['task_config'])
        self._init_model_config_run(data['model_config'], **additional_model_config)

    def predict(self, inputs, targets, forcings): 
       # Convert the constant fields to time-independent (drop time dim)
        inputs = to_static_vars(inputs)

        # It is crucial to tranpose the data so that level is last 
        # since the input includes 2D & 3D variables. 
        inputs = self._transpose(inputs, ['batch', 'time', 'lat', 'lon', 'level'])
        targets = self._transpose(targets, ['batch', 'time', 'lat', 'lon', 'level'])
        forcings = self._transpose(forcings, ['batch', 'time', 'lat', 'lon'])
        
        # TODO: use the extend_targets_template from rollout.py. 
        #targets_template = self.expand_time_dim(targets) * np.nan
        targets_template = targets 
        
        #print("Inputs:           ", inputs.dims.mapping)
        #print("Target Template:  ", targets_template.dims.mapping)
        #print("Forcings:         ", forcings.dims.mapping)
        
        run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
            run_forward.apply, self, self.norm_stats)), self))

        # @title Autoregressive rollout (keep the loop in JAX)
        predictions = rollout.chunked_prediction(
            run_forward_jitted,
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings)

        return predictions #, targets, inputs

    def _init_model_params_and_state(self, inputs, targets, forcings, optimizer):
        # Check that the norm stats haven't changed!!
        if 'level' in inputs.dims.keys(): 
            norm_stat_count = self.norm_stats['mean_by_level']['level'].shape[0]
            input_count = inputs['level'].shape[0]
                
            assert norm_stat_count == input_count, "Norm Stat is not compatiable with the inputs!"
                
        init_jitted = jax.jit(with_configs(run_forward.init, self, self.norm_stats))

        model_params, state = init_jitted(
                    rng=jax.random.PRNGKey(0),
                    inputs=inputs,
                    targets_template=targets,
                    forcings=forcings, 
                    )

        num = count_total_parameters(model_params)
        if self.verbose > 0:
            print(f'Num of Model Parameters: {num}')
                  
        return model_params, state
  
    def _init_optimizer(self, lr):
        # Setup optimizer with the current learning rate
        return optax.adam(lr, b1=0.9, b2=0.95, eps=1e-8)
        
    def _init_training_loss_diagnostics(self):
        self.training_loss = {'Phase 1' : np.zeros(self.n_epochs_phase1), 
                         'Phase 2' : np.zeros(self.n_epochs_phase2), 
                         'Phase 3' : np.zeros(self.n_epochs_phase3), 
                        }
        
        self.accum_diagnostics = {"Phase 1": {v : np.zeros(self.n_epochs_phase1) for v in self.target_vars},
                                  "Phase 2": {v : np.zeros(self.n_epochs_phase2) for v in self.target_vars},
                                  "Phase 3": {v : np.zeros(self.n_epochs_phase3) for v in self.target_vars},
                                 
                                 }

    def _init_learning_rate_scheduler(self, phase):
        if phase == 1:
            # Setup the learning rate schedule
            if self.verbose > 0:
                print('Initializing a linear learning rate scheduler...')
            start_learning_rate = 1e-6  # Start from 0
            end_learning_rate = 1e-3  # Increase to 1e-3
            scheduler = optax.linear_schedule(init_value=start_learning_rate, 
                                 end_value=end_learning_rate, 
                                 transition_steps=self.n_epochs_phase1)
            
        elif phase == 2:
            if self.verbose > 0:
                print('Initializing a cosine decay learning rate scheduler...')
            scheduler = optax.cosine_decay_schedule(init_value=1e-3, 
                                                  decay_steps=self.n_epochs_phase2, 
                                                  alpha=0)  # alpha=0 makes it decay to 0    
        else: 
            if self.verbose > 0:
                print('Initializing a constant learning rate scheduler...')
            scheduler = optax.constant_schedule(3e-7)
        
        return scheduler 

    def update_step(self, optimizer, params, grads, opt_state):
        """Performs a single update step by applying gradients to parameters."""
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    def _init_task_config_run(self, data): 

        domain_size =self.domain_size if self.domain_size else data['domain_size']
            
        self.task_config = graphcast.TaskConfig(
              input_variables=data['input_variables'],
              target_variables=data['target_variables'],
              forcing_variables=data['forcing_variables'],
              pressure_levels=data['pressure_levels'],
              input_duration=data['input_duration'],
              n_vars_2D = data['n_vars_2D'],
              domain_size = domain_size, 
              tiling = self.tiling, 
              train_lead_times = data.get('train_lead_times', None)
          )
        
        if self.verbose > 2:
            print(self.task_config)
        
    
    def _init_model_config_run(self, data, **additional_model_config):
        
        k_hop = data.get('k_hop', None)
        if k_hop is None: 
            k_hop = additional_model_config.get('k_hop', 8) 
        
        use_transformer = data.get('use_transformer', None)
        if use_transformer is None: 
            use_transformer = additional_model_config.get('use_transformer', False) 
        
        num_attn_heads = data.get('num_attn_heads', None)
        if num_attn_heads is None: 
            num_attn_heads = additional_model_config.get('num_attn_heads', 4) 
        
        self.model_config = graphcast.ModelConfig(
              resolution=int(data['resolution']),
              mesh_size=int(data['mesh_size']),
              latent_size=int(data['latent_size']),
              gnn_msg_steps=int(data['gnn_msg_steps']),
              hidden_layers=int(data['hidden_layers']),
              grid_to_mesh_node_dist=int(data['grid_to_mesh_node_dist']),
              loss_weights = data['loss_weights'],
              k_hop = k_hop,
              use_transformer = use_transformer,
              num_attn_heads = num_attn_heads
        )
        
        if self.verbose > 2:
            print(self.model_config)
        
    
    def _init_task_config(self, task_config):
        """Initialize the TaskConfig object used in the GraphCast code."""
        self.task_config = task_config
        
        if hasattr(task_config, 'train_lead_times'):
            self.train_lead_times = task_config.train_lead_times
        else:
            self.train_lead_times = None
            
        self.target_vars = task_config.target_variables
        
        
        if self.verbose > 2:
            print(self.task_config) 
        
        
    def _init_model_config(self, mesh_size, latent_size, 
                           gnn_msg_steps, hidden_layers, grid_to_mesh_node_dist, 
                           loss_weights, k_hop, use_transformer, num_attn_heads
                          ):
        # Weights used in the loss equation.
        if self.loss_weights is None:
            loss_weights = {v : 1.0 for v in self.target_vars}
        else:
            loss_weights = self.loss_weights
            
        #loss_weights['W'] = 2.0
        #loss_weights['UP_HELI_MAX'] = 2.0
        
        self.model_config = graphcast.ModelConfig(
              resolution=0,
              mesh_size=mesh_size,
              latent_size=latent_size,
              gnn_msg_steps=gnn_msg_steps,
              hidden_layers=hidden_layers,
              grid_to_mesh_node_dist=grid_to_mesh_node_dist, 
              loss_weights = loss_weights,
              k_hop = k_hop,
              use_transformer = use_transformer,
              num_attn_heads = num_attn_heads
        )
        
        if self.verbose > 2:
            print(self.model_config)
        
    
    def _load_norm_stats(self, path):     
        mean_by_level = xarray.load_dataset(os.path.join(path, 'mean_by_level.nc'))
        stddev_by_level = xarray.load_dataset(os.path.join(path, 'stddev_by_level.nc'))
        diffs_stddev_by_level = xarray.load_dataset(os.path.join(path, 'diffs_stddev_by_level.nc'))

        self.norm_stats = {'mean_by_level': mean_by_level, 
                      'stddev_by_level' : stddev_by_level,
                      'diffs_stddev_by_level' : diffs_stddev_by_level
                     }
        
    def save(self, model_params, state):
        
        model_data = {'parameters' : model_params, 
                'state' : state,
                'model_config' : self.model_config, 
                'task_config' : self.task_config, 
               }
        
        print('Saving model...')
        with open(self.out_path, 'wb') as io_byte:
            checkpoint.dump(io_byte, model_data)
             
    def plot_diagnostics(self):
        # Save the diagnostics. 
        plot_diagnostics(self.accum_diagnostics['Phase 1'], 1)
        plot_diagnostics(self.accum_diagnostics['Phase 2'], 2)
        if self.n_epochs_phase3 > 1:
            plot_diagnostics(self.accum_diagnostics['Phase 3'], 3)
    
    def plot_training_loss(self, save=True):
        fig, axes = plt.subplots(dpi=300, ncols=3, figsize=(12,4))
        for ax, (title, loss) in zip(axes.flat, self.training_loss.items()):
            ax.set(ylabel='Loss', xlabel='Epoch', title=title)
            ax.plot(loss)
            ax.grid(alpha=0.5)

        plt.tight_layout()
        if save:
            plt.savefig('training_results.png')
        else:
            return fig, axes

