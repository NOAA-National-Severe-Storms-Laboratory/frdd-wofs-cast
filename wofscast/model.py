"""
The main class WoFSCastModel builds on Google's GraphCast github demo:
    https://github.com/google-deepmind/graphcast/blob/main/graphcast_demo.ipynb

"""

import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

# For determinism (https://github.com/google/jax/discussions/10674)
#os.environ['XLA_FLAGS'] = (
#    '--xla_gpu_deterministic_ops=true '
#    '--xla_gpu_deterministic_ops=true '
#)
#os.environ['TF_DETERMINISTIC_OPS'] = '1' 

# XLA FLAGS set for GPU performance (https://jax.readthedocs.io/en/latest/gpu_performance_tips.html)
"""
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)
"""
import warnings
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
from . import autoregressive #_lam as autoregressive
from . import casting
from . import checkpoint
from . import graphcast_lam as graphcast
from . import normalization
from . import rollout
from . import xarray_jax
from . import xarray_tree
from .data_generator import (ZarrDataGenerator, 
                             add_local_solar_time, 
                             to_static_vars, 
                             replicate_for_devices) 

from .data_utils import add_derived_vars
from .toa_radiation import TOARadiationFlux

import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray #as xr

from .utils import count_total_parameters, save_model_params, load_model_params 

# For training the weights!
import optax
import jax
import numpy as np
import jax.numpy as jnp

from jax import device_put
from jax import pmap, device_put, local_device_count
from jax import tree_util

import time 
import wandb
from tqdm import tqdm 

from datetime import datetime

# Saves jax compiled code, which can speed up 
# compilation speeds for re-runs. If parameters are
# changed, then the code has to be re-compiled.
jax.config.update("jax_compilation_cache_dir", ".")


graphcast_name = 'params_GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz'

def add_batch_dim(ds, batch_size):
    # Repeat the data along the 'batch' dimension 
    ds = xarray.concat([ds] * batch_size, dim='batch')  

    return ds

def construct_wrapped_graphcast(model_config: graphcast.ModelConfig, 
                                task_config: graphcast.TaskConfig,
                                norm_stats: dict,
                                noise_level : Optional[float]=None,
                                gradient_checkpointing=False # For fine tuning on longer rollouts, then test turning it True.
                               ):
    """Constructs and wraps the GraphCast Predictor. Wrappers include 
    floating point precision convertion, normalization, and autoregression. 
    """
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
    predictor = autoregressive.Predictor(predictor, noise_level=noise_level, gradient_checkpointing=gradient_checkpointing)
    
    return predictor

# Function for deployment. Used to make predictions on new data and rollout. 
@hk.transform_with_state
def run_forward(model_config, task_config, norm_stats, noise_level, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config, norm_stats, noise_level)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, norm_stats, noise_level, inputs, targets, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config, norm_stats, noise_level)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))


# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn, model_obj, norm_stats, noise_level):
    return functools.partial(
      fn, model_config=model_obj.model_config, 
        task_config=model_obj.task_config, 
        norm_stats=norm_stats, noise_level=noise_level)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
    return lambda **kw: fn(**kw)[0]

# Always pass params and state, so the usage below are simpler
def with_params(fn, model_obj):
     return functools.partial(fn, params=model_obj.model_params, state=model_obj.state)

def with_optimizer(fn, optimizer):
    return functools.partial(fn, optimizer=optimizer)
    

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
    """
    
    def __init__(self, 
                 
                 # Model and input config
                 task_config : graphcast.TaskConfig = None, 
                 mesh_size : int = 3, 
                 loss_weights = None, 
                 noise_level: Optional[float] = None,
                 
                 # Model architecture
                 latent_size : int = 32, 
                 gnn_msg_steps : int =4, 
                 hidden_layers: int = 1, 
                 grid_to_mesh_node_dist: float = 0.6,
                 use_transformer : bool = False, 
                 k_hop : int = 8, 
                 num_attn_heads : int = 4, 
                 
                 n_steps : int = 100, 
                 learning_rate_scheduler : optax.Schedule = optax.constant_schedule(1e-4),
                 checkpoint : bool = True,
                 norm_stats_path : str = None, 
                 out_path : str = '/work/mflora/wofs-cast-data/model/wofscast.npz',
                 checkpoint_interval : int = 100,
                 parallel=True, 
                 
                 graphcast_pretrain=False,
                 verbose=1,
                 use_wandb = True,
                 adam_weight_decay = 0.1
                ):
        
        self.use_wandb = use_wandb
        self.wandb_config = {'mesh_size' : mesh_size, 
                             'latent_size' : latent_size,
                             'loss_weights' : loss_weights,
                             'gnn_msg_steps' : gnn_msg_steps,
                             'hidden_layers' : hidden_layers,
                             'grid_to_mesh_node_dist' : grid_to_mesh_node_dist,
                             'n_steps' : n_steps, 
                             'checkpoint_interval' : checkpoint_interval,
                             'task_config' : task_config,
                             'noise_level' : noise_level
                            }
        
        self.graphcast_pretrain = graphcast_pretrain
        if self.graphcast_pretrain:
            assert latent_size == 512, 'If graphcast_pretrain==True, latent_size must equal 512'
            assert gnn_msg_steps == 16, 'If graphcast_pretrain==True, gnn_msg_steps must equal 16'
            assert hidden_layers == 1, 'If graphcast_pretrain==True, hidden_layers must equal 1'
            assert use_transformer == False, 'If graphcast_pretrain==True, use_transformer=False'

        self.adam_weight_decay = adam_weight_decay    
        self.learning_rate_scheduler = learning_rate_scheduler 
        self.n_steps = n_steps 
        self.k_hop = k_hop
        self.loss_weights = loss_weights 
        self.noise_level = noise_level
        
        self.verbose = verbose
        
        # Ensure not to overwrite an existing model!
        out_path = modify_path_if_exists(out_path)
        self.out_path = out_path 
        
        self.checkpoint = checkpoint
        self.checkpoint_interval = checkpoint_interval
        
        # Initialize the GraphCast TaskConfig obj.
        self.norm_stats_path = norm_stats_path
        if task_config is not None: 
            self._init_task_config(task_config)
        
            # Initialize the GraphCast ModelConfig obj. 
            self._init_model_config(mesh_size, latent_size, 
                           gnn_msg_steps, hidden_layers, grid_to_mesh_node_dist, 
                           loss_weights, k_hop, use_transformer, num_attn_heads)
        
            # Load the normalization statistics. 
            self._load_norm_stats(norm_stats_path)
        
        self.clip_norm = 32.0 # used to clip gradients.  
        self.parallel = parallel
      
    def fit_generator(self, 
                      generator, 
                      model_params=None, 
                      state={}, 
                      return_params=False,
                      raise_errors=False
                      ):
        
        """Fit the WoFSCast model using the 3-Phase method outlined in Lam et al.
        using a generator method. 
        
            If model_params, state, and opt_state, then those are initialized.
            Otherwise, it is possible to continue training by passing those
            args in. 
        
        Parameters
        ---------------
            generator: A data generator that returns inputs, targets, forcings. 
            raise_errors : 
        """
        def train_step_parallel(params, 
                        state, 
                        opt_state,
                        inputs, 
                        targets, 
                        forcings, 
                        model_config, 
                        task_config, 
                        norm_stats,
                        noise_level, 
                        optimizer):
    
            def compute_loss(params, state, inputs, targets, forcings):
                (loss, diagnostics), next_state = loss_fn.apply(params, state, 
                                                        jax.random.PRNGKey(0), 
                                                        model_config, 
                                                        task_config, norm_stats, noise_level,
                                                        inputs, targets, forcings)
                return loss, (diagnostics, next_state)
    
            # Compute gradients and auxiliary outputs
            (loss, (diagnostics, next_state)), grads = jax.value_and_grad(compute_loss, has_aux=True)(params, state, 
                                                                                              inputs, targets, 
                                                                                              forcings)
    
            # Combine the gradient across all devices (by taking their mean).
            grads = jax.lax.pmean(grads, axis_name='devices')

            # combine the loss across devices
            loss = jax.lax.pmean(loss, axis_name='devices')
    
            updates, opt_state = optimizer.update(grads, opt_state, params=params)
            new_params = optax.apply_updates(params, updates)
    
            return new_params, opt_state , loss, diagnostics

        def train_step(params, 
               state, 
               opt_state,
               inputs, 
               targets, 
               forcings, 
               model_config, 
               task_config, 
               norm_stats,
               noise_level,
               optimizer):
    
            def compute_loss(params, state, inputs, targets, forcings):
                (loss, diagnostics), next_state = loss_fn.apply(params, state, 
                                                        jax.random.PRNGKey(0), 
                                                        model_config, 
                                                        task_config, 
                                                        norm_stats, 
                                                        noise_level,
                                                        inputs, targets, forcings)
                return loss, (diagnostics, next_state)
    
            # Compute gradients and auxiliary outputs
            (loss, (diagnostics, next_state)), grads = jax.value_and_grad(compute_loss, has_aux=True)(params, state, 
                                                                                              inputs, targets, 
                                                                                              forcings)
    
            # No need to average gradients across devices, as we are on a single device
            updates, opt_state = optimizer.update(grads, opt_state, params=params)
            new_params = optax.apply_updates(params, updates)
    
            return new_params, opt_state, loss, diagnostics

        
        # Initialize the optimizer. The setting used match Lam et al. (GraphCast)
        # and are hardcoded expect the learning rate scheduler, which is a class arg. 
        optimizer = optax.chain(
                optax.clip(self.clip_norm), # Clip gradients,.
                optax.adamw(
                learning_rate=self.learning_rate_scheduler,
                b1=0.9,
                b2=0.95,
                eps=1e-8,
                weight_decay=self.adam_weight_decay
            )
            )
        
        self.wandb_config['optimizer'] = optimizer 
        
        # Initialize the Weights & Biases project for logging and tracking 
        # training loss and other metrics. 
        project_name = os.path.basename(self.out_path).replace('.npz', '')
        
        if self.use_wandb:
            wandb.init(project='wofscast',
                   name = project_name,
                   config = self.wandb_config
                  ) 
        
        # Set the number of GPUs to the device count if parallel, otherwise set it to 1. 
        self.num_devices = jax.local_device_count() if self.parallel else 1

        if self.parallel and self.num_devices == 1:
            if raise_errors:
                raise ValueError('parallel=True, but only 1 GPU is available! Setting parallel=False')
            else:
                warnings.warn('parallel=True, but only 1 GPU is available! Setting parallel=False')
            self.parallel = False

        train_fn = with_optimizer(with_configs(train_step_parallel if self.parallel else train_step,
                                       self, self.norm_stats, self.noise_level), optimizer)

        if self.parallel:
            # Use xarray-based JAX pmap
            train_step_jitted = xarray_jax.pmap(train_fn, dim='devices', axis_name='devices')
        else:
            train_step_jitted = jax.jit(train_fn)
        
        #print('Starting JAX trace on line 411..')
        #jax.profiler.start_trace("/tmp/tensorboard")
        
        # Load a single batch. These pre-allocate space 
        # and we'll use a trick below to update these datasets 
        # with data from newly loaded batches. 
        inputs, targets, forcings = generator.generate()    
        
        if self.num_devices==1: 
            if 'devices' in inputs.dims:
                if raise_errors:
                    raise KeyError("Found 'devices' in the inputs dimensions, but num_devices=1")
                else:
                    warnings.warn("Found 'devices' in the inputs dimensions, but num_devices=1. Removing 'devices' dimension..")
                inputs = inputs.isel(devices=0)
                targets = targets.isel(devices=0)
                forcings = forcings.isel(devices=0) 
        
        ##print(f'{inputs.dims=}')
        #print(f'{targets=}')
        #print(f'{forcings=}')
        
        if model_params is None: 
        
            init_jitted = jax.jit(with_configs(run_forward.init, self, self.norm_stats, self.noise_level))

            # To initialize the model parameters and state, select data from 
            # one device and use only a single batch to speed up compilation time. 
            if self.verbose > 0:
                print("Initializing Model Parameters and State...")
            if self.num_devices==1:
                _inputs = inputs.isel(batch=[0])
                _targets = targets.isel(batch=[0])
                _forcings = forcings.isel(batch=[0])
        
            else:
                _inputs = inputs.isel(devices=0, batch=[0])
                _targets = targets.isel(devices=0, batch=[0])
                _forcings = forcings.isel(devices=0, batch=[0])
        
            model_params, state = init_jitted(
                    rng=jax.random.PRNGKey(0),
                    inputs=_inputs,
                    targets_template=_targets,
                    forcings=_forcings, 
                    )
            
            # The following will replace all up the initial embedding layers 
            # with weights from the GraphCast 36.7M parameter model. 
            # At the moment, there are no checks to ensure that the model 
            # built is compatiable with these weights. Sorry!
            if self.graphcast_pretrain:
                if self.verbose > 0:
                    print('\n Initializing from GraphCast 36.7M weights...')
            
                # Load the 36.7M GraphCast weights. 
                graphcast_path = os.path.join('/work/mflora/wofs-cast-data/graphcast_models', graphcast_name)

                with open(graphcast_path, 'rb') as f:
                    data = checkpoint.load(graphcast_path, dict)
            
                graphcast_params = data['params']
  
                model_params = update_params_with_graphcast(model_params, graphcast_params)

            num = count_total_parameters(model_params)
            if self.verbose > 0:
                print(f'\n Num of Model Parameters: {num}\n')
        
        opt_state = optimizer.init(model_params)
        
        # For GPU parallel processing, replicate the model parameters, model state, 
        # and the optimizer state on each device.
        model_params_replicated = replicate_for_devices(model_params, self.num_devices)
        state_replicated = replicate_for_devices(state, self.num_devices)
        opt_state_replicated = replicate_for_devices(opt_state, self.num_devices)
                
        del model_params, opt_state 
        
        # Run the code once to jit the train_step_jitted
        start_time = time.time()
        print(f'Compiling the model..')
        model_params_replicated, opt_state_replicated, loss, diagnostics = train_step_jitted(
                   model_params_replicated, 
                   state_replicated,
                   opt_state_replicated,
                   inputs, 
                   targets, 
                   forcings, 
               )
        run_time = time.time() - start_time
        print(f'Compiling Time: {run_time:.2f}')
        
        # Main processing loop.
        for step in tqdm(range(self.n_steps), total=self.n_steps, desc='Training'):
            start_time = time.time() 
            # Generate new data and update the pre-allocated space.
            # From the graph-ufs code: 
            # The purpose of the following code is best described as confusing
            # the jix.jat cache system. We start from a deepcopy of slice 0 where the jitting
            # is carried out, and sneakly update its values. If you use xarray update/copy etc
            # the cache system somehow notices, and either becomes slow or messes up the result
            # Copying variable values individually avoids both, fast and produces same results as before
        
            new_inputs, new_targets, new_forcings = generator.generate()

            for var_name, var in new_inputs.data_vars.items():
                inputs[var_name] = inputs[var_name].copy(deep=False, data=var.values)
            for var_name, var in new_targets.data_vars.items():
                targets[var_name] = targets[var_name].copy(deep=False, data=var.values)
            for var_name, var in new_forcings.data_vars.items():
                forcings[var_name] = forcings[var_name].copy(deep=False, data=var.values)
            
            #print(f'{inputs=}')
            #print(f'{targets=}')
            #print(f'{forcings=}')
            
            model_params_replicated, opt_state_replicated, loss, diagnostics = train_step_jitted(
                   model_params_replicated, 
                   state_replicated,
                   opt_state_replicated,
                   inputs, 
                   targets, 
                   forcings, 
               )
            
            # Convert the loss JAX array to numpy. 
            loss_val = np.mean(np.asarray(loss)).item()
            
            if np.isnan(loss_val):
                raise ValueError('Loss includes NaN value. Ending the training...')

            # Log metrics to wandb
            logs = {f"Loss": loss_val, 
                       "Epoch Time" : time.time() - start_time, 
                      }
            diagnostics = {f'{v}_loss' : np.mean(np.asarray(diagnostics[v])).item() for v in 
                           diagnostics.keys()} 
            
            if self.use_wandb: 
                wandb.log({**logs, **diagnostics})
 
            # Save the model ever so often!
            if step % self.checkpoint_interval == 0 and self.checkpoint:
                if self.verbose > 1:
                    print('Saving model params....')    
                self.save(model_params_replicated, state)
            
        #jax.profiler.stop_trace()
               
        # Save the final model params 
        print('Saving the final model...')
        self.save(model_params_replicated, state)
     
        if return_params:
            return jax.device_get(jax.tree_map(lambda x: x[0], model_params)), state
    
    def predict(self, inputs, targets, forcings, 
                initial_datetime=None, 
                n_steps=None, replace_bdry=True, 
                diffusion_model=None, 
                n_diffusion_steps=50): 
        """Predict using the WoFSCast"""
        
        # Ensure 'batch' dimension exists
        inputs = ensure_batch_dim(inputs, 'batch')
        targets = ensure_batch_dim(targets, 'batch')
        forcings = ensure_batch_dim(forcings, 'batch')
        
        extended_targets = targets
        extended_forcings = forcings
        
        if n_steps: 
            if initial_datetime is None:
                raise ValueError('If using n_steps, must provide an initial_datetime str or pd.Timestamp for the forcings')
            
            # Expects a pandas.Timestamp object. If it's a string or other format,
            # it will convert to a Timestamp object.
            if not isinstance(initial_datetime, pd.Timestamp):
                if isinstance(initial_datetime, str):
                    # Convert string to pandas.Timestamp
                    initial_datetime = pd.Timestamp(initial_datetime)
    
                else:
                    # If it's not a string, assume it's in datetime format and convert
                    # using the appropriate format string (adjust as needed).
                    init_dt_obj = datetime.strptime(initial_datetime, '%Y%m%d%H%M')
                    initial_datetime = pd.Timestamp(init_dt_obj)
                
                if n_steps and replace_bdry:
                    replace_bdry=False
                    warnings.warn('If using n_steps, then replace_bdry must be False. Setting it to False.')
                
            extended_targets = rollout.extend_targets_template(targets, 
                                                           required_num_steps=n_steps)
        
            extended_targets = extended_targets.transpose('batch', 'time', 'level', 'lat', 'lon')
        
            # Create the new datetime coordinate by adding the timedeltas to the initial datetime
            datetime_coord = initial_datetime + extended_targets['time'].data

            # Assign the new datetime coordinate to the dataset
            extended_targets = extended_targets.assign_coords(datetime=datetime_coord)

            if 'toa_radiation' in self.task_config.forcing_variables:
                # Add the local time and TOA radiation to the forcings datasets.
                extended_forcings = add_derived_vars(extended_targets.isel(batch=0).copy(deep=True))
                extended_forcings = TOARadiationFlux(
                    longitude_range="[0, 360]").add_toa_radiation(extended_forcings)
                
            else:
                # Select a single batch, but add the batch dim back later. 
                print(f'Adding forcing variables using add_local_solar_time')
                extended_forcings = add_local_solar_time(
                    extended_targets.isel(batch=0).copy(deep=True))

            extended_forcings = extended_forcings[self.task_config.forcing_variables]
            
            # Expand the batch size since the previous functions will drop it.  
            batch_size = inputs.dims['batch']
            extended_forcings = add_batch_dim(extended_forcings, batch_size)
            
            extended_forcings = extended_forcings.drop_vars('datetime', errors='ignore')
            extended_targets = extended_targets.drop_vars('datetime', errors='ignore')
    
        noise_level = None
        run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
            run_forward.apply, self, self.norm_stats, noise_level)), self))

        # @title Autoregressive rollout (keep the loop in JAX)
        predictions = rollout.chunked_prediction(
            run_forward_jitted,
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=extended_targets,
            forcings=extended_forcings, 
            replace_bdry=replace_bdry,
            diffusion_model=diffusion_model, 
            n_diffusion_steps=n_diffusion_steps,
        )

        return predictions

    # Load the model 
    def load_model(self, path, 
                   **additional_config # For backwards compat.
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
        
        self.domain_size = int(task_config['domain_size'])
            
        data['task_config'] = task_config_dict
   
        self.model_params = data['parameters']
        self.state = {}
        self._init_task_config_run(data['task_config'], **additional_config)
        self._init_model_config_run(data['model_config'], **additional_config)

        if self.norm_stats_path is None: 
            self.norm_stats_path = str(data.get('norm_stats_path', self.norm_stats_path))
        
        ###print(f'{self.norm_stats_path=}')
        self._load_norm_stats(self.norm_stats_path)
    
    def _init_task_config_run(self, data, **additional_config): 

        
        domain_size = additional_config.get('domain_size', None)
        if domain_size is None:
            domain_size = data.get('domain_size', 150)
        
        tiling = additional_config.get('tiling', None)
        if tiling is None:
            tiling = data.get('tiling', None) 
        
        ###print(f'{domain_size=}')
        
        self.task_config = graphcast.TaskConfig(
              input_variables=data['input_variables'],
              target_variables=data['target_variables'],
              forcing_variables=data['forcing_variables'],
              pressure_levels=data['pressure_levels'],
              input_duration=data['input_duration'],
              n_vars_2D = data['n_vars_2D'],
              domain_size = int(domain_size), 
              tiling = tiling, 
              train_lead_times = data.get('train_lead_times', None)
          )
        
        if self.verbose > 2:
            print(f'\n TaskConfig {self.task_config}')
        
    
    def _init_model_config_run(self, data, **additional_config):
        
        k_hop = data.get('k_hop', None)
        if k_hop is None: 
            k_hop = additional_config.get('k_hop', 8) 
        
        use_transformer = data.get('use_transformer', None)
        if use_transformer is None: 
            use_transformer = additional_config.get('use_transformer', False) 
        
        num_attn_heads = data.get('num_attn_heads', None)
        if num_attn_heads is None: 
            num_attn_heads = additional_config.get('num_attn_heads', 4) 
        
        mesh2grid_edge_normalization_factor = data.get('mesh2grid_edge_normalization_factor', None)
        if mesh2grid_edge_normalization_factor is None:
            mesh2grid_edge_normalization_factor = additional_config.get('mesh2grid_edge_normalization_factor', None)
        
        mesh_size = additional_config.get('mesh_size', None)
        if mesh_size is None:
            mesh_size = int(data['mesh_size'])
        
        grid_to_mesh_node_dist = additional_config.get('grid_to_mesh_node_dist', None)
        if grid_to_mesh_node_dist is None:
            grid_to_mesh_node_dist = int(data['grid_to_mesh_node_dist'])
        
        legacy_mesh = additional_config.get('legacy_mesh', False)
        
        self.model_config = graphcast.ModelConfig(
              resolution=int(data['resolution']),
              mesh_size=mesh_size,
              latent_size=int(data['latent_size']),
              gnn_msg_steps=int(data['gnn_msg_steps']),
              hidden_layers=int(data['hidden_layers']),
              grid_to_mesh_node_dist=grid_to_mesh_node_dist,
              loss_weights = data['loss_weights'],
              k_hop = k_hop,
              use_transformer = use_transformer,
              num_attn_heads = num_attn_heads, 
              mesh2grid_edge_normalization_factor = mesh2grid_edge_normalization_factor,
             legacy_mesh = legacy_mesh
        )
        
        if self.verbose > 2:
            print(f'\n ModelConfig: {self.model_config}')
        
    
    def _init_task_config(self, task_config):
        """Initialize the TaskConfig object used in the GraphCast code."""
        self.task_config = task_config
        
        if hasattr(task_config, 'train_lead_times'):
            self.train_lead_times = task_config.train_lead_times
        else:
            self.train_lead_times = None
            
        self.target_vars = task_config.target_variables
        
        if self.verbose > 2:
            print(f'\n TaskConfig {self.task_config}') 
        
        
    def _init_model_config(self, mesh_size, latent_size, 
                           gnn_msg_steps, hidden_layers, grid_to_mesh_node_dist, 
                           loss_weights, k_hop, use_transformer, num_attn_heads, **kwargs
                          ):
        # Weights used in the loss equation.
        if self.loss_weights is None:
            loss_weights = {v : 1.0 for v in self.target_vars}
        else:
            loss_weights = self.loss_weights
            
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
            print(f'\n ModelConfig: {self.model_config}')
        
    
    def _load_norm_stats(self, path):  
        """Load the normalization statistics"""
        mean_by_level = xarray.load_dataset(os.path.join(path, 'mean_by_level.nc'))
        stddev_by_level = xarray.load_dataset(os.path.join(path, 'stddev_by_level.nc'))
        diffs_stddev_by_level = xarray.load_dataset(os.path.join(path, 'diffs_stddev_by_level.nc'))
  
        self.norm_stats = {'mean_by_level': mean_by_level, 
                      'stddev_by_level' : stddev_by_level,
                      'diffs_stddev_by_level' : diffs_stddev_by_level
                     }
        
    def save(self, model_params, state):
        """Checkpoint the model parameters including task and model configs."""
        # Unreplicate model_params if using GPU parallelization for training. 
        # Using the suggested change from https://github.com/google/jax/discussions/15972
        # to limit increasing the memory. 
        if self.num_devices > 1:
            model_params = jax.device_get(jax.tree_map(lambda x: x[0], model_params))
            
        model_data = {'parameters' : model_params, 
                'state' : state,
                'model_config' : self.model_config, 
                'task_config' : self.task_config,
                'norm_stats_path' : self.norm_stats_path 
               }
        
        #print('\n Checkpoint the model parameters...')
        with open(self.out_path, 'wb') as io_byte:
            checkpoint.dump(io_byte, model_data)
             
def ensure_batch_dim(data, dim_name='batch'):
    """
    Ensure that the input xarray DataArray or Dataset has the specified dimension.
    If not, expand dimensions to include it.

    Parameters:
    data (xarray.DataArray or xarray.Dataset): The data to check and possibly modify.
    dim_name (str): The name of the dimension to ensure exists.

    Returns:
    xarray.DataArray or xarray.Dataset: The modified data with the specified dimension.
    """
    if dim_name not in data.dims:
        data = data.expand_dims(dim=dim_name)
    return data 


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

def update_params_with_graphcast(model_params, graphcast_params):
    def update_recursive(model_dict, graphcast_dict):
        for key, value in model_dict.items():
            if isinstance(value, dict) and key in graphcast_dict and isinstance(graphcast_dict[key], dict):
                # If both are dictionaries, recurse
                update_recursive(model_dict[key], graphcast_dict[key])
            elif key in graphcast_dict and model_dict[key].shape == graphcast_dict[key].shape:
                # If both shapes are identical, update the value
                model_dict[key] = graphcast_dict[key]
    
    update_recursive(model_params, graphcast_params)
    return model_params


