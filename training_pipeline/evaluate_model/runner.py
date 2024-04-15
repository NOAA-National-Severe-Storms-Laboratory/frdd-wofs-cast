import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

# @title Imports
import dataclasses
import datetime
import functools
import math
import re
from typing import Optional
from glob import glob

import cartopy.crs as ccrs
#from google.cloud import storage
from wofscast import autoregressive_lam as autoregressive
from wofscast import casting
from wofscast import checkpoint
from wofscast import data_utils
from wofscast import my_graphcast as graphcast
from wofscast import normalization
from wofscast import rollout
from wofscast import xarray_jax
from wofscast import xarray_tree
from IPython.display import HTML
import ipywidgets as widgets
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
import xarray #as xr
from wofscast.data_generator import to_static_vars, add_local_solar_time

from wofscast.utils import count_total_parameters, save_model_params, load_model_params 


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
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=False)
    
    return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, norm_stats, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config, norm_stats)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

# Always pass params and state, so the usage below are simpler
def with_params(fn, model_obj):
  return functools.partial(fn, params=model_obj.model_params, state=model_obj.state)

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn, model_obj, norm_stats):
    return functools.partial(
      fn, model_config=model_obj.model_config, task_config=model_obj.task_config, norm_stats=norm_stats)

class WoFSCastRunner:
    def __init__(self, model_path, 
                 total_forward_time_minutes = '60min', 
                 timestep_minutes = '10min',
                 norm_stats_path = '/work/mflora/wofs-cast-data/normalization_stats', 
                 domain_size = None
                ):
        
        model_data = self.load_model(model_path)
        self.timestep_minutes = timestep_minutes
        
        #TODO: use the extend_targets_template in rollout.py for real-time rollouts (without a known target field).
        # Can use to compute the forcing. 
        
        self.n_timesteps = int(total_forward_time_minutes.split('min')[0]) // int(timestep_minutes.split('min')[0])
        self.domain_size = domain_size
    
        self.train_lead_times = slice(timestep_minutes, total_forward_time_minutes)
    
        self.model_params = model_data['parameters']
        self.state = {}
        self._init_task_config(model_data['task_config'])
        self._init_model_config(model_data['model_config'])

        self._load_norm_stats(norm_stats_path)
    
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
    
    def _transpose(self, ds, dims):
        return ds.transpose(*dims, missing_dims='ignore')    
    
    def expand_time_dim(self, ds, forcings=False):
        # Repeat the time dimension n_timesteps number of times 
        # Use isel to select the slice and tile to repeat it
        repeated_slices = [ds.isel(time=0)] * self.n_timesteps
    
        # Concatenate the repeated slices along the 'time' dimension
        expanded_ds = xarray.concat(repeated_slices, dim='time')

        # Add a new datetime for the forcing variables. 
        # Generate the datetime range. 
        if forcings: 
            start_time = ds.datetime[0].values
            time_range = pd.date_range(start=start_time, periods=self.n_timesteps, freq=self.timestep_minutes)
            expanded_ds['time'] = time_range
            expanded_ds = expanded_ds.assign_coords(datetime=time_range)

            # Convert 'time' dimension to timedeltas from the first time point
            time_deltas = (expanded_ds['time'] - expanded_ds['time'][0]).astype('timedelta64[ns]')
            expanded_ds['time'] = time_deltas

        expanded_ds = self._transpose(expanded_ds, ['batch', 'time', 'lat', 'lon', 'level', 'datetime'])
    
        return expanded_ds

    
    # Load the model 
    def load_model(self, path):
        with open(path, 'rb') as f:
            data = checkpoint.load(f, dict)
    
        #Unravel the task config. 
        _TASK_CONFIG_KEYS = list(vars(graphcast.TaskConfig)['__dataclass_fields__'].keys())
    
        task_config = data['task_config']
    
        task_config_dict = {}
        for key in _TASK_CONFIG_KEYS: 
            if isinstance(task_config[key], dict):
                # unravel
                if key == 'pressure_levels':
                    task_config_dict[key] = [int(item) for _, item in task_config[key].items()]
                else:
                    task_config_dict[key] = [str(item) for _, item in task_config[key].items()]
            elif key == 'input_duration':
                task_config_dict[key] = str(task_config[key])
            else:
                task_config_dict[key] = task_config[key]
    
    
        data['task_config'] = task_config_dict
    
        return data 
    
    def _init_task_config(self, data): 

        domain_size =self.domain_size if self.domain_size else data['domain_size']
        
        self.task_config = graphcast.TaskConfig(
              input_variables=data['input_variables'],
              target_variables=data['target_variables'],
              forcing_variables=data['forcing_variables'],
              pressure_levels=data['pressure_levels'],
              input_duration=data['input_duration'],
              n_vars_2D = data['n_vars_2D'],
              domain_size = domain_size
          )
    
    def _init_model_config(self, data):
        self.model_config = graphcast.ModelConfig(
              resolution=int(data['resolution']),
              mesh_size=int(data['mesh_size']),
              latent_size=int(data['latent_size']),
              gnn_msg_steps=int(data['gnn_msg_steps']),
              hidden_layers=int(data['hidden_layers']),
              grid_to_mesh_node_dist=int(data['grid_to_mesh_node_dist']),
              loss_weights = data['loss_weights'],
        )
    
    def _load_norm_stats(self, path):     
        mean_by_level = xarray.load_dataset(os.path.join(path, 'mean_by_level.nc'))
        stddev_by_level = xarray.load_dataset(os.path.join(path, 'stddev_by_level.nc'))
        diffs_stddev_by_level = xarray.load_dataset(os.path.join(path, 'diffs_stddev_by_level.nc'))

        self.norm_stats = {'mean_by_level': mean_by_level, 
                      'stddev_by_level' : stddev_by_level,
                      'diffs_stddev_by_level' : diffs_stddev_by_level
                     }




