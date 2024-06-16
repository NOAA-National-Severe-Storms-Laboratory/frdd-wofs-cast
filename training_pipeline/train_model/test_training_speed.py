# Disable XLA preallocation
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
#os.environ['JAX_TRACEBACK_FILTERING'] = 'on'

import os


os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
    #'--xla_gpu_all_reduce_combine_threshold_bytes=33554432 '
    #'--xla_gpu_simplify_all_fp_conversions '
    #'--xla_gpu_enable_async_reduce_scatter=true ' 
    #'--xla_gpu_graph_level=0 '
    #'--xla_gpu_enable_async_all_reduce=true '
)

os.environ.update({
  "NCCL_LL128_BUFFSIZE": "-2",
  "NCCL_LL_BUFFSIZE": "-2",
   "NCCL_PROTO": "SIMPLE,LL,LL128",
 })

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))


from wofscast.data_generator import (add_local_solar_time, 
                                     to_static_vars, 
                                     dataset_to_input,
                                     load_chunk,
                                     ZarrDataGenerator
                                    )
from wofscast import data_utils
from wofscast.wofscast_task_config import DBZ_TASK_CONFIG, WOFS_TASK_CONFIG
import wofscast.graphcast_lam as graphcast 
from wofscast import (data_utils, 
                      casting, 
                      normalization,
                      autoregressive,
                      xarray_tree,
                      xarray_jax
                     )
from wofscast.utils import count_total_parameters


from os.path import join
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import haiku as hk
import jax
import functools

import dask 
import optax 
import jax.numpy as jnp
from jax import lax

import xarray as xr
import numpy as np

import time 
from tqdm import tqdm 


jax.config.update("jax_compilation_cache_dir", ".")

# Make sure NVTX annotations include full Python stack traces
#jax.config.update("jax_traceback_in_locations_limit", -1)


def replicate_for_devices(params, num_devices=None):
    if num_devices is None:
        num_devices = jax.local_device_count()
    return jax.device_put_replicated(params, jax.local_devices()) if num_devices > 1 else params


def get_files_for_year(year):
    year_path = join(base_path, year)
    with os.scandir(year_path) as it:
        return [join(year_path, entry.name) for entry in it if entry.is_dir() and entry.name.endswith('.zarr')]

def construct_wrapped_graphcast(model_config: graphcast.ModelConfig, 
                                task_config: graphcast.TaskConfig,
                                #norm_stats: dict
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
      diffs_stddev_by_level=diffs_stddev_by_level,#norm_stats['diffs_stddev_by_level'],
      mean_by_level=mean_by_level,#norm_stats['mean_by_level'],
      stddev_by_level=stddev_by_level,#norm_stats['stddev_by_level']
    )

    # Wraps everything so the one-step model can produce trajectories.
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    
    return predictor

# Function for deployment. Used to make predictions on new data and rollout. 
@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)

@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
    return functools.partial(
      fn, model_config=model_config, 
        task_config=task_config, 
      )

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
    return lambda **kw: fn(**kw)[0]

# Always pass params and state, so the usage below are simpler
def with_params(fn):
     return functools.partial(fn, params=params, state=state, optimizer=optimizer)
      
def train_step(params, 
               state, 
               opt_state,
               inputs, 
               targets, 
               forcings, 
               model_config, 
               task_config):

    
    def compute_loss(params, state, inputs, targets, forcings):
        (loss, diagnostics), next_state = loss_fn.apply(params, state, 
                                                        jax.random.PRNGKey(0), 
                                                        model_config, 
                                                        task_config, 
                                                        inputs, targets, forcings)
        return loss, (diagnostics, next_state)
    
    # Compute gradients and auxiliary outputs
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(compute_loss, has_aux=True)(params, state, 
                                                                                              inputs, targets, 
                                                                                              forcings)

    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    

    return new_params, opt_state, loss, diagnostics
            

          
def train_step_parallel(params, 
               state, 
               opt_state,
               inputs, 
               targets, 
               forcings, 
               model_config, 
               task_config, 
              ):
        
    def compute_loss(params, state, inputs, targets, forcings):
        (loss, diagnostics), next_state = loss_fn.apply(params, state, 
                                                        jax.random.PRNGKey(0), 
                                                        model_config, 
                                                        task_config, 
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
    
    return new_params, opt_state, loss, diagnostics


if __name__ == "__main__":    
    model_config = graphcast.ModelConfig(
              resolution=0,
              mesh_size=5,
              latent_size=512,
              gnn_msg_steps=16,
              hidden_layers=1,
              grid_to_mesh_node_dist=5, 
              loss_weights = None,
              k_hop = 8,
              use_transformer = False,
              num_attn_heads = 4
        )

    
    ### LOAD DATA ####
    
    base_path = '/work/mflora/wofs-cast-data/datasets_zarr'
    years = ['2019']
    num_devices = 2
    
    task_config = WOFS_TASK_CONFIG
    
    with ThreadPoolExecutor() as executor:
        paths = []
        for files in executor.map(get_files_for_year, years):
            paths.extend(files)
    
    start_time = time.time()        
    generator = ZarrDataGenerator(paths, task_config, 
                              batch_size=num_devices, 
                              num_devices=num_devices, 
                              preprocess_fn = add_local_solar_time
                             )
    
    inputs, targets, forcings = generator.generate()

    end_time = time.time()  # Start time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Elapsed time for loading 1 batch with generator: {elapsed_time:.6f} seconds")
    
    path = '/work/mflora/wofs-cast-data/normalization_stats'

    mean_by_level = xr.load_dataset(os.path.join(path, 'mean_by_level.nc'))
    stddev_by_level = xr.load_dataset(os.path.join(path, 'stddev_by_level.nc'))
    diffs_stddev_by_level = xr.load_dataset(os.path.join(path, 'diffs_stddev_by_level.nc'))
    
    ### TRAIN STEP ####
    start_time = time.time()  
    
    init_jitted = jax.jit(with_configs(run_forward.init))
    
    if num_devices==1:
        _inputs = inputs.isel(batch=[0])
        _targets = targets.isel(batch=[0])
        _forcings = forcings.isel(batch=[0])
        
    else:
        _inputs = inputs.isel(devices=0, batch=[0])
        _targets = targets.isel(devices=0, batch=[0])
        _forcings = forcings.isel(devices=0, batch=[0])
        
    params, state = init_jitted(
      rng=jax.random.PRNGKey(0),
      inputs=_inputs,
      targets_template=_targets,
      forcings=_forcings
    )
    
    end_time = time.time()  # Start time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Elapsed time for initializing model params and state: {elapsed_time:.6f} seconds")
    
    num = count_total_parameters(params)
    print(f'\n Num of Model Parameters: {num}\n')
    
    # setup optimiser
    optimizer = optax.adam(1e-3, b1=0.9, b2=0.999, eps=1e-8)
    opt_state = optimizer.init(params)
    
    if num_devices == 1:
        train_step_jitted = jax.jit(with_configs(train_step))
    else:
        train_step_jitted = xarray_jax.pmap(with_configs(train_step_parallel), dim='devices', axis_name='devices') 
    
    # For GPU parallel processing, replicate the model parameters, model state, 
    # and the optimizer state on each device.
    model_params_replicated = replicate_for_devices(params, num_devices)
    state_replicated= replicate_for_devices(state, num_devices)
    opt_state_replicated = replicate_for_devices(opt_state, num_devices)
    
    batch_size = 24
    n_steps = 10
        
    generator = ZarrDataGenerator(paths, 
                              task_config, 
                              target_lead_times=None,
                              batch_size=batch_size, 
                              num_devices=num_devices, 
                              preprocess_fn=add_local_solar_time,
                              prefetch_size=3
                             )
    
    # use the pre-allocated space.
    inputs, targets, forcings = generator.generate()
    
    # Run the code once to jit the train_step_jitted
    start_time = time.time()
    params, opt_state, loss, diagnostics = train_step_jitted(
                   model_params_replicated, 
                   state_replicated,
                   opt_state_replicated,
                   inputs, 
                   targets, 
                   forcings, 
               )
    run_time = time.time() - start_time
    print(f'Compiling Time: {run_time:.2f}')
    
    load_times = [] 
    step_times = [] 
    run_start_time = time.time()
    for step in range(n_steps):
        start_time = time.time()
        
        # Generate new data and update the pre-allocated space.
        # From the graph-ufs code: 
        # The purpose of the following code is best described as confusing
        # the jix.jat cache system. We start from a deepcopy of slice 0 where the jitting
        # is carried out, and sneakly update its values. If you use xarray update/copy etc
        # the cache system somehow notices, and either becomes slow or messes up the result
        # Copying variable values individually avoids both, fast and produces same results as before
        
        new_inputs, new_targets, new_forcings = generator.generate()
        load_times.append(time.time() - start_time) 
        
        for var_name, var in new_inputs.data_vars.items():
            inputs[var_name] = inputs[var_name].copy(deep=False, data=var.values)
        for var_name, var in new_targets.data_vars.items():
            targets[var_name] = targets[var_name].copy(deep=False, data=var.values)
        for var_name, var in new_forcings.data_vars.items():
            forcings[var_name] = forcings[var_name].copy(deep=False, data=var.values)
        
        #print(f'{inputs=}')
        #print(f'{targets=}')
        #print(f'{forcings=}')
            
        start_time = time.time()
        params, opt_state, loss, diagnostics = train_step_jitted(
                   model_params_replicated, 
                   state_replicated,
                   opt_state_replicated,
                   inputs, 
                   targets, 
                   forcings, 
               )
        step_times.append(time.time() - start_time)
        
      
    run_time = time.time() - run_start_time
    print(f'Load times : {np.mean(load_times[1:])}')
    print(f'Step times : {np.mean(step_times[1:])}')
    print(f'Total Run Time: {run_time:.4f}')
 
   