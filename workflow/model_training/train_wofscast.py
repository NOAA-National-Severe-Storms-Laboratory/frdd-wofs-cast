import os
import sys
import warnings
import argparse
import numpy as np
from os.path import join
from concurrent.futures import ThreadPoolExecutor
import optax, jax
from copy import copy 


#Before profiling memory, ensure your computations are stable (no NaNs or infs), as these can inflate memory usage.

#jax.config.update("jax_debug_nans", True)  # Debug NaNs
#jax.config.update("jax_debug_infs", True)  # Debug Infs


# Environment Configuration
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.90'

# Optional: Uncomment to set XLA GPU performance flags
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_triton_gemm_any=True '
#     '--xla_gpu_enable_async_collectives=true '
#     '--xla_gpu_enable_latency_hiding_scheduler=true '
#     '--xla_gpu_enable_highest_priority_async_stream=true '
# )

# Suppress specific RuntimeWarnings
warnings.filterwarnings(
    "ignore", 
    category=RuntimeWarning, 
    message="os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock."
)

# Add project directory to the system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

# WoFSCast imports
from wofscast.model import WoFSCastModel
from wofscast.graphcast_lam import TaskConfig
#from wofscast import wofscast_task_config
from wofscast.data_generator import (
    ZarrDataGenerator, 
    DataAssimDataLoader, 
    SingleZarrDataGenerator, 
    add_local_solar_time, 
    WoFSDataProcessor
)
from wofscast import checkpoint
from wofscast import losses
from wofscast.utils import get_random_subset, truncate_to_chunk_size
from wofscast.common.helpers import parse_arguments, load_configuration 

# Configurations
BASE_CONFIG_PATH = 'training_configs'

def get_files_for_year(year, base_path):
    """Get all zarr files within a directory for a given year."""
    year_path = join(base_path, year)
    with os.scandir(year_path) as it:
        return [join(year_path, entry.name) for entry in it if entry.is_dir() and entry.name.endswith('.zarr')] 

def get_paths(base_paths, years=['2019', '2020']):
    """Get file paths from multiple base paths and years."""
    paths = []

    # Use a thread pool to retrieve files in parallel
    with ThreadPoolExecutor() as executor:
        future_to_year = {
            executor.submit(get_files_for_year, year, base_path): (year, base_path) 
            for base_path in base_paths for year in years
        }
        for future in future_to_year:
            paths.extend(future.result())  # Collect results from futures

    return paths

if __name__ == '__main__':
    """Main script execution."""
    
    """ usage: stdbuf -oL python -u train_wofscast.py --config train_10min_v178_15min_offset.yaml > & logs/log_train_15min & """
    
    """ usage: stdbuf -oL python -u train_wofscast.py --config train_da.yaml > & logs/log_train_da & """
    
    """ usage: stdbuf -oL python -u train_wofscast.py --config finetune_full_domain_config.yaml > & logs/log_finetune_full & """
    
    """ usage: stdbuf -oL python -u train_wofscast.py --config train_10min_v178_90min_offset.yaml > & logs/log_finer_mesh & """
    
    
    
    args = parse_arguments()
    config_dict = load_configuration(BASE_CONFIG_PATH, args.config)

    # Extract configuration values
    fine_tune = config_dict.get('fine_tune', False)
    seed = config_dict.get('seed', 42)
    out_path = config_dict.get('model_save_path', 'wofscast_default.npz')
    ###task_config = getattr(wofscast_task_config, config_dict.get('task_config'))
    graphcast_pretrain = config_dict.get('graphcast_pretrain', False)
    batch_size = config_dict.get('batch_size', 24)
    loss_weights = config_dict.get('loss_weights', {})
    norm_stats_path = config_dict.get('norm_stats_path', None)
    
    do_add_solar_time = config_dict.get('add_local_solar_time', True) 
    decode_times = config_dict.get('decode_times', False)
    
    parallel = config_dict.get('parallel', True)
    use_wandb = config_dict.get('use_wandb', True)
    legacy_mesh = config_dict.get('legacy_mesh', True) 
    generator_name = config_dict.get('generator_name', 'ZarrDataGenerator') 
    
    lr_scheduler = config_dict.get('lr_scheduler', 'cosine_decay') 
    
    print(f'{lr_scheduler=}')
    
    if lr_scheduler == 'constant':
        scheduler = optax.constant_schedule(float(config_dict['peak_learning_rate']))
        n_steps = config_dict.get('n_fine_tune_steps', 10000) 
        
    elif lr_scheduler == 'cosine_decay':
        # General training settings
        warmup_steps = config_dict['warmup_steps']
        decay_steps = config_dict['decay_steps']
        n_steps = warmup_steps + decay_steps
        
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=float(config_dict['peak_learning_rate']),
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )

    if fine_tune: 
        # Determine whether the fine tuning includes loading data for 
        # more than one time step. 
        rollout = config_dict.get('rollout' , False)
        
        if rollout:
            target_lead_times = [slice(t[0], t[1]) for t in config_dict['target_lead_times']][0]
        else:
            target_lead_times = None
        
        # Fine-tuning settings
        n_steps = config_dict['n_fine_tune_steps']
        base_paths = config_dict['fine_tune_data_paths']
        new_save_path = config_dict['out_save_path']
        weights_path = config_dict['model_save_path'] 
        
        # Avoid overwriting existing checkpoint
        print('Performing fine tuning...')
        print(f'Using model weights from {weights_path}')
        print(f'Saving model to {new_save_path}..')

        trainer = WoFSCastModel(
            learning_rate_scheduler=scheduler, 
            checkpoint=True, 
            norm_stats_path=norm_stats_path, # Should come from the saved file! 
            out_path=new_save_path,
            checkpoint_interval=config_dict['checkpoint_interval'],  
            verbose=1, 
            n_steps = n_steps, 
            loss_weights=loss_weights,
            parallel=True, 
            use_wandb=use_wandb,
            legacy_mesh = legacy_mesh,
            wandb_config = config_dict, 
        )
        
        # Load the model
        print(f'Loading {weights_path}...')
        
        full_domain = config_dict.get('full_domain', False)
        
        if full_domain:
            # At the moment, we have a model trained on 150 x 150 grid point patches. 
            # Thus, for the full domain, if set tiling = (2,2), it will create a 2 x 2 
            # quilt of the original mesh to cover the full 300 x 300 WoFS domain.
            trainer.load_model(weights_path, **{'tiling' : 150, 'domain_size' : 300,
                                              'legacy_mesh' : legacy_mesh})
        else:
            trainer.load_model(weights_path)
            
        model_params, state = trainer.model_params, trainer.state 
        task_config = trainer.task_config
        
        loss_class = getattr(losses, config_dict.get('loss_metric', 'MSE'))
        lat_rng = config_dict.get('lat_rng', None)
        if lat_rng:
            lat_rng = slice(lat_rng[0], lat_rng[1])
        lon_rng = config_dict.get('lon_rng', None)
        if lon_rng:
            lon_rng = slice(lon_rng[0], lon_rng[1])
        
        loss_callable = loss_class(lat_rng = lat_rng, 
                                   lon_rng = lon_rng, 
                                   add_latitude_weight = config_dict.get('add_latitude_weight', False),
                                   add_level_weight = config_dict.get('add_level_weight',  False), 
                                  ) 
        # TaskConfig is "frozen", so it needs to recreated.
        variables_2d = config_dict.get('variables_2D', ['']) 
        task_config = TaskConfig(
            input_variables = task_config.input_variables, 
            target_variables = task_config.target_variables, 
            forcing_variables = task_config.forcing_variables, 
            pressure_levels = task_config.pressure_levels,
            input_duration = task_config.input_duration,
            n_vars_2D = len(variables_2d),
            domain_size = config_dict.get('domain_size', 150),
            tiling = config_dict.get('tile_size', None), 
            train_lead_times = target_lead_times,
            loss_callable = loss_callable
        ) 
        
        print(f'{target_lead_times=}')
        
        # Reset the task_config. 
        trainer.task_config = task_config
              
        
    else:
        base_paths = config_dict['data_paths'] 
        
        model_params, state = None, {}
        target_lead_times = None  # Defaults to TaskConfig's target lead times
        
        # Build the task config. 
        variables_2d = config_dict.get('variables_2D', ['']) 
        variables_3d = config_dict.get('variables_3D', ['']) 
        static_variables = config_dict.get('static_variables', ['']) 
    
        target_variables = variables_3d + variables_2d
        input_variables = target_variables + static_variables 
    
        loss_class = getattr(losses, config_dict.get('loss_metric', 'MSE'))
        lat_rng = config_dict.get('lat_rng', None)
        if lat_rng:
            lat_rng = slice(lat_rng[0], lat_rng[1])
        lon_rng = config_dict.get('lon_rng', None)
        if lon_rng:
            lon_rng = slice(lon_rng[0], lon_rng[1])
        
        loss_callable = loss_class(lat_rng = lat_rng, 
                                   lon_rng = lon_rng, 
                                   add_latitude_weight = config_dict.get('add_latitude_weight', False),
                                   add_level_weight = config_dict.get('add_level_weight',  False), 
                                  ) 
        
        task_config = TaskConfig(
            input_variables = input_variables, 
            target_variables = target_variables, 
            forcing_variables = config_dict.get('forcing_variables', None), 
            pressure_levels = config_dict.get('pressure_levels', None),
            input_duration = config_dict.get('input_duration', None),
            n_vars_2D = len(variables_2d),
            domain_size = config_dict.get('domain_size', 150),
            tiling = config_dict.get('tiling', None), 
            train_lead_times = config_dict.get('train_lead_times', None),
            loss_callable = loss_callable
        ) 
    
        print(f"""
            TaskConfig:
                input_variables   = {input_variables}
                target_variables  = {target_variables}
                forcing_variables = {config_dict.get('forcing_variables', None)}
                pressure_levels   = {config_dict.get('pressure_levels', None)}
                input_duration    = {config_dict.get('input_duration', None)}
                n_vars_2D         = {len(variables_2d)}
                domain_size       = {config_dict.get('domain_size', 150)}
                tiling            = {config_dict.get('tiling', None)}
                train_lead_times  = {config_dict.get('train_lead_times', None)}
                loss_callable     = {loss_callable}
                """
        )


        trainer = WoFSCastModel(
            task_config=task_config, 
            mesh_size=config_dict['mesh_size'], 
            latent_size=config_dict['latent_size'], 
            gnn_msg_steps=config_dict['gnn_msg_steps'], 
            hidden_layers=config_dict['hidden_layers'], 
            noise_level=config_dict.get('noise_level', None), 
            grid_to_mesh_node_dist=config_dict['grid_to_mesh_node_dist'],  
            use_transformer=config_dict['use_transformer'], 
            k_hop=config_dict['k_hop'],
            num_attn_heads=config_dict['num_attn_heads'], 
            n_steps=n_steps, 
            learning_rate_scheduler=scheduler, 
            checkpoint=True, 
            norm_stats_path=norm_stats_path,
            out_path=out_path,
            checkpoint_interval=config_dict['checkpoint_interval'],
            verbose=1, 
            loss_weights=loss_weights,
            parallel=parallel,
            graphcast_pretrain=graphcast_pretrain,
            use_wandb=use_wandb,
            legacy_mesh = legacy_mesh,
            wandb_config = config_dict, 
        )
    
    paths = get_paths(base_paths, years=['2019', '2020'])

    # Shuffle paths
    rs = np.random.RandomState(seed)
    rs.shuffle(paths)
    
    print(f'Number of Paths: {len(paths)}')
    
    preprocess_fn = None
    if do_add_solar_time:
        def preprocess_fn(ds):
            # Example preprocessing: Add local solar time
            ds = add_local_solar_time(ds)
            return ds
    
    to_legacy_lons = config_dict.get('to_legacy_lons', False)
    
    print(f'{to_legacy_lons=}')
    
    if to_legacy_lons:
        def preprocess_fn(ds):
            # Convert longs of ~260 back to 70-80
            # Seems to work ¯\_(ツ)_/¯
            ds = ds.assign_coords(lon =ds.lon-180)
            return ds

    pre_select_times = config_dict.get('pre_select_times', False)
    
    if pre_select_times:     
        def preprocess_fn(ds):
            ds = ds.isel(time=[1,2,3])
            return ds 
    
    if generator_name == 'ZarrDataGenerator': 
        print(f'Line 333 {target_lead_times=}')
        generator = ZarrDataGenerator(
            paths=paths, 
            task_config=task_config, 
            target_lead_times=target_lead_times,
            batch_size=batch_size, 
            num_devices=jax.local_device_count() if parallel else 1, 
            preprocess_fn=preprocess_fn,
            prefetch_size=2,
            random_seed=seed, 
            decode_times=decode_times,
        )
    elif generator_name == 'DataAssimDataLoader':
        generator = DataAssimDataLoader(
            known_variables = config_dict.get('known_variables', ['COMPOSITE_REFL_10CM']),
            unknown_variables = config_dict.get('unknown_variables', ['COMPOSITE_REFL_10CM']),
            skewed_variables = config_dict.get('skewed_variables', ['COMPOSITE_REFL_10CM']),
            gauss_filter_size = config_dict.get('gauss_filter_size', 10.0),
            paths=paths, 
            task_config=task_config, 
            target_lead_times=target_lead_times,
            batch_size=batch_size, 
            num_devices=jax.local_device_count() if parallel else 1, 
            preprocess_fn=preprocess_fn,
            prefetch_size=2,
            random_seed=seed, 
            decode_times=decode_times,
        )
        

    # Optional: Uncomment to use SingleZarrDataGenerator instead
    # zarr_path = '/work/mflora/wofs-cast-data/datasets_5min/training_dataset/wofs_data_2019-2020.zarr'
    # generator = SingleZarrDataGenerator(
    #     zarr_path=zarr_path, 
    #     task_config=task_config, 
    #     target_lead_times=None,
    #     batch_size=batch_size, 
    #     num_devices=2, 
    #     prefetch_size=2,
    #     random_seed=seed, 
    # )
    
    results = trainer.fit_generator(generator, model_params=model_params, state=state)
