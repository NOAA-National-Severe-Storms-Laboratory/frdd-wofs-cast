import os
import sys
import warnings
import argparse
import numpy as np
from os.path import join
from concurrent.futures import ThreadPoolExecutor
import optax
from copy import copy 

# Environment Configuration
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'

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
from wofscast import wofscast_task_config
from wofscast.data_generator import (
    ZarrDataGenerator, 
    SingleZarrDataGenerator, 
    add_local_solar_time, 
    WoFSDataProcessor
)
from wofscast import checkpoint
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
    
    """ usage: stdbuf -oL python -u train_wofscast.py --config train_10min_v178_config_updated.yaml > & log_training & """
    
    args = parse_arguments()
    config_dict = load_configuration(BASE_CONFIG_PATH, args.config)

    # Extract configuration values
    fine_tune = config_dict.get('fine_tune', False)
    seed = config_dict.get('seed', 42)
    out_path = config_dict.get('model_save_path', 'wofscast_default.npz')
    task_config = getattr(wofscast_task_config, config_dict.get('task_config'))
    graphcast_pretrain = config_dict.get('graphcast_pretrain', False)
    batch_size = config_dict.get('batch_size', 24)
    loss_weights = config_dict.get('loss_weights', {})
    norm_stats_path = config_dict.get('norm_stats_path', None)
    
    do_add_solar_time = config_dict.get('add_local_solar_time', True) 
    decode_times = config_dict.get('decode_times', False)
    
    if fine_tune: 
        # Determine whether the fine tuning includes loading data for 
        # more than one time step. 
        rollout = config_dict.get('rollout' , False)
        if rollout:
            target_lead_times = [slice(t[0], t[1]) for t in config_dict['target_lead_times']]
        else:
            target_lead_times = None
            
        # Fine-tuning settings
        n_steps = config_dict['n_fine_tune_steps']
        base_paths = config_dict['fine_tune_data_paths']
        
        # Avoid overwriting existing checkpoint
        model_path = copy(out_path)
        out_path = model_path.replace('.npz', '_fine_tune.npz') 
        
        new_save_path = config_dict.get('new_save_path', None)
        if new_save_path: 
            out_path = os.path.join(new_save_path, os.path.basename(out_path))
            
        print('Performing fine tuning...')
        print(f'Saving model to {out_path}..')
        
        scheduler = optax.constant_schedule(float(config_dict['fine_tune_learning_rate']))
        
        trainer = WoFSCastModel(
            learning_rate_scheduler=scheduler, 
            checkpoint=True, 
            norm_stats_path=norm_stats_path, # Should come from the saved file! 
            out_path=out_path,
            checkpoint_interval=config_dict['checkpoint_interval'],  
            verbose=1, 
            n_steps = n_steps, 
            loss_weights=loss_weights,
            parallel=True
        )
        
        # Load the model
        print(f'Loading {model_path}...')
        
        full_domain = config_dict.get('full_domain', False)
        
        if full_domain:
            # At the moment, we have a model trained on 150 x 150 grid point patches. 
            # Thus, for the full domain, if set tiling = (2,2), it will create a 2 x 2 
            # quilt of the original mesh to cover the full 300 x 300 WoFS domain.
            trainer.load_model(model_path, **{'tiling' : (2,2)})
        else:
            trainer.load_model(model_path)
            
        model_params, state = trainer.model_params, trainer.state 
        
    else:
        base_paths = config_dict['data_paths'] 
        
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
        
        model_params, state = None, {}
        target_lead_times = None  # Defaults to TaskConfig's target lead times
        
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
            parallel=True,
            graphcast_pretrain=graphcast_pretrain
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
    
    generator = ZarrDataGenerator(
        paths=paths, 
        task_config=task_config, 
        target_lead_times=target_lead_times,
        batch_size=batch_size, 
        num_devices=2, 
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
    
    trainer.fit_generator(generator, model_params=model_params, state=state)
