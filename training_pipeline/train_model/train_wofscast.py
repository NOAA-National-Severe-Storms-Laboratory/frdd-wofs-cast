# Disable XLA preallocation
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'

# XLA FLAGS set for GPU performance (https://jax.readthedocs.io/en/latest/gpu_performance_tips.html)
"""
os.environ['XLA_FLAGS'] = (
    #'--xla_gpu_enable_triton_softmax_fusion=true ' # Caused issues for the transformer layer. 
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)
"""


# WoFSCast 
import warnings
# Suppress the specific RuntimeWarning about os.fork(),  multithreaded code, and JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.")

import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from wofscast.model import WoFSCastModel
from wofscast.wofscast_task_config import WOFS_TASK_CONFIG, DBZ_TASK_CONFIG
from wofscast.data_generator import ZarrDataGenerator, add_local_solar_time
from wofscast import checkpoint
from wofscast.utils import get_random_subset,  truncate_to_chunk_size

import optax 

import os
from os.path import join
from concurrent.futures import ThreadPoolExecutor

# Using the Weights & Biases package: 
# Create an account : https://docs.wandb.ai/quickstart

def get_files_for_year(year):
    """Get all zarr files within a directory."""
    year_path = join(base_path, year)
    with os.scandir(year_path) as it:
        return [join(year_path, entry.name) for entry in it if entry.is_dir() and entry.name.endswith('.zarr')] 


# >>> python -m wandb login
if __name__ == '__main__':
    """ usage: stdbuf -oL python -u train_wofscast.py > & log_training & """
    
    # USER SET ARGS------------------------------------------------------------------------
    # Whether to initialize the model with existing model weights 
    # TaskConfig and ModelConfig are re-built using the checkpoint
    # provided. 
    fine_tune = False
    
    # Where the model weights are stored
    out_path = '/work/mflora/wofs-cast-data/model/wofscast_reproducibility_test.npz'
    
    # The task config contains details like the input variables, 
    # target variables, time step, etc.
    task_config = WOFS_TASK_CONFIG
    
    # Whether to use the 36.7M parameter GraphCast model weights 
    # Must set parameters identical to those paper 
    graphcast_pretrain = False
    
    # Number of samples processed during a single gradient descent step
    # If using multiple GPUs, batch_size / n_gpus samples are sent 
    # to each GPU. 
    batch_size = 32
    
    loss_weights = {
                    # Any variables not specified here are weighted as 1.0.
                    'U' : 1.0, 
                    'V': 1.0, 
                    'W': 1.0, 
                    'T': 1.0, 
                    'GEOPOT': 1.0, 
                    'QVAPOR': 1.0,
                    'T2' : 1.0, 
                    'COMPOSITE_REFL_10CM' : 1.0, 
                    #'UP_HELI_MAX' : 0.1,
                    'RAIN_AMOUNT' : 1.0,
                    }
    
    if fine_tune: 
        # For fine tuning, training is perform autoregressively on 
        # multi time step rollout. The lead time ranges to be evaluated 
        # are given below. Based on n_steps, each time range is 
        # evenly trained on. 
        n_steps = 10
        target_lead_times = [slice('10min', '20min'), slice('10min', '30min'), slice('10min', '40min')]
        
        # Location of the datasets with longer lead times. 
        base_path = '/work/mflora/wofs-cast-data/datasets_2hr_zarr'
   
        model_path = out_path.copy() 
    
        # Do not want to replace the existing checkpoint! 
        out_path = model_path.replace('.npz', '_fine_tune.npz') 
        
        # For fine tuning, we adopt the constant, but small learning rate. 
        scheduler = optax.constant_schedule(3e-6)
        
        # Build the TaskConfig and ModelConfig inputs. 
        trainer = WoFSCastModel(learning_rate_scheduler = scheduler, 
        
                 checkpoint=True, # Save the model periodically
            
                 norm_stats_path = '/work/mflora/wofs-cast-data/full_normalization_stats',
        
                 # Path where the model is saved. The file name (os.path.basename)
                 # is the named used for the Weights & Biases project. 
                 out_path = out_path,
                 
                 checkpoint_interval = 1, # How often to save the weights (in terms of epochs) 
                 verbose = 1, # Set to 3 to get all possible printouts
                 loss_weights = loss_weights,
                 parallel = True)    
        
        # Load the model, which will also load the TaskConfig and ModelConfig.
        trainer.load_model(model_path)
        model_params, state = trainer.model_params, trainer.state 
        
        
    else:
        # For general training, we adopt the linear increase in learning rate 
        # during a 'warm-up' period followed by a cosine decay in learning rate
        
        warmup_steps = 250
        decay_steps = 10000
        n_steps = warmup_steps + decay_steps
        
        scheduler = optax.warmup_cosine_decay_schedule(
              init_value=0,
              peak_value=1e-3,
              warmup_steps=warmup_steps,
              decay_steps=decay_steps,
              end_value=0.0,
            )
        
        model_params, state = None, {}
        target_lead_times= None # Defaults to target lead times in the TaskConfig.
        # Location of the dataset. 
        base_path = '/work/mflora/wofs-cast-data/datasets_zarr'
        
        trainer = WoFSCastModel(
                 task_config = task_config, 
                 mesh_size=5, # Number of Mesh refinements or more higher resolution layers. 
                 
                 # Parameters for the MLPs-------------------
                 latent_size=128, 
                 gnn_msg_steps=8, # Increasing this allows for connecting information from farther away. 
                 hidden_layers=1, 
                 grid_to_mesh_node_dist=5,  # Fraction of the maximum distance between mesh nodes on the 
                                             # finest mesh level. @ level 5, max distance ~ 4.5 km, 
                                             # so connecting to those grid points with 1-2 km 
        
                                             # OR integer as the distance 
                 #--------------------------------------------
                 # Parameters if using a transformer layer for processor (mesh)
                 # the transformer also relies on the latent_size arg above.
                 use_transformer = False, 
                 k_hop = 8,
                 num_attn_heads  = 4, 
        
                 n_steps = n_steps, 
                 learning_rate_scheduler = scheduler, 
        
                 checkpoint=True, # Save the model periodically
            
                 norm_stats_path = '/work/mflora/wofs-cast-data/full_normalization_stats',
        
                 # Path where the model is saved. The file name (os.path.basename)
                 # is the named used for the Weights & Biases project. 
                 out_path = out_path,
                 
                 checkpoint_interval = 250, # How often to save the weights (in terms of epochs) 
                 verbose = 1, # Set to 3 to get all possible printouts
                 loss_weights = loss_weights,
                 parallel = True,
                 graphcast_pretrain = graphcast_pretrain
        )
    
    
    years = ['2019', '2020']
    with ThreadPoolExecutor() as executor:
        paths = []
        for files in executor.map(get_files_for_year, years):
            paths.extend(files)
    
    print(f'Number of Paths: {len(paths)}')
    
    generator = ZarrDataGenerator(paths, 
                              task_config, 
                              target_lead_times=None,
                              batch_size=batch_size, 
                              num_devices=2, 
                              preprocess_fn=add_local_solar_time,
                              prefetch_size=3,
                              random_seed=42, 
                             )

    trainer.fit_generator(generator, 
                          model_params=model_params, 
                          state=state, 
                          )

