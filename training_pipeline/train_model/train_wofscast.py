# WoFSCast 
import warnings
# Suppress the specific RuntimeWarning about os.fork(),  multithreaded code, and JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.")



import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from wofscast.model import WoFSCastModel
from wofscast.wofscast_task_config import WOFS_TASK_CONFIG, DBZ_TASK_CONFIG
from wofscast.data_generator import ZarrDataGenerator
from wofscast import checkpoint

import os
from os.path import join
from concurrent.futures import ThreadPoolExecutor
    
# Using the Weights & Biases package: 
# Create an account : https://docs.wandb.ai/quickstart

# >>> python -m wandb login

def get_files_for_year(year):
    """Get all zarr files within a directory."""
    year_path = join(base_path, year)
    with os.scandir(year_path) as it:
        return [join(year_path, entry.name) for entry in it if entry.is_dir() and entry.name.endswith('.zarr')] 

def truncate_to_chunk_size(input_list, chunk_size=512):
    # Calculate the new length as the smallest multiple of chunk_size
    # that is greater than or equal to the length of the list
    new_length = ((len(input_list) + chunk_size - 1) // chunk_size) * chunk_size
    # If the list is already a multiple of chunk_size, no need to truncate
    if new_length > len(input_list):
        new_length -= chunk_size
    # Truncate the list
    return input_list[:new_length]
    
if __name__ == '__main__':
    """ usage: stdbuf -oL python -u train_wofscast.py > & log_training & """
    
    # Whether to initialize the model with existing model weights 
    # WARNING: Assumes the model.py args are the same and does not check! 
    fine_tune = False
    
    # Where the model weights are stored
    out_path = '/work/mflora/wofs-cast-data/model/wofscast_baseline.npz'
    
    model_params, state = None, {}
    if fine_tune: 
         # Warning about model parameters compatibility
        warnings.warn("""User must ensure model parameters are compatible with the model.py args below! 
        There is no check at the moment!""", UserWarning)
        
        # Load a checkpoint from an existing model. 
        model_path = '/work/mflora/wofs-cast-data/model/wofscast_baseline.npz'
        with open(model_path, 'rb') as f:
            data = checkpoint.load(f, dict)
            model_params, state = data['parameters'], {}
        
        # Do not want to replace the existing checkpoint! 
        out_path = model_path.replace('.npz', '_fine_tune.npz') 
    
    # The task config contains details like the input variables, 
    # target variables, time step, etc.
    task_config = WOFS_TASK_CONFIG
    
    # Data is lazily loaded into CPU memory @ cpu_batch_size_factor * gpu_batch_size
    # sized subsets. gpu_batch_size'd batches are loaded and fed to 
    # the GPU. 
    
    # In my testing, factors ~ 2-4 were optimal. 
    
    cpu_batch_size_factor = 2 
    gpu_batch_size = 64  
    n_workers = 16 
    
    generator_kwargs = dict(cpu_batch_size=cpu_batch_size_factor*gpu_batch_size, 
                            gpu_batch_size=gpu_batch_size,
                            n_workers = n_workers,
                           )
    
    loss_weights = {
                    # Any variables not specified here are weighted as 1.0.
                    'U' : 1.0, 
                    'V': 1.0, 
                    'W': 1.0, 
                    'T': 1.0, 
                    'GEOPOT': 1.0, 
                    'QVAPOR': 1.0,
                    'T2' : 0.5, 
                    'COMPOSITE_REFL_10CM' : 0.5, 
                    'UP_HELI_MAX' : 0.5,
                    'RAIN_AMOUNT' : 0.5,
                    }

    #loss_weights = {'COMPOSITE_REFL_10CM' : 1.0}
    
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
                 k_hop=8,
                 num_attn_heads  = 4, 
        
                 # Number of training epochs for the 2-phases (linearly increase;
                 # cosine decay).
                 n_epochs_phase1 = 2, 
                 n_epochs_phase2 = 2,
        
                 n_epochs_phase3 = 0, # Only use if fine tuning for > 1 step rollout. 
                 total_timesteps = 12, # 2+ hours of total rollout for training. 
                         
                 checkpoint=True, # Save the model periodically
            
                 norm_stats_path = '/work/mflora/wofs-cast-data/full_normalization_stats',
        
                 # Path where the model is saved. The file name (os.path.basename)
                 # is the named used for the Weights & Biases project. 
                 out_path = out_path,
                 
                 checkpoint_interval = 5, # How often to save the weights (in terms of epochs) 
                 verbose = 1, # Set to 3 to get all possible printouts
                 loss_weights = loss_weights,
                 use_multi_gpus = True,
                 generator_kwargs = generator_kwargs
    )
    
    base_path = '/work/mflora/wofs-cast-data/datasets_zarr'
    years = ['2019', '2020']
    
    with ThreadPoolExecutor() as executor:
        paths = []
        for files in executor.map(get_files_for_year, years):
            paths.extend(files)
    
    print(f'Number of Paths: {len(paths)}')
    
    # Ensure the file_paths are compatiable with the generator_chunk_size 
    paths = truncate_to_chunk_size(paths, chunk_size=gpu_batch_size)
    
    print(f'Number of Paths after truncation: {len(paths)}')
    
    trainer.fit_generator(paths[:128], model_params=model_params, state=state)

    # Plot the training loss and diagnostics. 
    trainer.plot_training_loss()
    trainer.plot_diagnostics()





