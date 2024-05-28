# WoFSCast 
import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from wofscast.model import WoFSCastModel
from wofscast.wofscast_task_config import WOFS_TASK_CONFIG, DBZ_TASK_CONFIG

import os
from os.path import join
from concurrent.futures import ThreadPoolExecutor

# Fine tuning. 
# TODO: Load model_params or add loading the model params into the Trainer, 
# for the fine tuning. 
    
# Using the Weights & Biases package: 
# Create an account : https://docs.wandb.ai/quickstart

# >>> python -m wandb login

def get_files_for_year(year):
    """Get all zarr files within a directory."""
    year_path = join(base_path, year)
    with os.scandir(year_path) as it:
        return [join(year_path, entry.name) for entry in it if entry.is_dir() and entry.name.endswith('.zarr')] 

if __name__ == '__main__':
    """ usage: stdbuf -oL python -u train_wofscast.py > & log_training & """
    
    # Data is lazily loaded into CPU memory @ cpu_batch_size_factor * gpu_batch_size
    # sized subsets. gpu_batch_size'd batches are loaded and fed to 
    # the GPU. 
    cpu_batch_size_factor = 4 
    gpu_batch_size = 32  

    loss_weights = {
                    # Any variables not specified here are weighted as 1.0.
                    'U' : 1.0, 
                    'V': 1.0, 
                    'W': 1.0, 
                    'T': 1.0, 
                    'GEOPOT': 1.0, 
                    'QVAPOR': 1.0,
                    'T2' : 0.1, 
                    'COMPOSITE_REFL_10CM' : 0.1, 
                    'UP_HELI_MAX' : 0.1,
                    'RAIN_AMOUNT' : 0.1,
                    }

    loss_weights = {'COMPOSITE_REFL_10CM' : 1.0}
    
    trainer = WoFSCastModel(
                 # The task config contains details like the input variables, 
                 # target variables, time step, etc.
                 task_config = WOFS_TASK_CONFIG, 

                 mesh_size=5, # Number of Mesh refinements or more higher resolution layers. 
                 
                 # Parameters for the MLPs-------------------
                 latent_size=128, 
                 gnn_msg_steps=8, # Increasing this allows for connecting information from farther away. 
                 hidden_layers=1, 
                 grid_to_mesh_node_dist=0.25, # Fraction of the maximum distance between mesh nodes on the 
                                             # finest mesh level. @ level 5, max distance ~ 4.5 km, 
                                             # so connecting to those grid points with 1-2 km
                 #--------------------------------------------
                 # Parameters if using a transformer layer for processor (mesh)
                 # the transformer also relies on the latent_size arg above.
                 use_transformer = False, 
                 k_hop=8,
                 num_attn_heads  = 4, 
        
                 # Number of training epochs for the 2-phases (linearly increase;
                 # cosine decay).
                 n_epochs_phase1 = 5, 
                 n_epochs_phase2 = 5,
        
                 n_epochs_phase3 = 0, # Only use if fine tuning for > 1 step rollout. 
                 total_timesteps = 12, # 2+ hours of total rollout for training. 
                 
                 cpu_batch_size_factor = cpu_batch_size_factor,
                 gpu_batch_size = gpu_batch_size,          
                 
                 checkpoint=True, # Save the model periodically
            
                 norm_stats_path = '/work/mflora/wofs-cast-data/full_normalization_stats',
                 # Path where the model is saved. The file name (os.path.basename)
                 # is the named used for the Weights & Biases project. 
                 out_path = '/work/mflora/wofs-cast-data/model/wofscast_dbz_weighted_loss.npz',
                 
                 checkpoint_interval = 1, # How often to save the weights (in terms of epochs) 
                 verbose=1, # Set to 3 to get all possible printouts
                 loss_weights = loss_weights,
                 use_multi_gpus = True
    )
    
    base_path = '/work/mflora/wofs-cast-data/datasets_zarr'
    years = ['2019', '2020']
    
    with ThreadPoolExecutor() as executor:
        paths = []
        for files in executor.map(get_files_for_year, years):
            paths.extend(files)

            
    N_SAMPLES = 1024        
            
    trainer.fit_generator(paths[:N_SAMPLES])

    # Plot the training loss and diagnostics. 
    trainer.plot_training_loss()
    trainer.plot_diagnostics()





