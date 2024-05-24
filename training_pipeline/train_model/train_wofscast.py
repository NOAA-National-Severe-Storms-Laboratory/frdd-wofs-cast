# WoFSCast 
import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from wofscast.model import WoFSCastModel
from wofscast.wofscast_task_config import WOFS_TASK_CONFIG, DBZ_TASK_CONFIG

#Third-party 
from glob import glob
import numpy as np 
import os


# Fine tuning. 
# TODO: Load model_params or add loading the model params into the Trainer, 
# for the fine tuning. 
    
#TODO: Build a data loader based on the advice in this blog: https://earthmover.io/blog/cloud-native-dataloader/
    
# Using the Weights & Biases package: 
# Create an account : https://docs.wandb.ai/quickstart

# >>> python -m wandb login

if __name__ == '__main__':
    
    # Get the training file paths. 
    
    """ usage: stdbuf -oL python -u train_wofscast.py > & log_training & """
    
    # Data is preloaded into CPU memory @ cpu_batch_size
    # subsets in size of gpu_batch_size are fed to the GPU 
    # one at a time. 
    cpu_batch_size = 128 
    gpu_batch_size = 32  


    loss_weights = {
                                  # Any variables not specified here are weighted as 1.0.
                                  # A single-level variable, but an important headline variable
                                  # and also one which we have struggled to get good performance
                                  # on at short lead times, so leaving it weighted at 1.0, equal
                                  # to the multi-level variables:
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
                 task_config = DBZ_TASK_CONFIG, 

                 mesh_size=5, 
                 # Parameters for the MLPs
                 latent_size=256, 
                 gnn_msg_steps=8, # Increasing this allows for connecting information from farther away. 
                 hidden_layers=2, 
                 grid_to_mesh_node_dist=5, # Distance in grid points (5 * 3km = 15 km) 
                 
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
                 
                 cpu_batch_size = cpu_batch_size,
                 gpu_batch_size = gpu_batch_size,          
                 
                 checkpoint=True, # Save the model periodically
            
                 norm_stats_path = '/work/mflora/wofs-cast-data/normalization_stats',
                 # Path where the model is saved. The file name (os.path.basename)
                 # is the named used for the Weights & Biases project. 
                 out_path = '/work/mflora/wofs-cast-data/model/wofscast_dbz_weighted_loss.npz',
                 
                 checkpoint_interval = 5, # How often to save the weights (in terms of epochs) 
                 verbose=1, 
                 loss_weights = loss_weights,
                 use_multi_gpus = True
    )
    
    base_path = '/work/mflora/wofs-cast-data/datasets_jsons'
    years = ['2019', '2020']
    paths = [join(base_path, year, file) for year in years for file in os.listdir(join(base_path, year))]
    
    trainer.fit_generator(paths)

    # Plot the training loss and diagnostics. 
    trainer.plot_training_loss()
    trainer.plot_diagnostics()





