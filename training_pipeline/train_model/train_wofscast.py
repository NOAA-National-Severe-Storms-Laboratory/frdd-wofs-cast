# WoFSCast 
import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from wofscast.model import WoFSCastModel
from wofscast.wofscast_task_config import WOFS_TASK_CONFIG, DBZ_TASK_CONFIG

#Third-party 
from glob import glob
import numpy as np 
import os


'''
# Deprecated
dir_path = 'datasets_2hr' if fine_tune else 'datasets'
data_paths = []
for year in ['2019', '2020']:
    data_paths.extend(glob(os.path.join(f'/work/mflora/wofs-cast-data/{dir_path}/{year}/wrf*.nc')))
rs = np.random.RandomState(123)
data_paths = rs.choice(data_paths, size=N_SAMPLES, replace=False)
'''

# Fine tuning. 
# TODO: Load model_params or add loading the model params into the Trainer, 
# for the fine tuning. 
    
#TODO: Build a data loader based on the advice in this blog: https://earthmover.io/blog/cloud-native-dataloader/
  
    
# Using the Weights & Biases package: 

# Create an account : https://docs.wandb.ai/quickstart

# >>> python -m wandb login
# 

    
if __name__ == '__main__':
    
    # Get the training file paths. 
    
    """ usage: stdbuf -oL python -u train_wofscast.py > & log_training & """
    
    # Number of samples sent to the GPU at one time. 
    # Note: When making this bigger, you may get an error about
    # a conversion error. It's poor traceback and basical
    batch_size = 32 
    generator_chunk_size = 512 # Number of samples loaded into CPU memory from the parent Zarr file. 


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

    
    loss_weights = {'COMPOSITE_REFL_10CM' : 10}
    
    trainer = WoFSCastModel(
                 # The task config contains details like the input variables, 
                 # target variables, time step, etc.
                 task_config = DBZ_TASK_CONFIG, 

                  mesh_size=5, 
                 # Parameters for the MLPs
                 latent_size=64, 
                 gnn_msg_steps=4, # Increasing this allows for connecting information from farther away. 
                 hidden_layers=1, 
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
                 batch_size=batch_size,
                 generator_chunk_size=generator_chunk_size,          
                 checkpoint=True,
                 norm_stats_path = '/work/mflora/wofs-cast-data/normalization_stats',
                 # Path where the model is saved. The file name (os.path.basename)
                 # is the named used for the Weights & Biases project. 
                 out_path = '/work/mflora/wofs-cast-data/model/wofscast_test.npz',
                 
                 checkpoint_interval = 5, # How often to save the weights (in terms of epochs) 
                 verbose=1, 
                 loss_weights = loss_weights,
                 use_multi_gpus = True
    )
    
    path = '/work/mflora/wofs-cast-data/wofcast_dataset_test1.zarr'
    trainer.fit_generator(path, client=None)

    # Plot the training loss and diagnostics. 
    trainer.plot_training_loss()
    trainer.plot_diagnostics()





