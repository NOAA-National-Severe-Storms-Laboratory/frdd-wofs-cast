from graphcast_trainer import WoFSCastTrainer 


from glob import glob
import numpy as np 
import os

generate_chunk_size = 32

trainer = WoFSCastTrainer(mesh_size=5, 
                 latent_size=128, 
                 gnn_msg_steps=8, # Increasing this allows for connecting information from farther away. 
                 hidden_layers=1, 
                 grid_to_mesh_node_dist=5,
                 n_epochs_phase1 = 5, 
                 n_epochs_phase2 = 5,
                 n_epochs_phase3 = 0,
                 total_timesteps = 12, # 2+ hours of total rollout for training. 
                 batch_size=8,
                 generator_chunk_size=generate_chunk_size,          
                 checkpoint=True,
                 norm_stats_path = '/work/mflora/wofs-cast-data/normalization_stats',
                 out_path = '/work/mflora/wofs-cast-data/model/wofscast_test.npz',
                 checkpoint_interval = 100, 
                 verbose=1)

fine_tune = True

# Get the training file paths. 
N_SAMPLES = 32 

dir_path = 'datasets_2hr' if fine_tune else 'datasets'

data_paths = []
for year in ['2019', '2020']:
    data_paths.extend(glob(os.path.join(f'/work/mflora/wofs-cast-data/{dir_path}/{year}/wrf*.nc')))
    
rs = np.random.RandomState(123)
data_paths = rs.choice(data_paths, size=N_SAMPLES, replace=False)

if fine_tune: 
    # TODO: Load model_params or add loading the model params into the Trainer, 
    # for the fine tuning. 
    
    #TODO: Build a data loader based on the advice in this blog: https://earthmover.io/blog/cloud-native-dataloader/
    
    trainer = WoFSCastTrainer(mesh_size=5, 
                 latent_size=128, 
                 gnn_msg_steps=8, # Increasing this allows for connecting information from farther away. 
                 hidden_layers=1, 
                 grid_to_mesh_node_dist=5,
                 n_epochs_phase1 = 0, 
                 n_epochs_phase2 = 0,
                 n_epochs_phase3 = 10,
                 total_timesteps = 12, # 2+ hours of total rollout for training. 
                 batch_size=8,
                 generator_chunk_size=generate_chunk_size,          
                 checkpoint=True,
                 norm_stats_path = '/work/mflora/wofs-cast-data/normalization_stats',
                 out_path = '/work/mflora/wofs-cast-data/model/wofscast_test.npz',
                 checkpoint_interval = 100, 
                 verbose=1)
    
    trainer.fit_generator(data_paths, client=None)
    
else:
    

# Plot the training loss and diagnostics. 
#trainer.plot_training_loss()
#trainer.plot_diagnostics()





