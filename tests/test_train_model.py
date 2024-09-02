# UNIT TESTS FOR TRAINING MODELS AND PRODUCING PREDCTIONS
# TEST INCLUDES GENERATING DUMMY DATA AND TRAINING A MODEL
# AND RUNNING EXISTING MODELS. CURRENT MODEL RUNS TEST 
# WOFSCAST AND DIFFUSION, APPLIED BOTH IN-STEP AND IN-POST

# AUTHOR : monte-flora 


import unittest
import os, sys
import glob

import numpy as np
import pandas as pd
import xarray as xr
import optax, jax
import gc

# Add project directory to the system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

# WoFSCast imports
from wofscast.model import WoFSCastModel
from wofscast import wofscast_task_config
from wofscast.data_generator import (add_local_solar_time, 
                                     dataset_to_input, 
                                     load_chunk, 
                                     shard_xarray_dataset)

from wofscast.graphcast_lam import TaskConfig
from wofscast.common.helpers import to_datetimes
from wofscast.normalization import PyTorchScaler
from wofscast.diffusion import apply_diffusion, EDMPrecond
from diffusers import UNet2DModel


class FakeGenerator:
    def __init__(self, task_config, num_devices=2):
        self.task_config = task_config
        self.num_devices = num_devices
        
    def generate(self):
        
        input_batch, target_batch, forcing_batch = [], [], []
        
        for _ in range(2):
            ds = get_fake_dataset()
            inputs, targets, forcings = get_fake_inputs_targets_forcings(ds, self.task_config)
            input_batch.append(inputs)
            target_batch.append(targets)
            forcing_batch.append(forcings)
            
        inputs = xr.concat(input_batch, dim='batch')
        targets = xr.concat(target_batch, dim='batch')
        forcings = xr.concat(forcing_batch, dim='batch')
        
        # Shard the dataset.
        inputs = shard_xarray_dataset(inputs, self.num_devices)
        targets = shard_xarray_dataset(targets, self.num_devices)
        forcings = shard_xarray_dataset(forcings, self.num_devices)
        
        return inputs, targets, forcings 
        
        


def get_fake_task_config():
    
    FORCING_VARS = [
            'local_solar_time_sin',
            'local_solar_time_cos',
    ]
    INPUT_VARS = ["U", "V", "W", "T2"]
    STATIC_VARS = ['HGT', 'XLAND']
    
    # Build a fake TaskConfig 
    task_config = TaskConfig(
        input_variables = INPUT_VARS + STATIC_VARS + FORCING_VARS,
        target_variables = INPUT_VARS, 
        forcing_variables = FORCING_VARS,
        pressure_levels = np.array([0,1,2,3]),
        input_duration = '20min', 
        n_vars_2D = 1,
        domain_size = 1,
        tiling=None, 
        train_lead_times = '10min'
     )
    
    return task_config

def get_fake_dataset():
    # Define the dimensions
    lat = np.linspace(30, 45, 10).astype(np.float32)
    long = np.linspace(80, 100, 20).astype(np.float32)
    level = np.arange(0, 6)  # 5 levels

    # Define the base datetime and time deltas
    base_datetime = pd.Timestamp('2024-01-01 00:00:00')
    time_deltas = pd.to_timedelta([0, 10, 20], unit='m')  # 3 time steps with hourly intervals

    # Create 3D variables (time, level, lat, long)
    U = np.random.rand(len(time_deltas), len(level), len(lat), len(long))
    V = np.random.rand(len(time_deltas), len(level), len(lat), len(long))
    W = np.random.rand(len(time_deltas), len(level), len(lat), len(long))

    # Create a 2D variable (time, lat, long)
    T2 = np.random.rand(len(time_deltas), len(lat), len(long))
    
    # Create static variables ( lat, long)
    HGT = np.random.rand(len(lat), len(long))
    XLAND = np.random.rand(len(lat), len(long))
    
    # Create the dataset
    ds = xr.Dataset(
        {
            "U": (["time", "level", "lat", "lon"], U),
            "V": (["time", "level", "lat", "lon"], V),
            "W": (["time", "level", "lat", "lon"], W),
            "T2": (["time", "lat", "lon"], T2),
            'HGT' : (["lat", "lon"], HGT),
            'XLAND' : (["lat", "lon"], XLAND),
            
        },
        coords={
            "lat": lat,
            "lon": long,
            "level": level,
            "time": time_deltas,
            "datetime": ("time", base_datetime + time_deltas),
        }
    )
    
    # Add the local solar time variables
    ds = add_local_solar_time(ds)
    
    # Convert the entire dataset to float32
    ds = ds.astype(np.float32)
    
    return ds 


def get_fake_inputs_targets_forcings(ds, task_config):     
    
    inputs, targets, forcings = dataset_to_input(ds, task_config, 
                                             target_lead_times=None,
                                             batch_over_time=False, # Deprecated
                                             n_target_steps=2 
                                            )
    
    
    return inputs, targets, forcings


def create_temp_norm_stats(
                           tmp_dir='/home/monte.flora/python_packages/frdd-wofs-cast/tests/test_data'):
    """
    Compute and save mean, standard deviation, and time difference standard deviation by level.
    If the files already exist in the specified directory, the function does nothing.
    
    Parameters:
    ds (xarray.Dataset): The dataset to compute statistics on.
    tmp_dir (str): The directory to save the resulting netCDF files.
    """
    ds = get_fake_dataset()
    
    # Define the file paths
    mean_by_level_path = os.path.join(tmp_dir, 'mean_by_level.nc')
    stddev_by_level_path = os.path.join(tmp_dir, 'stddev_by_level.nc')
    diffs_stddev_by_level_path = os.path.join(tmp_dir, 'diffs_stddev_by_level.nc')

    # Check if files already exist
    if os.path.exists(mean_by_level_path) and os.path.exists(stddev_by_level_path) and os.path.exists(diffs_stddev_by_level_path):
        print("Files already exist. Skipping computation.")
        return
    
    # Compute the statistics
    mean_by_level = ds.mean(dim=['time', 'lat', 'lon'])
    stddev_by_level = ds.std(dim=['time', 'lat', 'lon'], ddof=1)
    time_diffs = ds.diff(dim='time')
    diffs_stddev_by_level = time_diffs.std(dim=['time', 'lat', 'lon'], ddof=1)

    # Save the results to the specified directory
    mean_by_level.to_netcdf(mean_by_level_path)
    stddev_by_level.to_netcdf(stddev_by_level_path)
    diffs_stddev_by_level.to_netcdf(diffs_stddev_by_level_path)

class TestWoFSCastModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # This method runs once before any tests in the test case.
        # Some tests require a saved model state to load from. 
        # For the set up, we train and save such a model. 
        
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.tmp_dir = os.path.join(cls.test_dir, 'test_data')
        cls.model_path = os.path.join(cls.tmp_dir, 'wofscast_test.npz')
        
        # Setup a task config
        cls.task_config = get_fake_task_config()

        # Create a model and save it for later tests
        cls.norm_stats_path = cls.tmp_dir
        
        warmup_steps = 3
        decay_steps = 7
        n_steps = warmup_steps + decay_steps
        
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=1e-4,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )

        # Initialize and train a WoFSCastModel
        cls.model = WoFSCastModel(
            task_config=cls.task_config,
            mesh_size=3,
            latent_size=8,
            gnn_msg_steps=4,
            hidden_layers=8,
            grid_to_mesh_node_dist=5,
            n_steps=n_steps,
            learning_rate_scheduler=scheduler,
            checkpoint=False,
            norm_stats_path=cls.norm_stats_path,
            out_path=cls.model_path,
            checkpoint_interval=1,
            parallel=True,
            verbose=0,
            use_wandb=False
        )

        generator = FakeGenerator(cls.task_config)
        model_params, state = None, {}
        cls.model.fit_generator(generator, model_params=model_params, state=state)
        

    @classmethod
    def tearDownClass(cls):
        # This method runs once after all tests are done
        # Delete all .npz files created in the test directory
        npz_files = glob.glob(os.path.join(cls.tmp_dir, '*.npz'))
        for file in npz_files:
            try:
                os.remove(file)
                print(f"Deleted {file}")
            except OSError as e:
                print(f"Error deleting file {file}: {e}")

    # dtype issue; needs to be resolved!
    def test_fine_tuning(self):
        # Test initializing model parameters from existing parameters
        # and resuming training.
        
        out_path = self.model_path.replace('.npz', '_fine_tuned.npz')
        scheduler = optax.constant_schedule(3e-6)
        
        trainer = WoFSCastModel(
            task_config=self.task_config,
            mesh_size=3,
            latent_size=8,
            gnn_msg_steps=4,
            hidden_layers=8,
            grid_to_mesh_node_dist=5,
            n_steps=5,
            learning_rate_scheduler=scheduler,
            checkpoint=False,
            norm_stats_path=self.norm_stats_path,
            out_path=out_path,
            checkpoint_interval=1,
            parallel=True,
            verbose=0,
            use_wandb=False
        )
        
        trainer.load_model(self.model_path)
        model_params, state = trainer.model_params, trainer.state
        
        generator = FakeGenerator(self.task_config)
        
        trainer.fit_generator(generator, model_params=model_params, state=state)
        self.assertTrue(True)  # Dummy assertion just to check the method runs
        
    
    #works!
    def test_default_predict(self):
        # Test loading a saved model and performing rollout.

        ds = get_fake_dataset()
        inputs, targets, forcings = get_fake_inputs_targets_forcings(ds, self.task_config)
        
        model = WoFSCastModel()
        model.load_model(self.model_path)
        
        predictions = model.predict(inputs, targets, forcings, 
                                         initial_datetime='202105042200', 
                                         n_steps=5, replace_bdry=False)
        
        self.assertIsNotNone(predictions)
        

    #works!
    def test_wofscast_predict(self):
        # Test loading a saved WoFSCast model and performing rollout on WoFS Data.
        # Test also includes running a diffusion model in-step and in-post
        # Test example loads v178 model, a 10-min timestep on the 150 x 150 area.
        # Test diffusion model includes all variables. 
        
        test_dir = os.path.dirname(os.path.abspath(__file__))
        tmp_dir = os.path.join(test_dir, 'test_data')
        
        model_path = os.path.join(tmp_dir, 'wofscast_test_v178.npz') 
        data_path = os.path.join(tmp_dir, 'wrfwof_2021-05-15_020000_to_2021-05-15_041000__10min__ens_mem_09.zarr') 
        
        # Load the normalization statistics and initialize the 
        # scaler used for the diffusion model. 
        norm_stats_path = '/work/mflora/wofs-cast-data/full_normalization_stats'
        mean_by_level = xr.load_dataset(os.path.join(norm_stats_path, 'mean_by_level.nc'))
        stddev_by_level = xr.load_dataset(os.path.join(norm_stats_path, 'stddev_by_level.nc'))
        diffs_stddev_by_level = xr.load_dataset(os.path.join(norm_stats_path, 'diffs_stddev_by_level.nc'))

        scaler = PyTorchScaler(mean_by_level, stddev_by_level, diffs_stddev_by_level)
        
        dataset = load_chunk([data_path], 1, add_local_solar_time)
        dataset = dataset.compute()
        
        timestep = 10 
        steps_per_hour = 60 // timestep # 60 min / 5 min time steps
        hours = 3
        n_steps = steps_per_hour * hours 
        dts = to_datetimes(data_path, n_times = n_steps+2)
        
        # Load the base WoFSCast model.
        model = WoFSCastModel()
        model.load_model(model_path)
        
        domain_size = dataset.dims['lat']
        
        # Load the diffusion model.
        diffusion_model_path = os.path.join(tmp_dir, 'diffusion_all_vars_update_v1') 
        diffusion_model = UNet2DModel.from_pretrained(diffusion_model_path).to('cuda')
        n_channels = 1  # This arg is not used. 
        model_wrapped = EDMPrecond(domain_size, n_channels, diffusion_model, 
                               sigma_data=0.5, 
                               sigma_max=float('inf'), 
                               sigma_min=0.)
        sampler_kwargs = dict(
            sigma_min = 0.0002,
            sigma_max = 1000, 
            S_churn=0,#np.sqrt(2)-1, 
            S_min=0.02, 
            S_max=1000, 
            S_noise=1.5) 
        
        inputs, targets, forcings = dataset_to_input(dataset, model.task_config, 
                                             target_lead_times=slice('10min', '120min'),
                                             batch_over_time=False, n_target_steps=2)
        
        # Base WoFSCast model prediction
        predictions = model.predict(inputs, targets, forcings, 
                            initial_datetime=dts[0], 
                            n_steps=n_steps,
                            replace_bdry=False)

        # WoFSCast with diffusion in-step
        predictions_with_diff = model.predict(inputs, 
                            targets, 
                            forcings, 
                            initial_datetime=dts[0], 
                            n_steps=n_steps,             
                            diffusion_model=model_wrapped, 
                            scaler = scaler,             
                            n_diffusion_steps=10,
                            sampler_kwargs=sampler_kwargs, 
                            variables = None
                           )
        
        # WoFSCast with Diffusion applied in-post
        updated_predictions = predictions.copy(deep=True)

        for t in range(predictions.sizes['time']):
            # Select predictions for the current time coordinate
            these_predictions = updated_predictions.isel(time=[t])
    
            # Apply the diffusion process or any other update
            updated_prediction = apply_diffusion(these_predictions, targets, 
                                         model_wrapped, 
                                         scaler, num_steps=10, 
                                         sampler_kwargs=sampler_kwargs, 
                                         variables = None # Predicting all variables
                                        )
            
            # Assign the updated prediction back to the correct time index in the updated_predictions dataset
            updated_predictions[dict(time=[t])] = updated_prediction

        self.assertIsNotNone(predictions) 
 
if __name__ == '__main__':
    unittest.main()
