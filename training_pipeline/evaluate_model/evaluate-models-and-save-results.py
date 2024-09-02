# From WoFSCast
import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from wofscast.model import WoFSCastModel
from wofscast.data_generator import open_zarr
from plot_params import target_vars, display_name_mapper, units_mapper

from wofscast.data_generator import (load_chunk, 
                                     WRFZarrFileProcessor,
                                     WoFSDataProcessor, 
                                     dataset_to_input,
                                     add_local_solar_time,
                                     open_zarr
                                    )

# Third-party
import os 
import xarray 
import numpy as np 
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime, timedelta

import pandas as pd
import xarray as xr
from tqdm import tqdm
from wofscast.diffusion import EDMPrecond
from diffusers import UNet2DModel
import itertools 


import sys
sys.path.insert(0, '/home/monte.flora/python_packages/MontePython')
import monte_python

from pathlib import Path 
class MRMSDataLoader: 
    
    MRMS_PATH = '/work/rt_obs/MRMS/RAD_AZS_MSH/'
    
    def __init__(self, case_date, datetime_rng):
        self.case_date = case_date
        self.datetime_rng = datetime_rng 

    def find_mrms_files(self):
        """
        When given a start and end date, this function will find any MRMS RAD 
        files between those time periods. It will check if the path exists. 
        """
        year = str(self.datetime_rng[0].year) 
        mrms_filenames = [date.strftime('wofs_MRMS_RAD_%Y%m%d_%H%M.nc') for date in self.datetime_rng]

        mrms_filepaths = [Path(self.MRMS_PATH).joinpath(year, self.case_date, f) 
                          if Path(self.MRMS_PATH).joinpath(year, self.case_date, f).is_file() else None
                          for f in mrms_filenames 
                 ]
       
        return mrms_filepaths 
    
    def resize(self, ds, n_lat=300, n_lon=300, domain_size=150):
        """Resize the domain"""
        start_lat, start_lon = (n_lat - domain_size) // 2, (n_lon - domain_size) // 2
        end_lat, end_lon = start_lat + domain_size, start_lon + domain_size
        
        # Subsetting the dataset to the central size x size grid
        ds_subset = ds.isel(lat=slice(start_lat, end_lat), lon=slice(start_lon, end_lon))
        
        return ds_subset
    
    def load(self):

        files = self.find_mrms_files()
        
        # Initialize an empty list to store the datasets with 'mesh_consv' variable
        data = np.zeros((len(files), 150, 150))

        # Load 'mesh_consv' variable from each file and append to the datasets list
        for t, file in enumerate(files):
            if file is not None: 
                ds = xr.open_dataset(file, drop_variables=['lat', 'lon'])
                
                # Resize the output to 150 x 150
                ds = self.resize(ds)
                
                data[t,:,:] = ds['dz_consv'].values
    
                ds.close()
        

        return data

def get_case_date(path):
    name = os.path.basename(path)
    comps = name.split('_')
    
    start_date = comps[1]+'_'+comps[2]
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d_%H%M%S')
    
    if start_date_dt.hour < 14:
        case_date = start_date_dt.date() - timedelta(days=1)
    else:
        case_date = start_date_dt.date() 
        
    return case_date.strftime('%Y%m%d')

def to_datetimes(path, n_times = 13):  
    name, freq, ens_mem = os.path.basename(path).split('__')
    start_time_dt = datetime.strptime(name.split('_to')[0], 'wrfwof_%Y-%m-%d_%H%M%S')
    start_time = pd.Timestamp(start_time_dt)
    
    dt_list = pd.date_range(start=start_time, periods=n_times, freq=freq)
    
    # Remove the first 2 time steps, since those aren't plotted 
    # as they are used as the initial input. 
    return dt_list[2:]


def mean_preserving_time(x: xarray.DataArray) -> xarray.DataArray:
    return x.mean([d for d in x.dims if d != 'time'], skipna=True)

def _border_mask(shape, N=10):
    """
    Create a border mask for an array of given shape.

    Parameters:
    - shape: tuple, the shape of the array (NY, NX).
    - N: int, the width of the border where values should be True.

    Returns:
    - mask: jax.numpy.ndarray, a mask where border values are True and interior values are False.
    """
    NY, NX = shape
    mask = np.zeros(shape, dtype=bool)

    # Set the border to True
    mask[:N, :] = True  # Top border
    mask[-N:, :] = True  # Bottom border
    mask[:, :N] = True  # Left border
    mask[:, -N:] = True  # Right border

    return mask

# Create a border mask for the domain (slow to constantly recreate this!!!)
BORDER_MASK = _border_mask((150, 150), N=3)  # Adjust N as needed

# Function to calculate RMSE while ignoring the borders
def rmse_ignoring_borders(predictions, targets):
    # Set the errors at the borders to NaN
    err = (predictions - targets)**2
    err = xarray.where(BORDER_MASK, np.nan, err)  # Apply the border mask
    
    # Compute mean squared error while preserving 'time' dimension
    mse = mean_preserving_time(err)
    
    # Calculate the RMSE
    rmse = np.sqrt(mse)
    return rmse


def rmse_in_convection(predictions, targets, refl_mask):
    
    # Set the errors at the borders to NaN
    err = (predictions - targets)**2
    err = xarray.where(refl_mask, err, np.nan)  # Apply the refl mask
    
    # Compute mean squared error while preserving 'time' dimension
    mse = mean_preserving_time(err)
    
    # Calculate the RMSE
    rmse = np.sqrt(mse)
    return rmse

def accumulate_rmse(targets, predictions, target_vars, rmse_dict):
    """Accumulate RMSE for each prediction"""
    for var in target_vars: 
        # Compute full domain RMSE
        rmse = rmse_ignoring_borders(predictions[var], targets[var])
    
        # Compute RMSE where comp. refl > 3: 
        pred_refl_mask = np.where(predictions['COMPOSITE_REFL_10CM']>3, True, False)
        tar_refl_mask = np.where(targets['COMPOSITE_REFL_10CM']>3, True, False)
    
        # Combine the masks with logical OR to create the composite reflectivity mask
        refl_mask = np.logical_or(pred_refl_mask, tar_refl_mask)
        
        # Adjust mask dimensions if necessary
        if len(refl_mask.shape) < len(predictions[var].shape):
            refl_mask = np.expand_dims(refl_mask, axis=2)
        
        refl_mask = np.broadcast_to(refl_mask, predictions[var].shape)
    
        rmse_conv = rmse_in_convection(predictions[var], targets[var], refl_mask)
    
        rmse_dict['Full Domain'][var] += rmse
        rmse_dict['Convective Regions'][var] += rmse_conv
        
    return rmse_dict

def process_time_step(this_pred, this_tar, this_mrms):
        
    # Identify WoFSCast storm objects.
    labels_pred, pred_object_props = monte_python.label(
            input_data=this_pred['COMPOSITE_REFL_10CM'],
            method='single_threshold', 
            return_object_properties=True, 
            params={'bdry_thresh': 47})
        
    # Identify the WoFS storm objects.
    labels_tar, tar_object_props = monte_python.label(
            input_data=this_tar['COMPOSITE_REFL_10CM'],
            method='single_threshold', 
            return_object_properties=True, 
            params={'bdry_thresh': 47})
    
    # WoFS has a high reflectivity bias, so for percentile-matching
    # use a lower MRMS dBZ threshold. Using Skinner et al. (2018)
    # 40 MRMS ~ 44-45 WOFS. 
    labels_mrms, mrms_object_props = monte_python.label(
            input_data=this_mrms,
            method='single_threshold', 
            return_object_properties=True, 
            params={'bdry_thresh': 40})
    
    # Quality control
    labels_pred, pred_object_props = qcer.quality_control(
            this_pred['COMPOSITE_REFL_10CM'], labels_pred, pred_object_props, qc_params)

    labels_tar, tar_object_props = qcer.quality_control(
            this_tar['COMPOSITE_REFL_10CM'], labels_tar, tar_object_props, qc_params)
    
    labels_mrms, mrms_object_props = qcer.quality_control(
            this_mrms, labels_mrms, mrms_object_props, qc_params)
    
    # Update metrics for WoFSCast vs. WoFS
    obj_verifier.update_metrics(labels_tar, labels_pred)
    results1 = {f'wofscast_vs_wofs_{key}': getattr(obj_verifier, f"{key}_") 
                for key in ["hits", "false_alarms", "misses"]}
    
    obj_verifier.reset_metrics()
    
    # Update metrics for WoFSCast vs. MRMS 
    obj_verifier.update_metrics(labels_mrms, labels_pred)
    results2 = {f'wofscast_vs_mrms_{key}': getattr(obj_verifier, f"{key}_")
                    for key in ["hits", "false_alarms", "misses"]}
    
    obj_verifier.reset_metrics()
    
    # Update metrics for WoFS vs. MRMS 
    obj_verifier.update_metrics(labels_mrms, labels_tar)
    results3 = {f'wofs_vs_mrms_{key}': getattr(obj_verifier, f"{key}_") 
                    for key in ["hits", "false_alarms", "misses"]}
    
    obj_verifier.reset_metrics()
    
    return {**results1, **results2, **results3}
        
    return results
        
def replace_zeros(data): 
    return np.where(data==0, 1e-5, data)

def rmse_dict_to_dataframe(rmse_dict):
    """
    Convert a nested dictionary of xarray.DataArray objects to a pandas DataFrame.
    
    Parameters:
    - rmse_dict: dict, nested dictionary with RMSE values
    
    Returns:
    - pd.DataFrame, DataFrame with hierarchical indexing
    """
    data = []

    for key1, nested_dict in rmse_dict.items():
        for key2, data_array in nested_dict.items():
            # Ensure data_array is an xarray.DataArray
            if isinstance(data_array, xr.DataArray):
                # Extract values and timesteps
                values = data_array.values
                timesteps = data_array.coords['time'].values if 'time' in data_array.coords else range(len(values))
                
                for timestep, value in zip(timesteps, values):
                    data.append((key1, key2, timestep, value))

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Category', 'Variable', 'Time', 'RMSE'])
    
    # Set hierarchical index
    df.set_index(['Category', 'Variable', 'Time'], inplace=True)
    
    return df


if __name__ == "__main__":
    
    """ usage: stdbuf -oL python -u evaluate-models-and-save-results.py > & log_evaluate & """
    
    matcher = monte_python.ObjectMatcher(cent_dist_max = 14, # 40 km distance to match 
                                     min_dist_max = 14, 
                                     time_max=0, 
                                     score_thresh=0.2, 
                                     one_to_one = True)

    obj_verifier = monte_python.ObjectVerifier(matcher)
    qcer = monte_python.QualityControler()
    qc_params = [('min_area', 12)]
    
    # For the time series
    #MODEL_PATH = '/work/mflora/wofs-cast-data/model/wofscast_baseline_full_v2.npz'
    #MODEL_PATH = '/work/mflora/wofs-cast-data/model/wofscast_baseline.npz'

    #MODEL_PATH = '/work/mflora/wofs-cast-data/model/wofscast_reproducibility_test_seed_42.npz'
    
    MODEL_PATH = '/work/cpotvin/WOFSCAST/model/wofscast_test_v203.npz'
    
    #MODEL_PATH = '/work/mflora/wofs-cast-data/model/wofscast_best_10min_model.npz'

    #MODEL_PATH = '/work/mflora/wofs-cast-data/model/wofscast_10min_model_noise_v1.npz'
    
    tag = '_47dbz' #'_15min'
    
    use_diffusion = False
    model_wrapped = None
    if use_diffusion: 
        #wrap diffusers/pytorch model 
        domain_size = 160
        diffusion_model_path = "/work/mflora/wofs-cast-data/diffusion_models/diffusion_v2"
        diffusion_model = UNet2DModel.from_pretrained(diffusion_model_path).to('cuda')
        model_wrapped = EDMPrecond(domain_size, 1, diffusion_model)
    
    
    model = WoFSCastModel()

    model.load_model(MODEL_PATH)

    n_times = 12 

    rmse_dict = { 'Full Domain' : {v : np.zeros((n_times,)) for v in target_vars},
                  'Convective Regions' : {v : np.zeros((n_times,)) for v in target_vars},
            }
    

    metrics = ['hits', 'misses', 'false_alarms']
    subkeys = ['wofscast_vs_wofs', 'wofscast_vs_mrms', 'wofs_vs_mrms'] 

    cont_dict={}
    for m, s in itertools.product(metrics, subkeys):
        cont_dict[f'{s}_{m}'] = np.zeros((n_times))
    

    # Selecting a single ensemble member. 
    #base_path = '/work/mflora/wofs-cast-data/datasets_2hr_zarr/2021/*_ens_mem_09.zarr'
    base_path = '/work2/mflora/wofscast_datasets/dataset_10min_15min_init/2021/*_ens_mem_09.zarr'
    
    paths = glob(base_path)
    paths.sort()

    N_SAMPLES = 100

    rs = np.random.RandomState(42)
    random_paths = rs.choice(paths, size=N_SAMPLES, replace=False)

    N = len(random_paths)
  
    for path in tqdm(random_paths): 
        print(f"Evaluating {path}...")
        #dataset = load_chunk([path], gpu_batch_size=1, preprocess_fn=add_local_solar_time)
        
        dataset = open_zarr(path)
        dataset = add_local_solar_time(dataset)
        # Reset the levels coordinate
        #if '15min_init' in base_path:
        #    dataset.coords['level'] = np.arange(dataset.dims['level'])

        dataset = dataset.expand_dims(dim='batch', axis=0)
        
        dataset = dataset.compute() 
        inputs, targets, forcings = dataset_to_input(dataset, model.task_config, 
                                             target_lead_times=slice('10min', '120min'), 
                                             batch_over_time=False, n_target_steps=12)

        predictions = model.predict(inputs, targets, forcings, diffusion_model=model_wrapped)
        predictions = predictions.transpose('batch', 'time', 'level', 'lat', 'lon')
        
        # Load the MRMS data and compute object verification statistics 
        # against both the WoFS and MRMS. 
        case_date = get_case_date(path)
        dts = to_datetimes(path, n_times = len(predictions.time)+2)
        loader = MRMSDataLoader(case_date, dts)
        try:
            mrms_dz = loader.load() 
            
            results = [process_time_step(predictions.isel(time=t, batch=0), 
                                     targets.isel(time=t, batch=0),
                                     mrms_dz[t,...]
                                    )
                          for t in np.arange(n_times)]
    
            for t, result in enumerate(results):
                for key in cont_dict.keys():
                    cont_dict[key][t] += result[key]
            
            
        except OSError:
            print(f'Unable to load MRMS data for {case_date}')
        
        # Compute the RMSE statistics. 
        rmse_dict = accumulate_rmse(targets, predictions, target_vars, rmse_dict)

    for key in rmse_dict.keys():
        for v in target_vars:
            rmse_dict[key][v]/=N

    # Save the RMSE results. 
    df = rmse_dict_to_dataframe(rmse_dict)
    out_path = '/work/mflora/wofs-cast-data/verification_results'
    df.to_parquet(os.path.join(out_path, f"MSE_{os.path.basename(MODEL_PATH).replace('.npz','')}{tag}.parquet"))
              
    # Save objects-based results.
    data = {} 
    for subkey in subkeys: 
    
        cont_dict[f'{subkey}_hits'] = replace_zeros(cont_dict[f'{subkey}_hits'])
        cont_dict[f'{subkey}_misses'] = replace_zeros(cont_dict[f'{subkey}_misses'])
        cont_dict[f'{subkey}_false_alarms'] = replace_zeros(cont_dict[f'{subkey}_false_alarms'])

        pod = cont_dict[f'{subkey}_hits'] / (cont_dict[f'{subkey}_hits'] + cont_dict[f'{subkey}_misses'])
        sr = cont_dict[f'{subkey}_hits'] / (cont_dict[f'{subkey}_hits'] + cont_dict[f'{subkey}_false_alarms'])
        
        denom = (cont_dict[f'{subkey}_hits'] + cont_dict[f'{subkey}_misses'] + cont_dict[f'{subkey}_false_alarms'])
        csi = cont_dict[f'{subkey}_hits'] / denom
    
        data[f'{subkey}_POD'] = pod 
        data[f'{subkey}_SR'] = sr
        data[f'{subkey}_CSI'] = csi
        data[f'{subkey}_FB'] = pod / sr 
    
    df_object = pd.DataFrame(data)
    
    df_object.to_json(os.path.join(out_path, f"objects_{os.path.basename(MODEL_PATH).replace('.npz','')}{tag}.json"))


