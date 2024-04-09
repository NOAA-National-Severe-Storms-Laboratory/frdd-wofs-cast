#########################################
# Data Generator using Dask 
#########################################

from . import data_utils
from . import my_graphcast as graphcast

import os
import xarray as xr
from glob import glob
import numpy as np
import dataclasses
import random 

from tqdm import tqdm
from dask.diagnostics import ProgressBar
import dask
from dask import delayed, compute
import gc 


def to_static_vars(dataset):
    # Select the first time index for 'HGT' and 'XLAND' variables
    hgt_selected = dataset['HGT'].isel(time=0).drop('time')
    xland_selected = dataset['XLAND'].isel(time=0).drop('time')

    # Now, replace the 'HGT' and 'XLAND' in the original dataset with these selected versions
    dataset = dataset.drop_vars(['HGT', 'XLAND'])
    dataset['HGT'] = hgt_selected
    dataset['XLAND'] = xland_selected

    return dataset


def add_local_solar_time(data: xr.Dataset) -> xr.Dataset:
    """    
    Adds sine and cosine-transformed local solar time variables to the dataset,
    adjusted for longitude, and replicated across latitude.

    Args:
        data: The input dataset with 'time', 'lat', and 'lon' dimensions.

    Returns:
        xr.Dataset: The dataset with 'local_solar_time_sin' and 'local_solar_time_cos' variables added.
    """
    if not {'time', 'lat', 'lon'}.issubset(data.dims):
        missing_dims = {'time', 'lat', 'lon'} - set(data.dims)
        raise ValueError(f"Missing dimensions in the dataset: {missing_dims}")

    # Calculate the local solar time adjustment
    local_hours = (data.coords['datetime'].dt.hour + data.coords['lon'] / 15.0) % 24

    # Convert local_hours to radians for sine and cosine
    radians = (local_hours * 2 * np.pi) / 24

    # Calculate sine and cosine for the local solar time
    local_solar_time_sin = np.sin(radians)
    local_solar_time_cos = np.cos(radians)

    # Create DataArrays with 'time' and 'lon' dimensions
    local_solar_time_sin_da = xr.DataArray(local_solar_time_sin, dims=('time', 'lon'),
                                           coords={'time': data.coords['time'], 'lon': data.coords['lon']})
    local_solar_time_cos_da = xr.DataArray(local_solar_time_cos, dims=('time', 'lon'),
                                           coords={'time': data.coords['time'], 'lon': data.coords['lon']})

    # Replicate values across 'lat' dimension by broadcasting with an array of ones shaped (lat,)
    ones_lat = xr.DataArray(np.ones(data.dims['lat']), dims=['lat'], coords={'lat': data.coords['lat']})
    local_solar_time_sin_da, _ = xr.broadcast(local_solar_time_sin_da, ones_lat)
    local_solar_time_cos_da, _ = xr.broadcast(local_solar_time_cos_da, ones_lat)

    # Assign to the dataset
    data['local_solar_time_sin'] = local_solar_time_sin_da
    data['local_solar_time_cos'] = local_solar_time_cos_da
        
    return data

def load_wofscast_data(paths, lead_times, task_config, client): 
    """Loads a large number of netcdf files into memory using dask.distributed.
    Useful storing the full dataset in CPU RAM and then offloading small subsets
    to the GPU RAM batch by batch. 
    
    paths: list of paths: Path to my custom wrfwof files 
    lead_times: slice of shortest to longest lead time in the wrfwof files
    task_config: graphcast.TaskConfig: An object containing useful variables for the input/target building
    client: dask.distributed.Client
    
    """
    # Load all the data into memory. 
    dataset = xr.open_mfdataset(paths, 
                                concat_dim='batch', 
                                parallel=True, 
                                combine='nested',
                                preprocess=add_local_solar_time
                                    ) 
    
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(dataset,
                                                        target_lead_times=lead_times,
                                                        **dataclasses.asdict(task_config))
        
    # Convert the constant fields to time-independent (drop time dim) and transpose as needed
    inputs = to_static_vars(inputs)
        
    # Perform computation efficiently with dask.
    with ProgressBar():
        inputs, targets, forcings = dask.compute(inputs, targets, forcings)
        
    inputs = inputs.transpose('batch', 'time', 'lat', 'lon', 'level')
    targets = targets.transpose('batch', 'time', 'lat', 'lon', 'level')
    forcings = forcings.transpose('batch', 'time', 'lat', 'lon')
            
    return inputs, targets, forcings 


def wofscast_batch_generator(inputs, targets, forcings, batch_size=32, n_timesteps=1, seed=123):
    """Batcher for an xarray dataset. Useful for storing the full dataset in CPU RAM and then offloading small subsets
    to the GPU RAM batch by batch. Assumes 'inputs' and 'targets' are xarray DataArrays or Datasets."""
    np.random.seed(seed)  # Set the seed for reproducibility
    
    total_samples = len(inputs.batch)
    total_batches = total_samples // batch_size + (1 if total_samples % batch_size > 0 else 0)
    
    indices = np.random.permutation(total_samples)  # Shuffle indices
    
    targets = targets.isel(time=slice(0, n_timesteps))  # Pre-select timesteps
    forcings = forcings.isel(time=slice(0, n_timesteps))
    
    for batch_num in range(total_batches):
        batch_indices = indices[batch_num * batch_size : min((batch_num + 1) * batch_size, total_samples)]
        
        batch_inputs = inputs.isel(batch=batch_indices)
        batch_targets = targets.isel(batch=batch_indices)
        batch_forcings = forcings.isel(batch=batch_indices)
        
        yield batch_inputs, batch_targets, batch_forcings
        
    
def check_for_nans(dataset):

    # Iterate through each variable in the Dataset
    for var_name, data_array in dataset.items():
        # Find boolean mask of NaNs
        nan_mask = data_array.isnull()
    
        # Use np.where to find the indices of NaNs
        nan_indices = np.where(nan_mask)
    
        # `nan_indices` is a tuple of arrays, each array corresponds to indices along one dimension
        # Print the locations of NaNs
        print(f"NaN locations in {var_name}:")
        for dim, inds in zip(nan_mask.dims, nan_indices):
            print(f"  {dim}: {inds}") 
    
def read_netcdfs_dask(paths, dim, transform_func=None):
    """Reading multiple netcdf files into memory, using dask for efficiency"""
    @delayed
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds
        
    #datasets = [process_one_path(p) for p in tqdm(paths, desc="Loading WRFOUT files")]
    
    delayed_datasets = [process_one_path(p) for p in paths]
    
    with ProgressBar():
        datasets = compute(*delayed_datasets)
    
    combined = xr.concat(datasets, dim)
    
    return combined
    
def read_netcdfs(paths, dim, transform_func=None):
    """Reading multiple netcdf files into memory, using dask for efficiency"""
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds
        
    datasets = [process_one_path(p) for p in tqdm(paths, desc="Loading WRFOUT files")]
    
    combined = xr.concat(datasets, dim)
    
    return combined
    
    
def wofscast_data_generator(file_paths, train_lead_times, task_config, chunk_size=2500):
    # Helper function to divide file_paths into chunks
    def chunked_file_paths(file_paths, chunk_size):
        for i in range(0, len(file_paths), chunk_size):
            yield file_paths[i:i + chunk_size]

    # Your data processing loop, modified to process chunks of file paths
    for file_path_chunk in chunked_file_paths(file_paths, chunk_size):
        #inputs, target, forcings = load_wofscast_data(file_path_chunk, train_lead_times, task_config, client)
        
        dataset_result = read_netcdfs(file_path_chunk, dim='batch', transform_func=add_local_solar_time)
        
        # Check for NaNs!!
        ###check_for_nans(dataset)
        
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            dataset_result,
            target_lead_times=train_lead_times,
            **dataclasses.asdict(task_config)
        )
        
        dataset_result.close() 
        del dataset_result
        gc.collect() 
        
        inputs = to_static_vars(inputs)
        
        inputs = inputs.transpose('batch', 'time', 'lat', 'lon', 'level')
        targets = targets.transpose('batch', 'time', 'lat', 'lon', 'level')
        forcings = forcings.transpose('batch', 'time', 'lat', 'lon')
      
        
        yield inputs, targets, forcings 
        
        
        
        