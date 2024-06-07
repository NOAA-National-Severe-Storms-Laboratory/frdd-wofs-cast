#########################################
# Data Generator using Dask 
#########################################

from . import data_utils
from . import graphcast_lam as graphcast

import os
import xarray as xr
from glob import glob
import numpy as np
import pandas as pd
import dataclasses
import random 
import re 

from tqdm import tqdm
import dask
import gc 

#from numba import jit
import numpy as np
from datetime import datetime, timedelta
import math
import functools 

import fsspec
from jax import jit
import jax.numpy as jnp 

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from itertools import islice

# Set the Dask configuration to silence the performance warning
dask.config.set(**{'array.slicing.split_large_chunks': False})

def to_static_vars(dataset):
    # Select the first time index for 'HGT' and 'XLAND' variables
    hgt_selected = dataset['HGT'].isel(time=0).drop('time')
    xland_selected = dataset['XLAND'].isel(time=0).drop('time')

    # Now, replace the 'HGT' and 'XLAND' in the original dataset with these selected versions
    dataset = dataset.drop_vars(['HGT', 'XLAND'])
    dataset['HGT'] = hgt_selected
    dataset['XLAND'] = xland_selected

    return dataset

def extract_datetime_from_path(zarr_path):
    # Example string
    filename = os.path.basename(zarr_path)

    # Regular expression to match the datetime part
    datetime_pattern = r"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}"

    # Use re.search to find the pattern in the string
    match = re.search(datetime_pattern, filename)

    # Extract the matched part
    if match:
        datetime_str = match.group(0)
        datetime_str = datetime_str.replace("_", " ")
    else:
        raise ValueError(f'datetime_pattern {datetime_pattern} is not found in {zarr_path}')
        
    return datetime_str

def open_zarr(zarr_path):
    return xr.open_dataset(zarr_path, 
                           engine='zarr', consolidated=True, 
                           chunks={}, decode_times=False)

def load_dataset_from_zarr(path):
    """Load a dataset from a Zarr file"""
    if path.endswith('.json'):
        path = fsspec.get_mapper('reference://', fo=path, remote_protocol='file')

    ds = open_zarr(path) 
    
    # Check if the datetime coordinate exists; need for the local solar time calculation. 
    if 'datetime' not in ds.coords:
        # Extract the datetime from the path
        datetime_str = extract_datetime_from_path(path)
        datetime = pd.to_datetime(datetime_str)
        ds = ds.assign_coords(datetime=[datetime])

    ds = add_local_solar_time(ds)
    
    return ds

def load_and_concatenate(files, concat_dim):
    """Load multiple datasets and concatenate them along a specified dimension."""
    datasets = [load_dataset_from_zarr(file) for file in files]
    combined_dataset = xr.concat(datasets, dim=concat_dim)
    
    if concat_dim != 'batch': 
        # Add timing coordinates. Setting the time coordinate after time 
        # in nanoseconds since the first time entry. Neccesary 
        # for the inputs, targets, forcings extraction code. 
        try:
            combined_dataset = combined_dataset.rename({'Time': 'time'})
        except:
            print('Did not convert time dimension name.')  
        
        combined_dataset['time'] = combined_dataset['datetime'].values
        time_deltas = (combined_dataset['time'] - combined_dataset['time'][0]).astype('timedelta64[ns]')
        combined_dataset['time'] = time_deltas
        
        combined_dataset = combined_dataset.drop_vars('datetime', errors='ignore')
       
    return combined_dataset

def load_chunk(paths_chunk, batch_over_time=False, gpu_batch_size=32, preprocess_fn=None):
    if batch_over_time:
        if not isinstance(paths_chunk, list) or not all(isinstance(i, list) for i in paths_chunk):
            raise ValueError('paths must be a nested list if concatenating along a time dimension.')
        datasets_per_time = [load_and_concatenate(p, concat_dim='Time') for p in paths_chunk]
        
        # Apply preprocessing to the individual datasets. 
        if preprocess_fn:
            datasets_per_time = [preprocess_fn(ds) for ds in datasets_per_time]
        
        dataset = xr.concat(datasets_per_time, dim='batch')  
    else:
        dataset = load_and_concatenate(paths_chunk, concat_dim='batch')

        # Apply preprocessing to the dataset. 
        if preprocess_fn:
            dataset = preprocess_fn(dataset)     
        
    # Chunk the dataset
    dataset = dataset.chunk({'batch': gpu_batch_size})
    
    return dataset

def dataset_to_input(dataset, task_config, target_lead_times=None, 
                     batch_over_time=False, n_target_steps=1):
    
    DIMS = ('batch', 'time', 'lat', 'lon', 'level')
    
    if target_lead_times is None:
        target_lead_times = task_config.train_lead_times
    
    if batch_over_time:
        inputs, targets, forcings = data_utils.batch_extract_inputs_targets_forcings(
            dataset, 
            n_input_steps=2, 
            n_target_steps=n_target_steps, 
            target_lead_times=target_lead_times,
            **dataclasses.asdict(task_config)
        )
    else:
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                    dataset,
                    target_lead_times=target_lead_times,
                    **dataclasses.asdict(task_config)
        )
    
    if len(inputs.time) == 0:
        raise IndexError('target_lead_times is too long for dataset and inputs are empty!')
    
    inputs = to_static_vars(inputs)
        
    inputs = inputs.transpose(*DIMS, missing_dims='ignore')
    targets = targets.transpose(*DIMS, missing_dims='ignore')
    forcings = forcings.transpose(*DIMS, missing_dims='ignore')
    
    return inputs, targets, forcings

class ZarrDataGenerator:
    """
    A generator class to load and preprocess data from Zarr files for machine learning tasks.

    Parameters:
    -----------
    task_config : graphcast_lam.TaskConfig object
        Configuration dictionary for the task, containing parameters required for data processing.
    target_lead_times : list, optional
        List of lead times for the target variable. Default is None.
        If default, the target_lead_times in task_config is used. See the dataset_to_input function
        for more details. 
    n_target_steps : int, optional
        Number of target steps to predict. Default is 1. Used in the batch_extract_inputs_targets_forcings 
        funtion in the data_utils.py. 
    batch_over_time : bool, optional
        If True, batch data over time. Default is False. When providing a nested list of paths, where
        the inner nests are datasets to be concatenated over time, set to True. 
    preprocess_fn : function, optional
        A function to preprocess the data. Default is None. Useful option for processing 
        data after the concatenation over time. 
    cpu_batch_size : int, optional
        Batch size for loading data in parallel on CPU. Default is None. In which, 
        it is 2*gpu_batch_size.
    gpu_batch_size : int, optional
        Batch size for processing data on GPU. Default is 32.
    n_workers : int, optional
        Number of parallel workers for data loading. Default is 8.
    """
    
    def __init__(self, task_config, 
                 target_lead_times=None, 
                 n_target_steps : int =1, 
                 batch_over_time : bool =False,
                 preprocess_fn=None, 
                 cpu_batch_size=None, gpu_batch_size : int =32, 
                 n_workers : int = 8):
        
        self.task_config = task_config
        if cpu_batch_size:
            self.cpu_batch_size = cpu_batch_size
        else:
            self.cpu_batch_size = 2 * gpu_batch_size
            
        self.gpu_batch_size = gpu_batch_size 
        self.n_workers = n_workers 
        self.target_lead_times = target_lead_times
        self.batch_over_time = batch_over_time 
        self.n_target_steps = n_target_steps 
        self.preprocess_fn = preprocess_fn

    def __call__(self, paths):
        """
        Generates batches of data from the provided paths.

        Parameters:
        -----------
        paths : list
            List of file paths to the Zarr files.

        Yields:
        -------
        tuple
            A tuple containing inputs, targets, and forcings for each batch.
        """
        outer_start = 0
        np.random.shuffle(paths)
        
        chunk_futures = []
        loaded_chunks = []

        def with_params(fn, batch_over_time, gpu_batch_size, preprocess_fn):
            return functools.partial(fn, batch_over_time=batch_over_time, 
                                     gpu_batch_size=gpu_batch_size, 
                                     preprocess_fn=preprocess_fn)
        
        load_chunk_ = with_params(load_chunk, self.batch_over_time, 
                                  self.gpu_batch_size, self.preprocess_fn)

        # Using ProcessPoolExecutor for parallel loading
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            while outer_start < len(paths):
                outer_end = min(outer_start + self.cpu_batch_size, len(paths))
                paths_chunk = paths[outer_start:outer_end]
                chunk_futures.append(executor.submit(load_chunk_, paths_chunk))
                outer_start = outer_end

            for future in as_completed(chunk_futures):
                chunk = future.result()
                loaded_chunks.append(chunk)

        # Sequentially process the loaded chunks
        for chunk in loaded_chunks:
            total_samples = chunk.sizes['batch']
            chunk_indices = np.arange(total_samples)

            inner_start = 0
            while inner_start < len(chunk_indices):
                inner_end = min(inner_start + self.gpu_batch_size, len(chunk_indices))
                batch = chunk.isel(batch=slice(inner_start, inner_end))
                inputs, targets, forcings = dataset_to_input(batch, self.task_config, 
                                                             target_lead_times=self.target_lead_times,
                                                             batch_over_time=self.batch_over_time,
                                                             n_target_steps=self.n_target_steps 
                                                            )
                inputs, targets, forcings = dask.compute(inputs, targets, forcings, scheduler='threads')
                yield inputs, targets, forcings
                inner_start = inner_end

                
class TOARadiationFlux:
    def __init__(self):
        # Solar constant in W/m^2
        self.S0 = 1361

    @staticmethod
    @jit
    def calculate_solar_declination(day_of_year):
        """Calculate solar declination as a function of day of the year."""
        return 23.45 * jnp.sin(jnp.radians((360 / 365) * (day_of_year - 81)))

    @staticmethod
    @jit#(nopython=True)
    def calculate_hour_angle(utc_hour, longitude):
        """Calculate solar hour angle based on UTC time."""
        # Convert longitude to equivalent time (1 hour per 15 degrees)
        longitude_time = longitude / 15.0
        
        # Calculate solar time from UTC time and longitude
        solar_time = utc_hour + longitude_time
        
        # Hour angle, considering each hour is 15 degrees of rotation
        hour_angle = (solar_time - 12) * 15
        return hour_angle

    @staticmethod
    @jit
    def calculate_solar_zenith_angle(latitude, declination, hour_angle):
        """Calculate solar zenith angle."""
        latitude_rad = jnp.radians(latitude)
        declination_rad = jnp.radians(declination)
        hour_angle_rad = jnp.radians(hour_angle)

        cos_zenith = jnp.sin(latitude_rad) * jnp.sin(declination_rad) + \
                     jnp.cos(latitude_rad) * jnp.cos(declination_rad) * jnp.cos(hour_angle_rad)
        zenith_angle = jnp.degrees(jnp.arccos(cos_zenith))
        return zenith_angle
    
        
    def calculate_flux(self, date_times, lat_grid, lon_grid):
        NT = len(date_times)
        NY, NX = lat_grid.shape
        flux = jnp.zeros((NT, NY, NX), dtype=jnp.float32)

        # Loop through each datetime, calculate declination once per datetime
        for i, datetime_obj in enumerate(date_times):
            day_of_year = datetime_obj.timetuple().tm_yday
            declination = self.calculate_solar_declination(day_of_year)

            # Vectorized computation over latitude and longitude grids
            for j in range(NY):
                hour_angle = self.calculate_hour_angle(datetime_obj.hour + datetime_obj.minute / 60, 
                                                       lon_grid[j])
                zenith_angle = self.calculate_solar_zenith_angle(lat_grid[j], declination, hour_angle)
                
                # Calculate radiation flux
                flux = flux.at[i, j, :].set(
                    jnp.where(zenith_angle < 90, self.S0 * jnp.cos(jnp.radians(zenith_angle)), 0))
                
        return flux

def check_datetime_dtype(dataset, coord='datetime'):
    if np.issubdtype(dataset[coord].dtype, np.int64):
        dataset = xr.decode_cf(dataset)
        
    return dataset 

def add_local_solar_time(data: xr.Dataset) -> xr.Dataset:
    """    
    Adds sine and cosine-transformed local solar time variables to the dataset,
    adjusted for longitude, and replicated across latitude. Also adds 
    TOA (top-of-the-atmosphere) radiation. These variables are used as forcing
    inputs (known in the future; beyond the initial conditions) for the AI-NWP. 

    Args:
        data: The input dataset with 'time', 'lat', and 'lon' dimensions.

    Returns:
        xr.Dataset: The dataset with 'local_solar_time_sin' and 'local_solar_time_cos' variables added.
    """
    # Create an instance of TOARadiationFlux
    toa_radiation = TOARadiationFlux()
    
    time_dim = 'time'
    if {'Time'}.issubset(data.dims):
        time_dim = 'Time'
    
    if not {time_dim, 'lat', 'lon'}.issubset(data.dims):
        missing_dims = {time_dim, 'lat', 'lon'} - set(data.dims)
        raise ValueError(f"Missing dimensions in the dataset: {missing_dims}")

    data = check_datetime_dtype(data)
    
    # Calculate the local solar time adjustment
    local_hours = (data.coords['datetime'].dt.hour + data.coords['lon'] / 15.0) % 24

    # Convert local_hours to radians for sine and cosine
    radians = (local_hours * 2 * np.pi) / 24

    # Calculate sine and cosine for the local solar time
    local_solar_time_sin = np.sin(radians)
    local_solar_time_cos = np.cos(radians)

    # Create DataArrays with 'time' and 'lon' dimensions
    local_solar_time_sin_da = xr.DataArray(local_solar_time_sin, dims=(time_dim, 'lon'),
                                           coords={time_dim: data.coords[time_dim], 'lon': data.coords['lon']})
    local_solar_time_cos_da = xr.DataArray(local_solar_time_cos, dims=(time_dim, 'lon'),
                                           coords={time_dim: data.coords[time_dim], 'lon': data.coords['lon']})

    # Replicate values across 'lat' dimension by broadcasting with an array of ones shaped (lat,)
    ones_lat = xr.DataArray(np.ones(data.dims['lat']), dims=['lat'], coords={'lat': data.coords['lat']})
    local_solar_time_sin_da, _ = xr.broadcast(local_solar_time_sin_da, ones_lat)
    local_solar_time_cos_da, _ = xr.broadcast(local_solar_time_cos_da, ones_lat)

    # Assign to the dataset
    data['local_solar_time_sin'] = local_solar_time_sin_da.astype('float32')
    data['local_solar_time_cos'] = local_solar_time_cos_da.astype('float32')
    
    
    # Add TOA (top-of-the-atmo) radiation 
    # Calculate the TOA radiation flux for the defined dates, times, and grid
    '''
    lat_grid, lon_grid = np.meshgrid(data.lat, data.lon)  # Create 2D grid
    flux = toa_radiation.calculate_flux(pd.to_datetime(data.datetime.values), 
                                        lat_grid, lon_grid)
    
    data['toa_radiation'] = xr.DataArray(flux, dims=(time_dim, 'lat', 'lon'),
                                           coords={time_dim: data.coords[time_dim], 
                                                   'lat': data.coords['lat'],
                                                   'lon': data.coords['lon']})
    '''
    return data

    
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
    

    
class WoFSDataProcessor:
    STATIC_PATH_FOR_LATLON = '/work2/wofs_zarr/2019/20190525/2000/ENS_MEM_12/wrfwof_d01_2019-05-25_20:00:00.zarr'
    
    def __init__(self, domain_size=150):
        self.domain_size = domain_size 
        self._load_lat_and_lon()

    def _load_lat_and_lon(self):
        # Reset the lat/lon coordinates 
        tmp_ds = open_zarr(self.STATIC_PATH_FOR_LATLON) 
        # Latitude and longitude are expected to be 1d vectors. 
        self.lat_1d = tmp_ds['lat'].values
        self.lon_1d = tmp_ds['lon'].values
        
        tmp_ds.close()
    
    def set_ref_lat_and_lon(self, dataset: xr.Dataset) -> xr.Dataset:
        # Assign a reference 1D latitude and longitude for the coordiantes. 
        dataset = dataset.assign_coords(lat=self.lat_1d, lon=self.lon_1d)
        return dataset 
    
    def subset_vertical_levels(self, dataset: xr.Dataset) -> xr.Dataset:
        # Subset the vertical levels (every N layers). 
        dataset = dataset.isel(level=dataset.level[::3].values)
        return dataset 
    
    def unaccum_rainfall(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Calculate the difference in accumulated rainfall ('RAINNC') at each time step,
        with an assumption that the first time step starts with zero rainfall.
    
        Parameters:
        - ds: xarray.Dataset containing the 'RAINNC' variable
    
        Returns:
            - Modified xarray.Dataset with the new variable 'RAINNC_DIFF'
        """
        if 'RAINNC' not in dataset.data_vars:
            return dataset 
        
        # Calculate the difference along the time dimension
        rain_diff = dataset['RAINNC'].diff(dim='time')
    
        # Prepend a zero for the first time step. This assumes that the difference
        # for the first time step is zero since there's no previous time step to compare.
        # We use np.concatenate to add the zero at the beginning. Ensure that the dimensions match.
        # Adjust dimensions and coordinates according to your dataset's specific setup.
        initial_zero = xr.zeros_like(dataset['RAINNC'].isel(time=0))
        rain_diff_with_initial = xr.concat([initial_zero, rain_diff], dim='time')
    
        # Add the computed difference back to the dataset as a new variable
        dataset['RAIN_AMOUNT'] = rain_diff_with_initial
    
        dataset = dataset.drop_vars(['RAINNC'])
        
        return dataset     
    
    def resize(self, dataset: xr.Dataset) -> xr.Dataset:
        """Resize the domain"""
        n_lat, n_lon = dataset.dims['lat'], dataset.dims['lon']
        
        start_lat, start_lon = (n_lat - self.domain_size) // 2, (n_lon - self.domain_size) // 2
        end_lat, end_lon = start_lat + self.domain_size, start_lon + self.domain_size
        
        # Subsetting the dataset to the central size x size grid
        ds_subset = dataset.isel(lat=slice(start_lat, end_lat), lon=slice(start_lon, end_lon))
        
        return ds_subset
    
    
    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        
        funcs = [
            'set_ref_lat_and_lon', # Set a single reference lat/lon grid 
            #'resize', # Limit to the inner most 150 x 150 
            #'subset_vertical_levels', # Limit to every 3rd level.
            'unaccum_rainfall' # Convert accum rainfall to rain rate. 
        ]
    
        for func in funcs:
            dataset = getattr(self, func)(dataset)

        return dataset 
    
class WRFZarrFileProcessor:
    def __init__(self, base_path, years, resolution_minutes, 
                 restricted_dates=None, restricted_times=None, restricted_members=None):
        self.base_path = base_path
        self.years = years
        self.resolution_minutes = resolution_minutes
        self.restricted_dates = restricted_dates
        self.restricted_times = restricted_times
        self.restricted_members = restricted_members

    def get_nwp_files_at_resolution(self, directory):
        """
        Get file paths at a specified time resolution from a directory of WRFOUT zarr files.

        Args:
            directory (str): Path to the directory containing NWP output files.

        Returns:
            list: List of file paths at the specified time resolution.
        """
        files = sorted(os.listdir(directory))
        selected_files = []
        for file in files:
            try:
                timestamp_str = file.split('_')[2] + '_' + file.split('_')[3].replace('.zarr', '')
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d_%H:%M:%S')
            except ValueError:
                continue
            if (timestamp.minute * 60 + timestamp.second) % (self.resolution_minutes * 60) == 0:
                selected_files.append(os.path.join(directory, file))
        return selected_files

    def process_member_directory(self, mem_path):
        return self.get_nwp_files_at_resolution(mem_path)

    def process_datetime_directory(self, datetime_path):
        members_files = []
        with os.scandir(datetime_path) as members:
            for mem_entry in members:
                if mem_entry.is_dir() and (self.restricted_members is None or mem_entry.name in self.restricted_members):
                    mem_path = mem_entry.path
                    members_files.append(self.process_member_directory(mem_path))
        return members_files

    def process_date_directory(self, date_path):
        times_files = []
        with os.scandir(date_path) as init_times:
            for time_entry in init_times:
                if time_entry.is_dir() and (self.restricted_times is None or time_entry.name in self.restricted_times):
                    datetime_path = time_entry.path
                    times_files.extend(self.process_datetime_directory(datetime_path))
        return times_files

    def process_year_directory(self, year_path):
        dates_files = []
        with os.scandir(year_path) as dates:
            for date_entry in dates:
                if date_entry.is_dir() and (self.restricted_dates is None or date_entry.name in self.restricted_dates):
                    date_path = date_entry.path
                    dates_files.extend(self.process_date_directory(date_path))
        return dates_files

    def run(self):
        all_files = []
        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = []
            for year in self.years:
                year_path = os.path.join(self.base_path, year)
                futures.append(executor.submit(self.process_year_directory, year_path))
            for future in futures:
                all_files.extend(future.result())
        return all_files



        
        
        