#!/usr/bin/env python
# coding: utf-8

# ## Compute the normalization statistics for the GraphCast code

# In[1]:


import xarray as xr 
import numpy as np
from glob import glob

import random 
import os

import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from wofscast import graphcast_lam as graphcast
import dask 

from wofscast.data_generator import (add_local_solar_time, 
                                     to_static_vars, 
                                     load_chunk, 
                                     dataset_to_input,
                                     ZarrDataGenerator, 
                                     WRFZarrFileProcessor,
                                     WoFSDataProcessor
                                    )
from wofscast import data_utils
from wofscast.wofscast_task_config import (DBZ_TASK_CONFIG, 
                                           WOFS_TASK_CONFIG, 
                                           DBZ_TASK_CONFIG_1HR,
                                           DBZ_TASK_CONFIG_FULL
                                          )
from os.path import join


import random

def get_random_subset(input_list, subset_size, seed=123):
    """
    Get a random subset of a specified size from the input list.

    Parameters:
    -----------
    input_list : list
        The original list from which to draw the subset.
    subset_size : int
        The size of the subset to be drawn.
    seed : int, optional
        The seed for the random number generator. Default is None.

    Returns:
    --------
    list
        A random subset of the input list.
    """
    if subset_size > len(input_list):
        raise ValueError("subset_size must be less than or equal to the length of the input list")
    
    if seed is not None:
        random.seed(seed)

    return random.sample(input_list, subset_size)


from dask.diagnostics import ProgressBar

def compute_normalization_stats(paths, gpu_batch_size, task_config, save_path, 
                                batch_over_time=False, preprocess_fn=None): 

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        
        full_dataset = load_chunk(paths, batch_over_time, 
                                  gpu_batch_size, preprocess_fn) 

        full_dataset = full_dataset.chunk({'lat' : 50, 'lon' : 50, 'batch' : 128})
        
        # Setup computations using scattered data
        mean_by_level = full_dataset.mean(dim=['time', 'lat', 'lon', 'batch'])
        stddev_by_level = full_dataset.std(dim=['time', 'lat', 'lon', 'batch'], ddof=1)

        time_diffs = full_dataset.diff(dim='time')
        diffs_stddev_by_level = time_diffs.std(dim=['time', 'lat', 'lon', 'batch'], ddof=1)

        # Save results to NetCDF files (this triggers the computation)
        # Save results to NetCDF files (this triggers the computation)
        with ProgressBar():
            mean_by_level.to_netcdf(os.path.join(save_path, 'mean_by_level.nc'))
            stddev_by_level.to_netcdf(os.path.join(save_path, 'stddev_by_level.nc'))
            diffs_stddev_by_level.to_netcdf(os.path.join(save_path, 'diffs_stddev_by_level.nc'))

        # Close all datasets
        all_datasets = [full_dataset, mean_by_level, stddev_by_level, diffs_stddev_by_level]
        
        for ds in all_datasets:
            ds.close()

import os
from os.path import join
from concurrent.futures import ThreadPoolExecutor

base_path = '/work/mflora/wofs-cast-data/datasets_zarr'
years = ['2019', '2020']

def get_files_for_year(year):
    year_path = join(base_path, year)
    with os.scandir(year_path) as it:
        return [join(year_path, entry.name) for entry in it if entry.is_dir() and entry.name.endswith('.zarr')]
        #return [join(year_path, entry.name) for entry in it if entry.is_file()]
    
with ThreadPoolExecutor() as executor:
    paths = []
    for files in executor.map(get_files_for_year, years):
        paths.extend(files)

print(len(paths))

#random_paths = get_random_subset(paths, 4096)
# Save to NetCDF files
save_path = '/work/mflora/wofs-cast-data/full_normalization_stats/'

compute_normalization_stats(paths, 
                            gpu_batch_size=len(paths), 
                            task_config=WOFS_TASK_CONFIG, 
                            save_path=save_path)

'''
# ### Compute normalization statistics from DBZ_TASK_CONFIG_1HR
#%%time 
# Usage
base_path = '/work2/wofs_zarr/'
years = ['2019', '2020']
resolution_minutes = 10

# Specify the restrictions for testing
restricted_dates = None
restricted_times = ['1900', '2000', '2100', '2200', '2300', '0000', '0100', '0200', '0300']
restricted_members = ['ENS_MEM_1', 'ENS_MEM_12', 'ENS_MEM_17', 'ENS_MEM_5']#, 'ENS_MEM_10', 'ENS_MEM_11']

processor = WRFZarrFileProcessor(base_path, years, 
                             resolution_minutes, 
                             restricted_dates, 
                             restricted_times, restricted_members)

paths = processor.run()

random_paths = get_random_subset(paths, 6)

save_path = '/work/mflora/wofs-cast-data/normalization_stats_full_domain'

preprocessor = WoFSDataProcessor()

compute_normalization_stats(random_paths, gpu_batch_size=len(random_paths), 
                            task_config=DBZ_TASK_CONFIG_FULL, 
                            save_path=save_path, batch_over_time=True, 
                            preprocess_fn=preprocessor)
'''