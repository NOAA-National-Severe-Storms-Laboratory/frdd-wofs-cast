#!/usr/bin/env python
# coding: utf-8

# Compute the normalization statistics for the GraphCast code


""" usage: stdbuf -oL python -u compute_norm_stats.py --config dataset_5min_train_config.yaml > & log_compute_norm_stats & """
# Add the relative path to the wofscast package. 
import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from wofscast.utils import get_random_subset, load_yaml
from wofscast import data_utils
from wofscast import graphcast_lam as graphcast
from wofscast.data_generator import (add_local_solar_time, 
                                     to_static_vars, 
                                     load_chunk, 
                                     dataset_to_input,
                                     ZarrDataGenerator, 
                                     WRFZarrFileProcessor,
                                     WoFSDataProcessor
                                    )
from wofscast.wofscast_task_config import (DBZ_TASK_CONFIG, 
                                           WOFS_TASK_CONFIG, 
                                           WOFS_TASK_CONFIG_GC,
                                           DBZ_TASK_CONFIG_1HR,
                                           DBZ_TASK_CONFIG_FULL,
                                           WOFS_TASK_CONFIG_5MIN,
                                           WOFS_TASK_CONFIG_1HR
                                          )

import xarray as xr 
import numpy as np
from glob import glob
import time 
import random 
import dask 
import argparse 

from os.path import join
from concurrent.futures import ThreadPoolExecutor
from dask.diagnostics import ProgressBar


def compute_normalization_stats(paths, gpu_batch_size, save_path, 
                                compute_metrics=['mean', 'stddev', 'diffs_stddev'], 
                                preprocess_fn=None):

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        
        full_dataset = load_chunk(paths, gpu_batch_size, preprocess_fn) 
        
        full_dataset = full_dataset.chunk({'lat': 50, 
                                           'lon': 50, 
                                           'batch': gpu_batch_size})

        all_datasets = [full_dataset]
        
        if 'mean' in compute_metrics:
            mean_by_level = full_dataset.mean(dim=['time', 'lat', 'lon', 'batch'])
            all_datasets.append(mean_by_level)
            with ProgressBar():
                mean_by_level.to_netcdf(os.path.join(save_path, 'mean_by_level.nc'))
        
        if 'stddev' in compute_metrics:
            stddev_by_level = full_dataset.std(dim=['time', 'lat', 'lon', 'batch'], ddof=1)
            all_datasets.append(stddev_by_level)
            with ProgressBar():
                stddev_by_level.to_netcdf(os.path.join(save_path, 'stddev_by_level.nc'))
        
        if 'diffs_stddev' in compute_metrics:
            time_diffs = full_dataset.diff(dim='time')
            diffs_stddev_by_level = time_diffs.std(dim=['time', 'lat', 'lon', 'batch'], ddof=1)
            all_datasets.append(diffs_stddev_by_level)
            with ProgressBar():
                diffs_stddev_by_level.to_netcdf(os.path.join(save_path, 'diffs_stddev_by_level.nc'))

        # Close all datasets
        for ds in all_datasets:
            ds.close()
        
def get_files_for_year(year):
    year_path = join(base_path, year)
    with os.scandir(year_path) as it:
        return [join(year_path, entry.name) for entry in it if entry.is_dir() and entry.name.endswith('.zarr')]

if __name__ == "__main__": 
    
    # Config files are assumed to be stored in data_gen_configs/
    BASE_CONFIG_PATH = 'data_gen_configs'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config.yaml path')
    args = parser.parse_args()

    config_path = os.path.join(BASE_CONFIG_PATH, args.config)
    config_dict = load_yaml(config_path)
    
    base_path = config_dict['OUT_PATH']
    save_path = config_dict['OUT_NORM_PATH'] 
    n_samples_for_norm = config_dict['n_samples_for_norm']
    years = config_dict['years'] 
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True) 
    
    with ThreadPoolExecutor() as executor:
        paths = []
        for files in executor.map(get_files_for_year, years):
            paths.extend(files)
         
    random_paths = get_random_subset(paths, n_samples_for_norm)
    
    start_time = time.time() 
    # Configure Dask to use a single-threaded scheduler
    dask.config.set(scheduler='threads', num_workers=4)
      
    compute_normalization_stats(random_paths, 
                            gpu_batch_size=config_dict['batch_size'], 
                            save_path=save_path, 
                            compute_metrics=[
                                             #'mean', 
                                             #'stddev', 
                                             'diffs_stddev'
                                    ]
                           )

    end_time = time.time()
    time_to_run = end_time - start_time
    time_to_run_minutes = time_to_run / 60

    print(f'Time to Run: {time_to_run_minutes:.3f} mins')
    
''' Deprecated! 
    def preprocess_fn(dataset):
        _path = '/work/mflora/wofs-cast-data/datasets_zarr/2021/'
        latlon_path = os.path.join(_path, 'wrfwof_2021-05-15_040000_to_2021-05-15_043000__10min__ens_mem_09.zarr')
        fn = WoFSDataProcessor(latlon_path=latlon_path)
        
        dataset = fn(dataset)
        
        dataset = add_local_solar_time(dataset) 
        
        level_values = np.arange(dataset.dims['level'])
        dataset = dataset.assign_coords(level=("level", level_values))
        
        return dataset     
'''
    