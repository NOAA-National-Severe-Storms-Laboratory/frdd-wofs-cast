#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from wofscast.model import WoFSCastModel
from wofscast.border_mask import BORDER_MASK_NUMPY

from wofscast.data_generator import (load_chunk, 
                                     WRFZarrFileProcessor,
                                     WoFSDataProcessor, 
                                     dataset_to_input,
                                     ZarrDataGenerator,
                                     add_local_solar_time,
                                    )
from wofscast import checkpoint
from wofscast.wofscast_task_config import (DBZ_TASK_CONFIG, 
                                           WOFS_TASK_CONFIG, 
                                           DBZ_TASK_CONFIG_1HR,
                                           DBZ_TASK_CONFIG_FULL
                                          )

from wofscast.model_utils import dataset_to_stacked, variable_to_stacked
from wofscast.normalization import normalize 

import os
from os.path import join
from concurrent.futures import ThreadPoolExecutor
import random
import numpy as np
import xarray as xr
from tqdm import tqdm 

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time 

from typing import Optional, Mapping, Tuple, List

def get_files_for_year(year):
    """Get all zarr files within a directory."""
    year_path = join(base_path, year)
    with os.scandir(year_path) as it:
        return [join(year_path, entry.name) for entry in it if entry.is_dir() and entry.name.endswith('.zarr')] 

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

def pad_to_multiple_of_16(tensor):
    _, _, h, w = tensor.size()
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded_tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    return padded_tensor

import torch
from typing import Optional, Mapping, Tuple, List
from wofscast.model_utils import variable_to_stacked

def create_indices_array(dataset: xr.Dataset, 
                         sizes: Optional[Mapping[str, int]] = None, 
                         preserved_dims: Tuple[str, ...] = ('time', 'batch', 'lat', 'lon'),
                        ) -> torch.Tensor:
    """Creates an indices array for the variables in the dataset."""
    indices = []
    current_index = 0

    for i, name in enumerate(sorted(dataset.data_vars.keys())):
        variable = dataset.variables[name]
        stack_to_channels_dims = [d for d in variable.dims if d not in preserved_dims]
        if stack_to_channels_dims:
            variable = variable.stack(channels=stack_to_channels_dims)
        stacked_size = variable.sizes.get("channels", 1)
        indices.extend([i] * stacked_size)
        current_index += stacked_size

    return torch.tensor(indices)


if __name__ == "__main__":
    # Main script. 
    
    """ usage: stdbuf -oL python -u save_wofscast_to_torch_tensors.py > & log_torch & """
    
    task_config = WOFS_TASK_CONFIG
    OUT_BASE_PATH = '/work/mflora/wofs-cast-data/predictions'
    dataset_path = os.path.join(OUT_BASE_PATH, 'wofscast_normalized_with_residual_160_samples.pt')
    
    base_path = '/work/mflora/wofs-cast-data/datasets_zarr/'
    
    # Corey's best model so far. 
    MODEL_PATH = '/work/cpotvin/WOFSCAST/model/wofscast_test_v178.npz'
    norm_stats_path = '/work/mflora/wofs-cast-data/full_normalization_stats'
    
    model = WoFSCastModel(norm_stats_path = norm_stats_path)
    model.load_model(MODEL_PATH)
    
    years = ['2019', '2020']
    with ThreadPoolExecutor() as executor:
        paths = []
        for files in executor.map(get_files_for_year, years):
            paths.extend(files)
    
    n_batches = 160 #10000
    paths = get_random_subset(paths, n_batches, seed=42)

    predicted_list = []
    truth_list = []

    start_time = time.time()
    
    norm_stats_path = '/work/mflora/wofs-cast-data/full_normalization_stats'
    mean_by_level = xr.load_dataset(os.path.join(norm_stats_path, 'mean_by_level.nc'))
    stddev_by_level = xr.load_dataset(os.path.join(norm_stats_path, 'stddev_by_level.nc'))
    diffs_stddev_by_level = xr.load_dataset(os.path.join(norm_stats_path, 'diffs_stddev_by_level.nc'))
  
    save_channel_indices = True
    
    start_time = time.time()
    for path in tqdm(paths): 
        dataset = load_chunk([path], 1, add_local_solar_time)
        dataset = dataset.compute() 
        
        inputs, targets, forcings = dataset_to_input(dataset, task_config)
    
        predictions = model.predict(inputs, targets, forcings)
        if save_channel_indices:
            output_path = dataset_path.replace('wofscast_dataset', 'channel_indices')

            channel_indices = create_indices_array(predictions)
            torch.save({'channel_indices' : channel_indices }, 
                   output_path   
              )
            save_channel_indices=False
        
        # Compute the target residual (WoFS - WoFSCast) 
        target_residual = targets - predictions

        # Normalize the predictions and target fields 
        predictions_scaled = normalize(predictions, stddev_by_level, mean_by_level)
        targets_residual_scaled = normalize(target_residual, diffs_stddev_by_level, locations=None)
    
        pred_stacked = dataset_to_stacked(predictions_scaled).transpose('batch', 'channels', 'lat', 'lon').values 
        # (batch, channels, lat, lon)
        truth_stacked = dataset_to_stacked(targets_residual_scaled).transpose('batch', 'channels', 'lat', 'lon').values
    
        # Convert to tensors
        pred_stacked_tensor = torch.tensor(pred_stacked, dtype=torch.float32)
        truth_stacked_tensor = torch.tensor(truth_stacked, dtype=torch.float32)
    
        # Apply padding
        pred_stacked_tensor = pad_to_multiple_of_16(pred_stacked_tensor)
        truth_stacked_tensor = pad_to_multiple_of_16(truth_stacked_tensor)

        predicted_list.append(pred_stacked_tensor)
        truth_list.append(truth_stacked_tensor)

    # Concatenate all numpy arrays and convert to torch tensors
    predicted_all = torch.cat(predicted_list, dim=0)
    truth_all = torch.cat(truth_list, dim=0)

    print(f'{predicted_all.shape=}')
    print(f'{truth_all.shape=}')
    
    # Save the dataset
    torch.save({
        'input_images': predicted_all, # WoFSCast
        'target_images': truth_all, # Residual of WoFS and WoFSCast 
        'metadata': None  # Add metadata if available
    }, dataset_path)

    end_time = time.time()
    
    print(f"Total time : {(end_time - start_time) / 60:.2f} minutes")
    
    print(f"Dataset saved to {dataset_path}")

