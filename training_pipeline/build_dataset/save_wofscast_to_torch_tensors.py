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
                                     ZarrDataGenerator
                                    )
from wofscast import checkpoint
from wofscast.wofscast_task_config import (DBZ_TASK_CONFIG, 
                                           WOFS_TASK_CONFIG, 
                                           DBZ_TASK_CONFIG_1HR,
                                           DBZ_TASK_CONFIG_FULL
                                          )


import os
from os.path import join
from concurrent.futures import ThreadPoolExecutor
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import time 


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
    padded_tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))
    return padded_tensor


if __name__ == "__main__":
    # Main script. 
    
    """ usage: stdbuf -oL python -u save_wofscast_to_torch_tensors.py > & log_torch & """
    

    OUT_BASE_PATH = '/work/mflora/wofs-cast-data/predictions'
    
    base_path = '/work/mflora/wofs-cast-data/datasets_zarr/'
    MODEL_PATH = '/work/mflora/wofs-cast-data/model/wofscast_baseline.npz'
    norm_stats_path = '/work/mflora/wofs-cast-data/full_normalization_stats'
    
    model = WoFSCastModel(norm_stats_path = norm_stats_path)
    model.load_model(MODEL_PATH)
    
    years = ['2021'] #['2019', '2020']
    with ThreadPoolExecutor() as executor:
        paths = []
        for files in executor.map(get_files_for_year, years):
            paths.extend(files)
    
    n_batches = 4 #4096
    gpu_batch_size = 32 
    paths = get_random_subset(paths, n_batches, seed=42)

    generator = ZarrDataGenerator(WOFS_TASK_CONFIG, 
                              cpu_batch_size=2*gpu_batch_size, 
                              gpu_batch_size=gpu_batch_size, n_workers=24)

    gen = generator(paths)

    predicted_refl_list = []
    truth_refl_list = []

    start_time = time.time()
    
    for j, (inputs, targets, forcings) in enumerate(gen):                    
        predictions = model.predict(inputs, targets, forcings)
  
        predicted_refl = predictions['COMPOSITE_REFL_10CM'].values.squeeze() # shape (gpu_batch_size, ny, nx)
        truth_refl = targets['COMPOSITE_REFL_10CM'].values.squeeze() # shape (gpu_batch_size, ny, nx)
    
        # Add channel dimension
        predicted_refl = predicted_refl[:, np.newaxis, :, :]
        truth_refl = truth_refl[:, np.newaxis, :, :]
    
        # Convert to tensors
        predicted_refl_tensor = torch.tensor(predicted_refl, dtype=torch.float32)
        truth_refl_tensor = torch.tensor(truth_refl, dtype=torch.float32)
    
        # Apply padding
        predicted_refl_tensor = pad_to_multiple_of_16(predicted_refl_tensor)
        truth_refl_tensor = pad_to_multiple_of_16(truth_refl_tensor)

        predicted_refl_list.append(predicted_refl_tensor)
        truth_refl_list.append(truth_refl_tensor)

    # Concatenate all numpy arrays and convert to torch tensors
    predicted_refl_all = torch.cat(predicted_refl_list, dim=0)
    truth_refl_all = torch.cat(truth_refl_list, dim=0)

    # Create the ConditionalGOES16_Nowcast compatible dataset
    conditional_images = predicted_refl_all  # Use predicted reflectivity as conditional images
    next_image = truth_refl_all  # Use true reflectivity as next images

    # Save the dataset
    dataset_path = '/work/mflora/wofs-cast-data/predictions/wofscast_test_dataset.pt'
    torch.save({
        'next_image': next_image,
        'conditional_images': conditional_images,
        'metadata': None  # Add metadata if available
    }, dataset_path)

    end_time = time.time()
    
    print(f"Total time : {(end_time - start_time) / 60:.2f} minutes")
    
    print(f"Dataset saved to {dataset_path}")

