#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os 
sys.path.append('/home/monte.flora/python_packages/frdd-wofs-cast/')

from wofscast import data_utils
from wofscast import my_graphcast as graphcast

import xarray as xr
from glob import glob
import numpy as np
import dataclasses
import random 
import gc 

import dask 
from dask.diagnostics import ProgressBar
from dask.distributed import Client

DOMAIN_SIZE = 150
VARS_3D = ['U', 'V', 'W', 'T', 'GEOPOT']
VARS_2D = ['T2', 'COMPOSITE_REFL_10CM', 'UP_HELI_MAX', 'RAINNC']
STATIC_VARS = ['XLAND', 'HGT']

INPUT_VARS = VARS_3D + VARS_2D + STATIC_VARS
TARGET_VARS = VARS_3D + VARS_2D

FORCING_VARS = (
            "toa_incident_solar_radiation", # Only here to prevent the code from breaking!
            #"year_progress_sin",
            #"year_progress_cos",
            #"day_progress_sin",
            #"day_progress_cos",
        )
# Not pressure levels, but just vertical array indices at the moment. 
# When I created the wrfwof files, I pre-sampled every 2 levels. 
PRESSURE_LEVELS = np.arange(0,25)

# Loads data from the past 20 minutes (2 steps) and 
# creates a target over the next 10-60 min. 
INPUT_DURATION = '20min'
# 170 min is the max, but keeping it lower for testing the workflow
train_lead_times = slice('10min', '60min') 

TASK_WOFS = graphcast.TaskConfig(
      input_variables=INPUT_VARS,
      target_variables=TARGET_VARS,
      forcing_variables=FORCING_VARS,
      pressure_levels=PRESSURE_LEVELS,
      input_duration=INPUT_DURATION,
      n_vars_2D = len(VARS_2D),
      domain_size = DOMAIN_SIZE
  )


# In[3]:


def to_static_vars(dataset):
    # Select the first time index for 'HGT' and 'XLAND' variables
    hgt_selected = dataset['HGT'].isel(time=0).drop('time')
    xland_selected = dataset['XLAND'].isel(time=0).drop('time')

    # Now, replace the 'HGT' and 'XLAND' in the original dataset with these selected versions
    dataset = dataset.drop_vars(['HGT', 'XLAND'])
    dataset['HGT'] = hgt_selected
    dataset['XLAND'] = xland_selected

    return dataset

def wofscast_data_generator(paths, lead_times, task_config, client, batch_size=32, n_timesteps=1, seed=123):
    if seed is not None:
        random.seed(seed)
    random.shuffle(paths)
    
    total_batches = len(paths) // batch_size + (1 if len(paths) % batch_size > 0 else 0)

    for batch_num in range(total_batches):
        batch_paths = paths[batch_num*batch_size : (batch_num+1)*batch_size]
        
        # open_mfdataset already handles the dataset.close() for each file.
        dataset = xr.open_mfdataset(batch_paths, concat_dim='batch', 
                                    parallel=True, 
                                    combine='nested', 
                                    autoclose=True
                                    ) 
            
        inputs, targets = data_utils.extract_inputs_targets_forcings(dataset,
                                                        target_lead_times=lead_times,
                                                        **dataclasses.asdict(TASK_WOFS))
        
        # Convert the constant fields to time-independent (drop time dim) and transpose as needed
        inputs = to_static_vars(inputs).transpose('batch', 'time', 'lat', 'lon', 'level')
        targets = targets.isel(time=np.arange(n_timesteps)).transpose('batch', 'time', 'lat', 'lon', 'level')
        #forcings = forcings.isel(time=np.arange(n_timesteps)).transpose('batch', 'time', 'lat', 'lon')
        
        # Perform computation efficiently
        with ProgressBar():
            inputs, targets = dask.compute(inputs, targets)
        
        yield inputs, targets#, forcings

        
if __name__ == '__main__':
    client = Client()#dashboard_address=':5000')
    print(f"Dask Dashboard is available at {client.dashboard_link}")

        
    data_paths = glob(os.path.join('/work/mflora/wofs-cast-data/datasets/2021/wrf*.nc'))
    data_paths.sort()
    paths = data_paths[:12*4]

    i = 0
    for inputs, targets in wofscast_data_generator(paths, train_lead_times, TASK_WOFS, client, 
                                                         batch_size=12, n_timesteps=3):
        print(f'Epoch {i} Target Dims: ', targets.dims.mapping)
        i+=1 
   
    client.close()
    gc.collect()
    
    print(inputs) 





