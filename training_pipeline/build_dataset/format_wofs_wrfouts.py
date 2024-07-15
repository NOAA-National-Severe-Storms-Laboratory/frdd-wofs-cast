#!/usr/bin/env python
# coding: utf-8

# ## Format WoFS WRFOUTs into GraphCast-Friendly Format 
# 
# This notebook reads the raw WoFS WRFOUT files and formats them in xarray dataset formatted
# for the GraphCast code.
# 

""" usage: stdbuf -oL python -u format_wofs_wrfouts.py > & log_formatter & """

import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from wrfout_file_formatter import FileFormatter, filter_dates 

import numpy as np 

BASE_PATH = '/work2/wofs_zarr' #'/work2/wof/realtime/FCST/'

timestep_minutes = 60 # 60, 5
n_timesteps = 6 # 3, 24 
offset = 0 # 0, 60
duration_minutes = (n_timesteps)*timestep_minutes 

formatter = FileFormatter(n_jobs = 35, #35 for one timestep, 20 for multitimesteps
                          duration_minutes=duration_minutes, # 130 for 2 hours, .  
                          timestep_minutes=timestep_minutes, 
                          offset = offset, # Time in minutes after forecast initialization to start sampling. 
                          domain_size = 150, 
                          out_path = '/work/mflora/wofs-cast-data/datasets_2hr_hourly',
                          debug=False)

# Notes: 
# 15 hrs to process this configuration. 

# Only sample from top of the hour to potential sample 6 hrs of forecast. 
# Not sampling earlier in the data due to WoFS data quality.
init_times = ['1900', '1930', 
              '2000', '2030', 
              '2100', '2130', 
              '2200', '2230', 
              '2300', '2330',
              '0000', '0030',
              '0100', '0130',
              '0200', '0230',
              '0300']

# Subsample ensemble members to increase diversity in sample.
mems = np.arange(1, 18+1)

#init_times = ['2000', '0200', '0130']
#mems = [9, 12] 

process_multi_date = True

file_paths_set = []
if process_multi_date:
    all_dates = [] 
    years = ['2019', '2020', '2021']
    for year in years: 
        base_path = os.path.join(BASE_PATH, year)
        possible_dates = os.listdir(base_path)
    
        good_dates = filter_dates(possible_dates)#[:5]
        
        all_dates.extend(good_dates)
    
        all_file_paths = list(formatter.gen_file_paths(base_path, good_dates, init_times, mems))  
    
        file_paths_set.extend(all_file_paths)
        
    total_files = len(all_dates)*len(mems)*len(init_times)               
    print(f"Num of Samples: {total_files}")
    single_case=False
    
else:    
    year = '2021'
    good_dates = ['20210603']
    init_times = ['2200']
    mems = [9]
    base_path = os.path.join(BASE_PATH, year)
    
    all_file_paths = list(formatter.gen_file_paths(base_path, good_dates, init_times, mems))  
    print(f'{all_file_paths=}')
    file_paths_set.extend(all_file_paths)    
    single_case=True
    
ds = formatter.run(file_paths_set, single_case)
print('Finished!')

