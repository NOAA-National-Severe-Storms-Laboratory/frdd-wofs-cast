#Format WoFS WRFOUTs (netcdf or zarr) into GraphCast-Friendly Format 

# Add --debug to run a single case! 

""" usage: stdbuf -oL python -u format_wofs_wrfouts.py --config dataset_10min_test_full_domain_config.yaml  > & log_formatter & """

import sys, os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))
from wrfout_file_formatter import FileFormatter, filter_dates 
from wofscast.utils import load_yaml
import numpy as np 
import time
import argparse 
import yaml 

# Config files are assumed to be stored in data_gen_configs/
BASE_CONFIG_PATH = 'data_gen_configs'

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='config.yaml path')
parser.add_argument('--debug', action='store_true', help='whether to process a single date.')
args = parser.parse_args()

print(f'{args.debug=}')

config_path = os.path.join(BASE_CONFIG_PATH, args.config)
config_dict = load_yaml(config_path)

# Set the config parameters. 
BASE_PATH = config_dict['BASE_PATH']
OUT_PATH = config_dict['OUT_PATH']
timestep_minutes = config_dict['timestep_minutes']
n_timesteps = config_dict['n_timesteps']
offset = config_dict['offset']

years = config_dict['years']
mems = config_dict['mems']
init_times = config_dict['init_times']
n_jobs = config_dict.get('n_jobs', 35)
resize = config_dict.get('resize', True)
subset_vertical_levels = config_dict.get('subset_vertical_levels', True)
                              
duration_minutes = n_timesteps*timestep_minutes 

vars_to_keep = config_dict['VARS_TO_KEEP']

process_multi_date = False if args.debug else True
do_drop_vars=True
legacy = True 
overwrite=False

processes = [] 
if resize:
    processes.append('resize')
    
if subset_vertical_levels:
    processes.append('subset_vertical_levels')
    
subset_dates = config_dict.get('subset_dates', False)
   

formatter = FileFormatter(n_jobs = n_jobs, #35 for one timestep, 20 for multitimesteps
                          duration_minutes=duration_minutes, 
                          timestep_minutes=timestep_minutes, 
                          offset = offset, # Time in minutes after forecast initialization to start sampling. 
                          domain_size = 150, 
                          out_path = OUT_PATH,
                          debug=False, 
                          overwrite=overwrite, 
                          legacy=legacy,
                          do_drop_vars=do_drop_vars,
                          vars_to_keep=vars_to_keep,
                          processes = processes
                         )


file_paths_set = []
if process_multi_date:
    all_dates = [] 
    for year in years: 
        base_path = os.path.join(BASE_PATH, year)
        possible_dates = os.listdir(base_path)
        
        good_dates = filter_dates(possible_dates)#[:5]
        if subset_dates:
            good_dates = good_dates[:2] 
        
        all_dates.extend(good_dates)
    
        all_file_paths = list(formatter.gen_file_paths(base_path, good_dates, init_times, mems))  
    
        file_paths_set.extend(all_file_paths)
        
    total_files = len(all_dates)*len(mems)*len(init_times)               
    print(f"Num of Samples: {total_files}")
    single_case=False
    
else:    
    year = '2019'
    good_dates = ['20190513']
    init_times = ['0200']
    mems = [9]
    base_path = os.path.join(BASE_PATH, year)
    
    all_file_paths = list(formatter.gen_file_paths(base_path, good_dates, init_times, mems))  
    print(f'{all_file_paths=}')
    file_paths_set.extend(all_file_paths)    
    single_case=True

start_time = time.time()

ds = formatter.run(file_paths_set, single_case)

print(f'Finished! Time elapsed: {time.time()-start_time:.3f} secs')



