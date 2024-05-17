#!/usr/bin/env python
# coding: utf-8

# ## Goals
# 
# 
# 1. For a given ensemble member, combine WRFOUTs from forecast times into a single Zarr file
#     - Compute the add_local_solar_time during this phase? 
#     - Store full 3D (all vertical levels and 5 min resolution)? 
# 
# - Store data for ensemble members separately? 
# - Add dim for init times & ensemble members? 
# - upscaled 1 km 

# In[1]:


import pandas as pd
import xarray as xr 
from datetime import datetime, timedelta
import os
from multiprocessing import Pool
from functools import partial

from numcodecs import Blosc
import dask 
from dask.diagnostics import ProgressBar
import gc
import traceback 

def read_netcdf(path, transform_func=None, chunks={}):
    """Read single NetCDF lazily and optionally apply a transform_func"""
    dataset = xr.open_dataset(path, chunks=chunks, decode_times=False) 
    
    if transform_func:
        dataset = transform_func(dataset)

    return dataset 


def read_mfnetcdfs_dask(paths, dim, transform_func=None, chunks={}, load=True):
    """Read multiple NetCDF files into memory, using Dask for parallel loading."""
    # Absolutely, crucial to set threads_per_worker=1!!!!
    # https://forum.access-hive.org.au/t/netcdf-not-a-valid-id-errors/389/19
    #To summarise in this thread, it looks like a work-around in netcdf4-python to deal 
    #with netcdf-c not being thread safe was removed in 1.6.1. 
    #The solution (for now) is to make sure your cluster only uses 1 thread per worker.

    dataset = xr.open_mfdataset(paths, concat_dim=dim, combine='nested',
                                parallel=True, preprocess=transform_func,
                                chunks={}, decode_times=False) 
    
    if load:
        with ProgressBar():
            loaded_dataset= dataset.compute()
        return loaded_dataset

    return dataset 


class WRFPreProcessor:
    """
    WRFPreProcessor applies preprocessing steps to raw WRFOUT netcdf files. 
    The method utilizes dask's lazy loading feature in xarray. 
    """
    T_offset = 300. # Potential temperature offset for WRFOUT files. 
    
    def __init__(self, 
                 variables,
                 data_paths, 
                 time_resolution,
                 destagger_mapper = 
                 {'U': 'west_east_stag', 
                  'V': 'south_north_stag',
                  'W': 'bottom_top_stag',
                  'GEOPOT': 'bottom_top_stag'
                      }):
        
        self._variables = variables
        self._destagger_mapper = destagger_mapper
        self.time_resolution = time_resolution
    
        self.data_paths = data_paths
    
    def per_dataset(self, dataset):
        preprocess_steps = [
            'drop_variables', # Drop variables
            'rename_coords', # Rename coords to match ERA5
            'assign_lat_and_lon_coords', # Set Lat and Lon to 1d vectors. 
            'get_full_geopot', # Compute the full geopotential height field
            'destagger', # Destagger the U,V,W,GEOPOT, etc
            #'subset_vertical_levels', # Select the vertical layers to keep.
            'remove_negative_water_vapor', # Set negative water vapor == 0.
            'get_full_pot_temp', # Add the potential temperature offset
        ]
        for step in preprocess_steps:
            dataset = getattr(self, step)(dataset)
        
        return dataset
    
    def per_concat_dataset(self, dataset):
        preprocess_steps = [
            'rename_time_coord',
            'add_time_dim', # Add TimeDelta dim to use GraphCast data utils
            'unaccum_rainfall', # Convert accumulated rainfall to rain rate. 
        ]
        for step in preprocess_steps:
            dataset = getattr(self, step)(dataset)
        
        return dataset
    
    def drop_variables(self, dataset):
        all_vars = dataset.data_vars
        drop_variables = [v for v in all_vars if v not in self._variables] 
        
        return dataset.drop_vars(drop_variables+['XTIME'], 
                                errors='ignore')
    
    def get_full_geopot(self, dataset):
        """Combine the base and perturbation geopotential height"""
         # Combine geopotential perturbation + base state
        dataset['GEOPOT'] = dataset['PH'] + dataset['PHB']
        dataset = dataset.drop_vars(['PH', 'PHB'])
        
        return dataset 
    
    def get_full_pot_temp(self, dataset):
        """Add +300 K to the potential temperature field"""
        if 'T' in dataset.data_vars:
            dataset['T']+= self.T_offset
            
        return dataset
        
    def remove_negative_water_vapor(self, dataset):
        """Set negative water vapor == 0."""
        if 'QVAPOR' in dataset.data_vars: 
            dataset['QVAPOR'] = dataset['QVAPOR'].where(dataset['QVAPOR'] > 0, 0)

        return dataset 
    
    def rename_time_coord(self, dataset):
        return dataset.rename({'Time': 'time'})
    
    def rename_coords(self, dataset):
        """Renaming coordinate variables to align with the ERA5 naming convention. """
        return dataset.rename({'bottom_top' :'level', 
                    #'XLAT': 'latitude', 'XLONG' : 'longitude', 
                    'south_north' : 'lat', 'west_east' : 'lon'
               })
    
    def get_new_dim_name(self, stag_dim):
        """ Rename the existing staggered coordinates to the destaggered name for consistency."""
        dim_name_map = {'west_east_stag': 'lon', 'south_north_stag': 'lat', 'bottom_top_stag': 'level'}
        return dim_name_map.get(stag_dim, stag_dim)
    
    def destagger(self, dataset):
        """
        General function to destagger any given variables along their specified dimensions.

        Parameters:
            dataset : xarray.Dataset
                Input dataset.
            destagger_mapper : dict
                A mapping of variable names to their staggered dimensions.
                For example: {'U': 'west_east_stag', 'V': 'south_north_stag'}

        Returns:
            dataset : xarray.Dataset
            The dataset with destaggered variables.
        """
        for var, stag_dim in self._destagger_mapper.items():
            # Calculate the destaggered variable
            destaggered_var = 0.5 * (dataset[var] + dataset[var].roll({stag_dim: -1}, roll_coords=False))
            # Trim the last index of the staggered dimension
            destaggered_var = destaggered_var.isel({stag_dim: slice(None, -1)})
            # Rename the staggered dimension if a naming convention is provided
            # This step can be customized or made optional based on specific requirements
            new_dim_name = self.get_new_dim_name(stag_dim)  # Implement this method based on your context
            destaggered_var = destaggered_var.rename({stag_dim: new_dim_name})
            # Update the dataset with the destaggered variable
            dataset[var] = destaggered_var

        return dataset
    
    def subset_vertical_levels(self, dataset):
        # Subset the vertical levels (every N layers). 
        #TODO: Generalize this function!
        return dataset.isel(level=dataset.level[::3].values)
    
    def assign_lat_and_lon_coords(self, dataset):
        # Assign the 2D versions of 'xlat' and 'xlon' back to the dataset as coordinates
        # Renaming coordinate variables to align with the ERA5 naming convention.
        
        # Latitude and longitude are expected to be 1d vectors. 
        lat_1d = dataset['XLAT'].isel(lon=0, Time=0)
        lon_1d = dataset['XLONG'].isel(lat=0, Time=0)
        
        dataset = dataset.assign_coords(lat=lat_1d, lon=lon_1d)
    
        dataset = dataset.drop_vars(['XLAT', 'XLONG'])
        
        return dataset 
    
    def add_time_dim(self, dataset):
        """Add time dimensions/coords to make use of GraphCast data utils"""
        # Formating the time dimension for the graphcast code. 
        start_str = os.path.basename(self.data_paths[0]).split('_')[0] # wrfout or wrfwof 

        dts = [datetime.strptime(os.path.basename(f), f'{start_str}_d01_%Y-%m-%d_%H:%M:%S')
               for f in self.data_paths]
        time_range = [pd.Timestamp(dt) for dt in dts]

        num_time_points = dataset.sizes['time']

        dataset['time'] = time_range
        
        dataset = dataset.assign_coords(datetime=time_range)

        # Convert 'time' dimension to timedeltas from the first time point
        time_deltas = (dataset['time'] - dataset['time'][0]).astype('timedelta64[ns]')
        dataset['time'] = time_deltas
        
        return dataset  
    
    def unaccum_rainfall(self, dataset):
        """
        Calculate the difference in accumulated rainfall ('RAINNC') at each time step,
        with an assumption that the first time step starts with zero rainfall.
    
        Parameters:
        - ds: xarray.Dataset containing the 'RAINNC' variable
    
        Returns:
            - Modified xarray.Dataset with the new variable 'RAINNC_DIFF'
        """
        if 'RAINNC' not in self._variables:
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
    
    
import itertools 
from tqdm.notebook import tqdm 
from glob import glob
import re
    
def filter_dates(dates, month_range = ['03', '04', '05', '06', '07']):
    """
    Filter a list of dates to exclude January, February, November, and December. 

    Args:
    - dates: A list of dates in 'YYYYMMDD' format.

    Returns:
    - A list of dates that fall in March - July 
    """
    filtered_dates = [date for date in dates if date[4:6] in month_range]
    
    return filtered_dates
    

class WRFFileGenerator: 
    
    def __init__(self, duration_minutes='all', timestep_minutes=10, offset=0):
        self.duration_minutes = duration_minutes
        self.timestep_minutes = timestep_minutes 
        self.offset = offset 
    
    def parse_filename_datetime(self, filename):
        """
        Extract datetime object from a WRFOUT file path.
    
        Args:
            filename (str): Filename in the format wrfwof_d01_YYYY-MM-DD_HH:MM:SS
    
        Returns:
            datetime: Datetime object representing the timestamp in the filename.
        """
        # Convert string to datetime object
        return datetime.strptime(filename, 'wrfwof_d01_%Y-%m-%d_%H:%M:%S')
    
    def get_duration(self, directory_path):
        init_time = os.path.basename(os.path.dirname(directory_path))
        
        ###print(init_time, init_time[-2:])
        
        if init_time[-2:] == '30':
            return 180 # 3 hrs for bottom of the hour
        else:
            return 360 # 6 hrs for the top of the hour
        
    
    def get_wrfwofs_files(self, directory_path):
        """
        Load files for a given duration and timestep.
    
        Args:
        directory_path (str): Path to the directory containing the files. 

        Returns:
            list: List of filenames that match the given duration and timestep.
        """
        # List all wrfwof files in the directory
        files = glob(os.path.join(directory_path, 'wrfwof_d01_*'))
        files.sort()
                    
        return files 
        
    def file_path_generator(self, date_dir_path):
        """
        List all directories matching the pattern /work2/wof/realtime/FCST/YYYY/YYYYMMDD/HHMM/ENS_MEM_NN/

        Args:
        base_path (str): Base directory to start the search (e.g., '/work2/wof/realtime/FCST/2019/')

        Returns:
        list: A list of all matching directory paths.
        """
        ensemble_dirs = []
        # Regular expression to match the date directories and ensemble member directories
        date_pattern = re.compile(r"\d{8}$")  # YYYYMMDD
        time_pattern = re.compile(r"\d{4}$")  # HHMM
        ensemble_pattern = re.compile(r"ENS_MEM_\d{1,2}$")  # ENS_MEM_N or ENS_MEM_NN

        for root, dirs, files in os.walk(date_dir_path):
            # Filter directories to continue the walk
            dirs[:] = [d for d in dirs 
                       if date_pattern.match(d) or time_pattern.match(d) or ensemble_pattern.match(d)]
        
            # Check if any current directories are ensemble member directories
            for dir_name in dirs:
                if ensemble_pattern.match(dir_name):
                    yield os.path.join(root, dir_name)
    
    def gen_file_paths(self, date_dir_paths):
        for directory_path in date_dir_paths:
            for path in self.file_path_generator(directory_path):
                yield self.get_wrfwofs_files(path)
                
    def gen_single_paths(self, date_dir_paths):
        for directory_path in date_dir_paths:
            for path in self.file_path_generator(directory_path):
                for p in self.get_wrfwofs_files(path):
                    yield p            
                
             

def get_file_path(files, 
                  dir_replace=('/work2/wof/realtime/FCST/', 
                               '/work2/wofs_zarr/')):
    # Get the first path 
    path = files[0]
    # Replace the /work2 path with the new dir
    path = path.replace(dir_replace[0], dir_replace[1])
    # Get the path and not the current filename
    path = os.path.dirname(path)
    
    pattern = r'(ENS_MEM_)(\d+)'

    # Function to add leading zero if the number has less than 2 digits
    def add_leading_zero(match):
        ens_mem = match.group(1)  # The 'ENS_MEM_' part
        number = match.group(2)   # The number part
        return f'{ens_mem}{int(number):02}'  # Format number with leading zero if necessary

    # Replace the found pattern in the path using the add_leading_zero function
    new_path = re.sub(pattern, add_leading_zero, path)
    
    return new_path
    
def create_filename_from_list(file_paths, time_resolution):
    """
    Create a filename based on the first and last elements of a list of file paths.

    Args:
            file_paths (list): A list of file paths.

    Returns:
            str: A string representing the generated filename, which includes the start and end datetime.
    """
    if not file_paths:
        return "No files provided"

    # Extract start time from the first element
    start_time = os.path.basename(file_paths[0]).replace('wrfwof_d01_', '')  
    # Extract end time from the last element
    end_time = os.path.basename(file_paths[-1]).replace('wrfwof_d01_', '')  
    
    # Format the filename
    ens_mem = os.path.basename(os.path.dirname(file_paths[-1])).split('_')[-1]
    
    filename = f"wrfwof_{start_time}_to_{end_time}_{time_resolution}.zarr"

    # Cleaning up the datetime format to remove colons and make it filesystem-friendly
    for char in [":"]:
        filename = filename.replace(char, "")
    
    return filename
  
def replace_base_path(path, new_base_path, old_base_path='/work2/wof/realtime/FCST/'):
    new_path = path.replace(old_base_path, new_base_path)
    return new_path 
    

@dask.delayed
def _process_one_file(path, variables, timestep_minutes):
    """Process a single set of WRFOUT files and convert them into Zarr."""
    try:
        # Replace base path and set the output path
        out_path = replace_base_path(path,
                                     new_base_path='/work2/wofs_zarr/', 
                                     old_base_path='/work2/wof/realtime/FCST/')
        out_path = out_path + '.zarr'
        
        if os.path.exists(out_path):
            print(f'{out_path} already processed!')
            return None
        
        # Validate the input parameters
        if not isinstance(path, str) or not os.path.exists(path):
            raise ValueError(f"Invalid file path provided: {path}")

        if not isinstance(variables, list) or not variables:
            raise ValueError("Variables must be a non-empty list")

        if not isinstance(timestep_minutes, int) or timestep_minutes <= 0:
            raise ValueError(f"Invalid timestep: {timestep_minutes}")

        # Set the time resolution
        time_resolution = f'{timestep_minutes}min'
        
        # Initialize the preprocessor
        preprocessor = WRFPreProcessor(variables, path, time_resolution=time_resolution)

        # Read the NetCDF file using the specified preprocessor
        dataset = read_netcdf(path, transform_func=preprocessor.per_dataset, chunks={})
        
        # Configure the Blosc compressor
        compressor = Blosc(cname='zstd', clevel=7, shuffle=Blosc.BITSHUFFLE)

        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(out_path)
        if not os.path.exists(output_dir):
            print(f'Creating directory {output_dir}...')
            os.makedirs(output_dir, exist_ok=True)
        
        # Save the dataset to a Zarr file
        print(f'Saving to {out_path}...')
        dataset.to_zarr(out_path, mode='w', 
                        encoding={var: {'compressor': compressor} for var in dataset.variables})

        # Clean up
        dataset.close()
        del dataset
        gc.collect()

        return f'Processed {out_path} successfully!'

    except Exception as e:
        error_message = f"Error processing file {path}: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return error_message


if __name__ == "__main__":
    # Main script. 
    
    """ usage: stdbuf -oL python -u wrfwof_nc_to_zarr.py > & log_wrfwof_to_zarr & """
    
    # Configuration to use the multiprocessing scheduler
    dask.config.set(scheduler='processes')  # This tells Dask to use processes instead of threads

    
    VARS_3D_TO_KEEP = ['U', 'V', 'W', 'T', 'PH', 'PHB', 'QVAPOR']
    VARS_2D_TO_KEEP = ['T2', 'RAINNC', 'COMPOSITE_REFL_10CM', 'UP_HELI_MAX', 
                   'Q2', 'U10', 'V10', 'REL_VORT_MAX', 'SWDOWN', 'WSPD80', 
                   'W_UP_MAX', 'LWP'
                  ]
    CONSTANTS = ['HGT', 'XLAND']
    
    VARIABLES = VARS_3D_TO_KEEP + VARS_2D_TO_KEEP + CONSTANTS
    
    timestep_minutes = 5
    offset=0

    all_date_dir_paths = [] 
    years = ['2019', '2020', '2021']
    for year in years:
        base_path = os.path.join('/work2/wof/realtime/FCST/', year)
        all_dates = os.listdir(base_path)
        all_dates.sort()
    
        dates = filter_dates(all_dates)#[:1]
        all_date_dir_paths.extend([os.path.join(base_path, d) for d in dates])

    # Store the current time before starting
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")    
        
    generator = WRFFileGenerator(duration_minutes='all',
        timestep_minutes=timestep_minutes, offset=offset)

    # Need a correct number for the progress bar? 
    file_lists = generator.gen_single_paths(all_date_dir_paths)
    
    #file_lists = list(generator.gen_single_paths(all_date_dir_paths))
    #file_lists = file_lists[:12]

    # Create a list of delayed objects
    tasks = [_process_one_file(path, VARIABLES, timestep_minutes) for path in file_lists]

    with ProgressBar():
        results = dask.compute(*tasks, scheduler='processes', num_workers=12)
        
    # Calculate and print the elapsed time
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    days, remainder = divmod(elapsed_time.total_seconds(), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed time: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes")    
