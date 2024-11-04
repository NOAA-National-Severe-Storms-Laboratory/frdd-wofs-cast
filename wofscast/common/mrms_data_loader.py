from pathlib import Path 
import os
import pandas as pd
import numpy as np
import xarray as xr

def to_xarray_dataset(array, time, lat, lon, variable_name='data'):
    """
    Convert a NumPy array of shape (NT, lat, lon) into an xarray.Dataset.

    Parameters:
    - array: The input NumPy array with shape (NT, lat, lon).
    - time: Array-like time values corresponding to the NT dimension.
    - lat: Array-like latitude values.
    - lon: Array-like longitude values.
    - variable_name: The name of the variable to assign to the dataset (default: 'data').

    Returns:
    - xarray.Dataset: Dataset containing the input data with 'time', 'lat', and 'lon' coordinates.
    """
    # Create an xarray DataArray from the numpy array
    data_array = xr.DataArray(
        array,
        dims=['time', 'lat', 'lon'],  # Define the dimensions
        coords={'time': time, 'lat': lat, 'lon': lon},  # Assign coordinates
        name=variable_name  # Name of the variable
    )

    # Convert the DataArray to a Dataset
    dataset = data_array.to_dataset()

    return dataset




class MRMSDataLoader: 
    
    MRMS_PATH = '/work/rt_obs/MRMS/RAD_AZS_MSH/'
    MRMS_QPE_PATH = '/work/rt_obs/MRMS/QPE/'
    
    def __init__(self, case_date, domain_size=150, resize_domain=True):
        self.case_date = case_date
        self.domain_size = domain_size
        self.resize_domain = resize_domain 

    def find_mrms_files(self, datetime_rng):
        """
        When given a start and end date, this function will find any MRMS RAD 
        files between those time periods. It will check if the path exists. 
        """
        year = str(datetime_rng[0].year) 
        mrms_filenames = [date.strftime('wofs_MRMS_RAD_%Y%m%d_%H%M.nc') for date in datetime_rng]

        mrms_filepaths = [Path(self.MRMS_PATH).joinpath(year, self.case_date, f) 
                          if Path(self.MRMS_PATH).joinpath(year, self.case_date, f).is_file() else None
                          for f in mrms_filenames 
                 ]
       
        return mrms_filepaths 
    
    def find_qpe_files(self, datetime_rng):
        start_time, end_time = datetime_rng[0], datetime_rng[-1]
    
        year = str(start_time.year) 
        
        directory = os.path.join(self.MRMS_QPE_PATH, year, self.case_date)
        
        # List all files in the directory
        all_files = [f for f in os.listdir(directory) if 'wofs_MRMS_QPE' in f]
    
        # Filter files by the pattern and extract the datetime part
        selected_files = []
        for file in all_files:
            # Extract the timestamp from the filename
            timestamp_str = file.split('wofs_MRMS_QPE_')[-1].replace('.nc', '')
        
            # Convert the extracted timestamp string to a pandas Timestamp
            file_time = pd.to_datetime(timestamp_str, format='%Y%m%d_%H%M')
        
            # Check if the file time is within the start and end times
            if start_time <= file_time <= end_time:
                selected_files.append(os.path.join(directory, file))

        return selected_files

    
    def load(self, template_ds, mode='refl'):
        N = 300 
        
        datetimes = pd.to_datetime(template_ds.datetime)
        
        
        if mode == 'refl':
            files = self.find_mrms_files(datetimes)
        elif mode == 'qpe':
            files = self.find_qpe_files(datetimes)
         
        # If any files are missing, then return None 
        # so it can be skipped. 
        any_files_none = any([f is None for f in files])
        if any_files_none:
            return None

        # Use Dask to lazily load data
        ds = xr.open_mfdataset(files, chunks={}, combine='nested', 
                               concat_dim='time', engine='netcdf4',
                               drop_variables=['lat', 'lon'])
        if 'longitude' in ds.dims:
            ds = ds.rename({'longitude' : 'lon', 'latitude' : 'lat'})
        
        datetimes = pd.to_datetime(template_ds.datetime)
        times = template_ds.time.values
        
        # Resizing the dataset if necessary
        if self.resize_domain:
            start = (N - self.domain_size) // 2
            end = start + self.domain_size
            ds = ds.isel(lon=slice(start, end), lat=slice(start, end))
        
        ds = ds.assign_coords(lat = template_ds.lat)
        ds = ds.assign_coords(lon = template_ds.lon)
        if mode == 'refl':
            ds = ds.assign_coords(time = times)
            ds = ds.assign_coords(datetime = ('time', datetimes))
        
        ds = ds.compute() 
        
        return ds