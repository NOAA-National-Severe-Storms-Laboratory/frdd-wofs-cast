from pathlib import Path 
import os
import pandas as pd
import numpy as np
import xarray as xr

class MRMSDataLoader: 
    
    MRMS_PATH = '/work/rt_obs/MRMS/RAD_AZS_MSH/'
    MRMS_QPE_PATH = '/work/rt_obs/MRMS/QPE/'
    
    def __init__(self, case_date, datetime_rng, domain_size=150, resize_domain=True):
        self.case_date = case_date
        self.datetime_rng = datetime_rng 
        self.domain_size = domain_size
        self.resize_domain = resize_domain 

    def find_mrms_files(self):
        """
        When given a start and end date, this function will find any MRMS RAD 
        files between those time periods. It will check if the path exists. 
        """
        year = str(self.datetime_rng[0].year) 
        mrms_filenames = [date.strftime('wofs_MRMS_RAD_%Y%m%d_%H%M.nc') for date in self.datetime_rng]

        mrms_filepaths = [Path(self.MRMS_PATH).joinpath(year, self.case_date, f) 
                          if Path(self.MRMS_PATH).joinpath(year, self.case_date, f).is_file() else None
                          for f in mrms_filenames 
                 ]
       
        return mrms_filepaths 
    
    def find_qpe_files(self):
        start_time, end_time = self.datetime_rng[0], self.datetime_rng[-1]
    
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

    def resize(self, ds, n_lat=300, n_lon=300):
        """Resize the domain"""
        domain_size = self.domain_size
        start_lat, start_lon = (n_lat - domain_size) // 2, (n_lon - domain_size) // 2
        end_lat, end_lon = start_lat + domain_size, start_lon + domain_size
        
        # Subsetting the dataset to the central size x size grid
        ds_subset = ds.isel(lat=slice(start_lat, end_lat), lon=slice(start_lon, end_lon))
        
        return ds_subset
    
    def load_qpe(self):
        
        files = self.find_qpe_files()
        
        # Initialize an empty list to store the datasets with 'mesh_consv' variable
        data = np.zeros((len(files), self.domain_size, self.domain_size))

        # Load 'mesh_consv' variable from each file and append to the datasets list
        for t, file in enumerate(files):
            if file is not None: 
                ds = xr.open_dataset(file, drop_variables=['lat', 'lon'])
                
                # Resize the output to 150 x 150\
                if self.resize_domain: 
                    ds = self.resize(ds)
                
                data[t,:,:] = ds['qpe_consv'].values
    
                ds.close()
        
        # Sum over time
        data = data.sum(axis=0)
        
        return data
        
    
    def load(self):

        files = self.find_mrms_files()
        
        # Initialize an empty list to store the datasets with 'mesh_consv' variable
        data = np.zeros((len(files), self.domain_size, self.domain_size))

        # Load 'mesh_consv' variable from each file and append to the datasets list
        for t, file in enumerate(files):
            if file is not None: 
                ds = xr.open_dataset(file, drop_variables=['lat', 'lon'])
                
                # Resize the output to 150 x 150
                if self.resize_domain: 
                    ds = self.resize(ds)
                
                data[t,:,:] = ds['dz_consv'].values
    
                ds.close()
        
        return data
