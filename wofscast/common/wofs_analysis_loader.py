import os 
import xarray as xr 
import numpy as np
from glob import glob

class WoFSAnalysisLoader:
    # Location of the WoFS analysis files.
    BASE_WOFS_PATH = '/work2/wof/realtime/'
    def __init__(self, domain_size=150, resize=True): 
        self.domain_size = domain_size
        self.resize = resize
    
    def get_analysis_paths(self, datetime_list, year, case_date, mem): 
        
        file_paths = []

        for timestamp in datetime_list:
            year = str(timestamp.year)
            timestamp_str = timestamp.strftime('%Y%m%d%H%M')

            fname = f'wrfout_d01_{timestamp:%Y-%m-%d_%H:%M:%S}_{mem}'
            this_path = os.path.join(self.BASE_WOFS_PATH, year, case_date, timestamp_str, fname)    
            if os.path.exists(this_path):
                file_paths.append(this_path)
            else:
                file_paths.append(None)
        
        return file_paths
    
    def load(self, var, datetime_list, year, case_date, mem):
        N = 300
        
        paths = self.get_analysis_paths(datetime_list, year, case_date, mem)
        
        # Initialize data array with -1 for missing files or timesteps. 
        data = np.ones((len(paths), self.domain_size, self.domain_size))*-1.0
        
        for t, path in enumerate(paths):
            if path:
                ds = xr.open_dataset(path, chunks={}).isel(Time=-1)
                
                vals = ds[var].values
                
                if self.resize:
                    start = (N - self.domain_size) // 2
                    end = start + self.domain_size
                
                    vals = vals[start:end, start:end]
  
                data[t,:,:] = vals
    
                ds.close()
        
        return data