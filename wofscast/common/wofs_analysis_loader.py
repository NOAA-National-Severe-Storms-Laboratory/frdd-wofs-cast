import os 
import xarray as xr 
import numpy as np
import pandas as pd
from glob import glob

class WoFSAnalysisLoader:
    # Location of the WoFS analysis files.
    BASE_WOFS_PATH = '/work2/wof/realtime/'
    
    def __init__(self, domain_size=150, resize=True): 
        self.domain_size = domain_size
        self.resize = resize
        
        self._var_dim_map = {
            'U': 'west_east_stag', 
            'V': 'south_north_stag',
            'W': 'bottom_top_stag',
            'GEOPOT': 'bottom_top_stag',
        }
    
    def get_analysis_paths(self, datetime_list, year, case_date, mem): 
        file_paths = []
        for timestamp in datetime_list:
            timestamp_str = timestamp.strftime('%Y%m%d%H%M')
            fname = f'wrfout_d01_{timestamp:%Y-%m-%d_%H:%M:%S}_{mem}'
            this_path = os.path.join(self.BASE_WOFS_PATH, year, case_date, timestamp_str, fname)    
            file_paths.append(this_path if os.path.exists(this_path) else None)
        return file_paths
    
    def get_new_dim_name(self, stag_dim):
        """ Rename staggered coordinates to the destaggered name for consistency."""
        dim_name_map = {'west_east_stag': 'lon', 'south_north_stag': 'lat', 'bottom_top_stag': 'level'}
        return dim_name_map.get(stag_dim, stag_dim)
    
    def destagger(self, ds):
        """
        Destagger any given variables along their specified dimensions using vectorized operations.
        """
        for var, stag_dim in self._var_dim_map.items():
            if var in ds:
                destaggered_var = 0.5 * (ds[var] + ds[var].roll({stag_dim: -1}, roll_coords=False))
                destaggered_var = destaggered_var.isel({stag_dim: slice(None, -1)})
                new_dim_name = self.get_new_dim_name(stag_dim)
                destaggered_var = destaggered_var.rename({stag_dim: new_dim_name})
                ds[var] = destaggered_var
        return ds
    
    def load(self, template_ds, datetime_list, case_date, mem=9):
        N = 300
        
        year  = case_date[:4]
        
        paths = self.get_analysis_paths(datetime_list, year, case_date, mem)
        actual_paths = [p for p in paths if p]
        actual_times = [t for p, t in zip(paths, template_ds.time.values) if p]
        datetimes = [t for p, t in zip(paths, template_ds.datetime.values) if p]
        
        if all([p is None for p in paths]):
            return None
        
        _ds = xr.open_dataset(actual_paths[0], chunks={}, engine='netcdf4') 
        all_vars = _ds.data_vars
        
        drop_vars = [v for v in all_vars if v not in template_ds.data_vars]
        drop_vars += ['XLAT', 'XLONG', 'XTIME', 'XLAT_U', 'XLAT_V', 'XLONG_U', 'XLONG_V']
        
        drop_vars.remove('PH')
        drop_vars.remove('PHB')
                
        _ds.close() 
        
        # Use Dask to lazily load data
        ds = xr.open_mfdataset(actual_paths, chunks={}, combine='nested', concat_dim='Time', 
                                    drop_variables=drop_vars, engine='netcdf4')
        
        ds = ds.rename({'Time' : 'time'})
        
        ds['GEOPOT'] = ds['PH'] + ds['PHB']  # Combine PH and PHB into GEOPOT
        ds = ds.drop_vars(['PH', 'PHB'])
        
        ds['T']+=300. 
        
        ds = self.destagger(ds)

        # Rename
        ds = ds.rename({"south_north" : 'lat', 'west_east': 'lon', 'bottom_top' : 'level'})
 
        # Pre-select heights if necessary
        if 'level' in ds.dims:
            ds = ds.isel(level=ds.level[::3])  # Efficiently downsample
        
        # Resizing the dataset if necessary
        if self.resize:
            start = (N - self.domain_size) // 2
            end = start + self.domain_size
            ds = ds.isel(lon=slice(start, end), lat=slice(start, end))
        
        ds = ds.assign_coords(lat = template_ds.lat)
        ds = ds.assign_coords(lon = template_ds.lon)
        ds = ds.assign_coords(time = actual_times)
        ds = ds.assign_coords(datetime = ('time', datetimes))
        
        # Select the variable and convert it to a numpy array lazily
        ds = ds.compute()  # Load into memory after applying all operations
        
        return ds
      