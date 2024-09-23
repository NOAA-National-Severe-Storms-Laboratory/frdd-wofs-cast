from ..common.helpers import (get_case_date, 
                              to_datetimes, 
                              convert_rain_amount_to_inches, 
                              convert_T2_K_to_F)

from ..common.plot_utils import display_name_mapper, units_mapper
from ..common.mrms_data_loader import MRMSDataLoader 
from .metrics import ObjectBasedContingencyStats


import os 
from tqdm import tqdm 
import xarray as xr

# For the dataclass below. 
from dataclasses import dataclass, asdict
from typing import List, Dict

@dataclass
class EvaluatorConfig:
    data_path: str
    n_samples: int
    seed: int 
    model_path: str
    add_diffusion: bool
    load_ensemble: bool 
    spectra_variables: List[str]
    pmm_variables: List[str]
    fss_variables: List[str]
    fss_windows: List[int]
    fss_thresh_dict: Dict[str, List[float]]
    matching_distance_km: int
    grid_spacing_km : float
    out_base_path : str


class Evaluator:
    
    DIM_ORDER = ('batch', 'time', 'level', 'lat', 'lon')
    
    def __init__(self, model, object_ider, data_loader, metrics, 
                 unit_converter_funcs = [convert_T2_K_to_F, convert_rain_amount_to_inches]
                ): 
        self.model = model 
        self.data_loader = data_loader 
        self.metrics = metrics 
        self.object_ider = object_ider
        self.unit_converter_funcs = unit_converter_funcs
     
    def load_mrms_data(self, forecast, data_path):
        
        # Load the MRMS composite reflectivity. Loads all the separate time steps into 
        # one dataset. 
        case_date = get_case_date(data_path)
        dts = to_datetimes(data_path, n_times = len(forecast.time)+2)
        loader = MRMSDataLoader(case_date, dts) 

        try: 
            mrms_dataset = loader.load() 
            return mrms_dataset
        except OSError:
            print(f'Unable to load MRMS data for {data_path}, Skipping it...')
            return None 
    
    def add_units(self, forecast, targets):
        # Convert units. 
        unit_converter_funcs = [convert_T2_K_to_F, convert_rain_amount_to_inches]

        for func in self.unit_converter_funcs:
            forecast = func(forecast)
            targets = func(targets)
    
        for var in forecast.data_vars:
            var_key = var.replace('MAX', '') if 'MAX' in var else var 
    
            forecast[var].attrs['units'] = units_mapper.get(var_key, '')
            forecast[var].attrs['display_name'] = display_name_mapper.get(var_key, '')
            
            targets[var].attrs['units'] = units_mapper.get(var_key, '')
            targets[var].attrs['display_name'] = display_name_mapper.get(var_key, '')
            
        return forecast, targets 

    def evaluate(self, paths):
        
        wofs_dbz_thresh = 40  
            
        # TODO: Allow this to be optional! 
        wofs_v_mrms_metric = ObjectBasedContingencyStats(key='wofs_vs_mrms')
        wofscast_v_mrms_metric = ObjectBasedContingencyStats(key='wofscast_vs_mrms')
        
        for data_path in tqdm(paths): 
    
            inputs, targets, forcings = self.data_loader.load_inputs_targets_forcings(data_path)
            forecast = self.model.predict(inputs, targets, forcings)
    
            # TODO: make this more flexible, especially for adding other derived variables.
            forecast['WMAX'] = forecast['W'].max(dim='level')
            targets['WMAX'] = targets['W'].max(dim='level')
    
            # Convert units. 
            forecast, targets = self.add_units(forecast, targets)
    
            # Load the MRMS composite reflectivity. Loads all the separate time steps into 
            # one dataset. 
            mrms_dataset = self.load_mrms_data(forecast, data_path)
            if mrms_dataset is None:
                continue
    
            # Identify objects. TODO: Add UH tracks and any other field of interest! 
            # TODO: Also, add args to change these reflectivity thresholds.
            mrms_dataset = self.object_ider.label(mrms_dataset, 'comp_dz', params={'bdry_thresh' : 40})
            forecast = self.object_ider.label(forecast, 'COMPOSITE_REFL_10CM', params={'bdry_thresh' : wofs_dbz_thresh})
            targets = self.object_ider.label(targets, 'COMPOSITE_REFL_10CM', params={'bdry_thresh' : wofs_dbz_thresh})
    
            # Transpose to the same dimension order 
            targets = targets.transpose(*self.DIM_ORDER)
            forecast = forecast.transpose(*self.DIM_ORDER)
    
            # Update the metrics. 
            self.metrics = [metric.update(forecast, targets) for metric in self.metrics]
        
            # Matching against MRMS 
            wofs_v_mrms_metric.update(targets, mrms_dataset)
            wofscast_v_mrms_metric.update(forecast, mrms_dataset)
    
        metrics_ds = [metric.finalize() for metric in self.metrics]

        metrics_ds.append(wofs_v_mrms_metric.finalize())
        metrics_ds.append(wofscast_v_mrms_metric.finalize()) 

        # Setting combine_attrs="no_conflicts" to keep the 
        # n_pmm_storms attribute. 
        results_ds = xr.merge(metrics_ds, combine_attrs="no_conflicts")
        
        return results_ds
    
    def save(self, dataset, config, overwrite=False):
        
        model_name = os.path.basename(config.model_path).replace('.npz', '')
        
        if config.add_diffusion:
            out_path = os.path.join(config.out_base_path, f'{model_name}_diffusion_results.nc')
        else:
            out_path = os.path.join(config.out_base_path, f'{model_name}_results.nc')

        # Add the config variables as attributes for metadata in the future.
        for key, item in asdict(config).items():
            dataset.attrs[key] = str(item)
    
        # If overwrite is False, check for existing files and create a versioned filename
        if not overwrite and os.path.exists(out_path):
            base, ext = os.path.splitext(out_path)
            version = 1
        
            # Keep incrementing the version number until a non-existing file is found
            while os.path.exists(f"{base}_v{version}{ext}"):
                version += 1
        
            # Set the new versioned filename
            out_path = f"{base}_v{version}{ext}"
    
        # Save the dataset to the final out_path
        dataset.to_netcdf(out_path)
    
        return f'Saved results dataset to {out_path}!'