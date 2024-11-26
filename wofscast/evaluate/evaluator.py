from ..common.helpers import (get_case_date, 
                              to_datetimes, 
                              convert_rain_amount_to_inches, 
                              convert_T2_K_to_F)

from ..common.plot_utils import display_name_mapper, units_mapper
from ..common.mrms_data_loader import MRMSDataLoader 
from ..common.wofs_analysis_loader import WoFSAnalysisLoader
from .metrics import ObjectBasedContingencyStats, MSE, FractionsSkillScore

import os 
from tqdm import tqdm 
import xarray as xr
import numpy as np
import pandas as pd

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
    object_id_params : Dict

        
class Evaluator:
    
    DIM_ORDER = ('batch', 'time', 'level', 'lat', 'lon')
    
    def __init__(self, model, object_ider, data_loader, metrics, 
                 object_id_params, 
                 forecast_v_mrms_metrics, 
                 targets_v_mrms_metrics, 
                 forecast_v_analysis_metrics,
                 targets_v_analysis_metrics,
                 unit_converter_funcs = [convert_T2_K_to_F, 
                                         convert_rain_amount_to_inches],
                 replace_bdry=True
                ): 
        
        self.replace_bdry = replace_bdry
        
        # Parameters for identifying objects in the forecast, truth, and mrms datasets. 
        self.object_id_params = object_id_params

        self.model = model 
        self.data_loader = data_loader 
        self.metrics = metrics 
        
        # Metrics for comparing against an analysis dataset and against MRMS. 
        self.targets_v_analysis_metrics = targets_v_analysis_metrics
        self.forecast_v_analysis_metrics = forecast_v_analysis_metrics
        
        self.targets_v_mrms_metrics = targets_v_mrms_metrics
        self.forecast_v_mrms_metrics = forecast_v_mrms_metrics 
               
        self.object_ider = object_ider
        self.unit_converter_funcs = unit_converter_funcs
        
    def add_initial_conditions(self, dataset, inputs):
        return xr.concat([inputs.isel(time=-1), dataset], dim='time')

    def load_mrms_data(self, forecast, data_path):
        # Load the MRMS composite reflectivity. Loads all the separate time steps into 
        # one dataset. 
        case_date = get_case_date(data_path)
        dts = pd.to_datetime(forecast.datetime) 
        # TODO: Not flexible for the full domain!!!
        loader = MRMSDataLoader(case_date, domain_size=150, resize_domain=True) 
        try: 
            mrms_radar = loader.load(forecast, mode='refl') 
            mrms_qpe = loader.load(forecast, mode='qpe') 
            
            if mrms_radar is None:
                return None 
            
            if mrms_qpe is None:
                return None 
            
            
        except OSError:
            print(f'Unable to load MRMS data for {data_path}, Skipping it...')
            return None     
            
        mrms_qpe = mrms_qpe.expand_dims('batch', axis=0)
            
        qpe_var = 'qpe_consv' if 'qpe_consv' in mrms_qpe.data_vars else 'qpe_consv_15m'
            
        # Create accumulated rain variable and drop other QPE variables
        mrms_qpe = mrms_qpe.assign(accum_rain=mrms_qpe[qpe_var].sum(dim='time')).drop_vars(mrms_qpe.data_vars)
                        
        mrms_dataset = xr.merge([mrms_radar, mrms_qpe])
            
        mrms_dataset = mrms_dataset.transpose(*self.DIM_ORDER, missing_dims='ignore')
        mrms_dataset = mrms_dataset.drop_vars('datetime') 

        # Add the reflectivity value for the FSS. TODO: this is hardcoded and should be fixed
        # in the future. Problem should just create the same named variable between WRF and MRMS. 
        mrms_dataset['COMPOSITE_REFL_10CM'] = mrms_dataset['dz_consv'].copy() 
            
        return mrms_dataset
        
    def load_analysis_data(self, forecast, data_path):
        # Load the WoFS analyses. Loads all the separate time steps into 
        # one dataset. 
        case_date = get_case_date(data_path)
        dts = pd.to_datetime(forecast.datetime)  
        
        loader = WoFSAnalysisLoader()
        variables = list(forecast.data_vars)
        variables.remove('RAIN_AMOUNT') 
        variables.remove('WMAX') 

        wofs_analysis_dataset = loader.load(forecast, dts, case_date, mem=9)
        if wofs_analysis_dataset:
            wofs_analysis_dataset = wofs_analysis_dataset.expand_dims('batch', axis=0)
            wofs_analysis_dataset = wofs_analysis_dataset.drop_vars('datetime')
                
            wofs_analysis_dataset = self.add_units(wofs_analysis_dataset, unit_converter_funcs = [convert_T2_K_to_F])
            wofs_analysis_dataset['WMAX'] = wofs_analysis_dataset['W'].max(dim='level')

            wofs_analysis_dataset = wofs_analysis_dataset.transpose(*self.DIM_ORDER)     
                
            # Ensure the analysis dataset has the same time dimension as the forecast. 
            wofs_analysis_dataset = wofs_analysis_dataset.reindex(time=forecast.time, fill_value=np.nan)

        return wofs_analysis_dataset
    
    
    def add_units(self, dataset, unit_converter_funcs = [convert_T2_K_to_F, convert_rain_amount_to_inches]):
        # Convert units. 
        for func in unit_converter_funcs:
            dataset = func(dataset)
 
        for var in dataset.data_vars:
            var_key = var.replace('MAX', '') if 'MAX' in var else var 
    
            dataset[var].attrs['units'] = units_mapper.get(var_key, '')
            dataset[var].attrs['display_name'] = display_name_mapper.get(var_key, '')
            
        return dataset

    def add_derived_variables(self, forecast, targets):
        # TODO: make this more flexible, especially for adding other derived variables.
        forecast['WMAX'] = forecast['W'].max(dim='level')
        targets['WMAX'] = targets['W'].max(dim='level')
        
        forecast['accum_rain'] = forecast['RAIN_AMOUNT'].sum(dim='time')
        targets['accum_rain'] = targets['RAIN_AMOUNT'].sum(dim='time')
        
        return forecast, targets 
    
    def compute_percentile_thresholds(self, paths):
        """Use this method to compute the determine the percentile of 40 dBZ in the MRMS dataset. 
        Using that percentile, find the dBZ threshold for the forecast and truth datasets. 
        """
        MRMS_THRESH = 40.
        
        forecast_vals = []
        truth_vals = []
        mrms_vals = [] 
        
        for data_path in tqdm(paths): 
    
            inputs, targets, forcings = self.data_loader.load_inputs_targets_forcings(data_path)
            forecast = self.model.predict(inputs, targets, forcings)
            
            forecast = self.add_initial_conditions(forecast, inputs)
            targets = self.add_initial_conditions(targets, inputs)
    
            mrms_dataset = self.load_mrms_data(forecast, data_path)
            if mrms_dataset:  
                forecast_vals.append(forecast['COMPOSITE_REFL_10CM'].isel(time=-1).values.ravel())
                truth_vals.append(targets['COMPOSITE_REFL_10CM'].isel(time=-1).values.ravel())
                mrms_vals.append(mrms_dataset['COMPOSITE_REFL_10CM'].values.ravel()) 
                
        mrms_vals = np.array(mrms_vals).ravel()  

        # Compute the percentile corresponding to 40 dBZ in MRMS.
        percentile_40dbz = ((mrms_vals < MRMS_THRESH).sum() / len(mrms_vals)) * 100
        
        print(f"Percentile for 40 dBZ in MRMS: {percentile_40dbz}%")
        
        # Use that percentile to determine the corresponding dbz val
        forecast_threshold = np.percentile(forecast_vals, percentile_40dbz)
        truth_threshold = np.percentile(truth_vals, percentile_40dbz)        
        
        print(f"Forecast threshold for {percentile_40dbz}%: {forecast_threshold}")
        print(f"Truth threshold for {percentile_40dbz}%: {truth_threshold}")
        
        return {
                'forecast_threshold': forecast_threshold,
                'truth_threshold': truth_threshold
                }
                
    def evaluate(self, paths):
        
        for data_path in tqdm(paths): 
    
            inputs, targets, forcings = self.data_loader.load_inputs_targets_forcings(data_path)
            forecast = self.model.predict(inputs, targets, forcings, replace_bdry=self.replace_bdry)
    
            forecast = self.add_initial_conditions(forecast, inputs)
            targets = self.add_initial_conditions(targets, inputs)
    
            # Add derived variables to forecast & targets; ID storm objects. 
            forecast, targets = self.add_derived_variables(forecast, targets)
            
            if len(self.object_id_params['forecast'].keys()) > 1:
                raise ValueError('Code is not ready for object identifying more than 1 field!')
            
            for var in self.object_id_params['forecast'].keys():
                # TODO: Beware, this is not flexible yet! All labelled regions are called "storms"
                # and each iteration will replace the previous storms!.
                forecast = self.object_ider.label(forecast, var, 
                                              params={'bdry_thresh' : self.object_id_params['forecast'][var]})
                
                targets = self.object_ider.label(targets, var, 
                                             params={'bdry_thresh' : self.object_id_params['truth'][var]})

            # Load the WoFS analysis and MRMS radar and QPE variables. 
            wofs_analysis_dataset = self.load_analysis_data(forecast, data_path) 
            mrms_dataset = self.load_mrms_data(forecast, data_path)
      
            # Convert units. At the moment, its T2 K -> F and rain rate from mm to in. 
            forecast = self.add_units(forecast)
            targets = self.add_units(targets)
    
            # Transpose to the same dimension order 
            targets = targets.transpose(*self.DIM_ORDER)
            forecast = forecast.transpose(*self.DIM_ORDER)

            # Drop the datetime coordinate from all datasets, so that the only 
            # time axis is the *lead time* coorindate.
            targets = targets.drop_vars('datetime')
            forecast = forecast.drop_vars('datetime') 
  
            # Update the forecast vs. targets metrics. 
            self.metrics = [metric.update(forecast, targets) for metric in self.metrics]

            if mrms_dataset:         
                # Identify objects. TODO: Add UH tracks and any other field of interest! 
                # TODO: Also, add args to change these reflectivity thresholds.
                for var, thresh in self.object_id_params['mrms'].items():
                    mrms_dataset = self.object_ider.label(mrms_dataset,var,params={'bdry_thresh' : thresh})
   
                self.targets_v_mrms_metrics = [metric.update(targets, mrms_dataset) 
                                               for metric in self.targets_v_mrms_metrics]
                self.forecast_v_mrms_metrics = [metric.update(forecast, mrms_dataset) 
                                                for metric in self.forecast_v_mrms_metrics]
            
            # Update against WoFS analysis
            if wofs_analysis_dataset:
                self.targets_v_analysis_metics = [metric.update(targets, wofs_analysis_dataset) 
                                             for metric in self.targets_v_analysis_metrics ]
                
                self.forecast_v_analysis_metrics = [metric.update(forecast, wofs_analysis_dataset) 
                                                 for metric in self.forecast_v_analysis_metrics]
        
        metrics_ds = [metric.finalize() for metric in self.metrics]
        
        # Finalize WoFS and WoFSCast metrics and append to the main list
        metrics_ds.extend([
            *[metric.finalize() for metric in self.targets_v_analysis_metrics],
            *[metric.finalize() for metric in self.forecast_v_analysis_metrics],
            *[metric.finalize() for metric in self.targets_v_mrms_metrics],
            *[metric.finalize() for metric in self.forecast_v_mrms_metrics],
        ])

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
    
        print(f'Saved results dataset to {out_path}!')
    
        return f'Saved results dataset to {out_path}!'