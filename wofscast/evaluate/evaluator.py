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

        
class Evaluator:
    
    DIM_ORDER = ('batch', 'time', 'level', 'lat', 'lon')
    
    def __init__(self, model, object_ider, data_loader, metrics, 
                 unit_converter_funcs = [convert_T2_K_to_F, convert_rain_amount_to_inches]
                ): 
        self.model = model 
        self.data_loader = data_loader 
        self.metrics = metrics 
        
        # Metrics for comparing against an analysis dataset 
        self.targets_v_analysis_metics = [MSE(addon='targets_vs_analysis')]
        self.forecast_v_analysis_metrics = [MSE(addon='forecast_vs_analysis')]
        
        # Metrics for comparing against MRMS. 
        variables = ['accum_rain'] 
        self.targets_v_mrms_metrics = [MSE(addon='targets_vs_mrms', variables=variables), 
                                       #FractionsSkillScore(windows=[7, 15, 27], 
                                       #      thresh_dict={'accum_rain' : [0.5]},
                                       #      variables = ['accum_rain']),
                                       ObjectBasedContingencyStats(key='targets_vs_mrms'),
                                      ]
        self.forecast_v_mrms_metrics = [MSE(addon='forecast_vs_mrms', variables=variables), 
                                        #FractionsSkillScore(windows=[7, 15, 27], 
                                        #    thresh_dict={'accum_rain' : [0.5]},
                                        #     variables = ['accum_rain']),
                                        ObjectBasedContingencyStats(key='forecasts_vs_mrms'),
                                        ]
               
        self.object_ider = object_ider
        self.unit_converter_funcs = unit_converter_funcs
        
        self._var_dim_map = {'U': 'west_east_stag', 
                       'V': 'south_north_stag',
                       'W': 'bottom_top_stag',
                       'GEOPOT': 'bottom_top_stag'
                      }
        
    def add_initial_conditions(self, dataset, inputs):
        return xr.concat([inputs.isel(time=-1), dataset], dim='time')

    def load_mrms_data(self, forecast, data_path, mode='refl'):
        # Load the MRMS composite reflectivity. Loads all the separate time steps into 
        # one dataset. 
        case_date = get_case_date(data_path)
        dts = pd.to_datetime(forecast.datetime) 
        loader = MRMSDataLoader(case_date, domain_size=150, resize_domain=True) 

        try: 
            mrms_dataset = loader.load(forecast, mode=mode) 
            return mrms_dataset
        except OSError:
            print(f'Unable to load MRMS data for {data_path}, Skipping it...')
            return None 
    
    def load_analysis_data(self, forecast, data_path):
        # Load the WoFS analyses. Loads all the separate time steps into 
        # one dataset. 
        case_date = get_case_date(data_path)
        dts = pd.to_datetime(forecast.datetime)  
        
        anal_loader = WoFSAnalysisLoader()
        variables = list(forecast.data_vars)
        variables.remove('RAIN_AMOUNT') 
        variables.remove('WMAX') 

        #try:
        anal_ds = anal_loader.load(forecast, dts, case_date, mem=9)
        return anal_ds
        #except:
        #    print(f'Unable to load WoFS analys for {data_path}, Skipping it...')
        #    return None 
    
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
        
        return forecast, targets 
    
    def evaluate(self, paths):
        
        wofs_dbz_thresh = 47  
            
        for data_path in tqdm(paths): 
    
            inputs, targets, forcings = self.data_loader.load_inputs_targets_forcings(data_path)
            forecast = self.model.predict(inputs, targets, forcings)
    
            forecast = self.add_initial_conditions(forecast, inputs)
            targets = self.add_initial_conditions(targets, inputs)
    
            # Add derived variables to forecast & targets; ID storm objects. 
            forecast, targets = self.add_derived_variables(forecast, targets)
            forecast = self.object_ider.label(forecast, 'COMPOSITE_REFL_10CM', params={'bdry_thresh' : wofs_dbz_thresh})
            targets = self.object_ider.label(targets, 'COMPOSITE_REFL_10CM', params={'bdry_thresh' : wofs_dbz_thresh})
    
            # Load the WoFS analysis and MRMS radar and QPE variables. 
            wofs_analysis_dataset = self.load_analysis_data(forecast, data_path) 
            mrms_dataset = self.load_mrms_data(forecast, data_path)
            mrms_qpe = self.load_mrms_data(forecast, data_path, mode='qpe')
            
            # Convert units. 
            forecast = self.add_units(forecast)
            targets = self.add_units(targets)
    
            # Transpose to the same dimension order 
            targets = targets.transpose(*self.DIM_ORDER)
            forecast = forecast.transpose(*self.DIM_ORDER)

            # Update the forecast vs. targets metrics. 
            self.metrics = [metric.update(forecast, targets) for metric in self.metrics]

            if mrms_dataset: 
                mrms_dataset = mrms_dataset.transpose(*self.DIM_ORDER, missing_dims='ignore')
                mrms_qpe = mrms_qpe.expand_dims('batch', axis=0)
                
                # Identify objects. TODO: Add UH tracks and any other field of interest! 
                # TODO: Also, add args to change these reflectivity thresholds.
                mrms_dataset = self.object_ider.label(mrms_dataset, 'dz_consv', params={'bdry_thresh' : 40})
   
                # Update object matching statistics. 
                self.targets_v_mrms_metrics[-1].update(targets, mrms_dataset)
                self.forecast_v_mrms_metrics[-1].update(forecast, mrms_dataset)
                
                # Update MRMS QPE RMSE. First, accumulate rainfall. TODO: generalize this better! 
                try:
                    mrms_qpe['accum_rain'] = mrms_qpe['qpe_consv'].sum(dim='time')
                except:
                    mrms_qpe['accum_rain'] = mrms_qpe['qpe_consv_15m'].sum(dim='time')
                    
                forecast['accum_rain'] = forecast['RAIN_AMOUNT'].sum(dim='time')
                targets['accum_rain'] = targets['RAIN_AMOUNT'].sum(dim='time')
                
                
                #for i in [0,1]:
                self.targets_v_mrms_metrics[0].update(targets, mrms_qpe)
                self.forecast_v_mrms_metrics[0].update(forecast, mrms_qpe)
            
            # Update against WoFS analysis
            if wofs_analysis_dataset:
                wofs_analysis_dataset = wofs_analysis_dataset.expand_dims('batch', axis=0)
                wofs_analysis_dataset = self.add_units(wofs_analysis_dataset, unit_converter_funcs = [convert_T2_K_to_F])
                wofs_analysis_dataset['WMAX'] = wofs_analysis_dataset['W'].max(dim='level')

                wofs_analysis_dataset = wofs_analysis_dataset.transpose(*self.DIM_ORDER)                                                          
                self.targets_v_analysis_metics = [metric.update(targets, wofs_analysis_dataset) 
                                             for metric in self.targets_v_analysis_metics ]
                self.forecast_v_analysis_metrics = [metric.update(forecast, wofs_analysis_dataset) 
                                                 for metric in self.forecast_v_analysis_metrics]
            
        metrics_ds = [metric.finalize() for metric in self.metrics]
        
        # Finalize WoFS and WoFSCast metrics and append to the main list
        metrics_ds.extend([
            *[metric.finalize() for metric in self.targets_v_analysis_metics],
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