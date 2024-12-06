import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
# Set this lower, to allow for PyTorch Model to fit into memory
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.90' 

import sys
package_path = os.path.dirname(os.path.dirname(os.getcwd())) 
sys.path.insert(0, package_path)
from glob import glob 

from wofscast.common.wofs_data_loader import WoFSDataLoader
from wofscast.common.helpers import parse_arguments, load_configuration 

from wofscast.evaluate.metrics import (MSE,
                                       ObjectBasedContingencyStats,
                                       PowerSpectra,
                                       FractionsSkillScore,
                                       PMMStormStructure,
                                       )

from wofscast.evaluate.predictor import Predictor
from wofscast.evaluate.object_ider import ObjectIder
from wofscast.evaluate.evaluator import Evaluator, EvaluatorConfig 

import numpy as np
import argparse 
try:
    import torch 
    torch_imported = True
except:
    print('torch cannot be imported!') 
    torch_imported=False
    
BASE_CONFIG_PATH = 'eval_configs'


if __name__ == "__main__":
    
    """ usage: stdbuf -oL python -u evaluate.py --config eval_config-diffusion.yaml > & logs/log_eval & """
    """ usage: stdbuf -oL python -u evaluate.py --config eval_config.yaml > & logs/log_eval & """
    
    compute_percentile_thresholds = False 
    replace_bdry=True 
    
    args = parse_arguments()
    config_dict = load_configuration(BASE_CONFIG_PATH, args.config)
    config = EvaluatorConfig(**config_dict)

    # Selecting a single ensemble member. 
    paths = glob(config.data_path)
    paths.sort()
    
    # Randomly subsample the total paths. 
    n_samples=config.n_samples
    rs = np.random.RandomState(config.seed)
    if n_samples >= len(paths):
        random_paths = paths 
    else:
        random_paths = rs.choice(paths, size=n_samples, replace=False)

    print(f'{len(random_paths)=} {len(paths)=}')    
        
    # Hardcoded diffusion options. TODO: add them to the config.yaml.
    # Randy said the generation parameters are crucial! 
    # Increasing S_churn is super important!!
    sampler_kwargs = dict(
                    sigma_min = 0.0002,
                    sigma_max = 1000, 
                    S_churn=2.0, 
                    S_min=0.02, 
                    S_max=800, 
                    S_noise=1.05)
    
    if torch_imported:
        device = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda'
    else:
        device = 'cuda'
    
    model = Predictor(
        model_path = config.model_path,
        add_diffusion=config.add_diffusion,
        sampler_kwargs=sampler_kwargs,
        diffusion_device=device
    )

    time_range = slice('10min', '320min') if '6hr' in paths[0] else slice('10min', '110min')
    print(f'{time_range=}')
    
    data_loader = WoFSDataLoader(model.task_config, 
                             model.preprocess_fn, 
                             config.load_ensemble, 
                             model.decode_times,
                             time_range = time_range   
                                )  

    object_ider = ObjectIder()

    metrics = [MSE(),  
               ObjectBasedContingencyStats(config.matching_distance_km / config.grid_spacing_km), 
               PowerSpectra(variables=config.spectra_variables),
               FractionsSkillScore(windows=config.fss_windows, 
                               thresh_dict=config.fss_thresh_dict,
                             variables = config.fss_variables),
               PMMStormStructure(config.pmm_variables)
             ]
    
    mrms_variables = ['accum_rain']
    
    forecast_dbz_thresh = config.object_id_params['forecast']['COMPOSITE_REFL_10CM']
    targets_dbz_thresh = config.object_id_params['truth']['COMPOSITE_REFL_10CM']
    mrms_dbz_thresh = config.object_id_params['mrms']['dz_consv']
    
    targets_v_mrms_metrics = [
                            FractionsSkillScore(windows=[7, 15, 27, 30], 
                                            # Target, MRMS thresholds  
                                            thresh_dict={'COMPOSITE_REFL_10CM' : [(targets_dbz_thresh , mrms_dbz_thresh)],
                                                         'accum_rain' : [0.5], 
                                                        },
                                             variables = ['COMPOSITE_REFL_10CM', 'accum_rain'],
                                             addon='targets_vs_mrms'              
                                                          ),
                            ObjectBasedContingencyStats(key='targets_vs_mrms'),
                            MSE(addon='targets_vs_mrms', variables=mrms_variables), 
    ]

    forecast_v_mrms_metrics = [ 
                          FractionsSkillScore(windows=[7, 15, 27, 30], 
                                             # Forecast, MRMS thresholds  
                                            thresh_dict={'COMPOSITE_REFL_10CM' : [(forecast_dbz_thresh , mrms_dbz_thresh)],
                                                         'accum_rain' : [0.5], 
                                                        },
                                             variables = ['COMPOSITE_REFL_10CM', 'accum_rain'],
                                             addon='forecast_vs_mrms'             
                                                           ),
                          ObjectBasedContingencyStats(key='forecasts_vs_mrms'),
                          MSE(addon='forecast_vs_mrms', variables=mrms_variables)
    ]

    # Metrics for comparing against an analysis dataset 
    targets_v_analysis_metrics = [MSE(addon='targets_vs_analysis')]
    forecast_v_analysis_metrics = [MSE(addon='forecast_vs_analysis')]
    
    evaluator = Evaluator(model, object_ider, data_loader, metrics, config.object_id_params,
                           forecast_v_mrms_metrics, 
                           targets_v_mrms_metrics, 
                           forecast_v_analysis_metrics,
                           targets_v_analysis_metrics,
                           replace_bdry=replace_bdry
                         )
    
    if compute_percentile_thresholds:
        results_ds = evaluator.compute_percentile_thresholds(random_paths)
    else:
        results_ds = evaluator.evaluate(random_paths)
        evaluator.save(results_ds, config)
    
    
