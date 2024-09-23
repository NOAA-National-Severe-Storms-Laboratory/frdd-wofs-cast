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

BASE_CONFIG_PATH = 'eval_configs'


if __name__ == "__main__":
    
    """ usage: stdbuf -oL python -u evaluate.py --config eval_config.yaml > & logs/log_eval & """
    
    
    args = parse_arguments()
    config_dict = load_configuration(BASE_CONFIG_PATH, args.config)
    config = EvaluatorConfig(**config_dict)

    
    # Selecting a single ensemble member. 
    paths = glob(config.data_path)
    paths.sort()
    
    # Randomly subsample the total paths. 
    n_samples=config.n_samples
    rs = np.random.RandomState(config.seed)
    random_paths = rs.choice(paths, size=n_samples, replace=False)

    model = Predictor(
        model_path = config.model_path,
        add_diffusion=config.add_diffusion 
    )

    data_loader = WoFSDataLoader(model.task_config, 
                             model.preprocess_fn, 
                             config.load_ensemble, 
                             model.decode_times)  

    object_ider = ObjectIder()

    metrics = [MSE(),  
               ObjectBasedContingencyStats(config.matching_distance_km / config.grid_spacing_km), 
               PowerSpectra(variables=config.spectra_variables),
               FractionsSkillScore(windows=config.fss_windows, 
                               thresh_dict=config.fss_thresh_dict,
                             variables = config.fss_variables),
               PMMStormStructure(config.pmm_variables)
             ]
    
    evaluator = Evaluator(model, object_ider, data_loader, metrics)
    results_ds = evaluator.evaluate(random_paths)
    evaluator.save(results_ds, config)
    
    