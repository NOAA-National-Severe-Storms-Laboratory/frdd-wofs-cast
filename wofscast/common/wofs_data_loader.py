from ..data_generator import load_chunk, dataset_to_input 
from .helpers import get_case_date
from glob import glob
import os
import pandas as pd
from datetime import datetime

def to_datetimes(path, n_times):  
    name, freq, ens_mem = os.path.basename(path).split('__')
    
    original_time_str = name.split('_to')[0]
    # Try the first format with underscores
    try:
        start_time_dt = datetime.strptime(original_time_str, 'wrfwof_%Y-%m-%d_%H_%M_%S')
    except ValueError:
        # If the first format fails, try the second format without underscores
        start_time_dt = datetime.strptime(original_time_str, 'wrfwof_%Y-%m-%d_%H%M%S')
        
    start_time = pd.Timestamp(start_time_dt)
    
    dt_list = pd.date_range(start=start_time, periods=n_times, freq=freq)
    
    return dt_list


class WoFSDataLoader:
    def __init__(self, task_config, preprocess_fn=None, load_ensemble=True, decode_times=False, 
                 time_range = slice('10min', '110min') 
                ):
        self.task_config = task_config
        self.load_ensemble = load_ensemble
        self.preprocess_fn = preprocess_fn
        self._case_date = None
        self.decode_times = decode_times
        self.target_lead_times = time_range
           
    def get_paths(self, path):
        """Returns a sorted list of ensemble paths if load_ensemble is True; otherwise returns the path."""
        if self.load_ensemble:
            if isinstance(path, list):
                path = path[0]
                
            # Get all the ensemble members.
            paths = glob(f"{path.split('ens_mem')[0]}*")
            paths.sort()
        else:
            if not isinstance(path, list):
                paths = [path]
                      
        return paths 
    
    @property
    def case_date(self):
        return self._case_date
    
    @property
    def ens_mem(self):
        # Define how to retrieve ensemble member if necessary.
        pass
    
    def load_inputs_targets_forcings(self, path):
        """Loads the input, target, and forcing data based on the specified path."""
        paths = self.get_paths(path)
        
        dataset = load_chunk(paths, 1, self.preprocess_fn, decode_times=self.decode_times)

        dataset = dataset.compute() 
        
        dts = to_datetimes(path, n_times = dataset.dims['time'])
        
        dataset = dataset.assign_coords(datetime = ('time', dts))
        
        self._case_date = get_case_date(paths[0])
        
        inputs, targets, forcings = dataset_to_input(
            dataset, self.task_config, 
            target_lead_times=self.target_lead_times, 
            batch_over_time=False, 
            n_target_steps=2
        )

        return inputs, targets, forcings
