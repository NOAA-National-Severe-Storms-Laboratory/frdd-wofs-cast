from ..data_generator import load_chunk, dataset_to_input 
from .helpers import get_case_date
from glob import glob


class WoFSDataLoader:
    def __init__(self, task_config, preprocess_fn=None, load_ensemble=True, decode_times=False):
        self.task_config = task_config
        self.load_ensemble = load_ensemble
        self.preprocess_fn = preprocess_fn
        self._case_date = None
        self.decode_times = decode_times

    def get_target_slice_range(self):
        """Returns the slice range for target lead times based on the timestep."""
        #if self.config.timestep == 5:
        #    return slice('5min', '100min')
        return slice('10min', '120min')
        
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
        
        self._case_date = get_case_date(paths[0])
        
        target_lead_times = self.get_target_slice_range()
        
        inputs, targets, forcings = dataset_to_input(
            dataset, self.task_config, 
            target_lead_times=target_lead_times, 
            batch_over_time=False, 
            n_target_steps=2
        )
        
        return inputs, targets, forcings

