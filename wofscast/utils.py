import numpy as np 
import jax.numpy as jnp 

import multiprocessing as mp
import itertools
from multiprocessing.pool import Pool
from datetime import datetime
from tqdm import tqdm  
import traceback
from collections import ChainMap
import warnings
from copy import copy


def count_total_parameters(params_dict):
    """
    Count the total number of parameters in a nested dictionary of parameters.
    Assumes that the dictionary contains `Array` objects that have a `size` attribute.
    
    Args:
    - params_dict (dict): A nested dictionary of parameters.
    
    Returns:
    - int: The total number of parameters.
    """
    total_params = 0

    # Define a helper function to recurse through the dictionary
    def recurse_through_dict(d):
        nonlocal total_params
        for k, v in d.items():
            if isinstance(v, dict):
                recurse_through_dict(v)  # Recurse if value is a dictionary
            else:
                # Assume that the object has a 'size' attribute
                total_params += v.size
    
    recurse_through_dict(params_dict)
    return total_params

def flatten_dict(d, parent_key='', sep='//'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def save_model_params(d, file_path):
    flat_dict = flatten_dict(d)
    # Convert JAX arrays to NumPy for saving
    np_dict = {k: np.array(v) if isinstance(v, jnp.ndarray) else v for k, v in flat_dict.items()}
    np.savez(file_path, **np_dict)


def unflatten_dict(d, sep='//'):
    result_dict = {}
    for flat_key, value in d.items():
        keys = flat_key.split(sep)
        d = result_dict
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value
    return result_dict

def load_model_params(file_path):
    with np.load(file_path, allow_pickle=True) as npz_file:
        # Convert NumPy arrays back to JAX arrays
        jax_dict = {k: jnp.array(v) for k, v in npz_file.items()}
    return unflatten_dict(jax_dict)


# Ignore the warning for joblib to set njobs=1 for
# models like RandomForest
warnings.simplefilter("ignore", UserWarning)

class LogExceptions(object):
    def __init__(self, func):
        self.func = func

    def error(self, msg, *args):
        """ Shortcut to multiprocessing's logger """
        return mp.get_logger().error(msg, *args)
    
    def __call__(self, *args, **kwargs):
        try:
            result = self.func(*args, **kwargs)
                    
        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            print(traceback.format_exc())
            self.error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result

def to_iterator(*lists):
    """
    turn list
    """
    return itertools.product(*lists)

def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)

def run_parallel(
    func,
    args_iterator,
    nprocs_to_use,
    description=None,
    kwargs={}, 
):
    """
    Runs a series of python scripts in parallel. Scripts uses the tqdm to create a
    progress bar.
    Args:
    -------------------------
        func : callable
            python function, the function to be parallelized; can be a function which issues a series of python scripts
        args_iterator :  iterable, list,
            python iterator, the arguments of func to be iterated over
                             it can be the iterator itself or a series of list
        nprocs_to_use : int or float,
            if int, taken as the literal number of processors to use
            if float (between 0 and 1), taken as the percentage of available processors to use
        kwargs : dict
            keyword arguments to be passed to the func
    """
    iter_copy = copy(args_iterator)
    
    total = len(list(iter_copy))
    pbar = tqdm(total=total, desc=description)
    results = [] 
    def update(*a):
        # This is called whenever a process returns a result.
        # results is modified only by the main process, not by the pool workers. 
        pbar.update()
    
    if 0 <= nprocs_to_use < 1:
        nprocs_to_use = int(nprocs_to_use * mp.cpu_count())
    else:
        nprocs_to_use = int(nprocs_to_use)

    if nprocs_to_use > mp.cpu_count():
        raise ValueError(
            f"User requested {nprocs_to_use} processors, but system only has {mp.cpu_count()}!"
        )
        
    pool = Pool(processes=nprocs_to_use)
    ps = []
    for args in args_iterator:
        if isinstance(args, str):
            args = (args,)
         
        p = pool.apply_async(LogExceptions(func), args=args, callback=update)
        ps.append(p)
        
    pool.close()
    pool.join()

    results = [p.get() for p in ps]
    
    return results 


