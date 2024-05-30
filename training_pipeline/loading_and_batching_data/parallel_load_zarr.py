import xarray as xr
import fsspec
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import List, Optional

import os
from os.path import join
from concurrent.futures import ThreadPoolExecutor
import time 

def load_zarr_file(zarr_path: str, chunks: Optional[dict] = None) -> xr.Dataset:
    """Load a single Zarr file with optional chunking."""
    ds = xr.open_dataset(zarr_path, engine='zarr', 
                         consolidated=True, # Improve I/O!  
                         chunks=chunks)
    return ds

def load_and_concatenate_zarr_files(zarr_paths: List[str], concat_dim: str = 'batch', 
                                    chunks: Optional[dict] = None, max_workers: int = 4) -> xr.Dataset:
    """
    Load multiple Zarr files in parallel using multiprocessing and concatenate them along a specified dimension.

    Args:
        zarr_paths (List[str]): List of paths to Zarr files.
        concat_dim (str): Dimension along which to concatenate the datasets.
        chunks (dict, optional): Dictionary specifying chunk sizes for xarray.
        max_workers (int): Maximum number of processes to use for parallel loading.

    Returns:
        xr.Dataset: Concatenated xarray dataset.
    """
    start_time = time.time()  # Start timing
    
    datasets = []

    # Create a ProcessPoolExecutor with spawn context to avoid os.fork() issues
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context('spawn')) as executor:
        # Submit all the tasks to the executor
        future_to_zarr = {executor.submit(load_zarr_file, zarr_path, chunks): zarr_path for zarr_path in zarr_paths}
        
        # Collect the results as they complete
        for future in as_completed(future_to_zarr):
            zarr_path = future_to_zarr[future]
            try:
                ds = future.result()
                datasets.append(ds)
                ds.close()
            except Exception as e:
                print(f"Error loading {zarr_path}: {e}")

    # Concatenate the datasets along the specified dimension
    if datasets:
        concatenated_dataset = xr.concat(datasets, dim=concat_dim)
    else:
        raise ValueError("No datasets were loaded successfully.")
        
    end_time = time.time()  # End timing
    print(f"Time taken to load and concatenate {len(zarr_paths)} files: {end_time - start_time:.2f} seconds")    
        
    return concatenated_dataset   

# Example usage
if __name__ == "__main__":
    base_path = '/work/mflora/wofs-cast-data/datasets_jsons'#_zarr'
    years = ['2019']#, '2020']

    def get_files_for_year(year):
        year_path = join(base_path, year)
        with os.scandir(year_path) as it:
            #return [join(year_path, entry.name) for entry in it if entry.is_dir() and entry.name.endswith('.zarr')]
            return [join(year_path, entry.name) for entry in it if entry.is_file()]
    
    with ThreadPoolExecutor() as executor:
        paths = []
        for files in executor.map(get_files_for_year, years):
            paths.extend(files)

    print(len(paths))

    chunks = {}#'time': 1, 'latitude': 50, 'longitude': 50}  # Example chunk sizes, adjust as needed

    concatenated_dataset = load_and_concatenate_zarr_files(paths[:1024], concat_dim='batch', chunks=chunks, max_workers=4)
     
    start_time = time.time()  # Start timing

    concatenated_dataset = concatenated_dataset.compute() 
    
    end_time = time.time()  # End timing
    print(f"Time taken to load {len(paths)} files: {end_time - start_time:.2f} seconds")    
    
    
    # Load 1025 files in 17.33 + 6.85 = 24.18 s, workers=4 
    # workers = 12, +2 seconds
    
    

