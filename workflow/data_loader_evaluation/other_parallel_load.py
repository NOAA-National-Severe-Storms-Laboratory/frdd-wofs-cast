import dask
import xarray as xr
from dask.distributed import Client, as_completed
import dask.array as da
from typing import List, Optional
import time
import os
from os.path import join
from concurrent.futures import ThreadPoolExecutor

#def load_zarr_file(zarr_path: str, chunks: Optional[dict] = None) -> xr.Dataset:
#    """Load a single Zarr file with optional chunking."""
#    ds = xr.open_dataset(zarr_path, engine='zarr', consolidated=True, chunks=chunks)
#    return ds

def load_zarr_file(zarr_path: str, chunks=None) -> xr.Dataset:
    """Load a single Zarr file as a Dask array."""
    da_array = da.from_zarr(zarr_path)
    ds = xr.Dataset({'data': da_array})
    return ds



def load_and_concatenate_zarr_files_dask(paths: List[str], concat_dim: str = 'model_run_id', 
                                         chunks: Optional[dict] = None, max_workers: int = 4) -> xr.Dataset:
    """
    Load multiple Zarr files in parallel using Dask and concatenate them along a specified dimension.

    Args:
        paths (List[str]): List of paths to Zarr files.
        concat_dim (str): Dimension along which to concatenate the datasets.
        chunks (dict, optional): Dictionary specifying chunk sizes for xarray.
        max_workers (int): Maximum number of parallel workers.

    Returns:
        xr.Dataset: Concatenated xarray dataset.
    """
    start_time = time.time()  # Start timing
    
    # Initialize Dask client
    client = Client(n_workers=max_workers, threads_per_worker=1)
    
    # Use Dask's delayed execution to read Zarr files in parallel
    delayed_datasets = [dask.delayed(load_zarr_file)(path, chunks) for path in paths]
    
    # Compute the delayed datasets using the Dask client
    datasets = client.compute(delayed_datasets, sync=True)
    
    # Concatenate the datasets along the specified dimension
    concatenated_dataset = xr.concat(datasets, dim=concat_dim)
    
    end_time = time.time()  # End timing
    print(f"Time taken to load and concatenate {len(paths)} files: {end_time - start_time:.2f} seconds")
    
    # Optionally: Shutdown the Dask client
    client.close()
    
    return concatenated_dataset

# Example usage
if __name__ == "__main__":
    base_path = '/work/mflora/wofs-cast-data/datasets_zarr'
    years = ['2019']
    
    def get_files_for_year(year: str, base_path: str) -> List[str]:
        year_path = join(base_path, year)
        with os.scandir(year_path) as it:
            return [join(year_path, entry.name) for entry in it if entry.is_dir() and entry.name.endswith('.zarr')]

    with ThreadPoolExecutor() as executor:
        paths = []
        for files in executor.map(get_files_for_year, years, [base_path]*len(years)):
            paths.extend(files)

    print(f"Number of paths: {len(paths)}")
    
    chunks = {}  # Example chunk sizes, adjust as needed

    concatenated_dataset = load_and_concatenate_zarr_files_dask(paths[:1024], 
                                                                concat_dim='model_run_id', 
                                                                chunks=chunks, max_workers=4)
    
    start_time = time.time()  # Start timing

    # If early_compute is False, this ensures the computation happens here
    concatenated_dataset = concatenated_dataset.compute()
    
    end_time = time.time()  # End timing
    print(f"Time taken to load {len(paths)} files: {end_time - start_time:.2f} seconds")
