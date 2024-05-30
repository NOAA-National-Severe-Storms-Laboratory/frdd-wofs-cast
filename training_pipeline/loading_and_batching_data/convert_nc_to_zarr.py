import os
from os.path import join
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import xarray as xr
import zarr
import dask 


def netcdf_to_zarr(netcdf_path, compressor=None, chunk_sizes=None):
    """
    Convert a NetCDF file to Zarr format optimized for I/O speeds.

    Args:
        netcdf_path (str): Path to the input NetCDF file.
        compressor (zarr.Compressor, optional): Zarr compressor to use. Defaults to None.
        chunk_sizes (dict, optional): Dictionary specifying chunk sizes. Defaults to None.

    Returns:
        None
    """
    zarr_path = netcdf_path.replace('datasets_2hr', 'datasets_2hr_zarr').replace('.nc', '.zarr')
    
    if os.path.exists(zarr_path):
        return 'Done' 
    
    # Open the NetCDF file
    ds = xr.open_dataset(netcdf_path, chunks=chunk_sizes)

    # If no compressor is specified, use the default compressor
    if compressor is None:
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)

    # Set encoding for each variable to use the specified compressor
    encoding = {var: {'compressor': compressor} for var in ds.data_vars}

    # Write the dataset to Zarr format
    # Ensure output directory exists
    if not os.path.exists(os.path.dirname(zarr_path)):
        os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
    
    ds.to_zarr(zarr_path, mode='w', encoding=encoding, consolidated=True)
    ds.close()
    
    print(f'Saved {zarr_path}...')
    
    return 'Done'

if __name__ == '__main__':

    """ usage: stdbuf -oL python -u convert_nc_to_zarr.py > & log_nc_to_zarr & """
    base_path = '/work/mflora/wofs-cast-data/datasets_2hr'
    years = ['2019', '2020', '2021']

    def get_files_for_year(year):
        year_path = join(base_path, year)
        with os.scandir(year_path) as it:
            return [join(year_path, entry.name) for entry in it if entry.is_file()]

    with ThreadPoolExecutor() as executor:
        paths = []
        for files in executor.map(get_files_for_year, years):
            paths.extend(files)

    print(len(paths))
    
    # Using Dask's delayed execution with tqdm progress bar
    
    # Chunking the larger dataset by time, so that I/O in the future 
    # can efficiently load time subsets. 
    
    delayed_tasks = [dask.delayed(netcdf_to_zarr)(u, chunk_sizes={'time' : 1, 'datetime' : 1}) for u in paths]
    with tqdm(total=len(delayed_tasks), desc="Converting NetCDF to Zarr") as pbar:
        results = dask.compute(*delayed_tasks, scheduler='processes')
        for _ in results:
            pbar.update(1)