import os 
from datetime import datetime, timedelta
import pandas as pd
import numpy as np 
from scipy.ndimage import maximum_filter
import xarray as xr
import argparse 
from ..utils import load_yaml 

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Load and parse configuration file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config.yaml file relative to the base config directory.')
    return parser.parse_args()

def load_configuration(base_config_path, config_file):
    """Load the YAML configuration file."""
    config_path = os.path.join(base_config_path, config_file)
    return load_yaml(config_path)


def convert_rain_amount_to_inches(ds):
    """
    Converts the RAIN_AMOUNT variable from millimeters to inches in an xarray dataset.

    Parameters:
    ds (xarray.Dataset): The input dataset containing the RAIN_AMOUNT variable in millimeters.

    Returns:
    xarray.Dataset: The dataset with RAIN_AMOUNT converted to inches.
    """
    if 'RAIN_AMOUNT' in ds:
        ds['RAIN_AMOUNT'] = ds['RAIN_AMOUNT'] / 25.4  # 1 inch = 25.4 mm
        ds['RAIN_AMOUNT'].attrs['units'] = 'inches'
    else:
        raise ValueError("The dataset does not contain a variable named 'RAIN_AMOUNT'.")
    
    return ds


def convert_T2_K_to_F(ds):
    """
    Converts the T2 variable from Kelvin to degrees Fahrenheit in an xarray dataset.

    Parameters:
    ds (xarray.Dataset): The input dataset containing the T2 variable in Kelvin.

    Returns:
    xarray.Dataset: The dataset with T2 converted to degrees Fahrenheit.
    """
    if 'T2' in ds:
        ds['T2'] = (ds['T2'] - 273.15) * 9/5 + 32
        ds['T2'].attrs['units'] = 'deg F'
    else:
        raise ValueError("The dataset does not contain a variable named 'T2'.")
    
    return ds


def get_case_date(path):
    name = os.path.basename(path)
    comps = name.split('_')
    
    start_date = comps[1]+'_'+comps[2]
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d_%H%M%S')
    
    if start_date_dt.hour < 14:
        case_date = start_date_dt.date() - timedelta(days=1)
    else:
        case_date = start_date_dt.date() 
        
    return case_date.strftime('%Y%m%d')
    

def to_datetimes(path, n_times = 13):  
    name, freq, ens_mem = os.path.basename(path).split('__')
    start_time_dt = datetime.strptime(name.split('_to')[0], 'wrfwof_%Y-%m-%d_%H%M%S')
    start_time = pd.Timestamp(start_time_dt)
    
    dt_list = pd.date_range(start=start_time, periods=n_times, freq=freq)
    return dt_list[2:]

def get_qpe_datetimes(start_time, n_times = 13):  
    start_time = pd.Timestamp(start_time_dt)
    dt_list = pd.date_range(start=start_time, periods=n_times, freq=freq)
    return dt_list[2:]


# Assuming 'BORDER_MASK' is available and correctly sized for your 'preds' and 'tars'
def border_difference_check(preds, tars, border_mask):
    """Calculate the difference at the border and return a mask of differences."""
    border_diff = np.abs(preds - tars)
    # Apply the border mask to get differences only at the border
    border_diff_masked = np.where(border_mask, border_diff, np.nan)  # NaN where not border
    return np.nanmax(border_diff_masked)  # Get the maximum difference at the border

def _border_mask(shape, N=10):
    """
    Create a border mask for an array of given shape.

    Parameters:
    - shape: tuple, the shape of the array (NY, NX).
    - N: int, the width of the border where values should be True.

    Returns:
    - mask: jax.numpy.ndarray, a mask where border values are True and interior values are False.
    """
    NY, NX = shape
    mask = np.zeros(shape, dtype=bool)

    # Set the border to True
    mask[:N, :] = True  # Top border
    mask[-N:, :] = True  # Bottom border
    mask[:, :N] = True  # Left border
    mask[:, -N:] = True  # Right border

    return mask

def apply_maximum_filter(data, size=3):
    """
    Apply scipy's maximum filter to 2D data.

    Parameters:
    - data: 2D numpy array
    - size: size of the filter

    Returns:
    - filtered_data: 2D numpy array with the maximum filter applied
    """
    return maximum_filter(data, size=size)


def compute_nmep(dataset, var, threshold, filter_size=3):
    """
    Compute the Neighborhood Ensemble Probability (NMEP) for a given variable over all time steps using xarray,
    apply a maximum filter, and then threshold the results.

    Parameters:
    - dataset: xarray.Dataset, the input dataset containing the ensemble data.
    - var: str, the variable name to compute NMEP for.
    - threshold: float, the threshold value to binarize the data.
    - filter_size: int, the size of the maximum filter to apply.

    Returns:
    - xarray.Dataset, the dataset with the computed NMEP added as a new variable for each time step.
    """
    # Apply the maximum filter to the data
    filtered_data = xr.apply_ufunc(
        apply_maximum_filter,
        dataset[var],
        input_core_dims=[['lat', 'lon']],
        output_core_dims=[['lat', 'lon']],
        vectorize=True,
        kwargs={'size': filter_size}
    )
    
    # Binarize the data based on the threshold
    data_binary = filtered_data >= threshold
    
    # Compute the probability (mean) across the ensemble dimension for each time step
    # Replace 'batch' with the actual ensemble dimension name
    probs = data_binary.mean(dim='batch')
    
    # Add the computed NMEP to the dataset as a new variable
    dataset[f'{var}_nmep'] = probs
    
    return dataset

