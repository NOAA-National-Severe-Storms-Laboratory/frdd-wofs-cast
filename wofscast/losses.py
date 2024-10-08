# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loss functions (and terms for use in loss functions) used for weather."""

from typing import Mapping

from . import xarray_tree
import numpy as np
from typing_extensions import Protocol
import xarray

import xarray
import jax


LossAndDiagnostics = tuple[xarray.DataArray, xarray.Dataset]


class LossFunction(Protocol):
    """A loss function.

    This is a protocol so it's fine to use a plain function which 'quacks like'
    this. This is just to document the interface.
    """

    def __call__(
        self, predictions: xarray.Dataset, targets: xarray.Dataset, **optional_kwargs
    ) -> LossAndDiagnostics:
        """Computes a loss function.

        Args:
          predictions: Dataset of predictions.
          targets: Dataset of targets.
          **optional_kwargs: Implementations may support extra optional kwargs.

        Returns:
          loss: A DataArray with dimensions ('batch',) containing losses for each
            element of the batch. These will be averaged to give the final
            loss, locally and across replicas.
          diagnostics: Mapping of additional quantities to log by name alongside the
            loss. These will will typically correspond to terms in the loss. They
            should also have dimensions ('batch',) and will be averaged over the
            batch before logging.
        """


def threshold_tuned_loss(
    target,
    prediction,
    small_val_thresh,
    mid_val_thresh,
    large_val_thresh,
    small_val_penalty=10.0,
    mid_val_penalty=15.0,
    large_val_penalty=5.0,
):
    """
    Custom loss function that penalizes underpredictions above a certain threshold and overpredictions below a certain threshold.

    Parameters:
    - target: xarray.DataArray representing the true target values.
    - prediction: xarray.DataArray representing the predicted values.
    - large_val_thresh: Threshold above which underpredictions are heavily penalized.
    - small_val_thresh: Threshold below which overpredictions are heavily penalized.

    Returns:
    - loss: The calculated loss as an xarray.DataArray.
    """

    # Masks
    overpredict_small_val_mask = (prediction > target) & (target <= small_val_thresh)
    underpredict_mid_val_mask = (prediction < target) & (target <= mid_val_thresh)
    underpredict_large_val_mask = (prediction < target) & (target > large_val_thresh)

    # Loss calculations
    overpredict_loss = xarray.where(
        overpredict_small_val_mask, (prediction - target) * small_val_penalty, 0
    )
    underpredict_loss_mid = xarray.where(
        underpredict_mid_val_mask, (target - prediction) * mid_val_penalty, 0
    )
    underpredict_loss_large = xarray.where(
        underpredict_large_val_mask, (target - prediction) * large_val_penalty, 0
    )

    # Combine loss components
    combined_loss = overpredict_loss + underpredict_loss_mid + underpredict_loss_large

    return combined_loss


def custom_loss(predictions, targets):
    """Custom loss equation that heavily penalizes over and under prediction for a subset of variables,
    but otherwise, relies on MSE.

    predictions and targets are DataArray objects
    """
    custom_loss_params = {
        "COMPOSITE_REFL_10CM": {
            "small_val_thresh": 5.0,  # dBZ
            "mid_val_thresh": 15.0,  # dBZ
            "large_val_thresh": 40,
        },
        "UP_HELI_MAX": {
            "small_val_thresh": 5.0,  # UH units
            "mid_val_thresh": 15.0,  # UH units
            "large_val_thresh": 60,
        },
        "RAINNC": {
            "small_val_thresh": 0.1,  # mm
            "mid_val_thresh": 5.0,  # mm
            "large_val_thresh": 25.4,  # mm
        },
    }

    if predictions.name in custom_loss_params.keys():
        params = custom_loss_params[predictions.name]
        loss = threshold_tuned_loss(targets, predictions, **params)
    else:
        loss = (predictions - targets) ** 2

    return _mean_preserving_batch(loss)


def weighted_loss(predictions, targets):
    # Calculate the element-wise squared error
    mse_loss = (predictions - targets) ** 2

    # Create a mask to ignore target values less than or equal to 10
    mask = targets > 10

    # Apply additional weighting where the target values are greater than 10
    weight = xarray.where(mask, 10.0, 1.0).astype('bfloat16') 
    weighted_mse_loss = mse_loss * weight

    # Ignore the "correct" predictions of nothing (target <= 10)
    final_loss = weighted_mse_loss.where(mask, 0.0)

    return final_loss     
            

def limited_area_loss(predictions, targets):
    
    # Inner most 150 x 150 
    pred = predictions.isel(lat=slice(75, 225), lon=slice(75, 225))
    tars = targets.isel(lat=slice(75, 225), lon=slice(75, 225))
                            
    return (pred - tars) ** 2
                        
def weighted_mse_per_level(
    predictions: xarray.Dataset,
    targets: xarray.Dataset,
    per_variable_weights: Mapping[str, float],
) -> LossAndDiagnostics:
    """Latitude- and pressure-level-weighted MSE loss."""

    def loss(prediction, target):
        loss = (prediction - target) ** 2

        # For GenCast
        # loss = weight * ((prediction - target)**2) 
        # so weight needs to be a xarray_jax DataArray. 
        
        # print('No longer doing latitude-based weighting in the loss')
        # loss *= normalized_latitude_weights(target).astype(loss.dtype)

        # print('Turned off the pressure-level weighted loss')
        # if 'level' in target.dims:
        #  loss *= normalized_level_weights(target).astype(loss.dtype)

        return _mean_preserving_batch(loss)

    losses = xarray_tree.map_structure(loss, predictions, targets)
    #losses = xarray_tree.map_structure(limited_area_loss, predictions, targets)
      
    return sum_per_variable_losses(losses, per_variable_weights)

def _mean_preserving_batch(x: xarray.DataArray) -> xarray.DataArray:
    return x.mean([d for d in x.dims if d != "batch"], skipna=False)

def sum_per_variable_losses(
    per_variable_losses: Mapping[str, xarray.DataArray],
    weights: Mapping[str, float],
) -> LossAndDiagnostics:
    """Weighted sum of per-variable losses."""
    if weights is None:
        weights = {}

    if not set(weights.keys()).issubset(set(per_variable_losses.keys())):
        raise ValueError(
            "Passing a weight that does not correspond to any variable "
            f"{set(weights.keys())-set(per_variable_losses.keys())}"
        )

    weighted_per_variable_losses = {
        name: loss * weights.get(name, 1) for name, loss in per_variable_losses.items()
    }
    
    total = xarray.concat(
        weighted_per_variable_losses.values(), dim="variable", join="exact"
    ).sum("variable", skipna=False) 
    
    return total, per_variable_losses  # pytype: disable=bad-return-type

# Deprecated from the original GraphCast code.
#def normalized_level_weights(data: xarray.DataArray) -> xarray.DataArray:
#    """Weights proportional to pressure at each level."""
#    level = data.coords["level"]
#    return level / level.mean(skipna=False)

def normalized_level_weights(data: xarray.DataArray) -> xarray.DataArray:
    """Weights inversely proportional to level, giving more weight to lower levels."""
    level = data.coords["level"]
    
    # Invert the level values to give more weight to lower levels
    inverted_level = level.max(skipna=False) - level + 1
    
    # Normalize the weights so that they sum to 1 or keep relative weighting
    weights = inverted_level / inverted_level.mean(skipna=False)
    
    return weights



def normalized_latitude_weights(data: xarray.DataArray) -> xarray.DataArray:
    """Weights based on latitude, roughly proportional to grid cell area.

    This method supports two use cases only (both for equispaced values):
    * Latitude values such that the closest value to the pole is at latitude
      (90 - d_lat/2), where d_lat is the difference between contiguous latitudes.
      For example: [-89, -87, -85, ..., 85, 87, 89]) (d_lat = 2)
      In this case each point with `lat` value represents a sphere slice between
      `lat - d_lat/2` and `lat + d_lat/2`, and the area of this slice would be
      proportional to:
      `sin(lat + d_lat/2) - sin(lat - d_lat/2) = 2 * sin(d_lat/2) * cos(lat)`, and
      we can simply omit the term `2 * sin(d_lat/2)` which is just a constant
      that cancels during normalization.
    * Latitude values that fall exactly at the poles.
      For example: [-90, -88, -86, ..., 86, 88, 90]) (d_lat = 2)
      In this case each point with `lat` value also represents
      a sphere slice between `lat - d_lat/2` and `lat + d_lat/2`,
      except for the points at the poles, that represent a slice between
      `90 - d_lat/2` and `90` or, `-90` and  `-90 + d_lat/2`.
      The areas of the first type of point are still proportional to:
      * sin(lat + d_lat/2) - sin(lat - d_lat/2) = 2 * sin(d_lat/2) * cos(lat)
      but for the points at the poles now is:
      * sin(90) - sin(90 - d_lat/2) = 2 * sin(d_lat/4) ^ 2
      and we will be using these weights, depending on whether we are looking at
      pole cells, or non-pole cells (omitting the common factor of 2 which will be
      absorbed by the normalization).

      It can be shown via a limit, or simple geometry, that in the small angles
      regime, the proportion of area per pole-point is equal to 1/8th
      the proportion of area covered by each of the nearest non-pole point, and we
      test for this in the test.

    Args:
      data: `DataArray` with latitude coordinates.
    Returns:
      Unit mean latitude weights.
    """
    latitude = data.coords["lat"]

    if np.any(np.isclose(np.abs(latitude), 90.0)):
        weights = _weight_for_latitude_vector_with_poles(latitude)
    else:
        weights = _weight_for_latitude_vector_without_poles(latitude)

    return weights / weights.mean(skipna=False)


def _weight_for_latitude_vector_without_poles(latitude):
    """Weights for uniform latitudes of the form [+-90-+d/2, ..., -+90+-d/2]."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if not np.isclose(np.max(latitude), 90 - delta_latitude / 2) or not np.isclose(
        np.min(latitude), -90 + delta_latitude / 2
    ):
        raise ValueError(
            f"Latitude vector {latitude} does not start/end at "
            "+- (90 - delta_latitude/2) degrees."
        )
    return np.cos(np.deg2rad(latitude))


def _weight_for_latitude_vector_with_poles(latitude):
    """Weights for uniform latitudes of the form [+- 90, ..., -+90]."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if not np.isclose(np.max(latitude), 90.0) or not np.isclose(
        np.min(latitude), -90.0
    ):
        raise ValueError(
            f"Latitude vector {latitude} does not start/end at +- 90 degrees."
        )
    weights = np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(delta_latitude / 2))
    # The two checks above enough to guarantee that latitudes are sorted, so
    # the extremes are the poles
    weights[[0, -1]] = np.sin(np.deg2rad(delta_latitude / 4)) ** 2
    return weights


def _check_uniform_spacing_and_get_delta(vector):
    diff = np.diff(vector)

    print("Ignoring this error for WoFS! Make sure to come back to this!!!!")
    # if not np.all(np.isclose(diff[0], diff)):
    #  raise ValueError(f'Vector {diff} is not uniformly spaced.')
    return diff[0]
