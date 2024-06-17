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
from . import xarray_jax

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve
from jax import vmap, lax

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

def custom_window_loss(
    predictions: xarray.Dataset,
    targets: xarray.Dataset,
    per_variable_weights: Mapping[str, float],
) -> LossAndDiagnostics:

    min_nbrhd=1
    max_nbrhd=1
    mu_nbrhds = range(min_nbrhd, max_nbrhd+1, 2)
    wgts = [1.0 for nbrhd in mu_nbrhds] 
    mu_wgts = [wgt/sum(wgts) for wgt in wgts]

    min_nbrhd=0
    max_nbrhd=-1
    med_nbrhds = range(min_nbrhd, max_nbrhd+1, 2)
    wgts = [1.0 for nbrhd in med_nbrhds]
    med_wgts = [wgt/sum(wgts) for wgt in wgts]

    min_nbrhd=0#5#3
    max_nbrhd=-1#5
    max_nbrhds = range(min_nbrhd, max_nbrhd+1, 2)
    wgts = [1.0 for nbrhd in max_nbrhds]
    max_wgts = [wgt/sum(wgts) for wgt in wgts]

    min_nbrhd=0#5
    max_nbrhd=-1#5
    ssim_nbrhds = range(min_nbrhd, max_nbrhd+1, 2)
    wgts = [1.0 for nbrhd in ssim_nbrhds]
    ssim_wgts = [wgt/sum(wgts) for wgt in wgts]

    min_nbrhd=0#5
    max_nbrhd=-1#5
    perc_nbrhds = range(min_nbrhd, max_nbrhd+1, 2)
    perc_vals = [10, 90]
    wgts = [1.0] * (len(perc_nbrhds)*len(perc_vals))
    perc_wgts = [wgt/sum(wgts) for wgt in wgts]

    all_wgts = mu_wgts+med_wgts+max_wgts+ssim_wgts+perc_wgts
    all_wgts = all_wgts.copy()
    mu_wgts = [wgt/sum(all_wgts) for wgt in mu_wgts]
    med_wgts = [wgt/sum(all_wgts) for wgt in med_wgts]
    max_wgts = [wgt/sum(all_wgts) for wgt in max_wgts]
    ssim_wgts = [wgt/sum(all_wgts) for wgt in ssim_wgts]
    perc_wgts = [wgt/sum(all_wgts) for wgt in perc_wgts]

    sigma = 1.5
    thres = 0.01#0.001
    mse_thres = 0#0.01#0.01
    mse_cond_loss = False
    use_mae = False#True

    def compute_ssi(prediction, target, window_size, sigma, thres):

      max_targs = apply_percentile_filter(target, window_size, 100)
      max_preds = apply_percentile_filter(prediction, window_size, 100)


      kernel = gaussian_kernel(window_size, sigma)
      kernel = kernel.reshape(1, 1, *kernel.shape)
      mu1 = convolve(prediction, kernel, mode='same')
      mu2 = convolve(target, kernel, mode='same')

      mu1_sq = mu1 ** 2
      mu2_sq = mu2 ** 2
      mu1_mu2 = mu1 * mu2

      sigma1_sq = convolve(prediction ** 2, kernel, mode='same') - mu1_sq
      sigma2_sq = convolve(target ** 2, kernel, mode='same') - mu2_sq
      sigma12 = convolve(prediction * target, kernel, mode='same') - mu1_mu2

      C1 = (0.01) ** 2
      C2 = (0.03) ** 2

      ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
      ssim_map = jnp.clip(ssim_map, 0, 1)
      ssim_map = jnp.where(((jnp.abs(max_preds) >= thres) | (jnp.abs(max_targs) >= thres)), ssim_map, jnp.nan)

      return ssim_map
 
    def apply_uniform_filter(image, nbrhd):
        kernel = jnp.ones((nbrhd,nbrhd)).astype(jnp.bfloat16)
        kernel = kernel / kernel.size
        if image.ndim==4:
          kernel = kernel.reshape(1, 1, *kernel.shape)
        else:
          kernel = kernel.reshape(1, 1, 1, *kernel.shape)
        return convolve(image, kernel, mode='same')

    #'''
    def apply_percentile_filter(image, nbrhd, percentile):
      # Determine pad width based on image dimensions
      if image.ndim == 4:
        pad_width = ((0, 0), (0, 0), (nbrhd // 2, nbrhd // 2), (nbrhd // 2, nbrhd // 2))
      elif image.ndim == 5:
        pad_width = ((0, 0), (0, 0), (0, 0), (nbrhd // 2, nbrhd // 2), (nbrhd // 2, nbrhd // 2))
      else:
        raise ValueError("Unsupported number of dimensions. Expected 4D or 5D image.")

      padded_data = jnp.pad(image, pad_width, mode='reflect')

      def process_pixel(batch, channel, level, i, j):
        if image.ndim == 4:
            window = lax.dynamic_slice(padded_data, (batch, channel, i, j), (1, 1, nbrhd, nbrhd))
        else:
            window = lax.dynamic_slice(padded_data, (batch, channel, level, i, j), (1, 1, 1, nbrhd, nbrhd))
        return jnp.percentile(window, percentile, axis=(-2, -1))

      def process_channel(batch, channel):
        if image.ndim == 4:
            height, width = image.shape[2], image.shape[3]
        else:
            height, width = image.shape[3], image.shape[4]

        i_indices = jnp.arange(height)
        j_indices = jnp.arange(width)
        indices = jnp.array(jnp.meshgrid(i_indices, j_indices, indexing='ij')).reshape(2, -1).T

        # Map over all indices to process each pixel
        def process_index(idx):
            if image.ndim == 4:
                return process_pixel(batch, channel, None, idx[0], idx[1])
            else:
                return process_pixel(batch, channel, idx[0], idx[1], idx[2])

        return vmap(process_index)(indices).reshape(height, width)

    # Apply across batches and channels
      def process_batch(batch):
        return vmap(lambda channel: process_channel(batch, channel))(jnp.arange(image.shape[1]))

      processed_image = vmap(process_batch)(jnp.arange(image.shape[0]))
      return processed_image
    '''
    def apply_percentile_filter(image, nbrhd, percentile):
      # Padding the image
      if image.ndim==4:
        pad_width = ((0, 0), (0, 0), (nbrhd//2, nbrhd//2), (nbrhd//2, nbrhd//2))
      else:
        pad_width = ((0, 0), (0, 0), (0, 0), (nbrhd//2, nbrhd//2), (nbrhd//2, nbrhd//2))

      padded_data = jnp.pad(image, pad_width, mode='reflect')
    
      def process_pixel(batch, channel, level, i, j):
        if image.ndim==4:
          window = lax.dynamic_slice(padded_data, (batch, channel, i, j), (1, 1, nbrhd, nbrhd))
        else:
          window = lax.dynamic_slice(padded_data, (batch, channel, level, i, j), (1, 1, 1, nbrhd, nbrhd))
        return jnp.percentile(window, percentile, axis=(-2, -1))

      def process_channel(batch, channel):
        height, width = image.shape[2], image.shape[3]
        i_indices = jnp.arange(height)
        j_indices = jnp.arange(width)
        indices = jnp.array(jnp.meshgrid(i_indices, j_indices, indexing='ij')).reshape(2, -1).T
        
        # Map over all indices to process each pixel
        def process_index(idx):
            return process_pixel(batch, channel, idx[0], idx[1])
        
        return vmap(process_index)(indices).reshape(height, width)
    
      # Apply across batches and channels
      def process_batch(batch):
        return vmap(lambda channel: process_channel(batch, channel))(jnp.arange(image.shape[1]))
    
      processed_image = vmap(process_batch)(jnp.arange(image.shape[0]))
      return processed_image
    #'''
    def max_pooling(image, nbrhd):
      window_shape = (1, 1, nbrhd, nbrhd)
      strides = (1, 1, 1, 1)
      padding = 'VALID'
      pooled_image = lax.reduce_window(image, -jnp.inf, lax.max, window_shape, strides, padding)
      return pooled_image

    def gaussian_kernel(window_size, sigma):
      ax = jnp.arange(-window_size // 2 + 1., window_size // 2 + 1.)#.astype(jnp.bfloat16)
      xx, yy = jnp.meshgrid(ax, ax)
      kernel = jnp.exp(-0.5 * (jnp.square(xx) + jnp.square(yy)) / jnp.square(sigma))
      return kernel / jnp.sum(kernel)

    def apply_cond_loss(error, pred, true):
      low_val = -0.1
      high_val = 0.1
      over_penalty = 2.0
      under_penalty = 2.0

      error = jnp.where(((true < low_val) & (pred > true)), over_penalty*error, error) 
      error = jnp.where(((true > high_val) & (pred < true)), under_penalty*error, error)

      return error

    def loss(prediction, target):

      var = prediction.name
      prediction = xarray_jax.jax_data(prediction)
      target = xarray_jax.jax_data(target)

      if len(mu_nbrhds) > 0:

        mu_trues = [apply_uniform_filter(target, nbrhd) for nbrhd in mu_nbrhds]
        mu_preds = [apply_uniform_filter(prediction, nbrhd) for nbrhd in mu_nbrhds]

        diffs = [pred - true for true, pred in zip(mu_trues, mu_preds)] 
        if mse_thres > 0:
          #diffs = [jnp.where((jnp.abs(diff) >= mse_thres), diff, jnp.nan) for diff in diffs]
          diffs = [jnp.where(((jnp.abs(pred) >= mse_thres) | (jnp.abs(true) >= mse_thres)), diff, jnp.nan) for pred, true, diff in zip(mu_preds, mu_trues, diffs)]
        if use_mae:
          errors = [wgt * jnp.abs(diff) for wgt, diff in zip(mu_wgts, diffs)]
        else:
          errors = [wgt * diff ** 2 for wgt, diff in zip(mu_wgts, diffs)]
        if mse_cond_loss:
          errors = [apply_cond_loss(error, pred, true) for error, pred, true in zip(errors, mu_preds, mu_trues)]

        loss = jnp.nansum(jnp.stack(errors, axis=0), axis=0)#xr.concat(errors, dim='temp').sum(dim='temp')

      if len(med_nbrhds) > 0:

        med_trues = [apply_percentile_filter(target, nbrhd, 50) for nbrhd in med_nbrhds]
        med_preds = [apply_percentile_filter(prediction, nbrhd, 50) for nbrhd in med_nbrhds]

        diffs = [pred - true for true, pred in zip(med_trues, med_preds)]
        errors = [wgt * diff ** 2 for wgt, diff in zip(med_wgts, diffs)]
        try:
          loss += jnp.sum(jnp.stack(errors, axis=0), axis=0)
        except:
          loss = jnp.sum(jnp.stack(errors, axis=0), axis=0)

      if len(max_nbrhds) > 0:

        max_trues = [apply_percentile_filter(target, nbrhd, 100) for nbrhd in max_nbrhds]
        max_preds = [apply_percentile_filter(prediction, nbrhd, 100) for nbrhd in max_nbrhds]

        diffs = [pred - true for true, pred in zip(max_trues, max_preds)]
        errors = [wgt * diff ** 2 for wgt, diff in zip(max_wgts, diffs)]
        try:
          loss += jnp.sum(jnp.stack(errors, axis=0), axis=0)
        except:
          loss = jnp.sum(jnp.stack(errors, axis=0), axis=0)

      if len(perc_nbrhds) > 0:

        trues = [apply_percentile_filter(target, nbrhd, perc_val) for nbrhd, perc_val in zip(perc_nbrhds, perc_vals)]
        preds = [apply_percentile_filter(prediction, nbrhd, perc_val) for nbrhd, perc_val in zip(perc_nbrhds, perc_vals)]

        diffs = [pred - true for true, pred in zip(trues, preds)]
        errors = [wgt * diff ** 2 for wgt, diff in zip(perc_wgts, diffs)]
        try:
          loss += jnp.sum(jnp.stack(errors, axis=0), axis=0)
        except:
          loss = jnp.sum(jnp.stack(errors, axis=0), axis=0)

      if len(ssim_nbrhds) > 0:

        SSIs = [wgt*(1-compute_ssi(prediction, target, nbrhd, sigma, thres)) for wgt, nbrhd in zip(ssim_wgts, ssim_nbrhds)]

        try:
          loss += jnp.nansum(jnp.stack(SSIs, axis=0), axis=0)
        except:
          loss = jnp.nansum(jnp.stack(SSIs, axis=0), axis=0)
        loss = 1-compute_ssi(prediction, target, 11, sigma, thres)
        
      loss = xarray_jax.DataArray(loss, dims=['batch', 'time', 'lat', 'lon'])

      return _mean_preserving_batch(loss.astype(jnp.bfloat16))

    losses = xarray_tree.map_structure(loss, predictions, targets)

    return sum_per_variable_losses(losses, per_variable_weights)
 
def SSIM_loss(
    predictions: xarray.Dataset,
    targets: xarray.Dataset,
    per_variable_weights: Mapping[str, float],
) -> LossAndDiagnostics:

    window_size=11
    sigma = 1.5
    thres = 0#0.001

    def loss(prediction, target):    

      var = prediction.name
      L = 1.0

      prediction = xarray_jax.jax_data(prediction)
      target = xarray_jax.jax_data(target)

      def gaussian_kernel(window_size, sigma):
        ax = jnp.arange(-window_size // 2 + 1., window_size // 2 + 1.)
        xx, yy = jnp.meshgrid(ax, ax)
        kernel = jnp.exp(-0.5 * (jnp.square(xx) + jnp.square(yy)) / jnp.square(sigma))
        return kernel / jnp.sum(kernel)
    
      kernel = gaussian_kernel(window_size, sigma)
      kernel = kernel.reshape(1, 1, *kernel.shape)
      mu1 = convolve(prediction, kernel, mode='same')
      mu2 = convolve(target, kernel, mode='same')

      mu1_sq = mu1 ** 2
      mu2_sq = mu2 ** 2
      mu1_mu2 = mu1 * mu2
    
      sigma1_sq = convolve(prediction ** 2, kernel, mode='same') - mu1_sq
      sigma2_sq = convolve(target ** 2, kernel, mode='same') - mu2_sq
      sigma12 = convolve(prediction * target, kernel, mode='same') - mu1_mu2
    
      C1 = (0.01*L) ** 2
      C2 = (0.03*L) ** 2
    
      ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
      ssim_map = jnp.where(((jnp.abs(mu1) > thres) | (jnp.abs(mu2) > thres)), ssim_map, jnp.nan)

      loss = 1 - jnp.nanmean(jnp.clip(ssim_map, 0, 1), axis=(2,3))

      loss = xarray_jax.DataArray(loss, dims=['batch', 'time'])

      return _mean_preserving_batch(loss.astype(jnp.bfloat16))

    losses = xarray_tree.map_structure(loss, predictions, targets)

    return sum_per_variable_losses(losses, per_variable_weights)

def simple_FSS(
    predictions: xarray.Dataset,
    targets: xarray.Dataset,
    per_variable_weights: Mapping[str, float],
) -> LossAndDiagnostics:

    def create_uniform_filter_kernel(nbrhd):
        kernel = jnp.ones(nbrhd).astype(jnp.bfloat16)
        kernel = kernel / kernel.size
        return kernel

    def apply_neighborhood_filter(binary_fcst, nbrhd):
        kernel = create_uniform_filter_kernel((nbrhd, nbrhd))
        kernel = kernel.reshape(1, 1, *kernel.shape)  # Add dimensions for batch and channels
        return convolve(binary_fcst, kernel, mode='same')

    def loss(prediction, target):

      var = prediction.name

      obs_thres = 0.01#all_obs_thres[var]
      model_thres = 0.01#all_model_thres[var]
      nbrhd = 5
      buf_width = 0
      min_points = 0
      # Compute FSS for a single threshold
      # Required inputs: 2D obs/truth field ('obs'), 2D model/analysis field ('fcst'), FSS scale ('nbrhd_rad'), FSS threshold ('thres')
      # Optional inputs: FSS computations excluded from within 'buf_width' of domain boundary; FSS not computed if number of obs and fcst points exceeding 'thres' is less than 'min_points'

      total=0; total2=0; count=0
      nbrhd = int(nbrhd)
      buf_width = int(buf_width)
      if buf_width==0:
        buf_width = (nbrhd-1) // 2

      prediction = xarray_jax.jax_data(prediction)
      target = xarray_jax.jax_data(target)

      binary_fcst = (prediction >= model_thres).astype(jnp.bfloat16)
      binary_obs  = (target >= obs_thres).astype(jnp.bfloat16)

      NP_fcst = apply_neighborhood_filter(binary_fcst, nbrhd)
      NP_obs = apply_neighborhood_filter(binary_obs, nbrhd)

      NP_fcst = NP_fcst[:,:,buf_width:-buf_width, buf_width:-buf_width]
      NP_obs = NP_obs[:,:,buf_width:-buf_width, buf_width:-buf_width]

      mse = jnp.mean((NP_fcst - NP_obs) ** 2, axis=(2,3))
      potential_mse = jnp.mean(NP_fcst**2 + NP_obs**2, axis=(2,3))

      loss = jnp.where(potential_mse == 0, jnp.nan, (mse / potential_mse).astype(jnp.bfloat16)) 
      loss = xarray_jax.DataArray(loss, dims=['batch', 'time'])

      return _mean_preserving_batch(loss)

    losses = xarray_tree.map_structure(loss, predictions, targets)

    return sum_per_variable_losses(losses, per_variable_weights)

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
            

def weighted_mse_per_level(
    predictions: xarray.Dataset,
    targets: xarray.Dataset,
    per_variable_weights: Mapping[str, float],
) -> LossAndDiagnostics:
    """Latitude- and pressure-level-weighted MSE loss."""

    def loss(prediction, target):
        loss = (prediction - target) ** 2

        # print('No longer doing latitude-based weighting in the loss')
        # loss *= normalized_latitude_weights(target).astype(loss.dtype)

        # print('Turned off the pressure-level weighted loss')
        # if 'level' in target.dims:
        #  loss *= normalized_level_weights(target).astype(loss.dtype)

        return _mean_preserving_batch(loss)

    # losses = xarray_tree.map_structure(loss, predictions, targets)
    # losses = xarray_tree.map_structure(custom_loss, predictions, targets)

    losses = xarray_tree.map_structure(loss, predictions, targets)

    return sum_per_variable_losses(losses, per_variable_weights)


def _mean_preserving_batch(x: xarray.DataArray) -> xarray.DataArray:
    return x.mean([d for d in x.dims if d != "batch"], skipna=True)


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


def normalized_level_weights(data: xarray.DataArray) -> xarray.DataArray:
    """Weights proportional to pressure at each level."""
    level = data.coords["level"]
    return level / level.mean(skipna=False)


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
