import xarray as xr 
import numpy as np 

# For the object identification, matching, and verification
import sys
sys.path.insert(0, '/home/monte.flora/python_packages/MontePython')
import monte_python
from skimage.measure import regionprops


# For the spectra computation
from scipy import stats
from scipy.fft import dct

# for FSS
from scipy.ndimage import uniform_filter


class Metric:
    """Base class for time-perserving verification metrics"""
    def __init__(self):
        self.results_ = None
        self.n_updates = 0 
    
    def increment_updates(self):
        """Increments the number of updates."""
        self.n_updates += 1
    
    def raise_no_update_error(self):
        """Raise an error if finalize is called and n_updates==0"""
        if self.n_updates == 0:
            raise ZeroDivisionError
        
    def check_for_results(self):
        if self.results_ is None:
            raise AttributeError('Call .update() before calling .finalize') 
        
class MSE(Metric): 
    """A class for computing an accumulating MSE. Averaging occurs over all 
    dimensions except time. The MSE is computed for each call to .update() 
    and to a running total. Calling .finalize() computes the per-sample 
    average root-mean-squared-error.
    
    Attributes
    ---------------
    addon : str (default=None) 
        Used for adding a string to the variable names. 
        E.g., f'{n}_rmse_{self.addon}'. 
        
    variables : list of str (default=None)
        Variables to compute RMSE for. If None, then variables is set 
        to the list of variables found in both the forecast and truth dataset 
    """
    def __init__(self, addon=None, variables=None):
        self.addon = addon
        self.variables = variables
        super().__init__()
    
    def set_variables(self, forecast, truth):
        # Get the intersection of variables in the order they appear in forecast
        variables = [var for var in forecast.data_vars if var in truth.data_vars]
        self.variables = variables
    
    def mean_preserving_time(self, x : xr.Dataset) -> xr.Dataset:
        return x.mean([d for d in x.dims if d != 'time'], skipna=True)
    
    def finalize(self) -> xr.Dataset:
        """
        Divides the summed differences to create the mean squared difference
        Applies sqrt to get rmse.
        """
        self.check_for_results()
        
        self.results_ /= self.n_updates
        
        # Apply the square root to get RMSE while ignoring NaNs
        self.results_ = self.results_.apply(np.sqrt)

        # Rename the variables with the '_rmse' tag. 
        if self.addon: 
            mapper = {n : f'{n}_rmse_{self.addon}' for n in self.variables}
        else:
            mapper = {n : f'{n}_rmse' for n in self.variables}
            
        self.results_ = self.results_.rename(mapper)

        return self.results_ 
    
    def update(self, forecast : xr.Dataset, truth : xr.Dataset) -> xr.Dataset: 
        """
        Computes a time-preserving average of the 
        squared difference between the foreast and truth datasets.
        
        The results are added for each call to .update.  
        """
        if self.n_updates==0 and self.variables is None:
            self.set_variables(forecast, truth)
                
        diff = (forecast[self.variables]-truth[self.variables])**2
                
        diff = self.mean_preserving_time(diff)
                
        if self.results_ is None:
            self.results_ = diff.copy(deep=True) 
        else:    
            # Ignore NaNs in the summation; potentially troublesome!!
            self.results_ = self.results_.fillna(0) + diff.fillna(0)
            
            # Restore NaNs where both were NaN
            self.results_ = xr.where(
                np.isnan(self.results_) & np.isnan(diff),
                np.nan,
                self.results_,
            )
            
        
        self.increment_updates()
        
        return self
        
        
class ObjectBasedContingencyStats(Metric):
    
    METRICS = ["hits", "false_alarms", "misses"]
    
    def __init__(self, matching_dist : int  = 14, 
                 key : str = 'wofscast_vs_wofs'
                ):
        super().__init__()
        
        matcher = monte_python.ObjectMatcher(cent_dist_max = matching_dist, 
                                     min_dist_max = matching_dist, 
                                     time_max=0, 
                                     score_thresh=0.2, 
                                     one_to_one = True)
        
        self.obj_verifier = monte_python.ObjectVerifier(matcher)
        self.key = key 
        
    def _init_ds(self, dataset : xr.Dataset) -> xr.Dataset:
        
        # Create the time dimension (you can customize this as needed)
        time = dataset.time
        self._n_times = len(time)

        # Create an empty dataset with three variables
        empty_dataset = xr.Dataset(
            {
            f'{self.key}_hits': (['time'], np.zeros(self._n_times, dtype=np.float32)),
            f'{self.key}_misses': (['time'], np.zeros(self._n_times, dtype=np.float32)),
            f'{self.key}_false_alarms': (['time'], np.zeros(self._n_times, dtype=np.float32))
            },
            coords={'time': time}
        )
        
        return empty_dataset 
        
    def finalize(self) -> xr.Dataset:
        """From the total tally of hits, misses, and false alarms
        Computes POD, SR, CSI, and FB
        """
        # Compute POD, CSI, SR, FB
        for metric in ['pod', 'sr', 'csi', 'fb']:
            self.results_ = getattr(self, metric)(self.results_)
            
        return self.results_ 
    
    def update(self, forecast : xr.Dataset, truth : xr.Dataset) -> xr.Dataset: 
        """
        Matches objects in forecast and truth and keeps a running tally of 
        hits, misses, and false alarms.
        """
        if self.n_updates == 0:
            if 'storms' not in forecast.data_vars: 
                raise KeyError("'storms' does not exist in the forecast dataset")
            
            if 'storms' not in truth.data_vars: 
                raise KeyError("'storms' does not exist in the truth dataset")
            
            self.results_ = self._init_ds(forecast)
        
        for t in range(self._n_times):
            self.obj_verifier.update_metrics(truth.isel(time=t)['storms'].values, 
                                             forecast.isel(time=t)['storms'].values)
            
            for metric in self.METRICS:
                self.results_[f'{self.key}_{metric}'][t] +=  getattr(self.obj_verifier, f"{metric}_")     

            self.obj_verifier.reset_metrics()  
        
        self.increment_updates()    
            
        return self 
        
    def pod(self, ds): 
        # hits / hits + misses
        ds[f'{self.key}_pod'] = ds[f'{self.key}_hits'] / (ds[f'{self.key}_hits'] + ds[f'{self.key}_misses'])
        return ds 
    
    
    def sr(self, ds): 
        # hits / hits + false alarms
        ds[f'{self.key}_sr'] = ds[f'{self.key}_hits'] / (ds[f'{self.key}_hits'] + ds[f'{self.key}_false_alarms'])
        return ds 
    
    def csi(self, ds): 
        # hits / hits + misses + false alarms
        ds[f'{self.key}_csi'] = ds[
            f'{self.key}_hits'] / (ds[f'{self.key}_hits'] + ds[f'{self.key}_misses'] + ds[f'{self.key}_false_alarms'])
        return ds 
    
    def fb(self, ds): 
        # hits / hits + misses + false alarms
        ds[f'{self.key}_fb'] = ds[f'{self.key}_pod'] / ds[f'{self.key}_sr']
        return ds 
        
class PowerSpectra(Metric): 
    def __init__(self, grid_spacing_in_km = 3.0, variables=None, level=5):
        self.grid_spacing_in_km = grid_spacing_in_km
        self.variables = variables 
        self.level = level
        
        super().__init__()
        
    def _init_ds(self, dataset, waven):
        
         # Create the time dimension (you can customize this as needed)
        time = dataset.time
        n_times = len(time)
        
        n_bins = len(waven)

        if self.variables is None:
            self.variables = list(dataset.vars)
        
        forecast_data_dict = {f'{v}_forecast_spectra' : (['time', 'wave_num'], 
                                       np.zeros((n_times, n_bins), dtype=np.float32))
                    
                    for v in self.variables 
                    }
        
        truth_data_dict = {f'{v}_truth_spectra' : (['time', 'wave_num'], 
                                       np.zeros((n_times, n_bins), dtype=np.float32))
                    
                    for v in self.variables 
                    }
        
        data_dict = {**forecast_data_dict, **truth_data_dict}
        
        
        # Create an empty dataset with three variables
        empty_dataset = xr.Dataset(data_dict,
                        coords={'time': time, 
                            'wave_num' : waven
                           }
        )

        return empty_dataset 
    
    
    def update(self, forecast : xr.Dataset, truth : xr.Dataset) -> xr.Dataset: 
        # Drop the batch dim if its size == 1
        forecast = forecast.squeeze()
        truth = truth.squeeze()
        
        # 3D -> 2D, selecting a higher level, above the PBL. 
        if 'level' in forecast.dims: 
            forecast = forecast.isel(level=self.level)
            truth = truth.isel(level=self.level)

        # initilalize the results dataset. 
        if self.n_updates == 0:
            var = list(forecast.data_vars)[0]
            kvals, forecast_spectra, waven = self.get_spectra2D_DCT(forecast[var].isel(time=0).values)
            self.results_ = self._init_ds(forecast, waven)
            self._n_times = forecast.dims['time']    
            
        for t in range(self._n_times): 
            for var in self.variables:
                kvals, forecast_spectra, waven = self.get_spectra2D_DCT(forecast[var].isel(time=t).values)
                _, truth_spectra, _ = self.get_spectra2D_DCT(truth[var].isel(time=t).values)
                
                self.results_[f'{var}_forecast_spectra'][t] += forecast_spectra
                self.results_[f'{var}_truth_spectra'][t] += truth_spectra
                
        self.increment_updates()
                
        return self 
                
    def finalize(self) -> xr.Dataset: 
        
        # The spectra were cumulatively summed with the .update(), 
        # so need to divide by the count to get the average. 
        self.results_ /= self.n_updates
    
        return self.results_
    
    def get_spectra2D_DCT(self, data, varray = None):
        """
        Code based on Nate Snook's implementation of the algorithm in Surcel et al. (2014)

        Arguments:
            `field` is a 2D numpy array for which you would like to find the spectrum
            `dx` and `dy` are the distances between grid points in the x and y directions in meters

        Returns a tuple of (length_scale, spectrum), where `length_scale` is the length scale in meters,
            and `spectrum` is the power at each length scale.
        """
        dx=dy=self.grid_spacing_in_km
        
        _dct_type = 2
    
        u = data
    
        if type(varray) != type(None):
            v = varray
            
        # compute spectra
    
        ny, nx = u.shape

        if type(varray) == type(None):
        
            variance = 0.5*dct(dct(u, axis=0,
                               type=_dct_type, norm='ortho'), axis=1, type=_dct_type, norm='ortho')**2
        
        else:
        
            variance = 0.5*(dct(dct(u, 
                                axis=0, type=_dct_type, norm='ortho'), axis=1, 
                                type=_dct_type, norm='ortho')**2 \
                       +dct(dct(v, 
                                axis=0, type=_dct_type, norm='ortho'), axis=1, 
                            type=_dct_type, norm='ortho')**2)
           
        kfreq   = np.fft.fftfreq(nx) * nx
        kfreq2D = np.meshgrid(kfreq, kfreq)
    
        knrm   = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
        knrm2  = 0.5*(knrm[1:,:] + knrm[:-1,:])
        knrm   = 0.5*(knrm2[:,1:] + knrm2[:,:-1])
        knrm   = knrm[:ny//2,:nx//2].flatten()
    
        # In order to make this similar to the DFT, you need to shift variances
    
        variance2 = np.zeros((ny,nx//2))
        variance3 = np.zeros((ny//2,nx//2))
    
        for i in np.arange(1,nx//2):   
            variance2[:,i-1] = variance[:,2*i-1] + variance[:,2*i]

        for j in np.arange(1,ny//2):   
            variance3[j-1,:] = variance2[2*j-1,:] + variance2[2*j,:]

        variance = variance3.flatten()
    
        kbins = np.arange(0.5, nx//2+1, 1.)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        PSbins, _, _ = stats.binned_statistic(knrm, variance, statistic = "mean", bins = kbins)
    
        PSbins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    
        waven = 2.*kvals / nx
            
        return kvals, PSbins, waven 
 

class FractionsSkillScore(Metric):
    def __init__(self, windows, thresh_dict, variables, addon=None):
        super().__init__()
        
        self.windows = windows 
        self.thresh_dict = thresh_dict
        
        if isinstance(variables, str):
            variables = [variables]
        
        self.variables = variables 
        self.addon = addon
    
    def _init_ds(self, dataset):
        
         # Create the time dimension (you can customize this as needed)
        time = dataset.time
        n_times = len(time)
        n_windows = len(self.windows)
        
        obs_sq_dict = {f'{v}_obs_sq' : (['time', 'window', 'thresholds'], 
                                       np.zeros((n_times, n_windows, len(self.thresh_dict[v])), 
                                                dtype=np.float32))
                    
                    for v in self.variables
                    }

        fct_obs_dict = {f'{v}_fct_obs' : (['time', 'window', 'thresholds'], 
                                       np.zeros((n_times, n_windows, len(self.thresh_dict[v])), 
                                                dtype=np.float32))
                    
                    for v in self.variables
                    }
        
        fct_sq_dict = {f'{v}_fct_sq' : (['time', 'window', 'thresholds'], 
                                       np.zeros((n_times, n_windows, len(self.thresh_dict[v])), 
                                                dtype=np.float32))
                    
                    for v in self.variables
                    }
        
        data_dict={**obs_sq_dict, **fct_obs_dict, **fct_sq_dict}
        
        # Create an empty dataset with three variables
        empty_dataset = xr.Dataset(data_dict,
                        coords={'time': time, 
                                'window' : self.windows
                           }
        )

        return empty_dataset 
        
    def fss_accum(self, X_f, X_o, thresh, window):
        """Inspired by the pysteps package 
        https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/spatialscores.py
        """
        if isinstance(thresh, tuple):
            f_thresh, o_thresh = thresh
        else:
            f_thresh, o_thresh = thresh, thresh 

        if len(X_f.shape) != 2 or len(X_o.shape) != 2 or X_f.shape != X_o.shape:
            message = "X_f and X_o must be two-dimensional arrays"
            message += " having the same shape"
            raise ValueError(message)

        X_f = X_f.copy()
        X_f[~np.isfinite(X_f)] = f_thresh - 1
        X_o = X_o.copy()
        X_o[~np.isfinite(X_o)] = o_thresh - 1

        # Convert to binary fields with the given intensity threshold
        I_f = (X_f >= f_thresh).astype(float)
        I_o = (X_o >= o_thresh).astype(float)

        # Compute fractions of pixels above the threshold within a square
        # neighboring area by applying a 2D moving average to the binary fields
        if window > 1:
            S_f = uniform_filter(I_f, size=window, mode="constant", cval=0.0)
            S_o = uniform_filter(I_o, size=window, mode="constant", cval=0.0)
        else:
            S_f = I_f
            S_o = I_o
        
        return S_f, S_o 
        
    def fss_compute(self, variable): 
        v = variable
        numer = self.results_[f'{v}_fct_sq'] - 2.0 * self.results_[f"{v}_fct_obs"] + self.results_[f"{v}_obs_sq"]
        denom = self.results_[f"{v}_fct_sq"] + self.results_[f"{v}_obs_sq"]

        return 1.0 - numer / denom
        
    def update(self, forecast : xr.Dataset, truth : xr.Dataset) -> xr.Dataset: 
        
        if self.n_updates == 0:
            self.results_ = self._init_ds(forecast)
            self._n_times = forecast.dims['time']
            
        for t in range(self._n_times):
            for v in self.variables:
                forecast_vals = forecast.isel(time=t)[v].values.squeeze()
                truth_vals = truth.isel(time=t)[v].values.squeeze()
                
                for w, window in enumerate(self.windows):
                    for j, thresh in enumerate(self.thresh_dict[v]):
                        S_f, S_o = self.fss_accum(forecast_vals, truth_vals, thresh, window) 
                        
                        self.results_[f'{v}_fct_sq'][t,w,j] += np.nansum(S_f**2)
                        self.results_[f'{v}_obs_sq'][t,w,j] += np.nansum(S_o**2)
                        self.results_[f'{v}_fct_obs'][t,w,j] += np.nansum(S_f*S_o)
                        
        self.increment_updates()                
                        
        return self 
                        
    
    def finalize(self) -> xr.Dataset:
        for var in self.variables: 
            key = f'{var}_fss_{self.addon}' if self.addon else f'{var}_fss'
            self.results_[key] = self.fss_compute(var) 
            self.results_ = self.results_.drop_vars([f'{var}_fct_obs', f'{var}_fct_sq', f'{var}_obs_sq'])

        return self.results_ 

class PMMStormStructure(Metric):
    def __init__(self, variables, patch_size=30, matching_dist=14, buff=5):
        self.patch_size = patch_size
        self.buff = buff
        self.variables = variables
        self.matcher = monte_python.ObjectMatcher(
            cent_dist_max=matching_dist,
            min_dist_max=matching_dist,
            time_max=0,
            score_thresh=0.2,
            one_to_one=True
        )
        super().__init__()

    def _init_ds(self, forecast):
        """Initialize storage for forecast and truth patches."""
        # Lists to dynamically accumulate patches
        self.forecast_patches = []
        self.truth_patches = []

    def update(self, forecast: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
        """Update patches for forecast and truth at each time step."""
        if self.n_updates == 0:
            self._init_ds(forecast)
            self.domain_size = forecast.dims['lat']
            self._n_times = forecast.dims['time']

        for t in range(self._n_times):
            # Match objects between forecast and truth datasets
            forecast_storms = forecast.isel(time=t)['storms'].values
            truth_storms = truth.isel(time=t)['storms'].values
            forecast_labels, truth_labels, _ = self.matcher.match(forecast_storms, truth_storms)

            # Filter unmatched labels
            forecast_storms_matched = np.where(np.isin(forecast_storms, forecast_labels), forecast_storms, 0)
            truth_storms_matched = np.where(np.isin(truth_storms, truth_labels), truth_storms, 0)

            # Get object properties
            forecast_label_props = regionprops(
                forecast_storms_matched.astype(int),
                forecast['COMPOSITE_REFL_10CM'].isel(time=t, batch=0).values
            )
            truth_label_props = regionprops(
                truth_storms_matched.astype(int),
                truth['COMPOSITE_REFL_10CM'].isel(time=t, batch=0).values
            )

            all_pred_i, all_pred_j = self.get_all_i_j(forecast_label_props)
            all_true_i, all_true_j = self.get_all_i_j(truth_label_props)

            # Precompute half of the patch size
            half_patch_size = self.patch_size // 2

            forecast_at_t = forecast.isel(time=t, batch=0)
            truth_at_t = truth.isel(time=t, batch=0)

            for pred_i, pred_j, true_i, true_j in zip(all_pred_i, all_pred_j, all_true_i, all_true_j):
                # Skip patches outside the domain
                if (pred_i - half_patch_size < self.buff or pred_i + half_patch_size > self.domain_size - self.buff or
                        pred_j - half_patch_size < self.buff or pred_j + half_patch_size > self.domain_size - self.buff or
                        true_i - half_patch_size < self.buff or true_i + half_patch_size > self.domain_size - self.buff or
                        true_j - half_patch_size < self.buff or true_j + half_patch_size > self.domain_size - self.buff):
                    continue

                # Define patch boundaries
                imin_pred, imax_pred = pred_i - half_patch_size, pred_i + half_patch_size
                jmin_pred, jmax_pred = pred_j - half_patch_size, pred_j + half_patch_size
                imin_true, imax_true = true_i - half_patch_size, true_i + half_patch_size
                jmin_true, jmax_true = true_j - half_patch_size, true_j + half_patch_size

                # Extract forecast and truth patches at these patch coordinates
                these_forecast_patches = forecast_at_t[self.variables].isel(
                    lat=slice(imin_pred, imax_pred + 1),
                    lon=slice(jmin_pred, jmax_pred + 1)
                )

                these_truth_patches = truth_at_t[self.variables].isel(
                    lat=slice(imin_true, imax_true + 1),
                    lon=slice(jmin_true, jmax_true + 1)
                )

                # Append patches to the respective lists
                self.forecast_patches.append((these_forecast_patches, t))
                self.truth_patches.append((these_truth_patches, t))

        self.increment_updates()
        
        return self

    def finalize(self) -> xr.Dataset:
        """Aggregate the accumulated patches and compute the probability-matched mean (PMM)."""
        data_dict = {}

        n_samples_per_time = {t: 0 for t in range(self._n_times)}  # Initialize sample counter
        
        for var in self.variables:
            forecast_pmm = [] 
            truth_pmm = [] 
            for t in range(self._n_times):

                # Collect all forecast and truth patches for this variable and time index
                forecast_patches_t = [
                    patch[var].values for patch, patch_time in self.forecast_patches if patch_time == t
                ]
                truth_patches_t = [
                    patch[var].values for patch, patch_time in self.truth_patches if patch_time == t
                ]
                
                # Update sample count for this time step
                n_samples_per_time[t] = len(forecast_patches_t)

                # Concatenate patches if any are available
                if forecast_patches_t:
                    forecast_pmm.append(self.compute(np.array(forecast_patches_t)))
                else:
                    forecast_pmm.append(np.zeros((self.patch_size, self.patch_size)))

                if truth_patches_t:
                    truth_pmm.append(self.compute(np.array(truth_patches_t)))
                else:
                    truth_pmm.append(np.zeros((self.patch_size, self.patch_size)))

                # Store PMM results
                data_dict[f'{var}_forecast_pmm'] = (['time', 'patch_lat', 'patch_lon'], np.array(forecast_pmm))
                data_dict[f'{var}_truth_pmm'] = (['time', 'patch_lat', 'patch_lon'], np.array(truth_pmm)) 

        # Create an xarray.Dataset for the results
        results = xr.Dataset(data_dict)

        results['n_pmm_samples_per_time'] = (['time'], list(n_samples_per_time.values()))
        
        return results

    def compute(self, data):
        """Compute PMM from a collection of patches."""
        # Transpose the array to shape (ny, nx, ne)
        data = data.transpose(1, 2, 0)
                    
        # Compute field mean and rank
        field_mean = np.nanmean(data, axis=-1)
        rank_indices = np.argsort(np.ravel(field_mean))

        # Compute sorted per-member values
        sorted_per_mem = np.zeros((data.shape[0] * data.shape[1], data.shape[2]))
        for k in range(data.shape[2]):
            sorted_per_mem[:, k] = np.sort(np.ravel(data[:, :, k]))
        sorted_1D = np.nanmean(sorted_per_mem, axis=1)

        # Compute PMM
        pmm_1D = np.zeros(np.ravel(field_mean).shape)
        for count, inds in enumerate(rank_indices):
            pmm_1D[inds] = sorted_1D[count]
            
        pmm = np.reshape(pmm_1D, field_mean.shape)

        return pmm

    def get_all_i_j(self, label_props):
        """Extract centroid coordinates."""
        if not label_props:
            return [], []
        centroids = [region['Centroid'] for region in label_props]
        i_coords, j_coords = zip(*[(int(c[0]), int(c[1])) for c in centroids])
        return list(i_coords), list(j_coords)

   
class ObjectProperties(Metric):
    def __init__(self):
        super().__init__() 
    
    def update(self, forecast, truth):
        pass
    
    def finalize(self):
        pass
        
    