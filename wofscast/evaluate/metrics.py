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
    
    #@property    
    #def results(self):
    #    return self.results_


class MSE(Metric): 
    """A class for computing an accumulating MSE. The averaging occurs over all 
    dimensions except time. 
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
        self.results_ /= self.n_updates
        
        # Apply the square root to get RMSE. 
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
            self.results_ = diff
        else:    
            self.results_ += diff 
        
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
    def __init__(self, windows, thresh_dict, variables):
        super().__init__()
        
        self.windows = windows 
        self.thresh_dict = thresh_dict
        
        if isinstance(variables, str):
            variables = [variables]
        
        self.variables = variables 
    
    def _init_ds(self, dataset):
        
         # Create the time dimension (you can customize this as needed)
        time = dataset.time
        n_times = len(time)
        n_windows = len(self.windows)
        
        numer_data_dict = {f'{v}_numer' : (['time', 'window', 'thresholds'], 
                                       np.zeros((n_times, n_windows, len(self.thresh_dict[v])), 
                                                dtype=np.float32))
                    
                    for v in self.variables
                    }

        denom_data_dict = {f'{v}_denom' : (['time', 'window', 'thresholds'], 
                                       np.zeros((n_times, n_windows, len(self.thresh_dict[v])), 
                                                dtype=np.float32))
                    
                    for v in self.variables
                    }
        
        
        data_dict = {**numer_data_dict, **denom_data_dict}
        
        # Create an empty dataset with three variables
        empty_dataset = xr.Dataset(data_dict,
                        coords={'time': time, 
                                'window' : self.windows
                           }
        )

        return empty_dataset 
        
    def update(self, forecast : xr.Dataset, truth : xr.Dataset) -> xr.Dataset: 
        
        if self.n_updates == 0:
            self.results_ = self._init_ds(forecast)
            self._n_times = forecast.dims['time']
            
        for t in range(self._n_times):
            for v in self.variables:
                for w, window in enumerate(self.windows):
                    for j, thresh in enumerate(self.thresh_dict[v]):

                        forecast_binary = forecast.isel(time=t)[v].values >= thresh
                        truth_binary = truth.isel(time=t)[v].values >= thresh
    
                        NP_forecast = uniform_filter(forecast_binary.astype(float), window, mode='constant')
                        NP_truth = uniform_filter(truth_binary.astype(float), window, mode='constant')

                        numer = ((NP_forecast - NP_truth)**2).sum()
                        denom = (NP_forecast**2 + NP_truth**2).sum()
                    
                        self.results_[f'{v}_numer'][t,w,j] += numer
                        self.results_[f'{v}_denom'][t,w,j] += denom
                        
        self.increment_updates()                
                        
        return self 
                        
    
    def finalize(self) -> xr.Dataset:
        for var in self.variables: 
            se = self.results_[f'{var}_numer']
            potential_se = self.results_[f'{var}_denom']
            
            self.results_[f'{var}_fss'] = 1 - se / potential_se
            
            self.results_ = self.results_.drop_vars([f'{var}_numer', f'{var}_denom'])
            
        return self.results_ 
            
class PMMStormStructure(Metric):
    def __init__(self, variables, patch_size=30, matching_dist=14, buff=5):
        self.patch_size = patch_size
        self.buff = buff
        self.variables = variables
        self.matcher = monte_python.ObjectMatcher(cent_dist_max = matching_dist, 
                                     min_dist_max = matching_dist, 
                                     time_max=0, 
                                     score_thresh=0.2, 
                                     one_to_one = True)
        super().__init__()
    
    def _init_ds(self): 
        self.forecast_patches = {v : [] for v in self.variables}
        self.truth_patches = {v : [] for v in self.variables} 
        
    def update(self, forecast : xr.Dataset, truth : xr.Dataset) -> xr.Dataset: 
        
        if self.n_updates==0:
            self._init_ds()
            self.domain_size = forecast.dims['lat']
            self._n_times = forecast.dims['time']
            
        for t in range(self._n_times): 
        
            # Match objects
            forecast_storms = forecast.isel(time=t)['storms'].values
            truth_storms = truth.isel(time=t)['storms'].values
            forecast_labels, truth_labels, cent_dists = self.matcher.match(forecast_storms, truth_storms)
        
            # For removing any labels that is not matched. 
            forecast_storms_matched = np.where(np.isin(forecast_storms, forecast_labels), forecast_storms, 0)
            truth_storms_matched = np.where(np.isin(truth_storms, truth_labels), truth_storms, 0)
        
            # Get the object properties 
            forecast_label_props = regionprops(forecast_storms_matched.astype(int), 
                                               forecast['COMPOSITE_REFL_10CM'].isel(time=t, batch=0).values)
            truth_label_props = regionprops(truth_storms_matched.astype(int), 
                                            truth['COMPOSITE_REFL_10CM'].isel(time=t, batch=0).values)
        
            all_pred_i, all_pred_j = self.get_all_i_j(forecast_label_props)
            all_true_i, all_true_j = self.get_all_i_j(truth_label_props)
        
            # Precompute half of the patch size
            half_patch_size = self.patch_size // 2

            for nn, (pred_i, pred_j, true_i, true_j) in enumerate(zip(all_pred_i, 
                                                                  all_pred_j,
                                                                  all_true_i, 
                                                                  all_true_j)):
                # Calculate the minimum and maximum indices considering the patch size
                min_index = min(pred_i, pred_j, true_i, true_j) - half_patch_size
                max_index = max(pred_i, pred_j, true_i, true_j) + half_patch_size

                # Skip if the patch would be outside the domain boundaries
                if (min_index < self.buff) or (max_index > self.domain_size - self.buff):
                    continue

                for var in self.variables:
                    # Indices for the forecast patch around the predicted centroid
                    imin_pred = pred_i - half_patch_size
                    imax_pred = pred_i + half_patch_size
                    jmin_pred = pred_j - half_patch_size
                    jmax_pred = pred_j + half_patch_size

                    # Extract and store the forecast patch
                    forecast_patch = forecast[var].isel(time=t, batch=0)[imin_pred:imax_pred+1, jmin_pred:jmax_pred+1]
                    self.forecast_patches[var].append(forecast_patch)

                    # Indices for the truth patch around the true centroid
                    imin_true = true_i - half_patch_size
                    imax_true = true_i + half_patch_size
                    jmin_true = true_j - half_patch_size
                    jmax_true = true_j + half_patch_size

                    # Extract and store the truth patch
                    truth_patch = truth[var].isel(time=t, batch=0)[imin_true:imax_true+1, jmin_true:jmax_true+1]
                    self.truth_patches[var].append(truth_patch)

        self.increment_updates()            
                    
        return self 
    
    
    def finalize(self) -> xr.Dataset:
        # Compute probability-matched mean for each variable
        data_dict = {}
        for var in self.variables:
            data_dict[f'{var}_forecast_pmm'] = (['pmm_lat', 'pmm_lon'], self.compute(self.forecast_patches[var]))
            data_dict[f'{var}_truth_pmm'] = (['pmm_lat', 'pmm_lon'], self.compute(self.truth_patches[var]))
        
        results = xr.Dataset(data_dict)    
        
        results.attrs['n_pmm_storms'] = len(self.forecast_patches[var])
        
        return results 
    
    def compute(self, data):
        
        data = np.array(data)
        # Transpose the array to shape (ny, nx, ne)
        data = data.transpose(1, 2, 0)
        
        field_mean = np.mean(data, axis=-1)
        
        rank_indices = np.argsort(np.ravel(field_mean))
        
        sorted_per_mem = np.zeros((data.shape[0]*data.shape[1], data.shape[2]))
        for k in range(data.shape[2]):
            sorted_per_mem[:,k] = np.sort(np.ravel(data[:,:,k]))
        sorted_1D = np.mean(sorted_per_mem, axis=1)
        
        PMM_1D = np.zeros(np.ravel(field_mean).shape)
        for count, inds in enumerate(rank_indices):
            PMM_1D[inds] = sorted_1D[count]
            
        PMM = np.reshape(PMM_1D, field_mean.shape)
    
        return PMM
    
    # Define a helper function to extract centroid coordinates
    def get_all_i_j(self, label_props):
        if not label_props:
            return [], []
        # Extract centroids and convert to integers
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
        
    