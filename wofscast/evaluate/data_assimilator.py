from ..model import WoFSCastModel
from ..diffusion import DiffusionModel
from ..data_generator import add_local_solar_time


from scipy.ndimage import gaussian_filter

class DataAssimilator:
    def __init__(self, model_path, 
                 unknown_variables=['W', 'T', 'GEOPOT', 'QVAPOR', 'RAIN_AMOUNT', 'T2', 'U', 'V'], 
                 skewed_variables = ['W', 'RAIN_AMOUNT'], 
                 gauss_filter_size=10,
                 add_diffusion=False,
                 sampler_kwargs = dict(
                    sigma_min = 0.0002,
                    sigma_max = 1000, 
                    S_churn=0., #np.sqrt(2)-1, 
                    S_min=0.02, 
                    S_max=800, 
                    S_noise=1.05),
                 **additional_configs):

        self._load_model(model_path, add_diffusion, sampler_kwargs, **additional_configs)
    
        self._unknown_variables = unknown_variables
        self._skewed_variables = skewed_variables
        self._gauss_filter_size = gauss_filter_size
    
    def predict(self, inputs, targets, forcings, replace_bdry=True, n_diffusion_steps=20): 
        
        # Applying DA (smooth the unknown fields)
        inputs_smoothed = self.smooth(inputs)
        
        da_inputs = inputs.copy(deep=True)

        latest_time = da_inputs.isel(time=[-1])['time']
        da_targets = targets.isel(time=[0])
        da_forcings = forcings.isel(time=[0])
        
        for t in range(inputs.dims['time']):
            # The DA model is built on the inputs being valid at 00 and the targets
            # +10 min. So at the moment, need input times to be consistent with that.
            this_input = inputs_smoothed.isel(time=[t]).copy(deep=True)
            this_input = this_input.drop('datetime')
            this_time = this_input['time']
            this_input = this_input.assign_coords(time=latest_time)

            output = self.model.predict(this_input, da_targets, da_forcings, 
                                     replace_bdry=replace_bdry, 
                                      diffusion_model=self.diffusion_model, 
                                     n_diffusion_steps=n_diffusion_steps)
    
            # Reset to the true time so we can replace the values in da_inputs
            output = output.assign_coords(time=this_time)
            output = output.transpose('batch', 'time', 'lat', 'lon', 'level', missing_dims='ignore')
    
            # Update da_inputs with the predicted output at the correct time step
            #da_inputs.loc[dict(time=this_time)] = output
            # Update da_inputs with the predicted output at the correct time step, ignoring static variables
            for var_name, data_array in output.data_vars.items():
                if 'time' in data_array.dims:  # Only update variables with a time dimension
                    da_inputs[var_name].loc[dict(time=this_time)] = data_array
        
        return da_inputs
    
    def _load_model(self, model_path, add_diffusion, sampler_kwargs, **additional_configs):
        self.model = WoFSCastModel()
        self.model.load_model(model_path, **additional_configs)
        
        self.diffusion_model = None    
        if add_diffusion:    
            self.diffusion_model = DiffusionModel(sampler_kwargs=sampler_kwargs)
        
    def smooth(self, dataset):
        def apply_gaussian(arr, sigma):
            # Apply Gaussian filter with different sigma per dimension
            return gaussian_filter(arr, sigma=sigma)
       
        ds_smoothed = dataset.copy(deep=True)
        
        for variable in self._unknown_variables:
            key =  variable            
            if variable in self._skewed_variables:
                # For skewed variables like W or rain amount, the smoothed version 
                # is simply set to zero. 
                ds_smoothed[key] = xr.zeros_like(dataset[variable])
        else:
            #sigma = (self.gauss_filter_size if dim != 'time' else 0 for dim in dataset[variable].dims)
            # Exclude smoothing for 'time' and 'level' dimensions
            sigma = (self._gauss_filter_size if 
                     dim not in ['time', 'level'] else 0 for dim in dataset[variable].dims)
            ds_smoothed[key] = xr.apply_ufunc(
                        apply_gaussian, 
                        dataset[variable], 
                        kwargs={'sigma': sigma},  
                        dask="allowed"
                        )
            
        return ds_smoothed     