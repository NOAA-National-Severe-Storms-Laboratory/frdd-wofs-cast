from ..model import WoFSCastModel
from ..diffusion import DiffusionModel
from ..data_generator import add_local_solar_time

class Predictor:
    
    LEGACY_MODELS = [
    '/work/cpotvin/WOFSCAST/model/wofscast_test_v178.npz',
    '/work2/mflora/wofs-cast-data/model/wofscast_test_v178_fine_tune_v3.npz',
    '/work/cpotvin/WOFSCAST/model/wofscast_test_v203.npz', 
    '/work/cpotvin/WOFSCAST/model/wofscast_test_v204.npz', 
    '/work/mflora/wofs-cast-data/model/wofscast_v178_reproducibility_test_v1.npz',
    '/work/mflora/wofs-cast-data/model/wofscast_v178_more_data_v2.npz'
    #'/work/cpotvin/WOFSCAST/model/wofscast_test_v213.npz',
    ]
    
    def __init__(self, 
                 model_path, 
                 full_domain = False, 
                 add_diffusion=True, 
                 sampler_kwargs = dict(
                    sigma_min = 0.0002,
                    sigma_max = 1000, 
                    S_churn=0., #np.sqrt(2)-1, 
                    S_min=0.02, 
                    S_max=800, 
                    S_noise=1.05),
                 diffusion_device='cuda'
                ): 
        
    
        self._load_model(model_path, full_domain, add_diffusion, diffusion_device, sampler_kwargs)
    
    @property
    def task_config(self):
        return self.model.task_config
    
    @property
    def preprocess_fn(self):
        return self._preprocess_fn 
    
    @property
    def decode_times(self):
        return self._decode_times 
    
    def _load_model(self, model_path, full_domain, add_diffusion, diffusion_device, sampler_kwargs):    
        
        preprocess_fn = None
        #additional_configs={}
        decode_times = True
        if model_path in self.LEGACY_MODELS:
            preprocess_fn = add_local_solar_time
            decode_times = False
        
        print('Setting legacy_mesh = True in predictor.py') 
              
        additional_configs = {"legacy_mesh" : True}    
        
        self._preprocess_fn = preprocess_fn
        self._decode_times = decode_times
        
        self.model = WoFSCastModel()

        if full_domain:
            self.model.load_model(model_path, **{'tiling' : 150, 'domain_size' : 300})
        else:    
            self.model.load_model(model_path, **additional_configs)

        self.diffusion_model = None    
        if add_diffusion:    
            self.diffusion_model = DiffusionModel(sampler_kwargs=sampler_kwargs, 
                                                 device = diffusion_device,
                                                 )

            
    def predict(self, 
                inputs, 
                targets, 
                forcings, 
                replace_bdry=True, 
                n_diffusion_steps=20, 
                initial_datetime=None, 
                n_steps = None
               ): 
    
        predictions =  self.model.predict(inputs, targets, forcings, 
                            initial_datetime=initial_datetime, 
                            n_steps=n_steps, 
                            replace_bdry=replace_bdry, 
                            diffusion_model=self.diffusion_model, 
                            n_diffusion_steps=n_diffusion_steps
                           )
        
 
        if not hasattr(predictions, 'datetime'):
            # Create the new datetime coordinate by adding the timedeltas to the initial datetime
            datetime_coord = inputs.datetime[-1].values + predictions['time'].data
        
            predictions = predictions.assign_coords(datetime = ('time', datetime_coord))
            
        return predictions
        
        
        