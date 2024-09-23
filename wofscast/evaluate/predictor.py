from ..model import WoFSCastModel
from ..diffusion import DiffusionModel
from ..data_generator import add_local_solar_time

class Predictor:
    
    LEGACY_MODELS = [
    '/work/cpotvin/WOFSCAST/model/wofscast_test_v178.npz',
    '/work2/mflora/wofs-cast-data/model/wofscast_test_v178_fine_tune_v3.npz',
    '/work/cpotvin/WOFSCAST/model/wofscast_test_v203.npz'    
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
                ): 
        
    
        self._load_model(model_path, full_domain, add_diffusion, sampler_kwargs)
    
    @property
    def task_config(self):
        return self.model.task_config
    
    @property
    def preprocess_fn(self):
        return self._preprocess_fn 
    
    @property
    def decode_times(self):
        return self._decode_times 
    
    def _load_model(self, model_path, full_domain, add_diffusion, sampler_kwargs):    
        
        preprocess_fn = None
        additional_configs={}
        decode_times = True
        if model_path in self.LEGACY_MODELS:
            preprocess_fn = add_local_solar_time
            decode_times = False
            additional_configs = {"legacy_mesh" : True}    
        
        self._preprocess_fn = preprocess_fn
        self._decode_times = decode_times
        
        self.model = WoFSCastModel()

        if full_domain:
            self.model.load_model(model_path, **{'tiling' : (2,2)})
        else:    
            self.model.load_model(model_path, **additional_configs)

        self.diffusion_model = None    
        if add_diffusion:    
            self.diffusion_model = DiffusionModel(sampler_kwargs=sampler_kwargs)

            
    def predict(self, 
                inputs, 
                targets, 
                forcings, 
                replace_bdry=True, 
                n_diffusion_steps=20
               ): 
    
        return self.model.predict(inputs, targets, forcings, 
                            initial_datetime=None, 
                            n_steps=None, 
                            replace_bdry=replace_bdry, 
                            diffusion_model=self.diffusion_model, 
                            n_diffusion_steps=n_diffusion_steps
                           )