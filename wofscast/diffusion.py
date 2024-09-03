import torch.nn.functional as F
import torch 
from diffusers import UNet2DModel
import xarray as xr 
import numpy as np
import os

from .model_utils import dataset_to_stacked, stacked_to_dataset
from . import model_utils
from .normalization import PyTorchScaler

from typing import Tuple

class DiffusionModel:
    def __init__(self, 
                 refl_only=False,
                 domain_size=150, 
                 sigma_data=0.5, 
                 sigma_max = float('inf'),
                 sigma_min=0.,
                 sampler_kwargs = dict(
                    sigma_min = 0.0002,
                    sigma_max = 1000, 
                    S_churn=0,#np.sqrt(2)-1, 
                    S_min=0.02, 
                    S_max=1000, 
                    S_noise=1.5) ,
                 device='cuda', 
                 norm_stats_path = '/work/mflora/wofs-cast-data/full_normalization_stats'
                ): 
        
        self.norm_stats_path = norm_stats_path 
        self.device = device 
        self._load_model(refl_only, domain_size, sigma_data, sigma_max, sigma_min)
        self._load_scaler()
        self.sampler_kwargs = sampler_kwargs
        
    def sample(self, dataset, num_steps=20):
        # Apply the diffusion process or any other update
        return apply_diffusion(dataset, 
                               self.model_wrapped, 
                               self.scaler, num_steps=num_steps, 
                               sampler_kwargs=self.sampler_kwargs, 
                               variables = self.variables, 
                               device=self.device
                                        )
    
    def sample_in_post(self, dataset, num_steps=20):    
        # Notes:
        # 1. Lowering the S_churn improved the RMSE; I think thats lower noise though 
        # 2. Increaseing S_max improved prediction of the higher refl. values (subjectively)
        # 3. Lowering S_noise didn't seem to do much, but that was while S_churn = 0.0, otherwise
        #    it does seem to help if S_churn > 0.0. However, it makes the image more noisy 
        # 4. Setting sigma_data=0.5, really weakened storms!! Despite the model being trained on sigma_data=1.0
 
        # Make a copy of the predictions to update
        updated_dataset = dataset.copy(deep=True)

        # Loop through each time index and update the predictions
        for t in range(dataset.sizes['time']):
            # Select predictions for the current time coordinate
            these_data = updated_dataset.isel(time=[t])
    
            # Apply the diffusion process or any other update
            updated_data = self.sample(these_data, num_steps)
    
            # Assign the updated prediction back to the correct time index in the updated_predictions dataset
            updated_dataset[dict(time=[t])] = updated_data
            
        return updated_dataset
        
    def _load_scaler(self):
        # Load the normalization statistics and initialize the 
        # scaler used for the diffusion model. 
        mean_by_level = xr.load_dataset(os.path.join(self.norm_stats_path, 'mean_by_level.nc'))
        stddev_by_level = xr.load_dataset(os.path.join(self.norm_stats_path, 'stddev_by_level.nc'))
        diffs_stddev_by_level = xr.load_dataset(os.path.join(self.norm_stats_path, 'diffs_stddev_by_level.nc'))

        self.scaler = PyTorchScaler(mean_by_level, stddev_by_level, diffs_stddev_by_level)
        
    def _load_model(self, refl_only, domain_size, sigma_data, sigma_max, sigma_min):    
        
        if refl_only:
            n_channels=1
            self.variables = ['COMPOSITE_REFL_10CM']
            diffusion_model_path = "/work/mflora/wofs-cast-data/diffusion_models/diffusion_refl_only_v7"
        else:
            n_channels=105
            self.variables = None
            diffusion_model_path = "/work/mflora/wofs-cast-data/diffusion_models/diffusion_all_vars_update_v1"

        diffusion_model = UNet2DModel.from_pretrained(diffusion_model_path).to(self.device)
     
        self.model_wrapped = EDMPrecond(domain_size, n_channels, 
                                   diffusion_model, 
                                   sigma_data=sigma_data, 
                                   sigma_max=sigma_max, 
                                   sigma_min=sigma_min)
    @property     
    def params_count(self):
        total_params = sum(p.numel() for p in self.model_wrapped.parameters())
        print(f"Number of parameters: {total_params/1_000_000:.2f} Million")
        return total_params 
    






def pad_to_multiple_of_16(tensor):
    _, _, h, w = tensor.size()
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded_tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
    return padded_tensor

def crop_to_original_size(tensor, original_height, original_width):
    # Get the current dimensions
    _, _, height, width = tensor.shape
    # Calculate the crop indices
    start_y = (height - original_height) // 2
    start_x = (width - original_width) // 2
    # Crop the tensor
    cropped_tensor = tensor[:, :, start_y:start_y+original_height, start_x:start_x+original_width]
    return cropped_tensor


class EDMPrecond(torch.nn.Module):
    """ Original Func:: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/networks.py#L519
    
    This is a wrapper for your pytorch model. It's purpose is to apply the preconditioning that is talked about in Karras et al. (2022)'s EDM paper. 
    
    I adapted the linked function to take a conditional input. Note for now, the condition is concatenated to the dimension you want denoise (dim:0 for one channel prediction). 
    
    
    
    """
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # idk if this is still needed..?
        model,                              # pytorch model from diffusers 
        label_dim       = 0,                # Ignore this for now 
        use_fp16        = True,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data. this was the default from above
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model

    def forward(self, inputs, noisy_targets, sigma, force_fp32=False, **model_kwargs):
        
        """ 
        Call method. Preconditioning model from the Karras EDM Paper. 
        
        inputs : PyTorch Tensor of shape (batch, n_channels, ny, nx)
            Conditional input images. 
        targets : PyTorch Tensor of shape (batch, n_channels, ny, nx) 
            Target images with noise-level sigma applied. 
        """
        device = model_kwargs.get('device', 'cuda') 
        
        # To follow the naming convention from RJC and Karras original code. 
        x_condition = inputs.to(torch.float32)
        x_noisy = noisy_targets.to(torch.float32) 
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        
        #forcing dtype matching?
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and noisy_targets.device.type == 'cuda') else torch.float32
        
        #get weights from EDM 
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        #concatenate back with the scaling applied to the noisy image 
        model_input_images = torch.cat([x_noisy*c_in, x_condition], dim=1)
        
        #do the model call 
        F_x = self.model((model_input_images).to(dtype), c_noise.flatten(), return_dict=False)[0]
 
        #is this needed? RJC 
        assert F_x.dtype == dtype
        #do scaling from EDM 
        D_x = c_skip * x_noisy + c_out * F_x.to(torch.float32)
        
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    
class StackedRandomGenerator:  # pragma: no cover
    """
    Wrapper for torch.Generator that allows specifying a different random seed
    for each sample in a minibatch.
    """

    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(
                f"Expected first dimension of size {len(self.generators)}, got {size[0]}"
            )
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(
                f"Expected first dimension of size {len(self.generators)}, got {size[0]}"
            )
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )
    
#################### \Classes ########################


#################### Funcs ########################

def edm_sampler(
    net, latents, condition_images, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, 
    
    # Setting from GenCast. 
    sigma_min = 0.002, # Match CorrDiff
    sigma_max= 800, # Match CorrDiff
    rho = 7,
    #S_churn = 2.5,
    #S_min = 0.75,
    #S_max = 80, 
    #S_noise = 1.05, 
    
    # original settings.
    #sigma_min=0.002, 
    #sigma_max=80, 
    #rho=7,
    S_churn=0.0, 
    S_min=0, 
    S_max=float('inf'), 
    S_noise=1,
):
    """ adapted from: https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/generate.py 
    
    only thing i had to change was provide a condition as input to this func, then take that input and concat with generated image for the model call. 
    
    """
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        #need to concat the condition here 
        #model_input_images = torch.cat([x_hat, condition_images], dim=1)
        # Euler step.
        with torch.no_grad():
            denoised = net(condition_images, x_hat, t_hat, force_fp32=True, device=latents.device).to(torch.float64)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            #model_input_images = torch.cat([x_next, condition_images], dim=1)
            with torch.no_grad():
                #denoised = net(model_input_images, t_next).to(torch.float64)
                denoised = net(condition_images, x_next, t_next, force_fp32=True, device=latents.device).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def apply_diffusion(predictions, model, scaler, num_steps=100, sampler_kwargs={}, variables=None, device='cuda'):
    """Apply a diffusion model to the current time step output."""
    #run sampler 
    batch_size = len(predictions.batch)
    original_domain_size = predictions.dims['lat']
    
    rnd = StackedRandomGenerator(device, np.arange(0,batch_size,1).astype(int).tolist())
    
    # If the diffusion model is only meant to be applied to a subset of variables.
    if variables:
        n_channels = len(variables)
        if isinstance(variables, str):
            variables = [variables] 
        
        predictions_subset = predictions[variables].copy(deep=True) 
    else:
        variables = list(predictions.data_vars)
        predictions_subset = predictions.copy(deep=True) 
    
    # Normalize the input, drop the time dimension, and stack.
    predictions_scaled = scaler.scale(predictions_subset)
    
    pred_stacked = dataset_to_stacked(
        predictions_scaled.squeeze(dim='time')).transpose('batch', 'channels', 'lat', 'lon').values 
    # Move to the GPU.
    pred_stacked_tensor = torch.tensor(pred_stacked, dtype=torch.float32).to(device) 
 
    # Convert to torch tensor and add padding. Move the tensor to the GPU
    if original_domain_size == 150:
        predicted_tensor_pad = pad_to_multiple_of_16(pred_stacked_tensor)
    else:
        predicted_tensor_pad = F.pad(pred_stacked_tensor, (10, 10, 10, 10), mode='replicate')
        # (300,300) -> (320,320); hacky!
    
    latents = torch.randn(predicted_tensor_pad.shape, device=device)
    
    # The output is the normalized residual. 
    residual = edm_sampler(model, latents, predicted_tensor_pad, num_steps=num_steps, **sampler_kwargs)
    
    # Crop the tensor back to the original size
    residual_cropped = crop_to_original_size(residual, original_domain_size, original_domain_size)
    
    residual_xarray = xr.DataArray(
        data=residual_cropped.cpu().numpy(),
        dims=("batch", "channels", "lat", "lon"))
    
    residual_xarray =  residual_xarray.transpose("batch", "lat", "lon", "channels")
    
    residual_dataset = stacked_to_dataset( residual_xarray.variable, 
                                         predictions_subset.squeeze(dim='time'), 
                                         preserved_dims=("batch", "lat", "lon")) 
    
    # Add the time dimension back and tranpose 
    residual_dataset = residual_dataset.expand_dims('time').assign_coords(time=[predictions['time'][0].values])

    dims = ('batch', 'time', 'lat', 'lon', 'level')
    residual_dataset = residual_dataset.transpose(*dims, missing_dims='ignore')
    
    # Unnormalize the target residuals and add them to the unscaled 
    # inputs to get the updated predictions. 
    unscaled_residual = scaler.unscale_residuals(residual_dataset)
    for v in variables:
        predictions[v] = predictions[v] + unscaled_residual[v]
    
    return predictions






    