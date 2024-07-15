import torch.nn.functional as F
import torch 
from diffusers import UNet2DModel
import xarray as xr 
import numpy as np

def pad_to_multiple_of_16(tensor):
    _, _, h, w = tensor.size()
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded_tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))
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
    """ adapted from https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/networks.py#L519 """
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        model,                              # pytorch model from diffusers 
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
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

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        
        """ note for conditional, it expects x to have the condition in the channel dim. and the image to already be noised """
        
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        #split out the noisy image 
        x_noisy = torch.clone(x[:,[0]])
        x_condition = torch.clone(x[:,[1]])
        #concatinate back 
        model_input_images = torch.cat([x_noisy*c_in, x_condition], dim=1)
        F_x = self.model((model_input_images).to(dtype), c_noise.flatten(), return_dict=False)[0]

        assert F_x.dtype == dtype
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
    net, latents, condition_images,class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0.0, S_min=0, S_max=float('inf'), S_noise=1,
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
        model_input_images = torch.cat([x_hat, condition_images], dim=1)
        # Euler step.
        with torch.no_grad():
            denoised = net(model_input_images, t_hat).to(torch.float64)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            model_input_images = torch.cat([x_next, condition_images], dim=1)
            with torch.no_grad():
                denoised = net(model_input_images, t_next).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def unscale(data, std=13.0, mean=7.348):
    data = data * std
    data = data + mean
    return data

def scale(data, std=13.0, mean=7.348):
    scaled_data = (data - mean) / std
    return scaled_data

def apply_diffusion(predictions, model, num_steps=100):
    """Apply a diffusion model to the composite reflectivity predictions"""
    #run sampler 
    batch_size = len(predictions.batch)
    domain_size = 160 # Expected domain size for the diffuser. Made it larger for deeper U-net. 
    
    rnd = StackedRandomGenerator('cuda',np.arange(0,batch_size,1).astype(int).tolist())
    
    # Reorder so time dim is first. 
    predictions = predictions.transpose('time', 'batch', 'lat', 'lon', 'level')
    predicted_refl = predictions['COMPOSITE_REFL_10CM'].values # shape= time, batch, ny, nx 
    
    #scale data
    predicted_refl_scaled = scale(predicted_refl)

    
    data_over_time = []
    for t in range(predicted_refl.shape[0]):
        this_predicted_refl = predicted_refl_scaled[t,...]
    
        # Add channel dimension
        this_predicted_refl = this_predicted_refl[:, np.newaxis, :, :]

        # Convert to torch tensor and add padding. Move the tensor to the GPU
        predicted_refl_tensor = torch.tensor(this_predicted_refl, dtype=torch.float32).cuda()
        predicted_refl_tensor = pad_to_multiple_of_16(predicted_refl_tensor)
    
        latents = rnd.randn([batch_size, 1, domain_size, domain_size],device='cuda')
        images_batch = edm_sampler(model,latents, predicted_refl_tensor, num_steps=num_steps)
    
        # Crop the tensor back to the original size
        images_batch_cropped = crop_to_original_size(images_batch, 150, 150)
    
        # Convert the torch tensor back to numpy; also need to add the time dimension
        # Also unscale the data. 
        images_batch_np = unscale(images_batch_cropped.cpu().numpy())
    
        # Create a new DataArray with the updated predictions
        data = xr.DataArray(images_batch_np, 
                        dims=['batch', 'channel', 'lat', 'lon'])
    
        # Remove the channel dimension
        data = data.squeeze('channel')
        
        # Add the time dimension back
        data = data.expand_dims('time').assign_coords(time=[predictions['time'][t].values])
    
        data_over_time.append(data)
    
    data = xr.concat(data_over_time, dim='time')
    
    # Replace the original predictions with the new data
    predictions['COMPOSITE_REFL_10CM'] = data

    return predictions 