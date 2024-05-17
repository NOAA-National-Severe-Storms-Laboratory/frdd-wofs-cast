import numpy as np 
import jax.numpy as jnp

# Size of the border around the limited area 
# domain where the input is replaced with the
# data from the targets input. During training, 
# it would be the ground truth targets, but at
# deployment, it would be input from some parent 
# domain e.g., the HRRR or GFS. 
BORDER_MASK_SIZE = 5 
SHAPE = (150, 150) 

def create_border_mask(shape, border_mask_size=5, use_jax=True):
    """
    Create a border mask for an array of given shape.

    Parameters:
    - shape: tuple, the shape of the array (NY, NX).
    - N: int, the width of the border where values should be True.

    Returns:
    - mask: jax.numpy.ndarray, a mask where border values are True and interior values are False.
    """
    mask = np.zeros(shape, dtype=bool)

    # Set the border to True
    mask[:border_mask_size, :] = True  # Top border
    mask[-border_mask_size:, :] = True  # Bottom border
    mask[:, :border_mask_size] = True  # Left border
    mask[:, -border_mask_size:] = True  # Right border

    if use_jax:
        return jnp.array(mask, dtype=jnp.bfloat16)
    
    return mask 

BORDER_MASK_JAX = create_border_mask(SHAPE, BORDER_MASK_SIZE, use_jax=True)
BORDER_MASK_NUMPY = create_border_mask(SHAPE, BORDER_MASK_SIZE, use_jax=False)