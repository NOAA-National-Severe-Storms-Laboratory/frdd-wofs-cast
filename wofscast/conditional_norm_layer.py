import haiku as hk
import jax
import jax.numpy as jnp
import jraph

class ConditionalLayerNorm(hk.Module):
    def __init__(self, name, noise_level_dim=16, num_frequencies=32):
        super().__init__(name=name)
        self.noise_level_dim = noise_level_dim
        self.num_frequencies = num_frequencies

    def __call__(self, x, noise_level):
        # Create sine/cosine Fourier features
        frequencies = jnp.linspace(1.0, 16.0, self.num_frequencies)
        noise_level_features = jnp.concatenate([
            jnp.sin(frequencies * noise_level[:, None]),
            jnp.cos(frequencies * noise_level[:, None])
        ], axis=-1)

        # Pass through a 2-layer MLP
        mlp = hk.Sequential([
            hk.Linear(32),
            jax.nn.relu,
            hk.Linear(self.noise_level_dim)
        ])
        noise_level_encoding = mlp(noise_level_features)

        # Apply conditional layer norm
        ln = hk.LayerNorm(
            axis=-1,
            create_scale=False,
            create_offset=False
        )
        normalized_x = ln(x)
        
        # Create scale and offset using a linear layer
        scale_and_offset = hk.Linear(2 * x.shape[-1])(noise_level_encoding)
        scale, offset = jnp.split(scale_and_offset, 2, axis=-1)

        return normalized_x * (1 + scale) + offset