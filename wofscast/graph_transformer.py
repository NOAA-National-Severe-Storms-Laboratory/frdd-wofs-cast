import jax
import jax.numpy as jnp
import haiku as hk
from jax import random
from haiku import MultiHeadAttention

from typing import Optional, Callable, Mapping, Union
import dataclasses
from functools import partial

def create_adjacency_matrix(senders, receivers, num_nodes):
    """Create adjacency matrix from senders and receivers arrays."""
    # Initialize a zero matrix of shape (num_nodes, num_nodes)
    adjacency_matrix = jnp.zeros((num_nodes, num_nodes), dtype=jnp.int32)

    # Set entries where there is an edge
    adjacency_matrix.at[senders, receivers].set(1)

    return adjacency_matrix


@partial(jax.jit, static_argnames=['k'])
def compute_k_hop_adjacency_matrix(adjacency_matrix, k):
    """
    Compute the k-hop adjacency matrix using JAX, accelerated with JIT compilation.
    Handles dynamic looping within JAX's framework.
    """        
    # Initialize the result with the original adjacency matrix
    result = adjacency_matrix

    # Use JAX's lax for dynamic looping
    for _ in range(k-1):
        result = jnp.dot(result, adjacency_matrix)

    # Convert to binary connectivity matrix (1 if reachable, 0 otherwise)
    result = (result > 0).astype(jnp.int32)
    return result

def _layer_norm(x: jax.Array) -> jax.Array:
    """Applies a unique LayerNorm to `x` with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)

@dataclasses.dataclass
class Transformer(hk.Module):
    """A transformer stack."""

    num_heads: int  # Number of attention heads.
    num_layers: int  # Number of transformer (attention + MLP) layers to stack.
    attn_size: int  # Size of the attention (key, query, value) vectors.
    output_size : int # Final output size of D 
    dropout_rate: float  # Probability with which to apply dropout.
    widening_factor: int = 4  # Factor by which the MLP hidden layer widens.
    name: Optional[str] = None  # Optional identifier for the module.
    mask : jax.Array = None
        
    def __call__(
        self,
        embeddings: jax.Array,  # [Number of Nodes, Batch, Number of Features]
    ) -> jax.Array:  # [Number of Nodes, Batch, Number of Features]
        """Transforms input embedding sequences to output embedding sequences."""

        embeddings = jnp.transpose(embeddings, (1, 0, 2))  # [batch_size, n_nodes, n_features]
        
        initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
        batch_size, n_nodes, model_size = embeddings.shape 

        # Create an attention mask from the adjacency matrix
        # Assuming mask is [N, N], where N is the number of nodes -> [B,N,N]
        mask = jnp.stack([self.mask for b in range(batch_size)])
    
        attention_mask = (mask > 0).astype(jnp.float32)
        # Expand mask for multi-head attention
        attention_mask = attention_mask[:, None, :, :]  # [B, 1, N, N]
        attention_mask = jnp.broadcast_to(attention_mask, (attention_mask.shape[0], 
                                                           self.num_heads, 
                                                           attention_mask.shape[2], 
                                                           attention_mask.shape[2])) # [B, H, N, N]
         
        ###print(f'\n {embeddings.shape=}, {attention_mask.shape=}')
        
        h = embeddings
        for _ in range(self.num_layers):
            # First the attention block.
            attn_block = hk.MultiHeadAttention(
              num_heads=self.num_heads,
              key_size=self.attn_size,
              model_size=model_size,
              w_init=initializer,
          )
            h_norm = _layer_norm(h)
            h_attn = attn_block(h_norm, h_norm, h_norm, mask=attention_mask)
            h_attn = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_attn)
            h = h + h_attn

            # Then the dense block.
            dense_block = hk.Sequential([
              hk.Linear(model_size, w_init=initializer),
              jax.nn.gelu,
              hk.Linear(model_size, w_init=initializer),
              ])
            h_norm = _layer_norm(h)
            h_dense = dense_block(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_dense)
            h = h + h_dense

        # Match the output size!
        h = hk.Linear(self.output_size, w_init=initializer)(h)
        h = jax.nn.relu(h)
            
        # Reshape h to (N, B, D)
        h = jnp.transpose(h, [1, 0, 2])
                
        return _layer_norm(h)