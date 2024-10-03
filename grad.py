import numpy as np
import jax
import jax.numpy as jnp
def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
def selu_numpy(x, alpha=1.67, lambda_=1.05):
  return lambda_ * np.where(x > 0, x, alpha * np.exp(x) - alpha)
selu_jit = jax.jit(selu)

x = jnp.arange(1000000)
%timeit selu(x).block_until_ready()
%timeit selu_jit(x).block_until_ready()
y = np.arange(1000000)
%timeit selu_numpy(y)