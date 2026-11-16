import jax
import jax.numpy as jnp


@jax.jit
def activa(x):
    if x.sum() > 0:
        return x
    return -x

print(activa(jnp.array([1.0, -2.0, 3.0])))
