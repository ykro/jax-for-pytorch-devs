import jax
import jax.numpy as jnp


@jax.jit
def activa(x):
    return jnp.where(x.sum() > 0, x, -x)

print(activa(jnp.array([1.0, -2.0, 3.0])))
print(activa(jnp.array([-1.0, -2.0, 3.0])))
