import jax
import jax.numpy as jnp


@jax.jit
def bloque(x):
    print("trazando con", x.shape, x.dtype)
    h = x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6)
    return x + jax.nn.gelu(h)

a = jnp.ones((64, 128))
bloque(a)
bloque(a)
bloque(a)
print("tres llamadas listas")
bloque(jnp.ones((32, 128)))
