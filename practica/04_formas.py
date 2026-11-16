import jax
import jax.numpy as jnp

trazas = 0

@jax.jit
def resumen(x):
    global trazas
    trazas += 1
    return jnp.cos(x).sum()

for n in [37, 52, 61, 37, 90, 45, 118]:
    resumen(jnp.ones(n))
print("llamadas: 7  compilaciones:", trazas)
