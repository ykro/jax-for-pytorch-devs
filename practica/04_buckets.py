import jax
import jax.numpy as jnp

trazas = 0

@jax.jit
def resumen(x, mascara):
    global trazas
    trazas += 1
    return (jnp.tanh(x) * mascara).sum()

def con_bucket(n, buckets=(64, 128)):
    b = next(s for s in buckets if s >= n)
    x = jnp.pad(jnp.ones(n), (0, b - n))
    return resumen(x, jnp.arange(b) < n)

for n in [37, 52, 61, 37, 90, 45, 118]:
    con_bucket(n)
print("llamadas: 7  compilaciones:", trazas)
print("con bucket: ", float(con_bucket(37)))
print("sin relleno:", float(jnp.tanh(jnp.ones(37)).sum()))
