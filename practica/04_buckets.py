import jax
import jax.numpy as jnp

trazas = 0

@jax.jit
def resumen(x, mascara):
    global trazas
    trazas += 1
    return (jnp.cos(x) * mascara).sum()

def con_bucket(n, buckets=(64, 128), usar_mascara=True):
    b = next(s for s in buckets if s >= n)
    x = jnp.pad(jnp.ones(n), (0, b - n))
    mascara = jnp.arange(b) < n if usar_mascara else jnp.ones(b, bool)
    return resumen(x, mascara)

for n in [37, 52, 61, 37, 90, 45, 118]:
    con_bucket(n)
print("llamadas: 7  compilaciones:", trazas)
print("con máscara:", float(con_bucket(37)))
print("sin máscara:", float(con_bucket(37, usar_mascara=False)))
print("sin relleno:", float(jnp.cos(jnp.ones(37)).sum()))
