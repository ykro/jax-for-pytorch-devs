import jax
import jax.numpy as jnp

B = 128
k1, k2 = jax.random.split(jax.random.key(0))
X = jax.random.normal(k1, (B, 8))
y = jax.random.normal(k2, (B,))
w = jnp.zeros(8)

def loss_uno(w, x, y):
    return (x @ w - y) ** 2

g_batch = jax.grad(lambda w: jnp.mean((X @ w - y) ** 2))(w)
g_ejemplo = jax.vmap(jax.grad(loss_uno), in_axes=(None, 0, 0))(w, X, y)
print("gradiente del batch:", g_batch.shape)
print("uno por ejemplo:    ", g_ejemplo.shape)
iguales = jnp.allclose(g_ejemplo.mean(0), g_batch)
print("su promedio coincide:", bool(iguales))

normas = jnp.linalg.norm(g_ejemplo, axis=1)
recortados = g_ejemplo * jnp.minimum(1.0, 1.0 / normas)[:, None]
print("con norma > 1:", int((normas > 1).sum()), "de", len(normas))
print("promedio recortado:", recortados.mean(0).shape)
