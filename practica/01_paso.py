import jax
import jax.numpy as jnp

k1, k2 = jax.random.split(jax.random.key(0))
X = jax.random.normal(k1, (256, 8))
y = X @ jax.random.normal(k2, (8,)) + 0.5
params = {"w": jnp.zeros(8), "b": jnp.array(0.0)}

def loss(params, X, y):
    pred = X @ params["w"] + params["b"]
    return jnp.mean((pred - y) ** 2)

def paso(params, X, y):
    valor, grads = jax.value_and_grad(loss)(params, X, y)
    return jax.tree.map(lambda p, g: p - 0.1 * g, params, grads), valor

for i in range(50):
    params, valor = paso(params, X, y)
    if i % 10 == 0:
        print(f"paso {i} loss {valor:.4f}")
