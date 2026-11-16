import jax.numpy as jnp

pesos = jnp.zeros(4)
nuevos = pesos.at[1].set(0.5)
print("pesos: ", pesos)
print("nuevos:", nuevos)
