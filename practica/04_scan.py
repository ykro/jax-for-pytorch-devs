import jax
import jax.numpy as jnp

N = 500

def paso(estado, _):
    x, v = estado
    v = 0.9 * v - 0.1 * jnp.sin(x)
    return (x + v, v), x

def con_for(estado):
    for _ in range(N):
        estado, _ = paso(estado, None)
    return estado[0]

def con_scan(estado):
    final, _ = jax.lax.scan(paso, estado, None, length=N)
    return final[0]

inicio = (jnp.array(1.0), jnp.array(0.0))
print("iguales:", bool(jax.jit(con_for)(inicio) == jax.jit(con_scan)(inicio)))
print("ecuaciones con for: ", len(jax.make_jaxpr(con_for)(inicio).eqns))
print("ecuaciones con scan:", len(jax.make_jaxpr(con_scan)(inicio).eqns))
