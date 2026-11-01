"""Experiment 03: Eliminating Python Loops with jax.vmap(jax.grad(...)).

Computes per-example clipped gradients (as used in DP-SGD and per-sample
attribution) across a batch of examples. Compares a sequential loop of single-sample
gradients against a single vectorized `jax.jit(jax.vmap(jax.grad(...)))` call.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def single_example_loss(
    w: jax.Array, b: jax.Array, x_i: jax.Array, y_i: jax.Array
) -> jax.Array:
    """Loss for a SINGLE example (1D feature vector x_i, scalar/1D target y_i).

    Notice there is zero batch dimension in this function.
    """
    logits = jnp.tanh(x_i @ w + b)
    residual = logits - y_i
    return 0.5 * jnp.sum(residual * residual)


single_example_grad = jax.jit(jax.grad(single_example_loss, argnums=(0, 1)))


def loop_per_sample_clipped_grads(
    w: jax.Array, b: jax.Array, x_batch: jax.Array, y_batch: jax.Array, max_norm: float = 1.0
) -> tuple[jax.Array, jax.Array]:
    """Sequential Python loop computing per-example clipped gradients."""
    batch_size = x_batch.shape[0]
    gw_list = []
    gb_list = []
    for i in range(batch_size):
        gw, gb = single_example_grad(w, b, x_batch[i], y_batch[i])
        norm = jnp.sqrt(jnp.sum(gw * gw) + jnp.sum(gb * gb) + 1e-12)
        scale = jnp.minimum(1.0, max_norm / norm)
        gw_list.append(gw * scale)
        gb_list.append(gb * scale)
    return jnp.mean(jnp.stack(gw_list, axis=0), axis=0), jnp.mean(
        jnp.stack(gb_list, axis=0), axis=0
    )


@jax.jit
def vmap_per_sample_clipped_grads(
    w: jax.Array, b: jax.Array, x_batch: jax.Array, y_batch: jax.Array, max_norm: float = 1.0
) -> tuple[jax.Array, jax.Array]:
    """Vectorized per-example gradient computation pushed down into XLA via vmap."""
    grad_fn = jax.grad(single_example_loss, argnums=(0, 1))
    batched_grad_fn = jax.vmap(grad_fn, in_axes=(None, None, 0, 0))
    gw_batch, gb_batch = batched_grad_fn(w, b, x_batch, y_batch)

    norms = jnp.sqrt(
        jnp.sum(gw_batch * gw_batch, axis=(1, 2)) + jnp.sum(gb_batch * gb_batch, axis=1) + 1e-12
    )
    scales = jnp.minimum(1.0, max_norm / norms)
    gw_clipped = gw_batch * scales[:, None, None]
    gb_clipped = gb_batch * scales[:, None]
    return jnp.mean(gw_clipped, axis=0), jnp.mean(gb_clipped, axis=0)


def run_benchmark(batch_size: int = 128, in_dim: int = 64, out_dim: int = 16, n_iters: int = 20) -> dict[str, Any]:
    """Benchmark Python loop of per-sample gradients vs vmap(grad)."""
    rng = np.random.default_rng(101)
    w = jnp.asarray(rng.standard_normal((in_dim, out_dim), dtype=np.float32) * 0.1)
    b = jnp.zeros((out_dim,), dtype=jnp.float32)
    x_batch = jnp.asarray(rng.standard_normal((batch_size, in_dim), dtype=np.float32))
    y_batch = jnp.asarray(rng.standard_normal((batch_size, out_dim), dtype=np.float32))

    # Warmup both paths
    gw_loop, gb_loop = loop_per_sample_clipped_grads(w, b, x_batch, y_batch)
    gw_loop.block_until_ready()
    gw_vmap, gb_vmap = vmap_per_sample_clipped_grads(w, b, x_batch, y_batch)
    gw_vmap.block_until_ready()

    max_diff_gw = float(np.max(np.abs(np.asarray(gw_loop) - np.asarray(gw_vmap))))
    max_diff_gb = float(np.max(np.abs(np.asarray(gb_loop) - np.asarray(gb_vmap))))

    # Measure Python loop
    t0 = time.perf_counter()
    for _ in range(n_iters):
        gw_l, _ = loop_per_sample_clipped_grads(w, b, x_batch, y_batch)
        gw_l.block_until_ready()
    loop_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    # Measure vmap(grad)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        gw_v, _ = vmap_per_sample_clipped_grads(w, b, x_batch, y_batch)
        gw_v.block_until_ready()
    vmap_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    return {
        "batch_size": batch_size,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "n_iters": n_iters,
        "loop_ms": round(loop_ms, 4),
        "vmap_ms": round(vmap_ms, 4),
        "speedup": round(loop_ms / vmap_ms, 2),
        "max_diff_gw": max_diff_gw,
        "max_diff_gb": max_diff_gb,
    }


def main() -> None:
    res = run_benchmark()
    print("=" * 72)
    print("Experiment 03: Per-Sample Gradients with jax.vmap(jax.grad(...))")
    print("=" * 72)
    print(f"Batch Size              : {res['batch_size']} examples ({res['in_dim']} -> {res['out_dim']})")
    print(f"Python For-Loop Avg Time: {res['loop_ms']:.4f} ms")
    print(f"jax.vmap(grad) Avg Time : {res['vmap_ms']:.4f} ms")
    print(f"Vectorized Speedup      : {res['speedup']:.2f}x")
    print(f"Max |Grad W| Difference : {res['max_diff_gw']:.2e}")
    print(f"Max |Grad b| Difference : {res['max_diff_gb']:.2e}")


if __name__ == "__main__":
    main()
