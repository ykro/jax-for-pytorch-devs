"""Experiment 01: Eager Dispatch vs @jax.jit XLA Operator Fusion.

Compares PyTorch eager execution, JAX eager execution, and @jax.jit XLA-fused
execution on a multi-step residual + GELU + RMSNorm activation block.
Also inspects the intermediate Jaxpr and lowered StableHLO MLIR module.
"""

from __future__ import annotations

import math
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch


def pytorch_gelu_residual_block(
    x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Multi-step elementwise residual block in PyTorch eager mode."""
    h = x
    for _ in range(8):
        norm = h * torch.rsqrt(torch.mean(h * h, dim=-1, keepdim=True) + 1e-6)
        z = norm * scale + bias
        gelu = 0.5 * z * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * (z**3))))
        h = h + 0.25 * gelu
    return h


def jax_gelu_residual_block(x: jax.Array, scale: jax.Array, bias: jax.Array) -> jax.Array:
    """Identical multi-step elementwise residual block in JAX."""
    h = x
    for _ in range(8):
        norm = h * jax.lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + 1e-6)
        z = norm * scale + bias
        gelu = 0.5 * z * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * (z**3))))
        h = h + 0.25 * gelu
    return h


jax_jit_gelu_residual_block = jax.jit(jax_gelu_residual_block)


def run_benchmark(batch_size: int = 64, hidden_dim: int = 128, n_iters: int = 100) -> dict[str, Any]:
    """Run numerical equivalence, IR inspection, and timing comparison."""
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((batch_size, hidden_dim), dtype=np.float32)
    scale_np = rng.uniform(0.8, 1.2, size=(hidden_dim,)).astype(np.float32)
    bias_np = rng.uniform(-0.1, 0.1, size=(hidden_dim,)).astype(np.float32)

    x_torch = torch.from_numpy(x_np)
    scale_torch = torch.from_numpy(scale_np)
    bias_torch = torch.from_numpy(bias_np)

    x_jax = jnp.asarray(x_np)
    scale_jax = jnp.asarray(scale_np)
    bias_jax = jnp.asarray(bias_np)

    # Warmup
    out_torch = pytorch_gelu_residual_block(x_torch, scale_torch, bias_torch)
    out_jax_eager = jax_gelu_residual_block(x_jax, scale_jax, bias_jax).block_until_ready()
    out_jax_jit = jax_jit_gelu_residual_block(x_jax, scale_jax, bias_jax).block_until_ready()

    max_diff_torch_jit = float(np.max(np.abs(out_torch.numpy() - np.asarray(out_jax_jit))))
    max_diff_eager_jit = float(np.max(np.abs(np.asarray(out_jax_eager) - np.asarray(out_jax_jit))))

    # Measure PyTorch Eager
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = pytorch_gelu_residual_block(x_torch, scale_torch, bias_torch)
    torch_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    # Measure JAX Eager
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = jax_gelu_residual_block(x_jax, scale_jax, bias_jax).block_until_ready()
    jax_eager_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    # Measure JAX @jit (XLA fused)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = jax_jit_gelu_residual_block(x_jax, scale_jax, bias_jax).block_until_ready()
    jax_jit_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    jaxpr = jax.make_jaxpr(jax_gelu_residual_block)(x_jax, scale_jax, bias_jax)
    lowered_hlo = jax_jit_gelu_residual_block.lower(x_jax, scale_jax, bias_jax).as_text("stablehlo")

    return {
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "n_iters": n_iters,
        "torch_eager_ms": round(torch_ms, 4),
        "jax_eager_ms": round(jax_eager_ms, 4),
        "jax_jit_ms": round(jax_jit_ms, 4),
        "speedup_vs_jax_eager": round(jax_eager_ms / jax_jit_ms, 2),
        "speedup_vs_torch_eager": round(torch_ms / jax_jit_ms, 2),
        "max_diff_torch_jit": max_diff_torch_jit,
        "max_diff_eager_jit": max_diff_eager_jit,
        "jaxpr_num_equations": len(jaxpr.jaxpr.eqns),
        "stablehlo_lines": len(lowered_hlo.strip().splitlines()),
        "jaxpr_preview": "\n".join(str(jaxpr).splitlines()[:12]),
        "stablehlo_preview": "\n".join(lowered_hlo.strip().splitlines()[:12]),
    }


def main() -> None:
    res = run_benchmark()
    print("=" * 72)
    print("Experiment 01: Eager Dispatch vs @jax.jit XLA Operator Fusion")
    print("=" * 72)
    print(f"Backend Device          : {jax.devices()[0].device_kind} ({jax.default_backend()})")
    print(f"Tensor Shape            : ({res['batch_size']}, {res['hidden_dim']}) float32")
    print(f"Jaxpr Equations Traced  : {res['jaxpr_num_equations']} primitive ops")
    print(f"Lowered StableHLO Lines : {res['stablehlo_lines']} lines")
    print("-" * 72)
    print(f"PyTorch Eager Avg Time  : {res['torch_eager_ms']:.4f} ms")
    print(f"JAX Eager Avg Time      : {res['jax_eager_ms']:.4f} ms")
    print(f"JAX @jit Fused Avg Time : {res['jax_jit_ms']:.4f} ms")
    print(f"Speedup vs JAX Eager    : {res['speedup_vs_jax_eager']:.2f}x")
    print(f"Speedup vs PyTorch Eager: {res['speedup_vs_torch_eager']:.2f}x")
    print(f"Max |PyTorch - JAX JIT| : {res['max_diff_torch_jit']:.2e}")
    print("-" * 72)
    print("Jaxpr Preview (first 12 lines):")
    print(res["jaxpr_preview"])
    print("-" * 72)
    print("StableHLO Preview (first 12 lines):")
    print(res["stablehlo_preview"])


if __name__ == "__main__":
    main()
