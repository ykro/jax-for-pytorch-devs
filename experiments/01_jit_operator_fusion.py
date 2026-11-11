"""Experiment 01: Eager Dispatch vs @jax.jit XLA Operator Fusion.

Compares PyTorch eager execution, JAX eager execution, and @jax.jit XLA-fused
execution on a multi-step residual + GELU + RMSNorm activation block.
Also inspects the intermediate Jaxpr and lowered StableHLO MLIR module.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from collections.abc import Callable
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


def _median_ms(fn: Callable[[], Any], n_iters: int) -> float:
    """Median wall time of `fn()` in ms (robust to noisy neighbours on a shared machine)."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


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

    # Median per call after the warmup above (the first jit call, which compiles, is excluded)
    torch_ms = _median_ms(lambda: pytorch_gelu_residual_block(x_torch, scale_torch, bias_torch), n_iters)
    jax_eager_ms = _median_ms(
        lambda: jax_gelu_residual_block(x_jax, scale_jax, bias_jax).block_until_ready(), n_iters
    )
    jax_jit_ms = _median_ms(
        lambda: jax_jit_gelu_residual_block(x_jax, scale_jax, bias_jax).block_until_ready(), n_iters
    )

    jaxpr = jax.make_jaxpr(jax_gelu_residual_block)(x_jax, scale_jax, bias_jax)
    lowered = jax_jit_gelu_residual_block.lower(x_jax, scale_jax, bias_jax)
    lowered_hlo = lowered.as_text("stablehlo")
    # Fusions in the optimized program XLA compiles (one executable, several fused kernels inside)
    compiled_hlo = lowered.compile().as_text()
    xla_fusions = len(re.findall(r"^\s*(?:ROOT\s+)?%\S+ = \S+ fusion\(", compiled_hlo, re.MULTILINE))

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
        "xla_fusions": xla_fusions,
        "jaxpr_preview": "\n".join(str(jaxpr).splitlines()[:12]),
        "stablehlo_preview": "\n".join(lowered_hlo.strip().splitlines()[:12]),
    }


def report(res: dict[str, Any]) -> str:
    """Human-readable report of one run (printed by main() and by verify_speedup.py --verbose)."""
    lines = [
        "=" * 72,
        "Experiment 01: Eager Dispatch vs @jax.jit XLA Operator Fusion",
        "=" * 72,
        f"Backend Device          : {jax.devices()[0].device_kind} ({jax.default_backend()})",
        f"Tensor Shape            : ({res['batch_size']}, {res['hidden_dim']}) float32",
        f"Jaxpr Equations Traced  : {res['jaxpr_num_equations']} primitive ops",
        f"Lowered StableHLO Lines : {res['stablehlo_lines']} lines",
        f"Compiled XLA Fusions    : {res['xla_fusions']} fusions in one executable (8 residual blocks)",
        "-" * 72,
        f"PyTorch Eager Median    : {res['torch_eager_ms']:.4f} ms",
        f"JAX Eager Median        : {res['jax_eager_ms']:.4f} ms",
        f"JAX @jit Median         : {res['jax_jit_ms']:.4f} ms",
        f"Speedup vs JAX Eager    : {res['speedup_vs_jax_eager']:.2f}x",
        f"Speedup vs PyTorch Eager: {res['speedup_vs_torch_eager']:.2f}x",
        f"Max |PyTorch - JAX JIT| : {res['max_diff_torch_jit']:.2e}",
        "-" * 72,
        "Jaxpr Preview (first 12 lines):",
        res["jaxpr_preview"],
        "-" * 72,
        "StableHLO Preview (first 12 lines):",
        res["stablehlo_preview"],
    ]
    return "\n".join(lines)


def main() -> None:
    print(report(run_benchmark()))


if __name__ == "__main__":
    main()
