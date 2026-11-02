"""Experiment 05: The Dynamic Shapes Trap & Static Bucket Padding.

Demonstrates how variable sequence lengths trigger silent `@jax.jit`
recompilations because JAX traces on static `ShapedArray(shape, dtype)`
signatures. Fixes the recompilation storm using static bucket padding + masking.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

BUCKETS = (64, 128, 256)


def make_traced_encoder() -> tuple[Any, dict[str, int]]:
    """Create a @jax.jit masked pooling encoder that counts how many times Python tracing runs."""
    stats = {"trace_count": 0}

    @jax.jit
    def masked_encoder(tokens: jax.Array, mask: jax.Array, w: jax.Array) -> jax.Array:
        # Python side-effect executes ONLY during tracing/compilation!
        stats["trace_count"] += 1
        projected = jnp.tanh(tokens @ w)
        masked = projected * mask[:, None]
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        return jnp.sum(masked, axis=0) / denom

    return masked_encoder, stats


def select_bucket(length: int, buckets: tuple[int, ...] = BUCKETS) -> int:
    """Pick the smallest static bucket >= length."""
    for b in buckets:
        if length <= b:
            return b
    raise ValueError(f"Sequence length {length} exceeds max bucket {buckets[-1]}")


def pad_to_bucket(
    x: np.ndarray, buckets: tuple[int, ...] = BUCKETS
) -> tuple[jax.Array, jax.Array]:
    """Pad variable-length `(L, D)` array to `(B, D)` and build a `float32` mask `(B,)`."""
    seq_len, dim = x.shape
    bucket_len = select_bucket(seq_len, buckets)
    padded = np.zeros((bucket_len, dim), dtype=np.float32)
    padded[:seq_len] = x
    mask = np.zeros((bucket_len,), dtype=np.float32)
    mask[:seq_len] = 1.0
    return jnp.asarray(padded), jnp.asarray(mask)


def run_experiment() -> dict[str, Any]:
    """Compare naive dynamic-length JIT calls vs static bucketed JIT calls."""
    rng = np.random.default_rng(99)
    dim = 128
    w = jnp.asarray(rng.standard_normal((dim, dim), dtype=np.float32) / np.sqrt(dim))

    # 24 distinct sequence lengths between 12 and 250
    seq_lengths = [12 + i * 10 for i in range(24)]
    sequences = [rng.standard_normal((length, dim), dtype=np.float32) for length in seq_lengths]

    # 1. Naive dynamic shapes: every distinct sequence length triggers a new trace + XLA compile
    naive_encoder, naive_stats = make_traced_encoder()
    t0 = time.perf_counter()
    naive_outputs = []
    for seq in sequences:
        tokens = jnp.asarray(seq)
        mask = jnp.ones((seq.shape[0],), dtype=jnp.float32)
        out = naive_encoder(tokens, mask, w)
        out.block_until_ready()
        naive_outputs.append(np.asarray(out))
    naive_total_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Static bucketed shapes: warm up the 3 buckets once, then process all 24 sequences
    bucket_encoder, bucket_stats = make_traced_encoder()
    for b_len in BUCKETS:
        dummy_tokens = jnp.zeros((b_len, dim), dtype=jnp.float32)
        dummy_mask = jnp.ones((b_len,), dtype=jnp.float32)
        bucket_encoder(dummy_tokens, dummy_mask, w).block_until_ready()

    traces_after_warmup = bucket_stats["trace_count"]

    t0 = time.perf_counter()
    bucket_outputs = []
    for seq in sequences:
        padded_tokens, mask = pad_to_bucket(seq, BUCKETS)
        out = bucket_encoder(padded_tokens, mask, w)
        out.block_until_ready()
        bucket_outputs.append(np.asarray(out))
    bucket_total_ms = (time.perf_counter() - t0) * 1000.0

    extra_recompilations = bucket_stats["trace_count"] - traces_after_warmup
    max_output_diff = float(
        np.max([np.max(np.abs(a - b)) for a, b in zip(naive_outputs, bucket_outputs, strict=True)])
    )

    return {
        "num_sequences": len(sequences),
        "buckets": list(BUCKETS),
        "naive_trace_count": naive_stats["trace_count"],
        "bucket_warmup_traces": traces_after_warmup,
        "bucket_extra_recompilations": extra_recompilations,
        "naive_total_ms": round(naive_total_ms, 2),
        "bucket_total_ms": round(bucket_total_ms, 2),
        "speedup": round(naive_total_ms / bucket_total_ms, 2),
        "max_output_diff": max_output_diff,
    }


def main() -> None:
    res = run_experiment()
    print("=" * 72)
    print("Experiment 05: Dynamic Shapes Recompilation Trap & Bucket Padding")
    print("=" * 72)
    print(f"Distinct Sequence Lengths : {res['num_sequences']} sequences (lengths 12..242)")
    print(f"Static Buckets Configured : {res['buckets']}")
    print("-" * 72)
    print(f"Naive Dynamic Trace Count : {res['naive_trace_count']} XLA compilations")
    print(f"Naive Total Execution Time: {res['naive_total_ms']:.2f} ms")
    print(f"Bucketed Warmup Traces    : {res['bucket_warmup_traces']} XLA compilations")
    print(f"Extra Recompilations      : {res['bucket_extra_recompilations']} (0 cache misses in prod)")
    print(f"Bucketed Execution Time   : {res['bucket_total_ms']:.2f} ms")
    print(f"Latency Speedup           : {res['speedup']:.2f}x (bucket warmup compiles excluded)")
    print(f"Max Masked Output Diff    : {res['max_output_diff']:.2e}")


if __name__ == "__main__":
    main()
