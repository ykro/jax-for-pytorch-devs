"""Experiment 04: Sequential Loops in XLA with jax.lax.scan.

Compares a 500-step recurrent state-space / momentum trajectory unrolled via a
Python `for` loop vs `jax.lax.scan` compiled into a single XLA while loop.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def step_cell(
    carry: tuple[jax.Array, jax.Array], x_t: jax.Array, w_rec: jax.Array
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Single recurrent step updating hidden state h and momentum velocity v."""
    h, v = carry
    v_next = 0.9 * v + jnp.tanh(h @ w_rec + x_t)
    h_next = 0.95 * h + 0.05 * v_next
    return (h_next, v_next), h_next


def _median_ms(fn: Callable[[], Any], n_iters: int) -> float:
    """Median wall time of `fn()` in ms (robust to noisy neighbours on a shared machine)."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def python_loop_rollout(
    h0: jax.Array, v0: jax.Array, xs: jax.Array, w_rec: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Rollout 500 recurrent steps using a standard Python for-loop."""
    carry = (h0, v0)
    outputs = []
    for t in range(xs.shape[0]):
        carry, y_t = step_cell(carry, xs[t], w_rec)
        outputs.append(y_t)
    return carry[0], jnp.stack(outputs, axis=0)


@jax.jit
def lax_scan_rollout(
    h0: jax.Array, v0: jax.Array, xs: jax.Array, w_rec: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Rollout 500 recurrent steps inside XLA using jax.lax.scan."""

    def body_fn(
        carry: tuple[jax.Array, jax.Array], x_t: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        h, v = carry
        v_next = 0.9 * v + jnp.tanh(h @ w_rec + x_t)
        h_next = 0.95 * h + 0.05 * v_next
        return (h_next, v_next), h_next

    (h_final, _), ys = jax.lax.scan(body_fn, (h0, v0), xs)
    return h_final, ys


def run_benchmark(seq_len: int = 500, batch_size: int = 32, hidden_dim: int = 64, n_iters: int = 15) -> dict[str, Any]:
    """Compare Python loop dispatch overhead vs compiled jax.lax.scan."""
    rng = np.random.default_rng(2026)
    h0 = jnp.asarray(rng.standard_normal((batch_size, hidden_dim), dtype=np.float32) * 0.1)
    v0 = jnp.zeros((batch_size, hidden_dim), dtype=jnp.float32)
    xs = jnp.asarray(rng.standard_normal((seq_len, batch_size, hidden_dim), dtype=np.float32) * 0.1)
    w_rec = jnp.asarray(rng.standard_normal((hidden_dim, hidden_dim), dtype=np.float32) / np.sqrt(hidden_dim))

    # Warmup
    h_py, ys_py = python_loop_rollout(h0, v0, xs, w_rec)
    ys_py.block_until_ready()
    h_scan, ys_scan = lax_scan_rollout(h0, v0, xs, w_rec)
    ys_scan.block_until_ready()

    max_diff_final = float(np.max(np.abs(np.asarray(h_py) - np.asarray(h_scan))))
    max_diff_traj = float(np.max(np.abs(np.asarray(ys_py) - np.asarray(ys_scan))))

    # Median per call after the warmup above
    python_loop_ms = _median_ms(
        lambda: python_loop_rollout(h0, v0, xs, w_rec)[1].block_until_ready(), n_iters
    )
    scan_ms = _median_ms(lambda: lax_scan_rollout(h0, v0, xs, w_rec)[1].block_until_ready(), n_iters)

    scan_jaxpr = jax.make_jaxpr(lax_scan_rollout)(h0, v0, xs, w_rec)
    loop_traced_ops = len(jax.make_jaxpr(python_loop_rollout)(h0, v0, xs, w_rec).jaxpr.eqns)
    while_ops = lax_scan_rollout.lower(h0, v0, xs, w_rec).as_text().count("stablehlo.while")

    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "n_iters": n_iters,
        "python_loop_ms": round(python_loop_ms, 4),
        "lax_scan_ms": round(scan_ms, 4),
        "speedup": round(python_loop_ms / scan_ms, 2),
        "max_diff_final": max_diff_final,
        "max_diff_traj": max_diff_traj,
        "top_level_jaxpr_eqns": len(scan_jaxpr.jaxpr.eqns),
        "python_loop_traced_ops": loop_traced_ops,
        "stablehlo_while_ops": while_ops,
    }


def report(res: dict[str, Any]) -> str:
    """Human-readable report of one run (printed by main() and by verify_speedup.py --verbose)."""
    lines = [
        "=" * 72,
        "Experiment 04: 500-Step Trajectory Unrolling vs jax.lax.scan",
        "=" * 72,
        f"Sequence Length         : {res['seq_len']} steps, batch={res['batch_size']}, dim={res['hidden_dim']}",
        f"Top-Level Jaxpr Eqns    : {res['top_level_jaxpr_eqns']} (one jit call; scan sits inside it)",
        f"StableHLO while Loops   : {res['stablehlo_while_ops']} (the 500 steps stay inside the compiled program)",
        f"Python Loop Ops         : {res['python_loop_traced_ops']} ops dispatched from Python (500 jit steps + slicing)",
        f"Python Loop Median Time : {res['python_loop_ms']:.4f} ms",
        f"lax.scan Median Time    : {res['lax_scan_ms']:.4f} ms",
        f"Compiled Scan Speedup   : {res['speedup']:.2f}x",
        f"Max Trajectory Diff     : {res['max_diff_traj']:.2e}",
    ]
    return "\n".join(lines)


def main() -> None:
    print(report(run_benchmark()))


if __name__ == "__main__":
    main()
