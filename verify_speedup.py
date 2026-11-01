"""The Outside Judge: Deterministic Verification & Benchmark Runner.

Executes all 5 JAX vs PyTorch experiments, verifies numerical parity (`allclose`),
asserts >5x speedup on JIT operator fusion, `vmap(grad)`, and `lax.scan`,
asserts zero extra recompilations with static bucket padding, and writes
`benchmarks/latest_results.json`.
"""

from __future__ import annotations

import importlib
import json
import platform
from pathlib import Path
from typing import Any

import jax
import torch

exp01 = importlib.import_module("experiments.01_jit_operator_fusion")
exp02 = importlib.import_module("experiments.02_pure_functions_pytrees")
exp03 = importlib.import_module("experiments.03_vmap_batching")
exp04 = importlib.import_module("experiments.04_lax_scan_unrolling")
exp05 = importlib.import_module("experiments.05_dynamic_shapes_trap")


def run_all_verifications() -> dict[str, Any]:
    """Run all 5 experiments and enforce deterministic verification assertions."""
    r01 = exp01.run_benchmark()
    r02 = exp02.run_experiment()
    r03 = exp03.run_benchmark()
    r04 = exp04.run_benchmark()
    r05 = exp05.run_experiment()

    # Assertions for Experiment 01: JIT Operator Fusion
    assert r01["max_diff_torch_jit"] < 1e-5, f"Exp01 parity failed: {r01['max_diff_torch_jit']}"
    assert r01["speedup_vs_jax_eager"] >= 5.0, (
        f"Exp01 JIT speedup < 5.0x: {r01['speedup_vs_jax_eager']}x"
    )

    # Assertions for Experiment 02: Pure Functions & PyTrees
    assert r02["init_params_unchanged"] is True, "Exp02 mutated initial parameters in-place"
    assert r02["max_loss_diff"] < 1e-5, f"Exp02 loss parity failed: {r02['max_loss_diff']}"
    assert r02["max_w1_diff"] < 1e-5, f"Exp02 weight parity failed: {r02['max_w1_diff']}"

    # Assertions for Experiment 03: vmap(grad) Per-Example Gradients
    assert r03["max_diff_gw"] < 1e-5, f"Exp03 grad W parity failed: {r03['max_diff_gw']}"
    assert r03["max_diff_gb"] < 1e-5, f"Exp03 grad b parity failed: {r03['max_diff_gb']}"
    assert r03["speedup"] >= 5.0, f"Exp03 vmap speedup < 5.0x: {r03['speedup']}x"

    # Assertions for Experiment 04: lax.scan Recurrent Trajectory
    assert r04["max_diff_traj"] < 1e-5, f"Exp04 scan parity failed: {r04['max_diff_traj']}"
    assert r04["speedup"] >= 5.0, f"Exp04 scan speedup < 5.0x: {r04['speedup']}x"

    # Assertions for Experiment 05: Dynamic Shapes & Static Buckets
    assert r05["bucket_extra_recompilations"] == 0, (
        f"Exp05 had extra recompilations: {r05['bucket_extra_recompilations']}"
    )
    assert r05["max_output_diff"] < 1e-5, f"Exp05 masked output diff failed: {r05['max_output_diff']}"

    summary = {
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "jax_version": jax.__version__,
            "torch_version": torch.__version__,
            "jax_backend": jax.default_backend(),
            "jax_device": str(jax.devices()[0]),
        },
        "experiments": {
            "01_jit_operator_fusion": {
                k: v for k, v in r01.items() if k not in ("jaxpr_preview", "stablehlo_preview")
            },
            "02_pure_functions_pytrees": r02,
            "03_vmap_batching": r03,
            "04_lax_scan_unrolling": r04,
            "05_dynamic_shapes_trap": r05,
        },
        "all_verifications_passed": True,
    }
    return summary


def main() -> None:
    summary = run_all_verifications()
    bench_dir = Path(__file__).resolve().parent / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_json = bench_dir / "latest_results.json"
    out_txt = bench_dir / "terminal_output.txt"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    env = summary["environment"]
    e01 = summary["experiments"]["01_jit_operator_fusion"]
    e02 = summary["experiments"]["02_pure_functions_pytrees"]
    e03 = summary["experiments"]["03_vmap_batching"]
    e04 = summary["experiments"]["04_lax_scan_unrolling"]
    e05 = summary["experiments"]["05_dynamic_shapes_trap"]

    lines = [
        "=" * 78,
        "THE OUTSIDE JUDGE — JAX vs PyTorch Eager Verification Report",
        "=" * 78,
        (
            f"Runtime : Python {env['python_version']} | JAX {env['jax_version']} | "
            f"PyTorch {env['torch_version']} | Backend: {env['jax_backend']}"
        ),
        "-" * 78,
        f"{'Experiment':<34} | {'Baseline':<14} | {'JAX/XLA':<14} | {'Speedup / Check':<15}",
        "-" * 78,
        (
            f"{'01. @jit Operator Fusion':<34} | "
            f"{str(e01['jax_eager_ms']) + ' ms':<14} | "
            f"{str(e01['jax_jit_ms']) + ' ms':<14} | "
            f"{str(e01['speedup_vs_jax_eager']) + 'x (PASS)':<15}"
        ),
    ]
    delta_str = f"Δ={e02['max_w1_diff']:.1e} (PASS)"
    lines.extend(
        [
            (
                f"{'02. Pure PyTree vs nn.Module':<34} | "
                f"{'loss=' + str(e02['final_loss_torch']):<14} | "
                f"{'loss=' + str(e02['final_loss_jax']):<14} | "
                f"{delta_str:<15}"
            ),
            (
                f"{'03. vmap(grad) Per-Sample Clip':<34} | "
                f"{str(e03['loop_ms']) + ' ms':<14} | "
                f"{str(e03['vmap_ms']) + ' ms':<14} | "
                f"{str(e03['speedup']) + 'x (PASS)':<15}"
            ),
            (
                f"{'04. lax.scan 500-Step Rollout':<34} | "
                f"{str(e04['python_loop_ms']) + ' ms':<14} | "
                f"{str(e04['lax_scan_ms']) + ' ms':<14} | "
                f"{str(e04['speedup']) + 'x (PASS)':<15}"
            ),
            (
                f"{'05. Static Bucket Padding':<34} | "
                f"{str(e05['naive_trace_count']) + ' compiles':<14} | "
                f"{str(e05['bucket_extra_recompilations']) + ' extra comp':<14} | "
                f"{str(e05['speedup']) + 'x (PASS)':<15}"
            ),
            "-" * 78,
            f"Saved JSON report to: {out_json.relative_to(Path(__file__).resolve().parent)}",
            "ALL DETERMINISTIC CHECKS PASSED.",
        ]
    )
    report = "\n".join(lines)
    out_txt.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
