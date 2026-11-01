"""Automated pytest suite verifying all 5 JAX vs PyTorch experiments."""

from __future__ import annotations

import importlib

exp01 = importlib.import_module("experiments.01_jit_operator_fusion")
exp02 = importlib.import_module("experiments.02_pure_functions_pytrees")
exp03 = importlib.import_module("experiments.03_vmap_batching")
exp04 = importlib.import_module("experiments.04_lax_scan_unrolling")
exp05 = importlib.import_module("experiments.05_dynamic_shapes_trap")


def test_01_jit_operator_fusion() -> None:
    res = exp01.run_benchmark(batch_size=64, hidden_dim=128, n_iters=100)
    assert res["max_diff_torch_jit"] < 1e-5
    assert res["max_diff_eager_jit"] < 1e-5
    assert res["speedup_vs_jax_eager"] >= 5.0
    assert res["jaxpr_num_equations"] > 20
    assert res["stablehlo_lines"] > 10


def test_02_pure_functions_pytrees() -> None:
    res = exp02.run_experiment(n_steps=10)
    assert res["init_params_unchanged"] is True
    assert res["pytree_num_leaves"] == 9
    assert res["max_loss_diff"] < 1e-5
    assert res["max_w1_diff"] < 1e-5


def test_03_vmap_batching() -> None:
    res = exp03.run_benchmark(batch_size=128, in_dim=64, out_dim=16, n_iters=15)
    assert res["max_diff_gw"] < 1e-5
    assert res["max_diff_gb"] < 1e-5
    assert res["speedup"] >= 5.0


def test_04_lax_scan_unrolling() -> None:
    res = exp04.run_benchmark(seq_len=500, batch_size=32, hidden_dim=64, n_iters=10)
    assert res["max_diff_final"] < 1e-5
    assert res["max_diff_traj"] < 1e-5
    assert res["speedup"] >= 5.0
    assert res["top_level_jaxpr_eqns"] == 1


def test_05_dynamic_shapes_trap() -> None:
    res = exp05.run_experiment()
    assert res["naive_trace_count"] == 24
    assert res["bucket_warmup_traces"] == 3
    assert res["bucket_extra_recompilations"] == 0
    assert res["max_output_diff"] < 1e-5
    assert res["speedup"] >= 2.0
