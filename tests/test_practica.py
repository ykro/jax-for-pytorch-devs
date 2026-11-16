"""The snippets in practica/ are the ones typed in the practice video: each must run and print
what the video shows (the two *_roto.py files must fail with the error the video explains)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PRACTICA = Path(__file__).resolve().parent.parent / "practica"


def run(name: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(PRACTICA / name)]
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)


def test_01_mutar_roto_raises_type_error() -> None:
    r = run("01_mutar_roto.py")
    assert r.returncode != 0
    assert "TypeError: JAX arrays are immutable" in r.stderr


def test_01_mutar_keeps_original() -> None:
    lines = run("01_mutar.py").stdout.splitlines()
    assert lines == ["pesos:  [0. 0. 0. 0.]", "nuevos: [0.  0.5 0.  0. ]"]


def test_01_paso_loss_goes_down() -> None:
    lines = run("01_paso.py").stdout.splitlines()
    assert lines[0] == "paso 0 loss 13.7775"
    losses = [float(line.split("loss")[1]) for line in lines]
    assert len(losses) == 5
    assert losses[0] > 10.0
    assert losses[-1] < 1e-3


def test_02_trazar_traces_once_per_shape() -> None:
    lines = run("02_trazar.py").stdout.splitlines()
    assert lines == [
        "trazando con (64, 128) float32",
        "tres llamadas listas",
        "trazando con (32, 128) float32",
    ]


def test_02_if_roto_raises_tracer_bool_error() -> None:
    r = run("02_if_roto.py")
    assert r.returncode != 0
    assert "TracerBoolConversionError" in r.stderr


def test_02_if_uses_where() -> None:
    lines = run("02_if.py").stdout.splitlines()
    assert lines == ["[ 1. -2.  3.]", "[ 1.  2. -3.]"]


def test_03_vmap_grad_one_gradient_per_example() -> None:
    out = run("03_vmap_grad.py").stdout
    assert "gradiente del batch: (8,)" in out
    assert "uno por ejemplo:     (128, 8)" in out
    assert "su promedio coincide: True" in out
    assert "promedio recortado: (8,)" in out


def test_04_scan_one_equation() -> None:
    lines = run("04_scan.py").stdout.splitlines()
    assert lines == ["iguales: True", "ecuaciones con for:  2500", "ecuaciones con scan: 1"]


def test_04_formas_compiles_once_per_shape() -> None:
    assert run("04_formas.py").stdout.strip() == "llamadas: 7  compilaciones: 6"


def test_04_buckets_compile_once_per_bucket() -> None:
    lines = run("04_buckets.py").stdout.splitlines()
    assert lines[0] == "llamadas: 7  compilaciones: 2"
    con = float(lines[1].removeprefix("con máscara:"))
    sin_relleno = float(lines[3].removeprefix("sin relleno:"))
    assert abs(con - sin_relleno) < 1e-4


def test_04_buckets_mask_matters() -> None:
    """cos(0) = 1, so without the mask the 27 padded zeros of the 64 bucket add 27 to the sum."""
    lines = run("04_buckets.py").stdout.splitlines()
    con = float(lines[1].removeprefix("con máscara:"))
    sin = float(lines[2].removeprefix("sin máscara:"))
    assert abs((sin - con) - 27.0) < 1e-3
