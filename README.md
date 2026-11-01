# JAX & OpenXLA para Desarrolladores PyTorch

- **Por qué es relevante:** Pasar del dispatch imperativo operación por operación a la compilación de grafos completos en OpenXLA permite fusionar operadores en memoria caché del acelerador, vectorizar gradientes por ejemplo sin loops en Python y ejecutar trayectorias secuenciales en un único kernel compilado.
- **Para quién es:** Ingenieros de Machine Learning y desarrolladores que dominan PyTorch (`nn.Module`, `loss.backward()`, `optimizer.step()`) y buscan dominar el modelo mental funcional de JAX (`@jax.jit`, PyTrees, `vmap`, `lax.scan`).
- **Quién debería verlo:** Equipos técnicos que entrenan o sirven modelos en GPUs y Cloud TPUs y necesitan eliminar cuellos de botella de lanzamiento de kernels y recompilaciones por formas dinámicas.
- **Serie completa:** Parte de la guía práctica [JAX & OpenXLA Field Guide](https://github.com/ykro/jax-openxla-field-guide).

---

## Arquitectura y Modelo Mental

### 1. Eager Dispatch vs Fusión de Operadores con `@jax.jit`

En modo eager, cada operación aritmética lanza un kernel independiente y materializa buffers intermedios en la memoria del dispositivo (HBM). Al decorar con `@jax.jit`, JAX traza la función con `ShapedArray`, genera un `Jaxpr`, lo baja a `StableHLO` y el compilador OpenXLA fusiona las etapas elementwise y de reducción en un solo ejecutable PJRT.

![Eager Dispatch vs Fusión XLA con @jax.jit](diagrams/01-eager-vs-jit.svg)

### 2. Ciclo de Vida de `@jax.jit`: Trazado vs Caché PJRT

La primera invocación con una firma `(shape, dtype)` nueva ejecuta el código Python para construir el grafo `Jaxpr` y compilar el binario XLA. Las llamadas posteriores con la misma forma geométrica omiten Python y envían directamente al binario cacheado en PJRT.

![Ciclo de vida de @jax.jit](diagrams/02-jit-lifecycle.svg)

### 3. Vectorización con `vmap` y Loops Compilados con `lax.scan`

En lugar de iterar en Python sobre ejemplos de un batch o sobre pasos temporales, `jax.vmap` empuja el batch axis hacia las primitivas XLA y `jax.lax.scan` compila recurrencias largas en una única instrucción `WhileOp` que mantiene el estado (`carry`) en registros del acelerador.

![Flujo de datos de vmap y lax.scan](diagrams/03-vmap-scan-dataflow.svg)

---

## Resultados Reales de Verificación (`verify_speedup.py`)

Benchmarks deterministas ejecutados con Python `3.12.11`, JAX `0.12.0.1` y PyTorch `2.15.0`:

| Experimento | Archivo | Baseline (Eager / Loop) | JAX + OpenXLA | Speedup / Paridad Numérica |
| :--- | :--- | :--- | :--- | :--- |
| **01. `@jax.jit` Operator Fusion** | `experiments/01_jit_operator_fusion.py` | `1.4055 ms` (JAX Eager) / `0.8959 ms` (PyTorch) | `0.1589 ms` (`@jax.jit`) | **8.85x** (`max_diff = 3.34e-06`) |
| **02. Pure Functions & PyTrees** | `experiments/02_pure_functions_pytrees.py` | `loss = 0.845957` (`nn.Module` + SGD) | `loss = 0.845957` (`NamedTuple` PyTree) | **Paridad exacta** (`ΔW1 = 1.49e-08`) |
| **03. `vmap(grad)` Per-Sample Clip** | `experiments/03_vmap_batching.py` | `24.6432 ms` (loop Python, `B=128`) | `0.1776 ms` (`vmap(grad)`) | **138.78x** (`max_diff = 1.40e-09`) |
| **04. `lax.scan` 500-Step Rollout** | `experiments/04_lax_scan_unrolling.py` | `34.5272 ms` (loop Python, `T=500`) | `2.8744 ms` (`lax.scan`) | **12.01x** (`max_diff = 0.00e+00`) |
| **05. Static Bucket Padding** | `experiments/05_dynamic_shapes_trap.py` | `24 compilaciones` (`1061.29 ms`) | `0 recompilaciones extra` (`19.74 ms`) | **53.76x** (`max_diff = 8.94e-08`) |

Salida íntegra de terminal generada por `verify_speedup.py`:

```text
==============================================================================
THE OUTSIDE JUDGE — JAX vs PyTorch Eager Verification Report
==============================================================================
Runtime : Python 3.12.11 | JAX 0.12.0.1 | PyTorch 2.15.0 | Backend: cpu
------------------------------------------------------------------------------
Experiment                         | Baseline       | JAX/XLA        | Speedup / Check
------------------------------------------------------------------------------
01. @jit Operator Fusion           | 1.4055 ms      | 0.1589 ms      | 8.85x (PASS)   
02. Pure PyTree vs nn.Module       | loss=0.845957  | loss=0.845957  | Δ=1.5e-08 (PASS)
03. vmap(grad) Per-Sample Clip     | 24.6432 ms     | 0.1776 ms      | 138.78x (PASS) 
04. lax.scan 500-Step Rollout      | 34.5272 ms     | 2.8744 ms      | 12.01x (PASS)  
05. Static Bucket Padding          | 24 compiles    | 0 extra comp   | 53.76x (PASS)  
------------------------------------------------------------------------------
Saved JSON report to: benchmarks/latest_results.json
ALL DETERMINISTIC CHECKS PASSED.
```

---

## Estructura del Repositorio

```text
.
├── benchmarks/
│   ├── latest_results.json         # Métricas verificadas por verify_speedup.py
│   └── terminal_output.txt         # Salida completa de los 5 experimentos y el juez
├── diagrams/
│   ├── 01-eager-vs-jit.{html,svg,png}
│   ├── 02-jit-lifecycle.{html,svg,png}
│   └── 03-vmap-scan-dataflow.{html,svg,png}
├── experiments/
│   ├── 01_jit_operator_fusion.py   # Jaxpr, StableHLO y fusión XLA vs Eager
│   ├── 02_pure_functions_pytrees.py# Estado inmutable PyTree vs PyTorch nn.Module
│   ├── 03_vmap_batching.py         # Gradientes por ejemplo con jax.vmap(jax.grad)
│   ├── 04_lax_scan_unrolling.py    # 500 pasos recurrentes con jax.lax.scan
│   └── 05_dynamic_shapes_trap.py   # Trampa de formas dinámicas y static bucketing
├── gcp/
│   └── run_gpu_tpu_benchmark.sh    # Runner on-demand con auto-teardown en Cloud TPU v5e y Vertex AI L4 GPU
├── tests/
│   └── test_experiments.py         # Suite pytest determinista (5 tests)
├── Makefile
├── pyproject.toml
└── verify_speedup.py               # El Juez Externo (aserciones de paridad y speedup)
```

---

## Onboarding y Cómo Ejecutar Localmente o en Google Cloud

### 1. Instalación desde cero (macOS / Linux)

```bash
git clone https://github.com/ykro/jax-for-pytorch-devs.git
cd jax-for-pytorch-devs
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Recorrido guiado paso a paso (Orden de Onboarding)

Ejecuta cada script en orden para observar la transición desde PyTorch Eager hasta OpenXLA compilado.

```bash
# Paso 1 — Fusión de operadores con @jax.jit e inspección de Jaxpr y StableHLO
python experiments/01_jit_operator_fusion.py

# Paso 2 — Funciones puras y PyTrees inmutables frente a torch.nn.Module
python experiments/02_pure_functions_pytrees.py

# Paso 3 — Vectorización de gradientes per-sample con jax.vmap(jax.grad)
python experiments/03_vmap_batching.py

# Paso 4 — Loops recurrentes compilados en una sola primitiva con jax.lax.scan
python experiments/04_lax_scan_unrolling.py

# Paso 5 — Trampa de formas dinámicas y solución con static bucket padding
python experiments/05_dynamic_shapes_trap.py
```

### 3. Verificación completa (Linter + Tests + Juez Externo)

```bash
# Ejecutar únicamente el Juez Externo y generar benchmarks/latest_results.json
python verify_speedup.py

# Ejecutar ruff, pytest (5 tests) y verify_speedup.py en un solo comando
make all
```

### 4. Ejecución on-demand en Google Cloud (Cloud TPU v5e o NVIDIA L4 GPU)

El script `gcp/run_gpu_tpu_benchmark.sh` aprovisiona una Cloud TPU `v5litepod-1` (o un Custom Job en Vertex AI con GPU NVIDIA L4) en el proyecto `ai-experiments-487722` y garantiza la destrucción automática de la instancia mediante `trap cleanup EXIT`.

```bash
# Ejecutar benchmark en Cloud TPU v5e-1 on-demand (us-central1-a)
make gcp-tpu

# Ejecutar benchmark en Vertex AI Custom Job con GPU NVIDIA L4
make gcp-gpu
```
