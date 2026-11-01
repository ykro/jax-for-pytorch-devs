PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
PYTEST ?= $(shell if [ -x ../../.venv-shared/bin/pytest ]; then echo ../../.venv-shared/bin/pytest; else command -v pytest; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)

.PHONY: all experiments verify test lint clean gcp-gpu gcp-tpu

all: lint test verify

experiments:
	$(PYTHON) experiments/01_jit_operator_fusion.py
	$(PYTHON) experiments/02_pure_functions_pytrees.py
	$(PYTHON) experiments/03_vmap_batching.py
	$(PYTHON) experiments/04_lax_scan_unrolling.py
	$(PYTHON) experiments/05_dynamic_shapes_trap.py

verify:
	$(PYTHON) verify_speedup.py

test:
	$(PYTEST) -v

lint:
	$(RUFF) check .

gcp-gpu:
	bash gcp/run_gpu_tpu_benchmark.sh gpu

gcp-tpu:
	bash gcp/run_gpu_tpu_benchmark.sh tpu

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
