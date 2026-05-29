---
name: run-jno
description: run, smoke-test, verify, build, or exercise the jNO (jax-neural-operators) library; confirm a change works in the live library; run the test suite
---

jNO is a Python library — there is no server or GUI. The primary agent path is running the smoke driver (`.claude/skills/run-jno/smoke.py`) which exercises the full pipeline: import → domain → network → core → solve → eval → save/load. For correctness changes, the unit tests are the next layer.

## Prerequisites

`pixi` must be installed and the workspace environment must exist:

```bash
pixi install   # one-time; installs jno and all deps into the pixi env
```

No GPU is required — JAX falls back to CPU automatically. With a CUDA GPU present, JAX uses it by default.

## Run: smoke driver (agent path)

Exercises imports, domain construction, model controls, a 10-epoch solve, eval, and save/load round-trip. Completes in under 60 s (most time is JAX JIT compilation):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pixi run python .claude/skills/run-jno/smoke.py
```

Expected output ends with:

```
[6/6] eval + save/load OK

✓ smoke test passed
```

Exit code 0 = pass. The driver lives at `.claude/skills/run-jno/smoke.py`.

## Run: unit tests (non-slow)

Runs the full unit test suite minus the slow integration tests (~3.5 min):

```bash
pixi run pytest tests/ -x --tb=short -q -m "not slow" --ignore=tests/tutorial_examples_tests
```

Typical result: `1025 passed, ~9 skipped`.

## Run: single test file

```bash
pixi run pytest tests/test_lora.py -x --tb=short -q
pixi run pytest tests/test_derivatives.py -x --tb=short -q
```

## Run: tutorial smoke tests (slow — need GPU)

These run each tutorial script as a subprocess. They are marked `slow` and skipped by default:

```bash
JNO_TUTORIAL_SMOKE_TIMEOUT=360 pixi run pytest tests/test_tutorial_examples_smoke.py -x --tb=short -q
```

## Direct invocation of a tutorial example

Any `docs/tutorial_examples/**/*.py` can be run standalone:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pixi run python docs/tutorial_examples/01_basics/poisson_1d.py
```

## Dev workflow

```bash
pixi run fmt    # ruff format
pixi run lint   # ruff check --fix
pixi run test   # pytest -x --tb=short (non-slow)
pixi run ci-fmt && pixi run ci-lint && pixi run ci-test   # CI-equivalent (read-only)
```

## Gotchas

- **`statistics` object has no `.total_loss`**: the attribute is `history.training_logs[-1]["total_loss"]` — a list of dicts, not a flat array. This bites code that follows the (stale) docs description.
- **`net.freeze()` then `.reset()`**: `reset()` clears all controls including the optimizer; always call `.optimizer(...)` again after reset if you intend to train.
- **JAX GPU pre-allocation**: when running subprocesses or multiple pytest workers alongside a JAX-importing parent, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` or contexts fight over VRAM and crash. The conftest.py sets this for pytest; set it manually for standalone scripts.
- **JIT compile time**: the first `solve()` call spends 10–30 s compiling. For smoke purposes 10 epochs is enough — the wall time is dominated by JIT, not actual training.
- **`pixi run` vs bare `python`**: the library is installed in the pixi env. Running `python` outside `pixi run` will hit an unresolved `jno` import unless the env is activated manually.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: jno` | Run with `pixi run python`, not bare `python` |
| CUDA OOM on first `core()` | Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| Subprocess tests hang | Set `JNO_TUTORIAL_SMOKE_TIMEOUT` to a higher value |
| `AttributeError: 'statistics' object has no attribute 'total_loss'` | Use `history.training_logs[-1]["total_loss"][-1]` instead |
