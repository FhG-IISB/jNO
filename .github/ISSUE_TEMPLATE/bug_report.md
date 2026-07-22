---
name: Bug report
about: Report unexpected behaviour in jNO
title: ''
labels: bug
assignees: ''

---

## What happened

<!-- A short description of the unexpected behaviour. -->

## Minimal reproducer

<!--
The shortest script that reproduces the issue. A self-contained snippet
that someone can paste into a file and run is much more useful than
prose. Trim unrelated setup; keep imports.
-->

```python
import jno
# ...
```

## Expected vs actual

<!--
What did you expect to happen, and what did you observe instead? Paste
the full traceback (inside a ```python or ```text fence) if one exists.
-->

## Environment

- `jno.__version__`:
- `jax.__version__`:
- Python version (`python --version`):
- Operating system (e.g. Ubuntu 24.04, macOS 14.5):
- GPU / CUDA version (or "CPU only"):
- Install method (`pip install jax-numerical-operators[fem]`, `pixi install`, Docker image tag, source checkout):

## Additional context

<!--
Anything else relevant — does the bug only appear on GPU? Only with
batch size > N? Only after a particular `crux.solve(...)` configuration?
Links to related issues, papers, or upstream blackjax/optax/equinox
bugs.
-->
