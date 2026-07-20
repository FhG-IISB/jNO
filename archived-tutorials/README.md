# Archived tutorials

Tutorials kept for reference but **not** part of the published documentation site
or the tutorial test suite. They live outside `docs/`, so:

- they are not built by MkDocs and do not appear in the site navigation, and
- they are not collected by `tests/test_tutorial_examples.py` (which only walks
  `docs/tutorial_examples/`).

The underlying jNO capabilities each one demonstrates remain fully supported and
unit-tested in the main package; only the worked example has been retired from the
front-page tutorial set.

| Example | Demonstrates | Library feature (still live) |
| --- | --- | --- |
| `neural_galerkin_heat_1d` | Neural Galerkin — evolve network weights via parameter-Jacobian projection | existing `jno.fem` / trace API |
| `nn_enriched_fem_2d` | NN-enriched FEM — Fourier-feature prior + coarse-mesh correction | `∂(frozen network coefficient)/∂x` assembled into the RHS |
