# dev/: developer profiling harness (not shipped)

`prof.run(jitted_fn, *args, label=...)` profiles one call of a jitted jNO program and writes
`dev/runs/<label>/`: `run.json`, `summary.txt`, `compiled.hlo`, the profiler trace, and the figures
`headroom_ladder`, `icicle`, `launch_spectrum` (`.pdf` + `.png`). `prof.reparse(run_dir)` rebuilds a run from its
trace after a change to the parser or the figures.

Stages are `jax.named_scope("jno/<stage>")`: jNO marks `assemble` (FemLinearSystem.evaluate), `factor` and `solve`
(cudss_sym, the host callbacks of linear.py); a script adds its own (`with jax.named_scope("jno/readout"): ...`).
The transpose of a scope is attributed to the same stage as its adjoint. Kernels launched from CUDA graphs
(`command_buffer_N`) are joined to their scope through the kernel name. Solver backends report through
`jno.utils.profiling.record(...)` when `collect_stats(True)` is on.

Example: `dev/examples/toy.py`; the planar pilot lives in the transformers project (`planar/prof_grad.py`).
Run with `XLA_PYTHON_CLIENT_PREALLOCATE=false` when the card is shared. Plan: `plans/dev-profiling.md`.
