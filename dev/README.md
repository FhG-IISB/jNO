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

## Kernel counters (L2): `dev/kernels.py`
`python dev/kernels.py dev/runs/<label> --stage jno/solve [--top N] [--launches N] -- python the_script.py ARGS`
re-runs the script under Nsight Compute with an NVTX filter on that stage's XLA ranges (CUDA graphs and autotuning
off for the child, `JNO_PROF_NCU=1` so the harness skips its trace), refuses above `--budget` seconds of estimated
replay, and writes `kernels_<stage>.json` (per launch: time, FP64/FP32 flops, DRAM/L2/L1 bytes, hit rates, SM and
DRAM utilisation, occupancy, registers, stalls). The ladder then shows the stage's kernel bound (the roof at each
kernel's measured intensity) and `bind_strip` the fraction-of-peak cells per kernel kind. `--reparse` re-reads an
existing `.ncu-rep`. Development problem: `dev/examples/small3d.py` (3.5 k tets, every stage, ~15 s per cycle).

## Sizes, intrinsic bounds, the fan: `sizes=`, `dev/fan.py`
`prof.run(..., sizes=dict(cells=, nodes=, nnz=, n=, nrhs=, complex=))` lets `Result.intrinsic_bounds()` give the
minimal-work time per stage (assembly: the mesh and the triplets once; factor: nnz(L)^2/n flops; solve: two sweeps
over nnz(L) per right-hand side): the dashed rung of the ladder, the `x intrinsic` column of the table.
`python dev/fan.py RUN1 RUN2 ... --target N` fits per-stage time and memory against `sizes[cells]` (or `--x n`)
over several runs and extrapolates to a target size against the card's 8 GB; it says when the runs are
launch-dominated (then the exponents are below their asymptotic values).

## CPU counters per stage: `dev/cpu.py`
`JAX_PLATFORMS=cpu python dev/cpu.py RUN -- python the_script.py ARGS` re-runs the script pinned to the P-cores with
`perf record` around the steady-state calls; the harness first writes `/tmp/perf-<pid>.map` from XLA:CPU's gdb JIT
descriptor (`cpu.perf_map`), so every compiled kernel is symbolised by its HLO op and joined to its stage. Reports
the share of cycles, instructions, cache misses and FP64 scalar / 128-bit / 256-bit instructions per stage (the
vector share tells whether a stage is vectorised); library kernels (MKL, Eigen, oneDNN) are "library". A run on the
CPU: `JAX_PLATFORMS=cpu python dev/examples/small3d.py --cpu`. `--reparse` re-reads the run's `perf.data`.
