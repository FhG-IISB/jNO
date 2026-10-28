# Training

This section covers every aspect of the jNO training pipeline: constructing the core solver, attaching optimisers, schedules, multi-phase training, parallelism, evaluation, and per-model controls.

---

## Core Solver

`jno.core` is the central training object. It:

1. Builds the symbolic computation graph from your constraints.
2. Performs common sub-expression elimination (CSE).
3. Folds a sum of squared partials into one Laplacian node, so `u.xx + u.yy`
   and the shorthand `u.laplacian(x, y)` cost the same
   ([details](../operations.md#differentiation)).
4. Initialises all neural-network parameters.
5. Compiles a JIT-optimised step function.
6. Runs the training loop and returns training statistics.

```python
crux = jno.core(
    constraints=[pde.mse, 10.0 * boc.mse],   # weight a term by multiplying it
    mesh=(1, 1),                       # (batch_devices, model_devices); (1, 1) = every device, data-parallel
)
```

The random seed is not a constructor argument: set `JNO_SEED` in the environment or `seed` under
`[jno]` in `.jno.toml` (default `42`). Adaptive loss weights are covered in
[Adaptive Loss Weights](../adaptive/schedules.md#adaptive-loss-weights).

---

## Attaching Optimisers

**Every non-frozen model must have an optimiser before calling `solve()`.**

```python
u_net.optimizer(optax.adam).scale(lrs.exponential(1e-3, 0.9, 2000, 1e-5))
v_net.optimizer(optax.adamw).scale(lrs.warmup_cosine(5000, 500, 1e-3, 1e-4))
```

`model.optimizer()` returns `self` for chaining:

```python
u_net = jno.nn(foundax.mlp(2, key=key)).optimizer(optax.adam(1e-3))
```

### After `core.load()`

When loading a saved solver the `Model` references in the expression tree are disconnected from Python variables. Use `set_optimizer` to reassign:

```python
crux = jno.load("runs/crux.pkl")
crux.set_optimizer(optax.adam, scale=lrs(1e-4))
crux.solve(1000)
```

---

## Constraints

`jno.le(g, b)` / `jno.ge(g, b)` mark an entry of the `jno.core` list as a constraint `g <= b`
(`g >= b`) instead of a loss. There are two ways to enforce it:

- **A constrained optimiser** on the design variables: `jno.optimizers.mma(...)` or
  `jno.optimizers.gcmma(...)`. They take every constraint row into their own subproblem.
  `gcmma` keeps each step feasible.
- **An augmented Lagrangian** on the constraint itself, with any optimiser (optax,
  `ssbfgs`, `soap`, ...):

```python
vol_c = jno.le(volume, 0.4).auglag(mu=10.0, every=100)
crux = jno.core([loss, vol_c, (stress <= s_max).auglag(), jno.eq(mass, m0).auglag()])
crux.solve(5000)
lam = crux.eval([vol_c.multiplier])                          # the Lagrange multiplier
```

(`jno.auglag(c, ...)` is the same as `c.auglag(...)`.) `.auglag()` adds the penalty `(max(0, λ + μc)² − λ²)/(2μ)` to the loss, with `c = g − b`.
Every `every` steps it sets `λ ← max(0, λ + μc)`, using `c` averaged over the second half of
the window, and raises `μ` tenfold whenever the violation is positive and has not dropped to
a quarter (Nocedal & Wright 2006, §17.4). The constraint is still reported as its own row, so
the log, `early_stopping` and callbacks see `c`.

An equality `jno.eq(g, b)` (`g == b` cannot be written, because `==` on jNO expressions tests
whether two nodes are the same) is enforced only through `.auglag()`. Its penalty is
`λc + μc²/2` and its multiplier takes either sign (§17.3).

The multiplier of every constraint row, from `.auglag()` or from MMA's subproblem, is printed
next to the row as `λ[name]` and kept in `crux.training_logs[-1]["multipliers"]` (one column per
constraint row). It is the change in the loss per unit of slack in the bound, and `0` for an
inequality that is not active.

**Box bounds** on a parameter go with the optimiser: `mma`/`gcmma` take `lower=`/`upper=`, and
any optax optimiser gets them through `jno.optimizers.bounded`, which projects each step back
onto the box:

```python
rho.optimizer(jno.optimizers.bounded(optax.adam(1e-2), lower=0.0, upper=1.0))
```

!!! warning "Feasible only in the limit"
    Between multiplier updates the design may violate the constraint. If every iterate has to
    stay feasible, for example when an infeasible design breaks the next solve, use `gcmma`.
    `every` should be long enough for the optimiser to settle on the current penalty: tens of
    steps for a quasi-Newton optimiser, hundreds for Adam. Adam at a fixed rate keeps jittering
    by about its rate and, once its gradients fall below its `eps`, takes steps of that size
    again; over long runs this throws the multiplier off, so give it a decaying rate
    (`optax.exponential_decay`). An `.auglag()` constraint cannot be combined with
    `mma`/`gcmma` in the same core, and a strict `<` or `>` is refused.

---

## Per-Model Controls

Each model is fully independent with respect to its optimiser, trainability, LoRA configuration, and pretrained weight initialisation. See the **[Operations → Part B](../operations.md#part-b-operations-that-require-trainable-parameters)** sub-section for the full API covering freeze, masks, LoRA, dtype conversion, and diagnostics.
