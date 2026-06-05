# Bayesian sampling — `.bayesian(...)` per parameter

`model.bayesian(kernel_factory, **kw)` replaces a model's per-step gradient
update with one transition of a [blackjax](https://blackjax-devs.github.io/blackjax)
MCMC kernel.  The configurator mirrors `.optimizer(...)` — each parameter
in your script can independently choose to be **optimised** (point estimate
via optax) or **sampled** (posterior chain via blackjax).  Mix freely;
`crux.solve()` dispatches per-model.

After training the chain is available on the model as
`model.posterior_samples`, and on the `crux` itself via
`crux.eval([...], samples="chain")` which vmaps the evaluator over the
chain so nonlinear predictions push forward correctly.

## Quick example — Bayesian inverse problem

Recover `(A, B)` in `d(x) = A·sin(πx) + B·cos(πx)` from noisy
observations, with credible intervals:

```python
import blackjax, jax, jno
import jax.numpy as jnp

π = jno.np.pi
dom = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
x, _ = dom.variable("interior")

A_true, B_true = 3.14, -2.71
target = A_true * jno.np.sin(π * x) + B_true * jno.np.cos(π * x) \
       + jno.noise.gaussian(std=0.1)

k1, k2 = jax.random.split(jax.random.PRNGKey(0), 2)
a = jno.np.parameter((1,), key=k1, name="a")
b = jno.np.parameter((1,), key=k2, name="b")

for p in [a, b]:
    # inverse_mass_matrix defaults to identity of the right shape;
    # adapt=True (default) tunes step_size and IMM via blackjax.window_adaptation.
    p.bayesian(blackjax.nuts, step_size=1e-2, warmup=500, keep=1000)

residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - target
crux = jno.core([residual.mse], dom)
crux.solve(1500)

# Raw chains — leading axis = sample
a_chain = a.posterior_samples           # (1000, 1)
A_mean = jnp.mean(a_chain, axis=0)
A_lo, A_hi = jnp.quantile(a_chain, jnp.array([0.05, 0.95]), axis=0)
print(f"A = {A_mean[0]:.3f}  [{A_lo[0]:.3f}, {A_hi[0]:.3f}]")
```

## API

```python
Model.bayesian(
    kernel_factory,             # e.g. blackjax.nuts, blackjax.sgld
    *,
    prior=None,                 # callable: pytree -> log p(θ);  default Gaussian(σ=10)
    warmup=500,                 # outer epochs to discard before collecting
    keep=1000,                  # number of post-warmup samples to retain
    thin=1,                     # keep one sample every `thin` post-warmup steps
    **kernel_kwargs,            # forwarded to kernel_factory; `step_size=` is required
                                # EXCEPT for HMC-family kernels with adapt=True + warmup>0
                                # (window adaptation chooses one — defaults to 1.0).
)
```

`kernel_factory` is duck-typed at solve time:

| First parameter of factory   | Family               | Examples                          |
|------------------------------|----------------------|-----------------------------------|
| `logdensity_fn`              | Full-data MCMC       | `blackjax.nuts`, `blackjax.hmc`, `blackjax.mala` |
| `grad_estimator`             | Stochastic-gradient  | `blackjax.sgld`, `blackjax.sghmc` |

jno builds the appropriate closure from the live loss + context and rebuilds
the kernel inside the JIT graph each step.

## Output — chains by default

`crux.eval(...)` is **auto-chain-aware** per expression: if an expression's
dependency graph touches a model with ``posterior_samples`` set, the
evaluator is `vmap`-ped over that chain.  Otherwise the expression is
evaluated at the point value as before.  No `samples=` argument is needed
for the common case.

| Read                              | `.optimizer()` (point)   | `.bayesian()` (chain)                          |
|-----------------------------------|--------------------------|------------------------------------------------|
| `crux.eval([m])`                  | point value              | `(n_kept, *m_shape)` chain                     |
| `crux.eval([expr])`               | `(n_points, …)` point    | `(n_kept, n_points, …)` chain (auto)           |
| `m.posterior_samples`             | `None`                   | stacked module pytree (or array for `parameter`) |

No `.mean() / .std() / .quantile()` helpers are provided — compute whatever
summary you need from the chain with `jnp.mean`, `jnp.quantile`, arviz, or
your favourite plotting library.

### Nonlinear pushforward — handled automatically

For predictions through a neural network, the posterior mean over outputs
is **not** the output at the posterior mean of the weights.  This used to
require an explicit `samples="chain"`; the default now does the right
thing:

```python
u_chain = crux.eval([u])                          # (n_kept, n_points, 1)
u_mean = jnp.mean(u_chain, axis=0)
u_lo, u_hi = jnp.quantile(u_chain, jnp.array([0.05, 0.95]), axis=0)
```

### Escape hatches

```python
crux.eval([u], samples="chain")    # force chain (raises if no Bayesian deps)
crux.eval([u], samples="point")    # force point: evaluate at last sample,
                                   # skips the vmap. Quick debugging / sanity.
```

The `samples="point"` mode returns a single sample at the model's current
position — useful when you just want a quick number, but **not** a
substitute for the posterior summary on nonlinear outputs.

## Mixed: optimised + sampled

Different parameters use different update rules; they coexist in one
`crux.solve()`:

```python
encoder = jno.nn.wrap(foundax.mlp(2, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(0)))
head    = jno.nn.wrap(foundax.mlp(1, hidden_dims=32, num_layers=1, key=jax.random.PRNGKey(1)))

encoder.optimizer(optax.adam(1e-3))                          # point estimate
head.bayesian(blackjax.sgld, step_size=1e-5)                 # SGLD chain
```

Only `head.posterior_samples` is populated; `encoder.posterior_samples` is
`None`.

## Bayesian PINN — predictive bands

```python
import blackjax, foundax, jax, jno
import jax.numpy as jnp

π = jno.np.pi
dom = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
x, _  = dom.variable("interior")
xb, _ = dom.variable("boundary")

net = jno.nn.wrap(foundax.mlp(1, hidden_dims=32, num_layers=3,
                              key=jax.random.PRNGKey(0)))
net.bayesian(blackjax.sgld, step_size=1e-5, warmup=2000, keep=1000)

u    = net(x)
u_xx = jno.diff(u, x, order=2)
pde  = u_xx + (π ** 2) * jno.np.sin(π * x)        # u'' = -π² sin(πx)
bc   = net(xb) - 0.0

crux = jno.core([pde.mse, bc.mse], dom)
crux.solve(3000)

u_chain = crux.eval([u], samples="chain")
u_mean     = jnp.mean(u_chain, axis=0)
u_lo, u_hi = jnp.quantile(u_chain, jnp.array([0.05, 0.95]), axis=0)
```

## Custom prior

`prior=` takes any `pytree → float` returning the log-prior density.

```python
def laplace_prior(p, scale=1.0):
    return -sum(
        jnp.sum(jnp.abs(leaf))
        for leaf in jax.tree_util.tree_leaves(p)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    ) / scale

a.bayesian(blackjax.nuts, step_size=1e-2, prior=laplace_prior)
```

The default prior is a wide isotropic Gaussian with σ=10 over every
inexact-array leaf — effectively flat at typical parameter scales.

## Adaptation (NUTS / HMC)

For HMC-family kernels, `adapt=True` (default) runs
`blackjax.window_adaptation` for the first `warmup` steps **before** the
main solve loop.  Step size and inverse mass matrix are tuned
automatically; the loop then collects `keep` samples from epoch 0.

```python
a.bayesian(blackjax.nuts, step_size=1.0, warmup=500, keep=1000)   # adapt=True default
# → window adaptation tunes step_size + inverse_mass_matrix in 500 steps,
#   loop collects 1000 samples from the adapted state.
```

For non-adaptive kernels (`mala`, `sgld`, `sghmc`) `adapt=` is silently
ignored and `warmup=N` retains the classic "discard the first N samples"
meaning.

!!! warning "Mixed mode (Bayesian + optax)"
    Window adaptation runs once at the start, with all non-Bayesian models
    at their **initial** weights.  In mixed setups (e.g. a Bayesian
    coefficient + an optax-trained surrogate) the adapted step size is
    tuned against the *untrained* surrogate's logdensity and is typically
    wrong for the actual joint problem.  Set `adapt=False` and pick
    `step_size` by hand in that case.

## Logdensity-aware initializers (`.initialize()` extension)

`Model.initialize(...)` already takes a path, a pytree, or a stateless
`(shape, dtype, key) -> array` callable.  A fourth shape is now
accepted: any object with `requires_logdensity = True` whose
`__call__` runs *inside* `solve()` with access to the loss-derived
log-density.  **Pathfinder** (Zhang et al. 2022) is the first concrete
implementation.

```python
import jno, blackjax

a.initialize(jno.bayesian.pathfinder(maxiter=30, num_samples=200))
a.bayesian(blackjax.nuts, step_size=1e-2, warmup=100, adapt=True)
```

Pathfinder runs L-BFGS on the log-density and turns the inverse-Hessian
trajectory into a normal approximation to the posterior.  From the
fitted `q` jno extracts (a) a warm starting position (the MAP for K=1
chains; K i.i.d. samples for K>1 — proper over-dispersion) and (b) a
diagonal `inverse_mass_matrix` from the per-dimension variance of M
draws.  `.bayesian()`'s `warmup` and `adapt` then apply **after**
pathfinder.

### Behaviour matrix

| `.initialize(pathfinder(...))` | `adapt` | What runs |
|---|---|---|
| not set | `True` | window adaptation from the user's init (today's default) |
| not set | `False` | user's init, user's step_size — no warmup |
| set | `True` | pathfinder → window: warm position + IMM, then window refines step_size |
| set | `False` | pathfinder only: warm position + pathfinder's IMM, user's step_size kept |

### The protocol — `_BayesianInitializer`

Minimal, single-method contract.  Any class with
`requires_logdensity = True` is detected by `Model.initialize` and
dispatched the same way at solve time:

```python
class _BayesianInitializer:
    requires_logdensity: ClassVar[bool] = True

    def __call__(self, rng_key, logdensity_fn, position, num_chains):
        # → (new_position, extra_kwargs_update)
        ...
```

* `logdensity_fn` is already mask-wrapped for
  `.mask(M).bayesian()` groups — subclasses see only the masked subset.
* For `num_chains > 1`, return a `(K, *leaf)`-leading pytree (one warm
  position per chain).
* `extra_kwargs_update` is merged into the kernel handle's
  `extra_kwargs`; keys the kernel doesn't accept (e.g. an IMM update
  against a MALA kernel) are silently dropped.

### Composition

* **Masks** — `.mask(M).bayesian()` + pathfinder works: pathfinder
  runs on the masked subset's log-density; the unmasked complement
  stays at init.
* **Multi-chain** — pathfinder samples K distinct starting positions
  from the fitted `q`.  `init_jitter` is silently overridden when an
  initializer is set.
* **Non-IMM kernels** (MALA / SGLD / SGHMC) — warm position is
  applied; the IMM update is silently dropped (signature gate).
* **`substeps=`** + initializer → clear error (the initializer runs
  on the full loss; substep kernels see only substep-local
  constraints).
* **`.vi(...)`** + initializer → clear error (VI initialises its own
  variational distribution from the position; warm-start is
  redundant).

### Future initializers (same hook, no further core changes)

| Slot | Algorithm | Notes |
|---|---|---|
| `jno.bayesian.pathfinder(...)` | blackjax pathfinder | This release.  See [Tutorial 11](../tutorials/10-bayesian-pinns/pathfinder-init.md). |
| `jno.bayesian.laplace(...)` | MAP via optax + `jax.hessian` (diagonal or full) | This release.  See [Tutorial 12](../tutorials/10-bayesian-pinns/laplace-init.md).  MacKay 1992 §6; Daxberger et al. 2021 §2. |
| `jno.bayesian.svgd(...)` | blackjax svgd | This release.  See [Tutorial 13](../tutorials/10-bayesian-pinns/svgd-init.md).  K particles → K chain inits.  Liu & Wang 2016 §3. |
| `jno.bayesian.map(...)` | Fixed-step optax warm-start | Future.  No IMM output; user keeps step_size. |
| User-written subclass | anything that fits the contract | See [Tutorial 11](../tutorials/10-bayesian-pinns/pathfinder-init.md). |

A worked example lives at
[Tutorial 11](../tutorials/10-bayesian-pinns/pathfinder-init.md).

## Multiple chains

Pass `num_chains=K` (default `1`) to run K independent MCMC chains in
parallel via `jax.vmap`:

```python
a.bayesian(
    blackjax.nuts,
    step_size=1e-2,
    num_chains=4,
    init_jitter=0.1,   # per-chain Gaussian perturbation of the initial position
    warmup=300,
    keep=400,
)
```

After `crux.solve()`, `a.posterior_samples` has shape `(K, N, *param)`
— the canonical arviz layout `(chain, draw, *)`.  All Bayesian models
in a single `solve()` must share the same `num_chains`; mismatched
values raise at solve-start.

`init_jitter > 0` over-disperses the K starting positions so R-hat is
conservative (chains forced apart at start; if they reconverge to the
same distribution, that's strong evidence of convergence).

A single window-adaptation sweep runs at start (PyMC convention) and
its adapted step-size + inverse mass matrix are broadcast to all K
chains; per-chain adaptation can be enabled in a follow-up if needed.

## Convergence diagnostics

Two pure-JAX helpers on `jno.bayesian` operate directly on the
`(K, N, *param)` chain layout — no `arviz` dep:

| Helper | What it computes | Threshold |
|---|---|---|
| `jno.bayesian.rhat(chain)` | Vehtari et al. 2021 split, rank-normalised, folded R-hat | < 1.01 (strict) or < 1.05 (lenient) → converged |
| `jno.bayesian.ess(chain)` | Effective sample size via FFT-based autocorrelation + Geyer 1992 truncation | > 100 per parameter typically sufficient |

Both return arrays of shape `*param`, one diagnostic per parameter
component.  For `K=1` `rhat` falls back to a split-R-hat using the two
halves of the chain (Gelman et al. 2014 BDA3 §11.4).

Example:

```python
chain = a.posterior_samples              # (K, N, 1)
r = jno.bayesian.rhat(chain)             # → (1,)
e = jno.bayesian.ess(chain)              # → (1,)
print(f"A: R-hat = {float(r[0]):.4f}, ESS = {float(e[0]):.1f}")
```

See [Tutorial 08](../tutorials/10-bayesian-pinns/multichain-nuts.md)
for a worked end-to-end example.

## Per-step kernel diagnostics — `model.posterior_diagnostics`

Every blackjax kernel call returns an `info` NamedTuple alongside the
new state.  jno captures the fields that matter for the kernel family
in use and aggregates them across the chain into
`model.posterior_diagnostics`, a `{field: (K, N) array}` dict:

| Kernel family             | Captured fields                                       |
|---------------------------|-------------------------------------------------------|
| NUTS / HMC                | `is_divergent` (bool), `acceptance_rate` (float), `energy` (float) |
| MALA                      | `acceptance_rate` only                                 |
| SGLD / SGHMC (SG-MCMC)    | `None` — these kernels have no `info` NamedTuple       |
| Mean-field VI             | `None` — track ELBO via `history.total_loss` instead   |

```python
a.bayesian(blackjax.nuts, step_size=1e-2, warmup=500, keep=1000)
crux.solve(...)

diag = a.posterior_diagnostics              # {"is_divergent": (K,N), ...}
n_divergent = int(diag["is_divergent"].sum())
acc = float(diag["acceptance_rate"].mean())  # target: 0.6–0.8 for NUTS
```

**`is_divergent` is the single most diagnostic signal of an unhealthy
NUTS / HMC run.**  More than ~1% divergent transitions almost always
means the integrator's `step_size` is too large for the local
posterior curvature — drop `step_size`, raise `inverse_mass_matrix`
to match the geometry, or run window adaptation (`adapt=True`).

The same information surfaces in three other places so problems are
loud:

1. **wandb** (per print-rate chunk) — `posterior/<name>/n_is_divergent`,
   `posterior/<name>/mean_acceptance_rate`,
   `posterior/<name>/mean_energy`.
2. **Solve-end summary** — one log line per Bayesian model::

       INFO: Model 'a': 12/1000 divergent (1.20%), mean_accept=0.81, mean_energy=42.3

3. **Handle-creation log** — each `.bayesian()` configuration logs the
   diagnostic schema it'll track::

       INFO: Model 1: Bayesian sampling via 'nuts' (kind=full, ..., diagnostics=is_divergent, acceptance_rate, energy)

Kernels that surface no info object (SG-MCMC, VI) are flagged
explicitly at handle-creation time (`diagnostics=none (kernel API has
no info object)`) — never silently downgraded.

## Pure-Bayesian fastpath (automatic)

When a `solve()` call qualifies as **pure-Bayesian** — every Bayesian
model on the same `num_chains`, `warmup`, `keep`, and `thin`; no
`.optimizer()` models in the same solve; no `substeps=`; no
`offload_data=True`; no trackers; no adaptive resampling; and
`inner_steps == 1` — `solve()` auto-dispatches to a scan-based
fastpath that closes three perf gaps in the per-epoch Python loop:

1. **No outer `value_and_grad`.**  The slow path runs
   `jax.value_and_grad(loss_wrapper)(trainable)` every step and
   discards the gradients when no optax models are present.  The
   fastpath omits that pass entirely — the MCMC kernels still compute
   their own gradients via the existing logdensity / grad-estimator
   closures.
2. **One XLA dispatch per `print_rate` steps.**  `warmup` steps run in
   a `jax.lax.fori_loop` with no sample accumulation; the post-warmup
   phase runs in a `jax.lax.scan` chunked at `print_rate` outer
   iterations, with `thin` inner steps per outer iteration in a nested
   `fori_loop`.  Typical solve with `print_rate ≈ keep / 10`: ~10
   XLA dispatches instead of `epochs` of them.
3. **One host transfer per chunk.**  Samples are stacked inside XLA
   and returned as a single `(chunk_keep, K, *param)` tensor per
   Bayesian model.

The dispatch is fully automatic — there is no kwarg to set.  At
solve-start jno logs a single line so the decision is visible::

    INFO: MCMC fastpath: scan over 500 samples × thin=1 + 0 warmup, chunked at print_rate=80.

If your solve doesn't qualify (mixed-mode, substeps, streaming, …)
the per-epoch Python loop runs exactly as before with all its
features intact.  The fastpath's output (`posterior_samples`,
wandb metrics, `history`) matches the slow path's at the same
`print_rate` cadence, just with fewer datapoints between print
boundaries.

## Variational Inference (mean-field)

`Model.vi(...)` fits a variational approximation to the posterior
through the same `crux.solve()` driver as `.bayesian()` — but
optimises the **evidence lower bound** (ELBO) of a Gaussian product
`q(θ) = ∏_i N(μ_i, σ_i)` instead of running an MCMC chain.  After
solve(), `posterior_draws` i.i.d. samples are drawn from the fitted
`q` and stored on the model as `posterior_samples` in the same
`(1, N, *param)` layout as the MCMC path:

```python
import blackjax, optax

a.vi(
    blackjax.meanfield_vi,
    optimizer=optax.adam(1e-3),
    num_samples=8,           # MC samples per ELBO eval
    posterior_draws=500,     # draws from fitted q for posterior_samples
)
crux.solve(2000)              # 2000 ELBO optimisation steps
chain = a.posterior_samples   # (1, 500, *param) — draws from fitted q
```

Configuration mirrors the MCMC `.bayesian()` API: same downstream
`crux.eval(samples="auto")` plumbing, same `jno.bayesian.{rhat, ess}`
helpers (which trivially report ≈ 1 and ≈ N respectively for VI
draws — see caveat below).

| Aspect | MCMC | Mean-field VI |
|---|---|---|
| Mechanism | per-step Metropolis-Hastings / Langevin / leapfrog | per-step ELBO optimisation |
| Cost | High (many forward passes per sample) | Low (one MC ELBO eval per step) |
| Calibration | Asymptotically exact | Diagonal-covariance lower bound |
| Multi-modal | Multi-chain reveals modes | Captures one mode |
| ``posterior_samples`` shape | `(K, N, *param)` from collected chain | `(1, posterior_draws, *param)` drawn from fitted q |

Two manual overrides on blackjax's defaults at `init_state` time make
VI converge usefully on non-trivial models:

* ``state.mu = position`` (the model's initial weights), rather than
  blackjax's zeros — gives VI a sensible starting point on
  non-trivial architectures (numpyro autoguide convention).
* ``state.rho = -3.0`` everywhere (initial std ≈ 0.05), rather than
  blackjax's wider default — keeps the initial MC ELBO sample close
  to the mean so the gradient estimator is low-variance from the
  start.  The optimiser then *grows* rho where the posterior is
  genuinely wide.

!!! tip "Likelihood scaling for VI — `likelihood_scale=`"
    The canonical Gaussian-noise log-likelihood is a **sum** over data
    points, but jno's ``residual.mse`` returns the **mean** — so by
    default the likelihood term in the ELBO is N× too small and the
    prior dominates, leaving VI stuck near initialisation.

    Pass ``likelihood_scale=N_obs`` (or ``N_obs / sigma**2`` more
    generally) on ``.vi(...)`` so the ELBO uses the correct
    magnitude::

        a.vi(
            blackjax.meanfield_vi,
            optimizer=optax.adam(1e-3),
            likelihood_scale=N_obs,   # canonical sum-over-data weighting
        )

    The same kwarg works on ``.bayesian(...)`` for MCMC kernels —
    less critical (HMC's geometry is more robust to magnitude) but
    still correct.  Available on both Pattern A (global) and Pattern
    B/D (masked) configurators.

    See [Tutorial 09](../tutorials/10-bayesian-pinns/vi-bnn-regression.md)
    for a worked end-to-end example.

### Diagnostics caveat

`jno.bayesian.rhat` and `jno.bayesian.ess` still run on VI
`posterior_samples`, but they trivially report ≈ 1 and ≈ N
respectively because the draws are i.i.d. from the fitted `q`.  For
VI convergence monitoring, watch the ELBO trajectory in `history`
(via `crux.solve(...).total_loss` per-chunk values) instead.

### Mutual exclusion

A single model has **either** `.bayesian()` **or** `.vi()` — never
both.  Setting one after the other raises a clear error.  Models with
VI and models with MCMC **can coexist** in the same solve call —
each runs its own paradigm during the step loop.

## Composable per-mask backends (v1)

`.mask(M)` followed by `.bayesian(...)` (or `.vi(...)`) restricts
the posterior to the subset of the model's parameter pytree where
`M` is `True`.  Leaves outside the mask **stay at their initial
value** throughout `solve()` — they are not updated.

```python
import equinox as eqx

# Mark only the output layer ("head") as Bayesian; body stays at init.
all_false     = jax.tree_util.tree_map(lambda _: False, net.module)
head_all_true = jax.tree_util.tree_map(lambda _: True, net.module.output_layer)
head_mask     = eqx.tree_at(lambda m: m.output_layer, all_false,
                            replace=head_all_true)

net.mask(head_mask).bayesian(blackjax.sgld, step_size=1e-3,
                              warmup=1500, keep=400, thin=2)
```

After solve, `net.posterior_samples` stores the **full** pytree at every
sample.  Leaves inside the mask vary across the chain; leaves outside
the mask are constant.  `crux.eval([net(x)], samples="auto")`,
`jno.bayesian.{rhat, ess}`, and wandb stats all work uniformly — no
special case for masked solves.

A worked example lives at
[Tutorial 10](../tutorials/10-bayesian-pinns/masked-bnn-head.md).

### What's supported

* **Pattern A**: `.mask(M).bayesian(...)` (or `.vi(...)`) on a model
  with **no** global `.optimizer(...)`.  Body stays at init; masked
  subset is the posterior.
* **Pattern B** *(Phase 15)*: `.mask(M).bayesian(...)` + global
  `.optimizer(...)` on the same model.  Body is Adam-trained; head is
  MCMC-sampled.  **K=1 and K>1 both supported**: for K>1 the body's
  gradient is computed at the chain-0 representative head (SAEM
  simplification).  Tutorial: [T14](../tutorials/10-bayesian-pinns/pattern-b-bnn-head.md).
* **Pattern D** *(Phase 16)*: multiple disjoint `.mask().bayesian()`
  groups on the same model.  Each group's kernel state lives at its
  own composite key (`"<lid>.<group_idx>"`) in `opt_states`; the step
  loop iterates groups in sorted order (Metropolis-within-Gibbs cycle
  for K=1; SAEM-style chain-0 representative for K>1).
* **Pattern E** *(Phase 16)*: mixed VI + MCMC on disjoint masks of
  the same model.  MCMC accumulates per-step samples; at solve end
  the VI handle draws `posterior_draws` i.i.d. samples from its
  fitted distribution and splices them into the MCMC chain at the
  VI mask's leaves.  **Strict matching**: VI's `posterior_draws` must
  equal the MCMC group's `keep` (validated at solve start).
* **Masked + `num_chains > 1`** *(Phase 15)*: masked Bayesian solves
  with K parallel chains work for Pattern A, B, and D.
* **`.lora()` + `.bayesian()`** (no mask) — the LoRA partition already
  restricts trainable parameters to the LoRA adapters; `.bayesian()`
  samples that restricted subset.

### State storage (composite keys)

Internally, `opt_states` uses two key formats:

* **Bare** `"<lid>"` — optax states (one per layer with `.optimizer(...)`).
* **Composite** `"<lid>.<group_idx>"` — Bayesian / VI kernel states
  (one per masked group; bare-`.bayesian()` layers use `"<lid>.0"`).

A Pattern B + D layer therefore carries entries like
`{"1": optax_state, "1.0": kernel_g0, "1.1": kernel_g1}` simultaneously.
The helpers `jno.core._lid_of(k)`, `_group_idx_of(k)`, `_bay_key(lid, gi)`
parse / build these keys.

### What's still blocked

The composite-key scheme generalises further (multiple VI groups
beyond one per layer, overlapping masks, etc.) but the current
implementation only validates the patterns above.  Patterns not in
the supported list above fall back to clear `NotImplementedError` /
`ValueError` at solve start.

### What `posterior_samples` looks like

The chain stores the **full** model pytree (both masked and unmasked
leaves) with leading axes `(K, N, *param_shape)` for K=1 single-chain
solves.  Unmasked leaves are constant along the chain axis; masked
leaves vary.  This full-pytree storage is a memory cost: for very
narrow masks on wide models, sparse storage (varying leaves only +
a frozen-leaves snapshot, reassembled lazily) is documented as a v2
follow-up.  The user-facing API will not change.

## Wandb integration

When a wandb run is active, per-Bayesian-model statistics are logged at
the same print-rate cadence as the rest of the training metrics:

| Key                                          | Meaning                                                            |
|----------------------------------------------|--------------------------------------------------------------------|
| `posterior/<name>/n_samples`                 | Number of samples collected in the chain so far                    |
| `posterior/<name>/n_chains`                  | `num_chains` for this model (1 for the default)                    |
| `posterior/<name>/mean`                      | Running posterior mean (scalar parameters only)                    |
| `posterior/<name>/n_is_divergent`            | Running divergent-transition count (NUTS / HMC only)               |
| `posterior/<name>/mean_acceptance_rate`      | Running mean MH acceptance rate (NUTS / HMC / MALA)                |
| `posterior/<name>/mean_energy`               | Running mean Hamiltonian energy (NUTS / HMC only)                  |

`<name>` comes from the `name=` argument of `jno.np.parameter(...)` or
`jno.nn.wrap(..., name=...)`.  Multi-leaf modules (MLPs) only get the
chain length — full per-leaf statistics are out of scope for jno-side
logging; use arviz against `model.posterior_samples` instead.

!!! warning "Memory"
    A full chain costs ~`keep × #params × 4 bytes` per Bayesian model.  For
    large BNN PINNs increase `thin=` or decrease `keep=` to stay within
    GPU/CPU memory.

## References

- NUTS — Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler:
  Adaptively Setting Path Lengths in Hamiltonian Monte Carlo.* Journal of
  Machine Learning Research, 15(1), 1593–1623.
- SGLD — Welling, M., & Teh, Y. W. (2011). *Bayesian Learning via
  Stochastic Gradient Langevin Dynamics.* ICML 2011, 681–688.

The kernels themselves come from [blackjax](https://blackjax-devs.github.io/blackjax) —
jno only wires their `(state, key) → state` interface into the per-model
step dispatch.

## Limitations (this release)

- **VI (`blackjax.vi.*`)** has different mechanics (ELBO optimisation) and
  is not yet routed through `.bayesian()`.
- **Multi-chain** — chains are single-chain only.  Running K chains for
  R-hat / cross-chain diagnostics needs K separate `solve()` calls and
  manual stacking (or a follow-up that wires `jax.vmap` over seeds).
- **Discrete posteriors** (e.g. over `Choice` selections) need SMC and are
  out of scope.
- **Custom forward models outside the jNO tracer** (FEM solver, ODE
  integrator, finite volume) can't be wrapped in `.bayesian()` directly —
  the API expects the forward to be expressible as a jNO Placeholder
  expression.  For those cases use blackjax directly with jNO supplying
  the differentiable forward; see [Inverse FEM
  Diffusivity](../tutorials/10-bayesian-pinns/inverse-fem-diffusivity.md)
  for the pattern.

## Combining with `substeps=`

`substeps=` is supported with `.bayesian()` to enable the *two-stage
decoupled inference* pattern: substep 0 trains a surrogate via optax,
substep 1 runs one NUTS proposal on a Bayesian coefficient with the
surrogate `stop_gradient`-ed.  See the section index of the
[Bayesian PINNs tutorial chapter](../tutorials/10-bayesian-pinns/index.md).
When using substeps you must set `adapt=False` on the Bayesian model —
window adaptation runs against the full loss but the kernel only sees
the substep-local constraint set.
