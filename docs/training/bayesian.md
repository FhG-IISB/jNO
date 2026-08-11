# Bayesian sampling — `.bayesian(...)` per parameter

`model.bayesian(kernel_factory, **kw)` replaces a model's per-step gradient update with one transition
of a [blackjax](https://blackjax-devs.github.io/blackjax) MCMC kernel. The configurator mirrors
`.optimizer(...)` — each parameter can independently be **optimised** (point estimate via optax) or
**sampled** (posterior chain via blackjax); mix freely and `crux.solve()` dispatches per-model.

After training the chain is on the model as `model.posterior_samples`, and on the `crux` via
`crux.eval([...], samples="chain")` which vmaps the evaluator over the chain so nonlinear predictions
push forward correctly.

## Quick example — Bayesian inverse problem

Recover `(A, B)` in `d(x) = A·sin(πx) + B·cos(πx)` from noisy observations, with credible intervals:

```python
import blackjax, jax, jno
import jax.numpy as jnp

π = jno.np.pi
dom = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.01).domain()
x, _ = dom.variable("interior")

A_true, B_true = 3.14, -2.71
target = A_true * jno.np.sin(π * x) + B_true * jno.np.cos(π * x) + jno.noise.gaussian(std=0.1)

k1, k2 = jax.random.split(jax.random.PRNGKey(0), 2)
a = jno.np.parameter((1,), key=k1, name="a")
b = jno.np.parameter((1,), key=k2, name="b")

for p in [a, b]:
    # adapt=True (default) tunes step_size + inverse_mass_matrix via blackjax.window_adaptation.
    p.bayesian(blackjax.nuts, step_size=1e-2, warmup=500, keep=1000)

residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - target
crux = jno.core([residual.mse])
crux.solve(1500)

a_chain = a.posterior_samples           # (1000, 1) — leading axis = sample
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

`kernel_factory` is duck-typed at solve time by the **first parameter name** of the factory:

| First parameter of factory   | Family               | Examples                          |
|------------------------------|----------------------|-----------------------------------|
| `logdensity_fn`              | Full-data MCMC       | `blackjax.nuts`, `blackjax.hmc`, `blackjax.mala` |
| `grad_estimator`             | Stochastic-gradient  | `blackjax.sgld`, `blackjax.sghmc` |

jno builds the appropriate closure from the live loss + context and rebuilds the kernel inside the JIT
graph each step.

### Custom kernel factories — `(logdensity_fn, **kw) → SamplingAlgorithm`

You are **not** limited to the kernels above. `kernel_factory` is anything that returns a
[`blackjax.SamplingAlgorithm`](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/base/index.html)
(a `NamedTuple(init, step)`); jno detects the family from the first parameter name (`logdensity_fn` → a
closure `θ → log p(data | θ) + log p(θ)`; `grad_estimator` → a mini-batch gradient closure). A barebones
random-walk Metropolis factory, for illustration:

```python
import blackjax, jax
import jax.numpy as jnp

def my_rwm(logdensity_fn, step_size):
    def init_fn(position):
        return {"position": position, "logdensity": logdensity_fn(position)}

    def step_fn(rng_key, state):
        k1, k2 = jax.random.split(rng_key)
        prop = jax.tree.map(lambda p: p + step_size * jax.random.normal(k1, p.shape), state["position"])
        new_logd = logdensity_fn(prop)
        accept = jnp.log(jax.random.uniform(k2)) < new_logd - state["logdensity"]
        new_state = {
            "position":  jax.tree.map(lambda x, y: jnp.where(accept, y, x), state["position"], prop),
            "logdensity": jnp.where(accept, new_logd, state["logdensity"]),
        }
        return new_state, {"accepted": accept}

    return blackjax.SamplingAlgorithm(init_fn, step_fn)

net.bayesian(my_rwm, step_size=1e-2, warmup=500, keep=1000)
```

Your factory goes through the same warmup / thin / keep pipeline as the built-ins.

## Output — chains by default

`crux.eval(...)` is **auto-chain-aware** per expression: if an expression's dependency graph touches a
model with `posterior_samples` set, the evaluator is `vmap`-ped over that chain; otherwise it evaluates
at the point value. No `samples=` argument is needed for the common case.

| Read                              | `.optimizer()` (point)   | `.bayesian()` (chain)                          |
|-----------------------------------|--------------------------|------------------------------------------------|
| `crux.eval([m])`                  | point value              | `(n_kept, *m_shape)` chain                     |
| `crux.eval([expr])`               | `(n_points, …)` point    | `(n_kept, n_points, …)` chain (auto)           |
| `m.posterior_samples`             | `None`                   | stacked module pytree (or array for `parameter`) |

No `.mean()/.std()/.quantile()` helpers are provided — compute summaries from the chain with `jnp`,
arviz, or your plotting library. For a neural-network prediction the posterior mean over outputs is
**not** the output at the posterior mean of the weights; the auto-vmap does the right thing:

```python
u_chain = crux.eval([u])                          # (n_kept, n_points, 1)
u_mean = jnp.mean(u_chain, axis=0)
u_lo, u_hi = jnp.quantile(u_chain, jnp.array([0.05, 0.95]), axis=0)
```

Escape hatches: `crux.eval([u], samples="chain")` forces the chain (raises if no Bayesian deps);
`samples="point"` evaluates at the last sample (skips the vmap — quick debugging, **not** a substitute
for the posterior summary on nonlinear outputs).

## Mixed: optimised + sampled

Different parameters use different update rules in one `crux.solve()`:

```python
encoder = jno.nn(foundax.mlp(2, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(0)))
head    = jno.nn(foundax.mlp(1, hidden_dims=32, num_layers=1, key=jax.random.PRNGKey(1)))

encoder.optimizer(optax.adam(1e-3))                          # point estimate
head.bayesian(blackjax.sgld, step_size=1e-5)                 # SGLD chain
```

Only `head.posterior_samples` is populated; `encoder.posterior_samples` is `None`.

## Bayesian PINN — predictive bands

```python
import blackjax, foundax, jax, jno
import jax.numpy as jnp

π = jno.np.pi
dom = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.01).domain()
x, _  = dom.variable("interior")
xb, _ = dom.variable("boundary")

net = jno.nn(foundax.mlp(1, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(0)))
net.bayesian(blackjax.sgld, step_size=1e-5, warmup=2000, keep=1000)

u    = net(x)
u_xx = jno.diff(u, x, order=2)
pde  = u_xx + (π ** 2) * jno.np.sin(π * x)        # u'' = -π² sin(πx)
bc   = net(xb) - 0.0

crux = jno.core([pde.mse, bc.mse])
crux.solve(3000)

u_chain = crux.eval([u], samples="chain")
u_mean     = jnp.mean(u_chain, axis=0)
u_lo, u_hi = jnp.quantile(u_chain, jnp.array([0.05, 0.95]), axis=0)
```

## Priors — built-in factories

The `prior=` argument takes any `pytree → float` returning the log-prior density. Four factories live at
`jno.bayesian.priors.*`:

| Factory | Form | When to use |
|---|---|---|
| `priors.gaussian(sigma=10.0, fan_in_aware=False)` | $-\|\theta\|^2 / (2\sigma^2)$ | Wide default (σ=10) is "effectively flat"; smaller σ for shrinkage. `fan_in_aware=True` scales σ by 1/√fan_in per weight tensor. |
| `priors.laplace(scale=1.0)` | $-\|\theta\|_1 / \text{scale}$ | Sparse-friendly: encourages many components near zero. |
| `priors.student_t(df=4.0, scale=1.0)` | $-\frac{df+1}{2} \sum \log\big(1 + (\theta/\text{scale})^2 / df\big)$ | Heavy-tailed; practical substitute for horseshoe on individual weights. `df > 2` for finite variance. |
| `priors.layerwise_gaussian(base_sigma=1.0, default_sigma=1.0, fan_in_aware=True)` | Per-leaf $N(0, \sigma_\text{leaf}^2)$, $\sigma_\text{weight} = \text{base}/\sqrt{\text{fan\_in}}$, $\sigma_\text{bias} = \text{default}$ | The standard BNN-PINN prior (Sun et al. 2019, Wenzel et al. 2020). |

```python
a.bayesian(blackjax.nuts, step_size=1e-2, prior=jno.bayesian.priors.gaussian(sigma=10.0))   # wide default
a.bayesian(blackjax.nuts, step_size=1e-2, prior=jno.bayesian.priors.laplace(scale=0.5))      # sparse
head.mask(M).bayesian(blackjax.sgld, step_size=1e-3,
                      prior=jno.bayesian.priors.layerwise_gaussian())                         # BNN head
```

When `prior=None` the default is `priors.gaussian(sigma=10.0)` — effectively flat at typical parameter
scales, but for BNN weights at scale 0.01 it's overly wide and for outputs at scale 100 overly tight;
prefer a named factory for non-trivial problems.

Any `pytree → float` callable works as a **custom prior** (e.g. an L½ prior summing
`jnp.sqrt(jnp.abs(leaf))` over floating leaves).

!!! note "Masked priors see only the masked subset"
    When configured via `.mask(M).bayesian()` / `.mask(M).vi()`, the prior closure receives the
    **masked subset** of the position (whatever the kernel sees) — not the full pytree. The built-in
    factories iterate over the leaves they're handed, so masked and unmasked solves behave identically.
    A custom prior needing the full pytree (e.g. a hierarchical prior coupling masked and unmasked
    leaves) should use a global `.bayesian()` rather than `.mask(M).bayesian()`.

## Adaptation (NUTS / HMC)

For HMC-family kernels, `adapt=True` (default) runs `blackjax.window_adaptation` for the first `warmup`
steps **before** the main loop — step size and inverse mass matrix are tuned automatically, then the loop
collects `keep` samples from epoch 0. For non-adaptive kernels (`mala`, `sgld`, `sghmc`) `adapt=` is
silently ignored and `warmup=N` retains the classic "discard the first N samples" meaning.

!!! warning "Mixed mode (Bayesian + optax)"
    Window adaptation runs once at the start, with all non-Bayesian models at their **initial** weights.
    In mixed setups (a Bayesian coefficient + an optax-trained surrogate) the adapted step size is tuned
    against the *untrained* surrogate's logdensity and is typically wrong for the joint problem. Set
    `adapt=False` and pick `step_size` by hand there.

## Logdensity-aware initializers (`.initialize()` extension)

`Model.initialize(...)` accepts a fourth shape (beyond a path, a pytree, or a `(shape, dtype, key) ->
array` callable): any object with `requires_logdensity = True` whose `__call__` runs *inside* `solve()`
with access to the loss-derived log-density. **Pathfinder** (Zhang et al. 2022) is the first:

```python
a.initialize(jno.bayesian.pathfinder(maxiter=30, num_samples=200))
a.bayesian(blackjax.nuts, step_size=1e-2, warmup=100, adapt=True)
```

Pathfinder runs L-BFGS on the log-density and turns the inverse-Hessian trajectory into a normal
approximation to the posterior. From the fitted `q` jno extracts a warm starting position (MAP for K=1;
K i.i.d. over-dispersed samples for K>1) and a diagonal `inverse_mass_matrix`; `warmup`/`adapt` then apply
**after** pathfinder:

| `.initialize(pathfinder(...))` | `adapt` | What runs |
|---|---|---|
| not set | `True` | window adaptation from the user's init (default) |
| not set | `False` | user's init, user's step_size — no warmup |
| set | `True` | pathfinder → window: warm position + IMM, then window refines step_size |
| set | `False` | pathfinder only: warm position + pathfinder's IMM, user's step_size kept |

Composition: works with `.mask(M).bayesian()` (runs on the masked subset) and `num_chains > 1` (K distinct
starts); on non-IMM kernels (MALA/SGLD/SGHMC) the warm position is applied and the IMM update silently
dropped; with `substeps=` or `.vi(...)` it raises a clear error. Other initializers on the same hook:

| Slot | Algorithm | Notes |
|---|---|---|
| `jno.bayesian.pathfinder(...)` | blackjax pathfinder | Zhang et al. 2022. |
| `jno.bayesian.laplace(...)` | MAP via optax + `jax.hessian` (diagonal or full) | MacKay 1992 §6; Daxberger et al. 2021 §2. |
| `jno.bayesian.svgd(...)` | blackjax svgd | K particles → K chain inits. Liu & Wang 2016 §3. |

A worked example lives at [the Bayesian tutorials](../tutorials/10-bayesian-pinns/index.md).

## Multiple chains

Pass `num_chains=K` (default `1`) to run K independent MCMC chains in parallel via `jax.vmap`:

```python
a.bayesian(blackjax.nuts, step_size=1e-2, num_chains=4,
           init_jitter=0.1,   # per-chain Gaussian perturbation of the initial position
           warmup=300, keep=400)
```

After `crux.solve()`, `a.posterior_samples` has shape `(K, N, *param)` — the canonical arviz layout
`(chain, draw, *)`. All Bayesian models in a solve must share `num_chains` (mismatches raise at start).
`init_jitter > 0` over-disperses the K starts so R-hat is conservative. A single window-adaptation sweep
runs at start (PyMC convention) and its adapted step-size + IMM are broadcast to all K chains.

## Convergence diagnostics

Two pure-JAX helpers on `jno.bayesian` operate directly on the `(K, N, *param)` layout — no `arviz` dep:

| Helper | What it computes | Threshold |
|---|---|---|
| `jno.bayesian.rhat(chain)` | Vehtari et al. 2021 split, rank-normalised, folded R-hat | < 1.01 (strict) or < 1.05 (lenient) → converged |
| `jno.bayesian.ess(chain)` | Effective sample size via FFT autocorrelation + Geyer 1992 truncation | > 100 per parameter typically sufficient |

Both return arrays of shape `*param`, one diagnostic per parameter component:

```python
chain = a.posterior_samples              # (K, N, 1)
print(f"R-hat = {float(jno.bayesian.rhat(chain)[0]):.4f}, ESS = {float(jno.bayesian.ess(chain)[0]):.1f}")
```

`rhat(chain, strategy=...)` controls single-chain input: `"auto"` (default) does split-R-hat on the two
halves for K==1 and multichain R-hat for K≥2; `"multichain"` raises for K==1 (loud failure when you
expected multiple chains); `"split"` splits every chain in half for an extra stationarity check.

## Per-step kernel diagnostics — `model.posterior_diagnostics`

jno captures each kernel's per-step `info` and aggregates it across the chain into
`model.posterior_diagnostics`, a `{field: (K, N) array}` dict:

| Kernel family             | Captured fields                                       |
|---------------------------|-------------------------------------------------------|
| NUTS / HMC                | `is_divergent` (bool), `acceptance_rate` (float), `energy` (float) |
| MALA                      | `acceptance_rate` only                                 |
| SGLD / SGHMC (SG-MCMC)    | `None` — these kernels have no `info` NamedTuple       |
| Mean-field VI             | `None` — track ELBO via `history.total_loss` instead   |

```python
diag = a.posterior_diagnostics              # {"is_divergent": (K,N), ...}
n_divergent = int(diag["is_divergent"].sum())
acc = float(diag["acceptance_rate"].mean())  # target: 0.6–0.8 for NUTS
```

**`is_divergent` is the single most diagnostic signal of an unhealthy NUTS / HMC run.** More than ~1%
divergent transitions almost always means `step_size` is too large for the local posterior curvature —
drop `step_size`, tune `inverse_mass_matrix`, or run window adaptation (`adapt=True`). The same signal
surfaces in wandb (`posterior/<name>/n_is_divergent`, …), a solve-end summary line, and the
handle-creation log; kernels with no info object (SG-MCMC, VI) are flagged explicitly, never silently
downgraded.

## Pure-Bayesian fastpath (automatic)

When a solve is **pure-Bayesian** — every Bayesian model on the same `num_chains`/`warmup`/`keep`/`thin`;
no `.optimizer()` models, no `substeps=`, no `offload_data=True`, no trackers, no adaptive resampling,
`inner_steps == 1` — `solve()` auto-dispatches to a scan-based fastpath (no outer `value_and_grad`, one
XLA dispatch per `print_rate` steps, one host transfer per chunk). It is fully automatic (no kwarg); jno
logs one line at solve-start so the decision is visible. Non-qualifying solves run the per-epoch Python
loop unchanged, with identical output at the same `print_rate` cadence.

## Variational Inference (mean-field)

`Model.vi(...)` fits a variational approximation through the same `crux.solve()` driver as `.bayesian()`
— but optimises the **evidence lower bound** (ELBO) of a Gaussian product `q(θ) = ∏_i N(μ_i, σ_i)`
instead of running an MCMC chain. After solve, `posterior_draws` i.i.d. samples from the fitted `q` are
stored as `posterior_samples` in the same `(1, N, *param)` layout as the MCMC path:

```python
import blackjax, optax

a.vi(blackjax.meanfield_vi, optimizer=optax.adam(1e-3),
     num_samples=8,           # MC samples per ELBO eval
     posterior_draws=500)     # draws from fitted q for posterior_samples
crux.solve(2000)              # 2000 ELBO optimisation steps
chain = a.posterior_samples   # (1, 500, *param)
```

Downstream `crux.eval` and `jno.bayesian.{rhat, ess}` plumbing is identical to MCMC. Trade-offs:

| Aspect | MCMC | Mean-field VI |
|---|---|---|
| Mechanism | per-step Metropolis-Hastings / Langevin / leapfrog | per-step ELBO optimisation |
| Cost | High (many forward passes per sample) | Low (one MC ELBO eval per step) |
| Calibration | Asymptotically exact | Diagonal-covariance lower bound |
| Multi-modal | Multi-chain reveals modes | Captures one mode |

Two overrides on blackjax's `init_state` defaults, exposed as `Model.vi(...)` kwargs, make VI converge on
non-trivial models: `init_mu_at_position=True` (default) starts `state.mu` at the model's initial weights
(numpyro autoguide convention), and `init_log_std=-3.0` (default → σ ≈ 0.05) starts `state.rho` small so
the initial MC ELBO estimator is low-variance (the optimiser grows rho where the posterior is genuinely
wide).

!!! tip "Likelihood scaling for VI — `likelihood_scale=`"
    The canonical Gaussian log-likelihood is a **sum** over data points, but `residual.mse` returns the
    **mean** — so by default the ELBO likelihood term is N× too small and the prior dominates, leaving VI
    stuck near init. Pass `likelihood_scale=N_obs` (or `N_obs / sigma**2`) so the ELBO uses the correct
    magnitude. The same kwarg works on `.bayesian(...)` (less critical — HMC's geometry is more robust).

For VI convergence monitoring watch the ELBO in `history` (via `crux.solve(...).total_loss`) —
`jno.bayesian.{rhat, ess}` trivially report ≈ 1 and ≈ N because the draws are i.i.d. from `q`. A single
model has **either** `.bayesian()` **or** `.vi()`, never both (setting one after the other raises); VI and
MCMC models can coexist in one solve.

## Composable per-mask backends

`.mask(M)` followed by `.bayesian(...)` (or `.vi(...)`) restricts the posterior to the subset of the
pytree where `M` is `True`; leaves outside the mask stay at their initial value throughout `solve()`.

```python
import equinox as eqx

# Mark only the output layer ("head") as Bayesian; body stays at init.
all_false     = jax.tree_util.tree_map(lambda _: False, net.module)
head_all_true = jax.tree_util.tree_map(lambda _: True, net.module.output_layer)
head_mask     = eqx.tree_at(lambda m: m.output_layer, all_false, replace=head_all_true)

net.mask(head_mask).bayesian(blackjax.sgld, step_size=1e-3, warmup=1500, keep=400, thin=2)
```

After solve, `net.posterior_samples` stores the **full** pytree at every sample (masked leaves vary,
unmasked leaves constant); `crux.eval`, `jno.bayesian.{rhat, ess}`, and wandb stats work uniformly.
Supported combinations:

* **`.mask(M).bayesian()` with no global `.optimizer()`** — body stays at init; masked subset is the
  posterior.
* **`.mask(M).bayesian()` + a global `.optimizer()`** on the same model — body is Adam-trained, head is
  MCMC-sampled (K=1 and K>1; for K>1 the body's gradient uses the chain-0 head, an SAEM simplification).
* **Multiple disjoint `.mask().bayesian()` groups** on one model — Metropolis-within-Gibbs cycle (K=1) /
  chain-0 representative (K>1).
* **Mixed VI + MCMC on disjoint masks** — VI's `posterior_draws` must equal the MCMC group's `keep`.
* **Masked + `num_chains > 1`**, and **`.lora()` + `.bayesian()`** (samples the LoRA-restricted subset).

!!! warning "Masked head + optax body, K>1 — SAEM chain-0 representative"
    With `num_chains>1`, only chain 0 of the K head chains influences the body's optax update each step.
    All K chains still explore the head's posterior, but the body sees a single representative —
    SAEM-style joint inference, **not** K independent head+body solves. jno emits a one-line `WARNING` at
    solve-start. Pass `num_chains=1` for independent runs (one solve per chain).

Combinations outside the supported list fall back to a clear `NotImplementedError` / `ValueError` at
solve start.

## Wandb integration

When a wandb run is active, per-Bayesian-model statistics are logged at the print-rate cadence:

| Key                                          | Meaning                                                            |
|----------------------------------------------|--------------------------------------------------------------------|
| `posterior/<name>/n_samples`                 | Samples collected in the chain so far                              |
| `posterior/<name>/n_chains`                  | `num_chains` for this model                                        |
| `posterior/<name>/mean`                      | Running posterior mean (scalar parameters only)                    |
| `posterior/<name>/n_is_divergent`            | Running divergent-transition count (NUTS / HMC only)               |
| `posterior/<name>/mean_acceptance_rate`      | Running mean MH acceptance rate (NUTS / HMC / MALA)                |
| `posterior/<name>/mean_energy`               | Running mean Hamiltonian energy (NUTS / HMC only)                  |

`<name>` comes from the `name=` of `jno.np.parameter(...)` / `jno.nn(..., name=...)`. Multi-leaf
modules (MLPs) get only the chain length — use arviz against `model.posterior_samples` for per-leaf stats.

!!! warning "Memory"
    A full chain costs ~`keep × #params × 4 bytes` per Bayesian model. For large BNN PINNs increase
    `thin=` or decrease `keep=` to stay within memory.

## Combining with `substeps=`

`substeps=` enables *two-stage decoupled inference*: substep 0 trains a surrogate via optax, substep 1
runs one NUTS proposal on a Bayesian coefficient with the surrogate `stop_gradient`-ed. Set `adapt=False`
on the Bayesian model — window adaptation runs against the full loss but the kernel only sees the
substep-local constraint set. See the [Bayesian PINNs tutorial chapter](../tutorials/10-bayesian-pinns/index.md).

## References

- NUTS — Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler: Adaptively Setting Path Lengths in
  Hamiltonian Monte Carlo.* JMLR 15(1), 1593–1623.
- SGLD — Welling, M., & Teh, Y. W. (2011). *Bayesian Learning via Stochastic Gradient Langevin Dynamics.*
  ICML 2011, 681–688.

The kernels come from [blackjax](https://blackjax-devs.github.io/blackjax) — jno only wires their
`(state, key) → state` interface into the per-model step dispatch.

## Limitations

- **Discrete posteriors** (e.g. over `Choice` selections) need SMC and are out of scope.
- **Custom forward models outside the jNO tracer** (a FEM solver, ODE integrator, finite volume) can't be
  wrapped in `.bayesian()` directly — the API expects the forward to be a jNO Placeholder expression. For
  those, use blackjax directly with jNO supplying the differentiable forward; see the [Inverse FEM
  diffusivity pattern](../tutorials/10-bayesian-pinns/index.md).
