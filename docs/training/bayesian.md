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
    p.bayesian(
        blackjax.nuts,
        step_size=1e-2,
        inverse_mass_matrix=jnp.ones(1),
        warmup=500,
        keep=1000,
    )

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
    **kernel_kwargs,            # forwarded to kernel_factory; must include step_size=
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

(u_chain,) = crux.eval([u], samples="chain")
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

a.bayesian(blackjax.nuts, step_size=1e-2,
           inverse_mass_matrix=jnp.ones(1), prior=laplace_prior)
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

- **No combination with `substeps=`** — using both in one `solve()` raises
  a clear error.
- **No automatic adaptation** — supply `step_size` (and
  `inverse_mass_matrix` for HMC/NUTS) yourself.
- **VI (`blackjax.vi.*`)** has different mechanics (ELBO optimisation) and
  is not yet routed through `.bayesian()`.
- **Discrete posteriors** (e.g. over `Choice` selections) need SMC and are
  out of scope.
