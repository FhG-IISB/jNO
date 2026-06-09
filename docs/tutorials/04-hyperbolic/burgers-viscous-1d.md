# Burgers Viscous 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/burgers_viscous_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to chapter</a>
</div>

This example solves the viscous Burgers equation with a manufactured exact solution and demonstrates **RAD adaptive resampling** — a technique that concentrates collocation points in high-residual regions to accelerate convergence.

## Problem Setup

The PDE has the form `u_t + u u_x = ν u_xx + f`, with manufactured solution `u_exact = e^{-t} sin(πx)` and the corresponding forcing term `f`.

The domain `[0,1] × [0,1]` is discretised as a 2D rectangle with `x` (space) and `t` (time) treated as two spatial coordinates — the standard PINN formulation that makes full adaptive resampling available.

## Step 1: Define the Resampling Strategy

RAD fires every 200 epochs (starting at epoch 500 once the network has formed a rough solution), replacing 20 % of working-set points with draws from the full 513-node mesh pool, biased toward regions of high PDE residual.

```python
from jno import sampler

strategy = sampler.rad(
    resample_every=200,    # check for updates every 200 epochs
    resample_fraction=0.2, # replace 20 % of working-set points
    start_epoch=500,       # wait for a rough solution before resampling
    k=5,                   # cluster around the top-5 high-residual anchors
)
```

## Step 2: Build the Space-Time Domain

`mesh_size=0.05` on `[0,1]²` gives 513 interior nodes in the candidate pool. With `sample=(60, None)`, the network trains on a 60-point working set — an **8× pool-to-sample ratio** that gives RAD substantial room to move points into high-residual regions.

```python
domain = 1 * jno.domain.rect(
    mesh_size=0.05,
    x_range=(0.0, 1.0),
    y_range=(0.0, 1.0),
)
vars_int = domain.variable("interior", sample=(60, None), resampling_strategy=strategy)
x, t = vars_int[0], vars_int[1]

# IC boundary: t = 0 (bottom face of the rectangle)
vars_bot = domain.variable("bottom", sample=(20, None))
x0, t0 = vars_bot[0], vars_bot[1]
```

## Step 3: Define the Network and PDE

An MLP with 2 inputs `(x, t)` takes both coordinates directly. The `x*(1-x)` factor hard-encodes the Dirichlet BCs `u(0,t) = u(1,t) = 0`.

```python
net = jno.nn.wrap(foundax.mlp(2, hidden_dims=64, num_layers=4, key=jax.random.PRNGKey(3)))
net.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1e-3, warmup_steps=1, decay_steps=10, end_value=1e-5)))

u = net(x, t) * x * (1 - x)

u_t  = u.d(t)
u_x  = u.d(x)
u_xx = u_x.d(x)
pde  = u_t + u * u_x - ν * u_xx - source
```

## Step 4: Train and Evaluate

```python
u_0 = net(x0, t0) * x0 * (1 - x0)
ini = u_0 - jno.np.sin(π * x0)

crux    = jno.core([pde.mse, ini.mse])
history = crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
```

## What To Notice

- The nonlinear term `u u_x` creates residuals that are largest near `x = 0.5` (the maximum of `sin(πx)`). After the warm-up phase, RAD progressively moves interior points toward that region.
- The **pool-to-sample ratio** (8×) is what gives RAD room to relocate points. A ratio below 2× leaves too few candidates to draw from.
- The initial-condition boundary (`t = 0`, sampled via `"bottom"`) is kept fixed — only interior points are resampled.
- For the classic inviscid-limit Burgers benchmark (`ν = 0.01/π`, no forcing, IC `u(x,0) = -sin(πx)`), the steep gradient forms near `x = 0` at late times and RAD provides the largest gains. See [`adaptive/resampling.md`](../../adaptive/resampling.md) for that setup.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/burgers_viscous_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to 04 Hyperbolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/04_hyperbolic/burgers_viscous_1d.py"
```
