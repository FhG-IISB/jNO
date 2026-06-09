# Wave 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/wave_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to chapter</a>
</div>

Second-order-in-time wave equation — needs **both** an initial displacement and an initial velocity. Demonstrated with **soft enforcement** of all four conditions (displacement IC, velocity IC, and two Dirichlet BCs).

## Problem Setup

`u_tt = c² u_xx` on `(x, t) ∈ [0, 1] × [0, 1]`, with `u(0, t) = u(1, t) = 0`, `u(x, 0) = sin(π x)`, `u_t(x, 0) = 0`. Exact solution: `u(x, t) = cos(c π t) sin(π x)` — a half-period standing wave.

## Step 1: Build a Space-Time Domain

```python
c = 1.0
T_end = 1.0

domain = jno.domain(
    constructor=jno.domain.line(mesh_size=0.01),
    time=(0, T_end, 20),
)
x, t   = domain.variable("interior")
x0, t0 = domain.variable("initial")    # t = 0 — for both ICs
xb, tb = domain.variable("boundary")   # x = 0 / x = 1 at all t — for the BCs
```

## Step 2: Bare-Network Ansatz

```python
net = jno.nn.wrap(
    foundax.deeponet(n_sensors=1, coord_dim=1, n_outputs=1,
                     n_layers=6, basis_functions=128, hidden_dim=96,
                     activation=jax.nn.tanh, key=jax.random.PRNGKey(7))
)
net.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(
    init_value=1e-6, peak_value=1e-3,
    warmup_steps=200, decay_steps=49800, end_value=1e-7,
)))

u = net(t, x)
```

## Step 3: Four Constraints — PDE + 2 ICs + BC

```python
# Interior PDE residual:  u_tt − c² u_xx = 0
pde = u.d2(t) - c**2 * u.d2(x)

# Initial displacement:   u(x, 0) = sin(πx)
u0 = net(t0, x0)
ic_disp = u0 - jno.np.sin(π * x0)

# Initial velocity:        u_t(x, 0) = 0
ic_vel = u0.d(t0)

# Spatial boundary:        u(0, t) = u(1, t) = 0
bc = net(tb, xb)

crux = jno.core([pde.mse, ic_disp.mse, ic_vel.mse, bc.mse])
history = crux.solve(50000)
```

## What To Notice

- Hyperbolic problems are second-order in time, so two initial conditions are needed. The soft pattern handles them as two separate loss terms; the corresponding hard-ansatz trick (`sin(πx) + t² · net(t,x) · x(1−x)`) would absorb both ICs and the BCs into one expression, but it only works for unit-interval Dirichlet problems with this exact shape of IC.
- The velocity IC `u_t(x, 0) = 0` is computed as `net(t0, x0).d(t0)` — `.d()` is happily applied to the *boundary* evaluation; jNO traces the derivative through the network.
- Four competing losses takes longer to converge than the parabolic case. The warmup-cosine schedule above gives the optimiser time to find a balanced minimum before the LR decays.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/wave_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to 04 Hyperbolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/04_hyperbolic/wave_1d.py"
```
