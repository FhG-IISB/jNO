# Heat 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/03_parabolic/heat_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/03-parabolic/">Back to chapter</a>
</div>

Transient 1D heat equation — the introductory time-dependent example. Uses **soft enforcement** for both the initial condition and the spatial boundary conditions, which keeps the IC/BC roles explicit and works on any geometry (not just the unit interval).

## Problem Setup

`u_t = α u_xx` on `(x, t) ∈ [0, 1] × [0, 0.5]`, with `u(0, t) = u(1, t) = 0` (Dirichlet) and `u(x, 0) = sin(π x)` (initial). Exact solution: `u(x, t) = e^{-α π² t} sin(π x)`.

## Step 1: Build a Space-Time Domain

```python
α = 0.1
T_end = 0.5

domain = jno.domain.line(mesh_size=0.01, time=(0, T_end, 10))
x, t   = domain.variable("interior")   # full interior of space-time
x0, t0 = domain.variable("initial")    # t = 0 slice
xb, tb = domain.variable("boundary")   # x = 0 and x = 1 at all t
```

Three tags sampled — one for the PDE residual, one for the IC, one for the BC.

## Step 2: Bare-Network Ansatz

```python
net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1, coord_dim=1, n_outputs=1,
        n_layers=3, basis_functions=64, hidden_dim=32,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 10000, 0.9, end_value=1e-5)))

u = net(t, x)   # no multiplicative ansatz — IC + BC enforced via loss terms below
```

## Step 3: Three Constraints — PDE + IC + BC

```python
# Interior PDE residual:  u_t − α u_xx = 0
pde = u.d(t) - α * u.d2(x)

# Initial condition:  net(0, x) = sin(πx)
ic = net(t0, x0) - jno.np.sin(π * x0)

# Spatial boundary:  net(t, 0) = net(t, 1) = 0
bc = net(tb, xb)

crux = jno.core([pde.mse, ic.mse, bc.mse])
history = crux.solve(10000)
```

## What To Notice

- Each physical condition is its own constraint term — the IC, the BC, and the PDE are three separate scalars that the optimiser balances. There is no ansatz hiding any of them.
- For unit-interval Dirichlet problems a hard ansatz `u = sin(πx) + t · net(t,x) · x(1−x)` would work and remove two of the three losses (see the original `Laplace 1D` for the hard-ansatz pattern). The soft pattern shown here generalises to arbitrary geometries and to PDEs where no clean ansatz exists.
- DeepONet is used here in PINN mode (single instance, no parameter sweep). The branch/trunk split makes it expressive enough to capture both space and time dependence with a small network.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/03_parabolic/heat_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO/tutorials/03-parabolic/">Back to 03 Parabolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/03_parabolic/heat_1d.py"
```
