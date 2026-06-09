# Biharmonic 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/01_basics/biharmonic_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/01-basics/">Back to chapter</a>
</div>

Raises the PDE order. Solves a fourth-order biharmonic equation with **clamped** boundary conditions (both `u` and `u'` vanish at the endpoints) using a hard-enforced ansatz.

## Problem Setup

```text
u''''(x) = sin(π x),   x in [0, 1]

u(0) = u(1) = 0
u'(0) = u'(1) = 0
```

Exact solution: `u(x) = sin(π x) / π⁴`.

## Step 1: Create the Domain

```python
domain = jno.domain.line(mesh_size=0.05)
x, _ = domain.variable("interior")
```

## Step 2: Hard-Enforce the Clamped Boundary Conditions

```python
net = jno.nn.wrap(
    foundax.mlp(in_features=1, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(11))
)
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 10, 0.6, end_value=1e-5)))

u = net(x) * x**2 * (1 - x) ** 2
```

The `x²(1−x)²` factor and **its first derivative** both vanish at `x=0` and `x=1`, so all four clamped boundary conditions are enforced exactly by construction. The network only has to learn the *interior* shape that, when multiplied by `x²(1−x)²`, gives `sin(π x) / π⁴`.

## Step 3: Build the Fourth-Order Residual

```python
u_xxxx = u.d2(x).d2(x)
pde = u_xxxx - jno.np.sin(π * x)
```

`.d2(x).d2(x)` is the fourth derivative via two stacked second-derivative shortcuts — equivalent to `u.d(x).d(x).d(x).d(x)` but more compact, and far cleaner than four nested `jno.np.grad(...)` calls.

## Step 4: Solve

```python
crux = jno.core([pde.mse])
history = crux.solve(5000)
```

## What To Notice

- The clamped-BC ansatz `net(x) · x²(1−x)²` is genuinely doing work here — the exact solution `sin(πx)/π⁴` is **not** equal to the ansatz, so the network has to learn a non-trivial correction.
- Higher-order PDEs stay readable when you use the `.d2()` shortcut. Four nested `jno.np.grad(...)` calls would obscure the structure.
- Fourth-order accuracy is harder than second-order — expect to need a smaller mesh and longer training than `Laplace 1D` or `Poisson 1D`.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/01_basics/biharmonic_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/01-basics/">Back to 01 Basics</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/01_basics/biharmonic_1d.py"
```
