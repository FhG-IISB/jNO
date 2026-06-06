# Biharmonic 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/01_basics/biharmonic_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/01-basics/">Back to chapter</a>
</div>

This example raises the order of the PDE. Instead of a second derivative, it solves a fourth-order biharmonic problem with clamped boundary conditions.

## Problem Setup

We solve

```text
u''''(x) = 24,   x in [0, 1]
```

with clamped boundary conditions

```text
u(0) = u(1) = 0
u'(0) = u'(1) = 0
```

and exact solution

```text
u(x) = x^2 (1-x)^2
```

## Step 1: Create the Domain

```python
domain = jno.domain.line(mesh_size=pick(0.01, 0.1))
x, _ = domain.variable("interior")
```

The domain setup is the same as the earlier 1D examples.

## Step 2: Define the Exact Solution

```python
u_exact = x**2 * (1 - x) ** 2
```

This exact form is especially useful here because it also suggests a natural hard-constraint ansatz.

## Step 3: Encode the Boundary Conditions Directly

```python
net = jnn.nn.mlp(
    in_features=1,
    hidden_dims=32,
    num_layers=3,
    key=jax.random.PRNGKey(11),
)
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.6, pick(8_000, 10), 1e-5))

u = net(x) * x**2 * (1 - x) ** 2
```

Why this is powerful:

- `x^2(1-x)^2` vanishes at both endpoints.
- Its derivative also vanishes at both endpoints.
- That means the clamped boundary conditions are hard-enforced by construction.

## Step 4: Build the Fourth-Order Residual

```python
u_xxxx = jnn.grad(jnn.grad(jnn.grad(jnn.grad(u, x), x), x), x)
pde = u_xxxx - 24.0
error = jnn.tracker((u - u_exact).mse, interval=pick(200, 1))
```

This is the key learning point of the example: higher-order PDEs can still be expressed compactly when the symbolic operations remain composable.

## Step 5: Solve and Visualize

```python
crux = jno.core([pde.mse, error], domain)
history = crux.solve(pick(8_000, 10), profile=True)
```

The remainder of the script follows the same pattern as the other examples:

- plot training history
- evaluate predictions on sorted points
- compare with the exact solution
- save `solution.png`

## What To Notice

- Hard constraints become even more useful for higher-order problems.
- The symbolic gradient chain stays readable even for fourth derivatives.
- This is a strong template for beam-like or plate-like toy problems.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/01_basics/biharmonic_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/01-basics/">Back to 01 Basics</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/01_basics/biharmonic_1d.py"
```
