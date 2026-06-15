# Surrogate Inversion

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/05_coupled_and_inverse/surrogate_inversion.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/05-coupled-and-inverse/">Back to chapter</a>
</div>

This tutorial shows how to flip a trained PINN into an inverse solver by turning its **input** into a trainable parameter.

---

## The Idea

After training a forward PINN you have a differentiable surrogate $u_\theta : x \mapsto u(x)$.
The usual direction is *forward*: given $x$, predict $u$.

For *inversion*, we want to go the other way: given an observation $u_\text{obs}$, find $x^*$ such that

$$u_\theta(x^*) \approx u_\text{obs}.$$

Because the surrogate is differentiable end-to-end, we can simply declare the query input as a `jno.np.parameter` and minimise the residual with gradient descent — no new machinery needed.

---

## Problem

1D Poisson on $[0,1]$:

$$u'' + \pi^2 \sin(\pi x) = 0, \qquad u(0) = u(1) = 0$$

Exact solution: $u(x) = \sin(\pi x)$.

**Inverse task**: given the measurement $u_\text{obs} = \sin(\pi \cdot 0.3) \approx 0.809$, recover $x^* = 0.3$.

---

## Phase 1 — Forward Solve

Train a PINN in the usual way:

```python
domain = jno.domain.line(mesh_size=0.01)
x, _ = domain.variable("interior")

u_net = jno.nn.wrap(foundax.mlp(in_features=1, output_dim=1, hidden_dims=32, num_layers=3, key=k1))
u_net.optimizer(optax.adam(1e-3))

u = u_net(x) * x * (1 - x)                  # hard zero Dirichlet BCs
pde = u.dd(x) + π**2 * jno.np.sin(π * x)    # PDE residual

crux_fwd = jno.core([pde.mse])
crux_fwd.solve(3_000)
```

The network now acts as a cheap, differentiable surrogate of the Poisson solution.

---

## Phase 2 — Inversion

### Step 1: Freeze the Surrogate

```python
u_net.freeze()
```

The weights are now fixed. We will only optimise the query input.

### Step 2: Declare the Input as a Parameter

```python
x_query = jno.np.parameter((1,), name="x_query")
x_query.initialize(jax.nn.initializers.constant(0.1))  # initial guess
x_query.optimizer(optax.adam(5e-3))
```

`x_query` is a scalar trainable variable — exactly like a PDE coefficient in `inverse_parameter.py`, but here it is fed as the *network input*.

### Step 3: Build the Inverse Loss

```python
# Apply the same BC factor that was used during training
u_at_query = u_net(x_query) * x_query * (1 - x_query)

inv_loss = (u_at_query - u_obs) ** 2
```

### Step 4: Solve on a Single-Point Domain

The loss has no spatial dependence on a mesh — it is purely a function of `x_query`. Use `jno.domain.from_array` to create a minimal one-point domain:

```python
import numpy as np

inv_domain = jno.domain.from_array({"pt": np.zeros((1, 1))})
crux_inv = jno.core([inv_loss.mean])
crux_inv.solve(500)
```

### Step 5: Read Back the Result

```python
_x_q = crux_inv.eval([x_query])[0]
print(f"Recovered x = {float(_x_q):.4f}")   # → 0.3000
```

---

## What To Notice

- `u_net.freeze()` prevents the surrogate weights from changing during inversion.
- Swapping the *spatial variable* for a `jno.np.parameter` is the only change needed to go from forward to inverse mode.
- A single-point domain (`jno.domain.from_array`) is sufficient when the loss has no spatial structure.
- The inversion is gradient-based, so initialisation matters. For functions with multiple pre-images (e.g. $\sin$), initialise near the expected region.

---

## Going Further

- Add multiple observations at different locations to make the inverse problem uniquely determined.
- Combine with a **field-level inverse problem** (recovering a coefficient $k(x)$) — see the [Inverse Problems guide](../../inverse-problems.md).
- Extend to 2-D: the query becomes `jno.np.parameter((2,))` and the frozen 2D PINN evaluates at that point.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/05_coupled_and_inverse/surrogate_inversion.py" download>Download full script</a>
<a class="md-button" href="/jNO/tutorials/05-coupled-and-inverse/">Back to 05 Coupled and Inverse</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/05_coupled_and_inverse/surrogate_inversion.py"
```
