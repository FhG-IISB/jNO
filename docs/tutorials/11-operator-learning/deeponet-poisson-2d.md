# DeepONet — parametric Poisson 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/11_operator_learning/deeponet_poisson_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/11-operator-learning/">Back to chapter</a>
</div>

Train a DeepONet to solve a 1-parameter family of Poisson problems via PDE-residual learning. The network sees no ground-truth solutions — only the physics — and learns the operator `k → u(·)` for the entire range `k ∈ [0.5, 1.5]`.

## Problem Setup

```text
k Δu + 1 = 0   on [0, 2] × [0, 1],   u = 0 on ∂Ω,   k ~ U(0.5, 1.5)
```

500 random `k` values are sampled at the start of training; the solver replicates the spatial mesh across all 500 samples and computes the residual for every `(k, x, y)` triple in one forward pass.

## Step 1: Parametric Domain

Multiplying a domain by an integer `B` replicates it across `B` independent samples. This is the operator-learning pattern:

```python
N_SAMPLES = 500
dom = N_SAMPLES * jno.domain(jno.Shape.rect(0, 0, 2, 1, size=0.05))
x, y, _ = dom.variable("interior")

k_values = jax.random.uniform(jax.random.PRNGKey(0), shape=(N_SAMPLES, 1, 1), minval=0.5, maxval=1.5)
k = dom.variable("k", k_values)
```

`k_values` has one scalar `k` per sample; attaching it as a tensor variable on the domain makes it accessible inside the symbolic expression.

## Step 2: DeepONet Network

The branch input is the scalar `k` (a "function evaluated at one sensor"); the trunk input is the query coordinate `(x, y)`. The output is the dot product of the two encoded vectors:

```python
net = jno.nn(
    foundax.deeponet(
        n_sensors=1,         # branch input dimensionality
        coord_dim=2,         # trunk input dimensionality
        basis_functions=32,
        hidden_dim=128,
        activation=jax.numpy.tanh,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(optax.adam(optax.cosine_decay_schedule(1e-3, 20_000, alpha=1e-5 / 1e-3)))
```

## Step 3: Hard BCs + PDE Residual

```python
u = net(k, jno.np.concat([x, y], axis=-1)) * x * (2 - x) * y * (1 - y)
pde = k * (u.d2(x) + u.d2(y)) + 1.0
```

The multiplicative ansatz `x(2-x)y(1-y)` vanishes on all four edges and enforces the homogeneous Dirichlet BC for **every** sample, so the boundary doesn't need a loss term.

## Step 4: Solve

```python
crux = jno.core(constraints=[pde.mse])
crux.solve(epochs=20_000, batchsize=32)
```

`batchsize=32` means each gradient step uses 32 of the 500 parametric samples — a stochastic minibatch in `k`-space.

## What To Notice

- **One network, one training run, 500 PDE solutions.** After convergence, `crux.eval(u)` returns the solution field for every sampled `k` without any retraining.
- **Branch/trunk factorisation** is the operator-learning interpretation of "separation of variables in parameter space". It's cheap to scale (the trunk is the same for all samples), which makes DeepONet much faster than training one PINN per `k`.
- **Pure PDE-residual training.** No solution data is supplied — the network learns from physics alone. Compare with the [FNO](fno-poisson-2d.md) tutorials, which use a precomputed `(f, u)` dataset.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/11_operator_learning/deeponet_poisson_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO/tutorials/11-operator-learning/">Back to 11 Operator Learning</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/11_operator_learning/deeponet_poisson_2d.py"
```
