# HyCo: Hybrid-Cooperative PINN

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/05_coupled_and_inverse/hyco_poisson_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/05-coupled-and-inverse/">Back to chapter</a>
</div>

This tutorial implements the **Hybrid-Cooperative (HyCo)** learning framework from [Liverani, Steynberg & Zuazua (2025)](https://arxiv.org/abs/2509.14123) using `jno.fn.stop_gradient`.

---

## The Idea

A standard PINN must balance two potentially conflicting objectives: satisfying the PDE and fitting observations. HyCo instead trains **two specialised networks in parallel**:

| Model | Objective |
|-------|-----------|
| `u_phy` — physical model | Enforce the PDE residual |
| `u_syn` — synthetic model | Fit the sparse, noisy observations |

The two models are kept in sync through a **mutual interaction loss** evaluated at the interior collocation points. `jno.fn.stop_gradient` is the key ingredient: it ensures each interaction term only updates the *student* model's parameters, leaving the *reference* model's weights frozen for that gradient step.

---

## Loss Decomposition

$$
\mathcal{L} =
\underbrace{\mathcal{L}_\text{pde}}_{\nabla \to u_\text{phy}}
+\,\beta\,\underbrace{\mathcal{L}_\text{int}^\text{phy}}_{\nabla \to u_\text{phy}}
+\,\alpha\,\underbrace{\mathcal{L}_\text{data}}_{\nabla \to u_\text{syn}}
+\,\beta\,\underbrace{\mathcal{L}_\text{int}^\text{syn}}_{\nabla \to u_\text{syn}}
$$

where

$$
\mathcal{L}_\text{pde} = \|\mathcal{N}[u_\text{phy}]\|^2, \qquad
\mathcal{L}_\text{data} = \|u_\text{syn}(x^\text{obs}) - y^\text{obs}\|^2
$$

$$
\mathcal{L}_\text{int}^\text{phy} = \|u_\text{phy} - \operatorname{sg}(u_\text{syn})\|^2, \qquad
\mathcal{L}_\text{int}^\text{syn} = \|u_\text{syn} - \operatorname{sg}(u_\text{phy})\|^2
$$

Here $\operatorname{sg}(\cdot)$ denotes `stop_gradient`.  All four terms live in the **same** `jno.core` call — stop-gradient does the work of keeping the gradient paths separate.

---

## Problem

1D Poisson on $[0, 1]$:

$$u'' + \pi^2 \sin(\pi x) = 0, \qquad u(0) = u(1) = 0, \qquad u(x) = \sin(\pi x)$$

**Observations**: 7 randomly placed sensors with additive Gaussian noise ($\sigma = 0.05$).

---

## Setup

### Domains

A standard line domain provides the collocation points. Sensor coordinates are registered on the same domain object by passing the array directly to `variable`:

```python
# Dense collocation grid — PDE residual + interaction
domain = jno.domain.line(mesh_size=0.02)
x, _ = domain.variable("interior")

# Sparse noisy sensors — registered on the same domain
x_sen = np.linspace(0.1, 0.9, 7).reshape(-1, 1)
u_sen = np.sin(np.pi * x_sen) + rng.normal(0, 0.05, x_sen.shape)

(x_s,) = domain.variable("sensors", sample=x_sen, split=True, point_data=True)
```

### Networks

```python
u_phy_net = jno.nn.wrap(foundax.mlp(in_features=1, output_dim=1, hidden_dims=32, num_layers=3, key=k1))
u_syn_net = jno.nn.wrap(foundax.mlp(in_features=1, output_dim=1, hidden_dims=32, num_layers=3, key=k2))
for net in [u_phy_net, u_syn_net]:
    net.optimizer(optax.adam(1e-3))

u_phy = u_phy_net(x) * x * (1 - x)   # hard zero BCs
u_syn = u_syn_net(x) * x * (1 - x)
```

---

## The Four Loss Terms

```python
# Physical model: PDE residual
L_pde = (u_phy.dd(x) + π**2 * jno.np.sin(π * x)).mse

# Synthetic model: fit the noisy observations
u_syn_s = u_syn_net(x_s) * x_s * (1 - x_s)
u_obs   = jno.np.array(u_sen)
L_data  = (u_syn_s - u_obs).mse

# Mutual alignment — stop_gradient prevents cross-model gradient flow
L_int_phy = (u_phy - jno.fn.stop_gradient(u_syn)).mse
L_int_syn = (u_syn - jno.fn.stop_gradient(u_phy)).mse
```

### Why stop_gradient?

Without it, `L_int_phy` would backpropagate through *both* `u_phy` **and** `u_syn`, turning the interaction into a confusing cross-model gradient signal. With `stop_gradient`:

- Gradients of `L_int_phy` reach only `u_phy_net` — it is nudged toward `u_syn`'s predictions.
- Gradients of `L_int_syn` reach only `u_syn_net` — it is nudged toward `u_phy`'s predictions.

---

## Alternating updates via `substeps`

HyCo prescribes **alternating** updates — `u_phy` is updated first, then `u_syn` is updated using the *freshly updated* `u_phy`. Updating both simultaneously in a single optimizer step lets each model only see a stale snapshot of the other, which slows convergence and contaminates Adam momentum.

The `substeps` argument to `solve()` expresses the alternating schedule:

```python
α, β = 1.0, 1.0

crux = jno.core(
    [L_pde, β * L_int_phy, α * L_data, β * L_int_syn], 
)
# Each outer epoch runs two gradient steps in sequence:
#   substep 0: constraints [0, 1] → only u_phy_net updates
#   substep 1: constraints [2, 3] → only u_syn_net updates (sees fresh u_phy)
crux.solve(3_000, substeps=[[0, 1], [2, 3]])
```

Each substep keeps its **own** optimizer state, so Adam momentum for `u_phy` accumulates only from substep 0's gradients and never decays from being held inactive during substep 1. The shared `trainable` dict carries parameter updates between substeps, so substep 1 sees the freshly written `u_phy` weights.

You can also run multiple gradient steps per substep before alternating:

```python
crux.solve(1_500, substeps=[([0, 1], 2), ([2, 3], 2)])
# 1500 outer epochs × (2 phy + 2 syn) = 6000 effective gradient steps
```

The shorthand `[0, 1]` is equivalent to `([0, 1], 1)`. Within a single substep the `n` repeated steps share the same optimizer state so Adam momentum builds up continuously before the switch.

---

## Results

```
u_phy rel-L2 error : 3.7e-04  (physics model)
u_syn rel-L2 error : 5.4e-02  (synthetic model)
```

The physics model, guided by both the PDE and alignment with the data-fitted synthetic model, reaches near-exact accuracy.  The synthetic model, guided by 7 noisy observations and alignment with the physics model, settles on a smooth physically consistent solution — far better than overfitting to the raw data alone.

---

## What To Notice

- `jno.fn.stop_gradient` is the single syntactic addition that turns a standard two-model PINN into a cooperative system.
- `substeps=[[0, 1], [2, 3]]` expresses the alternating schedule HyCo requires — no manual outer loop needed.
- `jno.domain.from_array` with multiple tags keeps collocation and sensor points in the same domain object.
- Tune `α` and `β` to control how strongly each model is pulled toward the other.

---

## Going Further

- Replace the dense collocation grid with adaptive resampling — see [Adaptive Resampling](../../adaptive/resampling.md).
- Use the synthetic model as a **warm-start** for a harder PDE where the PINN struggles to find the solution from scratch.
- Extend to **2D** by replacing the 1D collocation grid with a 2D rect domain and the BC factor with $x(1-x)y(1-y)$.
- See [Liverani et al. (2025)](https://arxiv.org/abs/2509.14123) for analysis of the Gray-Scott and Helmholtz cases.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/05_coupled_and_inverse/hyco_poisson_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO/tutorials/05-coupled-and-inverse/">Back to 05 Coupled and Inverse</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/05_coupled_and_inverse/hyco_poisson_1d.py"
```
