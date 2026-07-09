# FNO2D — supervised Poisson 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/11_operator_learning/fno_poisson_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/11-operator-learning/">Back to chapter</a>
</div>

Train a Fourier Neural Operator to learn the Poisson solution operator from a precomputed dataset of `(f, u)` pairs. **Data-driven** — no PDE residual is computed at training time; the network is supervised on solution targets.

## Problem Setup

Solve `−∇²u = f` on `[0,1]²` with `u = 0` on the boundary. The forcing `f` is sampled from a truncated sinusoidal basis (`generate_poisson_data` in `create_domain.py`, shipped alongside the tutorial script); the analytic solution `u` is computed in closed form on the same basis.

## Step 1: Generate the Dataset

```python
from create_domain import build_domain_from_arrays, generate_poisson_data

GRID, SAMPLES = 16, 256
forcing, solution = generate_poisson_data(SAMPLES, GRID, n_modes=5, alpha=1.5, seed=42)
domain = build_domain_from_arrays(forcing, solution, GRID)
_f = domain.variable("_f")
_u = domain.variable("_u")
```

`forcing` and `solution` are `(256, 16, 16, 1)` arrays. The domain attaches them as named tensors so the symbolic expression can reference them directly.

## Step 2: FNO2D Architecture

```python
u = jno.nn(
    foundax.fno2d(
        in_features=1,
        hidden_channels=16,
        n_modes=6,            # number of Fourier modes retained per axis
        d_vars=1,
        n_layers=2,
        n_steps=1,
        d_model=(GRID, GRID),
        norm="layer",
        linear_conv=True,
        key=jax.random.PRNGKey(0),
    )
)
u.optimizer(optax.chain(
    optax.clip_by_global_norm(1e-3),
    optax.adamw(optax.cosine_decay_schedule(5e-4, 600, alpha=1e-7 / 5e-4), weight_decay=1e-6),
))
```

Each FNO layer lifts the input to a hidden channel, transforms to Fourier space, mixes the lowest-`n_modes` coefficients via a learnable linear map, transforms back, combines with a residual conv, and normalises. The spectral mixing is what makes FNO efficient on smooth solutions of elliptic PDEs.

## Step 3: Supervised Loss + Solve

```python
crux = jno.core([(_u - u(_f)).mse])
crux.solve(epochs=600, batchsize=32)
```

The loss is a pure supervised MSE between the network's prediction `u(_f)` and the ground-truth solution `_u`. No `.d(x)`, no `.integrate()`, no `.mse` on a residual — it's a regression problem.

## Result

![FNO prediction, ground-truth solution, and pointwise error for a held-out forcing on the unit square](/jNO/assets/fno_poisson_2d.png)

On eight held-out forcings drawn from an unseen RNG seed, the trained FNO reproduces the ground-truth Poisson solutions to a mean rel-$L^2 \approx 0.10$; the representative sample shown above sits at rel-$L^2 \approx 0.099$. All three panels are the operator's own output on data it never trained on.

## What To Notice

- **Data-driven, not physics-driven.** FNO's strength is that it generalises to new forcing functions *without retraining* — once the operator is learnt, evaluating `u(f_new)` for any in-distribution `f_new` is a single forward pass.
- **Spectral mixing is a strong prior** for elliptic / parabolic PDEs whose solutions are smooth in Fourier space. For problems with sharp gradients or discontinuities, the spectral cutoff (`n_modes=6` here) becomes the dominant approximation error.
- **Resolution independence** in principle — the same FNO trained at `GRID=16` can be evaluated at higher resolutions without architecture changes, because Fourier modes are mesh-agnostic. In practice the modes per axis (`n_modes`) needs to be matched to the target resolution.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/11_operator_learning/fno_poisson_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO/tutorials/11-operator-learning/">Back to 11 Operator Learning</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/11_operator_learning/fno_poisson_2d.py:code"
```
