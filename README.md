<p align="center">
  <img src="assets/logo.png" alt="jNO logo" width="500"/>
</p>

<p align="center">
    <a href="https://fhg-iisb.github.io/jNO/">
        <img src="https://img.shields.io/badge/docs-GitHub%20Pages-0aa?style=for-the-badge" alt="Dev Docs"/>
    </a>
    <a href="https://github.com/FhG-IISB/jno/actions/workflows/ci.yml">
        <img src="https://img.shields.io/github/actions/workflow/status/FhG-IISB/jno/ci.yml?branch=main&style=for-the-badge&label=tests" alt="Tests"/>
    </a>
    <a href="LICENSE">
        <img src="https://img.shields.io/badge/license-EPL--2.0-2ea44f?style=for-the-badge" alt="License"/>
    </a>
    <a href="CITATION.cff">
        <img src="https://img.shields.io/badge/cite-CITATION.cff-6b5b95?style=for-the-badge" alt="Citation"/>
    </a>
    <img src="https://img.shields.io/badge/docker-image%20available-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker image available"/>
</p>

Warning: This is a research-level repository. It may contain bugs and is subject to continuous change without notice.


# Install

Quick install from PyPI:

```bash
pip install jax-neural-operators
```


Foundation models and other neural operators  are maintained in a separate repository ([foundax](https://github.com/FhG-IISB/foundax)) so they can also be used independently (foundax is installed automatically with this repository).

# Example

```python
import jno
import jax
import jax.numpy as jnp
import optax
import foundax
import equinox as eqx
from jno import LearningRateSchedule as lrs
from jno.numpy import tracker

dir = jno.setup("./runs/poisson2d")

# ── Domain — rect with named boundary sides ────────────────────────────────────
dom        = jno.domain(constructor=jno.domain.rect(mesh_size=0.05, x_range=(0, 1), y_range=(0, 1)))
x,  y,  _  = dom.variable("interior")
xl, yl, _  = dom.variable("left")    # x = 0  →  soft Dirichlet

# ── Network — LoRA adapters on hidden layers ────────────────────────────────────
net = jno.nn.wrap(
    foundax.mlp(in_features=2, hidden_dims=64, num_layers=4,
                activation=jnp.tanh, key=jax.random.PRNGKey(0))
)
net.lora(rank=4, alpha=1.0, target="hidden_layers")  # parameter-efficient training
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 5_000, 1e-5))

# ── Forward pass — hard BCs on right (x=1), bottom (y=0), top (y=1) ───────────
π  = jno.np.pi
u  = net(jno.np.concat([x,  y ], axis=-1)) * (1 - x)  * y  * (1 - y)
ul = net(jno.np.concat([xl, yl], axis=-1)) * (1 - xl) * yl * (1 - yl)

# ── PDE: −∇²u = 2π²sin(πx)sin(πy),  exact u* = sin(πx)sin(πy) ───────────────
pde     = -(u.dd(x) + u.dd(y)) - 2 * π**2 * jno.np.sin(π * x) * jno.np.sin(π * y)
bc_left = ul                                          # soft: u(0, y) = 0

# ── Integral: ∫u dΩ → 4/π² ≈ 0.405 for the exact solution ────────────────────
vol_tracker = tracker(u.integrate(), interval=500)

# ── Gradient alignment — PDE vs. left-BC loss, output-layer params only ───────
all_false  = jax.tree_util.tree_map(lambda _: False, net.module)
out_mask   = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)
J_pde      = pde.mse.grad(net.mask(out_mask))         # ∂L_pde / ∂θ_out
J_bc       = bc_left.mse.grad(net.mask(out_mask))     # ∂L_bc  / ∂θ_out
grad_align = tracker(jno.np.dot(J_pde, J_bc), interval=500)

# ── Solve ──────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, bc_left.mse, vol_tracker, grad_align], domain=dom)
crux.solve(20_000).plot(f"{dir}/training.png")
jno.save(crux, f"{dir}/model.pkl")
```

