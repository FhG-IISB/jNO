# Gradient and Sensitivity Analysis

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/07_analysis/gradient_conflict.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/07-analysis/">Back to chapter</a>
</div>

This tutorial shows how to use `u.grad(net)` — the **parameter Jacobian** — to monitor what a PINN is learning *during* training. The key technique is computing the **cosine similarity** between domain regions as a `.tracker()`, so you can spot gradient conflict before the solve finishes.

## Concepts

### The Parameter Jacobian

Every trained network defines a mapping $\mathbf{u}(\mathbf{x}; \theta)$ from spatial points to outputs.
The parameter Jacobian is:

$$J_{i,p} = \frac{\partial u(\mathbf{x}_i)}{\partial \theta_p}$$

`u.grad(net)` returns a symbolic `NetworkGradient` expression. It is traced just like any other `Placeholder` — you can pass it to `jnn.function`, include it in a tracker, or evaluate it with `crux.eval`.

### Gradient Cosine Similarity

To compare how two groups of collocation points interact during training, compress each group's Jacobian rows into a single **sensitivity direction**:

$$\mathbf{g}_A = \frac{1}{|A|}\sum_{i \in A} J_i, \qquad \mathbf{g}_B = \frac{1}{|B|}\sum_{i \in B} J_i$$

Then compute the cosine similarity between $\mathbf{g}_A$ and $\mathbf{g}_B$:

$$\text{cos\_sim}(\mathbf{g}_A, \mathbf{g}_B) = \frac{\mathbf{g}_A \cdot \mathbf{g}_B}{\|\mathbf{g}_A\|\,\|\mathbf{g}_B\|}$$

| Value | Meaning |
|-------|---------|
| $\approx +1$ | Aligned — learning from group $A$ also helps group $B$ |
| $\approx 0$ | Orthogonal — the two groups are independent |
| $\approx -1$ | Conflict — improving group $A$ hurts group $B$ |

### Why use a sparse mask?

Computing the full Jacobian over all $P$ parameters is expensive. For in-training monitoring you only need a signal, not the exact answer. Restricting to the **output-layer weights** gives the dominant gradient directions at a fraction of the cost.

`net.mask(bool_pytree)` stores the selection; the next call to `u.grad(net.mask(...))` reads it:

```python
all_false   = jax.tree_util.tree_map(lambda _: False, u_net.module)
output_mask = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)

J = u.grad(u_net.mask(output_mask))   # traced; shape (N, P_out) at eval time
```

---

## Problem Setup

We solve the familiar 1D Poisson equation:

$$-u''(x) = \sin(\pi x), \quad x \in [0, 1], \quad u(0) = u(1) = 0$$

Exact solution: $u(x) = \sin(\pi x) / \pi^2$.

```python
domain = jno.domain.line(mesh_size=0.1)
x, _ = domain.variable("interior")

u_net = jno.nn(
    foundax.mlp(in_features=1, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(0))
).optimizer(optax.adam(optax.exponential_decay(1e-3, 10, 0.5, end_value=1e-5)))

u    = u_net(x) * x * (1 - x)        # hard BC: u(0) = u(1) = 0
u_xx = u.d2(x)
pde  = -u_xx - jno.np.sin(π * x)
```

---

## Step 1: Build the Sparse Mask

```python
import equinox as eqx, jax

all_false   = jax.tree_util.tree_map(lambda _: False, u_net.module)
output_mask = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)

# Symbolic Jacobian — only the output-layer weight, shape (N, P_out_weight)
J = u.grad(u_net.mask(output_mask))
```

`J` is a symbolic `NetworkGradient` node. Nothing is computed yet — it just records the network and the mask.

---

## Step 2: Define the Cosine Similarity as a Tracker

Wrap the cosine similarity calculation in a plain JAX function and use `jnn.function` to lift it into the symbolic graph, then attach it as a non-loss tracker:

```python
def _cos_sim_halves(J):
    mid = J.shape[0] // 2
    g_left  = J[:mid].mean(axis=0)
    g_right = J[mid:].mean(axis=0)
    denom = jnp.linalg.norm(g_left) * jnp.linalg.norm(g_right) + 1e-12
    return jnp.dot(g_left, g_right) / denom

cos_tracker = jno.np.function(_cos_sim_halves, [J]).tracker(200)
```

`.tracker(200)` means it is evaluated and logged every 200 epochs without contributing to the gradient.

---

## Step 3: Solve

```python
crux = jno.core([pde.mse, cos_tracker])
crux.solve(5000)
```

During training the log will show the cosine similarity alongside the PDE loss. For a smooth, symmetric solution like $\sin(\pi x)/\pi^2$ you expect the value to stay positive (ideally $>0.5$) throughout — meaning both halves of the domain reinforce the same parameter updates.

A value that drops toward zero or turns negative during training is a warning: the network is beginning to represent the two halves in nearly-orthogonal (or conflicting) parts of parameter space.

!!! note "What J measures during training"
    `u.grad(net)` gives $\partial u / \partial \theta$, the Jacobian of the **network output** w.r.t. parameters.
    For PDE losses that penalize derivatives of $u$, the cosine similarity gives a correct picture of *output-level* sensitivity.  The true loss gradient involves additional terms from differentiating through spatial derivatives, so treat this as a diagnostic signal rather than an exact measure of gradient conflict.

---

## Step 4: Post-training Analysis

After solving, evaluate the Jacobian directly. When `crux.eval` receives a **single expression**, it returns the raw array without a batch dimension:

```python
[J_sparse] = crux.eval([J])          # (N, P_out_weight) — no batch dim
N, P = J_sparse.shape
```

Compute the final cosine similarity:

```python
g_left  = J_sparse[:N // 2].mean(axis=0)
g_right = J_sparse[N // 2:].mean(axis=0)
cos_sim = float(
    jnp.dot(g_left, g_right)
    / (jnp.linalg.norm(g_left) * jnp.linalg.norm(g_right) + 1e-12)
)
print(f"cos_sim (left vs right) = {cos_sim:.4f}")
```

---

## Step 5: Neural Tangent Kernel (Full Jacobian)

For a deeper analysis, evaluate the **full** Jacobian (all parameters) after training:

```python
[J_full] = crux.eval([u.grad(u_net)])   # (N, P_total)

K       = J_full @ J_full.T
eigvals = jnp.sort(jnp.linalg.eigvalsh(K))[::-1]

eff_rank = float(jnp.sum(eigvals)**2 / (jnp.sum(eigvals**2) + 1e-12))
cond     = float(eigvals[0] / (eigvals[-1] + 1e-12))

print(f"Effective rank = {eff_rank:.2f}")
print(f"Condition number = {cond:.1f}")
```

The **effective rank** (participation ratio) tells you how many independent learning modes the network uses. The NTK condition number measures how uniformly different spatial patterns are learned: a high condition number means some patterns converge much slower than others.

---

## Step 6: Scale Analysis — Units and Non-Dimensionalization

Gradient conflict has a twin: **scale conflict**. When the additive terms of a *single* residual differ in magnitude by orders, the loss is ill-conditioned no matter how well the collocation points align. `jno.units` exposes that structure: annotate the coordinates and the field with `.unit(...)` / `.scale(...)`, and it audits dimensional consistency and reports — then rewrites away — the dimensionless group each term carries (the Fourier / Péclet-type numbers you would otherwise derive by hand).

Take a **thin, anisotropic** domain ($L_x = 1$, $L_y = \tfrac{1}{20}$). Geometry *alone* drives the two diffusion terms of the Laplacian $u_{xx} + u_{yy}$ to very different scales — no material coefficient required:

```python
Lx, Ly, U = 1.0, 0.05, 3.0
adom = jno.Shape.rect(0.0, 0.0, Lx, Ly, size=0.1).domain()
ax, ay, _ = adom.variable("interior", split=True)
ax = ax.unit("m").scale(Lx)          # characteristic length along x
ay = ay.unit("m").scale(Ly)          # 20× shorter characteristic length along y
au = jno.nn(foundax.mlp(in_features=2, hidden_dims=8, num_layers=2, key=jax.random.PRNGKey(1)))(ax, ay)
au = au.unit("K").scale(U)           # the field carries a temperature scale U
aniso = au.d2(ax) + au.d2(ay)        # anisotropic Laplacian — two terms in ONE residual
```

**Phase A — audit and report.** `jno.units.check` confirms both terms share a unit ($\text{K}\cdot\text{m}^{-2}$); `jno.units.nondimensionalize` gives each term's dimensionless magnitude $\pi_i = S_i / S_\text{ref}$:

```python
assert not jno.units.check(aniso).warnings            # dimensionally consistent
terms = jno.units.nondimensionalize(aniso).residuals[0].terms
scale_sep = terms[1].pi / terms[0].pi                 # → 400.0  = (Lx/Ly)²
```

The two terms differ by **400×** — exactly $(L_x/L_y)^2$. That is the "losses at different scales" that stalls plain gradient descent, surfaced *before* you ever train.

**Phase B — the transform.** `jno.units.rescale` rewrites the residual into its $O(1)$ dimensionless form: the hidden 400× separation resurfaces as an *explicit* leading coefficient, and the returned `Rescaler` maps coordinates onto the unit domain and a solution back to physical units ($u_\text{physical} = U \cdot \hat u$):

```python
transformed, rescaler = jno.units.rescale(aniso)
rdom = rescaler.rescaled_domain(adom)          # same problem, coordinates rescaled to O(1)
# ...  jno.core([transformed.mse], domain=rdom).solve(...)  # train the well-scaled problem
u_physical = rescaler.to_physical(u_hat)       # map the O(1) field back to physical units
```

!!! note "What `jno.units` operates on"
    `nondimensionalize` / `rescale` act on the **additive terms within one residual** ($\pi_i = S_i / S_\text{ref}$) — they extract the Fourier / Péclet numbers, not a ratio between two *separate* losses. Today only coordinates and the network output are annotatable through the public API; a bare material coefficient (e.g. a diffusivity $\alpha$) has no public `.unit` hook yet, so the demonstrated scale separation is purely geometric.

---

## Result

![Left: the trained network's solution lies on the exact sin(πx)/π² curve (rel L² ≈ 3×10⁻⁴). Right: the Neural Tangent Kernel eigenvalue spectrum decays over ~14 orders of magnitude, an effective rank near 1 — the ill-conditioning that makes some spatial modes converge far slower than others.](/jNO/assets/gradient_conflict.png)

The trained network matches the analytic $\sin(\pi x)/\pi^2$ to rel-$L^2\approx3\times10^{-4}$ (left). The **NTK eigenvalue spectrum** (right) is the model's own $K=JJ^\top$ after training: it collapses over ~14 orders of magnitude to an effective rank near 1, so a handful of modes dominate learning while the rest are almost frozen — the quantitative face of the gradient conflict.

---

## What To Notice

- The cosine similarity **during training** lets you catch gradient conflict early — long before the loss plateaus.
- A sparse mask (output layer only) makes the tracker cheap enough to run every 200 epochs even on large networks.
- `crux.eval([single_expr])` returns the array without a batch dimension; you don't need to strip a leading `[0]`.
- A low effective rank (close to 1) means the network is learning in a near-1D subspace — widen the network or increase collocation points.
- Cosine similarity $< 0.3$ between two groups that should behave similarly is a warning: consider adding more collocation points or using adaptive resampling.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/07_analysis/gradient_conflict.py" download>Download full script</a>
<a class="md-button" href="/jNO/tutorials/07-analysis/">Back to 07 Analysis</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/07_analysis/gradient_conflict.py:code"
```
