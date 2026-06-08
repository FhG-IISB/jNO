# Differential Operators

`jno.numpy` (importable as `jnn`) is a drop-in NumPy-compatible module that operates on symbolic `Placeholder` expressions. It provides automatic and finite-difference differential operators for formulating PDE residuals, plus the full NumPy math library for use inside constraints.

```python
import jno.numpy as jnn
```

---

## Differentiation Schemes

Pass any scheme string via the `scheme=` keyword on `jnn.grad`, `jnn.laplacian`, `jnn.hessian`, etc.

### All scheme strings

| Scheme string | Grad | Lap | Hessian | 1D | 2D | 3D | Temporal | Notes |
|---------------|:----:|:---:|:-------:|:--:|:--:|:--:|:--------:|-------|
| `"automatic_differentiation"` *(default)* | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Exact; any domain |
| `"automatic_differentiation:forward"` | ✅ | — | — | ✅ | ✅ | ✅ | ✅ | `jacfwd`; best when outputs ≫ inputs |
| `"automatic_differentiation:reverse"` | ✅ | — | — | ✅ | ✅ | ✅ | ✅ | `jacrev`; best when inputs ≫ outputs |
| `"automatic_differentiation:fwd-over-rev"` | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `jacfwd∘jacrev`; default Hessian |
| `"automatic_differentiation:fwd-over-fwd"` | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Fastest for many-input problems |
| `"automatic_differentiation:rev-over-rev"` | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Low memory |
| `"automatic_differentiation:rev-over-fwd"` | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `"finite_difference"` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Area-weighted; general unstructured meshes |
| `"finite_difference:lsq"` | ✅ | ✅ | — | ✅ | ✅ | ✅ | — | Least-squares; more accurate on irregular meshes |
| `"finite_difference:uniform"` | ✅ | ✅ | — | ✅ | ✅ | ✅ | — | Equal neighbour weights; fast on near-uniform meshes |
| `"finite_difference:inverse_distance"` | ✅ | ✅ | — | ✅ | ✅ | ✅ | — | Distance-weighted; better on highly stretched meshes |
| `"finite_difference:cotangent"` | — | ✅ | — | — | ✅ | — | — | Cotangent-weighted Laplacian; **2D only**; most accurate on triangulations |

FD schemes require `compute_mesh_connectivity=True`.

---

### Automatic Differentiation

Uses JAX `jax.jacrev` / `jax.jacfwd` — exact to machine precision on any domain.

```python
u_x  = jnn.grad(u, x)                             # reverse mode (default)
u_x  = jnn.grad(u, x, scheme="automatic_differentiation:forward")
lap  = jnn.laplacian(u, [x, y])                    # fwd-over-rev (default)
lap  = jnn.laplacian(u, [x, y], scheme="automatic_differentiation:fwd-over-fwd")
```

Set a project-wide default with `jno.setup(__file__, diff_type="forward", hessian_type="fwd-over-fwd")`.

---

### Finite Difference

Mesh-based; approximates derivatives using element stencils. Requires `compute_mesh_connectivity=True` on the domain.

```python
u_x = jnn.grad(u, x, scheme="finite_difference")                   # default sub-scheme
u_x = jnn.grad(u, x, scheme="finite_difference:lsq")               # least-squares
lap = jnn.laplacian(u, [x, y], scheme="finite_difference:cotangent")
```

---

## Differentiation

### Gradient (first derivative)

```python
# ∂u/∂x  — automatic differentiation (default)
u_x = jnn.grad(u, x)

# ∂u/∂x  — mesh-based finite differences
u_x = jnn.grad(u, x, scheme="finite_difference")
```

`jnn.grad` returns a `Jacobian` placeholder.

### Laplacian

```python
# ∇²u = ∂²u/∂x² + ∂²u/∂y²  — automatic differentiation (default)
lap = jnn.laplacian(u, [x, y])

# Finite-difference Laplacian (uses mesh connectivity — requires compute_mesh_connectivity=True)
lap = jnn.laplacian(u, [x, y], scheme="finite_difference")
```

`jnn.laplace` is an alias for `jnn.laplacian`.

### Jacobian (vector-valued)

```python
# J = [∂u/∂x, ∂u/∂y]
J = jnn.jacobian(u, [x, y])
```

### Hessian matrix

```python
# H[i,j] = ∂²u/∂xᵢ∂xⱼ
H = jnn.hessian(u, [x, y])
```

### Divergence

```python
# ∇·F = ∂Fx/∂x + ∂Fy/∂y
div_F = jnn.divergence([Fx, Fy], [x, y])
```

### Curl

```python
# 2D curl (scalar): ∂Fy/∂x − ∂Fx/∂y
curl = jnn.curl_2d(Fx, Fy, x, y)

# 3D curl (vector): [∂Fz/∂y−∂Fy/∂z, ∂Fx/∂z−∂Fz/∂x, ∂Fy/∂x−∂Fx/∂y]
curl_vec = jnn.curl_3d(Fx, Fy, Fz, x, y, z)
```

---

## Parameter Jacobian

`u.grad(net)` computes the Jacobian of a traced expression with respect to the **trainable parameters** of a network — as opposed to the spatial derivatives above, which differentiate with respect to input coordinates.

```python
# Full parameter Jacobian  ∂u/∂θ  — shape (B, N, P)
J_expr = u.grad(u_net)
[J] = crux.eval([J_expr])   # (B, N, P);  J[0] gives (N, P)
```

| Symbol | Shape | Meaning |
|--------|-------|---------|
| $B$ | batch | number of instances (usually 1 for a single solve) |
| $N$ | points | collocation points at which $u$ is evaluated |
| $P$ | params | number of selected trainable parameters (flattened) |

For multi-dimensional output the shape is `(B, N, D, P)`.

### Selecting a subset of parameters

For large networks, computing the full Jacobian over all $P$ parameters can be expensive. Restrict it to a subset using a boolean pytree built with `equinox.tree_at` and passed via `net.mask(...)`:

```python
import equinox as eqx, jax

# Boolean mask — True only for the output-layer weight matrix
all_false   = jax.tree_util.tree_map(lambda _: False, u_net.module)
output_mask = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)

[J_sparse] = crux.eval([u.grad(u_net.mask(output_mask))])  # (B, N, P_weight)
```

Multiple tensors can be selected at once:

```python
wb_mask = eqx.tree_at(
    lambda m: (m.output_layer.weight, m.output_layer.bias),
    all_false, (True, True),
)
[J_wb] = crux.eval([u.grad(u_net.mask(wb_mask))])
```

### Neural Tangent Kernel

The NTK is the Gram matrix $K = J J^T$ and is easy to compute once $J$ is available:

```python
J = J[0]            # (N, P) — strip batch dim
K = J @ J.T         # (N, N) Neural Tangent Kernel
```

$K_{ij}$ measures the correlation between the parameter-space sensitivities of points $i$ and $j$. Its eigenspectrum determines the learning dynamics: large eigenvalues correspond to fast-converging spatial modes, small eigenvalues to slow ones.

### Gradient cosine similarity

Compare how two groups of points interact during training by projecting their Jacobians onto a single direction and measuring alignment:

```python
g_A = J[group_A].mean(axis=0)   # aggregate sensitivity direction for group A
g_B = J[group_B].mean(axis=0)

cos_sim = float(
    jnp.dot(g_A, g_B) / (jnp.linalg.norm(g_A) * jnp.linalg.norm(g_B))
)
```

A value near +1 means both groups reinforce the same parameter updates; near -1 means they conflict.

### Using the Jacobian in a training loss

`u.grad(net)` can be included directly in a training constraint — for example to encourage the NTK to be well-conditioned — but this requires differentiating *through* `jax.jacrev`, which is second-order AD and expensive. Wrap in `.stop_gradient()` when you only want the current Jacobian as a constant regulariser:

```python
J_sg = u.grad(u_net).stop_gradient()        # J treated as a constant
ntk_reg = (J_sg @ J_sg.T - target_K).mse   # penalise NTK shape cheaply
```

See the [Gradient Conflict Analysis tutorial](tutorials/07-analysis/gradient-conflict.md) for a complete worked example.
