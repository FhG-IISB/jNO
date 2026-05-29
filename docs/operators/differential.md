# Differential Operators

`jno.numpy` (importable as `jnn`) provides automatic and finite-difference spatial operators for formulating PDE residuals.

```python
import jno.numpy as jnn
```

---

## Gradient (first derivative)

```python
# ∂u/∂x  — automatic differentiation (default)
u_x = jnn.grad(u, x)

# ∂u/∂x  — mesh-based finite differences
u_x = jnn.grad(u, x, scheme="finite_difference")
```

`jnn.grad` returns a `Jacobian` placeholder.

---

## Laplacian

```python
# ∇²u = ∂²u/∂x² + ∂²u/∂y²  — automatic differentiation (default)
lap = jnn.laplacian(u, [x, y])

# Finite-difference Laplacian (requires compute_mesh_connectivity=True)
lap = jnn.laplacian(u, [x, y], scheme="finite_difference")
```

`jnn.laplace` is an alias for `jnn.laplacian`.

---

## Jacobian (vector-valued)

```python
# J = [∂u/∂x, ∂u/∂y]
J = jnn.jacobian(u, [x, y])
```

---

## Hessian matrix

```python
# H[i,j] = ∂²u/∂xᵢ∂xⱼ
H = jnn.hessian(u, [x, y])
```

---

## Divergence

```python
# ∇·F = ∂Fx/∂x + ∂Fy/∂y
div_F = jnn.divergence([Fx, Fy], [x, y])
```

---

## Curl

```python
# 2D curl (scalar): ∂Fy/∂x − ∂Fx/∂y
curl = jnn.curl_2d(Fx, Fy, x, y)

# 3D curl (vector): [∂Fz/∂y−∂Fy/∂z, ∂Fx/∂z−∂Fz/∂x, ∂Fy/∂x−∂Fx/∂y]
curl_vec = jnn.curl_3d(Fx, Fy, Fz, x, y, z)
```

---

## Differentiation Schemes

| Scheme | Flag | Notes |
|--------|------|-------|
| Automatic Differentiation | `"automatic_differentiation"` (default) | Exact; uses JAX `jax.grad` / `jax.jacfwd`. |
| Finite Difference (mesh-based) | `"finite_difference"` | Approximation; uses FEM stencils. Requires `compute_mesh_connectivity=True` in the domain. |

