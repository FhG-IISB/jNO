# Differential Operators

`jno.numpy` (importable as `jnn`) is a drop-in NumPy-compatible module that operates on symbolic `Placeholder` expressions. It provides automatic and finite-difference differential operators for formulating PDE residuals, plus the full NumPy math library for use inside constraints.

```python
import jno.numpy as jnn
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

## Differentiation Schemes

| Scheme | Flag | Notes |
|--------|------|-------|
| Automatic Differentiation | `"automatic_differentiation"` (default) | Exact; uses JAX `jax.grad` / `jax.jacfwd`. |
| Finite Difference (mesh-based) | `"finite_difference"` | Approximation; uses FEM stencils. Requires `compute_mesh_connectivity=True` in the domain. |

---

## Mathematical Functions

### Trigonometric

```python
jnn.sin(x), jnn.cos(x), jnn.tan(x)
jnn.arcsin(x), jnn.arccos(x), jnn.arctan(x)
jnn.arctan2(y, x)   # alias: jnn.atan2
```

### Hyperbolic

```python
jnn.sinh(x), jnn.cosh(x), jnn.tanh(x)
jnn.arcsinh(x), jnn.arccosh(x), jnn.arctanh(x)
```

### Exponential / Logarithm

```python
jnn.exp(x), jnn.exp2(x), jnn.expm1(x)
jnn.log(x), jnn.log2(x), jnn.log10(x), jnn.log1p(x)
```

### Power / Root

```python
jnn.sqrt(x), jnn.cbrt(x), jnn.square(x)
jnn.power(x, n)
```

### Absolute / Rounding

```python
jnn.abs(x)
jnn.floor(x), jnn.ceil(x), jnn.round(x)
jnn.sign(x)
```

### Constants

```python
jnn.pi    # π
jnn.e     # e
jnn.inf   # ∞
jnn.nan   # NaN
```

---

## Reduction Operations

```python
jnn.sum(x)
jnn.mean(x)
jnn.std(x)
jnn.var(x)
jnn.min(x)
jnn.max(x)
jnn.median(x)
jnn.prod(x)
jnn.norm(x, ord=None, axis=None)
```

All support `axis=` and `keepdims=` keyword arguments.

---

## Reduction Properties on Placeholders

Every `Placeholder` expression exposes reduction properties. These return **scalar** Placeholder nodes suitable for use as constraints or trackers:

```python
expr.mse     # mean(expr²)      — most common loss term
expr.mae     # mean(|expr|)
expr.mean    # mean(expr)
expr.sum     # sum(expr)
expr.max     # max(expr)
expr.min     # min(expr)
expr.std     # std(expr)
```

Example:
```python
pde = jnn.laplacian(u, [x, y]) + 1.0
crux = jno.core([pde.mse], domain)       # minimise  mean((Δu+1)²)
```

---

## Array Operations

```python
jnn.concat([x, y], axis=-1)       # concatenate along last axis (default)
jnn.concatenate([x, y])           # alias for concat
jnn.stack([x, y], axis=0)         # stack along new axis
jnn.reshape(x, shape)
jnn.squeeze(x, axis=None)
jnn.expand_dims(x, axis)
jnn.transpose(x, axes=None)
```

---

## Comparison / Conditional

```python
jnn.where(condition, x, y)
jnn.maximum(x, y)
jnn.minimum(x, y)
```

Comparison operators are also available as methods on Placeholder objects:

```python
x > 0.5          # FunctionCall(greater, [x, Literal(0.5)])
x.equal(y)       # element-wise equality (traced)
x.not_equal(y)
```

---

## Linear Algebra

```python
jnn.dot(x, y)
jnn.matmul(x, y)
jnn.cross(x, y)

# Matrix multiply on Placeholder: x @ A
result = x @ A
```

---

## View Factor Operator (Radiation)

For radiation boundary conditions the domain can compute a view-factor matrix. Use `jnn.view_factor` to create an operator that applies it symbolically:

```python
# Get view factor matrix from domain
xb, yb, tb, nx, ny, VF = domain.variable("boundary", normals=True, view_factor=True)

# Wrap as a symbolic linear operator
VF_op = jnn.view_factor(VF)

# Apply: radiative heat flux received by each boundary point
q_inc = VF_op @ q_emitted      # F @ q (matrix-vector product)
q_inc = q_emitted @ VF_op      # q @ F

# Solve (I - αF)x = rhs
x = VF_op.solve(rhs, alpha)
```

---

## Custom Functions

Wrap arbitrary JAX functions for use inside symbolic expressions:

```python
def my_fn(x, y):
    return jnp.exp(-x**2) * jnp.sin(y)

result = jnn.function(my_fn, [x, y])
```

---

## Symbolic Arithmetic

Placeholders support standard Python arithmetic, enabling natural PDE notation:

```python
u = u_net(x, y)
v = v_net(x, y)

# Arithmetic
w = u + v
w = u * 2.0
w = u ** 2
w = -u

# Matrix multiplication
w = A @ u
```

---

## Trackers

Mark an expression as a *tracked metric* — it is evaluated and logged during training but **does not** contribute to the loss:

```python
from jno.numpy import tracker

val_error = tracker(jnn.mean(u(x, y) - u_exact(x, y)), interval=100)  # log every 100 epochs
crux = jno.core([pde.mse, boc.mse, val_error], domain)
```

---

## Constants Namespace

Load constant values from a file or dict and use them symbolically in expressions:

```python
C = jnn.constant("C", {
    "k": 1.5,
    "rho": 2700,
    "cp": 900,
    "physics": {"g": 9.81, "nu": 1.5e-5},
})

# Use in constraints
pde = -C.k * jnn.laplacian(u, [x, y]) - C.rho * jnn.grad(u, t)

# Load from file
C = jnn.constant("C", "params.json")      # JSON
C = jnn.constant("C", "params.yaml")      # YAML
C = jnn.constant("C", "params.toml")      # TOML
C = jnn.constant("C", "data.npz")         # NumPy npz
```
