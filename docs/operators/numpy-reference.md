# jno.numpy Reference

This page covers the full `jno.numpy` (alias `jnn`) utility API: math functions, reductions, array operations, symbolic arithmetic, and constants.

```python
import jno.numpy as jnn
```

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

Every `Placeholder` expression exposes reduction properties that return **scalar** nodes suitable as loss terms or trackers:

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
crux = jno.core([pde.mse])       # minimise  mean((Δu+1)²)
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

### Second-order tensors

All act on the **last two axes** and broadcast over any leading (e.g.
quadrature) axes.

```python
jnn.inv(A)          # inverse
jnn.det(A)          # determinant  → scalar per leading index
jnn.eigvalsh(A)     # ascending eigenvalues of a symmetric tensor → principal values
```

**Symmetric matrix functions** — `f(A) = V diag(f(λ)) Vᵀ` via `eigh`. Their
gradient stays finite at **repeated eigenvalues** (equal principal stretches —
the common case in finite-strain plasticity) using the Daleckiĭ–Kreĭn / Löwner
divided-difference form (Higham, *Functions of Matrices*, SIAM 2008, Thm 3.11).
Distinct from the element-wise scalar `log` / `exp` / `sqrt` above.

```python
jnn.logm(A)         # matrix logarithm    E = 0.5 * jnn.logm(jnn.matmul(F.T, F))   # Hencky strain
jnn.expm(A)         # matrix exponential  (inverse of logm on SPD)
jnn.sqrtm(A)        # matrix square root  U = jnn.sqrtm(jnn.matmul(F.T, F))         # right stretch
```

These are the tensor primitives a finite-strain constitutive update needs:
`inv` for the multiplicative split `F = Fₑ·Fₚ`, `det` for the volumetric part
`J = det F`, and `logm`/`expm` for a log-strain (Hencky) return map.

---

## Symbolic Arithmetic

Placeholders support standard Python arithmetic, enabling natural PDE notation:

```python
u = u_net(x, y)
v = v_net(x, y)

w = u + v
w = u * 2.0
w = u ** 2
w = -u
w = A @ u
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
