# jno.numpy API

`jno.numpy` (typically imported as `jnn`) provides a NumPy-like API that works seamlessly with jNO's symbolic computation graph. Functions accept both regular JAX arrays and jNO `Placeholder` objects (variables, network outputs, etc.).

```python
import jno.numpy as jnn
```

---

## Constants

```python
jnn.pi      # π
jnn.π       # π (Unicode alias)
jnn.e       # Euler's number
jnn.inf     # Infinity
jnn.nan     # Not a number
```

---

## Differential Operators

These are the core operators for defining PDEs. They create symbolic nodes in the computation graph that are evaluated using automatic differentiation or finite differences during training.

### Gradient

```python
# First-order partial derivative ∂u/∂x
u_x = jnn.grad(u(x, y), x)

# Using finite differences instead of AD
u_x_fd = jnn.grad(u(x, y), x, scheme='finite_difference')
```

### Laplacian

```python
# ∇²u = ∂²u/∂x² + ∂²u/∂y²
lap_u = jnn.laplacian(u(x, y), [x, y])

# Finite difference scheme
lap_u_fd = jnn.laplacian(u(x, y), [x, y], scheme='finite_difference')

# Alias
lap_u = jnn.laplace(u(x, y), [x, y])
```

### Hessian

```python
# Full Hessian matrix H[i,j] = ∂²u/∂xᵢ∂xⱼ
H = jnn.hessian(u(x, y), [x, y])
```

### Jacobian

```python
# Jacobian matrix
J = jnn.jacobian(u(x, y), [x, y])
```

### Divergence

```python
# div(F) = ∂Fx/∂x + ∂Fy/∂y
div_F = jnn.divergence([Fx, Fy], [x, y])
```

### Curl

```python
# 2D curl (scalar): ∂Fy/∂x - ∂Fx/∂y
curl_2d = jnn.curl_2d(Fx, Fy, x, y)

# 3D curl (vector)
curl_3d = jnn.curl_3d(Fx, Fy, Fz, x, y, z)
```

---

## Trigonometric Functions

| Function | Description |
|----------|-------------|
| `jnn.sin(x)` | Sine |
| `jnn.cos(x)` | Cosine |
| `jnn.tan(x)` | Tangent |
| `jnn.arcsin(x)` | Inverse sine |
| `jnn.arccos(x)` | Inverse cosine |
| `jnn.arctan(x)` | Inverse tangent |
| `jnn.arctan2(y, x)` | Two-argument arctangent |

## Hyperbolic Functions

| Function | Description |
|----------|-------------|
| `jnn.sinh(x)` | Hyperbolic sine |
| `jnn.cosh(x)` | Hyperbolic cosine |
| `jnn.tanh(x)` | Hyperbolic tangent |
| `jnn.arcsinh(x)` | Inverse hyperbolic sine |
| `jnn.arccosh(x)` | Inverse hyperbolic cosine |
| `jnn.arctanh(x)` | Inverse hyperbolic tangent |

## Exponential and Logarithmic Functions

| Function | Description |
|----------|-------------|
| `jnn.exp(x)` | Exponential eˣ |
| `jnn.exp2(x)` | 2ˣ |
| `jnn.expm1(x)` | eˣ - 1 |
| `jnn.log(x)` | Natural logarithm |
| `jnn.log2(x)` | Base-2 logarithm |
| `jnn.log10(x)` | Base-10 logarithm |
| `jnn.log1p(x)` | log(1 + x) |

## Power and Root Functions

| Function | Description |
|----------|-------------|
| `jnn.sqrt(x)` | Square root |
| `jnn.cbrt(x)` | Cube root |
| `jnn.square(x)` | Square |
| `jnn.power(x, y)` | x raised to the power y |

## Rounding and Absolute Value

| Function | Description |
|----------|-------------|
| `jnn.abs(x)` | Absolute value |
| `jnn.floor(x)` | Floor |
| `jnn.ceil(x)` | Ceiling |
| `jnn.round(x)` | Round to nearest integer |
| `jnn.clip(x, min, max)` | Clip values to range |

## Activation Functions

| Function | Description |
|----------|-------------|
| `jnn.sigmoid(x)` | Sigmoid: 1 / (1 + exp(-x)) |
| `jnn.softplus(x)` | Softplus: log(1 + exp(x)) |
| `jnn.relu(x)` | ReLU: max(0, x) |
| `jnn.leaky_relu(x, slope)` | Leaky ReLU |
| `jnn.elu(x, alpha)` | ELU |
| `jnn.gelu(x)` | GELU |
| `jnn.swish(x)` | Swish: x · sigmoid(x) |
| `jnn.sign(x)` | Sign function |
| `jnn.heaviside(x, h0)` | Heaviside step function |

---

## Array Manipulation

```python
jnn.concat([a, b], axis=-1)      # Concatenate along axis
jnn.concatenate([a, b], axis=-1)  # Alias for concat
jnn.stack([a, b], axis=0)         # Stack along new axis
jnn.reshape(x, shape)             # Reshape
jnn.squeeze(x, axis=None)         # Remove single-dimensional entries
jnn.expand_dims(x, axis)          # Add dimension
jnn.transpose(x, axes=None)       # Transpose
```

---

## Reduction Operations

All reduction operations accept `axis` and `keepdims` parameters:

```python
jnn.sum(x, axis=None)
jnn.mean(x, axis=None)
jnn.median(x, axis=None)
jnn.std(x, axis=None)
jnn.var(x, axis=None)
jnn.min(x, axis=None)
jnn.max(x, axis=None)
jnn.prod(x, axis=None)
jnn.norm(x, ord=None, axis=None)
```

---

## Comparison and Selection

```python
jnn.maximum(x, y)       # Element-wise maximum
jnn.minimum(x, y)       # Element-wise minimum
jnn.where(cond, x, y)   # Conditional selection
```

---

## Linear Algebra

```python
jnn.dot(x, y)       # Dot product
jnn.matmul(x, y)    # Matrix multiplication
jnn.cross(x, y)     # Cross product
```

---

## View Factor Operator

For radiation heat transfer problems:

```python
VF = jnn.view_factor(F)   # Create view factor operator
result = VF @ x            # F @ x
result = x @ VF            # x @ F
result = VF.solve(rhs, alpha)  # Solve (I - αF)x = rhs
```

---

## Constants and Parameters

### Named Constants

Load constants from dictionaries or files:

```python
C = jnn.constant("C", {"k": 1.0, "m": 2.0, "physics": {"gravity": 9.81}})
C.k                     # -> Constant(C.k=1.0)
C.physics.gravity       # -> Constant(C.physics.gravity=9.81)

# From files
C = jnn.constant("C", "params.json")
C = jnn.constant("C", "config.yaml")
```

### Trainable Parameters

```python
a = jnn.parameter((1,), name="a")()   # Scalar trainable parameter
b = jnn.parameter((3,), name="b")()   # Vector trainable parameter
```

---

## Custom Functions

Wrap arbitrary functions for use in the computation graph:

```python
custom_fn = jnn.function(my_jax_function, args=[x, y], name="my_fn")
```

---

## Trackers

Monitor quantities during training without adding them to the loss:

```python
tracker = jnn.tracker(jnn.mean(u - u_exact), interval=100)
```

---

## Array Creation

Standard JAX array creation (returns regular arrays, not placeholders):

```python
jnn.zeros(shape)
jnn.ones(shape)
jnn.linspace(start, stop, num)
jnn.arange(start, stop, step)
jnn.full(shape, fill_value)
```

---

## See Also

- [Architecture Guide](Architecture-Guide.md) — neural network models
- [Training and Solving](Training-and-Solving.md) — using these operators in the training loop
