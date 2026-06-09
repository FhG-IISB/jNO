# Integration Operators

`jno.numpy` (importable as `jnn`) supports **mesh-based numerical integration** of symbolic expressions over boundaries and volumes. Integration is the companion to differentiation: where `.d(x)` computes derivatives pointwise, `.integrate()` collapses a field to a single scalar by summing over the mesh.

```python
import jno.numpy as jnn
```

---

## Core API

### Method syntax

```python
# Volume integral  ∫_Ω expr dV
vol = expr.integrate()

# Boundary integral  ∫_∂Ω expr ds
bnd = expr_on_boundary.integrate()
```

### Function syntax (alias)

```python
vol = jnn.integrate(expr)
bnd = jnn.integrate(expr_on_boundary)
```

`integrate()` returns an `Integral` placeholder — a scalar node in the expression graph that reduces to a Python float when evaluated.

---

## Region auto-detection

The integration region (**boundary** vs **volume**) is determined automatically from the `Variable` tags inside the expression.  You do not pass any region argument.

```python
dom = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))

x, y, t         = dom.variable("interior")   # volume region
x_b, y_b, t_b  = dom.variable("boundary")   # boundary region
x_w, y_w, t_w  = dom.variable("wall")       # any named boundary group

∫_Ω  u.integrate()           # interior tag → volume weights
∫_∂Ω u_b.integrate()         # boundary tag → arc-length / surface weights
```

Internally, the evaluator checks whether the tag's points lie on the global mesh boundary.  Any physical group registered as a boundary in the domain receives arc-length (2D) or area (3D) weights; all others receive volume weights.

---

## Integration weights

| Region | Weight per node | Source |
|--------|----------------|--------|
| 1-D volume | ½ × adjacent segment lengths | `mesh_connectivity["nodal_volumes"]` |
| 2-D volume | ⅓ × incident triangle areas | `mesh_connectivity["nodal_volumes"]` |
| 3-D volume | ¼ × incident tetrahedron volumes | `mesh_connectivity["nodal_volumes"]` |
| 2-D boundary | ½ × adjacent boundary edge lengths | `mesh_connectivity["nodal_ds"]` |
| 3-D boundary | ⅓ × incident boundary face areas | `mesh_connectivity["nodal_ds"]` |

Weights are precomputed once at domain creation and cached on the domain object; they are **not** recomputed inside the training loop.

---

## Volume integrals

```python
dom = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
x, y, _ = dom.variable("interior")

u = net(jno.np.concat([x, y], axis=-1))

# ∫_Ω u dA  — scalar placeholder, suitable as a loss or tracker
vol_mean = u.integrate()
```

Use as a loss term:

```python
# Enforce prescribed volume average
target_mean = 2.0 / 3.0
integral_loss = (u.integrate() - target_mean).square()
crux = jno.core([pde_loss, integral_loss])
```

Use as a tracker (logged, not differentiated):

```python
from jno.numpy import tracker

vol_mean = tracker(u.integrate(), interval=500)
crux = jno.core([pde_loss, vol_mean])
```

---

## Boundary integrals

```python
dom = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
x_b, y_b, _ = dom.variable("boundary")

u_b = net(jno.np.concat([x_b, y_b], axis=-1))

# ∫_∂Ω u ds  — perimeter-weighted integral
boundary_mean = u_b.integrate()
```

---

## Flux integrals

For flux integrals of the form `∫_∂Ω F·n ds`, request boundary normals and compute the dot product explicitly before calling `.integrate()`:

```python
x_b, y_b, _, nx, ny = dom.variable("boundary", normals=True)

u_b = net(jno.np.concat([x_b, y_b], axis=-1))

# Outward flux  ∫_∂Ω ∂u/∂n ds
flux = (u_b.d(x_b) * nx + u_b.d(y_b) * ny).integrate()

# Divergence theorem check: for F = (x, y), flux = 2·area
div_thm = (x_b * nx + y_b * ny).integrate()   # ≈ 2.0 on unit square
```

jno does **not** compute F·n automatically — you specify the integrand explicitly, giving full control over what is integrated.

---

## Named boundary groups

For domains with named boundary regions, request variables from each group separately:

```python
dom = jno.domain(constructor=jno.domain.polygon(...))

x_left,  y_left,  _, nx_l, ny_l = dom.variable("left",  normals=True)
x_right, y_right, _, nx_r, ny_r = dom.variable("right", normals=True)

# Separate flux integrals per boundary segment
flux_left  = (u(x_left,  y_left)  * nx_l).integrate()
flux_right = (u(x_right, y_right) * nx_r).integrate()
```

---

## JIT compatibility

`Integral` nodes are fully compatible with `jax.jit`.  Integration weights are precomputed at domain setup and embedded as JAX constants during the first trace.  Subsequent JIT calls hit the compiled cache without recomputing any numpy operations.

```python
import jax

@jax.jit
def loss(params, ctx):
    ...
    return (u.integrate() - target).square()
```

Gradients flow through the integral, making it suitable as a differentiable loss term:

```python
import equinox as eqx

grad_fn = eqx.filter_jit(eqx.filter_grad(loss_fn))
grads = grad_fn(model)
```

---

## Common patterns

### Volume mean as a soft constraint

```python
x, y, _ = dom.variable("interior")
u = net(jno.np.concat([x, y], axis=-1))
target = 2.0 / 3.0

losses = [
    pde_residual.mse,
    (u.integrate() - target).square(),
]
```

### Boundary flux conservation

```python
x_b, y_b, _, nx, ny = dom.variable("boundary", normals=True)
u_b = net(jno.np.concat([x_b, y_b], axis=-1))

# Enforce zero net outward flux
net_flux = (u_b.d(x_b) * nx + u_b.d(y_b) * ny).integrate()
losses = [pde.mse, net_flux.square()]
```

### Tracking a physical observable

```python
from jno.numpy import tracker

x, y, _ = dom.variable("interior")
u = net(jno.np.concat([x, y], axis=-1))

vol_mean = tracker(u.integrate(), interval=200)   # logged every 200 epochs
crux = jno.core([pde.mse, bc.mse, vol_mean])
```
