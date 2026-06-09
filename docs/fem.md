# FEM and Variational Formulations

jNO supports two hybrid approaches that combine finite-element machinery with neural networks:

- **VPINN** (Variational PINN): a neural network is the trial function; the weak-form residual is minimised with jNO's standard training loop.
- **FEM system assembly**: the weak form is assembled into a linear system $Au = b$ or a nonlinear residual operator $R(u) = 0$ that can be solved with a direct or iterative linear solver.

Both approaches use the same domain setup and weak-form syntax.

---

## Domain setup

Call `domain.init_fem` after creating the domain to activate FEM mode. This builds the mesh connectivity, quadrature points, and basis functions needed for assembly.

```python
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.1))

domain.init_fem(
    element_type="TRI3",    # element family: "TRI3", "TRI6", "QUAD4", …
    quad_degree=2,          # quadrature degree for numerical integration
    bcs=[
        jno.dirichlet(["left", "right", "top", "bottom"]),   # homogeneous Dirichlet
    ],
    fem_solver=True,        # True = enable direct FEM solve path
)
```

For vector-valued unknowns (e.g. elasticity with displacement `(u_x, u_y)`):

```python
domain.init_fem(
    element_type="TRI3",
    quad_degree=2,
    bcs=[jno.dirichlet(["left", "right"], (0.0, 0.0))],
    vec=2,
)
```

---

## Boundary conditions

### Dirichlet (essential)

`jno.dirichlet(tags, values=None)` marks boundaries where the unknown is prescribed.

```python
jno.dirichlet("left")                          # zero on "left"
jno.dirichlet(["left", "right"], 0.0)          # zero on both sides
jno.dirichlet("top", lambda x, y: jnp.sin(x)) # spatially varying
jno.dirichlet("wall", (0.0, 1.0))             # vector: u_x=0, u_y=1
jno.dirichlet("wall", {"x": 0.0, "y": 1.0})  # same, dict form
```

### Neumann (natural)

`jno.neumann(tags)` marks boundaries where flux terms from the weak form should be assembled into the surface integrals. Zero Neumann (no flux) is the default and requires no explicit declaration.

```python
jno.neumann("right")          # include "right" boundary in surface assembly
jno.neumann(["top", "right"])
```

---

## Weak-form symbols

After `init_fem`, retrieve the trial and test function symbols from the domain:

```python
u, phi = domain.fem_symbols()
# u   — TrialFunction (the unknown)
# phi — TestFunction  (the test/weight function)
```

Quadrature coordinates are accessed as tagged variables:

```python
xg, yg, _ = domain.variable("fem_gauss", split=True)    # volume Gauss points
xr, yr, _ = domain.variable("gauss_right", split=True)  # Gauss points on "right" boundary
```

The boundary Gauss variable name follows the pattern `"gauss_<tag>"`.

---

## Building the weak form

Use `jno.np.grad` on `u` and `phi` with respect to the Gauss-point coordinates. The syntax is identical to the PINN workflow.

**Poisson equation** $-\Delta u = f$:

```python
import jno.numpy as jnn

ux  = jnn.grad(u,   xg)
uy  = jnn.grad(u,   yg)
phix = jnn.grad(phi, xg)
phiy = jnn.grad(phi, yg)
f = 1.0  # forcing term

# Weak form: ∫ ∇u · ∇φ dΩ − ∫ f φ dΩ = 0
weak = ux * phix + uy * phiy - f * phi
```

**Neumann boundary flux** (add to weak form):

```python
xr, yr, _ = domain.variable("gauss_right", split=True)
u_r  = u   # TrialFunction evaluated at right boundary Gauss points
phi_r = phi

# ∫_∂Ω_right g φ ds  with prescribed flux g = 1
weak_bc = weak - 1.0 * phi_r   # subtract Neumann load
```

---

## Assembly

### Linear system — `"fem_system"`

For linear PDEs, assemble directly into a stiffness matrix and load vector:

```python
A, b = weak.assemble(domain, target="fem_system")
# A — sparse or dense stiffness matrix
# b — load vector
```

`FemLinearSystem` supports addition so separate volume and boundary contributions can be combined:

```python
sys_vol = vol_form.assemble(domain, target="fem_system")
sys_bnd = bnd_form.assemble(domain, target="fem_system")
A, b = sys_vol + sys_bnd
```

Solve with any linear solver:

```python
import jax.numpy as jnp

u_h = jnp.linalg.solve(A, b)
```

### Nonlinear residual — `"fem_residual"`

For nonlinear PDEs, assemble a residual operator:

```python
R = weak.assemble(domain, target="fem_residual")
# R — FemResidualOperator

# Evaluate residual at a candidate solution
r = R.residual(u_flat)      # R(u)

# Linearise for Newton iterations
J, rhs = R.linearize(u_flat)   # J, -R(u)  →  solve J Δu = -R(u)
```

A simple Newton loop:

```python
u_h = jnp.zeros(R.size)
for _ in range(20):
    J, rhs = R.linearize(u_h)
    u_h = u_h + jnp.linalg.solve(J, rhs)
```

### VPINN path — use with jNO training

When the trial function is a neural network rather than a FE basis, pass the weak form directly to `jno.core` as a residual:

```python
net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=64, num_layers=4,
                               key=jax.random.PRNGKey(0)))
net.optimizer(optax.adam(1e-3))

u_nn = net(xg, yg) * xg * (1 - xg) * yg * (1 - yg)

ux  = jnn.grad(u_nn,  xg)
uy  = jnn.grad(u_nn,  yg)
phix = jnn.grad(phi, xg)
phiy = jnn.grad(phi, yg)

weak_vpinn = ux * phix + uy * phiy - 1.0 * phi

crux = jno.core([weak_vpinn.mse])
crux.solve(10000)
```

---

## Complete example — Poisson FEM solve

```python
import jax.numpy as jnp
import jno
import jno.numpy as jnn

domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.1))

domain.init_fem(
    element_type="TRI3",
    quad_degree=2,
    bcs=[jno.dirichlet(["left", "right", "top", "bottom"])],
    fem_solver=True,
)

u, phi = domain.fem_symbols()
xg, yg, _ = domain.variable("fem_gauss", split=True)

ux   = jnn.grad(u,   xg)
uy   = jnn.grad(u,   yg)
phix = jnn.grad(phi, xg)
phiy = jnn.grad(phi, yg)

f = 1.0 + 0.0 * xg   # uniform forcing

weak = ux * phix + uy * phiy - f * phi

A, b = weak.assemble(domain, target="fem_system")
u_h  = jnp.linalg.solve(A, b)
```

See the **[08 FEM and Variational PINNs](tutorials/08-fem-and-varpinns/poisson-2d-fem.md)** tutorials for worked VPINN and FEM examples.
