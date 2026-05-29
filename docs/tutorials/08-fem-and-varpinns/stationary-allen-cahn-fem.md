# Stationary Allen-Cahn FEM

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/stationary_allen_cahn_fem.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

This example solves a nonlinear stationary Allen-Cahn problem using classical FEM machinery.

## Problem Setup

The weak form corresponds to a stationary Allen-Cahn equation with a nonlinear cubic term.

## Step 1: Define the Nonlinear Weak Residual

The script expresses the nonlinear form directly in terms of FEM operators rather than pointwise PINN losses.

```python
eps = 0.05
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.12))
domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[
        domain.dirichlet("left", u_left),
        domain.dirichlet("right", u_right),
    ],
    fem_solver=True,
)

u, phi = domain.fem_symbols()
xg, yg, _ = domain.variable("fem_gauss", split=True)

ux   = jno.np.grad(u, xg)
uy   = jno.np.grad(u, yg)
phix = jno.np.grad(phi, xg)
phiy = jno.np.grad(phi, yg)

# Weak form: ∫ eps^2 grad(u)·grad(phi) + (u^3 - u) phi dΩ = 0
weak = eps**2 * (ux * phix + uy * phiy) + (u**3 - u) * phi
```

## Step 2: Build the Jacobian and Nonlinear Solve Loop

Because the problem is nonlinear, a residual alone is not enough; the script also uses Jacobian information for iterative solution.

```python
op = weak.assemble(domain, target="fem_residual")

def residual_np(u_np):
    return np.asarray(op.residual(jnp.asarray(u_np)))

def jacobian_np(u_np):
    J = op.jacobian(jnp.asarray(u_np))
    return np.asarray(to_dense(J))
```

## Step 3: Solve With a Classical Nonlinear Method

A SciPy-style root or nonlinear solve routine is used to converge the weak-form system.

```python
u0  = exact_u_num(x_nodes, y_nodes).reshape(-1)
sol = spo.root(
    residual_np,
    np.asarray(u0),
    jac=jacobian_np,
    method="hybr",
)
u_fem = jnp.asarray(sol.x).reshape(-1)
```

## What To Notice

- Weak-form nonlinear solves are structurally different from PINN optimization.
- This example is useful for comparing classical and neural treatments of the same PDE family.
- It also shows how jNO's weak-form abstractions extend beyond linear problems.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/stationary_allen_cahn_fem.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to 08 FEM and Variational PINNs</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/stationary_allen_cahn_fem.py"
```
