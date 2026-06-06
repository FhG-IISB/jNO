# Poisson 2D FEM

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/poisson_2d_fem.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

This example is a pure finite-element solve with no neural network component.

## Problem Setup

The script assembles the weak form of a manufactured Poisson problem and solves the resulting linear system.

## Step 1: Define the Weak Form

Instead of writing a pointwise PDE residual, the script builds bilinear and linear forms using FEM symbols.

```python
domain = jno.domain.rect(mesh_size=0.18)
domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[domain.dirichlet(["left", "right", "bottom", "top"], 0.0)],
    fem_solver=True,
)

u, phi = domain.fem_symbols()
xg, yg, _ = domain.variable("fem_gauss", split=True)

du_dx = jno.np.grad(u, xg)
du_dy = jno.np.grad(u, yg)
phi_x = jno.np.grad(phi, xg)
phi_y = jno.np.grad(phi, yg)

# Weak form: ∫ grad(u)·grad(phi) dΩ - ∫ f phi dΩ = 0
weak = du_dx * phi_x + du_dy * phi_y - source_f(xg, yg) * phi
```

## Step 2: Assemble the Linear System

The weak form is transformed into a matrix system through jNO's FEM assembly workflow.

```python
A, b = weak.assemble(domain, target="fem_system")

A_dense = to_dense(A)
b_dense = jnp.asarray(b)
u_fem = jnp.linalg.solve(A_dense, b_dense).reshape(-1)
```

## Step 3: Solve and Compare

The script computes the FEM solution and compares it to a known exact field.

```python
coords = np.asarray(domain.mesh.points)[:, :2]
x = jnp.asarray(coords[:, 0:1])
y = jnp.asarray(coords[:, 1:2])

u_exact = exact_u_num(x, y).reshape(-1)
rel_l2 = jnp.linalg.norm(u_exact - u_fem) / (jnp.linalg.norm(u_exact) + 1e-14)
print(f"Relative L2 error: {float(rel_l2):.6e}")
```

## What To Notice

- This chapter is about weak forms rather than PINN residuals.
- The assembly pipeline is useful as a reference even if your final method is neural.
- Pure FEM examples help validate the underlying PDE setup.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/poisson_2d_fem.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to 08 FEM and Variational PINNs</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/poisson_2d_fem.py"
```
