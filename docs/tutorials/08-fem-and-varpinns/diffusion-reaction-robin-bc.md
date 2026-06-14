# Diffusion-Reaction Robin BC

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/diffusion_reaction_robinBC.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

This example compares a variational PINN against a classical FEM reference under mixed Dirichlet and Robin boundary conditions.

## Problem Setup

The PDE is a diffusion-reaction or Helmholtz-like problem on a 2D domain with boundary integrals contributing Robin terms.

## Step 1: Build a Weak Form With Boundary Terms

The script includes boundary quadrature contributions directly in the variational residual.

```python
domain = jno.domain.rect(mesh_size=0.22)
domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[
        domain.dirichlet("left",   lambda p: p[1]),
        domain.dirichlet("bottom", 0.0),
        domain.neumann(["right", "top"]),
    ],
    fem_solver=True,
)

u, phi = domain.fem_symbols()
xg, yg, _ = domain.variable("fem_gauss",  split=True)
xr, yr, _ = domain.variable("gauss_right", split=True)
xt, yt, _ = domain.variable("gauss_top",   split=True)

du_dx = u.d(xg);  du_dy = u.d(yg)
phi_x = phi.d(xg); phi_y = phi.d(yg)

vol          = du_dx*phi_x + du_dy*phi_y + sigma*u*phi - source_f(xg, yg)*phi
robin_right  = alpha_right * u * phi - robin_rhs_right(xr, yr) * phi
robin_top    = alpha_top   * u * phi - robin_rhs_top(xt, yt)   * phi
weak         = vol + robin_right + robin_top
```

## Step 2: Train the VPINN

Instead of minimizing a pointwise PDE residual, the network minimizes a weak-form objective.

```python
net = jno.nn.wrap(
    foundax.mlp(2, hidden_dims=32, num_layers=3,
                activation=jax.nn.tanh, key=jax.random.PRNGKey(0))
)

def apply_hard_bc(raw, x, y):
    return y + x * y * raw   # satisfies u(0,y)=y and u(x,0)=0

xg2, yg2, _ = train_domain.variable("fem_gauss", split=True)
u_gauss = apply_hard_bc(net(xg2, yg2), xg2, yg2)

pde  = weak.assemble(train_domain, u_net=u_gauss, target="vpinn")
crux = jno.core(constraints=[pde.mse])
net.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1e-3, warmup_steps=2, decay_steps=100, end_value=1e-5)))
crux.solve(epochs=1000)
```

## Step 3: Compare Against FEM

A finer FEM solve is used as a reference to assess how well the variational PINN captures the solution.

```python
A, b  = weak.assemble(fem_domain, target="fem_system")
u_fem = jnp.linalg.solve(to_dense(A), jnp.asarray(b)).reshape(-1)

u_exact = exact_u_num(x_nodes, y_nodes).reshape(-1)
rel_l2_fem  = jnp.linalg.norm(u_exact - u_fem) / (jnp.linalg.norm(u_exact) + 1e-14)
print(f"FEM  relative L2: {float(rel_l2_fem):.6e}")
```

## What To Notice

- Robin conditions fit naturally into a weak-form workflow.
- This script is a strong reference for FEM versus VPINN comparisons.
- Boundary quadrature tags are one of the key implementation ideas.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/diffusion_reaction_robinBC.py" download>Download full script</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to 08 FEM and Variational PINNs</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/diffusion_reaction_robinBC.py"
```
