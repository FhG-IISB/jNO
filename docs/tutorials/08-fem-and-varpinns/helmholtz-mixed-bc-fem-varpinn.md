# Helmholtz Mixed BC FEM VarPINN

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/helmholtz_mixedBC_fem_varpinn.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

This example trains a variational PINN for a 2D Helmholtz problem and compares it against a fine FEM reference solution.

## Problem Setup

The PDE has the Helmholtz form with mixed Dirichlet and Neumann data and a manufactured exact solution.

## Step 1: Write a Weak Residual

The residual is assembled in variational form rather than as a pointwise strong-form loss.

```python
domain = jno.domain.rect(mesh_size=0.22)
domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[
        domain.dirichlet("left",   0.0),
        domain.dirichlet("bottom", lambda p: jnp.sin(jnp.pi * p[0])),
        domain.neumann(["right", "top"]),
    ],
    fem_solver=True,
)

u, phi = domain.fem_symbols()
xg, yg, _ = domain.variable("fem_gauss",  split=True)
xr, yr, _ = domain.variable("gauss_right", split=True)
xt, yt, _ = domain.variable("gauss_top",   split=True)

k_sq = 0.0 * xg + k_val**2
vol  = (u.d(xg) * phi.d(xg)
      + u.d(yg) * phi.d(yg)
      - k_sq * u * phi
      - source_f(xg, yg) * phi)

weak = vol - exact_flux_right(xr, yr)*phi - exact_flux_top(xt, yt)*phi
```

## Step 2: Combine Hard Boundary Handling With VPINN Training

The neural ansatz handles some constraints structurally while the weak form captures the PDE and remaining conditions.

```python
net = jno.nn.wrap(
    foundax.mlp(2, hidden_dims=32, num_layers=4,
                activation=jax.nn.tanh, key=jax.random.PRNGKey(0))
)

def apply_hard_bc(u_pred, x, y):
    # u(0,y)=0  and  u(x,0)=sin(pi*x)
    return sin(pi * x) + x * y * u_pred

xg2, yg2, _ = train_domain.variable("fem_gauss", split=True)
u_gauss = apply_hard_bc(net(xg2, yg2), xg2, yg2)

pde  = weak.assemble(train_domain, u_net=u_gauss, target="vpinn")
crux = jno.core(constraints=[pde.mse], domain=train_domain)
net.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1e-3, warmup_steps=1, decay_steps=10, end_value=1e-5)))
crux.solve(epochs=10)
```

## Step 3: Compare Against FEM

The script solves a classical FEM reference problem on a finer mesh and visualizes the difference.

```python
A_fem, b_fem = weak.assemble(fem_domain, target="fem_system")
u_fem = jnp.linalg.solve(to_dense(A_fem), jnp.asarray(b_fem)).reshape(-1)

u_exact = exact_u_num(x_f, y_f).reshape(-1)
rel_l2_fem  = jnp.linalg.norm(u_exact - u_fem) / (jnp.linalg.norm(u_exact) + 1e-14)
rel_l2_vpinn = jnp.linalg.norm(u_true - u_vpinn) / (jnp.linalg.norm(u_true) + 1e-14)
print(f"VPINN relative L2: {float(rel_l2_vpinn):.6e}")
print(f"FEM   relative L2: {float(rel_l2_fem):.6e}")
```

## What To Notice

- This is a representative VPINN workflow in the tutorial suite.
- Weak forms can improve how derivative-heavy PDEs are handled.
- Direct FEM comparison makes the example especially useful for benchmarking.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/helmholtz_mixedBC_fem_varpinn.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to 08 FEM and Variational PINNs</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/helmholtz_mixedBC_fem_varpinn.py"
```
