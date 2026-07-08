# Variational PINN: Poisson with a network trial

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/vpinn_poisson_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A **variational PINN** keeps the FEM *test* space but replaces the *trial* with a **neural network**:
instead of solving a linear system for FE coefficients, you minimise the weak-form residual
**test-projected** onto the FE basis. In jNO it is authored exactly like any other `jno.fem` problem
— write the weak form with `u = net(x,y)` and hand it to `jno.fem`; no `init_fem`, no `weak.assemble`.

We solve Poisson $-\Delta u = f$ on the unit square with exact $u = x(1-x)y(1-y)$
(so $f = 2[x(1-x)+y(1-y)]$).

## A network trial in the weak form

`dom.fem_symbols()` gives the trial symbol `u` and the FE test function `phi`. Build the trial from a
network with a **hard-BC ansatz** that vanishes on the boundary, write the standard weak form, and
`jno.fem` detects the network (a `ModelCall`) and returns a trainable test-projected residual:

```python
u, phi = dom.fem_symbols()
xi, yi, _ = dom.variable("interior", split=True)
xb, yb, _ = dom.variable("boundary", split=True)

u_net = net(xi, yi) * (xi * (1 - xi) * yi * (1 - yi))   # network trial, vanishes on the boundary
vi    = phi.bind(x=xi, y=yi)
f     = 2.0 * (xi * (1 - xi) + yi * (1 - yi))

pde = jno.fem([
    jnn.grad(u_net, xi) * jnn.grad(vi, xi) + jnn.grad(u_net, yi) * jnn.grad(vi, yi) - f * vi,
    u(xb, yb) - 0.0,                                     # Dirichlet declaration (see below)
])
jno.core([pde.mse], domain=dom).solve(2500)             # minimise the test-projected residual
```

## The Dirichlet condition is not optional

`u(boundary) - 0` looks redundant — the hard-BC ansatz already vanishes on the boundary — but it is
**required**. It tells `jno.fem` which FE test functions live on the boundary so their residual is
masked. A test function that does *not* vanish on the boundary carries the exact solution's
$\partial u/\partial n$ flux — an **irreducible** term that makes the loss minimum *not* the PDE
solution, so training would diverge from it. With the declaration the residual is tested on the
interior only and the network converges.

## The result

![VPINN solution and pointwise error vs the analytic solution](/jNO/assets/vpinn_poisson_2d.png)

The trained network matches the analytic $x(1-x)y(1-y)$ to rel-$L^2 \approx 7\times10^{-4}$.

## What to notice

- **Same entry as FEM.** A network trial vs an FE trial is the only difference; `jno.fem` routes by
  detecting the `ModelCall`, and the returned `.mse` is an ordinary jNO loss for `jno.core`.
- **Declare the Dirichlet boundary** (`u(boundary) - g`) so its test functions are masked.
- Single-field 2-D/3-D for now (1-D and coupled multi-field raise a clear error).
