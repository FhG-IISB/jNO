# Fredholm Integral Equation of the Second Kind

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/06_integration/fredholm_integral_equation.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/06-integration/">Back to chapter</a>
</div>

This example solves a **Fredholm integral equation of the second kind** — an equation where the unknown function appears both outside and inside an integral.  The exact solution is known, so the trained network's accuracy can be measured directly.

## Equation

$$u(x) = f(x) + \int_0^1 x \cdot t \; u(t) \, dt, \qquad x \in [0,1]$$

with forcing term chosen so that the exact solution is

$$u^*(x) = \sin(\pi x)$$

### Derivation of f

Substituting $u^* = \sin(\pi x)$ into the equation:

$$\sin(\pi x) = f(x) + \int_0^1 x \cdot t \cdot \sin(\pi t) \, dt$$

The integral evaluates to (integration by parts):

$$\int_0^1 t \cdot \sin(\pi t) \, dt = \frac{1}{\pi}$$

so $x \cdot \int_0^1 t \cdot \sin(\pi t) \, dt = x / \pi$, giving:

$$f(x) = \sin(\pi x) - \frac{x}{\pi}$$

## Why this is interesting

Integral equations cannot be solved by the standard PINN approach of differentiating pointwise residuals — there is no ODE or PDE to differentiate.  The `.integrate()` operator fills this gap.  The key insight here is that the degenerate (separable) kernel $K(x,t) = x \cdot t$ lets the double-argument integral collapse into a simple product:

$$\int_0^1 x \cdot t \cdot u(t) \, dt = x \cdot \underbrace{\int_0^1 t \cdot u(t) \, dt}_{C}$$

$C$ is a **scalar** independent of $x$.  jno computes it with one `.integrate()` call over the full mesh.

## Step 1: Set up the domain

```python
domain = jno.domain.line(mesh_size=0.01)
x, _ = domain.variable("interior")
```

The 1D mesh on $[0,1]$ doubles as the integration mesh.  No separate quadrature rule is needed.

## Step 2: Define the forcing term

```python
pi_val = float(jnp.pi)
f = jno.np.sin(π * x) - x / pi_val
```

## Step 3: Build the model

```python
net = jno.nn.wrap(
    foundax.mlp(in_features=1, hidden_dims=64, num_layers=4,
                activation=jax.nn.tanh, key=jax.random.PRNGKey(0))
)
u = net(x)
```

## Step 4: Formulate the residual

```python
# C = ∫₀¹ t · u(t) dt  — scalar, the same for every x
C = (x * u).integrate()

# Pointwise residual  R(xᵢ) = u(xᵢ) − f(xᵢ) − xᵢ · C
residual = u - f - x * C
```

`(x * u).integrate()` evaluates the integrand `t · u(t)` at **all mesh nodes** (here `x` is the dummy variable $t$), then reduces to a scalar using nodal volume weights.  Multiplying by the outer `x` broadcasts that scalar back to the collocation grid.

Both uses of the network — as the integrand inside `C` and as the outer `u` — go through the **same** shared parameters.  Gradients flow through both paths simultaneously.

## Step 5: Solve

```python
EPOCHS = 50_000
crux = jno.core([residual.mse], domain)
crux.solve(EPOCHS)
```

## What to notice

- **No boundary conditions are needed.** The integral equation is posed on the interior only; boundary values emerge naturally from the trained solution.
- **`.integrate()` is differentiable.** `jax.grad` and `eqx.filter_grad` propagate through it, so the scalar $C$ receives gradient updates alongside the pointwise residual terms.
- **JIT-friendly.** Mesh weights are precomputed at domain creation and embedded as JAX constants.  The integral adds no overhead beyond a single dot-product per step.
- **Relative L2 error < 5 %** is achievable with a modest 4-layer MLP and 50 000 Adam steps.

## Script

```python
--8<-- "tutorial_examples/06_integration/fredholm_integral_equation.py"
```

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/06_integration/fredholm_integral_equation.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/06-integration/">Back to 06 Integration</a>
</div>
