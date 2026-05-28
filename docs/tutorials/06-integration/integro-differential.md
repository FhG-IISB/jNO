# Integro-Differential Equation

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/06_integration/integro_differential.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/06-integration/">Back to chapter</a>
</div>

This example solves an **integro-differential equation (IDE)** — an equation where the unknown appears under both a derivative and an integral at the same time.  It shows how `.d(x)` and `.integrate()` compose naturally in the same residual.

## Equation

$$u'(x) + u(x) = g(x) + \int_0^1 u(t)\, dt, \qquad x \in [0,1], \quad u(0) = 0$$

Exact solution: $u^*(x) = \sin(\pi x)$

### Derivation of g

$$u'(x) = \pi\cos(\pi x), \qquad u(x) = \sin(\pi x), \qquad \int_0^1 \sin(\pi t)\,dt = \frac{2}{\pi}$$

$$g(x) = \pi\cos(\pi x) + \sin(\pi x) - \frac{2}{\pi}$$

## Why this is different

In a standard PINN the residual involves only pointwise quantities — derivatives at $x$.  Here the residual also includes $\int_0^1 u(t)\,dt$, which is a **scalar** that couples the solution at every mesh point.

jno handles this transparently: `.integrate()` returns a scalar placeholder that flows through the same computation graph as `.d(x)`.  Both appear in the same MSE loss with no extra bookkeeping.

## Hard boundary condition

The Dirichlet condition $u(0) = 0$ is enforced by the network ansatz

$$u(x) = \text{net}(x) \cdot x$$

Multiplying by $x$ forces $u(0) = 0$ for any weight configuration, so the optimizer never needs to "discover" the boundary condition — it is **automatically satisfied** throughout training.

## Building the residual

```python
x, _ = domain.variable("interior")

u  = net(x) * x   # hard BC: u(0) = 0

C  = u.integrate()           # scalar: ∫₀¹ u(t) dt
du = u.d(x)                  # (N, 1): u'(x) at every collocation point

residual = du + u - g - C    # IDE residual
```

`C` is a scalar that is the same for every row — JAX broadcasting adds it to the `(N, 1)` arrays `du`, `u`, and `g` without any manual reshaping.

## Shape summary

| Expression | Shape | Note |
|---|---|---|
| `u` | `(N, 1)` | network output |
| `u.d(x)` | `(N, 1)` | pointwise derivative |
| `u.integrate()` | scalar | integral over all mesh points |
| `g` | `(N, 1)` | forcing term |
| `residual` | `(N, 1)` | broadcast scalar + vectors |

## Step-by-step

**Step 1 — Domain and variable**

```python
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.05))
x, _ = domain.variable("interior")
```

**Step 2 — Hard-BC ansatz**

```python
u = net(x) * x   # u(0) = 0 for any net weights
```

**Step 3 — Compose operators**

```python
C  = u.integrate()   # scalar: feeds into every row of the residual
du = u.d(x)          # pointwise: (N, 1)
```

**Step 4 — Form and solve**

```python
residual = du + u - g - C
crux = jno.core([residual.mse], domain)
crux.solve(30_000)
```

## What to notice

- **`.integrate()` returns a scalar** here — no `var=` argument, because there is no outer collocation variable to hold fixed.  The result is a single number representing $\int_0^1 u(t)\,dt$.
- **`.d(x)` and `.integrate()` compose** — both produce `Placeholder` objects that participate in the same expression graph.
- **Gradients flow through both operators** — the loss is differentiable with respect to the network weights simultaneously through the derivative term and the integral term.
- **Relative L2 error < 10%** is achieved with 21 interior points and 30 000 steps.

## Contrast with the Fredholm tutorials

| Feature | Fredholm separable | Fredholm non-separable | IDE (this example) |
|---|---|---|---|
| Integral result | scalar | `(N, 1)` | scalar |
| `.integrate(var=...)` | no | yes | no |
| Derivative in residual | no | no | yes |
| Boundary condition | none | none | hard BC ansatz |

## Script

```python
--8<-- "tutorial_examples/06_integration/integro_differential.py"
```

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/06_integration/integro_differential.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/06-integration/">Back to 06 Integration</a>
</div>
