# Getting Started

The fastest path from a fresh install to a first PDE solve. Complete
[Installation](Installation.md) first, then build the example up one step at a time.

We solve a **2-D Poisson** problem on the unit square — first with a physics-informed network
(PINN), then, at the end, the *same problem* through the FEM and FDM front doors. All three are
written in one language and checked against the same exact solution:

$$-\nabla^2 u = 2\pi^2 \sin(\pi x)\sin(\pi y), \quad u\big|_{\partial\Omega}=0
\quad\Rightarrow\quad u^\ast = \sin(\pi x)\sin(\pi y).$$

---

## 1. Set up a run

`jno.setup()` initialises logging and returns a run directory in one call.

```python
import jno, jax, optax, foundax

run = jno.setup("./runs/getting-started")
```

## 2. Define the domain

A [domain](Domain-and-Geometry.md) holds the geometry and the points sampled on it. `variable(...)`
returns the coordinates of a named region (`"interior"`, `"boundary"`, …); a domain is a source of
(effectively infinite) collocation points for a PINN.

```python
dom = jno.Shape.rect(0, 0, 1, 1, size=0.04).domain()
x, y, _ = dom.variable("interior")     # interior collocation coordinates
```

## 3. Create a network

Every [model](foundation_models/index.md) comes from **foundax** and is wrapped with `jno.nn(...)` to
gain jNO's training controls. Attach an [optimizer](training/index.md) (schedules, LoRA, freezing, …
all chain off the model):

```python
net = jno.nn(foundax.mlp(2, hidden_dims=64, num_layers=4, key=jax.random.PRNGKey(0)))
net.optimizer(optax.adam(1e-3))
```

## 4. Write the PDE residual

Call the network on the coordinates and take derivatives with the [differential
operators](operations.md#differentiation) — here the concise `u.dd(x)` (second derivative). Multiplying by
`x(1-x)y(1-y)` makes the ansatz vanish on `∂Ω`, so the Dirichlet BC is enforced **exactly** with no loss
term:

```python
import jno.numpy as jnn

pi = jnn.pi
u = net(jnn.concat([x, y], axis=-1)) * x * (1 - x) * y * (1 - y)   # hard u = 0 on ∂Ω
f = 2 * pi**2 * jnn.sin(pi * x) * jnn.sin(pi * y)
pde = u.dd(x) + u.dd(y) + f                                        # −∇²u = f  ⇒  residual = ∇²u + f
```

## 5. Solve

A [`jno.core`](training/index.md) collects the constraints (here the single PDE residual, driven to
zero in mean-square) and `solve()` trains through them:

```python
crux = jno.core([pde.mse])
crux.solve(epochs=10_000).plot(f"{run}/training.png")
jno.save(crux, f"{run}/model.pkl")
```

During training jNO prints one line per print-interval — `L` is the total loss, `C0, C1, …` the
per-constraint losses:

```text
Epoch  1000/10000 | L: 1.2345e-03 | C0: 1.2345e-03
```

## 6. Evaluate the prediction

[Evaluate](training/evaluation.md) the trained model on its own output — on a finer mesh if you like:

```python
pred, xt, yt = crux.eval([u, x, y], domain=jno.Shape.rect(0, 0, 1, 1, size=0.01).domain())
print(pred.shape)                       # the learned field, sampled on the fine mesh
```

---

## The same problem, the other two ways

Nothing above was PINN-specific except the trial function. The identical BVP — same domain, same
forcing, same boundary condition — goes through `jno.fem` as a **weak** form and `jno.fdm` as a
**strong** one. The forcing `fq` is the same expression, now bound to quadrature coordinates:

```python
xi, yi, _ = dom.variable("interior", split=True)
xb, yb, _ = dom.variable("boundary", split=True)
fq = 2 * pi**2 * jnn.sin(pi * xi) * jnn.sin(pi * yi)

# FEM — the weak form, with a test function v
U, V = dom.fem_symbols()
Ui, Vi = U.bind(x=xi, y=yi), V.bind(x=xi, y=yi)
u_fem = jno.fem([Ui.x * Vi.x + Ui.y * Vi.y - fq * Vi,    # ∫∇u·∇v − ∫f·v = 0
                 U(xb, yb) - 0.0]).solve()

# FDM — the strong form, collocated at the nodes, no test function
w  = dom.unknown()
wi = w.bind(x=xi, y=yi)
u_fdm = jno.fdm([-wi.d2(xi) - wi.d2(yi) - fq,            # −Δu = f
                 w(xb, yb) - 0.0]).solve()
```

Measured against the exact $u^\ast=\sin(\pi x)\sin(\pi y)$ on the same `size=0.04` mesh:

| Method | What you wrote | relative $L^2$ error |
|---|---|---|
| PINN | a network + the strong-form residual as a loss | `0.0002` |
| FEM | the weak form as a term list | `0.0002` |
| FDM | the strong form as a term list | `0.0078` |

The FDM number is the honest one to notice: strong-form collocation on an *unstructured* triangular
mesh is less accurate here than P1 FEM on the same nodes. Use `structured=True` on the domain for
the regular-grid stencils FDM is really built for.

The point is not the ranking — it is that switching method changed the *term list*, not the
framework. And because each of these is a differentiable node, any of them can be dropped into an
objective to recover a coefficient or a shape; see [Concepts](concepts.md#composition-concretely).

---

## Where to go next

- **Geometry** — build real shapes (CSG, curved boundaries, mesh density): [Domain & Geometry](Domain-and-Geometry.md).
- **Operators** — every derivative / integral you can write into a residual: [Operations](operations.md#differentiation).
- **Training** — schedules, resampling, callbacks, parallelism: [Machine learning](training/index.md).
- **Model controls** — freeze, mask, LoRA, dtype, tuning: [Operations → Part B](operations.md#part-b-operations-that-require-trainable-parameters).
- **Numerical methods** — assemble and solve a weak form or a stencil:
  [FEM](fem/index.md), [FDM](fdm.md), [RCWA](rcwa.md), and
  [solvers & preconditioners](solvers.md).
- **Tutorials** — worked end-to-end examples (PINN, operator learning, FEM, Bayesian): [Tutorials](tutorials/01-basics/laplace-1d.md).
