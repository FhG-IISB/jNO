# Inverse problem on a complex domain (FEM tomography)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/inverse_conductivity_lshape.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

The differentiable-FEM story meets the complex-geometry one. On an **L-shaped** part we recover a
hidden conductivity field $k(x)$ — a buried high-conductivity inclusion — from the measured response
to a known source, by differentiating the FEM solve end to end:

$$\text{forward: } -\nabla\!\cdot(k(x)\,\nabla u) = f,\quad u|_{\partial\Omega}=0,\qquad \text{unknown } k(x)>0.$$

## The whole inverse problem, in a few lines

`k = jno.np.parameter(phi)` is a trainable P1 field on the trial space. `fem.solve()` is the
**differentiable** forward solve, and `crux.solve` minimises the data misfit plus an H1 smoothness
prior (field inversion is ill-posed without one):

```python
d = jno.domain([[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]]).build_mesh(0.06)   # L-shape, not a square
...
k   = jno.np.parameter(phi, name="k")
fem = jno.fem([k * (ui.x*vi.x + ui.y*vi.y) - f*vi, u(xb, yb) - 0.0])
k.initialize(jax.nn.initializers.constant(1.0))                          # start from a flat guess
k.optimizer(optax.adam(2e-2))
crux = jno.core([(fem.solve() - u_obs).mse, 2e-3 * k.regularize("h1seminorm").mean], domain=...)
crux.solve(700)                                                          # gradients flow through the solve
```

The synthetic data `u_obs` comes from the same assembly evaluated at the true field —
`fem.operator.evaluate({"k": k_true})` — so there is one weak form for both the forward and the
inverse direction.

## The result

![Two L-shaped panels: the true conductivity field with a single bright Gaussian inclusion in the
lower arm, and the recovered field, which reproduces the inclusion at the same location with nearly
the same peak.](/jNO/assets/inverse_conductivity_lshape.png)

The reconstruction recovers the buried inclusion at the right place with nearly the right peak
($\sim$2.0 vs 2.2) — rel-L2 $\sim2.5\times10^{-2}$.

## What to notice

- **The complex domain changes nothing about the workflow:** the L-shape is one vertex list; the
  inverse machinery is identical to a square.
- **`fem.solve()` is differentiable** — `crux.solve` backpropagates the data misfit through the
  linear FEM solve to every nodal value of `k(x)`.
- **Field inversion needs a prior:** `k.regularize("h1seminorm")` keeps the ill-posed reconstruction
  smooth; without it the recovered field is dominated by noise.
- The same pattern recovers a *scalar* parameter (drop the field, use `jno.np.parameter((1,))`) or a
  *transient* coefficient (assemble with `time=...` and train through the trajectory).

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/inverse_conductivity_lshape.py"
```
