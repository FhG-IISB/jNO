# Adaptive mesh on top of the inverse design (minimal mesh)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/adaptive_inverse_lshape.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

Adaptive refinement usually resolves a *forward* solution. Here we wrap it around an **inverse
solve**: at every round we recover an unknown parameter on the current mesh, *then* estimate,
mark and refine. The mesh ends up minimal for the **recovered design**, and the recovered
parameter **de-biases toward the truth** as the singularity is resolved — accuracy per degree of
freedom that uniform refinement cannot match.

The model is a reaction–diffusion problem on the re-entrant **L-shape**, whose harmonic corner mode
$u_\text{sing}=r^{2/3}\sin(2\varphi/3)$ is *value*-singular at the corner:

$$-\nabla^2 u + \kappa\,u = f,\qquad f=\kappa_\text{true}\,u_\text{sing},\qquad u|_{\partial\Omega}=u_\text{sing},\qquad \text{unknown scalar }\kappa.$$

At $\kappa=\kappa_\text{true}$ the exact solution is exactly $u_\text{sing}$. Because $\kappa$ multiplies
the **mass (reaction) term** it enters through the operator, so the inverse is differentiable via
implicit diff — a parameter in the Dirichlet data would not be. Observations are the closed-form
$u_\text{sing}$ at mesh nodes, **weighted to the corner** ($r<0.2$): that is exactly where a coarse
mesh under-resolves the singularity and biases the fit, and exactly the bias adaptation removes.

## The loop, in one call

Each round is a full differentiable inverse solve on the frozen mesh, followed by a
[Zienkiewicz–Zhu](https://doi.org/10.1002/nme.1620240206) estimate → Dörfler mark →
`domain.refine`. `run_adaptive_inverse` drives it; you supply `build_inverse(domain) -> (crux, state_op)`:

```python
from jno.utils.solver.fem_adapt import AdaptSpec, run_adaptive_inverse

kappa = jno.np.parameter((1,), name="kappa"); kappa.optimizer(optax.adam(1e-1))
best = {}

def build_inverse(d):
    if "k" in best:                                    # warm-start the finer round
        kappa.initialize(jax.nn.initializers.constant(best["k"]))
    s, w = corner_observations(d)                      # closed-form obs, corner-weighted
    fem = reaction_diffusion_fem(d, kappa)             # kappa in the mass term (operator)
    return jno.core([(w * (fem.solve() - s)).mse], domain=dummy), fem.solve()

def readout(crux):                                     # optimized value lives in the crux
    best["k"] = float(np.asarray(crux.eval([kappa])).reshape(-1)[0]); return best["k"]

run_adaptive_inverse(d, build_inverse,
                     AdaptSpec(theta=0.6, max_iters=6, refine_factor=1.6, eps=0.01),
                     n_opt=250, readout=readout)
```

`eps` is the **"good enough" stop**: the loop keeps refining until the recovered κ stops
changing between rounds by more than `eps` (‖Δκ‖/‖κ‖), then ends — so you don't guess a
round count. It uses a **patience of 2** (two consecutive rounds under `eps`) because a
single flat step can be a false plateau. Honest framing: this is a *plateau detector* ("κ
has stopped moving as I refine"), not a certified error bound — the lever for more accuracy
is the `max_iters` / `max_dofs` budget. `max_iters` remains a hard cap; whichever fires first
wins.

Refinement is a non-differentiable **outer** Python loop; each `crux.solve` is a fully
differentiable inverse solve on the currently frozen mesh, so gradients reach `kappa` unchanged. The
optimized value lives in the `crux` instance (not written back to the `jno.np` object), so we snapshot
it in `readout` and reseed the next round with `kappa.initialize(...)` to warm-start.

## The result

![Top: four L-shaped meshes refining hard at the re-entrant corner, tinted by the recovered singular
state, labelled 41→663 dofs with kappa climbing 3.82→4.93. Bottom: a log-log plot of recovered-kappa
error versus mesh DOFs, with the adaptive curve dropping well below the uniform baseline.](/jNO/assets/adaptive_inverse_lshape.png)

Starting from a deliberately wrong guess $\kappa=2$ (truth $5$), the recovered value marches
$3.82\to4.93$ as the corner is resolved, and with a low `eps` and a bigger budget it keeps going —
past $\kappa\approx4.99$ at a few thousand DOFs. At matched cost the adaptive run reaches
$|\kappa-\kappa_\text{true}|\approx0.07$ with **663 DOFs**, versus $\approx0.19$ for uniform at **1102
DOFs** — roughly a factor of a few fewer DOFs for the same parameter accuracy. `eps` sets *when* it
stops; `max_iters` / `max_dofs` set *how far* it can go.

## What to notice

- **The mesh is minimal for the *recovered* design**, not for a fixed forward guess — the estimator
  runs on the state at the current recovered `kappa` each round.
- **The parameter de-biases with resolution:** a coarse mesh under-resolves the corner singularity
  and systematically biases the fit; refining there removes it.
- **Put the parameter in the operator, not the Dirichlet data.** `kappa` multiplies the mass term and
  flows through `implicit`-diff; a parameter in the boundary data is not currently differentiable.
- **Scalar / low-dimensional parameters only.** A *field* parameter tied to mesh vertices changes
  shape on every remesh and would need solution transfer between the non-nested meshes.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/adaptive_inverse_lshape.py"
```
