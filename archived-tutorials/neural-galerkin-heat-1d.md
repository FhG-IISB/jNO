# Neural Galerkin: evolving a network's weights in time

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/neural_galerkin_heat_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A steady VPINN or Deep-Ritz network is trained *once*. **Neural Galerkin** instead makes the network
**time-dependent through its weights** $\theta(t)$: it projects the PDE $u_t = \mathcal N(u)$ onto the
tangent space of the parametrisation and marches the resulting ODE for the weights,

$$ M(\theta)\,\dot\theta = F(\theta), \qquad M = \int \partial_\theta u\,(\partial_\theta u)^\top\,dx,
   \qquad F = \int \partial_\theta u\,\cdot\,\mathcal N(u)\,dx. $$

The **parameter-Jacobian** $J = \partial u/\partial\theta$ (N collocation points × P weights) is the
whole engine — here computed with `jax.jacfwd` over the network weights — and each step is a
regularised least-squares projection of the spatial operator $\mathcal N(u)$ onto $J$.

Method: **Neural Galerkin** — Bruna, Peherstorfer & Vanden-Eijnden, *J. Comput. Phys.* 496 (2024),
§2–3 (arXiv:2203.01360).

Test problem: 1-D heat $u_t = \nu u_{xx}$ on $[0,1]$, $u(x,0)=\sin(\pi x)$, $u=0$ at the ends —
exact $u(x,t) = e^{-\nu\pi^2 t}\sin(\pi x)$.

## Two stages: fit the IC, then march the weights

The initial condition is fit like any jNO network — a `crux.solve` regression onto $\sin(\pi x)$,
with a hard-BC ansatz $u = x(1-x)\,\mathrm{net}(x)$ that vanishes at the ends:

```python
u = (x * (1 - x) * net(x)).scalar.bind(x=x)
crux = jno.core([(u - jnn.sin(np.pi * x)).mse])
crux.solve(8000)                       # θ(0) so u(·,0) ≈ sin(πx)
```

The **time evolution is a bring-your-own-integrator loop** (the same pattern as the transient-FEM
tutorials) — Neural Galerkin marches an ODE for $\theta$, it is not gradient training:

```python
trainable, static = eqx.partition(net.module, eqx.is_array)
theta, unravel = ravel_pytree(trainable)

def u_point(th, xp):                   # the SAME ansatz, as a pure function of the weights
    mod = eqx.combine(unravel(th), static)
    return xp * (1 - xp) * jnp.ravel(mod(jnp.reshape(xp, (1,))))[0]

@jax.jit
def ng_step(th):
    J    = jax.vmap(lambda xp: jax.jacfwd(lambda t: u_point(t, xp))(th))(xs)      # (N, P)
    u_xx = jax.vmap(lambda xp: jax.grad(jax.grad(lambda z: u_point(th, z)))(xp))(xs)  # (N,)
    theta_dot = jnp.linalg.solve(J.T @ J + LAM * jnp.eye(P), J.T @ (NU * u_xx))   # Tikhonov projection
    return th + DT * theta_dot
```

## Two things that make or break it

Both were verified while building this tutorial:

- **N ≥ P.** More collocation points than trainable weights, or $\dot\theta$ is underdetermined and
  the dynamics are garbage. We use a modest network ($P\approx140$) and ~400 points.
- **Regularise the projection.** The Gram $M = J^\top J$ is notoriously ill-conditioned — a raw
  least-squares solve blows up (we saw the march diverge to >8000 % error). A small Tikhonov term
  $(J^\top J + \lambda I)$ keeps $\dot\theta$ bounded and the march stable.

## The result

![Neural Galerkin marched profile vs the analytic heat solution](/jNO/assets/neural_galerkin_heat_1d.png)

The IC fit reaches rel-$L^2 \approx 1\times10^{-3}$; marching the weights to $t=0.5$ (5000 explicit
steps) matches $e^{-\nu\pi^2 t}\sin(\pi x)$ to rel-$L^2 \approx 4\times10^{-5}$.

## What to notice

- **The weights are the state.** No spatial mesh evolves — the network's parameters carry the time
  dependence, and $J=\partial u/\partial\theta$ is the moving basis. This is why Neural Galerkin
  shines for solutions with sharp, *moving* features that a fixed grid resolves poorly.
- **`crux.solve` for the IC, a BYO loop for time.** The initial fit is ordinary jNO training; the
  evolution is a numerical ODE march, exactly like the transient-FEM tutorials hand $(M, A)$ to their
  own integrator.
- **`jax.jacfwd` over the weights** gives the parameter-Jacobian directly — the same object jNO
  exposes as `u.grad(net)` for post-hoc analysis (NTK, ENGD).
