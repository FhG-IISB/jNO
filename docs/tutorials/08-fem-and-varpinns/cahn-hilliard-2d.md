# Cahn–Hilliard Phase Separation (transient · nonlinear · 4th-order)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/cahn_hilliard_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

The Cahn–Hilliard equation models phase separation as the mass-conserving gradient flow of the
Ginzburg–Landau free energy

$$
E[u] = \int \Bigl[ \tfrac14 (u^2-1)^2 + \tfrac{\kappa}{2}\,|\nabla u|^2 \Bigr]\,dx ,
\qquad
\partial_t u = \Delta\mu,\quad \mu = u^3 - u - \kappa\,\Delta u .
$$

Eliminating the chemical potential $\mu$ gives a single **4th-order** PDE
$\partial_t u = \Delta(u^3-u) - \kappa\,\Delta^2 u$, whose primal weak form needs an **$H^2$-conforming**
space — exactly the Argyris $C^1$ element. Three `jno.fem` capabilities compose in one form: a **transient**
term, a genuinely **nonlinear** term, and the **biharmonic** $\int\Delta u\,\Delta v$.

```python
u, phi = d.fem_symbols(space="Argyris")
ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
form = (ui.t * vi                                              # transient
        + (3*ui*ui - 1) * (ui.x*vi.x + ui.y*vi.y)             # nonlinear  ∇(u³-u)·∇v
        + KAPPA * laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]))   # biharmonic (C¹)
fem = jno.fem([form, u(ci[0], ci[1]) - u0])
```

## Bring your own (direct) solver for a stiff operator

The biharmonic is $h^{-4}$-conditioned, so `jno.fem`'s default matrix-free Newton–Krylov step converges
slowly. `solve()` lets you supply your own integrator; here a backward-Euler step with a **dense direct**
Newton solve — assembled only from the block's `mass` / `residual` / `jacobian` — is ~100× faster:

```python
def direct_newton(block, args, save_ts):
    def step(uprev, t_next):
        mass = dense(block.mass(t_next, args))
        def newton(_i, un):                                    # G = M(uₙ-uₚ)/dt + R(uₙ)
            g = mass @ (un - uprev)/dt + block.residual(un, t_next, args)
            jac = mass/dt + dense(block.jacobian(un, t_next, args))
            return un + jnp.linalg.solve(jac, -g)              # direct solve, not Krylov
        un = lax.fori_loop(0, 4, newton, uprev)
        return un, un
    _, ys = lax.scan(step, block.state0, save_ts[1:])
    return ys

traj = fem.solve(solve_fn=direct_newton)
```

## The physics *is* the test

Cahn–Hilliard has two exact invariants — no manufactured solution needed:

1. **Mass conservation.** Testing with $v\equiv 1$ kills every spatial term, so $\tfrac{d}{dt}\int u = 0$.
2. **Energy dissipation.** $\tfrac{dE}{dt} = -\int|\nabla\mu|^2 \le 0$: $E$ is a strict Lyapunov functional.

Driving a $+1$ droplet in a $-1$ sea, curvature-driven coarsening shrinks it while the discrete solution
conserves mass to machine precision and dissipates the discrete free energy **monotonically**:

```
Cahn–Hilliard (Argyris C¹) — droplet coarsening:
  mass ∫u:  -0.409597 → -0.409597   (drift 1.83e-15)
  energy E:  0.27939 → 0.19830      (monotone: True)
  max|u|:    0.986  (bounded — no overshoot past the ±1 wells)
```

![Free energy dissipates monotonically while mass is conserved; the +1 droplet coarsens in the -1 sea.](/jNO/assets/cahn_hilliard_2d.png)

The mass line is dead flat while the free energy falls monotonically — the two defining properties of the
Cahn–Hilliard flow, reproduced by the conforming $C^1$ discretisation (energy computed by per-cell
quadrature reconstruction of $u$ and $\nabla u$ from the solution DOFs).

!!! note "References"
    J.W. Cahn, J.E. Hilliard, *Free energy of a nonuniform system. I. Interfacial free energy*,
    J. Chem. Phys. **28** (1958) 258–267. J.H. Argyris, I. Fried, D.W. Scharpf (1968); R.C. Kirby,
    SMAI J. Comput. Math. **4** (2018) — the $C^1$ element.
