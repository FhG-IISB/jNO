# Two droplets merge: a diffuse-interface flow (Cahn–Hilliard–Navier–Stokes)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/droplet_merge_chns_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

Two liquid drops sit side by side, their rims 0.08 apart. Surface tension pulls them together; they
touch, fuse into one, and the merged drop relaxes toward a circle. That is a **change of topology** —
two boundaries become one — and it needs no special machinery here, because the interface is not a
mesh boundary at all. It is a field.

A phase field $\phi$ is $+1$ in the liquid and $-1$ outside, with a $\tanh$ layer of width $\sim\varepsilon$
between. It rides beside the flow (Jacqmin 1999; Yue, Feng, Liu & Shen 2004):

$$\rho\,(\partial_t\mathbf u+\mathbf u\!\cdot\!\nabla\mathbf u)=-\nabla p+\eta\,\Delta\mathbf u+\mu\nabla\phi,\qquad \nabla\!\cdot\!\mathbf u=0$$

$$\partial_t\phi+\mathbf u\!\cdot\!\nabla\phi=\nabla\!\cdot(M\nabla\mu),\qquad \mu=\lambda\Big(-\Delta\phi+\frac{\phi^3-\phi}{\varepsilon^2}\Big),\qquad \sigma=\frac{2\sqrt2}{3}\frac{\lambda}{\varepsilon}$$

The chemical potential $\mu$ is the variational derivative of the interface energy, and $\mu\nabla\phi$
is the capillary force it exerts on the fluid.

## The model is the term list

Nothing in jNO is specific to two-phase flow. The four fields are ordinary FEM symbols — a Taylor–Hood
velocity/pressure pair, and P1 fields for $\phi$ and $\mu$ — and each equation is a weak form:

```python
momentum = (RHO * dot(u_.t, v_) + RHO * dot(dot(grad_u, u_), v_) + ETA * ddot(grad_u, grad_v)
            - p_ * trace(grad_v) - mu * (phi.x * v_[0] + phi.y * v_[1]))          # capillary force μ∇φ
cahn_hilliard = phi.t * psi + (u_[0] * phi.x + u_[1] * phi.y) * psi + MOBILITY * (mu.x * psi.x + mu.y * psi.y)
chemical_potential = mu * chi - LAM * (phi * phi * phi - phi) / EPS**2 * chi - LAM * (phi.x * chi.x + phi.y * chi.y)
```

The fourth-order Cahn–Hilliard operator is split into two second-order equations through $\mu$, so plain
$H^1$ elements carry it.

## A mesh that follows the interface

The interface is where the resolution matters, and it moves. `remesh(criterion=...)` refines on any
traced field — here $1-\phi^2$, which is $\approx 1$ on the interface and $\approx 0$ in both bulks — and on a
march it is evaluated on the **live** state at every remesh, at a fixed vertex budget:

```python
phi_now = c.bind(x=xi, y=yi)
traj = fem.solve(adapt=jno.solve.remesh(criterion=1.0 - phi_now * phi_now, every=3, max_dofs=n0),
                 nonlinear=jno.solve.newton(direct=True))
```

The starting mesh is already graded toward the two rims (`size=` takes a callable). Remeshing moves
resolution to where a feature *is going*; it cannot recover what a coarse first step already lost.

## What it produces

8,636 DOFs on 798 vertices, 12 steps, three remeshes — **57 s** on this machine (GPU).

| | start | end |
|---|---|---|
| $\phi$ midway between the drops | $-0.185$ | $+1.035$ (one drop) |
| width / height of the liquid | 2.28 | 0.99 (a circle) |
| liquid area $\int(1+\phi)/2$ | 0.22006 | 0.22185 (**+0.81 %**) |

![Phase field at t = 0, 0.30 and 0.60](/jNO/assets/droplet_merge_chns_2d.png)

*The computed $\phi$ at $t = 0$, $0.30$ and $0.60$, each drawn on the adapted mesh that step ran on — the
fine band tracks the interface as the drops fuse. The colour saturates at $\pm1$; the bulk itself sits at
$\pm1.03$, the known Cahn–Hilliard shift for a curved interface.*

!!! success "Conservation, stated exactly"
    Cahn–Hilliard conserves $\int\phi$ **exactly** on a fixed mesh: summing the $\phi$ equation over the
    P1 partition of unity leaves only $\int\phi\,\nabla\!\cdot\mathbf u$, and $\phi$ lies in the P1
    pressure-test space, so the discrete continuity equation makes that term vanish. The script asserts
    it — between remeshes the liquid area is constant to $10^{-8}$. The +0.81 % over the whole run is
    therefore entirely the three state transfers, not the model.

The same model passes two static checks in `tests/test_fem_chns_droplet.py`. A single drop carries the
Laplace pressure jump $\sigma/R$ to **+4.0 %** at $\varepsilon/R = 0.12$ — an $O(\varepsilon/R)$
diffuse-interface error that shrinks with $\varepsilon$ — and its chemical potential settles at the
Gibbs–Thomson value $\sigma/2R$ to +4.3 %, while the spurious currents die away to $3\times10^{-6}$ of the
capillary velocity $\sigma/\eta$.

## Scope — what this is not

- **Planar 2-D.** A planar liquid thread cannot pinch off, so *breakup* needs an axisymmetric or 3-D run.
- **Matched density and viscosity.** A density ratio enters through $\rho(\phi)$ as a state-dependent
  mass (it assembles, and a static drop at ratio 1000 stays static), but large-ratio *dynamics* are not
  checked here.
- **A diffuse interface.** $\varepsilon/R = 0.22$ in this run; results carry an $O(\varepsilon/R)$ error,
  and a drop slowly shrinks by the Cahn–Hilliard bulk shift. The rims start about 1.4 interface widths
  apart, so the first contact is helped by diffusion across the gap, not by flow alone.

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/droplet_merge_chns_2d.py:code"
```

**References.** D. Jacqmin, *J. Comput. Phys.* **155** (1999) 96–127. P. Yue, J. J. Feng, C. Liu &
J. Shen, *J. Fluid Mech.* **515** (2004) 293–317.
