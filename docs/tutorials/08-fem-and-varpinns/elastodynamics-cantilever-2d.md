# Ringing Cantilever (elastodynamics + modal analysis, 2nd-order time)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/elastodynamics_cantilever_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

The **vector** sibling of the [vibrating membrane](wave-membrane-2d.md): a plane-stress beam clamped
at the root, obeying Newton's second law for a continuum,

$$\rho\,u_{tt} = \nabla\!\cdot\sigma(u),\qquad \sigma(u)=\lambda(\nabla\!\cdot u)\,I + 2\mu\,\varepsilon(u).$$

The displacement $u=(u_x,u_y)$ carries a *second* time derivative `ui.tt`, so `jno.fem` auto-reduces
the problem to the first-order system in $y=[u,\,v{=}u_t]$ and integrates it with the
energy-conserving trapezoidal rule ($\theta=\tfrac12$).

## The weak form is the whole problem

```python
eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
weak = rho * inner(ui.tt, vi, n_contract=1) + lam * trace(eu) * trace(ep) + 2.0 * mu * inner(eu, ep, n_contract=2)
fem  = jno.fem([weak, u(xl, yl) - (0.0, 0.0), u(xi0, yi0) - (0.0, 0.0), ui0.t - (0.0, 0.0)])
```

`ui.tt` triggers the second-order route; the vector value shape `(2,)` makes it elastodynamics. The
state is $y=[u;v]$ of size $2N$, split with `fem.offsets` (`[0, N, 2N]`).

## Proving the *dynamics*, not just conservation

The trapezoidal rule conserves a quadratic invariant of **any** linear block — even a
frequency-wrong one — so energy conservation alone says nothing about the physics. To check the
*frequency* we do a small modal analysis on the assembled operators: the generalized eigenproblem
$K\varphi=\omega^2 M\varphi$ gives the fundamental bending mode $(\omega_1,\varphi_1)$. Released from
that mode at rest, the exact solution is $u(t)=\varphi_1\cos(\omega_1 t)$, so the tip must trace a
clean cosine — a direct check that the augmented $[u,v]$ block reproduces $M\ddot u + Ku = 0$ at the
right speed. As a bonus $\omega_1$ matches Euler–Bernoulli beam theory
$\omega_1\approx(1.875/L)^2\sqrt{EI/\rho A}$.

!!! warning "Soft modes need float64"
    The fundamental bending mode of a slender beam is *soft* — its modal stiffness is orders of
    magnitude below $\lVert K\rVert$ — so float32 assembly round-off shifts $\omega_1$ by a few
    percent (the beam rings at the wrong speed while energy is still conserved: a silent error). The
    script enables `jax_enable_x64` up front; `jno.fem` also **warns** when a second-order-time form
    is assembled without it.

## Result

![Tip deflection of the jNO elastodynamics solve overlaid on the analytic modal cosine over two fundamental periods; the two curves coincide.](/jNO/assets/elastodynamics_cantilever_2d.png)

The tip rings as the exact modal cosine $\varphi_1\cos(\omega_1 t)$ to rel-$L^2\approx2\times10^{-3}$
over two periods, the discrete energy is conserved, the clamped root stays fixed to machine
precision, and the FEM fundamental frequency matches beam theory to ~1%.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/elastodynamics_cantilever_2d.py:code"
```
