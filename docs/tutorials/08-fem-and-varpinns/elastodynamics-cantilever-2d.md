# Ringing Cantilever (elastodynamics + modal analysis, vector 2nd-order in time)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/elastodynamics_cantilever_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

The transient sibling of the [static cantilever](linear-elasticity-cantilever.md). A plane-stress
beam clamped at the root obeys $\rho\,u_{tt}=\nabla\!\cdot\sigma(u)$ — **vector** displacement with a
**second time derivative**. `jno.fem` reduces it to the first-order block in $y=[u,v{=}u_t]$ and
integrates with the energy-conserving trapezoidal rule.

## Vector + second order in one weak form

`fem_symbols(value_shape=(2,), order=2)` gives a P2 vector field (constant-strain TRI3 is too stiff
in bending); `ui.tt` makes it second order. The spatial part is the same isotropic elasticity form as
the static beam:

```python
u, phi = d.fem_symbols(value_shape=(2,), order=2)
eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
weak = rho * inner(ui.tt, vi, n_contract=1) \
     + lam * trace(eu) * trace(ep) + 2.0 * mu * inner(eu, ep, n_contract=2)
fem = jno.fem([weak, u(xl, yl) - (0.0, 0.0), ...])   # clamped root
```

## Proving the dynamics with a modal release

Energy conservation alone only shows the *integrator* is conservative — the trapezoidal rule
conserves a quadratic invariant of **any** linear block, even a mis-assembled one. To check the
*frequency* is right, take the assembled mass/stiffness blocks of the augmented system and solve the
generalized eigenproblem $K\varphi=\omega^2 M\varphi$ for the fundamental mode, then release the beam
from it:

```python
M_uu, K_uu = M[:N, :N], A[N:, :N]              # blocks of the augmented system (y = [u; v])
evals, evecs = scipy.linalg.eigh(K_uu[free, free], M_uu[free, free])
omega1, phi1 = np.sqrt(evals[0]), evecs[:, 0]  # fundamental frequency + shape
y = np.concatenate([phi1, np.zeros(N)])        # release from the mode, at rest
```

Released from a pure mode the exact solution is $u(t)=\varphi_1\cos(\omega_1 t)$, so the tip traces a
clean cosine — a direct check that the $[u,v]$ block reproduces $M\ddot u + Ku = 0$ at the right
frequency.

## What to notice

- A single vector field with `ui.tt` covers elastodynamics — no first-order rewrite by hand.
- The tip rings as the analytic $\varphi_1\cos(\omega_1 t)$ to well under 1 % over two periods, total
  energy is conserved, and the clamped root holds to machine precision (the vector Dirichlet rows of
  the augmented system are exact).
- `jno.fem` doubles as a **modal-analysis** tool: `fem.M` / `fem.operator.A` feed straight into a
  generalized eigensolver, and the fundamental frequency matches Euler–Bernoulli beam theory
  $\omega_1\approx(1.875/L)^2\sqrt{EI/\rho A}$ to ~1 %.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/elastodynamics_cantilever_2d.py"
```
