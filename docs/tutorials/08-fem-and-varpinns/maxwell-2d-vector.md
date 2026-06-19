# Time-harmonic 2D Maxwell — in-plane vector $E$ (complex vector FEM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/maxwell_2d_vector.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

Frequency-domain Maxwell for the in-plane electric field $E = (E_x, E_y)$ (TE polarisation) is the
**curl–curl** equation

$$ \nabla\times(\nabla\times E) \;-\; k^2\,E \;=\; J, \qquad k^2 = \omega^2\mu\varepsilon, $$

with $E$ **complex-valued** ($k^2$ is complex in a lossy medium). This is the genuine *vector* complex
problem — not a scalar Helmholtz.

## A complex vector = two coupled real vector fields

A complex field has no native FEM unknown in jNO. A complex **vector** is carried as **two real vector
fields** $(E_r, E_i)$ — for a 2-D field that is $4$ real DOFs per node — and the complex equation is
split into its real and imaginary parts, giving a **coupled multifield system** that `jno.fem`
assembles and solves with the very same machine behind the two-temperature and Stokes examples. (At
the trace level this is what `placeholder.vector.complex` → `ComplexVectorView` exposes: `.real` /
`.imag` are the two vector parts.)

```python
Er, Pr = d.fem_symbols(value_shape=(2,), names=("Er", "Pr"), order=2)  # real part of E + its test
Ei, Qi = d.fem_symbols(value_shape=(2,), names=("Ei", "Qi"), order=2)  # imaginary part of E + its test

curl = lambda F: F.x[1] - F.y[0]        # 2-D scalar curl  ∂Fy/∂x − ∂Fx/∂y
div  = lambda F: F.x[0] + F.y[1]
# real/imag split of  curl(curl E) + s·div-penalty − k²·E − J ,  k² = KR + i·KI
eq_re = curl(erb)*curl(prb) + s*div(erb)*div(prb) - (KR*dot(erb,prb) - KI*dot(eib,prb)) - dot(Jr, prb)
eq_im = curl(eib)*curl(qib) + s*div(eib)*div(qib) - (KR*dot(eib,qib) + KI*dot(erb,qib)) - dot(Ji, qib)
fem = jno.fem([eq_re, eq_im, *boundary_conditions])   # ordinary real coupled solve
```

Two details for a **nodal** (Lagrange) discretisation of curl–curl:

- the scalar curl is `F.x[1] - F.y[0]` — *grad-then-index*, because the backend differentiates the
  trial, not a component of it (`F[1].x` is not supported);
- nodal elements need a **grad–div penalty** $+\,s\,(\nabla\!\cdot E)(\nabla\!\cdot v)$ to suppress the
  spurious curl kernel. It is **consistent** here because the exact field is divergence-free, so the
  penalty vanishes at the solution. With it, P1 converges $\sim\!\mathcal{O}(h^2)$ and **P2 is
  essentially exact**.

## The result

![Left: |E|, the computed magnitude, a smooth cellular pattern. Right: the computed Re(E) vector
field, a rotational (curl-type) field.](/jNO/assets/maxwell_2d_vector.png)

Left: $|E| = \sqrt{|E_x|^2 + |E_y|^2}$ (computed). Right: the computed $\mathrm{Re}(E)$ as a vector
field — the rotational, divergence-free structure of a curl eigenmode. Both panels are the **actual
finite-element solution**, nothing painted in.

## What to notice

- **Verified, not asserted:** a manufactured divergence-free $E$ with $\mathrm{Re}(E)\neq\mathrm{Im}(E)$
  and a genuinely complex $k^2 = 30 + 4i$ (so the real and imaginary parts truly couple). At P2 the
  recovered field matches the closed-form $E$ to $L^2$ relative error $\approx 5\times10^{-4}$.
- **Generality by reuse:** complex lowers onto the coupled-multifield path, so it inherits linear,
  nonlinear and transient for free — and the *same* trace expression drives a PINN.
- Honest scope: this uses nodal Lagrange elements with grad–div stabilisation (the right tool for a
  smooth field); general Maxwell with reentrant corners wants Nédélec edge elements.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/maxwell_2d_vector.py"
```
