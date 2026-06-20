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

## A complex field is one symbol — `complex=True`

`domain.fem_symbols(..., complex=True)` returns a **complex** trial/test. You write the weak form once
with ordinary complex algebra (`*` is the complex product, `1j` is just the imaginary unit, `.real` /
`.imag` / `.conj` / `.dot`), and `jno.fem` lowers `weak.real` for you. Under the hood a complex vector
is carried as **two coupled real vector fields** $(E_r, E_i)$ — $4$ real DOFs per node — and the
real-part lowering lands on the **coupled multifield system** `jno.fem` already assembles for the
two-temperature and Stokes examples. There is no separate "complex solver": one notion of complex,
`1j` everywhere, the tracer carries the rest.

```python
E, v = d.fem_symbols(value_shape=(2,), complex=True, order=2)   # a complex vector field + its test
Eb, vb = E.bind(x=x, y=y), v.bind(x=x, y=y)

curl = lambda F: F.x[1] - F.y[0]        # 2-D scalar curl  ∂Fy/∂x − ∂Fx/∂y
div  = lambda F: F.x[0] + F.y[1]
k2 = KR + 1j*KI                         # complex coefficient — a plain Python complex
J  = jno.complex(Jr, Ji)                # complex forcing (data) = re + 1j*im

weak = curl(Eb)*curl(vb) + s*div(Eb)*div(vb) - k2*Eb.dot(vb) - J.dot(vb)   # `*` is the complex product
fem  = jno.fem([weak.real, *boundary_conditions])   # `.real` lowers onto the coupled real solve
```

`weak.real` is one expression whose real part mixes both test fields; `jno.fem` distributes it into the
per-field ($E_r$, $E_i$) blocks automatically. Two details for a **nodal** (Lagrange) discretisation of
curl–curl:

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
