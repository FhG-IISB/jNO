# 2D Helmholtz with a PML (complex FEM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/helmholtz_pml_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A time-harmonic scattering / radiation problem: a point source radiates outward and a **Perfectly
Matched Layer (PML)** absorbs the outgoing wave with no reflection, so a truncated box behaves like
open space. The PML is a **complex coordinate stretch** $s = 1 + i\,\sigma(x)/k$ ($\sigma$ ramps up
in a frame, $0$ in the physical core) — so the weak form has **complex** coefficients and the
solution $u$ is complex.

## Complex weak form via `1j`

A `1j` coefficient (Python's native imaginary unit) makes the weak form complex. `jno.fem` then
solves the problem with a **real-equivalent** method (it splits each term into real
$\mathrm{Re}$/$\mathrm{Im}$ sub-forms,
assembles both through the ordinary real FEM path, solves the block
$\begin{bmatrix}A_r&-A_i\\A_i&A_r\end{bmatrix}\begin{bmatrix}u_r\\u_i\end{bmatrix}=\begin{bmatrix}b_r\\b_i\end{bmatrix}$
and recombines to $u=u_r+i\,u_i$). The PML's anisotropic stretched operator reads directly:

```python
Sx, Sy = 1.0 + 1j * sx / k, 1.0 + 1j * sy / k  # complex coordinate stretch
weak = (Sy / Sx) * (ui.x * vi.x) + (Sx / Sy) * (ui.y * vi.y) - k**2 * Sx * Sy * (u * vi) - src * vi
fem = jno.fem([weak, u(xb, yb) - 0.0], quad_degree=3)  # detects complex -> real-equivalent solve
u = fem.solve()                                        # complex128 field u_r + i u_i
```

## The result

![Re(u) with PML shows clean concentric outgoing wavefronts absorbed at the PML interface; without
PML the wave reflects off the walls into a standing-wave pattern; |u| decays into the
frame.](/jNO/assets/helmholtz_pml_2d.png)

Left: $\mathrm{Re}(u)$ **with** the PML — concentric outgoing wavefronts, smoothly absorbed at the
dashed PML interface. Middle: the same problem **without** the PML ($\sigma=0$, $u=0$ walls) — the
wave reflects and resonates. Right: $|u|$, decaying into the absorbing frame.

## What to notice

- A complex coefficient (a `1j`) is the only signal needed — `fem.is_complex` is `True` and
  `fem.solve()` returns a `complex128` field.
- The complex solve uses the **real-equivalent block**; the underlying real FEM backend is never
  asked to assemble a complex matrix.
- **PML quality, no analytic solution required:** a *converged* PML's physical-core field is
  independent of the absorber strength $\sigma_0$ — here the relative change from $\sigma_0=40$ to
  $60$ is $\sim 9\times10^{-4}$, i.e. the truncation is effectively reflection-free.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/helmholtz_pml_2d.py:code"
```
