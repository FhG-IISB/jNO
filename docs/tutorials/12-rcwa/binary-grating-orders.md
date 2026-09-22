# Binary Grating — where the diffraction orders go

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/12_rcwa/01_binary_grating_orders.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

A plane wave at normal incidence hits a periodic slab — a high-index ridge filling half of each unit
cell. The **grating equation** decides which orders escape as propagating waves:

$$\sin\theta_m = \sin\theta_i + m\,\frac{\lambda}{\Lambda}
\qquad\Longrightarrow\qquad
\text{order } m \text{ propagates} \iff \left|m\,\frac{\lambda}{\Lambda}\right| \le 1 .$$

Everything else is evanescent. So sweeping the period $\Lambda$ through $\lambda$ switches the $\pm1$
orders on at $\Lambda=\lambda$ — the **Rayleigh anomaly**. Below it the structure is *sub-wavelength*:
only $m=0$ survives and the slab behaves as a homogeneous effective medium, which is the regime
metasurfaces work in. Above it the ±1 orders open, and the very same slab sends most of what it
transmits into them.

## The term list is the whole problem

It is the scalar-Helmholtz weak form you would hand `jno.fem` — nothing RCWA-specific in it:

```python
sol = jno.rcwa([
    ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi),  # ∇u·∇v − k₀²·ε·u·v
    -(1j * K0 * ut) * vt,                    # outgoing radiation (top ambient)
    -(1j * K0 * ub - 2j * K0) * vb,          # incident plane wave + radiation (bottom)
    ul - ur,                                 # Floquet periodicity in x
    uf - ubk,                                # Floquet periodicity in y
], orders=160, grid=256).solve()
```

`jno.rcwa` infers the period from the Floquet ties, the ambients from the two z-normal radiation
faces, the layer stack and $\varepsilon$ from the volume coefficient, and the wavelength from
$\varepsilon$ in the vacuum superstrate. It infers the **polarization** from the equation: the ridges
run along $y$, and for such a grating the scalar equation is exactly Maxwell's for $E_y$, so it is solved
TE — the same equation `jno.fem` would solve (see [Polarization](../../rcwa.md#polarization)). Only the
truncation is genuinely yours: `orders` Fourier terms and a `grid` for sampling $\varepsilon$. At these
values the reflectance at $\Lambda=1.5$ is within $2\times10^{-3}$ of an independent RCWA with exact
Fourier coefficients (0.8441). Inspect what was inferred with `rc.spec`, which works **without** the
`[rcwa]` backend:

```python
RcwaSpec(period=(1.5, 0.3), layers=3, wavelength=1.0, k_in=(0.0, 0.0), source_face='bottom')
rc.spec.polarization       # (0j, (1+0j)) -- E along y
rc.spec.polarization_from  # 'y: every layer is invariant along y and the incidence has no y component, ...'
```

!!! tip "The mesh does not have to resolve the pattern"
    Because `eps` is an analytic `jno.fn`, it is evaluated **directly on the RCWA grid** rather than
    interpolated off the mesh. The tetrahedral mesh only carries the tags and periodicity, so
    `size=0.25` is plenty — the grating profile stays exact.

## Two regimes, one structure

```text
Λ = 0.6  (λ/Λ = 1.667, sub-wavelength) — grating equation allows m ∈ [0]
  R = 0.00187   T = 0.99813   R + T = 1.000000
  T(-1) = 0.00000   T(+0) = 0.99813   T(+1) = 0.00000

Λ = 1.5  (λ/Λ = 0.667, diffractive)    — grating equation allows m ∈ [-1, 0, 1]
  R = 0.84594   T = 0.15406   R + T = 1.000000
  T(-1) = 0.06879   T(+0) = 0.01648   T(+1) = 0.06879
```

At $\Lambda=0.6$ the slab is nearly transparent: as an effective medium it is close to a half-wave
layer, and only the straight-through order exists. $\Lambda=1.5$ happens to sit on the flank of a narrow
reflection resonance of this slab — an independent RCWA puts $R$ at 0.35 for $\Lambda=1.44$ and 0.87
for $1.49$ — so there it reflects 85 %, and of what it transmits, 89 % goes into $\pm1$. At $\Lambda=0.6$ those orders are *exactly* dark
— not small, zero — because they are not propagating solutions at all.

## What is verified

There is no closed-form field here, but four statements are exact and are asserted in the script:

| Check | Holds to |
|---|---|
| Energy conservation $R+T=1$ (lossless) | $10^{-15}$ |
| Grating-equation cutoff: orders outside $\lvert m\lambda/\Lambda\rvert\le1$ carry zero power | exactly `0.0` |
| Mirror symmetry $T(+1)=T(-1)$ at normal incidence | $\approx 10^{-10}$ |
| Completeness: propagating orders sum to $T$ | $10^{-6}$ |

!!! measured "The $\pm1$ symmetry floor does not improve with `orders`"
    $\lvert T(+1)-T(-1)\rvert$ measures $7.5\times10^{-11}$ at `orders=160` and $6.9\times10^{-11}$ at
    `orders=320` — it is **not** Fourier truncation. The $\pm1$ pair is degenerate at normal incidence
    and the modal eigensolve splits it at its own floor. Energy conservation, computed from the same
    solution, holds to machine precision. Assert against the measured floor, not against zero.

!!! danger "$\Lambda = m\lambda$ is singular"
    There order $m$ is exactly grazing — a **Wood–Rayleigh anomaly** — and the modal problem is
    degenerate. `jno.rcwa` raises with the period and wavelength named rather than returning a NaN
    efficiency, so the sweep below deliberately steps around $\Lambda=1$ and $\Lambda=2$.

## Result

![Diffraction efficiency against grating period. Through the shaded sub-wavelength region the ±1 orders are flat at exactly zero; at period equal to wavelength they switch on and take most of the transmitted power, while R+T stays pinned at one across the whole sweep.](/jNO/assets/rcwa_binary_grating.png)

The $\pm1$ curve is flat at exactly zero through the shaded region and turns on at $\Lambda=\lambda$,
matching the cutoff to the sweep's resolution; $R+T$ stays at $1$ across all 13 periods. Beyond the
cutoff the $\pm1$ orders carry up to about 0.6 of the incident power; the sweep's coarse steps pass over
the resonance near $\Lambda=1.5$ described above.

## Going further

Every efficiency is a **differentiable** JAX scalar, so this is already an inverse-design objective —
put a `jno.np.parameter` in `eps` and `jax.grad` a target order fraction through the modal solve. See
[RCWA](../../rcwa.md) for anisotropic $\hat\varepsilon$, in-plane PML, internal sources, `sol.jones()`
for polarization, and `sol.aerial()` for imaging.

## Full script

```python
--8<-- "tutorial_examples/12_rcwa/01_binary_grating_orders.py:code"
```
