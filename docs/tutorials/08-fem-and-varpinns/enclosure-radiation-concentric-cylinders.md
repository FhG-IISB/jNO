# Conduction + Grey-Body Enclosure Radiation (concentric cylinders)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/enclosure_radiation_concentric_cylinders.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

Two solid rings separated by a **vacuum gap**: heat conducts from the hot inner edge through the inner
ring, **radiates across the gap** to the outer ring, then conducts out to the cold edge. Radiation is
*nonlinear* ($T^4$) and *nonlocal* — every gap-facing element exchanges with every other through the
view-factor matrix — so it cannot be a local weak term. jNO supplies the **view matrix**; you write the
grey-body radiosity **as math in `jno.np`** and couple it to the conduction FEM as a consistent surface
load. There is no `jno.radiation()` helper.

![Conduction + radiation across a vacuum gap](/jNO/assets/enclosure_radiation_concentric_cylinders.png)

## The view matrix is a first-class object

`domain.enclosure(tags)` discretises the radiating surfaces into mesh-aligned **elements** and returns a
quality-gated view factor — fully geometry-determined, so a *concave* surface keeps its self-view
($F_{22}=1-r_1/r_2$ for the outer cylinder):

```python
gap = d.enclosure(["inner_gap", "outer_gap"])   # name the surfaces once (axisymmetric=True for r,z)
gap.check()                                      # closure (Σ_j F_ij→1) + reciprocity (A_i F_ij = A_j F_ji)
F   = gap.view_factor
eps = gap.emissivity({"inner_gap": 0.8, "outer_gap": 0.6})
rho = 1.0 - eps
```

## You write the radiosity; it stays traced

```python
def q_rad(u):                        # net flux per element:  q = σ·G·T⁴, G=(I−F)(I−ρF)⁻¹diag(ε)
    Ts = gap.field(u)                # nonlocal gather: per-element temperature from the solution
    J  = jno.np.linalg.solve(jno.np.eye(gap.size) - rho[:, None] * F, eps * SIGMA * (Ts + KELVIN)**4)
    return J - F @ J
```

`gap.field(u)` gathers the per-element temperature; `gap.load(q)` scatters a per-element flux back to the
FEM nodes as the consistent surface load $\int_\Gamma q\,v\,ds$. Because the radiosity is `jno.np`, a
trainable `jno.np.parameter` emissivity flows through it for inverse problems.

## Couple and solve

The radiation BC $-k\,\partial T/\partial n = q_{rad}$ enters `fem.residual` as a load, so you solve
$A u = b - \texttt{gap.load}(q_{rad}(u))$. jNO imposes no solver; the Dirichlet conditions are
penalty-enforced (so $A$ is ill-conditioned) — a **direct** sparse solve with a Picard/Newton outer loop
on the radiation converges robustly:

```python
A, b = fem.operator                  # A is BCOO sparse — do NOT densify
lu   = scipy.sparse.linalg.splu(csr(A))
T = lu.solve(b)
for _ in range(200):                 # Picard on the (mild) radiation nonlinearity
    Tn = lu.solve(b - gap.load(q_rad(T)))
    if abs(Tn - T).max() < 1e-5: break
    T = 0.7 * T + 0.3 * Tn
```

## It matches the closed form

The coupled surface temperatures and heat flow reproduce the analytic two-surface series
$Q = 2\pi r_1\sigma(T_{s1}^4-T_{s2}^4)/(1/\varepsilon_1+(r_1/r_2)(1/\varepsilon_2-1))$ to under 1 %
(`Ts1` 874.7 K, `Ts2` 360.6 K). The radial-profile panel shows the logarithmic conduction profile in each
ring and the radiative temperature jump across the gap.

Reference: M. F. Modest, *Radiative Heat Transfer*, 3rd ed., Ch. 4–5 (view factors; the net-radiation /
radiosity method for diffuse-grey enclosures).
