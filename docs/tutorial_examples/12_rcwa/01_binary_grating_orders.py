# --8<-- [start:code]
"""Binary dielectric grating: where the diffraction orders go, via ``jno.rcwa``.

A plane wave at normal incidence hits a periodic slab -- a high-index ridge filling half of each
unit cell. The **grating equation** decides which orders escape as propagating waves,

    sin(θ_m) = sin(θ_i) + m·λ/Λ ,   so order m propagates  ⟺  |m·λ/Λ| ≤ 1  (θ_i = 0 here),

and everything else is evanescent. Sweeping the period Λ through λ therefore switches the ±1
orders on at Λ = λ -- the **Rayleigh anomaly** (Rayleigh, *Proc. R. Soc. A* 79, 399, 1907). Below
it the structure is *sub-wavelength*: only m=0 survives, all the transmitted light goes straight
ahead, and the slab acts as a homogeneous effective medium -- the regime metasurfaces work in.
Above it the same slab is a **beam splitter**.

The whole problem is the scalar-Helmholtz term list you would hand ``jno.fem``:

    ∇u·∇v − k₀²·ε·u·v = 0        volume (ε carries the pattern)
    −i k₀ u·v                     outgoing radiation, top ambient
    −(i k₀ u − 2i k₀)·v           incident plane wave + radiation, bottom ambient
    u(left) − u(right) = 0        Floquet periodicity in x
    u(front) − u(back) = 0        Floquet periodicity in y

``jno.rcwa`` infers the period from the Floquet ties, the ambients from the two z-normal radiation
faces, the layer stack and ε from the volume coefficient, and the wavelength from ε in the vacuum
superstrate -- so only ``orders`` (the Fourier truncation) is genuinely ours to choose. Because ε
is an analytic ``jno.fn``, it is sampled directly on the RCWA grid, so the tetrahedral mesh only
has to carry the tags and can stay coarse. The modal method itself is Moharam & Gaylord,
*J. Opt. Soc. Am.* 71, 811 (1981).

What is verified (no analytic field solution needed -- these are exact statements):
  1. energy conservation, R + T = 1, on a lossless structure;
  2. the grating-equation cutoff -- orders outside |m·λ/Λ| ≤ 1 carry exactly zero power;
  3. mirror symmetry -- T(+1) = T(−1) for a symmetric grating at normal incidence (to ~2e-8: the
     ±1 pair is degenerate here, and the modal eigensolve splits it at a floor that more Fourier
     orders do not lower -- unlike the energy balance, which holds to machine precision);
  4. completeness -- the propagating order efficiencies sum to the total transmission.

Λ = m·λ is excluded throughout: there order m is exactly grazing (a Wood-Rayleigh anomaly), the
modal problem is singular, and ``jno.rcwa`` raises rather than handing back a NaN efficiency.
"""

import jax

jax.config.update("jax_enable_x64", True)  # modal eigen-decomposition of a high-contrast stack

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jno  # noqa: E402

WL = 1.0  # vacuum wavelength -- the yardstick every length below is measured against
EPS_RIDGE = 11.0  # ridge permittivity (n ≈ 3.32, silicon-like)
Z0, Z1 = 0.4, 0.6  # slab occupies z ∈ (0.4, 0.6); vacuum ambients above and below
PY = 0.3  # y-period, deliberately sub-wavelength -> a 1-D grating
DUTY = 0.5  # ridge fills the left half of each cell
K0 = 2 * jnp.pi / WL
ORDERS = 40


def grating(period, orders=ORDERS):
    """Solve one binary grating and return its order-resolved efficiencies."""
    d = jno.Shape.box(0, 0, 0, period, PY, 1.0, size=0.25).domain()  # coarse: ε is analytic
    e = 1e-6
    d.tag("bottom", lambda x, y, z: z < e)
    d.tag("top", lambda x, y, z: z > 1.0 - e)
    d.tag("left", lambda x, y, z: x < e)
    d.tag("right", lambda x, y, z: x > period - e)
    d.tag("front", lambda x, y, z: y < e)
    d.tag("back", lambda x, y, z: y > PY - e)

    u, v = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)

    def on(tag):  # bind u, v on a named face
        s = d.variable(tag, split=True)
        return u.bind(x=s[0], y=s[1], z=s[2]), v.bind(x=s[0], y=s[1], z=s[2])

    (ut, vt), (ub, vb) = on("top"), on("bottom")
    ul, ur, uf, ubk = on("left")[0], on("right")[0], on("front")[0], on("back")[0]

    ridge = jno.fn(lambda x, y, z: jnp.where((Z0 < z) & (z < Z1) & (x < DUTY * period), 1.0, 0.0), [xi, yi, zi])
    eps = 1.0 + (EPS_RIDGE - 1.0) * ridge  # 1 in the ambients, EPS_RIDGE in the ridge

    sol = jno.rcwa(
        [
            ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi),  # ∇u·∇v − k₀²εuv
            -(1j * K0 * ut) * vt,  # outgoing (top)
            -(1j * K0 * ub - 2j * K0) * vb,  # incident + outgoing (bottom)
            ul - ur,  # Floquet, x
            uf - ubk,  # Floquet, y
        ],
        orders=orders,
    ).solve()

    real = lambda a: float(jnp.real(jnp.asarray(a)))  # noqa: E731
    return dict(
        T=real(sol.efficiency("T")),
        R=real(sol.efficiency("R")),
        orders={m: real(sol.order(m, 0)) for m in (-2, -1, 0, 1, 2)},
    )


def propagating(period):
    """The transmitted orders the grating equation lets escape at normal incidence."""
    return [m for m in (-2, -1, 0, 1, 2) if abs(m * WL / period) <= 1.0]


# --- two regimes, side by side ------------------------------------------------------------------
SUB, DIF = 0.6, 1.5  # sub-wavelength (Λ < λ) and diffractive (Λ > λ)
res = {P: grating(P) for P in (SUB, DIF)}

print("\nBinary dielectric grating -- where the orders go   (λ = 1, ε_ridge = 11, 50% duty)")
for P, r in res.items():
    prop = propagating(P)
    tag = "sub-wavelength" if P < WL else "diffractive"
    print(f"\n  Λ = {P}  (λ/Λ = {WL / P:.3f}, {tag}) -- grating equation allows m ∈ {prop}")
    print(f"    R = {r['R']:.5f}   T = {r['T']:.5f}   R + T = {r['R'] + r['T']:.6f}")
    print("    " + "   ".join(f"T({m:+d}) = {r['orders'][m]:.5f}" for m in (-1, 0, 1)))

for P, r in res.items():
    prop, o = propagating(P), r["orders"]
    assert abs(r["R"] + r["T"] - 1.0) < 1e-6, f"lossless structure must conserve energy (Λ={P})"
    for m in o:
        if m not in prop:
            assert o[m] < 1e-9, f"order {m} is evanescent at Λ={P} but carries {o[m]:.2e}"
    # The ±1 pair is degenerate at normal incidence, so the modal eigensolve splits it at its own
    # floor: measured |T(+1) − T(−1)| ≈ 2e-8, and it does NOT shrink with `orders` (2.09e-8 at 40,
    # 1.93e-8 at 80) — it is not Fourier truncation. Energy conservation, by contrast, holds to 1e-15.
    assert abs(o[+1] - o[-1]) < 1e-6, f"symmetric grating at normal incidence: T(+1) must equal T(-1) (Λ={P})"
    assert abs(sum(o[m] for m in prop) - r["T"]) < 1e-6, f"propagating orders must sum to T (Λ={P})"

# the physical punchline, stated as a test rather than prose
assert res[SUB]["orders"][+1] == 0.0, "below the Rayleigh cutoff the ±1 orders must be exactly dark"
assert res[DIF]["orders"][+1] + res[DIF]["orders"][-1] > 0.8 * res[DIF]["T"], (
    "above the cutoff this grating should throw most of its light into ±1"
)

# --- sweep the period across the Rayleigh anomaly at Λ = λ ---------------------------------------
# Λ = m·λ puts order m exactly grazing (a Wood-Rayleigh anomaly): the modal problem is singular
# there and `jno.rcwa` raises rather than returning a NaN efficiency. Step around Λ = 1 and Λ = 2.
periods = np.concatenate([np.linspace(0.55, 0.95, 5), np.linspace(1.05, 1.95, 8)])
sweep = [grating(float(P)) for P in periods]
print(f"\n  swept {len(periods)} periods across the Rayleigh anomaly at Λ = λ = {WL}")

first_on = next(P for P, r in zip(periods, sweep) if r["orders"][+1] > 1e-6)
assert first_on > WL, f"the ±1 orders must not switch on below Λ = λ (got {first_on})"
assert all(abs(r["R"] + r["T"] - 1.0) < 1e-6 for r in sweep), "energy conserved across the whole sweep"
print(f"  ±1 orders first carry power at Λ = {first_on:.2f}  (cutoff is Λ = λ = {WL})")
# --8<-- [end:code]

# --- figure: the orders switching on at the Rayleigh anomaly -------------------------------------
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10})
TEAL, AMBER, GREY = "#0D9488", "#B45309", "#94A3B8"
fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.axvspan(periods[0], WL, color=GREY, alpha=0.13, lw=0)
ax.plot(periods, [r["orders"][0] for r in sweep], "o-", color=TEAL, lw=2.0, ms=4, label=r"$T(0)$")
ax.plot(
    periods, [r["orders"][+1] + r["orders"][-1] for r in sweep], "s-", color=AMBER, lw=2.0, ms=4, label=r"$T(+1)+T(-1)$"
)
ax.plot(periods, [r["R"] + r["T"] for r in sweep], ":", color="#1A202C", lw=1.4, label=r"$R+T$")
ax.axvline(WL, color="#1A202C", lw=1.0, ls="--")
ax.annotate(r"$\Lambda=\lambda$", xy=(WL, 1.03), ha="center", fontsize=9)
ax.text(0.72, 0.5, "sub-wavelength\n(only $m=0$)", ha="center", fontsize=8.5, color="#475569")
ax.set_xlabel(r"grating period $\Lambda\;/\;\lambda$")
ax.set_ylabel("diffraction efficiency")
ax.set_title("Binary grating — the ±1 orders switch on at the Rayleigh cutoff")
ax.set_ylim(-0.03, 1.12)
ax.margins(x=0)
ax.legend(loc="center right", frameon=False, fontsize=9)
assets = Path(__file__).resolve().parents[2] / "assets"
assets.mkdir(exist_ok=True)
fig.savefig(assets / "rcwa_binary_grating.png")
print(f"  saved figure -> {assets / 'rcwa_binary_grating.png'}")
