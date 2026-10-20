"""Band structure and modal inverse design — the eigensolver compositions a photonic-crystal /
metasurface workflow needs, written test-first.

``FEM.eigs``'s own docstring advertises "photonic band structure" as a use case. It does not work
today, for three independent reasons, and each is pinned here by a test with an **analytic** oracle
so the fix has something to turn green:

1. **A Bloch tie crashes the eigensolve.** ``_fem.py`` assembles the mass form as a *separate*
   ``jno.fem(list(mass))``, which carries no Bloch phase, so ``M`` stays ``n × n`` while a phased
   ``K`` is the real-equivalent ``2n × 2n`` block: ``dot_general ... got (74,) and (148,)``. A plain
   (unphased) periodic tie works, which is what localises it.
2. **A tie cannot compose with a Dirichlet pin** (explicit ``NotImplementedError``) — the case every
   *slab* needs: periodic in-plane, PEC or PML above and below.
3. **A non-symmetric coupled pencil is silently wrong above ~50 DOFs.** This one is new and is the
   worst of the three, because it is not loud. The auxiliary-field ("Aux-E") route to dispersive
   materials — carry the polarisation ``P`` as a second field so a Lorentz pole becomes an ordinary
   *linear* pencil in ``λ = ω²`` — is correct at ``h = 0.1, 0.05, 0.04`` and then, at ``h = 0.03``
   and below, silently returns the ``ε∞``-only spectrum instead. The routing probe is not at fault:
   it reports ``nonsymmetric`` at every mesh size tested (relative asymmetry 1.3e-01 down to
   3.4e-04, never near the 1e-8 tolerance).

Plus the enabler the modal objectives need:

4. **Eigenvector derivatives are NaN by design** (``eigen.py:_no_eigenvector_grad``) — deliberately,
   because a silent zero would be worse. Any objective on the *field pattern* (radiative Q, mode
   volume, overlap, far-field polarisation) needs them. The degenerate case is included because a
   symmetry-protected BIC sits exactly there, and only *subspace-invariant* functionals have a
   derivative at a repeated eigenvalue — the test is written that way.

Analytic oracles used
---------------------
* Doubly Bloch-tied unit square, ``-Δu = λu``: ``λ = (kx + 2πm)² + (ky + 2πn)²``.
* Bloch in x, Dirichlet in y: ``λ = (kx + 2πm)² + (nπ)²``.
* Lorentz slab, ``ε(ω) = ε∞ + Δε ω₀²/(ω₀² − ω²)``, Dirichlet ends: ``k² = λ ε(λ)`` with ``k = nπ``,
  i.e. ``ε∞ λ² − λ(ε∞ω₀² + Δεω₀² + k²) + k²ω₀² = 0``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

PI = np.pi
EPS_INF, DELTA_EPS, W0 = 2.0, 3.0, 5.0


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _square(size):
    """Unit square with the four edges tagged, corners included (a predicate, per the docs)."""
    e = 1e-9
    d = jno.domain(jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size))
    d.tag("bz_l", lambda x, y: x < e)
    d.tag("bz_r", lambda x, y: x > 1 - e)
    d.tag("bz_b", lambda x, y: y < e)
    d.tag("bz_t", lambda x, y: y > 1 - e)
    return d


def _laplace_bands(d, kx, ky, k=4, names=("bzu", "bzv")):
    """``-Δu = λu`` on the unit cell, Bloch-tied in BOTH directions (so no pin is needed)."""
    u, phi = d.fem_symbols(names=names)
    xi, yi, *_ = d.variable("interior", split=True)
    xl, yl, *_ = d.variable("bz_l", split=True)
    xr, yr, *_ = d.variable("bz_r", split=True)
    xb, yb, *_ = d.variable("bz_b", split=True)
    xt, yt, *_ = d.variable("bz_t", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    px = 1.0 if kx == 0.0 else np.exp(-1j * kx)
    py = 1.0 if ky == 0.0 else np.exp(-1j * ky)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, u(xl, yl) - px * u(xr, yr), u(xb, yb) - py * u(xt, yt)])
    lam, _X = fem.eigs(mass=[ui * vi], k=k)
    return np.sort(np.real(np.asarray(lam)))


def _band_exact(kx, ky, k=4, rng=(-1, 0, 1)):
    return np.sort(np.array([(kx + 2 * PI * m) ** 2 + (ky + 2 * PI * n) ** 2 for m in rng for n in rng]))[:k]


def _lorentz_exact(n):
    """Both polariton branches of ``k² = λ ε(λ)`` at ``k = nπ``."""
    k2 = (n * PI) ** 2
    a = EPS_INF
    b = -(EPS_INF * W0**2 + DELTA_EPS * W0**2 + k2)
    c = k2 * W0**2
    disc = b * b - 4 * a * c
    return sorted([(-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)])


def _lorentz_slab(mesh_size, k=4):
    """Auxiliary-field ("Aux-E") Lorentz medium: the polarisation ``P`` is a SECOND FIELD, written by
    the user in the trace. With no loss every ``ω`` appears as ``ω²``, so the pole becomes an
    ordinary LINEAR generalised eigenproblem in ``λ = ω²`` — no polynomial eigensolver needed.

        eq_u :  ∫u'v'                       = λ ∫(ε∞ u + P) v
        eq_P :  ∫(ω₀²P − Δε ω₀² u) q        = λ ∫P q
    """
    d = jno.domain(constructor=jno.domain.line(mesh_size=mesh_size))
    u, v = d.fem_symbols(names=("lzu", "lzv"))
    p, q = d.fem_symbols(names=("lzp", "lzq"))
    xi, *_ = d.variable("interior", split=True)
    l0 = d.variable("left", split=True)[0]
    r0 = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), v.bind(x=xi)
    pi_, qi = p.bind(x=xi), q.bind(x=xi)
    fem = jno.fem([ui.x * vi.x, W0**2 * pi_ * qi - DELTA_EPS * W0**2 * ui * qi, u(l0) - 0.0, u(r0) - 0.0])
    lam, _X = fem.eigs(mass=[EPS_INF * ui * vi + pi_ * vi, pi_ * qi], k=k)
    return np.sort(np.real(np.asarray(lam)))


# --------------------------------------------------------------------------------------
# 1. Band structure — the Γ point works, any Bloch phase does not
# --------------------------------------------------------------------------------------
def test_periodic_eigs_matches_analytic_at_gamma():
    """The baseline that already works, and the control for everything below: at ``k = 0`` a Bloch
    tie is an ordinary periodic tie, and the spectrum converges to ``0, 4π², 4π², 4π²``."""
    errs = []
    for size in (0.16, 0.10, 0.07):
        lam = _laplace_bands(_square(size), 0.0, 0.0)
        assert abs(lam[0]) < 1e-8, f"the constant mode must sit at λ = 0, got {lam[0]:.3e}"
        errs.append(abs(lam[1] - 4 * PI**2) / (4 * PI**2))
    assert errs[0] > errs[1] > errs[2], f"P1 should converge from above: {errs}"
    assert errs[-1] < 0.02, f"first nonzero band {errs[-1] * 100:.2f}% off 4π² at h = 0.07"


def test_bloch_eigs_matches_analytic_band():
    """The headline: a band at a general Brillouin-zone point.

    FIXED (feat/bloch-eigs): the mass form is now lifted into the same real-equivalent 2n layout the
    Bloch-tied K lives in, the reduction is taken in that space, and the doubled spectrum the block
    carries is thinned back to the complex one.
    """
    kx, ky = 1.3, 0.7
    lam = _laplace_bands(_square(0.07), kx, ky, k=3)
    exact = _band_exact(kx, ky, k=3)
    rel = np.abs(lam - exact) / exact
    assert rel.max() < 0.03, f"bands {np.round(lam, 4)} vs analytic {np.round(exact, 4)}"


def test_bloch_band_is_continuous_across_the_brillouin_zone():
    """A band DIAGRAM, not one point: ω₁(kx) must track the analytic curve along Γ→X and stay
    continuous — the sweep is what an inverse-design loop actually runs."""
    d = _square(0.09)
    ks = np.linspace(0.0, PI, 5)
    got = np.array([_laplace_bands(d, float(kx), 0.0, k=1)[0] for kx in ks])
    exact = np.array([_band_exact(float(kx), 0.0, k=1)[0] for kx in ks])
    assert np.all(np.diff(got) > -1e-9), f"ω₁ must rise monotonically along Γ→X, got {got}"
    assert np.abs(got - exact).max() / max(exact.max(), 1e-30) < 0.03


def _pc_bands(kx, ky, eps_rod, r=0.2, size=0.055, nb=3):
    """TM (scalar E_z) bands of a square lattice of dielectric rods in air, lattice constant a = 1:
    ``∇²E + (ω/c)² ε(x) E = 0`` -> ``K x = λ M(ε) x`` with ``λ = (ω/c)²``. The rod is a smoothed
    indicator so ε is written in the trace and no meshed region is needed."""
    e = 1e-9
    d = jno.domain(jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size))
    d.tag("pc_l", lambda x, y: x < e)
    d.tag("pc_r", lambda x, y: x > 1 - e)
    d.tag("pc_b", lambda x, y: y < e)
    d.tag("pc_t", lambda x, y: y > 1 - e)
    u, phi = d.fem_symbols(names=("pcu", "pcv"))
    xi, yi, *_ = d.variable("interior", split=True)
    xl, yl, *_ = d.variable("pc_l", split=True)
    xr, yr, *_ = d.variable("pc_r", split=True)
    xb, yb, *_ = d.variable("pc_b", split=True)
    xt, yt, *_ = d.variable("pc_t", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    rr = jno.np.sqrt((xi - 0.5) ** 2 + (yi - 0.5) ** 2 + 1e-12)
    eps = 1.0 + (eps_rod - 1.0) * 0.5 * (1.0 - jno.np.tanh(60.0 * (rr - r)))
    px = 1.0 if kx == 0.0 else np.exp(-1j * kx)
    py = 1.0 if ky == 0.0 else np.exp(-1j * ky)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, u(xl, yl) - px * u(xr, yr), u(xb, yb) - py * u(xt, yt)])
    lam, _X = fem.eigs(mass=[eps * ui * vi], k=nb)
    return np.sort(np.real(np.asarray(lam)))


def _bz_path(n=4):
    """Γ → X → M → Γ around the irreducible Brillouin zone of the square lattice."""
    pts = [(t * PI, 0.0) for t in np.linspace(0, 1, n, endpoint=False)]
    pts += [(PI, t * PI) for t in np.linspace(0, 1, n, endpoint=False)]
    pts += [((1 - t) * PI, (1 - t) * PI) for t in np.linspace(0, 1, n + 1)]
    return pts


@pytest.mark.slow
def test_photonic_crystal_tm_bandgap_opens():
    """The capability, end to end: a square lattice of dielectric rods opens a TM band gap and an
    empty lattice does not.

    Oracle in two parts. (1) With ``ε = 1`` everywhere the cell is an empty lattice, so bands 1 and
    2 must OVERLAP across Γ→X→M→Γ — a positive gap there would mean the Bloch phase was not acting.
    (2) With ε = 12 rods at r = 0.2a a gap must open, and its lower edge must land near the
    textbook first-TM-gap edge (Joannopoulos et al., *Photonic Crystals*, ch. 5: ωa/2πc ≈ 0.24–0.29
    for ε ≈ 11.4, r = 0.2a). Only the LOWER edge is pinned: the upper edge depends on where band 2's
    minimum sits, and this test samples a coarse k-path on a coarse mesh with a tanh-smoothed rod.
    """
    empty = np.array([_pc_bands(kx, ky, 1.0) for kx, ky in _bz_path()])
    assert empty[:, 1].min() < empty[:, 0].max(), (
        f"an EMPTY lattice must have overlapping bands 1 and 2, got a gap of {empty[:, 1].min() - empty[:, 0].max():.4f}"
    )
    rods = np.array([_pc_bands(kx, ky, 12.0) for kx, ky in _bz_path()])
    lo, hi = rods[:, 0].max(), rods[:, 1].min()
    assert hi > lo, f"no TM gap opened: band1 max {lo:.4f}, band2 min {hi:.4f}"
    f_lo = np.sqrt(lo) / (2 * PI)  # reduced frequency omega a / 2 pi c
    assert 0.22 < f_lo < 0.34, f"lower gap edge at ωa/2πc = {f_lo:.4f}, expected ≈ 0.24–0.29"


def test_bloch_eigs_accepts_its_own_modes_as_a_warm_start():
    """A band sweep should be able to hand each k-point the previous one's modes as ``X0`` — they
    are an excellent guess, since adjacent k-points have nearly identical fields.

    That is a round-trip through the real-equivalent layout: ``eigs`` returns COMPLEX modes on the
    n-space, and the solve runs on the 2n ``[Re; Im]`` block, so the guess has to be expanded back
    into it. Measured on a 2788-DOF cell over 12 k-points, warm-starting took a LOBPCG sweep from
    5.88 s to 4.85 s (and the cold dense sweep was 9.27 s).
    """
    d = _square(0.09)
    X0 = None
    for kx in (1.20, 1.25, 1.30):
        u, phi = d.fem_symbols(names=("wsu", "wsv"))
        xi, yi, *_ = d.variable("interior", split=True)
        xl, yl, *_ = d.variable("bz_l", split=True)
        xr, yr, *_ = d.variable("bz_r", split=True)
        xb, yb, *_ = d.variable("bz_b", split=True)
        xt, yt, *_ = d.variable("bz_t", split=True)
        ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
        fem = jno.fem(
            [ui.x * vi.x + ui.y * vi.y, u(xl, yl) - np.exp(-1j * kx) * u(xr, yr), u(xb, yb) - np.exp(-1j * 0.7) * u(xt, yt)]
        )
        # X0= seeds the iterative block, so it needs precond= — jNO refuses the combination
        # without it rather than accepting the guess and silently ignoring it.
        pc = jno.precond.jacobi()
        cold, Xc = fem.eigs(mass=[ui * vi], k=3, precond=pc)
        warm, Xw = fem.eigs(mass=[ui * vi], k=3, precond=pc, X0=X0) if X0 is not None else (cold, Xc)
        c, w = np.sort(np.real(np.asarray(cold))), np.sort(np.real(np.asarray(warm)))
        assert np.abs(c - w).max() / max(np.abs(c).max(), 1e-30) < 1e-6, (
            f"a warm start changed the answer at kx={kx}: {np.round(c, 6)} vs {np.round(w, 6)}"
        )
        X0 = Xw
    assert X0 is not None and np.iscomplexobj(np.asarray(X0)), "modes should come back complex"


# --------------------------------------------------------------------------------------
# 2. Slab geometry — a tie plus a pin
# --------------------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="_fem.py:_eigs_constraint_maps returns EITHER the periodic prolongation OR the Dirichlet "
    "gather and raises when both are present. The periodic record already carries `kept_nodes`, "
    "so the composition is: build P_per, then gather the main DOFs that are not pinned. A pinned "
    "SECONDARY implies its main partner is pinned (|c| = 1, so u_sec = c u_main = 0 => u_main = 0), "
    "which is consistent.",
)
def test_bloch_tie_composes_with_a_dirichlet_pin():
    """The slab case: Bloch-periodic in x, PEC (homogeneous Dirichlet) at y = 0 and y = 1, so
    ``λ = (kx + 2πm)² + (nπ)²``. Every metasurface unit cell has this shape."""
    kx = 1.3
    d = _square(0.08)
    u, phi = d.fem_symbols(names=("slu", "slv"))
    xi, yi, *_ = d.variable("interior", split=True)
    xl, yl, *_ = d.variable("bz_l", split=True)
    xr, yr, *_ = d.variable("bz_r", split=True)
    xb, yb, *_ = d.variable("bz_b", split=True)
    xt, yt, *_ = d.variable("bz_t", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y,
            u(xl, yl) - np.exp(-1j * kx) * u(xr, yr),
            u(xb, yb) - 0.0,
            u(xt, yt) - 0.0,
        ]
    )
    lam, _X = fem.eigs(mass=[ui * vi], k=3)
    got = np.sort(np.real(np.asarray(lam)))
    exact = np.sort(np.array([(kx + 2 * PI * m) ** 2 + (n * PI) ** 2 for m in (-1, 0, 1) for n in (1, 2)]))[:3]
    assert (np.abs(got - exact) / exact).max() < 0.05, f"{np.round(got, 4)} vs {np.round(exact, 4)}"


# --------------------------------------------------------------------------------------
# 3. Dispersive materials via an auxiliary field — right on coarse meshes, silently wrong on fine
# --------------------------------------------------------------------------------------
def test_lossless_lorentz_auxiliary_field_is_a_linear_pencil():
    """The route itself is sound, and it needs NO new solver feature: a lossless Lorentz pole
    carried by an auxiliary polarisation field is an ordinary linear generalised eigenproblem, and
    on a coarse mesh jNO gets both polariton branches right."""
    lam = _lorentz_slab(0.05, k=4)
    lower = [_lorentz_exact(n)[0] for n in (1, 2, 3)]
    assert (np.abs(lam[:3] - lower) / lower).max() < 0.02, f"{np.round(lam[:3], 4)} vs {np.round(lower, 4)}"
    # and the coupling is load-bearing: without it the answer would be the eps-infinity-only k²/ε∞
    assert abs(lam[0] - PI**2 / EPS_INF) / (PI**2 / EPS_INF) > 0.5, "dispersion had no effect"


def test_coupled_nonsymmetric_eigs_never_returns_a_non_eigenvalue():
    """Refinement must converge, or say it cannot — it must never quietly change physics.

    THE BUG (found here, now guarded): the identical Lorentz slab was correct at h = 0.05 and 0.04
    and, from h = 0.03, silently returned ``4.9385, 19.79, 25, 25`` — the BLOCK-DIAGONAL spectrum,
    i.e. the dispersive coupling dropped, with the polarisation modes parked at exactly ω₀² = 25.
    Not one of those is an eigenvalue of the pencil. The trigger is the dense/Arnoldi cutoff in
    ``nonsymmetric_geneigh`` (``n <= max(64, 4k+16)``): the sizes are 42, 52, 68, 102, so the flip
    happens exactly where the exact dense path stops being used.

    The cause is NOT jNO's assembly — ``scipy.linalg.eig`` on the very same matrices converges
    correctly at every size (1.88568, 1.88435, 1.88335, 1.88258 against the exact 1.88199). It is
    ARPACK: this ``M`` is non-symmetric (a one-sided coupling block, as every auxiliary-field
    dispersive, poroelastic, thermo-elastic or piezoelectric pencil has), which breaks scipy's
    documented precondition for a generalized ``sigma=None`` solve. scipy returns the same wrong
    values when called directly.

    So the fix is a residual gate rather than a convergence fix: relative residuals separate the
    two cases with no ambiguity at all — **1e-14 converged against 0.33–0.41 wrong**. Whichever
    path runs, the answer is now either right or refused.
    """
    exact = _lorentz_exact(1)[0]
    outcomes = []
    for size in (0.05, 0.04, 0.03, 0.02):
        try:
            lam1 = _lorentz_slab(size, k=4)[0]
            err = abs(lam1 - exact) / exact
            assert err < 0.05, (
                f"h={size}: returned {lam1:.6f} against the exact {exact:.6f} ({err * 100:.1f}% off) "
                f"without raising — a wrong answer got through the residual gate"
            )
            outcomes.append(f"h={size}: {lam1:.5f} ok")
        except RuntimeError as exc:
            msg = " ".join(str(exc).split())
            assert "do not satisfy the pencil" in msg, f"h={size}: unexpected RuntimeError: {msg[:200]}"
            outcomes.append(f"h={size}: refused")
        except Exception as exc:  # noqa: BLE001 - a jax-wrapped callback error carries the text too
            msg = " ".join(str(exc).split())
            assert "do not satisfy the pencil" in msg, f"h={size}: unexpected {type(exc).__name__}: {msg[:200]}"
            outcomes.append(f"h={size}: refused")
    assert any("ok" in o for o in outcomes), f"every mesh refused — the gate is too tight: {outcomes}"


# --------------------------------------------------------------------------------------
# 3b. Contour integration — every eigenvalue inside a region, differentiably
# --------------------------------------------------------------------------------------
def test_contour_finds_exactly_the_modes_inside_the_circle():
    """``fem.eigs(contour=...)`` asks a different question — "everything inside this region" — which
    is the one a band structure actually poses. It must agree with the k-smallest path on the modes
    they both see, and return COMPLEX modes on the n-space."""
    d = _square(0.09)
    u, phi = d.fem_symbols(names=("ctu", "ctv"))
    xi, yi, *_ = d.variable("interior", split=True)
    xl, yl, *_ = d.variable("bz_l", split=True)
    xr, yr, *_ = d.variable("bz_r", split=True)
    xb, yb, *_ = d.variable("bz_b", split=True)
    xt, yt, *_ = d.variable("bz_t", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    rr = jno.np.sqrt((xi - 0.5) ** 2 + (yi - 0.5) ** 2 + 1e-12)
    eps = 1.0 + 11.0 * 0.5 * (1.0 - jno.np.tanh(60.0 * (rr - 0.2)))
    fem = jno.fem(
        [ui.x * vi.x + ui.y * vi.y, u(xl, yl) - np.exp(-1j * 1.3) * u(xr, yr), u(xb, yb) - np.exp(-1j * 0.7) * u(xt, yt)]
    )
    ref = np.sort(np.real(np.asarray(fem.eigs(mass=[eps * ui * vi], k=6)[0])))
    lam, X = fem.eigs(mass=[eps * ui * vi], contour=jno.solve.contour(centre=10.0, radius=8.0, points=48, probes=12))
    got = np.sort(np.real(np.asarray(lam)))
    inside = ref[np.abs(ref - 10.0) < 8.0]
    assert len(got) == len(inside), f"contour found {len(got)} modes, {len(inside)} are inside: {got} vs {inside}"
    assert np.abs(got - inside).max() / np.abs(inside).max() < 1e-5, f"{np.round(got, 6)} vs {np.round(inside, 6)}"
    assert np.iscomplexobj(np.asarray(X)) and np.asarray(X).shape[1] == len(got)


def test_contour_refuses_a_saturated_probing_block():
    """The one knob that bites: the probing block must be wider than the number of eigenvalues
    inside, or the answer is silently wrong. Saturation is detected and refused."""
    d = _square(0.12)
    u, phi = d.fem_symbols(names=("csu", "csv"))
    xi, yi, *_ = d.variable("interior", split=True)
    xl, yl, *_ = d.variable("bz_l", split=True)
    xr, yr, *_ = d.variable("bz_r", split=True)
    xb, yb, *_ = d.variable("bz_b", split=True)
    xt, yt, *_ = d.variable("bz_t", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem(
        [ui.x * vi.x + ui.y * vi.y, u(xl, yl) - np.exp(-1j * 1.3) * u(xr, yr), u(xb, yb) - np.exp(-1j * 0.7) * u(xt, yt)]
    )
    with pytest.raises(ValueError, match="SATURATED"):
        fem.eigs(mass=[ui * vi], contour=jno.solve.contour(centre=200.0, radius=200.0, points=32, probes=3))


def test_contour_rejects_the_k_smallest_arguments():
    """A region and "the k at one end" are different questions; mixing them would silently ignore one."""
    d = _square(0.14)
    u, phi = d.fem_symbols(names=("cru", "crv"))
    xi, yi, *_ = d.variable("interior", split=True)
    xb, yb, *_ = d.variable("bz_b", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0])
    with pytest.raises(ValueError, match="names a REGION"):
        fem.eigs(mass=[ui * vi], contour=jno.solve.contour(centre=10.0, radius=5.0), sigma=1.0)


def test_contour_differentiates_a_mode_shape_objective():
    """THE point of the contour route: a MODE-SHAPE objective differentiates through it.

    ``tr(P_S W)`` — the share of the enclosed invariant subspace sitting inside the rod — is
    basis-independent, so it is well defined even at a degeneracy, and it is the shape a radiative-Q
    / mode-volume / field-overlap objective has. The Arnoldi path returns NaN for exactly this; here
    it agrees with central differences, because the whole method is linear solves plus an SVD.
    """
    from jno.utils.solver.eigen import contour_geneigh

    d = jno.domain(constructor=jno.domain.line(mesh_size=0.02))
    u, v = d.fem_symbols(names=("cdu", "cdv"))
    xi, *_ = d.variable("interior", split=True)
    l0 = d.variable("left", split=True)[0]
    r0 = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), v.bind(x=xi)
    dn = lambda A: np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)
    rod = 0.5 * (1.0 - jno.np.tanh(40.0 * (jno.np.sqrt((xi - 0.5) ** 2 + 1e-12) - 0.15)))
    A_ = dn(jno.fem([ui.x * vi.x, u(l0) - 0.0, u(r0) - 0.0]).operator[0])
    M1 = dn(jno.fem([ui * vi]).operator[0])
    M2 = dn(jno.fem([rod * ui * vi]).operator[0])
    pts = np.asarray(d.mesh.points)[:, 0]
    pin = [int(np.argmin(pts)), int(np.argmax(pts))]
    fr = np.array([i for i in range(A_.shape[0]) if i not in pin])
    A_, M1, M2 = [jnp.asarray(X[np.ix_(fr, fr)], dtype=jnp.complex128) for X in (A_, M1, M2)]
    wgt = jnp.asarray(np.exp(-(((pts[fr] - 0.5) / 0.15) ** 2)))

    def in_rod(er):
        lam, X = contour_geneigh(
            A_, M1 + (er - 1.0) * M2, centre=60.0, radius=55.0, points=128, probes=10, rank=5, tol=1e-12, maxiter=2000
        )
        Q, _ = jnp.linalg.qr(X)  # orthonormal basis of the enclosed subspace
        return jnp.real(jnp.sum(jnp.abs(Q) ** 2 * wgt[:, None])) / Q.shape[1]

    g_ad = float(jax.grad(in_rod)(6.0))
    g_fd = float((in_rod(6.0 + 1e-4) - in_rod(6.0 - 1e-4)) / 2e-4)
    assert np.isfinite(g_ad), "contour eigenvector gradient is NaN"
    assert abs(g_fd) > 1e-6, "objective is flat — the test would be vacuous"
    assert abs(g_ad - g_fd) / abs(g_fd) < 1e-5, f"AD {g_ad:.8f} vs FD {g_fd:.8f}"


# --------------------------------------------------------------------------------------
# 4. Eigenvector adjoints — the enabler for every mode-shape objective
# --------------------------------------------------------------------------------------
def _pencil(theta, size=0.12):
    """``(A(θ), B)`` for ``-∇·(κ(θ)∇u) = λ u`` on the unit square, κ scaled on the right half.

    Built through ``fem.operator.evaluate`` — the idiom ``test_fem_trainable_coordinates.py`` uses
    for its own finite-difference gradient checks — because ``FEM.eigs`` does not accept a
    runtime-parametric fem at all (``TypeError: 'FemLinearSystem' object is not subscriptable``),
    which is itself a gap this file records. ``theta`` multiplies ONE additive term directly, as the
    affine-lowering contract requires ("one trainable scalar per additive term", not nested).
    """
    d = jno.domain(jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size))
    u, phi = d.fem_symbols(names=("evu", "evv"))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    th = jno.np.parameter((1,), name="theta")
    th.dtype(jnp.float64)
    th.initialize(jax.nn.initializers.constant(0.0))
    mask = 0.5 * (1.0 + jno.np.tanh(40.0 * (xi - 0.5)))  # smooth right-half indicator
    fem = jno.fem([(ui.x * vi.x + ui.y * vi.y) + th * (mask * (ui.x * vi.x + ui.y * vi.y)), u(xb, yb) - 0.0])
    B = jno.fem([ui * vi]).operator[0]
    A, _b = fem.operator.evaluate({"theta": jnp.atleast_1d(theta)})
    return A, B


def _degenerate_pencil(theta, sym=True, size=0.12):
    """Two IDENTICAL, decoupled 1-D Laplacians: the spectrum is exactly two-fold degenerate
    (measured 9.952703 twice), which is the situation a symmetry-protected BIC sits in.

    ``sym=True`` adds the symmetric coupling ``θ(b·φ_a + a·φ_b)``, which splits the pair to
    ``μ ∓ θ`` — so the degeneracy is lifted at a known rate and there is something to differentiate.
    """
    d = jno.domain(constructor=jno.domain.line(mesh_size=size))
    a, pa = d.fem_symbols(names=("dga", "dgpa"))
    b, pb = d.fem_symbols(names=("dgb", "dgpb"))
    xi, *_ = d.variable("interior", split=True)
    l0 = d.variable("left", split=True)[0]
    r0 = d.variable("right", split=True)[0]
    ai, pai = a.bind(x=xi), pa.bind(x=xi)
    bi, pbi = b.bind(x=xi), pb.bind(x=xi)
    th = jno.np.parameter((1,), name="theta_deg")
    th.dtype(jnp.float64)
    th.initialize(jax.nn.initializers.constant(0.0))
    second = bi.x * pbi.x + (th * ai * pbi if sym else 0.0 * pbi)
    fem = jno.fem([ai.x * pai.x + th * bi * pai, second, a(l0) - 0.0, a(r0) - 0.0, b(l0) - 0.0, b(r0) - 0.0])
    B = jno.fem([ai * pai, bi * pbi]).operator[0]
    A, _b = fem.operator.evaluate({"theta_deg": jnp.atleast_1d(theta)})
    return A, B


def _mode_weight_objective(A, B, k, cols):
    """A mode-shape functional: how much of the mode (or of the degenerate SUBSPACE) sits to the
    right. Summing ``|x|²`` over ALL columns of the eigenspace makes it a trace over the spectral
    projector, i.e. invariant to the basis chosen inside a degenerate eigenspace — which is the
    only kind of eigenvector functional that HAS a derivative at a repeated eigenvalue."""
    lam, X = jno.solve.eigs(k=k, which="smallest")(A, B)
    S = X[:, cols] if jnp.ndim(X) == 2 else jnp.reshape(X, (-1, 1))
    wgt = jnp.linspace(0.0, 1.0, jnp.shape(S)[0])
    return jnp.real(jnp.sum(jnp.abs(S) ** 2 * wgt[:, None]))


def _fd(f, x0, eps):
    return (f(x0 + eps) - f(x0 - eps)) / (2 * eps)


def test_eigenvector_gradient_matches_fd_at_a_simple_eigenvalue():
    """GOOD NEWS, pinned as a regression guard: on a symmetric pencil with a well-separated
    eigenvalue, the eigenVECTOR gradient already works and agrees with finite differences. So the
    missing piece is NOT "eigenvector adjoints" in general — it is the degenerate case below and
    the non-symmetric/complex path, which is exactly where photonics lives."""
    obj = lambda t: _mode_weight_objective(*_pencil(t), k=1, cols=slice(0, 1))
    g_ad = float(jax.grad(obj)(0.0))
    g_fd = float(_fd(obj, 0.0, 1e-5))
    assert np.isfinite(g_ad), "eigenvector gradient is NaN at a simple eigenvalue"
    assert abs(g_fd) > 1e-9, "objective is flat — the test would be vacuous"
    assert abs(g_ad - g_fd) / abs(g_fd) < 1e-4, f"AD {g_ad:.8g} vs FD {g_fd:.8g}"


@pytest.mark.xfail(
    strict=True,
    reason="THE enabler for symmetry-protected BICs. At an EXACT two-fold degeneracy (two identical "
    "decoupled Laplacians: 9.952703 twice) the eigenvector gradient is NaN, from "
    "eigen.py:_no_eigenvector_grad. Only subspace-invariant functionals have a derivative at a "
    "repeated eigenvalue, so the objective here is a trace over the spectral projector, which does. "
    "Nelson's method with the repeated-eigenvalue extension is the rule to implement; jNO already "
    "detects the defective case via the eigenvalue condition number, which is the scaffolding.",
)
def test_eigenvector_gradient_at_an_exact_degeneracy():
    obj = lambda t: _mode_weight_objective(*_degenerate_pencil(t), k=2, cols=slice(0, 2))
    g_ad = float(jax.grad(obj)(0.0))
    g_fd = float(_fd(obj, 0.0, 1e-4))
    assert np.isfinite(g_ad), "subspace-invariant eigenvector gradient is NaN at a degeneracy"
    assert abs(g_ad - g_fd) <= 1e-6 + 1e-3 * abs(g_fd), f"AD {g_ad:.8g} vs FD {g_fd:.8g}"


def test_the_degeneracy_is_real_and_splits_as_expected():
    """Guards the test above: if the pair were only NEARLY degenerate, the NaN would prove nothing."""
    lam0 = np.sort(np.real(np.asarray(jno.solve.eigs(k=3, which="smallest")(*_degenerate_pencil(0.0))[0])))
    lam1 = np.sort(np.real(np.asarray(jno.solve.eigs(k=3, which="smallest")(*_degenerate_pencil(0.5))[0])))
    assert abs(lam0[0] - lam0[1]) < 1e-10, f"pair is not exactly degenerate: {lam0[:2]}"
    assert lam1[1] - lam1[0] > 0.5, f"the symmetric coupling must split the pair, got {lam1[:2]}"


@pytest.mark.xfail(
    strict=True,
    reason="FEM.eigs on a runtime-parametric fem raises TypeError: 'FemLinearSystem' object is not "
    "subscriptable. Gradient-based modal design needs the parameter to reach the eigensolve through "
    "the front door, not only through fem.operator.evaluate.",
)
def test_eigs_accepts_a_runtime_parametric_form():
    """The front-door version of the gradient tests above."""
    d = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
    u, v = d.fem_symbols(names=("pgu", "pgv"))
    xi, *_ = d.variable("interior", split=True)
    l0 = d.variable("left", split=True)[0]
    r0 = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), v.bind(x=xi)
    th = jno.np.parameter((1,), name="theta_front")
    th.dtype(jnp.float64)
    th.initialize(jax.nn.initializers.constant(1.0))
    fem = jno.fem([th * ui.x * vi.x, u(l0) - 0.0, u(r0) - 0.0])
    lam, _X = fem.eigs(mass=[ui * vi], k=2)
    got = np.sort(np.real(np.asarray(lam)))
    assert abs(got[0] - PI**2) / PI**2 < 0.05, f"λ₁ = {got[0]:.4f} vs π² = {PI**2:.4f}"
