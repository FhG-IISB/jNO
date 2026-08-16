"""Morley non-conforming quadratic triangle (6 DOF) — the cheap biharmonic element — through ``jno.fem``.

Morley's DOFs are the value at the 3 vertices and the normal derivative at the 3 edge midpoints. It is
**non-conforming** for H² (neither C⁰ nor C¹), yet it passes the patch test and converges for the biharmonic
(energy O(h), L² O(h²)). It routes through the non-nodal assembler with an ``n_verts + n_edges`` layout and the
affine ``M(cell)`` DOF-transform (vertex-value rows are affine-invariant; the edge-normal rows use the
globally-oriented physical normal — the same machinery as Argyris). With only 6 DOF and a quadratic basis it is
far cheaper than the 21-DOF quintic Argyris element, so it clears the Argyris construction memory ceiling and
scales to much finer meshes.

The crucial modelling subtlety: for the **non-conforming** element the biharmonic bilinear form must be the
full-Hessian inner product ``∫ D²u : D²v`` — NOT ``∫ Δu·Δv`` (which is singular here, since functions like
``xy`` have ``Δu = 0`` but ``D²u ≠ 0`` and would be spurious kernel modes).

References: L.S.D. Morley, *The triangular equilibrium element in the solution of plate bending problems*,
Aeronautical Quarterly 19 (1968) 149–169; P.G. Ciarlet, *The FEM for Elliptic Problems* (2002), §6.
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_morley_dual_basis_and_p2_reproduction():
    """Element de-risk (the Bell-killer analogue): the reference dual basis is δ, and the *physical* element
    reproduces an arbitrary P2 polynomial (value, gradient, Laplacian) exactly on random affine cells — this
    catches functional/sign/orientation bugs before assembly noise can hide them."""
    import jax.numpy as jnp

    from jno.utils.solver.fem_elements import (
        _arg_ref_normals,
        _mor_ref_functionals,
        _mor_reference_coeffs,
        morley_pushforward,
        morley_triangle,
    )
    from jno.utils.solver.fem_topology import BASIX_TRIANGLE_EDGES

    S = _mor_reference_coeffs()
    refn = _arg_ref_normals()
    D = np.array([_mor_ref_functionals(S[:, k], np.eye(2), refn) for k in range(6)]).T
    assert np.linalg.norm(D - np.eye(6)) < 1e-11, "reference dual basis must satisfy ℓ_i(ψ_k)=δ_ik"

    spec = morley_triangle(quad_degree=4)
    nv_val, ne_grad = spec.ref_aux
    rng = np.random.default_rng(0)
    c = rng.normal(size=6)  # p(x,y) = c0 + c1 x + c2 y + c3 x² + c4 xy + c5 y²
    pv = lambda P: (
        c[0]
        + c[1] * P[..., 0]
        + c[2] * P[..., 1]
        + c[3] * P[..., 0] ** 2
        + c[4] * P[..., 0] * P[..., 1]
        + c[5] * P[..., 1] ** 2
    )  # noqa: E731
    pg = lambda P: np.stack(
        [c[1] + 2 * c[3] * P[..., 0] + c[4] * P[..., 1], c[2] + c[4] * P[..., 0] + 2 * c[5] * P[..., 1]], -1
    )  # noqa: E731
    plap = 2 * c[3] + 2 * c[5]
    for _ in range(4):
        V = rng.normal(size=(3, 2)) * np.array([2.0, 1.5]) + np.array([1.0, -0.5])
        J = np.stack([V[1] - V[0], V[2] - V[0]], axis=1)
        en = np.array(
            [(lambda d: np.array([-d[1], d[0]]) / np.linalg.norm(d))(V[b] - V[a]) for (a, b) in BASIX_TRIANGLE_EDGES]
        )
        mids = np.array([0.5 * (V[a] + V[b]) for (a, b) in BASIX_TRIANGLE_EDGES])
        dof = np.concatenate([pv(V), np.einsum("ed,ed->e", pg(mids), en)])
        phi, grad, hess = (
            np.asarray(a)
            for a in morley_pushforward(
                jnp.asarray(spec.ref_values),
                jnp.asarray(spec.ref_grads),
                jnp.asarray(spec.ref_hess),
                jnp.asarray(J),
                jnp.linalg.det(jnp.asarray(J)),
                jnp.asarray(en),
                (jnp.asarray(nv_val), jnp.asarray(ne_grad)),
            )
        )
        xq = V[0] + spec.quad_points @ J.T
        assert np.abs(phi @ dof - pv(xq)).max() < 1e-11, "physical element must reproduce a P2 value exactly"
        assert np.abs(np.einsum("qmd,m->qd", grad, dof) - pg(xq)).max() < 1e-10, "…and its gradient"
        assert np.abs(np.einsum("qm,m->q", hess[:, :, 0, 0] + hess[:, :, 1, 1], dof) - plap).max() < 1e-10, (
            "…and its Laplacian"
        )


def _l2(uh, ue):
    return float(np.linalg.norm(uh - ue) / np.linalg.norm(ue))


def test_morley_biharmonic_convergence():
    """Clamped biharmonic ``Δ²u = f`` with the manufactured ``u* = sin(πx)sin(πy)`` on the unit square. The
    non-conforming Morley element (full-Hessian form) converges at the optimal L² rate ≈ 2, and — the practical
    payoff — assembles on a refined mesh (nv≈500) that the 21-DOF Argyris element cannot (its construction
    OOMs). The value DOFs are the first ``n_verts`` entries of the solution."""
    import jax.numpy as jnp

    sin = jno.np.sin
    dense = lambda A: jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)  # noqa: E731
    solver = lambda A, b: jnp.linalg.solve(dense(A), jnp.asarray(b).reshape(-1))  # noqa: E731

    errs, hs = [], []
    for ms in (0.16, 0.11, 0.075):
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=ms)
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        u, phi = d.fem_symbols(space="Morley")
        ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
        f = 4 * PI**4 * sin(PI * xi) * sin(PI * yi)
        g = sin(PI * xb) * sin(PI * yb)  # clamped to the exact trace (value + ∂ₙ)
        Hu, Hv = jno.np.hessian(ui, [xi, yi]), jno.np.hessian(vi, [xi, yi])
        fem = jno.fem([jno.np.inner(Hu, Hv, n_contract=2) - f * vi, u(xb, yb) - g])
        assert fem.is_linear
        sol = np.asarray(fem.solve(solver)).reshape(-1)
        pts = np.asarray(d.mesh.points)[:, :2]
        nv = pts.shape[0]
        uh = sol[np.arange(nv)]  # Morley value DOFs
        ue = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
        errs.append(_l2(uh, ue))
        hs.append(ms)

    rates = [np.log(errs[i] / errs[i + 1]) / np.log(hs[i] / hs[i + 1]) for i in range(len(errs) - 1)]
    assert errs[-1] < errs[0] and errs[-1] < 0.03, f"Morley biharmonic must converge: {errs}"
    assert np.mean(rates) > 1.5, f"L² order must be near the non-conforming optimum 2: {rates}"


def test_morley_laplacian_form_is_singular_full_hessian_is_not():
    """Guards the modelling subtlety: ``∫Δu·Δv`` gives a singular Morley system (spurious ``Δu=0`` modes),
    whereas the full-Hessian ``∫D²u:D²v`` is coercive. A regression against silently re-introducing the wrong
    form."""

    sin = jno.np.sin
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, phi = d.fem_symbols(space="Morley")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 4 * PI**4 * sin(PI * xi) * sin(PI * yi)
    g = sin(PI * xb) * sin(PI * yb)
    dense = lambda A: np.asarray(A.todense() if hasattr(A, "todense") else A)  # noqa: E731

    lap = jno.np.laplacian
    fem_lap = jno.fem([lap(ui, [xi, yi]) * lap(vi, [xi, yi]) - f * vi, u(xb, yb) - g])
    Hu, Hv = jno.np.hessian(ui, [xi, yi]), jno.np.hessian(vi, [xi, yi])
    fem_hes = jno.fem([jno.np.inner(Hu, Hv, n_contract=2) - f * vi, u(xb, yb) - g])
    assert np.linalg.cond(dense(fem_lap.A)) > 1e20, "the Laplacian form should be (near-)singular for Morley"
    assert np.linalg.cond(dense(fem_hes.A)) < 1e12, "the full-Hessian form should be well-conditioned"


@pytest.mark.slow  # non-nodal C¹ assembly at n=24 peaks at ~3 GiB in one sparse reduce — OOMs 8 GiB GPUs
def test_morley_periodic_biharmonic_convergence():
    """Periodic-in-y biharmonic ``Δ²u = f`` on the non-nodal (C¹-ish) Morley element — the regression that pins
    the **edge-derivative periodic tie sign**. The DOF-level periodic prolongation must tie both the vertex value
    DOFs and the edge-normal-derivative DOFs across the matched top/bottom boundary; the edge block carries a
    ``sign(n_secondary·n_main)`` weight, and a flipped sign gives a wrong solution.

    The manufactured ``u* = sin(πx) sin(2πy)`` is periodic in y (period 1) and has ``∂ᵧu = 2π cos(2πy) sin(πx)``
    NON-ZERO and EQUAL at y=0 and y=1 — so the tie genuinely couples the normal-derivative DOFs and a wrong sign
    (which would force ``∂ᵧu(top) = −∂ᵧu(bottom)``) fails to recover it. The x-walls are clamped to the exact
    trace. Recovery at the optimal Morley L² rate ≈ 2 confirms the tie sign. ``Δ²[sin(πx)sin(2πy)] = 25π⁴ u*``.

    Runs on CPU: the non-nodal assembly at the finer mesh peaks at ~3 GiB in one sparse reduce, which OOMs
    8 GiB GPUs — the pin validates the periodic tie sign, not GPU capacity."""
    import jax.numpy as jnp

    sin = jno.np.sin
    solver = lambda A, b: jnp.linalg.solve(  # noqa: E731
        jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A), jnp.asarray(b).reshape(-1)
    )

    errs, hs = [], []
    with jax.default_device(jax.devices("cpu")[0]):
        for n in (12, 24):
            d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=n).domain()
            xi, yi, _ = d.variable("interior", split=True)
            xl, yl, _ = d.variable("left", split=True)
            xr, yr, _ = d.variable("right", split=True)
            xt, yt, _ = d.variable("top", split=True)
            xb, yb, _ = d.variable("bottom", split=True)
            u, phi = d.fem_symbols(space="Morley")
            ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
            f = 25 * PI**4 * sin(PI * xi) * sin(2 * PI * yi)
            Hu, Hv = jno.np.hessian(ui, [xi, yi]), jno.np.hessian(vi, [xi, yi])
            fem = jno.fem(
                [
                    jno.np.inner(Hu, Hv, n_contract=2) - f * vi,
                    u(xl, yl) - sin(PI * xl) * sin(2 * PI * yl),  # clamped x-walls to the exact trace (value + ∂ₙ)
                    u(xr, yr) - sin(PI * xr) * sin(2 * PI * yr),
                    u(xt, yt) - u(xb, yb),  # periodic in y — ties value AND edge-normal-derivative DOFs
                ]
            )
            assert fem.is_linear
            sol = np.asarray(fem.solve(solver)).reshape(-1)
            pts = np.asarray(d.mesh.points)[:, :2]
            nv = pts.shape[0]
            uh = sol[np.arange(nv)]  # Morley value DOFs
            ue = np.sin(PI * pts[:, 0]) * np.sin(2 * PI * pts[:, 1])
            errs.append(_l2(uh, ue))
            hs.append(1.0 / n)

    rate = np.log(errs[0] / errs[1]) / np.log(hs[0] / hs[1])
    assert errs[-1] < 0.05 and errs[-1] < errs[0], f"periodic Morley biharmonic must converge: {errs}"
    assert rate > 1.5, f"L² order must be near the non-conforming optimum 2 (wrong tie sign breaks this): {rate}"


@pytest.mark.slow  # two non-nodal C¹ fields — assembly peaks at ~6 GiB, OOMs 8 GiB GPUs
def test_morley_periodic_multifield_decoupled_convergence():
    """Two DECOUPLED periodic biharmonic Morley fields with DIFFERENT manufactured solutions — the regression
    that pins the **per-field block-concatenation** (one periodic prolongation ``P`` per field). If a tie is
    applied to the wrong field's DOF block, or a block is mis-offset, one field fails to converge.

    ``ψ* = sin(πx)sin(2πy)`` (``Δ²ψ*=25π⁴ψ*``) and ``φ* = sin(2πx)sin(2πy)`` (``Δ²φ*=64π⁴φ*``) — both periodic
    in y, both clamped-x to their exact trace, NO coupling. Both must recover at the Morley L² rate ≈ 2. Runs on
    CPU: two C¹ fields peak at ~6 GiB in the sparse reduce, which OOMs 8 GiB GPUs."""
    import jax.numpy as jnp

    sin = jno.np.sin
    solver = lambda A, b: jnp.linalg.solve(  # noqa: E731
        jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A), jnp.asarray(b).reshape(-1)
    )

    eps, eph, hs = [], [], []
    with jax.default_device(jax.devices("cpu")[0]):
        for n in (12, 24):
            d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=n).domain()
            xi, yi, _ = d.variable("interior", split=True)
            xl, yl, _ = d.variable("left", split=True)
            xr, yr, _ = d.variable("right", split=True)
            xt, yt, _ = d.variable("top", split=True)
            xb, yb, _ = d.variable("bottom", split=True)
            psi, vps = d.fem_symbols(names=("psi", "vps"), space="Morley")
            phi, vph = d.fem_symbols(names=("phi", "vph"), space="Morley")
            pib, vpi = psi.bind(x=xi, y=yi), vps.bind(x=xi, y=yi)
            fib, vfi = phi.bind(x=xi, y=yi), vph.bind(x=xi, y=yi)
            fps = 25 * PI**4 * sin(PI * xi) * sin(2 * PI * yi)
            fph = 64 * PI**4 * sin(2 * PI * xi) * sin(2 * PI * yi)
            H = jno.np.hessian
            fem = jno.fem(
                [
                    jno.np.inner(H(pib, [xi, yi]), H(vpi, [xi, yi]), n_contract=2) - fps * vpi,  # field 0: ψ
                    jno.np.inner(H(fib, [xi, yi]), H(vfi, [xi, yi]), n_contract=2) - fph * vfi,  # field 1: φ (decoupled)
                    psi(xl, yl) - sin(PI * xl) * sin(2 * PI * yl),  # ψ clamped-x to exact trace
                    psi(xr, yr) - sin(PI * xr) * sin(2 * PI * yr),
                    phi(xl, yl) - sin(2 * PI * xl) * sin(2 * PI * yl),  # φ clamped-x to exact trace
                    phi(xr, yr) - sin(2 * PI * xr) * sin(2 * PI * yr),
                    psi(xt, yt) - psi(xb, yb),  # periodic-y on BOTH fields
                    phi(xt, yt) - phi(xb, yb),
                ]
            )
            assert fem.is_linear
            sol = np.asarray(fem.solve(solver)).reshape(-1)
            off = fem.offsets
            pts = np.asarray(d.mesh.points)[:, :2]
            nv = pts.shape[0]
            ups = np.sin(PI * pts[:, 0]) * np.sin(2 * PI * pts[:, 1])
            uph = np.sin(2 * PI * pts[:, 0]) * np.sin(2 * PI * pts[:, 1])
            eps.append(_l2(sol[off[0] : off[0] + nv], ups))  # field-0 value DOFs
            eph.append(_l2(sol[off[1] : off[1] + nv], uph))  # field-1 value DOFs
            hs.append(1.0 / n)

    rps = np.log(eps[0] / eps[1]) / np.log(hs[0] / hs[1])
    rph = np.log(eph[0] / eph[1]) / np.log(hs[0] / hs[1])
    assert eps[-1] < 0.05 and rps > 1.5, f"ψ field must recover (per-field block): errs={eps} rate={rps}"
    assert eph[-1] < 0.05 and rph > 1.5, f"φ field must recover (no block bleed): errs={eph} rate={rph}"


def test_morley_periodic_rejects_argyris_and_nonconforming():
    """Fail-loud guards (constraint: never fail silently). Periodic ties on the non-nodal path are Morley-only:
    the C¹ **Argyris** element (extra per-vertex derivative DOFs whose periodic signs are not wired) raises a
    clear ``NotImplementedError`` at assembly rather than silently mis-tying."""
    sin = jno.np.sin
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=6).domain()
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xt, yt, _ = d.variable("top", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    u, phi = d.fem_symbols(space="Argyris")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 25 * PI**4 * sin(PI * xi) * sin(2 * PI * yi)
    Hu, Hv = jno.np.hessian(ui, [xi, yi]), jno.np.hessian(vi, [xi, yi])
    with pytest.raises(NotImplementedError, match="[Mm]orley"):
        jno.fem(
            [
                jno.np.inner(Hu, Hv, n_contract=2) - f * vi,
                u(xl, yl) - sin(PI * xl) * sin(2 * PI * yl),
                u(xr, yr) - sin(PI * xr) * sin(2 * PI * yr),
                u(xt, yt) - u(xb, yb),  # periodic-y on Argyris -> must raise, not silently mis-tie
            ]
        )


def test_morley_periodic_rejects_a_bloch_phase():
    """A Bloch / quasi-periodic tie ``u(A) - c*u(B)`` must refuse on Morley rather than drop the phase.

    The C¹ prolongation carries REAL edge-normal orientation signs; a complex phase needs a complex
    prolongation (``P^H A P`` mixes Re and Im). The nodal Lagrange and N1E builders both read the phase off
    the tie spec, so this element is the odd one out and has to say so.

    This also pins the tie-spec ARITY. The spec is ``(main, secondary, comp, field_key, phase)`` and this
    builder used to unpack exactly four, so appending the Bloch phase broke *every* periodic Morley
    problem with ``ValueError: too many values to unpack (expected 4)`` — a plain-periodic tie included.
    The two convergence tests above are the regression for that; this one covers the phase itself."""
    sin = jno.np.sin
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=6).domain()
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xt, yt, _ = d.variable("top", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    u, phi = d.fem_symbols(space="Morley")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 25 * PI**4 * sin(PI * xi) * sin(2 * PI * yi)
    Hu, Hv = jno.np.hessian(ui, [xi, yi]), jno.np.hessian(vi, [xi, yi])
    with pytest.raises(NotImplementedError, match="Bloch|quasi-periodic"):
        jno.fem(
            [
                jno.np.inner(Hu, Hv, n_contract=2) - f * vi,
                u(xl, yl) - sin(PI * xl) * sin(2 * PI * yl),
                u(xr, yr) - sin(PI * xr) * sin(2 * PI * yr),
                u(xt, yt) - 0.5 * u(xb, yb),  # quasi-periodic phase c=0.5 -> refuse, never silently drop
            ]
        )
