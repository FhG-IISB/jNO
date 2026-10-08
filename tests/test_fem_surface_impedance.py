"""General boundary integrands on the non-nodal path, and ``u.across`` -- the two-face coupling a thin
conductor needs once it is removed from the mesh.

A foil of thickness ``t`` and conductivity ``σ`` carrying a 1-D field across its thickness is the exact
layered-conductor two-port of Dowell (P. L. Dowell, "Effects of eddy currents in transformer windings",
Proc. IEE 113(8), 1387-1394, 1966): ``E± = Z11 H± ...`` with ``Zc = γ/σ``, ``Z11 = Zc coth γt``,
``Z12 = Zc csch γt``. Inverted, the admittance ``Y = (1/Zc)[[coth, -csch], [csch, -coth]]`` acts on
``E = -jωA``; per face it is ``c11 A·v + c12 A_across·v`` with ``c11 = jωσ coth(γt)/γ`` and
``c12 = -jωσ csch(γt)/γ`` -- both finite as ``ω -> 0`` (``±1/(μt)`` plus ``jωσt/3`` / ``jωσt/6``). Outside
the foil the field is linear in the thickness coordinate, which P1/P2 and N1E_k reproduce exactly, so
the discrete solution must equal the closed form to round-off: in the net-current mode, the
external-field (proximity) mode, and a mix of the two.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for meshing")

import jax  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

inner, vec = jno.np.inner, jno.np.vector
MU0, SIG, T_FOIL = 4e-7 * np.pi, 5.8e7, 35e-6


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _dense(A):
    return np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)


def _two_port(f, t=T_FOIL):
    w = 2 * np.pi * f
    g = np.sqrt(1j * w * MU0 * SIG)
    coth, csch = 1 / np.tanh(g * t), 1 / np.sinh(g * t)
    return w, g / SIG, coth, csch, 1j * w * SIG * coth / g, -1j * w * SIG * csch / g


@pytest.mark.parametrize("k", [1, 2])
def test_general_boundary_path_equals_the_legacy_impedance_mass(k):
    """A boundary integrand the legacy patterns do not recognise (here the impedance mass written as a
    component sum) goes through the shared evaluator per facet -- and assembles the SAME matrix."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.6).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E", order=k)
    x, y, z = d.variable("interior", split=True)[:3]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    vol = inner(ui.vector.curl(x, y, z), vi.vector.curl(x, y, z)) + inner(ui, vi)
    nvec = d.variable("boundary", normals=True)
    tu, tv = u.vector.cross(nvec), v.vector.cross(nvec)
    A0 = _dense(jno.fem([vol]).A)
    S_legacy = _dense(jno.fem([vol, 2.0 * inner(tu, tv)]).A) - A0
    S_general = _dense(jno.fem([vol, 2.0 * (tu[0] * tv[0]) + 2.0 * (tu[1] * tv[1]) + 2.0 * (tu[2] * tv[2])]).A) - A0
    assert np.abs(S_legacy).max() > 0.1
    np.testing.assert_allclose(S_general, S_legacy, atol=1e-13 * np.abs(S_legacy).max())


def test_two_dimensional_h_curl_boundary_term_assembles():
    """2-D H(curl) boundary terms (the scalar tangential trace u×n = u_x n_y - u_y n_x) were refused
    by the pattern path; through the general path ``∮ (u×n)(v×n)`` of a constant field u = (1, 2) on
    the unit square equals its boundary integral, |u·t|² summed over the four sides = 2·1 + 2·4."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.4)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="N1E", order=2)
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    xb, yb, _, nx, ny = d.variable("boundary", normals=True, split=True)
    ub, vb = u.bind(x=xb, y=yb), v.bind(x=xb, y=yb)
    M = _dense(jno.fem([inner(ui, vi)]).A)
    S = _dense(jno.fem([inner(ui, vi), (ub[0] * ny - ub[1] * nx) * (vb[0] * ny - vb[1] * nx)]).A) - M
    a = np.linalg.solve(M, np.asarray(jno.fem([inner(ui, vi) - (1.0 * vi[0] + 2.0 * vi[1])]).b).reshape(-1))
    np.testing.assert_allclose(a @ S @ a, 2 * 1.0 + 2 * 4.0, rtol=1e-11)


def _foil_2d(order, f, Ht, Hb, Hd=0.5e-3, L=1e-3):
    w, Zc, coth, csch, c11, c12 = _two_port(f)
    t = T_FOIL
    d = jno.domain.csg.from_regions({"top": box(0, t / 2, L, Hd), "bot": box(0, -Hd, L, -t / 2)}, mesh_size=2e-4, time=None)
    e = 1e-9
    d.tag("ztop", lambda x, y: np.abs(y - Hd) < e)
    d.tag("zbot", lambda x, y: np.abs(y + Hd) < e)
    d.tag("ftop", lambda x, y: np.abs(y - t / 2) < e)
    d.tag("fbot", lambda x, y: np.abs(y + t / 2) < e)
    A, v = d.fem_symbols(names=("A", "v"), order=order)
    xi, yi, _ = d.variable("interior", split=True)
    Ai, vi = A.bind(x=xi, y=yi), v.bind(x=xi, y=yi)

    def on(tag):
        c = d.variable(tag, split=True)
        return A.bind(x=c[0], y=c[1]), v.bind(x=c[0], y=c[1])

    (At, vt), (Ab, vb), (_a, vzt), (_b, vzb) = on("ftop"), on("fbot"), on("ztop"), on("zbot")
    fem = jno.fem(
        [
            (1 / MU0) * (Ai.x * vi.x + Ai.y * vi.y),
            c11 * At * vt + c12 * A.across("fbot", domain=d) * vt,
            c11 * Ab * vb + c12 * A.across("ftop", domain=d) * vb,
            Ht * vzt,
            (-Hb) * vzb,
        ]
    )
    sol = np.asarray(fem.solve()).reshape(-1)
    # closed form: H_x is uniform in each air layer; the foil's admittance gives E± = -jωA± on its faces
    Y = np.array([[coth, -csch], [csch, -coth]]) / Zc
    Ep, Em = np.linalg.solve(Y, [Ht, Hb])
    Ap, Am = Ep / (-1j * w), Em / (-1j * w)
    zz = np.asarray(d.mesh.points)[:, 1]
    ex = np.where(zz > 0, Ap - MU0 * Ht * (zz - t / 2), Am - MU0 * Hb * (zz + t / 2))
    return np.abs(sol[: zz.size] - ex).max() / np.abs(ex).max()


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("Ht,Hb", [(1.0, -1.0), (1.0, 1.0), (0.3, 2.0)])
def test_two_dimensional_foil_two_port_is_exact(order, Ht, Hb):
    """2-D A_z (Lagrange) with the foil removed and its two faces coupled by ``across``: exact at 10 MHz
    (t/δ = 1.7) for the net-current, external-field and mixed drives (measured 1e-15)."""
    assert _foil_2d(order, 1e7, Ht, Hb) < 1e-12


def test_foil_two_port_low_frequency_limit_does_not_blow_up():
    """At 100 Hz (δ/t = 190) c11 and c12 are each ≈ ±1/(μt) and their sum carries the conductance --
    still exact to the expected cancellation, ε·(δ/t)²: measured 2e-11."""
    assert _foil_2d(1, 100.0, 1.0, -1.0) < 1e-8
    assert _foil_2d(2, 100.0, 0.3, 2.0) < 1e-8


def _foil_3d(k, f, Ht, Hb, Hd=0.4e-3, L=0.6e-3):
    """The same foil in 3-D, N1E_k: H = H_slab ŷ prescribed by ∮(n×H)·v on the outer faces; the two
    slabs are meshed INDEPENDENTLY (non-matching faces), which the linear field tolerates exactly."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    from jno.utils.solver.fem_nonnodal import nonnodal_field_at_points

    w, Zc, coth, csch, c11, c12 = _two_port(f)
    t = T_FOIL
    S = jno.Shape
    d = S.regions(top=S.box(0, 0, t / 2, L, L, Hd, size=2.5e-4), bot=S.box(0, 0, -Hd, L, L, -t / 2, size=2.5e-4)).domain()
    e = 1e-9
    preds = {
        "ztop": lambda x, y, z: np.abs(z - Hd) < e,
        "zbot": lambda x, y, z: np.abs(z + Hd) < e,
        "ftop": lambda x, y, z: np.abs(z - t / 2) < e,
        "fbot": lambda x, y, z: np.abs(z + t / 2) < e,
        "x0t": lambda x, y, z: (np.abs(x) < e) & (z > 0),
        "x1t": lambda x, y, z: (np.abs(x - L) < e) & (z > 0),
        "x0b": lambda x, y, z: (np.abs(x) < e) & (z < 0),
        "x1b": lambda x, y, z: (np.abs(x - L) < e) & (z < 0),
    }
    for nm, p in preds.items():
        d.tag(nm, p)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E", order=k)
    x, y, z = d.variable("interior", split=True)[:3]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    # a tiny gauge mass: a source-free air region leaves the gradient part of A otherwise undetermined
    terms = [(1 / MU0) * inner(ui.vector.curl(x, y, z), vi.vector.curl(x, y, z)) + 1e-6 * inner(ui, vi)]
    # n×H with H = H_slab ŷ: z faces n×ŷ = ∓x̂, x faces ±ẑ
    for nm, (gx, gz) in {
        "ztop": (-Ht, 0),
        "zbot": (Hb, 0),
        "x0t": (0, -Ht),
        "x1t": (0, Ht),
        "x0b": (0, -Hb),
        "x1b": (0, Hb),
    }.items():
        c = d.variable(nm, split=True)[:3]
        terms.append(inner(vec(gx + 0 * c[0], 0 * c[0], gz + 0 * c[0]), v.bind(x=c[0], y=c[1], z=c[2])))
    for nm, other in (("ftop", "fbot"), ("fbot", "ftop")):
        nvec = d.variable(nm, normals=True)
        c = d.variable(nm, split=True)[:3]
        ub, vb = u.bind(x=c[0], y=c[1], z=c[2]), v.bind(x=c[0], y=c[1], z=c[2])
        tu, tv = ub.vector.cross(nvec), vb.vector.cross(nvec)
        terms.append(c11 * inner(tu, tv) + c12 * inner(jno.np.cross(u.across(other, domain=d), nvec), tv))
    A, b = jno.fem(terms).operator
    ii = np.asarray(A.indices)
    As = sp.csr_matrix((np.asarray(A.data), (ii[:, 0], ii[:, 1])), shape=A.shape)
    sol = spla.spsolve(As.tocsc(), np.asarray(b).reshape(-1))
    if not np.iscomplexobj(As.data):  # the fused real-equivalent block
        n = A.shape[0] // 2
        sol = sol[:n] + 1j * sol[n:]
    Y = np.array([[coth, -csch], [csch, -coth]]) / Zc
    Ep, Em = np.linalg.solve(-Y, [Ht, Hb])  # the 3-D orientation flips H relative to the 2-D one
    Ap, Am = Ep / (-1j * w), Em / (-1j * w)
    rng = np.random.default_rng(0)
    topo = d._fem_nonnodal_topology
    P, C, dm = np.asarray(d.mesh.points), topo["cells"], topo["dofmap"]
    # gauge-invariant data: the tangential A ON the foil faces, and B in the air
    Xs = np.column_stack(
        [
            L * (0.05 + 0.9 * rng.random(20)),
            L * (0.05 + 0.9 * rng.random(20)),
            np.r_[np.full(10, t / 2), np.full(10, -t / 2)],
        ]
    )
    val = np.asarray(nonnodal_field_at_points(P, C, dm, sol, Xs))
    ex = np.where(Xs[:, 2] > 0, Ap, Am)
    Xv = np.column_stack(
        [
            rng.random(20) * L,
            rng.random(20) * L,
            np.r_[t / 2 + rng.random(10) * (Hd - t / 2), -t / 2 - rng.random(10) * (Hd - t / 2)],
        ]
    )
    B = np.asarray(nonnodal_field_at_points(P, C, dm, sol, Xv, derivative="curl"))
    Bex = MU0 * np.where(Xv[:, 2] > 0, Ht, Hb)
    eA = np.abs(val[:, :2] - np.column_stack([ex, 0 * ex])).max() / np.abs(ex).max()
    eB = np.abs(B - np.column_stack([0 * Bex, Bex, 0 * Bex])).max() / np.abs(Bex).max()
    return eA, eB


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("Ht,Hb", [(1.0, -1.0), (1.0, 1.0)])
def test_three_dimensional_foil_two_port_is_exact(k, Ht, Hb):
    """3-D N1E_k at 10 MHz: tangential A on both faces and B in the air match the closed form (measured
    4e-14 at k = 1, 3e-12 at k = 2, both modes)."""
    eA, eB = _foil_3d(k, 1e7, Ht, Hb)
    assert eA < 1e-9 and eB < 1e-8, (eA, eB)


def test_across_names_a_real_boundary_region():
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.5)
    A, v = d.fem_symbols(names=("A", "v"))
    with pytest.raises(ValueError, match="not a boundary region"):
        A.across("nowhere", domain=d)


@pytest.mark.parametrize("family,degree", [("N1E", 1), ("N1E", 2), ("N2E", 2), ("RT", 2), ("Lagrange", 3)])
def test_basis_at_points_batch_matches_per_cell(family, degree):
    """The batched across-pairing basis equals the per-cell evaluation it replaced."""
    from jno.utils.solver.fem_dofmap import basis_at_points, basis_at_points_batch, build_dofmap

    rng = np.random.default_rng(0)
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    P = np.asarray(d.mesh.points)[:, :3]
    C = np.asarray(d.mesh.cells_dict["tetra"])
    dm = build_dofmap(C, family, degree)
    cs = rng.integers(0, len(C), 40)
    lam = rng.dirichlet(np.ones(4), 40)
    X = np.einsum("nk,nkd->nd", lam, P[C[cs]])
    ref = np.stack([basis_at_points(dm, P, C, int(c), X[i : i + 1])[0] for i, c in enumerate(cs)])
    np.testing.assert_allclose(basis_at_points_batch(dm, P, C, cs, X), ref, rtol=0, atol=1e-12)


def _foil_2d_hybrid(order, f, Ht, Hb, Hd=0.5e-3, L=1e-3):
    """The foil MESHED (as μ0, σ = 0) between the air layers, its two faces now interior facets named by
    ``d.tag(..., region=<the foil>)``; the meshed foil carries the static block 1/(μt)[[1,-1],[-1,1]], so
    only the conduction part of the two-port goes on the faces."""
    w, Zc, coth, csch, c11, c12 = _two_port(f)
    t = T_FOIL
    c11p, c12p = c11 - 1 / (MU0 * t), c12 + 1 / (MU0 * t)
    d = jno.domain.csg.from_regions(
        {"top": box(0, t / 2, L, Hd), "cu": box(0, -t / 2, L, t / 2), "bot": box(0, -Hd, L, -t / 2)}, mesh_size=2e-4, time=None
    )
    e = 1e-9
    d.tag("ztop", lambda x, y: np.abs(y - Hd) < e)
    d.tag("zbot", lambda x, y: np.abs(y + Hd) < e)
    d.tag("ftop", lambda x, y: np.abs(y - t / 2) < e, region="interior_cu")
    d.tag("fbot", lambda x, y: np.abs(y + t / 2) < e, region="interior_cu")
    A, v = d.fem_symbols(names=("A", "v"), order=order)
    xi, yi, _ = d.variable("interior", split=True)
    Ai, vi = A.bind(x=xi, y=yi), v.bind(x=xi, y=yi)

    def on(tag):
        c = d.variable(tag, split=True)
        return A.bind(x=c[0], y=c[1]), v.bind(x=c[0], y=c[1])

    (At, vt), (Ab, vb), (_a, vzt), (_b, vzb) = on("ftop"), on("fbot"), on("ztop"), on("zbot")
    fem = jno.fem(
        [
            (1 / MU0) * (Ai.x * vi.x + Ai.y * vi.y),
            c11p * At * vt + c12p * A.across("fbot", domain=d) * vt,
            c11p * Ab * vb + c12p * A.across("ftop", domain=d) * vb,
            Ht * vzt,
            (-Hb) * vzb,
        ]
    )
    sol = np.asarray(fem.solve()).reshape(-1)
    Y = np.array([[coth, -csch], [csch, -coth]]) / Zc
    Ep, Em = np.linalg.solve(Y, [Ht, Hb])
    Ap, Am = Ep / (-1j * w), Em / (-1j * w)
    zz = np.asarray(d.mesh.points)[:, 1]
    ex = np.where(zz > 0, Ap - MU0 * Ht * (zz - t / 2), Am - MU0 * Hb * (zz + t / 2))
    air = np.abs(zz) >= t / 2 - 1e-12  # inside the foil the meshed field is the linear interpolant
    return np.abs(sol[: zz.size] - ex)[air].max() / np.abs(ex).max()


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("f,tol", [(1e7, 1e-12), (100.0, 1e-8)])
def test_foil_on_interior_facets_of_a_meshed_body_is_exact(order, f, tol):
    """Surface terms on a body's faces INSIDE a conforming mesh (`region=`): exact at 10 MHz (measured
    1e-15) and at 100 Hz (2e-11, the c11/c12 cancellation), net-current and mixed drives."""
    for Ht, Hb in ((1.0, -1.0), (0.3, 2.0)):
        assert _foil_2d_hybrid(order, f, Ht, Hb) < tol
