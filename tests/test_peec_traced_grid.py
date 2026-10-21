"""The lattice's GEOMETRY as a design variable: `solve(geometry={solid: (x0, y0, z0, x1, y1, z1)})`.

A solid lattice used to be frozen at build: cell counts, occupancy AND coordinates. Only the first two
are structural. On a grid whose every conductor face is a grid line (`grid=jno.peec.faces(...)`) the
coordinates of the lines can move at solve time while the cell counts, the ownership and the node
numbering stay as built: a trace's width, length and position, a die's position and a layer's
thickness all become differentiable, with no density approximation and no rebuild.

The oracles: the traced pass at the reference reproduces the host build to the last bit (self term
to its quadrature); the gradient agrees with central differences of REBUILT networks with the same
cell counts (the same discretisation on both sides, so they must agree); the self and mutual terms'
derivatives agree with closed forms; and the uniform FFT path refuses a moved solid by name rather
than returning the answer for the geometry it was built on.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.kernel import bar_self, bar_self_jax, pair_matrix
from jno.utils.solver.peec import bar_filaments

jax.config.update("jax_enable_x64", True)

mm, CU, MU0 = 1e-3, 5.8e7, 4e-7 * np.pi
P = 1.0 * mm


# ---- the self term ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ell, w, t",
    [(1.0, 1.0, 1.0), (1.0, 0.8, 0.5), (0.5, 1.0, 0.57), (1.0, 1.0, 3.5e-3), (0.6, 1.0, 1e-3), (8.0, 1.0, 0.1)],
)
def test_the_jax_self_term_matches_the_host_integral(ell, w, t):
    """Same integral, traceable. Checked at a cube, lattice aspect ratios, and the sliver a current
    SHEET makes (a 1 mm bar 3.5 um thick), where a careless rule loses tens of percent."""
    ref = float(bar_self(ell * mm, w * mm, t * mm))
    got = float(bar_self_jax(jnp.asarray(ell * mm), jnp.asarray(w * mm), jnp.asarray(t * mm)))
    assert abs(got - ref) <= 2e-6 * abs(ref), (got, ref)


def test_the_self_term_derivative_is_exact_in_the_length():
    """`Lp = (mu0/4pi) l^2 g(l, w, t)` is homogeneous of degree one in (l, w, t), so Euler's theorem
    gives `l dLp/dl + w dLp/dw + t dLp/dt = Lp` exactly -- an oracle that needs no second formula."""
    ell, w, t = 2.0 * mm, 0.8 * mm, 0.3 * mm
    Lp = lambda ell, w, t: ell**2 * bar_self_jax(ell, w, t)
    g = jax.grad(Lp, argnums=(0, 1, 2))(ell, w, t)
    assert abs(ell * g[0] + w * g[1] + t * g[2] - Lp(ell, w, t)) <= 1e-9 * Lp(ell, w, t)


# ---- the lattice pass ----------------------------------------------------------------------------


def _two_traces(gap=3.0, pitch=P):
    """Two coplanar traces, the second's width a design variable. Every face is a grid line."""
    a = jno.shape.box(0, 0, 0, 12 * mm, 2 * mm, 0.5 * mm, size=(pitch,) * 3).attach(sigma=CU).name("t1")
    b = (
        jno.shape.box(0, (2 + gap) * mm, 0, 12 * mm, (4 + gap) * mm, 0.5 * mm, size=(pitch,) * 3)
        .attach(sigma=CU)
        .name("t2")
    )
    return a, b


def test_the_traced_pass_at_the_reference_reproduces_the_build():
    """`lattice["regeom"](reference edges)` must give back exactly the arrays the host pass built,
    or the two paths are different discretisations and a traced solve answers a different question."""
    a, b = _two_traces()
    fil = bar_filaments([a, b], sigma=[CU, CU], edges=jno.peec.faces(pitch=P), freq=1e6)
    lat = fil.lattice
    assert lat["graded"] and "regeom" in lat
    pos, mom, self_g, ln, area, skin, nodes = lat["regeom"](tuple(jnp.asarray(e) for e in lat["edges"]))
    for got, ref, name in (
        (pos, fil.pos, "pos"),
        (mom, fil.mom, "mom"),
        (ln, fil.length, "length"),
        (area, fil.area, "area"),
        (skin, fil.skin, "skin"),
        (nodes, fil.nodes, "nodes"),
    ):
        assert np.allclose(np.asarray(got), np.asarray(ref), rtol=0, atol=1e-15), name
    assert np.allclose(np.asarray(self_g), np.asarray(fil.self_g), rtol=2e-6, atol=0), "self term"


def test_faces_puts_a_grid_line_on_every_face_and_nothing_finer_than_asked():
    a, b = _two_traces(gap=3.0)
    fil = bar_filaments([a, b], sigma=[CU, CU], edges=jno.peec.faces(pitch=P))
    ey = np.asarray(fil.lattice["edges"][1])
    for face in (0.0, 2.0, 5.0, 7.0):
        assert np.min(np.abs(ey - face * mm)) < 1e-12 * mm, (face, ey)
    assert np.diff(ey).min() > 0.5 * P and np.diff(ey).max() <= P + 1e-12


def _net(a, b, freq=1e6):
    d = (a + b).domain()
    d.tag("A", lambda x, y, z: x < 0.9 * mm)
    d.tag("B", lambda x, y, z: x > 11.1 * mm)
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freq, grid=jno.peec.faces(pitch=P))


def _ext(sh):
    bnd = np.asarray(sh.bounds(), float).reshape(2, -1)
    return tuple(float(v) for v in np.concatenate([bnd[0], bnd[1]]))


def test_a_traced_solve_at_the_reference_equals_the_untraced_one():
    a, b = _two_traces()
    e = _net(a, b)
    ref = float(jnp.real(e.solve(matrix_free=False).L))
    got = float(jnp.real(e.solve(matrix_free=False, geometry={"t2": _ext(b)}).L))
    assert abs(got - ref) <= 1e-9 * abs(ref), (got, ref)


def test_the_moved_lines_are_the_reference_at_the_reference_and_stretch_between_faces():
    """`_moved_edges` is what a caller compares with the built lines to decide on a rebuild: at the
    reference it is the reference, and a moved face moves its own line while the lines between two
    faces keep their fraction of the gap -- here t2's upper face, so the cells inside t2 stretch."""
    a, b = _two_traces(gap=3.0)
    e = _net(a, b)._build()
    ref = [np.asarray(v) for v in e._solid_info[2]["edges"]]
    same = e._moved_edges({"t2": _ext(b)})
    assert all(np.allclose(np.asarray(m), r, rtol=0, atol=1e-12 * mm) for m, r in zip(same, ref))
    x0, y0, z0, x1, y1, z1 = _ext(b)
    moved = [np.asarray(v) for v in e._moved_edges({"t2": (x0, y0, z0, x1, y1 + 1.0 * mm, z1)})]
    ey, ry = moved[1], ref[1]
    k0, k1 = int(np.argmin(np.abs(ry - y0))), int(np.argmin(np.abs(ry - y1)))
    assert ey[k1] == pytest.approx(y1 + 1.0 * mm, abs=1e-15)
    assert ey[k0] == pytest.approx(y0, abs=1e-15)
    stretch = np.diff(ey[k0 : k1 + 1]) / np.diff(ry[k0 : k1 + 1])
    assert np.allclose(stretch, (y1 - y0 + 1.0 * mm) / (y1 - y0))
    assert np.allclose(moved[0], ref[0], rtol=0, atol=1e-12 * mm) and np.allclose(moved[2], ref[2], rtol=0, atol=1e-12 * mm)


def test_the_gradient_in_a_trace_face_matches_rebuilt_networks():
    """The claim. d L / d(y0 of t2), by AD through the moved grid, against central differences of
    networks REBUILT at y0 +/- h. Both sides are the same cell counts and the same quadrature, so
    they must agree; a small rebuild step keeps the counts (`faces` rounds gap / pitch)."""
    a, b = _two_traces()
    e = _net(a, b)
    x0, y0, z0, x1, y1, z1 = _ext(b)

    def L_of(yy):
        return jnp.real(e.solve(matrix_free=False, geometry={"t2": (x0, yy, z0, x1, y1, z1)}).L)

    g = float(jax.grad(L_of)(y0))
    h = 0.02 * mm
    Ls = []
    for s in (-1.0, 1.0):
        bb = jno.shape.box(x0, y0 + s * h, z0, x1, y1, z1, size=(P,) * 3).attach(sigma=CU).name("t2")
        Ls.append(float(jnp.real(_net(a, bb).solve(matrix_free=False).L)))
    fd = (Ls[1] - Ls[0]) / (2 * h)
    assert abs(g - fd) <= 2e-3 * abs(fd), (g, fd)
    assert g > 0, "the loop widens with the gap, so L must grow"


def test_a_moved_solid_on_the_uniform_lattice_is_refused():
    a, b = _two_traces()
    d = (a + b).domain()
    d.tag("A", lambda x, y, z: x < 0.9 * mm)
    d.tag("B", lambda x, y, z: x > 11.1 * mm)
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    e = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6)
    with pytest.raises(ValueError, match="faces"):
        e.solve(geometry={"t2": _ext(b)})


def test_an_unknown_solid_and_a_face_off_the_grid_are_refused():
    a, b = _two_traces()
    e = _net(a, b)
    with pytest.raises(ValueError, match="names no conductor"):
        e.solve(matrix_free=False, geometry={"t9": _ext(b)})
    x0, y0, z0, x1, y1, z1 = _ext(b)
    with pytest.raises(ValueError, match="6 values"):
        e.solve(matrix_free=False, geometry={"t2": (x0, y0, x1, y1)})


def test_lines_that_would_cross_are_refused_when_concrete():
    a, b = _two_traces()
    e = _net(a, b)
    x0, y0, z0, x1, y1, z1 = _ext(b)
    with pytest.raises(ValueError, match="increasing"):
        e.solve(matrix_free=False, geometry={"t2": (x0, 1.0 * mm, z0, x1, y1, z1)})  # y0 below t1's top


def test_a_shared_face_given_two_values_is_refused_when_concrete():
    """Two touching solids share a grid line; both may name it, but not with different values."""
    a = jno.shape.box(0, 0, 0, 12 * mm, 2 * mm, 0.5 * mm, size=(P,) * 3).attach(sigma=CU).name("t1")
    b = jno.shape.box(0, 2 * mm, 0, 12 * mm, 5 * mm, 0.5 * mm, size=(P,) * 3).attach(sigma=CU).name("t2")
    e = _net(a, b)
    with pytest.raises(ValueError, match="shared"):
        e.solve(matrix_free=False, geometry={"t1": (0, 0, 0, 12 * mm, 2.1 * mm, 0.5 * mm), "t2": _ext(b)})


# ---- the hierarchical operator on a frozen structure ------------------------------------------------


def _welded():
    """Two traces and one wire between them, which is what puts a module on the welded path."""
    a, b = _two_traces()
    w = (
        jno.shape.line(
            [
                (11 * mm, 1 * mm, 0.5 * mm),
                (11 * mm, 1 * mm, 1.5 * mm),
                (11 * mm, 6 * mm, 1.5 * mm),
                (11 * mm, 6 * mm, 0.5 * mm),
            ],
            r=0.1 * mm,
            size=P,
        )
        .attach(sigma=3.8e7)
        .name("w")
    )
    d = (a + b + w).domain()
    d.tag("A", lambda x, y, z: (x < 0.9 * mm) & (y < 2.1 * mm))
    d.tag("B", lambda x, y, z: (x < 0.9 * mm) & (y > 4.9 * mm))
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6, grid=jno.peec.faces(pitch=P)), b


def test_the_hierarchical_path_takes_a_traced_lattice_and_is_differentiable():
    """`hmatrix.build` refuses a tracer, so the structure is built once from the reference geometry
    and refilled from the moved one. At the reference that is the same operator; away from it the
    gradient must agree with central differences of the SAME traced solve."""
    e, b = _welded()
    op = jno.solve.hierarchical(tol=1e-8)
    ref = float(jnp.real(e.solve(operator=op).L))
    x0, y0, z0, x1, y1, z1 = _ext(b)

    def L_of(yy):
        return jnp.real(e.solve(operator=op, geometry={"t2": (x0, yy, z0, x1, y1, z1)}).L)

    assert abs(float(L_of(y0)) - ref) <= 1e-9 * ref
    g = float(jax.grad(L_of)(y0))
    h = 0.02 * mm
    fd = float((L_of(y0 + h) - L_of(y0 - h)) / (2 * h))
    assert abs(g - fd) <= 2e-3 * abs(fd), (g, fd)
    assert g > 0


# ---- a closed form ----------------------------------------------------------------------------------


def test_the_mutual_term_derivative_matches_the_filament_closed_form():
    """Two thin parallel bars a distance d apart: `M = (mu0 l / 2 pi)[asinh(l/d) - sqrt(1 + (d/l)^2)
    + d/l]` for filaments (Grover), and dM/dd follows. Bars 1 % of d wide reproduce it to O(1e-4)."""
    ell, dd, w = 4 * mm, 2 * mm, 0.02 * mm
    a = jno.shape.box(0, 0, 0, ell, w, w, size=(ell / 2, w, w)).attach(sigma=CU).name("a")
    b = jno.shape.box(0, dd, 0, ell, dd + w, w, size=(ell / 2, w, w)).attach(sigma=CU).name("b")
    fil = bar_filaments([a, b], sigma=[CU, CU], edges=jno.peec.faces(pitch=ell / 2))
    lat = fil.lattice
    ia, ib = (
        [i for i, p in enumerate(np.asarray(fil.part)) if p == 0],
        [i for i, p in enumerate(np.asarray(fil.part)) if p == 1],
    )
    assert len(ia) == 1 and len(ib) == 1, "one bar each: two cells along x, one bar between their centres"
    # face index of b's lower y face
    ey = np.asarray(lat["edges"][1])
    k = int(np.argmin(np.abs(ey - dd)))

    def M_of(d_):
        ey_new = jnp.asarray(ey).at[k].set(d_).at[k + 1].set(d_ + w)
        pos, mom, self_g, *_ = lat["regeom"]((jnp.asarray(lat["edges"][0]), ey_new, jnp.asarray(lat["edges"][2])))
        K = pair_matrix(pos, mom, lambda r: 1.0 / r, self_g, group=fil.group) * (MU0 / (4 * jnp.pi))
        return K[ia[0], ib[0]]

    # two cells along x give ONE bar, centre to centre: its length is l / 2
    lb = ell / 2
    M_cf = lambda d_: (MU0 * lb / (2 * np.pi)) * (np.arcsinh(lb / d_) - np.sqrt(1 + (d_ / lb) ** 2) + d_ / lb)
    # 3 Gauss points along a bar 2.5x longer than the separation: the rule, not the width, sets the
    # 1.6e-3 here (the line docstring measures 1.2 % for three points on collinear neighbours)
    assert abs(float(M_of(dd)) - M_cf(dd)) <= 3e-3 * M_cf(dd)
    g = float(jax.grad(M_of)(dd))
    hh = 1e-6 * mm
    g_cf = (M_cf(dd + hh) - M_cf(dd - hh)) / (2 * hh)
    assert abs(g - g_cf) <= 1e-2 * abs(g_cf), (g, g_cf)
    # and the derivative of the QUADRATURE itself is exact: AD against central differences of M_of
    fd = float((M_of(dd + hh) - M_of(dd - hh)) / (2 * hh))
    assert abs(g - fd) <= 1e-6 * abs(fd), (g, fd)


def test_a_crossing_design_under_a_trace_stays_finite_and_is_inactive_when_feasible():
    """An optimiser's line search visits infeasible designs. Traced, a crossing cannot raise, so a
    cell keeps a fifth of its reference width; at a feasible design the floor must not act."""
    a, b = _two_traces()
    e = _net(a, b)
    x0, y0, z0, x1, y1, z1 = _ext(b)
    L_of = lambda yy: jnp.real(e.solve(matrix_free=False, geometry={"t2": (x0, yy, z0, x1, y1, z1)}).L)
    bad = float(jax.jit(L_of)(1.0 * mm))  # y0 below t1's top: would cross
    assert np.isfinite(bad)
    ok = float(jax.jit(L_of)(y0 - 0.3 * mm))
    ref = float(L_of(y0 - 0.3 * mm))
    assert abs(ok - ref) <= 1e-12 * abs(ref)


def test_the_hierarchical_structure_is_built_once_per_network(monkeypatch):
    """An untraced solve and the traced solves after it share one structure: `hmatrix.build` runs
    once per block, not once per trace (it was keyed on the sliced block, a fresh object per call)."""
    import jno.utils.solver.hmatrix as HM
    import jno.utils.solver.peec as PS

    calls = []
    real = HM.build
    monkeypatch.setattr(HM, "build", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    PS._HSTRUCT_CACHE.clear()
    e, b = _welded()
    op = jno.solve.hierarchical(tol=1e-8)
    e.solve(operator=op)
    n0 = len(calls)
    assert n0 >= 1
    x0, y0, z0, x1, y1, z1 = _ext(b)
    for yy in (y0, y0 - 0.1 * mm):
        jax.jit(lambda v: jnp.real(e.solve(operator=op, geometry={"t2": (x0, v, z0, x1, y1, z1)}).L))(yy)
    assert len(calls) == n0, f"the structure was rebuilt under the trace: {len(calls)} builds for {n0} blocks"


def test_the_near_values_in_jax_equal_the_host_ones_at_the_reference():
    """The preconditioner's near field follows the geometry: at the reference the jax values must
    be the host values, pair for pair."""
    from jno.utils.solver.peec import _near_values, near_block

    e, _b = _welded()
    nr, nc, nv = near_block(e.fil, lambda r: 1.0 / r, mu_scale=1.0, reach=2.0)
    got = np.asarray(_near_values(e.fil, nr, nc, lambda r: 1.0 / r, 1.0))
    assert got.shape == nv.shape and np.allclose(got, nv, rtol=1e-12, atol=0)
