"""A Dirichlet pin on a `domain.tag` point set (a corner node, a stretch of an edge) works on a domain
built from named regions, as on a single-region one: the tag is not a boundary region, so it classified as a
volume term and was refused with "a residual with the trial but no test function must live on a boundary region"."""

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("kind", ["corner", "edge_band"])
def test_a_point_set_pin_on_a_multi_region_domain_fixes_its_nodes(kind):
    W = 1.0
    lower = jno.shape.rect(0, 0, 1, W / 2).name("lower")
    upper = jno.shape.rect(0, W / 2, 1, W).name("upper")
    dom = (lower + upper).domain()
    if kind == "corner":
        dom.tag("pinned", lambda x, y: (x < 1e-9) & (y > W - 1e-9))
    else:  # the part of the top edge with |x - 0.5| < 0.3
        dom.tag("pinned", lambda x, y: (y > W - 1e-9) & (abs(x - 0.5) < 0.3))
    u, v = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    xl, yl, _ = dom.variable("left", split=True)
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y - 1.0 * vi,
            u(xl, yl) - 0.0,
            u(*dom.variable("pinned", split=True)) - 2.0,
        ]
    )
    assert "dirichlet@pinned" in fem.classification
    sol = np.asarray(fem.solve()).reshape(-1)
    pts = np.asarray(fem.points)
    sel = pts[:, 0] < 1e-9
    sel &= pts[:, 1] > W - 1e-9
    if kind == "edge_band":
        sel = (pts[:, 1] > W - 1e-9) & (np.abs(pts[:, 0] - 0.5) < 0.3)
    assert sel.any()
    np.testing.assert_allclose(sol[sel], 2.0, atol=1e-10)


def _band_problem(pred, boundary_value=2.0):
    d = jno.shape.rect(0, 0, 1, 1, size=0.1).domain()
    d.tag("band", pred)
    x, y, _ = d.variable("interior", split=True)
    b, k = d.variable("boundary", split=True), d.variable("band", split=True)
    u, v = d.fem_symbols()
    a, t = u.bind(x=x, y=y), v.bind(x=x, y=y)
    weak = a.x * t.x + a.y * t.y - 1.0 * t
    pts = np.asarray(d.mesh.points)[:, :2]
    band = np.asarray(pred(pts[:, 0], pts[:, 1]))
    interior = (pts > 1e-9).all(1) & (pts < 1 - 1e-9).all(1)
    return d, (weak, u(b[0], b[1]) - boundary_value, u(k[0], k[1]) - 2.0), band, interior


def test_a_point_set_pin_wholly_inside_the_domain_fixes_its_nodes():
    """A tag whose predicate holds only at interior nodes is an interior pin. It used to pin NOTHING, with no error
    (the interior fallback intersected the tag with the boundary). Oracle: the pinned nodes equal the value and the
    rest equals the same problem with those rows replaced by identity rows by hand."""

    def pred(x, y):
        return (np.abs(x - 0.5) < 0.12) & (np.abs(y - 0.5) < 0.12)

    d, terms, band, interior = _band_problem(pred)
    assert band.sum() > 0 and not (band & ~interior).any()
    sol = np.asarray(jno.fem(list(terms)).solve()).reshape(-1)
    np.testing.assert_allclose(sol[band], 2.0, atol=1e-10)

    A, rhs = jno.fem(list(terms[:2])).operator  # the boundary pin only
    A, rhs = np.array(A.todense()), np.array(rhs).reshape(-1)
    for i in np.flatnonzero(band):
        A[i, :] = 0.0
        A[i, i] = 1.0
        rhs[i] = 2.0
    np.testing.assert_allclose(sol, np.linalg.solve(A, rhs), rtol=1e-5, atol=1e-6)  # the iterative solve, to its tolerance


def test_a_pin_on_a_tag_that_reaches_the_boundary_pins_its_boundary_part_and_says_so(capfd):
    """A tag that also touches the boundary keeps the deliberate reading of a boundary pin (a thick band around a
    cylinder pins the cylinder, not the fluid around it). Its interior nodes stay free, which used to go unsaid:
    one warning now names the tag and the counts."""

    def pred(x, y):
        return np.abs(x - 0.5) < 0.12  # a band across the square, top to bottom

    d, terms, band, interior = _band_problem(pred, boundary_value=0.0)
    fem = jno.fem(list(terms))
    out, err = capfd.readouterr()
    n_free = int((band & interior).sum())
    assert n_free > 0 and f"{n_free} interior node(s)" in out + err and "'band'" in out + err
    sol = np.asarray(fem.solve()).reshape(-1)
    np.testing.assert_allclose(sol[band & ~interior], 2.0, atol=1e-10)  # the boundary part is pinned
    assert (np.abs(sol[band & interior] - 2.0) > 1e-3).all()  # the interior part is not
