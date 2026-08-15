"""``domain.by_tag`` / ``domain.attach`` on a boundary tag — a per-FACET coefficient.

``by_region`` exists so a multi-material volume is ONE equation instead of one term per region. This
is the same thing for the boundary: ``d.by_tag({"wall": 25.0, "lid": 5.0})`` lets a single surface
term carry a coefficient that varies across tags, instead of a term per tag::

    robin = d.h * (ub - T_inf) * vb          # one term, h varies across the boundary

It desugars to ``sum_t TagMask(t) * value``. The mask is built from the assembler's own facet
selection (``fem_native._region_faces``), *not* by re-evaluating the tag predicate — so a
``TagMask("wall")`` covers exactly the facets a Dirichlet condition on ``"wall"`` pins, and no
tolerance-tight predicate is re-run under float32 where ``x > 1 - 1e-9`` rounds to ``x > 1.0``.

Scope, asserted below: nodal Lagrange surface terms only. A volume term, a non-nodal space and 1-D
each raise rather than integrating over the wrong thing.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno


def _square(size=0.34):
    """Unit square with `left` / `right` boundary tags and the symbols a Robin term needs."""
    d = jno.Shape.rect(0, 0, 1, 1, size=size).domain()
    d.tag("left", lambda x, y: x < 1e-9)
    d.tag("right", lambda x, y: x > 1 - 1e-9)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    binds = {}
    for tag in ("interior", "boundary", "left", "right"):
        a, b, _ = d.variable(tag, split=True)
        binds[tag] = (u.bind(x=a, y=b), v.bind(x=a, y=b))
    ui, vi = binds["interior"]
    return d, u, v, binds, ui.x * vi.x + ui.y * vi.y


def _A(fem):
    A = fem.A
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _b(fem):
    return np.asarray(fem.b).reshape(-1)


# --------------------------------------------------------------------------------------
# the equivalence that proves the feature
# --------------------------------------------------------------------------------------
def test_one_term_over_the_boundary_equals_the_per_tag_loop():
    """THE test: ``h * ub * vb`` over the whole boundary with a per-tag ``h`` must assemble the
    identical operator to two terms, one per tag. The surface twin of
    ``test_by_region_stiffness_matches_per_region_loop``."""
    d, _u, _v, binds, stiff = _square()
    ub, vb = binds["boundary"]
    ul, vl = binds["left"]
    ur, vr = binds["right"]

    one = _A(jno.fem([stiff, d.by_tag({"left": 3.0, "right": 7.0}) * ub * vb]))
    loop = _A(jno.fem([stiff, 3.0 * ul * vl, 7.0 * ur * vr]))
    np.testing.assert_allclose(one, loop, rtol=1e-5, atol=1e-5)
    # ...and it is not a no-op: the Robin terms genuinely changed the operator.
    assert not np.allclose(one, _A(jno.fem([stiff])), atol=1e-4)


def test_the_load_vector_matches_the_per_tag_loop_too():
    """A per-tag surface SOURCE, not just an operator: the same equivalence must hold for ``b``."""
    d, _u, _v, binds, stiff = _square()
    _ub, vb = binds["boundary"]
    _ul, vl = binds["left"]
    _ur, vr = binds["right"]

    one = _b(jno.fem([stiff, d.by_tag({"left": 2.0, "right": 5.0}) * vb]))
    loop = _b(jno.fem([stiff, 2.0 * vl, 5.0 * vr]))
    np.testing.assert_allclose(one, loop, rtol=1e-5, atol=1e-5)
    assert np.abs(one).max() > 0.0


def test_the_mask_selects_exactly_the_dirichlet_facets():
    """The mask comes from the assembler's own facet selection, so a tag's TagMask must cover exactly
    the facets a Dirichlet condition on that tag pins -- one selection rule for the whole library, not
    a second one that can quietly disagree with the first."""
    d, u, _v, binds, stiff = _square()
    _ub, vb = binds["boundary"]

    # Rows pinned by a Dirichlet condition on `left`.
    A_dir = _A(jno.fem([stiff, u(*d.variable("left", split=True)[:2]) - 0.0]))
    pinned = {i for i in range(A_dir.shape[0]) if np.isclose(A_dir[i, i], 1.0) and np.allclose(np.delete(A_dir[i], i), 0.0)}

    # Nodes touched by a `left`-only surface load.
    load = _b(jno.fem([stiff, d.by_tag({"left": 1.0}) * vb]))
    touched = {i for i in range(load.shape[0]) if abs(load[i]) > 1e-9}

    assert touched, "the by_tag surface load touched no node at all"
    assert touched <= pinned, "by_tag reached nodes the Dirichlet selection for the same tag does not"


def test_three_tags_each_keep_their_own_value():
    """Extremes: more than two tags, including a zero one -- a zero coefficient must not be confused
    with an absent tag."""
    d = jno.Shape.rect(0, 0, 1, 1, size=0.34).domain()
    d.tag("l", lambda x, y: x < 1e-9)
    d.tag("r", lambda x, y: x > 1 - 1e-9)
    d.tag("b", lambda x, y: y < 1e-9)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    xb, yb, _ = d.variable("boundary", split=True)
    ub, vb = u.bind(x=xb, y=yb), v.bind(x=xb, y=yb)
    per = {}
    for t in ("l", "r", "b"):
        a, c, _ = d.variable(t, split=True)
        per[t] = (u.bind(x=a, y=c), v.bind(x=a, y=c))

    stiff = ui.x * vi.x + ui.y * vi.y
    vals = {"l": 4.0, "r": 0.0, "b": 9.0}
    one = _A(jno.fem([stiff, d.by_tag(vals) * ub * vb]))
    loop = _A(jno.fem([stiff] + [vals[t] * per[t][0] * per[t][1] for t in ("l", "r", "b")]))
    np.testing.assert_allclose(one, loop, rtol=1e-5, atol=1e-5)


def test_default_fills_the_untagged_boundary():
    """``default=`` covers the facets no listed tag claims -- the same semantics as ``by_region``."""
    d, u, v, binds, stiff = _square()
    ub, vb = binds["boundary"]
    strict = _A(jno.fem([stiff, d.by_tag({"left": 3.0}) * ub * vb]))
    filled = _A(jno.fem([stiff, d.by_tag({"left": 3.0}, default=1.0) * ub * vb]))
    assert not np.allclose(strict, filled, atol=1e-4), "default= reached no extra facet"


def test_a_view_typed_value_survives_by_tag():
    """The same view-preservation `by_region` has: a typed coefficient must come back typed."""
    from jno.trace.views import ScalarView

    d, _u, _v, _binds, _stiff = _square()
    xb, _yb, _ = d.variable("boundary", split=True)
    out = d.by_tag({"left": ScalarView(2.0 + 0.0 * xb), "right": ScalarView(1.0 + 0.0 * xb)})
    assert type(out).__name__ == "ScalarView"


def test_mixed_view_types_across_tags_raise():
    from jno.trace.views import MatrixView, VectorView

    d, _u, _v, _binds, _stiff = _square()
    xb, yb, _ = d.variable("boundary", split=True)
    vec = jno.np.vector
    m = MatrixView(vec(1.0 + 0.0 * xb, 0.0 * yb, 0.0 * xb, 1.0 + 0.0 * yb).expr).from_flat(2, 2)
    w = VectorView(vec(1.0 + 0.0 * xb, 0.0 * yb).expr)
    with pytest.raises(ValueError, match="mix view types"):
        d.by_tag({"left": m, "right": w})


# --------------------------------------------------------------------------------------
# attach on a tag
# --------------------------------------------------------------------------------------
def test_attach_on_a_tag_reads_back_as_the_per_facet_coefficient():
    d, _u, _v, binds, stiff = _square()
    ub, vb = binds["boundary"]
    ul, vl = binds["left"]
    ur, vr = binds["right"]
    d.attach("left", h=3.0).attach("right", h=7.0)

    one = _A(jno.fem([stiff, d.h * ub * vb]))
    loop = _A(jno.fem([stiff, 3.0 * ul * vl, 7.0 * ur * vr]))
    np.testing.assert_allclose(one, loop, rtol=1e-5, atol=1e-5)


def test_attach_returns_self_so_declarations_chain():
    d, *_ = _square()
    assert d.attach("left", h=1.0) is d


def test_attach_records_the_kind_from_what_the_target_owns():
    d, *_ = _square()
    d.tag("blob", lambda x, y: (x > 0.3) & (x < 0.7) & (y > 0.3) & (y < 0.7))  # cells, no facets
    d.attach("left", h=1.0)
    d.attach("blob", q=1.0)
    kinds = d.__dict__["_attachment_kind"]
    assert kinds["left"] == "surface" and kinds["blob"] == "volume"
    assert "TagMask(left)" in str(d.h)
    assert "RegionMask(blob)" in str(d.q)  # an interior tag is still a VOLUME coefficient


def test_a_tag_owning_both_cells_and_facets_is_ambiguous_and_raises():
    """`domain.tag` names any subset, interior or boundary. A half-plane owns both, so whether its
    properties are volume or surface quantities cannot be decided -- and guessing picks the wrong
    mask, which is wrong physics rather than an error."""
    d, *_ = _square()
    d.tag("halfplane", lambda x, y: x < 0.5)
    with pytest.raises(ValueError, match="owns both interior cells and boundary facets"):
        d.attach("halfplane", k=1.0)


def test_attach_rejects_an_unknown_target():
    d, *_ = _square()
    with pytest.raises(ValueError, match="unknown target"):
        d.attach("nope", k=1.0)


def test_a_property_declared_on_both_a_region_and_a_tag_raises_on_read():
    """One name, two meanings: integrated over cells in one place and over facets in another."""
    d = (
        jno.Shape.rect(0, 0, 1, 1, size=0.5).name("a").attach(rho=1.0)
        + jno.Shape.rect(0, 0, 2, 1, size=0.5).name("b").attach(rho=2.0)
    ).domain()
    d.tag("edge", lambda x, y: y < 1e-9)
    d.attach("edge", rho=5.0)
    with pytest.raises(AttributeError, match="declared on both a volume region and a boundary tag"):
        d.rho


def test_a_surface_property_needs_no_completeness_rule():
    """Unlike regions, tags are NOT a partition of the boundary -- untagged boundary is deliberately
    natural in jNO -- so declaring `h` on one tag only must work, not raise."""
    d, _u, _v, binds, stiff = _square()
    ub, vb = binds["boundary"]
    d.attach("left", h=3.0)
    assert np.all(np.isfinite(_A(jno.fem([stiff, d.h * ub * vb]))))


# --------------------------------------------------------------------------------------
# scope limits — each must be LOUD
# --------------------------------------------------------------------------------------
def test_a_tag_mask_in_a_volume_term_raises():
    """There is no facet to indicate in a volume term. Defaulting to 1 would integrate over the whole
    domain and defaulting to 0 would drop the term -- both silent."""
    d, _u, _v, _binds, stiff = _square()
    with pytest.raises(NotImplementedError, match="per-tag surface integration"):
        jno.fem([d.by_tag({"left": 3.0}) * stiff]).A


def test_a_tag_owning_no_facet_raises_rather_than_integrating_over_nothing():
    d, _u, _v, binds, stiff = _square()
    ub, vb = binds["boundary"]
    d.tag("interior_blob", lambda x, y: (x > 0.4) & (x < 0.6) & (y > 0.4) & (y < 0.6))
    with pytest.raises(ValueError, match="owns no boundary facet"):
        jno.fem([stiff, d.by_tag({"interior_blob": 1.0}) * ub * vb]).A


def test_by_tag_rejects_an_unknown_tag():
    d, *_ = _square()
    with pytest.raises(ValueError, match="unknown tag"):
        d.by_tag({"nope": 1.0})


def test_by_tag_in_1d_raises():
    """1-D boundary 'facets' are vertices and this path threads no mask."""
    d = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
    d.tag("lo", lambda x: x < 1e-9)
    u, v = d.fem_symbols()
    (xi,) = d.variable("interior", split=True)[:1]
    (xb,) = d.variable("boundary", split=True)[:1]
    ui, vi = u.bind(x=xi), v.bind(x=xi)
    ub, vb = u.bind(x=xb), v.bind(x=xb)
    with pytest.raises(NotImplementedError, match="per-tag surface integration"):
        jno.fem([ui.x * vi.x, d.by_tag({"lo": 3.0}) * ub * vb]).A


def test_by_tag_on_a_non_nodal_space_raises():
    """The non-nodal assembler CLASSIFIES boundary terms by pattern and lifts the coefficient out, so
    an unevaluated TagMask could ride into a host-assembled surface mass and weight every facet alike.
    Rejected explicitly rather than left to chance."""
    pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")
    inner = jno.np.inner
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    d.tag("x0", lambda x, y, z: x < 1e-9)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    cb = d.variable("boundary", split=True)
    ub, vb = u.bind(x=cb[0], y=cb[1], z=cb[2]), v.bind(x=cb[0], y=cb[1], z=cb[2])
    cu, cv = u.vector.curl(c[0], c[1], c[2]), v.vector.curl(c[0], c[1], c[2])
    with pytest.raises(NotImplementedError, match="not supported on a non-nodal space"):
        jno.fem([inner(cu, cv), d.by_tag({"x0": 2.0}) * inner(ub, vb)]).A
