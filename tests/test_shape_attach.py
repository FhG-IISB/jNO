"""``Shape.attach`` — material properties declared on the region, read back as ``d.<name>``.

``.attach(k=220.0)`` on a named region makes ``d.k`` the per-region coefficient over every region
that declared a ``k`` — the same object ``d.by_region({...})`` returns, so it drops straight into a
weak form. A value may be anything the jNO stack treats as a coefficient: a scalar, an array, a
symbolic expression, a typed view (``ScalarView`` / ``VectorView`` / ``MatrixView``), a trainable
``jno.np.parameter``, or a plain function of the coordinates.

Also covers the two prerequisites the feature exposed: ``by_region`` accepting ``Shape.regions``
sub-region names at all, and ``Shape.regions`` accepting a ``{name: shape}`` dict for names that are
not valid Python identifiers.

Boundary-tag attachment (``d.attach("wall", h=25.0)`` → ``d.h`` as a per-facet coefficient) lives in
``tests/test_fem_by_tag.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno


def _two_regions(**kw):
    """A 1x1 inclusion inside a 2x1 plate, each carrying whatever `.attach(...)` kwargs are given."""
    a = jno.Shape.rect(0, 0, 1, 1, size=0.3).name("a").attach(**kw.get("a", {}))
    b = jno.Shape.rect(0, 0, 2, 1, size=0.3).name("b").attach(**kw.get("b", {}))
    return a + b


def _poisson(d, k):
    """`k` as the conductivity of a pinned Poisson problem -> the dense stiffness matrix."""
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    A = jno.fem([k * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0]).A
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


# --------------------------------------------------------------------------------------
# attach -> d.<name>
# --------------------------------------------------------------------------------------
def test_attached_property_is_the_by_region_coefficient():
    d = _two_regions(a={"k": 220.0}, b={"k": 0.186}).domain()
    assert str(d.k) == str(d.by_region({"a": 220.0, "b": 0.186}))


def test_several_properties_are_independent():
    d = _two_regions(a={"k": 1.0, "eta": 0.44}, b={"k": 2.0, "eta": 0.11}).domain()
    assert str(d.eta) == str(d.by_region({"a": 0.44, "b": 0.11}))
    assert str(d.k) == str(d.by_region({"a": 1.0, "b": 2.0}))


def test_missing_in_one_region_raises_and_names_it():
    """Hard error, not a silent default: a region that forgot a property must be reported."""
    d = _two_regions(a={"k": 1.0}, b={}).domain()
    with pytest.raises(AttributeError, match=r"region\(s\) \['b'\] never attached a 'k'"):
        d.k


def test_unattached_name_is_a_normal_attribute_error():
    d = _two_regions(a={"k": 1.0}, b={"k": 2.0}).domain()
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        d.nope


@pytest.mark.parametrize("clash", ["mesh", "by_region", "tag", "dimension"])
def test_name_colliding_with_a_domain_attribute_raises_at_build(clash):
    """`__getattr__` only fires when normal lookup FAILS, so a colliding name would be silently
    shadowed by the real attribute. It has to be caught when the domain is built."""
    with pytest.raises(ValueError, match="collide with existing jno.domain attributes"):
        _two_regions(a={clash: 1.0}, b={clash: 2.0}).domain()


def test_attach_merges_and_last_wins():
    s = jno.Shape.rect(0, 0, 1, 1).name("a").attach(k=1.0, eta=9.0).attach(k=2.0)
    assert s._attach == {"k": 2.0, "eta": 9.0}


def test_attach_survives_name_sized_and_curved():
    s = jno.Shape.rect(0, 0, 1, 1).attach(k=1.0).name("a").sized(0.1).curved()
    assert s._attach == {"k": 1.0}
    assert s._region_name == "a"


def test_attach_survives_boolean_and_transform_ops():
    s = (jno.Shape.rect(0, 0, 2, 2).attach(k=1.0) - jno.Shape.rect(0, 0, 1, 1)).translate((1.0, 0.0))
    assert s._attach == {"k": 1.0}


def test_attach_does_not_affect_geometric_equality_or_hashing():
    """`_attach` is compare=False: two geometrically identical shapes stay equal (and hashable)
    however they are materialled."""
    plain = jno.Shape.rect(0, 0, 1, 1)
    assert plain == plain.attach(k=1.0)
    assert len({plain, plain.attach(k=1.0)}) == 1


def test_attached_callable_is_resolved_against_the_domain_coordinates():
    """A spatially varying value has to be built from `d.variable(...)`, which does not exist while
    the geometry plan is being written — so a plain function is called at read time instead."""
    d = _two_regions(a={"k": lambda x, y: 2.0 + 0.5 * y}, b={"k": 1.0}).domain()
    x, y, _t = d.variable("interior", split=True)
    assert str(d.k) == str(d.by_region({"a": 2.0 + 0.5 * y, "b": 1.0}))


def test_attached_callable_sees_only_spatial_coordinates():
    """Two arguments in 2-D — the time slot `variable(split=True)` also returns is not passed."""
    d = _two_regions(a={"k": lambda *c: float(len(c))}, b={"k": 0.0}).domain()
    assert "Literal(2.0)" in str(d.k)


def test_attached_value_may_be_a_jax_array():
    jnp = pytest.importorskip("jax.numpy")
    v = jnp.asarray(3.0)
    d = _two_regions(a={"k": v}, b={"k": 1.0}).domain()
    assert str(d.k) == str(d.by_region({"a": v, "b": 1.0}))


# --------------------------------------------------------------------------------------
# attach through the Shape.regions dict form (names that are not Python identifiers)
# --------------------------------------------------------------------------------------
def test_attach_through_the_regions_dict_form():
    d = jno.Shape.regions(
        {
            "Quartz.1": jno.Shape.rect(0, 0, 1, 1, size=0.3).attach(k=1.0),
            "Quartz.2": jno.Shape.rect(0, 0, 2, 1, size=0.3).attach(k=2.0),
        }
    ).domain()
    assert str(d.k) == str(d.by_region({"Quartz.1": 1.0, "Quartz.2": 2.0}))


# --------------------------------------------------------------------------------------
# a Shape-built domain must reach the same machinery a polygon-built one does
# --------------------------------------------------------------------------------------
def test_region_masks_partition_when_a_background_region_encloses_everything():
    """`by_region` sums RegionMask*value, so the masks have to be a PARTITION. Declaring a bounding
    region last to pick up the leftover void is the normal idiom, and that region contains every cell
    -- if priority is not subtracted out, its coefficient is added on top of every other region's.
    Silent and physical rather than an error: on the furnace it added 0.186 to the insulation's
    k = 0.5 and pulled the solution 780 K cold."""
    from jno.utils.solver.fem_utils import _cell_region_mask

    inner = jno.Shape.rect(2, 2, 4, 4, size=0.5).name("inner")
    background = jno.Shape.rect(0, 0, 6, 6, size=0.5).name("background")  # encloses `inner`
    d = (inner + background).domain()

    masks = np.array([_cell_region_mask(d, nm) for nm in d._shape_regions])
    total = masks.sum(axis=0)
    assert set(np.unique(total)) == {1.0}, "region masks must partition the cells exactly once"
    assert masks[0].sum() > 0, "the enclosed region must still own its own cells"


def test_attached_returns_the_raw_mapping_for_partial_consumers():
    """`d.<name>` is the coefficient and demands every region declare the property; `d.attached(name)`
    is the mapping and does not -- an enclosure spans only some regions, and `enclosure.emissivity`
    takes a {tag: eps} dict."""
    d = _two_regions(a={"eps": 0.8}, b={"k": 1.0}).domain()
    assert d.attached("eps") == {"a": 0.8}
    with pytest.raises(AttributeError):
        d.eps  # 'b' never declared one, so the COEFFICIENT is still an error


def test_attached_rejects_an_undeclared_property():
    d = _two_regions(a={"k": 1.0}, b={"k": 2.0}).domain()
    with pytest.raises(KeyError, match="no region declared"):
        d.attached("nope")


# --------------------------------------------------------------------------------------
# the two silent drops: a region is a region whether or not it has a sibling
# --------------------------------------------------------------------------------------
def test_a_single_named_region_round_trips_its_attachment():
    """`domain.__init__` gated on the `("regions", ...)` node, which a lone named shape is not -- so
    its attachment was collected from nowhere and `d.k` reported a bare "no attribute 'k'". Nothing
    told the user the declaration had been ignored."""
    d = jno.Shape.rect(0, 0, 1, 1, size=0.4).name("a").attach(k=2.0).domain()
    assert str(d.k) == str(d.by_region({"a": 2.0}))
    assert d.attached("k") == {"a": 2.0}


def test_attach_without_a_name_raises_at_build():
    """The other half of the same hole: with no region name there is nothing to attach to."""
    with pytest.raises(ValueError, match="no region name"):
        jno.Shape.rect(0, 0, 1, 1, size=0.4).attach(k=2.0).domain()


def test_a_single_named_region_still_masks_every_cell():
    """The lone region owns the whole mesh, so its coefficient must not restrict anything away."""
    d = jno.Shape.rect(0, 0, 1, 1, size=0.4).name("a").attach(k=3.0).domain()
    np.testing.assert_allclose(_poisson(d, d.k), _poisson(d, 3.0), rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------------------
# the value-type spectrum — each asserted to ASSEMBLE, not merely to build an expression
# --------------------------------------------------------------------------------------
def test_scalar_attachment_assembles_the_per_region_loop_matrix():
    # float32 assembly: `d.k` and the explicit `by_region` build equal-but-distinct expression trees,
    # so the two sums round differently in the last bit or two. Compare at float32 precision, not
    # bitwise (a 1e-6 atol on entries of order 7 is BELOW the float32 rounding of that magnitude).
    d = _two_regions(a={"k": 3.0}, b={"k": 7.0}).domain()
    np.testing.assert_allclose(_poisson(d, d.k), _poisson(d, d.by_region({"a": 3.0, "b": 7.0})), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("value", [0.0, -2.5, 1e6])
def test_extreme_scalar_values_assemble(value):
    """Zero, negative and large: a zero coefficient must not be mistaken for an absent one."""
    d = _two_regions(a={"k": value}, b={"k": 1.0}).domain()
    A = _poisson(d, d.k)
    assert np.all(np.isfinite(A))
    np.testing.assert_allclose(A, _poisson(d, d.by_region({"a": value, "b": 1.0})), rtol=1e-5, atol=1e-5)


def test_callable_attachment_assembles():
    d = _two_regions(a={"k": lambda x, y: 2.0 + 0.5 * y}, b={"k": 1.0}).domain()
    assert np.all(np.isfinite(_poisson(d, d.k)))


def test_trainable_attachment_recovers():
    """An attached `jno.np.parameter` must stay trainable -- otherwise a per-region material could be
    declared but never fitted. Mirrors `test_by_region_trainable_value_recovers`, through `attach`."""
    import jax
    import optax

    truth_d = _two_regions(a={"k": 4.0}, b={"k": 1.0}).domain()
    u, v = truth_d.fem_symbols()
    xi, yi, _ = truth_d.variable("interior", split=True)
    xb, yb, _ = truth_d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    f = 5.0 * (xi * (1 - xi) + yi * (1 - yi))
    truth = jno.fem([truth_d.k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0]).solve()

    kA = jno.np.parameter((1,), name="kA", key=jax.random.PRNGKey(0))
    kA.initialize(jax.nn.initializers.constant(1.0))
    d = _two_regions(a={"k": kA}, b={"k": 1.0}).domain()
    u2, v2 = d.fem_symbols()
    xi2, yi2, _ = d.variable("interior", split=True)
    xb2, yb2, _ = d.variable("boundary", split=True)
    ui2, vi2 = u2.bind(x=xi2, y=yi2), v2.bind(x=xi2, y=yi2)
    f2 = 5.0 * (xi2 * (1 - xi2) + yi2 * (1 - yi2))
    fem = jno.fem([d.k * (ui2.x * vi2.x + ui2.y * vi2.y) - f2 * vi2, u2(xb2, yb2) - 0.0])
    crux = jno.core([(fem.solve() - truth).mse], domain=d)
    kA.optimizer(optax.adam(3e-1))
    crux.solve(250)
    rec = float(np.asarray(crux.eval([kA])).reshape(-1)[0])
    assert abs(rec - 4.0) < 0.3, f"an attached trainable value should recover k=4.0 (got {rec:.3f})"


def test_scalar_view_attachment_assembles():
    """A ScalarView value stays a ScalarView through `by_region` and integrates as a coefficient."""
    from jno.trace.views import ScalarView

    d = _two_regions().domain()
    x, _y, _ = d.variable("interior", split=True)
    sv = ScalarView(2.0 + 0.0 * x)
    d2 = _two_regions(a={"k": sv}, b={"k": 1.0}).domain()
    assert type(d2.k).__name__ == "ScalarView"
    assert np.all(np.isfinite(_poisson(d2, d2.k)))


def test_vector_view_attachment_survives_and_its_components_integrate():
    """A per-region velocity: the VectorView must come back OUT of `by_region` as a VectorView, or
    `b[0]` -- the whole point of a vector-valued material -- is not reachable."""
    from jno.trace.views import VectorView

    d = _two_regions().domain()
    x, y, _ = d.variable("interior", split=True)
    vec = jno.np.vector
    bA = VectorView(vec(1.0 + 0.0 * x, 0.0 * y).expr)
    bB = VectorView(vec(0.0 * x, 1.0 + 0.0 * y).expr)
    d2 = _two_regions(a={"b": bA}, b={"b": bB}).domain()
    assert type(d2.b).__name__ == "VectorView"

    u, v = d2.fem_symbols()
    xi, yi, _ = d2.variable("interior", split=True)
    xb, yb, _ = d2.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    bb = d2.b
    A = jno.fem([ui.x * vi.x + ui.y * vi.y + (bb[0] * ui.x + bb[1] * ui.y) * vi, u(xb, yb) - 0.0]).A
    A = np.asarray(A.todense() if hasattr(A, "todense") else A)
    assert np.all(np.isfinite(A))
    # An advection term is not symmetric -- proof the vector coefficient actually entered the operator.
    assert not np.allclose(A, A.T, atol=1e-6)


def test_mixing_view_types_across_regions_raises():
    """Silently, `RegionMask * MatrixView + RegionMask * VectorView` returns whichever `_rewrap` ran
    last, and the assembler then contracts the wrong rank -- wrong physics, no error."""
    from jno.trace.views import MatrixView, VectorView

    d = _two_regions().domain()
    x, y, _ = d.variable("interior", split=True)
    vec = jno.np.vector
    m = MatrixView(vec(1.0 + 0.0 * x, 0.0 * y, 0.0 * x, 1.0 + 0.0 * y).expr).from_flat(2, 2)
    w = VectorView(vec(1.0 + 0.0 * x, 0.0 * y).expr)
    with pytest.raises(ValueError, match="mix view types"):
        d.by_region({"a": m, "b": w})


def test_jax_array_attachment_assembles():
    """A jax array is a legal coefficient -- the attach path must not force it back to a python float."""
    import jax.numpy as jnp

    d = _two_regions(a={"k": jnp.asarray(2.0)}, b={"k": 1.0}).domain()
    assert np.all(np.isfinite(_poisson(d, d.k)))


def test_attachment_in_3d():
    d = (
        jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).name("a").attach(k=1.0)
        + jno.Shape.box(0, 0, 0, 2, 1, 1, size=0.5).name("b").attach(k=9.0)
    ).domain()
    assert d.attached("k") == {"a": 1.0, "b": 9.0}
    assert "RegionMask(a)" in str(d.k) and "RegionMask(b)" in str(d.k)


def test_many_regions_all_contribute():
    """Five regions, five different values -- the sum must carry every one of them."""
    shapes = None
    for i in range(5):
        s = jno.Shape.rect(0, 0, 1 + i, 1, size=0.4).name(f"r{i}").attach(k=float(i))
        shapes = s if shapes is None else shapes + s
    d = shapes.domain()
    assert d.attached("k") == {f"r{i}": float(i) for i in range(5)}
    assert all(f"RegionMask(r{i})" in str(d.k) for i in range(5))
