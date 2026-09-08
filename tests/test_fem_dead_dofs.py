"""A field whose every term is region-restricted has DOFs that sit in no equation at all.

Region-restricted terms make coupled multiphysics writable — Navier-Stokes on ``fluid``, elasticity on
``solid``, one term list. But ``fem_symbols`` has no region argument, so every field still carries DOFs
over the **whole** mesh. Those outside the region its terms integrate on appear in no row, and the
system is structurally singular.

What that looked like before this check: the matrix-free default raised a generic "problem may be
singular/ill-posed" naming no cause, and a direct ``lu`` slot could return garbage instead. The fix a
user had to know to write is a pin per field per unused region — boilerplate nobody guesses.

**Reach is measured with the assembler's own resolution** (``_cell_region_mask``: a cell is in a region
iff its centroid is), intersected with each field's connectivity, so the check cannot disagree with what
was assembled. ``domain.tag_node_mask`` is deliberately not used for volume regions — measured on the
two-block domain below it returns a mutually exclusive and incomplete partition (40 + 32 of 78 nodes,
nothing shared at the interface, 6 interior nodes in neither), which would report well-posed DOFs as
dead. That measurement is asserted directly in ``test_the_tag_mask_is_not_a_usable_reach_oracle`` so the
reason this code does not take the obvious shortcut stays visible.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _two_region(size=0.5):
    return (
        jno.Shape.box(0, 0, 0, 1, 1, 1).name("lower").sized(size)
        + jno.Shape.box(0, 0, 1, 1, 1, 2).name("upper").sized(size)
    ).domain()


def _pieces(d):
    u, phi = d.fem_symbols()
    lo, up = d.variable("lower", split=True), d.variable("upper", split=True)
    ob = d.variable("ob", where=lambda x, y, z: (z < 1e-9) | (z > 2 - 1e-9), split=True)
    lap = lambda w, z, r: (  # noqa: E731
        w.bind(x=r[0], y=r[1], z=r[2]).x * z.bind(x=r[0], y=r[1], z=r[2]).x
        + w.bind(x=r[0], y=r[1], z=r[2]).y * z.bind(x=r[0], y=r[1], z=r[2]).y
        + w.bind(x=r[0], y=r[1], z=r[2]).z * z.bind(x=r[0], y=r[1], z=r[2]).z
    )
    return u, phi, lo, up, ob, lap


def test_a_field_restricted_to_one_region_is_refused_by_name():
    """The defect. ``u`` is governed only on `lower`, so its DOFs on `upper` have no equation.

    Refused at SOLVE, not at build. A form covering one region is a legitimate intermediate --
    ``jno.core([femL, fdmR, ...])`` and ``jno.dd.couple`` are built from exactly those, one per
    subdomain, with each partner governing what the other omits. Raising in ``jno.fem`` made those
    unbuildable and broke five domain-decomposition tests; structural singularity is a property of a
    system somebody solves alone, so that is where it is reported.
    """
    d = _two_region()
    u, phi, lo, _up, ob, lap = _pieces(d)
    fem = jno.fem([lap(u, phi, lo), u(ob[0], ob[1], ob[2]) - 0.0])  # builds: it may be a subdomain
    with pytest.raises(ValueError, match="appear in no term") as e:
        fem.solve()
    msg = str(e.value)
    assert "'u'" in msg, msg  # names the FIELD, not just a block index
    assert "'lower'" in msg, msg  # and the region its terms are restricted to
    assert "rigid-body" in msg  # and states the limit: this is structural, not a rank test
    assert "couple" in msg, msg  # and points a subdomain author at the coupling entry points


def test_a_one_region_form_still_builds_for_use_as_a_subdomain():
    """The reason the refusal moved: `jno.dd.couple` needs to CONSTRUCT such a form.

    Reduced to the essential shape, so it fails for the right reason if the build-time raise ever
    returns -- the domain-decomposition suite would catch it too, but only through a much longer path.
    """
    d = _two_region()
    u, phi, lo, _up, ob, lap = _pieces(d)
    fem = jno.fem([lap(u, phi, lo), u(ob[0], ob[1], ob[2]) - 0.0])
    assert fem is not None and int(fem.dofs) > 0


def test_a_term_over_the_missing_region_makes_it_build_and_solve():
    """The fix the message suggests has to actually work, or the message is wrong.

    A cheap ``eps * u * phi`` over the otherwise-ungoverned region reaches every one of its DOFs,
    because reach is measured over that region's CELLS."""
    d = _two_region()
    u, phi, lo, up, ob, lap = _pieces(d)
    reg = lambda r: u.bind(x=r[0], y=r[1], z=r[2]) * phi.bind(x=r[0], y=r[1], z=r[2])  # noqa: E731
    fem = jno.fem([lap(u, phi, lo), 1e-8 * reg(up), u(ob[0], ob[1], ob[2]) - 0.0])
    assert np.isfinite(np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))).all()


def test_a_volume_region_pin_covers_the_whole_region():
    """A Dirichlet pin on a volume sub-region reaches every ungoverned DOF of that region.

    It did not always. The pin fell through to ``domain.tag_node_mask``, which for a volume region is
    a proximity test against the region's *sampled* points rather than a containment test, and reached
    **32 of 33** nodes -- one interior node of `upper` that sampling missed stayed unconstrained, with
    nothing raised. ``_region_node_ids_from_cells`` resolves it from mesh topology instead, on the same
    ``_cell_region_mask`` the assembler integrates over, so the Dirichlet node set cannot disagree with
    the cells the terms were applied to.

    This test previously asserted the DEFECT (``n_dead == 33 and n_pin == 32``) and read the pin
    through ``tag_node_mask("upper", ...)``. That returns ``None`` for anything that is not a
    ``domain.tag`` -- and "upper" is a ``Shape.name`` region -- so it raised ``TypeError`` from the
    commit that introduced it and never once ran its assertions. The behaviour it was guarding was
    then fixed, leaving it wrong twice over.
    """
    from jno.utils.solver.fem_native import _region_node_ids_from_cells
    from jno.utils.solver.fem_utils import _cell_region_mask

    d = _two_region()
    u, phi, lo, up, ob, lap = _pieces(d)
    jno.fem([lap(u, phi, d.variable("interior", split=True)), u(ob[0], ob[1], ob[2]) - 0.0])
    pts = np.asarray(d._fem_native_dof_points_all[0])
    cells = np.asarray(d._fem_native_assembly_cells_all[0])
    reached = np.zeros(len(pts), dtype=bool)
    reached[np.unique(cells[np.asarray(_cell_region_mask(d, "lower")).reshape(-1) > 0])] = True
    dead = set(np.flatnonzero(~reached).tolist())
    assert dead, "the fixture must leave part of `upper` ungoverned, or this test asserts nothing"

    ids = set(np.asarray(_region_node_ids_from_cells(d, "upper", cells)).tolist())
    assert dead <= ids, f"the pin misses {len(dead - ids)} of the {len(dead)} ungoverned DOFs"

    assert d.tag_node_mask("upper", pts) is None, (
        "tag_node_mask now resolves a Shape-region name; this test's premise -- that a volume-region "
        "pin must NOT be resolved through it -- needs rechecking"
    )


def test_terms_covering_every_region_do_not_raise():
    """The no-false-positive case: both regions governed, so no DOF is dead. This is the configuration
    the coupled-multiphysics examples are written in, and it must stay silent."""
    d = _two_region()
    u, phi, lo, up, ob, lap = _pieces(d)
    fem = jno.fem([1.0 * lap(u, phi, lo), 10.0 * lap(u, phi, up), u(ob[0], ob[1], ob[2]) - (ob[2] > 1.0) * 1.0])
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))
    pts = np.asarray(d.built_mesh.points)
    mid = sol[np.abs(pts[:, 2] - 1.0) < 1e-9]
    # two conductivities in series, k=1 then k=10: u(interface) = R1/(R1+R2) = 1/(1+1/10)
    assert abs(float(mid.mean()) - 1.0 / (1.0 + 1.0 / 10.0)) < 1e-9


def test_a_whole_domain_term_is_never_flagged():
    """The scan short-circuits when every field has a whole-domain term — which is every ordinary
    single-region problem, so it cannot regress them."""
    d = _two_region()
    u, phi, _lo, _up, ob, _lap = _pieces(d)
    co = d.variable("interior", split=True)
    ui, vi = u.bind(x=co[0], y=co[1], z=co[2]), phi.bind(x=co[0], y=co[1], z=co[2])
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z, u(ob[0], ob[1], ob[2]) - 0.0])
    assert np.isfinite(np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))).all()


def test_the_tag_mask_is_not_a_usable_reach_oracle():
    """Why the check resolves regions by CELL centroid rather than by node tag.

    ``tag_node_mask`` on a volume sub-region is exclusive and incomplete — it is a sampling-based
    proximity test, not a point-in-region test. Building the reach on it would mark well-posed interior
    DOFs as dead. Pinned here because that shortcut is the obvious one to reach for later.
    """
    d = _two_region()
    u, phi, _lo, _up, ob, lap = _pieces(d)
    co = d.variable("interior", split=True)
    jno.fem([lap(u, phi, co), u(ob[0], ob[1], ob[2]) - 0.0])  # publishes the assembly layout below
    pts = np.asarray(d.built_mesh.points)
    lo = np.asarray(d.tag_node_mask("lower", pts), dtype=bool)
    up = np.asarray(d.tag_node_mask("upper", pts), dtype=bool)
    assert not (lo & up).any(), "the two volume masks are mutually exclusive -- no shared interface"
    assert (~lo & ~up).any(), "and incomplete -- some interior nodes belong to neither"

    # the cell-centroid resolution the check actually uses does cover every node
    from jno.utils.solver.fem_utils import _cell_region_mask

    cells = np.asarray(d._fem_native_assembly_cells_all[0])
    covered = np.zeros(len(np.asarray(d._fem_native_dof_points_all[0])), dtype=bool)
    for r in ("lower", "upper"):
        covered[np.unique(cells[np.asarray(_cell_region_mask(d, r)).reshape(-1) > 0])] = True
    assert covered.all(), "every DOF must be reached once both regions are integrated over"
