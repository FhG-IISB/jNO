"""A ``jno.Shape`` domain draws collocation points from the geometry, and does not mesh to do it.

The contract, in one line: **the PINN source does not change.** The same `.domain()`, the same
`tag(name, where)`, the same `variable(name, sample=(n, None))` — what changes is that no mesher
runs, the points are continuous rather than a fixed node set, and `n` means what it says.

A mesh is still built the moment something genuinely needs one (`jno.fem`, `.integrate()`, a facet
predicate); that is announced in one line rather than happening invisibly.
"""

import math

import numpy as np
import pytest

import jno

pytestmark = pytest.mark.filterwarnings("ignore")


def _meshless(d):
    """Whether a mesh exists, asked *without* triggering the lazy build that reading `.mesh` would."""
    return d.__dict__.get("_mesh") is None


# --------------------------------------------------------------------------- no mesh is built


@pytest.mark.parametrize(
    "make, dim",
    [
        (lambda: jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve().domain(), 1),
        (lambda: jno.Shape.rect(0.0, 0.0, 1.0, 1.0).domain(), 2),
        (lambda: jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).domain(), 3),
        (lambda: (jno.Shape.rect(0.0, 0.0, 1.0, 1.0) - jno.Shape.disk(0.5, 0.5, 0.25)).domain(), 2),
        (lambda: (jno.Shape.box(0, 0, 0, 1, 1, 1) - jno.Shape.sphere(0.5, 0.5, 0.5, 0.3)).domain(), 3),
        (lambda: jno.Shape.disk(0.0, 0.0, 1.0).extrude(2.0).domain(), 3),
        (lambda: jno.Shape.rect(1, 0, 2, 1).revolve((0, 0, 0), (0, 1, 0), 2 * math.pi).domain(), 3),
        (lambda: jno.Shape.rect(1, 0, 2, 1).revolve((0, 0, 0), (0, 1, 0), math.pi).domain(), 3),
    ],
    ids=["1d-line", "2d-rect", "3d-box", "2d-csg", "3d-csg", "extrude", "revolve", "revolve-half"],
)
def test_a_shape_domain_starts_without_a_mesh(make, dim):
    d = make()
    assert _meshless(d), "constructing a domain must not run a mesher"
    assert d.dimension == dim


def test_sampling_and_tagging_never_trigger_a_mesh():
    d = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).domain()
    d.tag("hot", lambda x, y, z: (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 < 0.04)
    d.variable("interior", sample=(3000, None), split=True)
    d.variable("hot", sample=(500, None), split=True)
    d.variable("left", sample=(200, None), normals=True, split=True)
    assert _meshless(d), "tagging and sampling are geometry operations; neither needs a mesh"


# --------------------------------------------------------------------------- the tag vocabulary


def test_auto_named_boundaries_exist_before_any_mesh():
    """The names come from the same classifier the mesher uses — applied to analytic boundary
    points instead of gmsh entities — so a CSG cut introduces its own name too."""
    assert set(jno.Shape.rect(0, 0, 2, 1).domain()._geometry_tags) >= {
        "interior",
        "boundary",
        "left",
        "right",
        "top",
        "bottom",
    }
    holed = (jno.Shape.rect(0, 0, 1, 1) - jno.Shape.disk(0.5, 0.5, 0.25)).domain()
    assert "arc" in holed._geometry_tags, "the cut introduced a circular edge; it should be nameable"
    void = (jno.Shape.box(0, 0, 0, 1, 1, 1) - jno.Shape.sphere(0.5, 0.5, 0.5, 0.3)).domain()
    assert "surface" in void._geometry_tags, "the spherical void's face should be nameable"


def test_an_auto_named_boundary_samples_only_its_own_face():
    d = jno.Shape.rect(0.0, 0.0, 2.0, 1.0).domain()
    d.variable("left", sample=(400, None), normals=True, split=True)
    pts = np.asarray(d.context["left"]).reshape(-1, 2)
    nrm = np.asarray(d.context["n_left"]).reshape(-1, 2)
    assert len(pts) == 400
    assert np.allclose(pts[:, 0], 0.0)
    assert pts[:, 1].min() < 0.05 and pts[:, 1].max() > 0.95  # spread along the whole edge
    assert np.allclose(nrm, [-1.0, 0.0])


def test_a_boundary_predicate_is_recognised_and_sampled_on_the_boundary():
    """The case that decides whether "tagging works the same": `x < tol` names a face. It is
    classified by measuring which draw accepts it, not by inspecting the lambda."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).domain()
    d.tag("wall", lambda x, y: x < 1e-9)
    d.tag("blob", lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.04)
    assert d._geometry_tags["wall"][0] == "boundary"
    assert d._geometry_tags["blob"][0] == "interior"

    d.variable("wall", sample=(300, None), normals=True, split=True)
    w = np.asarray(d.context["wall"]).reshape(-1, 2)
    assert np.allclose(w[:, 0], 0.0) and len(w) == 300
    assert np.allclose(np.asarray(d.context["n_wall"]).reshape(-1, 2), [-1.0, 0.0])

    d.variable("blob", sample=(500, None), split=True)
    b = np.asarray(d.context["blob"]).reshape(-1, 2)
    assert (((b[:, 0] - 0.5) ** 2 + (b[:, 1] - 0.5) ** 2) < 0.04).all()


def test_a_boundary_tag_becomes_a_boundary_region_without_a_mesh():
    """`jno.fem` reads `_boundary_regions` to tell an essential condition from a domain residual;
    a mesh-free boundary tag that registered no region would be read as the latter."""
    d = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).domain()
    d.tag("inlet", lambda x, y, z: x < 1e-9)
    assert "inlet" in d._boundary_regions
    assert _meshless(d)


# --------------------------------------------------------------------------- continuous points


def test_the_count_is_honoured_and_uncapped():
    """On a mesh the count was silently clipped to the node count, with a warning. There is no node
    set here, so `n` means `n` — including counts far beyond any mesh of this shape."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).domain()
    d.variable("interior", sample=(50_000, None), split=True)
    assert np.asarray(d.context["interior"]).shape == (1, 1, 50_000, 2)


def test_successive_draws_are_different_points():
    d = jno.Shape.disk(0.0, 0.0, 1.0).domain()
    d.variable("interior", sample=(400, None), split=True)
    first = np.asarray(d.context["interior"]).copy()
    d.sample({"interior": (400, None)})
    second = np.asarray(d.context[[k for k in d.context if k.startswith("interior")][-1]])
    assert not np.allclose(np.sort(first, axis=2), np.sort(second, axis=2))


def test_points_land_inside_the_shape_they_name():
    holed = (jno.Shape.rect(0, 0, 1, 1) - jno.Shape.disk(0.5, 0.5, 0.25)).domain()
    holed.variable("interior", sample=(5000, None), split=True)
    pts = np.asarray(holed.context["interior"]).reshape(-1, 2)
    assert ((pts[:, 0] - 0.5) ** 2 + (pts[:, 1] - 0.5) ** 2 >= 0.25**2 - 1e-12).all()
    assert (pts >= -1e-12).all() and (pts <= 1 + 1e-12).all()


def test_no_count_without_a_declared_size_refuses_rather_than_guessing():
    """`variable(tag)` with no count could mean collocation points or the node set, and which is
    right depends on what the caller does *afterwards*. With no `size=` there is no declared
    resolution to fall back on, so it refuses and names both ways out — guessing would be silent,
    since finite differences over one collocation point return a number, just the wrong one."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).domain()  # no size=
    with pytest.raises(ValueError, match="no mesh size was declared"):
        d.variable("interior", split=True)

    d.variable("interior", sample=(500, None), split=True)  # the explicit way out
    assert np.asarray(d.context["interior"]).shape == (1, 1, 500, 2)
    assert _meshless(d)


def test_a_declared_size_makes_the_no_count_form_mean_that_mesh_s_nodes():
    """`size=` IS the declaration of a resolution: asking for a mesh of that density and then for
    'the interior' unambiguously means its nodes. That is what a convergence study over `size`
    rests on, so it keeps working — and an explicit count still opts out to mesh-free."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain()
    assert _meshless(d), "declaring a size still does not mesh until the nodes are asked for"
    d.variable("interior", split=True)
    assert not _meshless(d)
    assert np.asarray(d.context["interior"]).shape[2] == len(d._mesh_pool["interior"])

    sized = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain()
    sized.variable("interior", sample=(9_000, None), split=True)  # a count still wins
    assert np.asarray(sized.context["interior"]).shape == (1, 1, 9_000, 2)
    assert _meshless(sized)


def test_resampling_candidates_are_generated_not_reshuffled():
    """The no-count default IS a resampling strategy, so its candidates must come from the geometry
    every time. Drawing them from the reference pool would make "redrawn every step" mean "redrawn
    from the same frozen cloud every step" — the exact thing this feature removes."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).domain()
    d.variable("interior", sample=(50, None), split=True)
    a, _ = d.draw_candidates("interior")
    b, _ = d.draw_candidates("interior")
    assert len(a) >= 1000 and len(b) >= 1000
    assert not np.allclose(np.sort(a, axis=0), np.sort(b, axis=0))
    # and a boundary tag's candidates carry normals
    pts, nrm = d.draw_candidates("left")
    assert nrm is not None and np.allclose(nrm, [-1.0, 0.0]) and np.allclose(pts[:, 0], 0.0)


# --------------------------------------------------------------------------- the deferred mesh


def test_reading_the_mesh_builds_it_once():
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain()
    assert _meshless(d)
    mesh = d.mesh
    assert mesh is not None and len(mesh.points) > 10
    assert d.mesh is mesh, "the deferred build must happen once, not per read"
    assert d.mesh_connectivity is not None, "the post-mesh pipeline must have run"


def test_a_tag_declared_while_mesh_free_survives_the_deferred_build():
    """The tag carried only a predicate while there was no mesh. Once there is one, it must gain
    the mesh-derived half too — otherwise an essential condition on it silently becomes a
    whole-domain residual."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain()
    d.tag("wall", lambda x, y: x < 1e-9)
    _ = d.mesh  # trigger
    assert "wall" in d._tag_predicates
    assert "wall" in d._mesh_pool and len(d._mesh_pool["wall"]) > 0
    on_wall = np.asarray(d._mesh_pool["wall"])
    assert np.allclose(on_wall[:, 0], 0.0), "the replayed pool must be the mesh nodes on that face"


def test_a_facet_predicate_asks_for_the_mesh_it_needs():
    """A facet predicate reads facet centroids and normals — a statement about the discretisation,
    not the geometry. It should build the mesh rather than refuse."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    assert _meshless(d)
    d.tag("top", lambda x, n, names: x[:, 1] > 1.0 - 1e-6)
    assert not _meshless(d)
    assert "top" in d._boundary_regions


# --------------------------------------------------------------------------- it actually trains


def test_a_mesh_free_pinn_solves_poisson_to_the_exact_solution():
    """End to end on the geometry alone: -Lap u = 2 pi^2 sin(pi x) sin(pi y), hard Dirichlet."""
    foundax = pytest.importorskip("foundax")
    optax = pytest.importorskip("optax")
    import jax

    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).domain()
    x, y, _t = d.variable("interior", sample=(2000, None), split=True)

    net = jno.nn(foundax.mlp(in_features=2, hidden_dims=48, num_layers=4, key=jax.random.PRNGKey(0)))
    net.optimizer(optax.adam(optax.exponential_decay(3e-3, 1000, 0.6, end_value=1e-6)))

    pi = jno.np.pi
    u = (x * (1 - x) * y * (1 - y) * net(x, y)).scalar.bind(x=x, y=y)
    f = 2 * pi**2 * jno.np.sin(pi * x) * jno.np.sin(pi * y)

    crux = jno.core([(u.xx + u.yy + f).mse])
    crux.solve(4000)

    _u, _x, _y = crux.eval([u, x, y])
    pred = np.asarray(_u).reshape(-1)
    exact = np.sin(np.pi * np.asarray(_x).reshape(-1)) * np.sin(np.pi * np.asarray(_y).reshape(-1))
    rel = np.linalg.norm(pred - exact) / np.linalg.norm(exact)
    assert rel < 1e-2, f"mesh-free PINN relative L2 {rel:.3e}"
    assert _meshless(d), "training must not have built a mesh"


def test_a_time_dependent_shape_domain_carries_the_time_axis():
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).domain(time=(0.0, 1.0, 11))
    d.variable("interior", sample=(128, None), split=True)
    assert np.asarray(d.context["interior"]).shape == (1, 11, 128, 2)
    assert _meshless(d)


@pytest.mark.parametrize(
    "make, why",
    [
        (
            lambda: jno.Shape.rect(0, 0, 1, 1, size=0.3).name("core").domain(),
            "a named region's tags are the mesher's conforming sub-bodies",
        ),
        (
            lambda: jno.Shape.rect(0, 0, 1, 1, size=0.3).structured().domain(),
            "a structured plan is a lattice, not a Shape, by the time it is built",
        ),
    ],
    ids=["named-region", "structured"],
)
def test_a_plan_that_cannot_be_served_still_meshes(make, why):
    """Each of these keeps the eager path *by name*, rather than being half-served mesh-free.

    A plan with no closed form is no longer on this list: ``sweep``/``fillet`` are served by a
    boundary tessellation instead (see ``test_shape_tessellation.py``). What remains is the work no
    amount of point sampling reconstructs -- conforming sub-bodies, and lattice structure."""
    assert not _meshless(make()), why


def test_batching_draws_independently_per_row():
    d = 4 * jno.Shape.rect(0.0, 0.0, 1.0, 1.0).domain()
    d.variable("interior", sample=(200, None), split=True)
    arr = np.asarray(d.context["interior"])
    assert arr.shape == (4, 1, 200, 2)
    assert not np.allclose(arr[0], arr[1]), "each batch row is its own Monte-Carlo draw"


def test_normals_are_refused_on_an_interior_tag():
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).domain()
    with pytest.raises(ValueError, match="boundary"):
        d.variable("interior", sample=(10, None), normals=True, split=True)


def test_a_curved_boundary_samples_exactly_on_the_curve():
    """A tessellation would put these on chords; the analytic sampler puts them on the circle."""
    d = jno.Shape.disk(1.0, 2.0, 0.5).domain()
    d.variable("arc", sample=(500, None), normals=True, split=True)
    pts = np.asarray(d.context["arc"]).reshape(-1, 2)
    r = np.linalg.norm(pts - np.array([1.0, 2.0]), axis=1)
    assert np.allclose(r, 0.5, atol=1e-12)
    assert math.isclose(float(r.std()), 0.0, abs_tol=1e-12)


def test_an_attached_mesh_is_not_overwritten_by_the_pending_plan():
    """Attaching a mesh retires the deferred plan, instead of the plan later building over it.

    The failure this pins is silent, which is why it is worth a test of its own: ``_apply_mesh``
    wrote ``_mesh`` but left ``_lazy_plan`` in place, so the *next* read of ``.mesh`` saw a pending
    plan, ran the mesher, and handed back the generated mesh -- the caller's mesh gone, with no
    error anywhere. Anything computed afterwards was computed on a different mesh than the one that
    was supplied.
    """
    meshio = pytest.importorskip("meshio")

    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    cells = np.array([[0, 1, 2]])
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain()
    assert _meshless(d), "a Shape domain starts mesh-free -- otherwise this proves nothing"

    d._apply_mesh(
        meshio.Mesh(
            np.c_[pts, np.zeros(len(pts))],
            [("triangle", cells)],
            cell_sets={"interior": [np.arange(1)], "boundary": [np.array([], dtype=np.int64)]},
        )
    )
    # Reading `.mesh` is exactly what used to trigger the overwrite.
    assert len(d.mesh.points) == 3, f"the attached mesh was replaced: {len(d.mesh.points)} points"
    np.testing.assert_allclose(np.asarray(d.mesh.points)[:, :2], pts)
