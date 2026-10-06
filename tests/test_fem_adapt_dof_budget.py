"""``max_dofs`` is a DOF budget, and every remesh path holds it within 20 %.

The budget used to be handed to the mesher as a VERTEX count, and the steady loop compared it with the
vertex count as well. A field with several DOFs per vertex overshot it by that factor: a Taylor-Hood
march (P2 velocity + P1 pressure, about nine DOFs per vertex) asked for ``max_dofs=4000`` produced
23,793 DOFs. The name says DOFs, so ``fem.dofs`` -- every unknown of the assembled system -- is what is
counted now.

Oracles:
  * the count predicted on a new mesh is the one the assembler produces there: exact, for P1-P3 on
    triangles, quadrilaterals and tetrahedra, a vector field, a Taylor-Hood pair and a complex field;
  * every remesh of a march lands within 20 % of ``max_dofs``, on the isotropic and the anisotropic
    path, for scalar P1, vector P1, scalar P2 and a mixed vector-P2 + scalar-P1 layout. Measured before
    the fix (budget 3000, 2-D heat march): scalar P2 4.4x, vector P1 2.1x;
  * no round of a steady loop exceeds ``1.2 * max_dofs``, and the loop stops once the budget binds;
  * a budget the edge-size window cannot reach raises instead of proceeding over it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

pytest.importorskip("mmgpy", reason="mmgpy required for adaptive remeshing (fem_adapt imports it)")

from jno.utils.solver.fem_adapt import _DOF_BUDGET_TOL, _dof_counter  # noqa: E402
from jno.utils.solver.fem_native import mesh_cell_type  # noqa: E402

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# ── the count: the assembler's own DOF count on a mesh it has not seen ────────────────────────────
def _poisson(d, kind):
    """A steady problem of the given field layout on ``d`` -- built only to be counted, never solved."""
    dim = int(d.dimension)
    X = d.variable("interior", split=True)[:dim]
    B = d.variable("boundary", split=True)[:dim]
    at = dict(zip("xyz", X))
    axes = "xyz"[:dim]
    dot = lambda a, b: sum(getattr(a, c) * getattr(b, c) for c in axes)  # noqa: E731  ∇a·∇b
    if kind in ("P1", "P2", "P3"):
        u, v = d.fem_symbols(order=int(kind[1]))
        ui, vi = u.bind(**at), v.bind(**at)
        return jno.fem([dot(ui, vi) - vi, u(*B) - 0.0])
    if kind == "vecP2":
        u, v = d.fem_symbols(value_shape=(2,), order=2)
        ui, vi = u.bind(**at), v.bind(**at)
        return jno.fem([dot(ui[0], vi[0]) + dot(ui[1], vi[1]) - vi[0], u(*B) - 0.0])
    if kind == "TH":
        u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
        p, q = d.fem_symbols(names=("p", "q"), order=1)
        ui, vi, pi, qi = u.bind(**at), v.bind(**at), p.bind(**at), q.bind(**at)
        div = lambda w: w.x[0] + w.y[1]  # noqa: E731
        return jno.fem([dot(ui[0], vi[0]) + dot(ui[1], vi[1]) - pi * div(vi) - vi[0], qi * div(ui), u(*B) - 0.0, p.pin()])
    if kind == "complex":
        u, v = d.fem_symbols()
        ui, vi = u.bind(**at), v.bind(**at)
        return jno.fem([dot(ui, vi) + 1j * ui * vi - vi, u(*B) - 0.0])
    raise ValueError(kind)


def _mesh_of(d):
    dim = int(d.dimension)
    cell = mesh_cell_type(d, dim)
    return np.asarray(d.mesh.points)[:, :dim], np.asarray(d.mesh.cells_dict[cell]), cell


_TRI = (lambda: jno.shape.rect(0, 0, 1, 1, size=0.3).domain(), lambda: jno.shape.rect(0, 0, 1, 1, size=0.17).domain())
_QUAD = (
    lambda: jno.shape.rect(0, 0, 1, 1).quad().structured(n=3).domain(),
    lambda: jno.shape.rect(0, 0, 1, 1).quad().structured(n=5).domain(),
)
_TET = (
    lambda: jno.shape.box(0, 0, 0, 1, 1, 1, size=0.6).domain(),
    lambda: jno.shape.box(0, 0, 0, 1, 1, 1, size=0.4).domain(),
)


@pytest.mark.parametrize(
    "meshes, kind",
    [pytest.param(_TRI, k, id=f"tri-{k}") for k in ("P1", "P2", "P3", "vecP2", "TH", "complex")]
    + [pytest.param(_QUAD, k, id=f"quad-{k}") for k in ("P1", "P2")]
    + [pytest.param(_TET, k, id=f"tet-{k}") for k in ("P2", "P3")],
)
def test_the_count_on_a_new_mesh_is_the_assembled_one(meshes, kind):
    """Read the layout off a problem on one mesh, count it on ANOTHER, and assemble there: the two agree
    exactly. (Measured: 205 / 442 for P2 / P3 on 58 vertices, 468 for Taylor-Hood, 1301 for P3 tets.)"""
    first, second = meshes
    count = _dof_counter(_poisson(first(), kind))
    other = second()
    assert count(*_mesh_of(other)) == _poisson(other, kind).dofs


# ── a march: every remesh within 20 % of the budget ───────────────────────────────────────────────
def _march(kind):
    """A 2-D heat march with a sharp hump, so the remesher has a feature to concentrate on."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.12).domain(time=(0.0, 0.05, 6))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    at = dict(x=xi, y=yi, t=ti)
    dot = lambda a, b: a.x * b.x + a.y * b.y  # noqa: E731  ∇a·∇b
    hump = jno.fn(
        lambda x, y: jnp.sin(PI * x) * jnp.sin(PI * y) + jnp.exp(-((x - 0.3) ** 2 + (y - 0.6) ** 2) / 0.005), [x0, y0]
    )
    if kind in ("P1", "P2"):
        u, v = d.fem_symbols(order=int(kind[1]))
        ui, vi = u.bind(**at), v.bind(**at)
        return jno.fem([ui.t * vi + 0.1 * dot(ui, vi), u(xb, yb) - 0.0, u(x0, y0) - hump])
    if kind == "vecP1":
        u, v = d.fem_symbols(value_shape=(2,))
        ui, vi = u.bind(**at), v.bind(**at)
        heat = sum(ui.t[c] * vi[c] + 0.1 * dot(ui[c], vi[c]) for c in range(2))
        return jno.fem([heat, u(xb, yb) - 0.0, u(x0, y0)[0] - hump, u(x0, y0)[1] - hump])
    # the Taylor-Hood LAYOUT (vector P2 + scalar P1), as two decoupled heat problems
    u, w = d.fem_symbols(value_shape=(2,), names=("u", "w"), order=2)
    s, r = d.fem_symbols(names=("s", "r"), order=1)
    ui, wi, si, ri = u.bind(**at), w.bind(**at), s.bind(**at), r.bind(**at)
    return jno.fem(
        [
            sum(ui.t[c] * wi[c] + 0.1 * dot(ui[c], wi[c]) for c in range(2)),
            si.t * ri + 0.1 * dot(si, ri),
            u(xb, yb) - 0.0,
            s(xb, yb) - 0.0,
            u(x0, y0)[0] - hump,
            u(x0, y0)[1] - hump,
            s(x0, y0) - hump,
        ]
    )


@pytest.mark.parametrize("anisotropic", [False, True], ids=["isotropic", "anisotropic"])
@pytest.mark.parametrize("kind", ["P1", "vecP1", "P2", "P2vec+P1"])
def test_a_march_holds_the_dof_budget(kind, anisotropic):
    budget = 1500
    fem = _march(kind)
    traj = fem.solve(adapt=jno.solve.remesh(anisotropic=anisotropic, every=2, max_dofs=budget))
    remeshed = [h["n_dofs"] for h in fem.adapt_history if h.get("remeshed", True)]
    assert len(remeshed) == 2, fem.adapt_history
    # the recorded count is the size of the state actually marched on that mesh
    assert np.asarray(traj.states[-1]).size == remeshed[-1]
    lo, hi = (1.0 - _DOF_BUDGET_TOL) * budget, (1.0 + _DOF_BUDGET_TOL) * budget
    assert all(n <= hi for n in remeshed), f"max_dofs={budget}, remeshes gave {remeshed} DOFs"
    assert all(n >= lo for n in remeshed), f"max_dofs={budget} is the size to hold; remeshes gave {remeshed} DOFs"


def test_an_unreachable_budget_raises():
    """``hmax`` stops the mesh coarsening to 40 DOFs: the remesh must say so, not proceed over budget."""
    fem = _march("P1")
    with pytest.raises(RuntimeError, match=r"max_dofs=40.*could not be held under the DOF budget"):
        with pytest.warns(UserWarning, match="Raise hmax"):
            fem.solve(adapt=jno.solve.remesh(every=2, max_dofs=40, hmax=0.1))


# ── the steady loop: the budget caps the rounds and stops the loop ────────────────────────────────
@pytest.mark.parametrize("anisotropic", [False, True], ids=["isotropic", "anisotropic"])
def test_a_steady_loop_never_rounds_past_the_budget(anisotropic):
    """P2 (about four DOFs per vertex): the loop used to stop on the VERTEX count, so its last round grew
    to several times the budget. Every round now stays under 1.2 x max_dofs, and the loop ends there."""
    budget = 2500
    d = jno.shape.rect(0, 0, 1, 1, size=0.15).domain()
    u, v = d.fem_symbols(order=2)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    peak = jno.fn(lambda x, y: jnp.exp(-((x - 0.3) ** 2 + (y - 0.6) ** 2) / 0.002), [xi, yi])
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 500.0 * peak * vi, u(xb, yb) - 0.0])
    fem.solve(adapt=jno.solve.remesh(anisotropic=anisotropic, refine_factor=2.0, max_iters=8, max_dofs=budget))
    dofs = [h["n_dofs"] for h in fem.adapt_history]
    assert max(dofs) <= (1.0 + _DOF_BUDGET_TOL) * budget, f"max_dofs={budget}, rounds gave {dofs} DOFs"
    assert len(dofs) < 8, f"the budget did not stop the loop: {dofs}"
    assert dofs[-1] >= (1.0 - _DOF_BUDGET_TOL) * budget, f"the loop stopped short of the budget: {dofs}"
    assert fem.dofs == dofs[-1]  # the history counts the system the loop solved, not its vertices
