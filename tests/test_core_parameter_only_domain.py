"""A parameter-only fit needs no domain of its own.

An inverse problem's loss — ``(fem.solve() - u_obs).mse`` — reads no collocation coordinate: the
Variables live inside the solve, consumed by the assembler. The domain core carries is then only a
loop driver, which is why these fits used to be written with a hand-rolled placeholder
``jno.domain.from_array({"_": np.zeros((1, 1))})``. They must work without one, including when the
loss spans *several* solves on *different* meshes — the case that decides whether a coupled
process -> device inverse problem can be written at all.
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jno  # noqa: E402

TOL = 0.05


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _alpha(name="alpha", start=2.0, lr=5e-2):
    a = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name=name)
    a.initialize(jax.nn.initializers.constant(start))
    a.dtype(jnp.float64)
    a.optimizer(optax.adam(lr))
    return a


def _poisson(alpha, mesh_size=0.25):
    """``-alpha * lap u = f`` on the unit square; exact ``u = x(1-x)y(1-y)`` at ``alpha = 1``."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    return jno.fem([alpha * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)


def _observe(fem, at=1.0):
    a, b = fem.operator.evaluate({"alpha": at})
    return jnp.linalg.solve(a.todense(), jnp.asarray(b).reshape(-1))


def _recovered(crux, alpha):
    return float(np.asarray(crux.eval([alpha])[0]).reshape(-1)[0])


def test_single_solve_inverse_needs_no_domain():
    """One solve: the mesh is discovered from the graph, so no placeholder domain is needed."""
    alpha = _alpha()
    fem = _poisson(alpha)
    crux = jno.core([(fem.solve() - _observe(fem)).mse])
    assert crux.domain is not None
    crux.solve(120)
    rec = _recovered(crux, alpha)
    assert abs(rec - 2.0) > 0.5, "parameter did not move -- no gradient reached it"
    assert abs(rec - 1.0) < TOL, f"recovered alpha={rec:.4f}"


def test_two_solves_on_two_meshes_is_not_an_ambiguous_domain():
    """The regression: a loss spanning two solves on two meshes collocates on neither, so it is a
    parameter-only fit -- not the 'constraints reference 2 distinct domains' conflict it used to
    raise. This is the shape of a coupled process -> device inverse problem."""
    alpha = _alpha()
    coarse, fine = _poisson(alpha, mesh_size=0.3), _poisson(alpha, mesh_size=0.2)
    loss = (coarse.solve() - _observe(coarse)).mse + (fine.solve() - _observe(fine)).mse

    crux = jno.core([loss])  # must not raise
    crux.solve(120)
    rec = _recovered(crux, alpha)
    assert abs(rec - 2.0) > 0.5, "parameter did not move -- no gradient reached it"
    assert abs(rec - 1.0) < TOL, f"recovered alpha={rec:.4f} from two coupled solves"


def test_two_solves_match_the_placeholder_domain_they_replace():
    """The fix must be equivalent to the placeholder it retires, not merely non-raising."""
    truth = {}
    for tag, dom in (("inferred", None), ("placeholder", jno.domain.from_array({"_": np.zeros((1, 1))}))):
        alpha = _alpha()
        coarse, fine = _poisson(alpha, mesh_size=0.3), _poisson(alpha, mesh_size=0.2)
        loss = (coarse.solve() - _observe(coarse)).mse + (fine.solve() - _observe(fine)).mse
        crux = jno.core([loss]) if dom is None else jno.core([loss], domain=dom)
        crux.solve(120)
        truth[tag] = _recovered(crux, alpha)
    assert abs(truth["inferred"] - truth["placeholder"]) < 1e-6, truth


def test_collocated_variables_on_two_domains_still_raise():
    """The guard that matters is kept: two domains core would actually *sample* is a real conflict."""
    d1 = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.4)
    d2 = jno.domain(box(0.0, 0.0, 2.0, 2.0), mesh_size=0.4)
    x1, y1, _ = d1.variable("interior", split=True)
    x2, y2, _ = d2.variable("interior", split=True)
    with pytest.raises(ValueError, match="distinct"):
        jno.core([(x1 * y1).mse, (x2 * y2).mse])


def test_one_collocated_domain_beside_a_solve_mesh_resolves_to_the_collocated_one():
    """A PINN term on its own domain, plus a solve on another mesh: only one domain is sampled, so
    it is the one core must drive its loop with -- previously this raised as 'ambiguous'."""
    alpha = _alpha()
    fem = _poisson(alpha, mesh_size=0.3)
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.4)
    xi, yi, _ = d.variable("interior", split=True)

    crux = jno.core([(fem.solve() - _observe(fem)).mse, (xi * yi * 0.0).mse])  # must not raise
    assert crux.domain is d, "the collocated domain must win over the solve's mesh"
