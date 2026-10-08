"""``jno.np.vector`` with a bare number next to a per-point component.

``jno.np.vector(1.0, 0.0, x)`` must mean the same field as ``jno.np.vector(1.0 + 0.0*x, 0.0*x, x)``:
the number broadcasts over the points. It once hung ``jno.fem`` forever instead: the broadcast path in
``jno.np.concat`` called ``max(...)``, which inside ``jno/jnp_ops.py`` is the trace reduction
``jno.np.max``, not the builtin. A trace node went into a reshape shape, and JAX's formatter for the
invalid shape iterated that node without end. Every FEM test here runs under a wall-clock limit so a
regression fails instead of hanging the suite.
"""

from __future__ import annotations

import contextlib
import signal

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for meshing")
pytest.importorskip("basix")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver.fem_nonnodal import nonnodal_field_at  # noqa: E402

inner, grad = jno.np.inner, jno.np.grad
vector = jno.np.vector


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@contextlib.contextmanager
def _time_limit(seconds):
    """Fail (not hang) if the block runs longer than ``seconds``. The old hang was a pure-Python loop, which
    SIGALRM interrupts."""

    def _raise(signum, frame):
        raise TimeoutError(f"exceeded {seconds} s -- the bare-number vector hang is back")

    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def test_concat_broadcasts_a_number_against_per_point_components():
    """The evaluator-level contract: a 0-d number and an (N, 1) component give (N, 2), and equal the
    explicit ``1.0 + 0.0*x`` spelling exactly."""
    x = jnp.linspace(0.0, 1.0, 5)[:, None]  # a per-point scalar, (N, 1), as the FEM kernels hand it over
    fn = jno.np.concat([1.0, x]).fn
    with _time_limit(10):
        out = fn(jnp.asarray(1.0), x)
    np.testing.assert_array_equal(out, jnp.concatenate([1.0 + 0.0 * x, x], axis=-1))
    out3 = jno.np.concat([1.0, 0.0, x]).fn(jnp.asarray(1.0), jnp.asarray(0.0), x)
    np.testing.assert_array_equal(out3, jnp.concatenate([1.0 + 0.0 * x, 0.0 * x, x], axis=-1))


def _n1e_projection(tdim, components, k):
    """L² projection onto N1E_k: (u, v) = (g, v). Returns the read-back at random points of every cell,
    those points, and the raw solution."""
    if tdim == 2:
        d = jno.domain(box(0, 0, 1, 1), mesh_size=0.45)
    else:
        d = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.9).domain()
    u, v = d.fem_symbols(value_shape=(tdim,), names=("u", "v"), space="N1E", order=k)
    c = d.variable("interior", split=True)[:tdim]
    X = dict(zip("xyz", c))
    ui, vi = u.bind(**X), v.bind(**X)
    with _time_limit(60):
        fem = jno.fem([inner(ui, vi) - inner(vector(*components(*c)), vi)])
        sol = np.asarray(fem.solve()).reshape(-1)
    dm = d._fem_nonnodal_topology["dofmap"]
    pts = np.asarray(d.mesh.points)[:, :tdim]
    cells = d._fem_nonnodal_topology["cells"]
    rp = np.random.default_rng(0).dirichlet(np.ones(tdim + 1), 4)[:, :tdim]
    val = np.asarray(nonnodal_field_at(pts, cells, dm, jnp.asarray(sol), ref_points=rp))
    J = pts[cells[:, 1:]] - pts[cells[:, :1]]  # (n_cells, tdim, tdim): edge vectors from vertex 0
    P = pts[cells[:, 0]][:, None, :] + np.einsum("qa,cad->cqd", rp, J)
    return val, P, sol


@pytest.mark.parametrize(
    "tdim, k, bare, broadcast, exact",
    [
        # (1, x) lies in N1E_2 (all linear vector fields): its projection is exact.
        (
            2,
            2,
            lambda x, y: (1.0, x),
            lambda x, y: (1.0 + 0.0 * x, x),
            lambda P: np.stack([np.ones_like(P[..., 0]), P[..., 0]], -1),
        ),
        # (-y, x, 1) = (0,0,1)×(x,y,z) + (0,0,1) lies in N1E_1: exact at the lowest order, three components.
        (
            3,
            1,
            lambda x, y, z: (-y, x, 1.0),
            lambda x, y, z: (-y, x, 1.0 + 0.0 * x),
            lambda P: np.stack([-P[..., 1], P[..., 0], np.ones_like(P[..., 0])], -1),
        ),
    ],
    ids=["2d-N1E2", "3d-N1E1"],
)
def test_bare_number_component_builds_and_solves_like_the_broadcast_spelling(tdim, k, bare, broadcast, exact):
    val_b, P, sol_b = _n1e_projection(tdim, bare, k)
    val_w, _, sol_w = _n1e_projection(tdim, broadcast, k)
    np.testing.assert_array_equal(sol_b, sol_w)  # the same linear system, bit for bit
    np.testing.assert_allclose(val_b, exact(P), atol=1e-7)  # the analytic field, to the BiCGStab tolerance


def test_bare_number_in_a_vector_dirichlet_value_lagrange_laplace():
    """Vector P1 Laplace, -Δu = 0 with u = (1, x) on the boundary: (1, x) is harmonic and linear, so P1
    reproduces it at every node. The bare-number boundary value must give exactly that."""

    def solve(g):
        d = jno.domain(box(0, 0, 1, 1), mesh_size=0.3)
        u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"))
        xi, yi, *_ = d.variable("interior", split=True)
        xb, yb, *_ = d.variable("boundary", split=True)
        gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
        with _time_limit(60):
            fem = jno.fem([inner(gu, gv, 2), u(xb, yb) - vector(*g(xb))])
            sol = np.asarray(fem.solve()).reshape(-1, 2)
        return np.asarray(d.mesh.points)[:, :2], sol

    P, sol_b = solve(lambda x: (1.0, x))
    _, sol_w = solve(lambda x: (1.0 + 0.0 * x, x))
    np.testing.assert_array_equal(sol_b, sol_w)
    np.testing.assert_allclose(sol_b, np.stack([np.ones(len(P)), P[:, 0]], -1), atol=1e-7)
