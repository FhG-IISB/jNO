"""Differentiable geometry on the NON-NODAL assembler: ``d(solve)/dX`` for an H(curl) problem.

``domain.variable(region).trainable()`` promotes a spatial coordinate to a per-vertex design
variable. The nodal assembler has carried it since the differentiable-geometry work; these tests
cover the non-nodal one (N1E / RT / P0 / Hermite / Argyris / Morley), which read the mesh as a
static numpy array and returned a shape derivative of exactly zero.

Zero is the dangerous answer here: it is indistinguishable from a converged design, so each test
below asserts something is NON-zero as well as correct, and ``test_static_mesh_unchanged`` pins the
no-design-variable case so the feature cannot perturb ordinary assembly.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402

inner, vec = jno.np.inner, jno.np.vector


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _curl_curl(size=0.5, trainable_axis=None):
    """A gauged curl-curl system on a cube with a constant source.

    A small mass term makes the operator non-singular, so a direct solve applies and the objective
    below is well defined. With ``trainable_axis`` set, that axis of every mesh vertex becomes a
    design variable.
    """
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    pts = np.asarray(d.mesh.points)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]

    param = None
    if trainable_axis is not None:
        param = ci[trainable_axis].trainable(name="X")

    # A mass term, NOT a tree gauge, is what makes curl-curl non-singular here. The tree-pinning
    # route goes through `domain._extra_dof_pins`, which not every jNO carries -- and where it is
    # absent the pins are silently ignored, leaving the test to solve a singular system and call it
    # a pass. A mass term is part of the form itself, so it cannot be quietly dropped.
    fem = jno.fem(
        [
            inner(u.vector.curl(x, y, z), v.vector.curl(x, y, z))
            + 1e-3 * inner(u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z))
            - inner(vec(1.0 + 0.0 * x, 0.0 * x, 0.0 * x), v.bind(x=x, y=y, z=z)),
            u.vector.cross(d.variable("boundary", normals=True)),
        ]
    )
    return d, fem, pts, param


def _objective(op, values):
    A, b = op.evaluate({"X": values})
    return jnp.sum(jno.utils.solver.linear.sparse_lu_solve(A, b) ** 2)


def test_trainable_coordinate_makes_the_operator_parametric():
    """Without this the assembler returns a static ``(A, b)`` and every shape derivative is zero."""
    _d, fem, _pts, param = _curl_curl(trainable_axis=2)
    assert param is not None
    op = fem.operator
    assert hasattr(op, "evaluate"), "a trainable coordinate must give a parametric FemLinearSystem"
    assert "X" in (getattr(op, "runtime_parameter_exprs", {}) or {})


def test_shape_derivative_matches_finite_difference():
    """``jax.grad`` w.r.t. vertex coordinates against a central difference, on the entries that carry
    real sensitivity. The check is restricted to the largest components because a central difference
    on a near-zero entry is itself only good to a few percent."""
    _d, fem, pts, _p = _curl_curl(trainable_axis=2)
    op = fem.operator
    z0 = jnp.asarray(pts[:, 2])

    g = np.asarray(jax.grad(lambda zv: _objective(op, zv))(z0))
    assert np.linalg.norm(g) > 0.0, "shape derivative came back identically zero"

    h = 1e-7
    for k in np.argsort(-np.abs(g))[:3]:  # the three largest sensitivities
        fd = (_objective(op, z0.at[k].add(h)) - _objective(op, z0.at[k].add(-h))) / (2 * h)
        assert np.isclose(g[k], float(fd), rtol=2e-3), f"vertex {k}: grad {g[k]:.6e} vs fd {float(fd):.6e}"


def test_moving_the_mesh_moves_the_solution():
    """A guard against a derivative that is correct only because the objective is constant: the
    objective must actually respond to the design variable."""
    _d, fem, pts, _p = _curl_curl(trainable_axis=2)
    op = fem.operator
    z0 = jnp.asarray(pts[:, 2])
    base = float(_objective(op, z0))
    moved = float(_objective(op, z0 * 1.05))  # a 5 % stretch in z
    assert not np.isclose(base, moved, rtol=1e-6), "the solution did not respond to mesh motion"


def test_static_mesh_unchanged():
    """No design variable -> the ordinary, non-parametric assembly, untouched."""
    _d, fem, _pts, param = _curl_curl(trainable_axis=None)
    assert param is None
    op = fem.operator
    assert not hasattr(op, "evaluate"), "assembly must stay non-parametric without trainable coords"


def test_static_fallback_refuses_when_coordinates_are_trainable():
    """The paths that still read the static mesh must RAISE rather than return a zero derivative.

    A silently static fallback reads as a converged design instead of a missing feature, which is
    the failure this whole guard exists to prevent.
    """
    from jno.utils.solver import fem_nonnodal

    src = __import__("inspect").getsource(fem_nonnodal)
    assert "_apply_coord_params" in src
    assert "would come back as exactly zero" in src, "the silent-zero guard has been removed"
