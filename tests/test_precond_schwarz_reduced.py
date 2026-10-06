"""Pattern-built preconditioners (``schwarz``, ``fsai``) on a REDUCED system -- a periodic tie or slip.

A periodic tie, the exact slip elimination ``n·u = 0`` and hanging-node constraints make every solve run on
``P^T A P``. ``schwarz`` and ``fsai`` build their pattern eagerly from a representative operator, and that
used to be the FULL one: every such solve failed ("incompatible shapes (162, 1), (144, 1)" for Schwarz, an
IndexError for FSAI). ``schwarz(nullspace="rigid")`` also built its rigid-body modes on the full space.

Oracles: a sparse-direct solve of the same system (a preconditioner changes how fast a Krylov method
converges, never what it converges to), and, for the modes themselves, the defining property -- a rigid
translation is in the kernel of the unconstrained (tie-only) elasticity operator, and prolongs back to the
translation on EVERY node, the eliminated face included.

METIS (``pymetis``, the ``[metis]`` extra) partitions for ``parts > 1``. Where it is not installed the
multi-part cases swap in a plain index-strip partition: the partition is not what is under test, and
any partition gives a valid preconditioner.
"""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

EPS = 1e-9
L, H = 2.0, 1.0


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.fixture
def _partition(monkeypatch):
    """METIS if installed, else contiguous index strips (a valid, if poor, partition)."""
    if importlib.util.find_spec("pymetis") is None:
        import jno.utils.solver.schwarz as sw

        monkeypatch.setattr(sw, "_partition", lambda G, parts: (np.arange(G.shape[0]) * parts) // G.shape[0])


def _strip(*, clamp=True, slip=False, n=8):
    """Linear elasticity on ``[0, L] x [0, H]``, periodic in x (the left face eliminated onto the right).

    ``clamp``: the bottom is held (u = 0), so the system is non-singular. ``slip``: instead of the tie, the
    top and bottom carry ``n·u = 0`` (exactly eliminated) and the left face is clamped."""
    d = jno.shape.rect(0.0, 0.0, L, H).structured(n=n).domain()
    d.tag("left", lambda x, y: x < EPS)
    d.tag("right", lambda x, y: x > L - EPS)
    d.tag("bot", lambda x, y: (y < EPS) & (x > EPS))
    d.tag("slipwall", lambda x, y: ((y < EPS) | (y > H - EPS)) & (x > EPS))
    x, y = d.variable("interior", split=True)[:2]
    u, v = d.fem_symbols(value_shape=(2,))
    ε = lambda w: jno.np.symgrad(w, [x, y])  # noqa: E731
    ddot = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    tr = jno.np.trace
    vb = v.bind(x=x, y=y)
    f = (-0.3 * vb[0] + 0.1 * vb[1] * jno.np.sin(np.pi * x)) if clamp else 0.0 * vb[0]
    terms = [2.0 * ddot(ε(u), ε(v)) + tr(ε(u)) * tr(ε(v)) + f]
    if slip:
        xl, yl, _ = d.variable("left", split=True)
        c = d.variable("slipwall", normals=True, split=True)
        us = u(c[0], c[1])
        terms += [u(xl, yl)[0] - 0.0, u(xl, yl)[1] - 0.0, c[-2] * us[0] + c[-1] * us[1] - 0.0]
    else:
        if clamp:
            xb, yb, _ = d.variable("bot", split=True)
            terms += [u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0]
        xl, yl, _ = d.variable("left", split=True)
        xr, yr, _ = d.variable("right", split=True)
        terms += [u(xl, yl) - u(xr, yr)]
    return jno.fem(terms)


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


def _reduced_size(fem):
    from jno.utils.solver.fem_utils import _periodic_blocks

    return int(_periodic_blocks(fem._periodic)[2][-1])


def test_the_rigid_translations_restrict_exactly_and_stay_in_the_kernel():
    """On the tie-only (singular) operator: the reduced translations are annihilated by ``P^T A P`` and
    prolong back to the translation on every node, the eliminated left face included."""
    from jno.precond import _near_null_space, _on_solved_space
    from jno.utils.solver.fem_utils import prolong_periodic

    fem = _strip(clamp=False)
    A_red = _on_solved_space(fem, fem._op[0])
    n_red, n_full = int(A_red.shape[0]), int(fem.blocks[-1].stop)
    pts = np.asarray(fem.field_points[0])
    assert n_full - n_red == 2 * int(np.sum(pts[:, 0] < EPS))  # the whole left face went, counted off the mesh
    Z = _near_null_space(fem, n_red)
    assert Z.shape == (n_red, 3)  # two translations and the in-plane rotation
    scale = float(np.abs(np.asarray(A_red.todense())).max())
    for c in range(2):
        assert float(np.abs(np.asarray(A_red @ jnp.asarray(Z[:, c]))).max()) < 1e-12 * scale * np.sqrt(n_red)
        U = np.asarray(prolong_periodic(fem._periodic, Z[:, c])).reshape(-1, 2)
        want = np.zeros_like(U)
        want[:, c] = 1.0
        np.testing.assert_array_equal(U, want)
    # the rotation cannot be periodic in x: the reduced space does not contain it, and the operator must not
    # annihilate what the gather made of it (a sanity check that the column is a real, distinct mode)
    assert float(np.abs(np.asarray(A_red @ jnp.asarray(Z[:, 2]))).max()) > 1e-6 * scale


@pytest.mark.parametrize("parts", [1, 4])
@pytest.mark.parametrize("nullspace", [None, "rigid"])
def test_schwarz_on_a_periodic_elasticity_gives_the_direct_answer(_partition, parts, nullspace):
    fem = _strip()
    ref = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))
    spec = jno.precond.schwarz(parts=parts, nullspace=nullspace)
    got = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-12, maxiter=2000), precond=spec))
    assert _rel(got, ref) < 1e-9
    assert spec._pattern.n == _reduced_size(fem) < fem.dofs  # the partition is of the system actually solved
    if nullspace == "rigid":
        assert spec._pattern.null.shape == (_reduced_size(fem), 3)


def test_schwarz_rigid_on_a_slip_reduced_elasticity_gives_the_direct_answer(_partition):
    fem = _strip(slip=True)
    assert fem._periodic is not None and fem._periodic.get("coupling") == "slip"
    ref = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))
    spec = jno.precond.schwarz(parts=4, nullspace="rigid")
    got = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-12, maxiter=2000), precond=spec))
    assert _rel(got, ref) < 1e-9
    assert spec._pattern.n == _reduced_size(fem) < fem.dofs


def test_fsai_on_a_periodic_elasticity_gives_the_direct_answer():
    """FSAI builds its pattern through the same representative operator (it failed with an IndexError)."""
    fem = _strip()
    ref = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))
    got = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-12, maxiter=2000), precond=jno.precond.fsai()))
    assert _rel(got, ref) < 1e-9


def test_a_full_space_nullspace_array_on_a_reduced_system_is_refused():
    """A (n_full, k) array reshaped to (n_red, -1) could silently come out as garbage columns
    (162 x 8 = 144 x 9 here); it is refused by name instead."""
    fem = _strip()
    wrong = np.ones((fem.dofs, 8))
    with pytest.raises(ValueError, match="REDUCED"):
        fem.solve(linear=jno.solve.cg(), precond=jno.precond.schwarz(parts=1, nullspace=wrong))
