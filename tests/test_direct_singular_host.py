"""A singular operator on the HOST sparse-direct path fails by name, not deep inside scipy.

On the CPU the sparse-direct solve is SuperLU (scipy), reached two ways: ``jno.solve.lu(backend="host")``
(jNO's own callback around ``splu``) and the default ``jno.solve.lu()`` / ``newton(direct=True)`` (JAX's
``spsolve``, whose callback calls scipy's ``spsolve``). A singular operator used to surface as
"failed to factorize matrix at line 413 in file .../dpanel_bmod.c" or "Factor is exactly singular",
wrapped in a JAX callback error naming neither the matrix nor the cause -- or, for ``spsolve`` on an
exactly singular matrix, as a NaN that the Newton verdict then blamed on ``max_steps``.

Oracle: ``u**3 v - v`` with ``u = 1`` on the boundary has the tangent ``3 u**2 M``, which is EXACTLY zero
on every interior row at ``u0 = 0`` -- singular by construction -- and regular from ``u0 = 1/2``, where
Newton converges to the exact root ``u = 1``.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

pytest.importorskip("meshio")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# The DEFAULT sparse-direct solve is SuperLU only on the CPU; on a GPU it is cuSolver, which raises its own
# "Singular matrix in linear solve" (loud, but not this message). These two tests pin the CPU path.
_on_cpu = pytest.mark.skipif(
    jax.default_backend() != "cpu", reason="the default sparse-direct solve is SuperLU only on the CPU backend"
)


def _cubic():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=4).domain()
    u, v = d.fem_symbols(names=("u", "v"), order=1)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    return jno.fem([ui * ui * ui * vi - 1.0 * vi, u(xb, yb) - 1.0])


def _newton(fem, x0, linear=None):
    return np.asarray(fem.solve(x0=x0, nonlinear=jno.solve.newton(direct=True), linear=linear)).reshape(-1)


def test_host_lu_names_a_singular_tangent_with_its_size():
    fem = _cubic()
    n = fem.dofs
    with pytest.raises(RuntimeError, match=rf"could not factorize the {n}x{n} operator: a singular / structurally"):
        _newton(fem, np.zeros(n), jno.solve.lu(backend="host"))
    assert fem.stats["error"].startswith("RuntimeError: host SuperLU could not factorize"), fem.stats


def test_host_lu_is_unaffected_on_a_regular_tangent():
    fem = _cubic()
    got = _newton(fem, np.full(fem.dofs, 0.5), jno.solve.lu(backend="host"))
    np.testing.assert_allclose(got, 1.0, atol=1e-9)


@_on_cpu
def test_default_direct_newton_blames_the_singular_tangent_not_max_steps():
    """JAX's spsolve answers an exactly singular matrix with NaN; the verdict must say so."""
    fem = _cubic()
    with pytest.raises(RuntimeError, match=r"did not converge: it produced a non-finite residual") as err:
        _newton(fem, np.zeros(fem.dofs))
    assert "singular tangent" in str(err.value) and "max_steps=" not in str(err.value)
    np.testing.assert_allclose(_newton(fem, np.full(fem.dofs, 0.5)), 1.0, atol=1e-9)


@_on_cpu
def test_a_superlu_breakdown_inside_jax_spsolve_is_named(monkeypatch):
    """SuperLU's internal ABORT on a rank-deficient panel. It is a matter of pivoting luck which singular
    matrix aborts rather than returns NaN or a finite wrong answer (measured: a 3334-DOF three-field
    u-p-s saddle with continuous P1 stress aborts, the same form at 2582 DOFs does not), so the abort is
    injected at scipy -- below JAX's own callback, so the wrapping the user sees is the real one."""
    import scipy.sparse.linalg as spla

    real = spla.spsolve

    def abort(*a, **k):
        raise RuntimeError("failed to factorize matrix at line 413 in file ../SuperLU/SRC/dpanel_bmod.c")

    fem = _cubic()
    n = fem.dofs
    monkeypatch.setattr(spla, "spsolve", abort)
    with pytest.raises(RuntimeError, match=rf"the solved system is {n}x{n}\): a singular / structurally") as err:
        _newton(fem, np.full(n, 0.5))
    assert "dpanel_bmod" in str(err.value), "the SuperLU message itself must be kept"
    assert fem.stats["error"].startswith("RuntimeError: host SuperLU could not factorize"), fem.stats
    monkeypatch.setattr(spla, "spsolve", real)
    np.testing.assert_allclose(_newton(fem, np.full(n, 0.5)), 1.0, atol=1e-9)
