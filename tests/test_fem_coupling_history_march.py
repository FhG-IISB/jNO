"""A nonlocal ``jno.Coupling`` on a step-history form marched over ``domain(tau=...)``.

The Coupling wrapper rebuilt the residual operator and kept only the residual, so the step-history
layout and the state readout the assembler hangs on it were dropped: ``fem.solve()`` no longer saw a
march, ran a plain steady solve, and the first residual raised on an unbuffered ``u.i(-1)``. The wrapper
also swallowed the march's third residual argument (the load-path time). It now forwards both.

Oracle: a coupling whose effect has a LOCAL spelling. ``c(U) = k M U``, with ``M`` the consistent P1 mass
matrix assembled by a separate ``jno.fem`` on the same mesh, is exactly the weak term ``k u v``. The march
with the nonlocal coupling must equal the march with the local term to solver tolerance, with and without
a periodic tie (the march reduces the coupled residual as ``Pᵀ r(P ũ)``), and differ from the march with
neither -- so the coupling demonstrably acts.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

DT, NS, K, EPS = 0.02, 5, 3.0, 1e-9
PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _square(**grid):
    d = jno.shape.rect(0, 0, 1, 1).structured(n=8).domain(**grid)
    for name, pred in {
        "l": lambda x, y: x < EPS,
        "r": lambda x, y: x > 1 - EPS,
        "b": lambda x, y: y < EPS,
        "t": lambda x, y: y > 1 - EPS,
    }.items():
        d.tag(name, pred)
    return d


def _mass():
    """The consistent P1 mass matrix of the same 8x8 mesh, and its node coordinates."""
    d = _square()
    u, v = d.fem_symbols()
    V = d.variable("interior", split=True)
    fem = jno.fem([u.bind(x=V[0], y=V[1]) * v.bind(x=V[0], y=V[1])])
    return jnp.asarray(fem.A), np.asarray(fem.points)


def _march(reaction, *, tie, M=None):
    """Backward-Euler heat written with the step history, ``(u - u.i(-1))/dt - Δu + [k u] = f``, u = 0 on
    y = 0, 1; ``reaction`` in {None, "local", "nonlocal"} adds nothing, the weak term ``k u v``, or the
    Coupling ``U -> k M U``."""
    d = _square(tau=(DT, DT * NS, NS))
    u, v = d.fem_symbols(names=("u", "v"))
    V = d.variable("interior", split=True)
    ub, vb = u.bind(x=V[0], y=V[1], t=V[2]), v.bind(x=V[0], y=V[1], t=V[2])
    on = lambda r: d.variable(r, split=True)[:2]  # noqa: E731
    f = 10.0 * (1.0 + jno.np.sin(2 * PI * V[0]) + V[0])
    pde = (ub - u.i(-1)) / DT * vb + ub.x * vb.x + ub.y * vb.y - f * vb
    if reaction == "local":
        pde = pde + K * ub * vb
    terms = [pde, u(*on("b")) - 0.0, u(*on("t")) - 0.0]
    if tie:
        terms.append(u(*on("l")) - u(*on("r")))
    if reaction == "nonlocal":
        terms.append(jno.Coupling(lambda U: K * (M @ U), name="mass"))
    fem = jno.fem(terms)
    s = fem.solve()
    return np.asarray(s.fn() if hasattr(s, "fn") else s), np.asarray(fem.points)


@pytest.mark.parametrize("tie", [False, True], ids=["plain", "periodic"])
def test_a_nonlocal_coupling_on_a_tau_march_equals_its_local_spelling(tie):
    M, X_mass = _mass()
    local, X = _march("local", tie=tie)
    nonlocal_, X2 = _march("nonlocal", tie=tie, M=M)
    neither, _X = _march(None, tie=tie)
    np.testing.assert_array_equal(X, X_mass)  # the mass matrix is indexed like the march's DOFs
    np.testing.assert_array_equal(X, X2)
    assert local.shape == (NS, X.shape[0])
    scale = np.abs(local).max()
    err = np.abs(nonlocal_ - local).max() / scale
    assert err < 1e-8, f"the nonlocal k M u is {err:.2e} off the local k u v"
    gap = np.abs(neither - local).max() / scale
    assert gap > 0.05, f"the reaction barely changes the march ({gap:.2e}); the comparison would be vacuous"
