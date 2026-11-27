"""Unknowns with their own spaces in one problem: velocity P_N and pressure P_{N-2} (spectral Stokes/Navier-Stokes).

The pressure lives on the Gauss nodes of degree N-2 (``space="spectral:gauss"``), where the continuity rows are
collocated: the classic P_N - P_{N-2} pair, which has no spurious pressure modes. Oracles: Kovasznay flow (exact
steady Navier-Stokes) converges exponentially; the pressure is unique up to the one constant the gauge fixes.
"""

import jax
import jno.numpy as jnn
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


RE = 40.0
LAM = RE / 2 - np.sqrt(RE**2 / 4 + 4 * np.pi**2)


def _kovasznay(n):
    d = jno.shape.rect(-0.5, -0.5, 1.0, 1.5).domain()
    V = d.unknown(space="spectral", value_shape=(2,), order=(n, n), name="v")
    P = d.unknown(space="spectral:gauss", order=(n - 2, n - 2), name="p")
    x, y, _ = d.variable("interior", split=True)
    xp, yp, _ = d.variable("interior", nodes=P, split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    v = V.bind(x=x, y=y)
    p = P.bind(x=x, y=y)
    ex = lambda X, Y: jnn.stack(  # noqa: E731
        [1 - jnn.exp(LAM * X) * jnn.cos(2 * np.pi * Y), LAM / (2 * np.pi) * jnn.exp(LAM * X) * jnn.sin(2 * np.pi * Y)],
        axis=-1,
    )
    cons = [
        v.grad() @ v + p.grad() - (1 / RE) * v.laplacian(),
        V.bind(x=xp, y=yp).div(),
        V(xb, yb) - ex(xb, yb),
        P.bind(x=xp, y=yp).mean,
    ]
    sol_v, sol_p = jno.spectral(cons).solve()
    Pv = V.basis.nodes
    uex = 1 - np.exp(LAM * Pv[:, 0]) * np.cos(2 * np.pi * Pv[:, 1])
    vex = LAM / (2 * np.pi) * np.exp(LAM * Pv[:, 0]) * np.sin(2 * np.pi * Pv[:, 1])
    err_v = max(np.abs(np.asarray(sol_v)[:, 0] - uex).max(), np.abs(np.asarray(sol_v)[:, 1] - vex).max())
    Pp = P.basis.nodes
    pex = 0.5 * (1 - np.exp(2 * LAM * Pp[:, 0]))
    sp = np.asarray(sol_p)
    err_p = np.abs((sp - sp.mean()) - (pex - pex.mean())).max()
    return err_v, err_p


def test_kovasznay_converges_exponentially():
    errs = [_kovasznay(n) for n in (12, 16, 20)]
    ev = [e[0] for e in errs]
    assert ev[1] < 0.05 * ev[0] and ev[2] < 0.05 * ev[1], errs
    assert ev[2] < 1e-7 and errs[2][1] < 1e-5, errs


def test_pressure_on_gauss_nodes_has_one_null_mode():
    d = jno.shape.rect(-1.0, -1.0, 1.0, 1.0).domain()
    V = d.unknown(space="spectral", value_shape=(2,), order=(10, 10))
    P = d.unknown(space="spectral:gauss", order=(8, 8))
    x, y, _ = d.variable("interior", split=True)
    xp, yp, _ = d.variable("interior", nodes=P, split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    v, p = V.bind(x=x, y=y), P.bind(x=x, y=y)
    prob = jno.spectral([-v.laplacian() + p.grad(), V.bind(x=xp, y=yp).div(), V(xb, yb) - 0.0, P.bind(x=xp, y=yp).mean])
    assert P.basis.grid.axes[0].rule == "gauss"
    n = prob._Ntot
    R = prob._residual_fn()
    J = np.asarray(jax.jacfwd(R)(jax.numpy.zeros(n)))
    s = np.linalg.svd(J, compute_uv=False)
    assert s[-1] / s[0] > 1e-12  # nonsingular with the gauge: no spurious pressure mode
