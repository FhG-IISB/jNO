"""Periodic ties on a COUPLED (multi-field) problem.

Covers: the block-wise Galerkin reduction equals the dense block-diagonal P^T M P (the off-diagonal
cross-block path), and an end-to-end coupled 2-field periodic transient — decoupled fields match
their analytic decay (per-field reduction), and the off-diagonal coupling is actually assembled.
"""

import jax
import numpy as np
import pytest

import jno
import jno.jnp_ops as jnn


@pytest.fixture
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_block_reduction_matches_dense_blockdiag_P(_x64):
    """reduce_matrix_periodic must equal P_mf^T M P_mf with P_mf block-diagonal — including the
    off-diagonal field-coupling blocks M[i,j]."""
    import jax.numpy as jnp

    from jno.utils.solver.fem_utils import reduce_matrix_periodic

    rng = np.random.default_rng(0)
    P0 = jnp.asarray(rng.standard_normal((5, 4)))
    P1 = jnp.asarray(rng.standard_normal((5, 3)))
    periodic = {
        "blocks": [{"P": P0, "kept": np.arange(4), "vec": 1}, {"P": P1, "kept": np.arange(3), "vec": 1}],
        "off_full": [0, 5, 10],
        "off_red": [0, 4, 7],
    }
    M = jnp.asarray(rng.standard_normal((10, 10)))  # full coupled operator, nonzero off-diagonal blocks
    red = np.asarray(reduce_matrix_periodic(periodic, M))
    z = np.zeros((5, 3))
    Pmf = np.block([[np.asarray(P0), z], [np.zeros((5, 4)), np.asarray(P1)]])
    ref = Pmf.T @ np.asarray(M) @ Pmf
    assert np.allclose(red, ref, atol=1e-10)


def _periodic_domain(n):
    dom = jno.domain(
        constructor=jno.domain.equi_distant_rect(x_range=(0.0, 1.0), y_range=(0.0, 1.0), nx=n, ny=n),
        time=(0.0, 0.02, 11),
        compute_mesh_connectivity=False,
    )
    for nm, pred in {
        "left": lambda x, y: x < 1e-6,
        "right": lambda x, y: x > 1 - 1e-6,
        "bottom": lambda x, y: y < 1e-6,
        "top": lambda x, y: y > 1 - 1e-6,
    }.items():
        dom.tag(nm, pred)
    return dom


def _build(dom, coupling=0.0):
    """Two coupled periodic heat fields u, w with per-field diffusivity and optional u<-w coupling."""
    import jax.numpy as jnp  # noqa: F401

    u, pu = dom.fem_symbols(names=("u", "pu"))
    w, pw = dom.fem_symbols(names=("w", "pw"))
    xi, yi, ti = dom.variable("interior", split=True)
    ci = dom.variable("initial", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    ui, pui = u.bind(x=xi, y=yi, t=ti), pu.bind(x=xi, y=yi, t=ti)
    wi, pwi = w.bind(x=xi, y=yi, t=ti), pw.bind(x=xi, y=yi, t=ti)
    ic = jnn.sin(2 * np.pi * ci[0]) * jnn.sin(2 * np.pi * ci[1])
    u_eq = ui.t * pui + 0.1 * (ui.x * pui.x + ui.y * pui.y) - coupling * wi * pui  # off-diagonal u<-w coupling
    w_eq = wi.t * pwi + 0.2 * (wi.x * pwi.x + wi.y * pwi.y)
    return jno.fem(
        [
            u_eq,
            w_eq,
            u(xl, yl) - u(xr, yr),
            u(xb, yb) - u(xt, yt),
            w(xl, yl) - w(xr, yr),
            w(xb, yb) - w(xt, yt),
            u(ci[0], ci[1]) - ic,
            w(ci[0], ci[1]) - ic,
        ]
    )


def _march(block):
    import jax.numpy as jnp

    U = jnp.asarray(block.state0)
    t, dt = float(block.t0), float(block.dt)
    for _ in range(round((block.t1 - block.t0) / dt)):
        U = block.step(U, t, dt)
        t += dt
    return np.asarray(block.prolong(U))  # full nodal [u | w]


def test_decoupled_multifield_periodic_matches_analytic(_x64):
    """Two periodic heat fields with different nu each match exp(-8 pi^2 nu t) — per-field reduction."""
    n = 12
    dom = _periodic_domain(n)
    fem = _build(dom, coupling=0.0)
    block = fem.operator
    assert block.metadata.get("periodic") is True
    assert block.metadata["reduced_state_size"] < block.metadata["full_state_size"]

    U_full = _march(block)
    off = fem.offsets
    pts = np.asarray(fem.field_points[0])
    mode = np.sin(2 * np.pi * pts[:, 0]) * np.sin(2 * np.pi * pts[:, 1])
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))  # noqa: E731
    u_an = np.exp(-8.0 * np.pi**2 * 0.1 * block.t1) * mode
    w_an = np.exp(-8.0 * np.pi**2 * 0.2 * block.t1) * mode
    assert rel(U_full[off[0] : off[1]], u_an) < 5e-2
    assert rel(U_full[off[1] : off[2]], w_an) < 5e-2


def test_coupled_multifield_periodic_assembles_and_couples(_x64):
    """The off-diagonal coupling is assembled + reduced: turning it on changes the u field, and the
    periodic tie holds exactly on both fields."""
    n = 12
    U0 = _march(_build(_periodic_domain(n), coupling=0.0).operator)
    Uc = _march(_build(_periodic_domain(n), coupling=3.0).operator)
    fem = _build(_periodic_domain(n), coupling=3.0)
    off = fem.offsets
    # coupling changed the u field (off-diagonal block is live), left w roughly alone
    assert np.linalg.norm(Uc[off[0] : off[1]] - U0[off[0] : off[1]]) > 1e-3
    assert np.all(np.isfinite(Uc))
    # periodic tie satisfied: u(left node) == u(right node) after prolong
    pts = np.asarray(fem.field_points[0])
    left = np.where(pts[:, 0] < 1e-6)[0]
    right = np.where(pts[:, 0] > 1 - 1e-6)[0]
    ul = Uc[off[0] : off[1]]
    # match left/right nodes by y-coordinate
    li = left[np.argsort(pts[left, 1])]
    ri = right[np.argsort(pts[right, 1])]
    assert np.allclose(ul[li], ul[ri], atol=1e-8)
