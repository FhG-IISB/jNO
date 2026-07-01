"""Kirchhoff-plate boundary conditions on the C¹ (Argyris) and Morley elements — the granular essential trace.

A 4th-order plate field has two independent essential boundary traces: the **deflection** ``u(region) - g``
and the **rotation** ``u.dn(region) - h`` (``∂u/∂n = h``). The classical BCs compose from them — clamped =
both, simply-supported = deflection only, guided = rotation only, free = neither (natural ``M_n=V_n=0``). The
decisive test is that ``u(region)-g`` alone genuinely *frees* the rotation: the same uniform-load square plate
must give **different** deflections for clamped vs simply-supported, each matching Timoshenko's tabulated
coefficient (``w_max = 0.00126`` clamped, ``0.00406`` simply-supported, for ``q=D=a=1``).

Reference: S. Timoshenko, S. Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed. (1959), Table 35.
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import Polygon, box  # noqa: E402

import jno  # noqa: E402

inner, hess, lap = jno.np.inner, jno.np.hessian, jno.np.laplacian
_dense = lambda A: jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)  # noqa: E731
_solve = lambda A, b: np.asarray(jnp.linalg.solve(_dense(A), jnp.asarray(b).reshape(-1)))  # noqa: E731


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _plate_center_deflection(space, clamped, mesh_size=0.06):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, phi = d.fem_symbols(space=space)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    q = 1.0 + 0.0 * xi
    bih = (
        lap(ui, [xi, yi]) * lap(vi, [xi, yi])
        if space == "Argyris"
        else inner(hess(ui, [xi, yi]), hess(vi, [xi, yi]), n_contract=2)
    )
    terms = [bih - q * vi, u(xb, yb) - 0.0]  # deflection w=0 (simply-supported)
    if clamped:
        terms.append(u.dn(xb, yb) - 0.0)  # + rotation ∂w/∂n=0 (clamped)
    sol = np.asarray(jno.fem(terms).solve(_solve)).reshape(-1)
    pts = np.asarray(d.mesh.points)[:, :2]
    nv = pts.shape[0]
    w = sol[6 * np.arange(nv)] if space == "Argyris" else sol[np.arange(nv)]
    return float(w[np.argmin((pts[:, 0] - 0.5) ** 2 + (pts[:, 1] - 0.5) ** 2)])


@pytest.mark.parametrize("space", ["Argyris", "Morley"])
def test_clamped_vs_simply_supported_matches_timoshenko(space):
    """The reason the granular BC exists: `u(reg)-g` alone (simply-supported) must give a *different*, and
    larger, deflection than `u(reg)-g` + `u.dn(reg)-0` (clamped) — each matching Timoshenko's coefficient. A
    silently-still-clamped value BC would give the two the same answer; this is what a grep cannot catch."""
    w_clamped = _plate_center_deflection(space, clamped=True)
    w_simple = _plate_center_deflection(space, clamped=False)
    assert abs(w_clamped / 0.00126 - 1.0) < 0.06, f"{space} clamped w_max={w_clamped:.5f} (Timoshenko 0.00126)"
    assert abs(w_simple / 0.00406 - 1.0) < 0.06, f"{space} simply-supported w_max={w_simple:.5f} (Timoshenko 0.00406)"
    assert w_simple > 2.5 * w_clamped, "u(reg)-g alone must FREE the rotation (simply-supported ≠ clamped)"


def test_free_edge_cantilever_plate():
    """A free edge is written by *not* pinning it: clamp one edge, leave the other three free. With the
    ν-weighted plate energy the free edges get the natural M_n=V_n=0, so the plate cantilevers — the tip
    deflects, the clamped edge stays flat."""
    NU = 0.3
    d = jno.domain(box(0.0, 0.0, 1.0, 0.4), mesh_size=0.06)
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    u, phi = d.fem_symbols(space="Morley")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    q = 1.0 + 0.0 * xi
    plate = (1 - NU) * inner(hess(ui, [xi, yi]), hess(vi, [xi, yi]), n_contract=2) + NU * lap(ui, [xi, yi]) * lap(
        vi, [xi, yi]
    )
    fem = jno.fem([plate - q * vi, u(xl, yl) - 0.0, u.dn(xl, yl) - 0.0])  # clamp LEFT only; other edges free
    pts = np.asarray(d.mesh.points)[:, :2]
    nv = pts.shape[0]
    w = np.asarray(fem.solve(_solve)).reshape(-1)[np.arange(nv)]
    w_clamp = np.abs(w[pts[:, 0] < 1e-9]).max()
    w_tip = np.abs(w[pts[:, 0] > 1 - 1e-9]).max()
    assert w_tip > 1e-4 and w_tip > 20 * max(w_clamp, 1e-30), f"cantilever must deflect at the free tip: {w_tip:.2e}"


def test_argyris_rotation_nonaxis_raises_morley_does_not():
    """The rotation pin needs the (n,t) frame, wired for axis-aligned edges only on Argyris → a non-axis-aligned
    edge must raise loudly (never a silent wrong pin). Morley — whose edge DOF *is* ∂u/∂n — handles any
    orientation."""
    diamond = Polygon([(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)])  # all edges non-axis-aligned

    def _solve_clamped(space):
        d = jno.domain(diamond, mesh_size=0.15)
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        u, phi = d.fem_symbols(space=space)
        ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
        bih = (
            lap(ui, [xi, yi]) * lap(vi, [xi, yi])
            if space == "Argyris"
            else inner(hess(ui, [xi, yi]), hess(vi, [xi, yi]), n_contract=2)
        )
        return jno.fem([bih - (1.0 + 0.0 * xi) * vi, u(xb, yb) - 0.0, u.dn(xb, yb) - 0.0]).solve(_solve)

    with pytest.raises(NotImplementedError, match="axis-aligned"):
        _solve_clamped("Argyris")
    sol = np.asarray(_solve_clamped("Morley")).reshape(-1)  # Morley: any orientation
    assert np.all(np.isfinite(sol)) and np.abs(sol).max() > 0
