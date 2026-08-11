"""Periodic ties composed with a nonlocal ``Coupling`` term (Phase 4 of the compose work).

A ``Coupling`` folds a nonlocal residual ``c(u)`` into ``R(u) = A u - b + c(u)``, promoting a steady
linear form to a nonlinear ``FemResidualOperator``. A periodic tie reduces the system. The two compose
for free: the existing nonlinear-periodic solve path already solves ``Pᵀ r(P u_red) = 0``, so the
*coupled* residual reduces through the same wrap -- no per-combination branch. (Previously this raised
NotImplementedError.) These tests pin: the promotion+reduction is recognised, a zero coupling recovers
the plain periodic linear solve across the two solve paths, and a constant-load coupling reproduces the
*independent* plain periodic solve of the equivalent extra-source problem.

Run with x64 (FEM assembly/solves are float64).
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi


_DUMMY = jno.domain.from_array({"_": np.zeros((1, 1))})  # global FEM solve -> crux needs a driver domain


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _eval(fem):
    """Evaluate a solve to a full nodal array. A plain linear periodic solve is already a JAX array; a
    coupled (nonlinear) periodic solve is a *traced* node, evaluated via a throwaway crux -- the
    established nonlinear-periodic pattern (the residual may carry trainable params)."""
    out = fem.solve()
    if isinstance(out, jax.Array):
        return np.asarray(out).reshape(-1)
    crux = jno.core([out.mean], domain=_DUMMY)
    return np.asarray(crux.eval([out])).reshape(-1)


def _build(extra_source=None, coupling=None, mesh_size=0.07):
    """Periodic-in-x, Dirichlet-in-y reaction-diffusion ``-Δu + u = f (+ extra_source)`` with manufactured
    ``u* = cos(2πx) sin(πy)`` (``f = (5π²+1) u*``). Optionally add a constant extra local source and/or a
    nonlocal ``coupling`` (a bare ``w -> (n_dofs,)`` function passed in the jno.fem list)."""
    dom = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    dom.tag("left", lambda x, y: (x < 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("right", lambda x, y: (x > 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("bottom", lambda x, y: y < 1e-6)
    dom.tag("top", lambda x, y: y > 1 - 1e-6)
    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = (5 * PI**2 + 1.0) * jno.np.cos(2 * PI * xi) * jno.np.sin(PI * yi)
    weak = ui.x * vi.x + ui.y * vi.y + ui * vi - f * vi
    if extra_source is not None:
        weak = weak - extra_source * vi
    terms = [weak, u(xb, yb) - 0.0, u(xt, yt) - 0.0, u(xl, yl) - u(xr, yr)]
    if coupling is not None:
        terms.append(coupling)
    return jno.fem(terms)


def test_periodic_coupling_promotes_and_reduces():
    """A periodic tie + a Coupling: the local form is promoted to nonlinear AND the tie reduces it."""
    fem = _build(coupling=lambda w: jnp.zeros_like(jnp.asarray(w).reshape(-1)))
    assert fem._mode == "nonlinear", "the coupling must promote the linear periodic form to nonlinear"
    assert fem._periodic is not None, "the u(left)-u(right) tie must still reduce the coupled system"
    assert fem._periodic["n_red"] < fem._periodic["n_full"], "the tie must eliminate the secondary-face DOFs"


def test_periodic_coupling_zero_is_plain_periodic():
    """Extreme: a coupling returning zeros must give exactly the plain periodic *linear* solve -- the
    promotion to a nonlinear residual + the periodic reduction must not perturb the solution. This also
    cross-checks the nonlinear-periodic solve path against the linear-periodic path."""
    u_plain = _eval(_build())  # linear periodic path
    u_zero = _eval(_build(coupling=lambda w: jnp.zeros_like(jnp.asarray(w).reshape(-1))))  # nonlinear+periodic path
    assert np.allclose(u_zero, u_plain, atol=1e-6), "zero coupling perturbed the periodic solve"

    pts = np.asarray(_build().points)
    u_star = np.cos(2 * PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    rel = float(np.linalg.norm(u_plain - u_star) / np.linalg.norm(u_star))
    assert rel < 2e-2, f"periodic solve does not recover u*: rel-L2 {rel:.3e}"


def test_periodic_coupling_constant_load_matches_extra_source():
    """Correctness vs an independent reference. A constant-load coupling ``c(w) = -extra`` makes the
    coupled residual ``A u - b - extra = 0`` i.e. ``A u = b + extra``. Choosing ``extra`` to be the FE
    load of a constant source ``s0`` (obtained as the difference of two independent linear assemblies'
    RHS vectors), the coupled+periodic solve must reproduce the plain periodic *linear* solve of the
    equivalent extra-source problem -- a reference that never touches the coupling/nonlinear path."""
    s0 = 3.0  # a constant extra source
    b_plain = np.asarray(_build()._op[1]).reshape(-1)  # raw (A, b): full RHS, un-reduced
    b_loaded = np.asarray(_build(extra_source=s0)._op[1]).reshape(-1)
    extra = jnp.asarray(b_loaded - b_plain)  # the FE load of s0*vi (Dirichlet rows cancel -> 0 there)

    u_ref = _eval(_build(extra_source=s0))  # plain linear periodic, equivalent problem
    u_coupled = _eval(_build(coupling=lambda w: -extra))  # nonlinear+periodic path
    rel = float(np.linalg.norm(u_coupled - u_ref) / np.linalg.norm(u_ref))
    assert rel < 1e-5, f"periodic+coupling disagrees with the equivalent extra-source periodic solve: {rel:.3e}"
    # and the coupling genuinely moved the solution off the no-load periodic solve
    u_plain = _eval(_build())
    assert float(np.linalg.norm(u_coupled - u_plain)) > 1e-3, "the coupling did not change the solution"
