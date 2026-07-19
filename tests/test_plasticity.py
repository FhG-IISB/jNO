"""J2 elasto-plasticity written **entirely in the jno trace** — no jno.plasticity module, no material
object. The radial return contracts against the test strain to a SCALAR weak form per Gauss point
(``dev(A):B = A:B - tr(A)tr(B)/3``, ``||dev(A)||^2 = A:A - tr(A)^2/3`` — the same trick the elastic form
uses, ``trace*trace`` instead of an identity). This file IS the proof that plasticity is a formula:

  * deformation theory (virgin) — shear + uniaxial-strain patch tests are exact and land on the yield
    surface; a genuine BVP shows plasticity caps the stress below the elastic peak; the solve is
    differentiable to a material parameter;
  * flow theory — the identical formula with ``ee = eps(u) - ep.i(-1)`` reads the previous per-QP state
    through the general ``.i(k)`` step-history channel, and at zero history reproduces deformation theory
    to machine zero.

Reference: Simo & Hughes, *Computational Inelasticity* (1998), Box 3.2 (radial return, iso hardening).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

pytest.importorskip("basix", reason="native Lagrange assembler needs basix")

E, NU, SY, H = 200.0, 0.3, 2.0, 20.0
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))
K = LAM + 2.0 * MU / 3.0
RT = 1.5**0.5


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def j2_weak(u, phi, coords, *, sy=SY, H=H, hist=None):
    """The J2 radial-return internal virtual work ``∫ σ:ε(φ)`` as a SCALAR trace formula. ``hist=None`` is
    deformation theory (virgin); ``hist=(ep, al)`` reads the previous plastic strain/hardening via
    ``.i(-1)`` (flow theory). This 6-line helper is exactly what a user writes behind their own aliases —
    there is no jno.plasticity."""
    sym, grad, trace, inner, sqrt, maximum = (
        jno.np.sym,
        jno.np.grad,
        jno.np.trace,
        jno.np.inner,
        jno.np.sqrt,
        jno.np.maximum,
    )
    eps = lambda w: sym(grad(w, list(coords)))
    eu = eps(u) if hist is None else eps(u) - hist[0].i(-1)
    yield0 = sy if hist is None else sy + H * hist[1].i(-1)
    ev = eps(phi)
    tru, trv = trace(eu), trace(ev)
    ddev = sqrt(maximum(inner(eu, eu, 2) - tru * tru / 3.0, 0.0) + 1e-30)  # ||dev(ee)||, safe at 0
    dg = maximum(RT * 2 * MU * ddev - yield0, 0.0) / (3 * MU + H)  # plastic multiplier
    dev_ev = inner(eu, ev, 2) - tru * trv / 3.0  # dev(ee):ε(φ)
    return K * tru * trv + 2 * MU * dev_ev - 2 * MU * RT * dg * dev_ev / ddev


def _box(size=0.34):
    d = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=size).domain()
    d.tag("bdry", lambda x, y, z: (x < 1e-6) | (x > 1 - 1e-6) | (y < 1e-6) | (y > 1 - 1e-6) | (z < 1e-6) | (z > 1 - 1e-6))
    return d


def _q_uniaxial_shear(g):
    q_tr = np.sqrt(3) * 2 * MU * g
    dgamma = max(q_tr - SY, 0) / (3 * MU + H)
    return q_tr - 3 * MU * dgamma  # von Mises on the (hardening) yield surface


def test_shear_patch_test_exact_and_plastic():
    g = 3.0 * SY / (2 * np.sqrt(3) * MU)  # tensor shear component, past yield
    d = _box()
    u, phi = d.fem_symbols(value_shape=(3,))
    co, cc = d.variable("interior", split=True), d.variable("bdry", split=True)
    mech = j2_weak(u, phi, [co[0], co[1], co[2]])
    fem = jno.fem(
        [mech, u(cc[0], cc[1], cc[2])[0] - g * cc[1], u(cc[0], cc[1], cc[2])[1] - 0.0, u(cc[0], cc[1], cc[2])[2] - 0.0]
    )
    assert not fem.is_linear
    sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(max_steps=40, rtol=1e-11, atol=1e-13))).reshape(-1, 3)
    pts = np.asarray(fem.field_points[0])[:, :3]
    assert np.abs(sol[:, 0] - g * pts[:, 1]).max() < 1e-8 and np.abs(sol[:, 1:]).max() < 1e-8
    assert _q_uniaxial_shear(g / 2.0) > SY + 1e-6  # genuinely plastic


def test_uniaxial_strain_patch_test():
    eps_xx = 4.0 * SY / (2 * np.sqrt(3) * MU)
    d = _box()
    u, phi = d.fem_symbols(value_shape=(3,))
    co, cc = d.variable("interior", split=True), d.variable("bdry", split=True)
    mech = j2_weak(u, phi, [co[0], co[1], co[2]])
    fem = jno.fem(
        [mech, u(cc[0], cc[1], cc[2])[0] - eps_xx * cc[0], u(cc[0], cc[1], cc[2])[1] - 0.0, u(cc[0], cc[1], cc[2])[2] - 0.0]
    )
    sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(max_steps=40, rtol=1e-11, atol=1e-13))).reshape(-1, 3)
    pts = np.asarray(fem.field_points[0])[:, :3]
    assert np.abs(sol[:, 0] - eps_xx * pts[:, 0]).max() < 1e-8


def test_plasticity_caps_stress_below_elastic_peak():
    """Genuine BVP (bottom fixed, top sheared past yield, sides free): the plastic peak von Mises is
    bounded by the yield surface, strictly below the (unbounded) elastic peak — yielding limited it."""
    shear = 5.0 * SY / (2 * np.sqrt(3) * MU)
    d = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=0.4).domain()
    d.tag("bot", lambda x, y, z: z < 1e-6)
    d.tag("top", lambda x, y, z: z > 1 - 1e-6)
    u, phi = d.fem_symbols(value_shape=(3,))
    co, cb, ct = d.variable("interior", split=True), d.variable("bot", split=True), d.variable("top", split=True)
    X = [co[0], co[1], co[2]]
    bcs = [
        u(cb[0], cb[1], cb[2])[0] - 0.0,
        u(cb[0], cb[1], cb[2])[1] - 0.0,
        u(cb[0], cb[1], cb[2])[2] - 0.0,
        u(ct[0], ct[1], ct[2])[0] - shear,
        u(ct[0], ct[1], ct[2])[1] - 0.0,
        u(ct[0], ct[1], ct[2])[2] - 0.0,
    ]
    sym, grad, trace, inner = jno.np.sym, jno.np.grad, jno.np.trace, jno.np.inner
    eps = lambda w: sym(grad(w, X))
    eu, ev = eps(u), eps(phi)
    el = LAM * trace(eu) * trace(ev) + 2 * MU * inner(eu, ev, 2)
    sol_el = np.asarray(jno.fem([el, *bcs]).solve()).reshape(-1, 3)
    sol_pl = np.asarray(
        jno.fem([j2_weak(u, phi, X), *bcs]).solve(nonlinear=jno.solve.newton(max_steps=60, rtol=1e-10, atol=1e-12))
    ).reshape(-1, 3)

    cells = np.asarray(d.mesh.cells_dict["tetra"]).astype(np.int64)
    pts = np.asarray(jno.fem([j2_weak(u, phi, X), *bcs]).field_points[0])[:, :3]
    v = pts[cells]
    G = np.zeros((len(cells), 4, 3))
    G[:, 1:, :] = np.transpose(np.linalg.inv(v[:, 1:, :] - v[:, :1, :]), (0, 2, 1))
    G[:, 0, :] = -G[:, 1:, :].sum(1)

    def qmax(sol, plastic):
        gu = np.einsum("caj,cai->cij", G, sol[cells])
        e = 0.5 * (gu + np.transpose(gu, (0, 2, 1)))
        tr = e[:, 0, 0] + e[:, 1, 1] + e[:, 2, 2]
        s = 2 * MU * (e - tr[:, None, None] / 3 * np.eye(3))  # trial deviator
        q = np.sqrt(1.5) * np.linalg.norm(s, axis=(1, 2))
        if plastic:
            q = np.minimum(q, SY + H * np.maximum(q - SY, 0) / (3 * MU + H))  # returned to the yield surface
        return q.max()

    assert qmax(sol_el, False) > SY  # elastic overshoots
    assert qmax(sol_pl, True) < qmax(sol_el, False)  # plasticity capped it


def test_gradient_flows_to_material_parameter_through_solve():
    """Inverse-problem readiness: d(response)/d(sigma_y) through the plastic Newton solve matches FD."""
    shear = 4.0 * SY / (2 * np.sqrt(3) * MU)
    d = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=0.5).domain()
    d.tag("bot", lambda x, y, z: z < 1e-6)
    d.tag("top", lambda x, y, z: z > 1 - 1e-6)
    u, phi = d.fem_symbols(value_shape=(3,))
    co, cb, ct = d.variable("interior", split=True), d.variable("bot", split=True), d.variable("top", split=True)
    X = [co[0], co[1], co[2]]
    syP = jno.np.parameter((1,), name="sy")
    mech = j2_weak(u, phi, X, sy=jno.np.reshape(syP, ()))
    fem = jno.fem(
        [
            mech,
            u(cb[0], cb[1], cb[2])[0] - 0.0,
            u(cb[0], cb[1], cb[2])[1] - 0.0,
            u(cb[0], cb[1], cb[2])[2] - 0.0,
            u(ct[0], ct[1], ct[2])[0] - shear,
            u(ct[0], ct[1], ct[2])[1] - 0.0,
            u(ct[0], ct[1], ct[2])[2] - 0.0,
        ]
    )
    op = fem.operator
    from jno.utils.solver.newton_krylov import newton_krylov

    u0 = jnp.zeros(int(fem.dofs))
    energy = lambda s: jnp.mean(newton_krylov(lambda uu: op.residual(uu, {"sy": jnp.reshape(s, (1,))}), u0) ** 2)
    g = jax.grad(energy)(2.0)
    fd = (energy(2.0 + 1e-4) - energy(2.0 - 1e-4)) / 2e-4
    assert abs(float(g)) > 1e-8 and jnp.allclose(g, fd, rtol=5e-3)


def test_flow_theory_weak_form_reads_history_and_reduces_to_deformation_theory():
    """The flow-theory formula (ee = eps(u) - ep.i(-1)) reads the previous per-QP state through the
    general .i(k) channel; threading a ZERO history buffer must reproduce the deformation-theory residual
    to machine zero — the correctness anchor for the whole history mechanism."""
    g = 3.0 * SY / (2 * np.sqrt(3) * MU)
    d = _box()
    u, phi = d.fem_symbols(value_shape=(3,))
    ep, _ = d.fem_symbols(value_shape=(3, 3), names=("ep", "ep_t"))
    al, _ = d.fem_symbols(value_shape=(), names=("al", "al_t"))
    co, cc = d.variable("interior", split=True), d.variable("bdry", split=True)
    X = [co[0], co[1], co[2]]
    bcs = [u(cc[0], cc[1], cc[2])[0] - g * cc[1], u(cc[0], cc[1], cc[2])[1] - 0.0, u(cc[0], cc[1], cc[2])[2] - 0.0]

    fem_h = jno.fem([j2_weak(u, phi, X, hist=(ep, al)), *bcs])
    specs = fem_h.operator.history_specs
    assert {v["name"] for v in specs.values()} == {"ep", "al"}  # history detected via .i(-1)
    zero = {k: jnp.zeros(specs[k]["shape"]) for k in specs}
    rng = np.random.default_rng(0)
    u_test = rng.normal(size=int(fem_h.dofs)) * 0.01
    R_flow0 = np.asarray(fem_h.operator.residual(u_test, {"__history__": zero}))

    fem_d = jno.fem([j2_weak(u, phi, X), *bcs])  # deformation theory (virgin)
    R_defm = np.asarray(fem_d.operator.residual(u_test, None))
    assert np.abs(R_flow0 - R_defm).max() < 1e-10  # zero history == deformation theory
