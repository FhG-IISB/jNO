"""Tensor formulas in weak forms — the identity ``jno.np.identity`` broadcasts correctly, and **finite
strain** is just a formula (no module): St. Venant-Kirchhoff ``F = I + ∇u``, ``E = ½(FᵀF − I)``,
``S = λ tr(E) I + 2μ E``, internal virtual work ``∫ (F S):∇δu``. A large-stretch patch test verifies the
formula; a small-stretch cantilever verifies it reduces to linear elasticity.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

pytest.importorskip("basix", reason="native Lagrange assembler needs basix")

E, NU = 200.0, 0.3
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _box(size=0.4):
    d = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=size).domain()
    d.tag("bdry", lambda x, y, z: (x < 1e-6) | (x > 1 - 1e-6) | (y < 1e-6) | (y > 1 - 1e-6) | (z < 1e-6) | (z > 1 - 1e-6))
    return d


def test_identity_tensor_in_weak_form_matches_scalar_elastic_form():
    """σ = λ tr(ε) I + 2μ ε written with the tensor identity must give the same solve as the scalar
    elastic form (λ trace·trace + 2μ inner). Confirms jno.np.identity broadcasts over the quad axis."""
    sym, grad, trace, inner, I3 = jno.np.sym, jno.np.grad, jno.np.trace, jno.np.inner, jno.np.identity(3)
    g = 0.05
    d = _box()
    u, phi = d.fem_symbols(value_shape=(3,))
    co, cc = d.variable("interior", split=True), d.variable("bdry", split=True)
    X = [co[0], co[1], co[2]]
    eps = lambda w: sym(grad(w, X))
    bcs = [u(cc[0], cc[1], cc[2])[0] - g * cc[1], u(cc[0], cc[1], cc[2])[1] - 0.0, u(cc[0], cc[1], cc[2])[2] - 0.0]
    sig = LAM * trace(eps(u)) * I3 + 2 * MU * eps(u)  # tensor stress (uses I)
    sol_tensor = np.asarray(jno.fem([inner(sig, eps(phi), 2), *bcs]).solve()).reshape(-1, 3)
    scalar = LAM * trace(eps(u)) * trace(eps(phi)) + 2 * MU * inner(eps(u), eps(phi), 2)
    sol_scalar = np.asarray(jno.fem([scalar, *bcs]).solve()).reshape(-1, 3)
    assert np.abs(sol_tensor - sol_scalar).max() < 1e-10


def _svk_virtual_work(u, phi, X):
    """St. Venant-Kirchhoff internal virtual work ∫ (F S):∇δu as a pure trace formula."""
    grad, trace, inner, einsum, I3 = jno.np.grad, jno.np.trace, jno.np.inner, jno.np.einsum, jno.np.identity(3)
    H = lambda w: grad(w, X)  # displacement gradient
    F = I3 + H(u)  # deformation gradient
    Egl = 0.5 * (einsum("...ki,...kj->...ij", F, F) - I3)  # Green-Lagrange ½(FᵀF − I)
    S = LAM * trace(Egl) * I3 + 2 * MU * Egl  # 2nd Piola-Kirchhoff
    P = einsum("...ij,...jk->...ik", F, S)  # 1st PK = F S
    return inner(P, H(phi), 2)


def test_finite_strain_large_stretch_patch_test():
    """u = (0.2 x, 0, 0) prescribed on the whole boundary (20% stretch, far outside small strain): the
    finite-strain solve must recover it exactly — a large-deformation patch test."""
    delta = 0.2
    d = _box()
    u, phi = d.fem_symbols(value_shape=(3,))
    co, cc = d.variable("interior", split=True), d.variable("bdry", split=True)
    X = [co[0], co[1], co[2]]
    fem = jno.fem(
        [
            _svk_virtual_work(u, phi, X),
            u(cc[0], cc[1], cc[2])[0] - delta * cc[0],
            u(cc[0], cc[1], cc[2])[1] - 0.0,
            u(cc[0], cc[1], cc[2])[2] - 0.0,
        ]
    )
    assert not fem.is_linear
    sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(max_steps=40, rtol=1e-10, atol=1e-12))).reshape(-1, 3)
    pts = np.asarray(fem.field_points[0])[:, :3]
    assert np.abs(sol[:, 0] - delta * pts[:, 0]).max() < 1e-8


def test_finite_strain_reduces_to_linear_elasticity_at_small_strain():
    """A cantilever sheared a TINY amount: finite strain and linear elasticity must agree to O(strain²)."""
    shear = 2e-4  # small
    d = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=0.5).domain()
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
    lin = LAM * trace(eps(u)) * trace(eps(phi)) + 2 * MU * inner(eps(u), eps(phi), 2)
    sol_lin = np.asarray(jno.fem([lin, *bcs]).solve()).reshape(-1, 3)
    sol_fs = np.asarray(
        jno.fem([_svk_virtual_work(u, phi, X), *bcs]).solve(
            # atol=1e-14 was BELOW the achievable residual on CPU (it floors at ~1.14e-14 against a
            # 1.07e-14 gate), so Newton raised a spurious non-convergence there while passing on GPU.
            # This is a SOLVER tolerance, not the property under test: the assertion below is the
            # physics (rel < 1e-3), for which 1e-12 is still four orders tighter than needed.
            nonlinear=jno.solve.newton(max_steps=30, rtol=1e-12, atol=1e-12)
        )
    ).reshape(-1, 3)
    rel = np.abs(sol_fs - sol_lin).max() / max(np.abs(sol_lin).max(), 1e-30)
    assert rel < 1e-3  # finite strain -> linear as strain -> 0


# ==================================================================================================
# `jno.np.diff(psi, F)` — the 1st Piola-Kirchhoff stress BY AUTODIFF, so a hyperelastic material is
# written as its stored energy rather than a hand-derived stress. The oracle is the hand-written
# formula already verified above: P = ∂ψ/∂F must reproduce it, not merely agree qualitatively.
# ==================================================================================================
def _kinematics(u, X):
    grad, einsum, I3 = jno.np.grad, jno.np.einsum, jno.np.identity(3)
    F = I3 + grad(u, X)  # ONE node — `diff` matches its `wrt` by identity
    C = einsum("...ki,...kj->...ij", F, F)
    return F, C, I3


def _clamped_stretch(d, u, delta=0.12):
    cc = d.variable("bdry", split=True)
    return [
        u(cc[0], cc[1], cc[2])[0] - delta * cc[0],
        u(cc[0], cc[1], cc[2])[1] - 0.0,
        u(cc[0], cc[1], cc[2])[2] - 0.0,
    ]


def test_diff_reproduces_the_hand_written_svk_stress():
    """∂ψ_SVK/∂F, by autodiff through the trace, must equal the hand-derived P = F S exactly."""
    trace, inner, einsum, grad = jno.np.trace, jno.np.inner, jno.np.einsum, jno.np.grad
    d = _box()
    u, phi = d.fem_symbols(value_shape=(3,))
    co = d.variable("interior", split=True)
    X = [co[0], co[1], co[2]]
    F, C, I3 = _kinematics(u, X)
    Egl = 0.5 * (C - I3)

    psi = LAM / 2 * trace(Egl) ** 2 + MU * inner(Egl, Egl, 2)  # SVK stored energy
    P_auto = jno.np.diff(psi, F)  # <- the feature
    P_hand = einsum("...ij,...jk->...ik", F, LAM * trace(Egl) * I3 + 2 * MU * Egl)

    bcs = _clamped_stretch(d, u)
    fem_a = jno.fem([inner(P_auto, grad(phi, X), 2), *bcs])
    fem_h = jno.fem([inner(P_hand, grad(phi, X), 2), *bcs])
    assert not fem_a.is_linear  # the Diff marker must keep the form on the residual path

    # Compare the RESIDUALS at a common, non-trivial state — a far tighter check than comparing solves.
    rng = np.random.default_rng(0)
    uk = np.asarray(rng.normal(scale=0.02, size=fem_a.dofs))
    ra, rh = np.asarray(fem_a.residual(uk)), np.asarray(fem_h.residual(uk))
    assert np.abs(ra - rh).max() < 1e-11 * max(1.0, np.abs(rh).max()), f"max |Δresidual| = {np.abs(ra - rh).max():.3e}"


def test_diff_reproduces_the_hand_written_neo_hookean_stress():
    """P = μ(F − F⁻ᵀ) + λ ln(J) F⁻ᵀ, the form documented in docs/fem/formulations.md, recovered from its energy."""
    trace, inner, grad, det, inv = jno.np.trace, jno.np.inner, jno.np.grad, jno.np.det, jno.np.inv
    d = _box()
    u, phi = d.fem_symbols(value_shape=(3,))
    co = d.variable("interior", split=True)
    X = [co[0], co[1], co[2]]
    F, C, I3 = _kinematics(u, X)
    J = det(F)

    psi = MU / 2 * (trace(C) - 3.0) - MU * jno.np.log(J) + LAM / 2 * jno.np.log(J) ** 2
    P_auto = jno.np.diff(psi, F)
    Finv_T = jno.np.transpose(inv(F), (0, 2, 1))
    P_hand = MU * (F - Finv_T) + LAM * jno.np.log(J) * Finv_T

    bcs = _clamped_stretch(d, u)
    fem_a = jno.fem([inner(P_auto, grad(phi, X), 2), *bcs])
    fem_h = jno.fem([inner(P_hand, grad(phi, X), 2), *bcs])
    rng = np.random.default_rng(1)
    uk = np.asarray(rng.normal(scale=0.02, size=fem_a.dofs))
    ra, rh = np.asarray(fem_a.residual(uk)), np.asarray(fem_h.residual(uk))
    assert np.abs(ra - rh).max() < 1e-11 * max(1.0, np.abs(rh).max()), f"max |Δresidual| = {np.abs(ra - rh).max():.3e}"


def test_diff_energy_form_solves_the_large_stretch_patch_test():
    """End to end: the energy spelling must recover a 20% stretch exactly, like the hand-written one."""
    trace, inner, grad = jno.np.trace, jno.np.inner, jno.np.grad
    delta = 0.2
    d = _box()
    u, phi = d.fem_symbols(value_shape=(3,))
    co = d.variable("interior", split=True)
    X = [co[0], co[1], co[2]]
    F, C, I3 = _kinematics(u, X)
    Egl = 0.5 * (C - I3)
    psi = LAM / 2 * trace(Egl) ** 2 + MU * inner(Egl, Egl, 2)
    fem = jno.fem([inner(jno.np.diff(psi, F), grad(phi, X), 2), *_clamped_stretch(d, u, delta)])
    sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(max_steps=40, rtol=1e-10, atol=1e-12))).reshape(-1, 3)
    pts = np.asarray(fem.field_points[0])[:, :3]
    assert np.abs(sol[:, 0] - delta * pts[:, 0]).max() < 1e-8


def test_diff_raises_when_wrt_does_not_occur_in_the_energy():
    """`substitute` matches by IDENTITY, so an inline-rebuilt `wrt` matches nothing and would silently
    differentiate to zero. That must be a loud error at trace-construction time, not a zero stress."""
    trace, grad = jno.np.trace, jno.np.grad
    d = _box()
    u, _ = d.fem_symbols(value_shape=(3,))
    co = d.variable("interior", split=True)
    X = [co[0], co[1], co[2]]
    F, C, I3 = _kinematics(u, X)
    psi = MU * trace(C)
    with pytest.raises(ValueError, match="does not occur"):
        jno.np.diff(psi, I3 + grad(u, X))  # a FRESH node, not the `F` inside `psi`
