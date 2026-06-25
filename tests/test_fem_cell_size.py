"""Tests for the `dom.cell_size` symbol (element size h, for SUPG/GLS stabilization).

Covers: it resolves to the true mesh size at quad points, an h-scaled stabilization term
classifies as global + linear, and adding a SUPG term actually changes the assembled operator.
"""

import jax
import numpy as np
import pytest

import jno
from jno.utils.solver.term_kind import classify_term

dense = lambda A: np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)  # noqa: E731


@pytest.fixture
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _square(n):
    return jno.domain(
        constructor=jno.domain.equi_distant_rect(x_range=(0.0, 1.0), y_range=(0.0, 1.0), nx=n, ny=n),
        compute_mesh_connectivity=False,
    )


def test_cell_size_resolves_to_mesh_size(_x64):
    """∫_Ω h dΩ = Σ_i (∫ h φ_i) ≈ the element size (1/N) on the unit square with a uniform mesh."""
    n = 8
    dom = _square(n)
    u, v = dom.fem_symbols()
    xi, yi = dom.variable("interior", split=True)[:2]
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    h = dom.cell_size
    fem = jno.fem([ui * vi - h * vi])  # M u = b, with load b = ∫ h φ
    b = dense(fem.b).reshape(-1)
    assert abs(float(b.sum()) - 1.0 / n) < 1e-3 / n


def test_supg_term_classifies_global_and_linear(_x64):
    """An h-scaled grad·grad stabilization term has spatial gradients on both sides → global, linear."""
    dom = _square(6)
    u, v = dom.fem_symbols()
    xi, yi = dom.variable("interior", split=True)[:2]
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    h = dom.cell_size
    supg = h * (ui.x * vi.x + ui.y * vi.y)
    k = classify_term(dom, supg)
    assert not k.is_local
    assert k.linear


def test_supg_changes_the_assembled_operator(_x64):
    """Adding a streamline (SUPG) stabilization term must modify the assembled advection-diffusion operator."""
    n = 8
    dom = _square(n)
    dom.tag("bnd", lambda x, y: (x < 1e-6) | (x > 1 - 1e-6) | (y < 1e-6) | (y > 1 - 1e-6))
    u, v = dom.fem_symbols()
    xi, yi = dom.variable("interior", split=True)[:2]
    xb, yb = dom.variable("bnd", split=True)[:2]
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    h = dom.cell_size
    nu = 1e-3
    bx, by = 1.0, 0.0
    adv = bx * ui.x + by * ui.y
    galerkin = adv * vi + nu * (ui.x * vi.x + ui.y * vi.y)
    supg = (h / 2.0) * adv * (bx * vi.x + by * vi.y)

    A_galerkin = dense(jno.fem([galerkin, u(xb, yb) - 0.0]).A)
    A_supg = dense(jno.fem([galerkin + supg, u(xb, yb) - 0.0]).A)
    assert not np.allclose(A_galerkin, A_supg)  # the stabilization changed the operator
    assert np.abs(A_supg - A_galerkin).max() > 0
