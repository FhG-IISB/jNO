"""Axisymmetric FEM, written **by hand** in the weak form — jNO applies no ring measure for you.

A meridional ``(r, z)`` mesh is just a 2-D mesh; what makes a form axisymmetric is the measure the
user writes into it. For a **scalar** field that is exactly the Cartesian integrand times ``2πr``:

    ∫ (∂ᵣu ∂ᵣv + ∂zu ∂zv) 2πr dr dz     is the weak form of   (1/r)(r u_ᵣ)_ᵣ + u_zz

so ``k * (ui.x*vi.x + ui.y*vi.y) * (2*pi*r)`` is complete and needs no library support. These tests
pin that pattern against closed-form results, on oracles where the Cartesian and axisymmetric answers
genuinely differ so a test cannot pass by accident:

* **Radial conduction** in an annulus is logarithmic, ``T(r) = T_a + ΔT ln(r/a)/ln(b/a)``
  (Carslaw & Jaeger, *Conduction of Heat in Solids*, 2nd ed., §7.2), where the plain 2-D form on the
  same mesh gives a straight line.
* **Total heat flow** ``Q = 2πk ΔT / ln(b/a)`` per unit height — only right if the measure is on the
  volume term.
* **A ring source** integrates to the revolved volume, not the meridional area.
* **A boundary flux** integrates over the ring area ``2πb·h``, not the edge length.

For a **vector** field the ring measure alone is NOT the axisymmetric form: the hoop strain
``ε_θθ = u_r/r`` has no 2-D counterpart and must be written out explicitly too. That is why jNO does
not offer to do this weighting automatically — it would be right for scalars and quietly wrong for
vectors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

TWO_PI = 2.0 * np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _annulus_domain(a, b, h, mesh_size):
    """The meridional rectangle [a, b] x [0, h] of an annular cylinder — an ordinary 2-D domain."""
    pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
    from shapely.geometry import box

    d = jno.domain(box(a, 0.0, b, h), mesh_size=mesh_size)
    d.tag("inner", lambda x, y: jnp.abs(x - a) < 1e-9)  # jnp: jno traces tag predicates
    d.tag("outer", lambda x, y: jnp.abs(x - b) < 1e-9)
    return d


def _solve_radial(a, b, h, mesh_size, T_a, T_b, k=1.0, axisymmetric=True):
    """Steady conduction across the annulus. ``axisymmetric`` decides whether the user writes the
    ring measure into the form — the mesh is identical either way.

    Each call builds its own domain: a domain carries per-build state, so driving two ``jno.fem``
    builds off one instance is not supported."""
    d = _annulus_domain(a, b, h, mesh_size)
    u, v = d.fem_symbols()
    r, z, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=r, y=z), v.bind(x=r, y=z)
    ra, za, _ = d.variable("inner", split=True)
    rb, zb, _ = d.variable("outer", split=True)

    grad = k * (ui.x * vi.x + ui.y * vi.y)
    # the whole difference, written in the form (multiplying by a bare 1.0 instead would leave a
    # degenerate node the term classifier no longer reads as a volume term)
    vol = grad * (TWO_PI * r) if axisymmetric else grad
    fem = jno.fem([vol, u(ra, za) - T_a, u(rb, zb) - T_b])
    return np.asarray(fem.solve()).reshape(-1), np.asarray(d.mesh.points)[:, :2]


def test_handwritten_ring_measure_gives_logarithmic_radial_conduction():
    """Same mesh, same terms — only the ``2πr`` the user writes decides log vs linear."""
    a, b, T_a, T_b = 1.0, 3.0, 100.0, 0.0
    T_axi, pts = _solve_radial(a, b, 0.6, 0.09, T_a, T_b, axisymmetric=True)
    rr = pts[:, 0]
    log_exact = T_a + (T_b - T_a) * np.log(rr / a) / np.log(b / a)
    lin_exact = T_a + (T_b - T_a) * (rr - a) / (b - a)
    assert np.abs(T_axi - log_exact).max() / (T_a - T_b) < 5e-3, "hand-written ring measure must give ln(r)"
    assert np.abs(T_axi - lin_exact).max() / (T_a - T_b) > 5e-2, "premise: log and linear must differ"

    T_cart, pts_c = _solve_radial(a, b, 0.6, 0.09, T_a, T_b, axisymmetric=False)
    lin_c = T_a + (T_b - T_a) * (pts_c[:, 0] - a) / (b - a)
    assert np.abs(T_cart - lin_c).max() / (T_a - T_b) < 5e-3, "without it the same form stays Cartesian"


def test_handwritten_ring_measure_gives_the_closed_form_heat_flow():
    """``Q = 2πk ΔT / ln(b/a)`` per unit height, recovered as the reaction on the inner surface."""
    a, b, h, k, T_a, T_b = 1.0, 3.0, 0.5, 7.0, 100.0, 0.0
    T, _ = _solve_radial(a, b, h, 0.07, T_a, T_b, k=k, axisymmetric=True)

    d = _annulus_domain(a, b, h, 0.07)  # identical mesh, fresh domain for the pure-Neumann operator
    u, v = d.fem_symbols()
    r, z, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=r, y=z), v.bind(x=r, y=z)
    pure = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) * (TWO_PI * r)])  # no Dirichlet rows -> true flux
    A = pure.operator[0]
    A = A.todense() if hasattr(A, "todense") else np.asarray(A)
    inner = np.abs(np.asarray(d.mesh.points)[:, 0] - a) < 1e-9
    Q = float((np.asarray(A) @ T)[inner].sum()) / h

    Q_exact = TWO_PI * k * (T_a - T_b) / np.log(b / a)
    assert abs(Q - Q_exact) / Q_exact < 1e-2, f"Q {Q:.3f} vs closed form {Q_exact:.3f}"


def test_handwritten_ring_measure_integrates_the_revolved_volume():
    """A constant body source assembles to ``f·π(b²−a²)·h``, the revolved volume."""
    a, b, h, f0 = 1.0, 2.5, 0.4, 3.0
    d = _annulus_domain(a, b, h, 0.06)
    u, v = d.fem_symbols()
    r, z, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=r, y=z), v.bind(x=r, y=z)

    load = np.asarray(jno.fem([(ui.x * vi.x + ui.y * vi.y - f0 * vi) * (TWO_PI * r)]).operator[1]).reshape(-1)
    exact = f0 * np.pi * (b**2 - a**2) * h
    assert abs(load.sum() - exact) / exact < 1e-9, f"revolved source {load.sum():.6f} vs {exact:.6f}"

    cart = np.asarray(jno.fem([ui.x * vi.x + ui.y * vi.y - f0 * vi]).operator[1]).reshape(-1)
    assert abs(cart.sum() - f0 * (b - a) * h) / (f0 * (b - a) * h) < 1e-9, "unweighted stays Cartesian"


def test_handwritten_ring_measure_on_a_boundary_flux():
    """A flux on the outer wall integrates over the ring area ``g·2πb·h``, not the edge length ``g·h``."""
    a, b, h, g = 1.0, 2.0, 0.5, 4.0
    d = _annulus_domain(a, b, h, 0.06)
    u, v = d.fem_symbols()
    r, z, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=r, y=z), v.bind(x=r, y=z)
    rb, zb, _ = d.variable("outer", split=True)

    # the boundary term carries its OWN 2*pi*r, evaluated on the boundary coordinate
    fem = jno.fem([(ui.x * vi.x + ui.y * vi.y) * (TWO_PI * r), -g * v.bind(x=rb, y=zb) * (TWO_PI * rb)])
    load = np.asarray(fem.operator[1]).reshape(-1)
    exact = g * TWO_PI * b * h
    assert abs(load.sum() - exact) / exact < 1e-9, f"ring flux {load.sum():.6f} vs exact {exact:.6f}"


def test_enclosure_axisymmetric_load_is_per_full_revolution():
    """``Enclosure.load`` with ``axisymmetric=True`` is per full revolution (measure ``2πr ds``), so a
    hand-written form it is added to must carry the same factor. Locks the convention numerically."""
    from jno.domain.enclosure import Enclosure

    R, n, q0 = 0.7, 6, 3.0
    rr = np.linspace(0.0, R, n + 1)
    e0 = np.c_[rr[:-1], np.zeros(n)]
    e1 = np.c_[rr[1:], np.zeros(n)]
    areas = TWO_PI * (0.5 * (e0[:, 0] + e1[:, 0])) * np.linalg.norm(e1 - e0, axis=1)
    gap = Enclosure(
        domain=None,
        tags=["disc"],
        F=np.zeros((n, n)),
        elements=np.c_[np.arange(n), np.arange(1, n + 1)],
        element_tags=np.array(["disc"] * n, dtype=object),
        areas=areas,
        normals=np.tile([0.0, 1.0], (n, 1)),
        midpoints=0.5 * (e0 + e1),
        axisymmetric=True,
        endpoints=(e0, e1),
    )
    load = np.asarray(gap.load(jnp.full(n, q0), size=n + 1))
    assert abs(load.sum() - q0 * np.pi * R**2) < 1e-12 * q0 * np.pi * R**2, (
        "the scattered load must equal flux x the full revolved disc area"
    )
