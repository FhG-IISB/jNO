"""The 1-D spectral bases and the tensor grid behind ``dom.unknown(space="spectral")``.

Oracles are closed forms: a trigonometric polynomial (Fourier) or a polynomial (Chebyshev, Legendre) of the
space's own degree is differentiated, interpolated, integrated and transformed **exactly** (to round-off);
an analytic non-polynomial field converges geometrically. The dense and FFT paths of the Chebyshev axis are
checked against each other, since the switch between them is a size threshold users never see.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno._spectral.bases import Chebyshev, Fourier, Legendre, axis_from_spec
from jno._spectral.grid import Grid
from jno._spectral.space import SpectralSpace, parse_space


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _err(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# ── Fourier ─────────────────────────────────────────────────────────────────────────────────────────────


class TestFourier:
    def test_nodes_exclude_the_periodic_image(self):
        ax = Fourier(8, 0.0, 2 * np.pi)
        assert ax.nodes[0] == 0.0 and ax.nodes[-1] < 2 * np.pi
        assert np.allclose(np.diff(ax.nodes), 2 * np.pi / 8)

    @pytest.mark.parametrize("n", [16, 17])
    def test_trig_polynomial_derivatives_are_exact(self, n):
        ax = Fourier(n, 0.0, 2.0)  # period 2: k = π m
        x = ax.nodes
        u = np.sin(np.pi * x) + 0.5 * np.cos(3 * np.pi * x)
        du = np.pi * np.cos(np.pi * x) - 1.5 * np.pi * np.sin(3 * np.pi * x)
        d2u = -(np.pi**2) * np.sin(np.pi * x) - 4.5 * np.pi**2 * np.cos(3 * np.pi * x)
        assert _err(ax.diff(u, 1), du) < 1e-12
        assert _err(ax.diff(u, 2), d2u) < 1e-11
        assert _err(ax.diff_matrix(1) @ u, du) < 1e-12

    def test_analytic_periodic_field_to_round_off(self):
        ax = Fourier(32, 0.0, 2 * np.pi)
        x = ax.nodes
        u = np.exp(np.sin(x))
        assert _err(ax.diff(u), np.cos(x) * u) < 1e-13

    def test_interpolation_is_the_band_limited_interpolant(self):
        ax = Fourier(16, -1.0, 1.0)
        f = lambda x: np.cos(np.pi * x) + np.sin(2 * np.pi * x) ** 2  # modes up to 4 < 8
        xq = np.linspace(-1, 1, 37)
        assert _err(ax.interp(f(ax.nodes), xq), f(xq)) < 1e-13
        assert _err(ax.interp_matrix(ax.nodes), np.eye(16)) < 1e-15

    @pytest.mark.parametrize("n0,n1", [(16, 24), (17, 32), (16, 9), (24, 16)])
    def test_resample_pads_exactly_and_truncates_by_projection(self, n0, n1):
        a0, a1 = Fourier(n0, 0.0, 1.0), Fourier(n1, 0.0, 1.0)
        f = lambda x: np.sin(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x)  # modes 1, 2: in every space here
        assert _err(a0.resample(f(a0.nodes), n1), f(a1.nodes)) < 1e-13
        assert _err(a0.to(f(a0.nodes), a1), f(a1.nodes)) < 1e-13
        # the complex path agrees with the real one
        z = jnp.asarray(f(a0.nodes)) * (1 + 0.5j)
        assert _err(a0.resample(z, n1), f(a1.nodes) * (1 + 0.5j)) < 1e-13

    def test_truncation_drops_unresolved_modes(self):
        a0 = Fourier(32, 0.0, 1.0)
        x = a0.nodes
        u = np.sin(2 * np.pi * x) + np.sin(2 * np.pi * 10 * x)  # mode 10 is not in a 16-node space
        assert _err(a0.resample(u, 16), np.sin(2 * np.pi * Fourier(16, 0.0, 1.0).nodes)) < 1e-13

    def test_transform_round_trip_and_coefficients(self):
        ax = Fourier(12, 0.0, 2 * np.pi)
        u = 2.0 + np.cos(ax.nodes) + 3 * np.sin(2 * ax.nodes)
        c = np.asarray(ax.forward(u))
        assert abs(c[0] - 2.0) < 1e-14 and abs(c[1] - 0.5) < 1e-14 and abs(c[2] + 1.5j) < 1e-14
        assert _err(ax.backward(c), u) < 1e-14

    def test_quadrature_is_the_mean(self):
        ax = Fourier(10, 0.0, 3.0)
        assert abs(float(ax.integrate(np.cos(2 * np.pi * ax.nodes / 3.0) + 1.0)) - 3.0) < 1e-14


# ── Chebyshev ───────────────────────────────────────────────────────────────────────────────────────────


class TestChebyshev:
    @pytest.mark.parametrize("rule", ["lobatto", "gauss"])
    def test_polynomial_of_the_space_degree_is_differentiated_exactly(self, rule):
        ax = Chebyshev(14, -2.0, 3.0, rule=rule)
        x = ax.nodes
        p = np.polynomial.Polynomial(np.linspace(1, -1, 14))  # degree 13 = n - 1
        assert _err(ax.diff(p(x), 1), p.deriv(1)(x)) / np.abs(p.deriv(1)(x)).max() < 1e-12
        assert _err(ax.diff(p(x), 2), p.deriv(2)(x)) / np.abs(p.deriv(2)(x)).max() < 1e-11

    def test_lobatto_nodes_ascend_and_include_the_ends(self):
        ax = Chebyshev(9, 1.0, 2.0)
        assert ax.nodes[0] == pytest.approx(1.0) and ax.nodes[-1] == pytest.approx(2.0)
        assert np.all(np.diff(ax.nodes) > 0)

    def test_runge_function_converges_geometrically(self):
        """1/(1+25x²) has poles at ±i/5: the error falls like ρ^-N with ρ = 1/5 + √(1+1/25) (Bernstein
        ellipse), slowly but geometrically -- measured here, across the dense (64) and FFT (128) paths."""
        errs = {}
        for n in (32, 64, 128):
            ax = Chebyshev(n)
            ax.dense_max = 100  # 128 takes the FFT path
            x = ax.nodes
            u = 1.0 / (1.0 + 25 * x * x)
            du = -50 * x / (1.0 + 25 * x * x) ** 2
            errs[n] = _err(ax.diff(u), du)
        rate = np.log(errs[64] / errs[128]) / 64  # e ~ N² ρ^-N: rate = ln ρ - ln 4 / 64
        ln_rho = np.log(0.2 + np.sqrt(1.04))
        assert 0.8 * ln_rho < rate < 1.3 * ln_rho
        assert errs[128] < 1e-7

    @pytest.mark.parametrize("rule,degree", [("lobatto", 9), ("gauss", 9)])
    def test_quadrature_is_exact_on_polynomials(self, rule, degree):
        ax = Chebyshev(10, 0.0, 2.0, rule=rule)
        p = np.polynomial.Polynomial(np.arange(1, degree + 2, dtype=float))
        exact = p.integ()(2.0) - p.integ()(0.0)
        assert abs(float(ax.integrate(p(ax.nodes))) - exact) / abs(exact) < 1e-14

    def test_modal_coefficients_are_chebyshev_coefficients(self):
        ax = Chebyshev(8)
        c_true = np.array([1.0, -0.5, 0.25, 0, 0, 2.0, 0, 0])
        u = np.polynomial.chebyshev.chebval(ax.nodes, c_true)
        assert _err(ax.forward(u), c_true) < 1e-14
        assert _err(ax.backward(c_true), u) < 1e-14

    def test_fft_path_matches_dense_path(self):
        n = 129
        fast = Chebyshev(n, -1.0, 2.0)
        dense = Chebyshev(n, -1.0, 2.0)
        fast.dense_max, dense.dense_max = 64, 10**9
        assert fast._fast() and not dense._fast()
        x = fast.nodes
        u = np.exp(np.sin(2 * x)) + x**3
        assert _err(fast.forward(u), dense.forward(u)) < 1e-13
        assert _err(fast.backward(dense.forward(u)), u) < 1e-13
        du = 2 * np.cos(2 * x) * np.exp(np.sin(2 * x)) + 3 * x**2
        assert _err(fast.diff(u), du) / np.abs(du).max() < 1e-11
        assert _err(fast.diff(u), dense.diff(u)) / np.abs(du).max() < 1e-10
        # complex input splits into real and imaginary parts
        z = jnp.asarray(u) * (1 - 2j)
        assert _err(fast.forward(z), dense.forward(u) * (1 - 2j)) < 1e-12

    def test_interpolation_exact_at_nodes_and_for_polynomials(self):
        ax = Chebyshev(11, -1.0, 1.0)
        assert _err(ax.interp_matrix(ax.nodes), np.eye(11)) < 1e-15
        p = np.polynomial.Polynomial([0.3, 1, -2, 0.5, 0, 1])
        xq = np.linspace(-1, 1, 50)
        assert _err(ax.interp(p(ax.nodes), xq), p(xq)) < 1e-13


# ── Legendre ────────────────────────────────────────────────────────────────────────────────────────────


class TestLegendre:
    def test_gauss_nodes_and_weights_match_numpy(self):
        ax = Legendre(20, -1.0, 1.0, rule="gauss")
        x, w = np.polynomial.legendre.leggauss(20)
        assert _err(ax.nodes, x) < 1e-14 and _err(ax.weights, w) < 1e-14

    @pytest.mark.parametrize("n", [2, 3, 8, 40])
    def test_gll_rule_is_exact_to_degree_2n_minus_3(self, n):
        ax = Legendre(n, 0.0, 1.0)
        assert ax.nodes[0] == 0.0 and ax.nodes[-1] == 1.0
        deg = 2 * n - 3
        x = ax.nodes
        approx = float(ax.integrate(x**deg))
        assert abs(approx - 1.0 / (deg + 1)) < 1e-13
        # nodes are the zeros of P'_{n-1} inside
        if n > 2:
            dP = np.polynomial.legendre.Legendre.basis(n - 1).deriv()
            assert np.abs(dP(2 * x[1:-1] - 1)).max() < 1e-10

    def test_differentiation_and_modal_transform(self):
        ax = Legendre(12, -1.0, 1.0)
        c = np.zeros(12)
        c[[0, 3, 11]] = [1.0, -2.0, 0.5]
        L = np.polynomial.legendre.Legendre(c)
        assert _err(ax.forward(L(ax.nodes)), c) < 1e-13
        assert _err(ax.diff(L(ax.nodes)), L.deriv()(ax.nodes)) / np.abs(L.deriv()(ax.nodes)).max() < 1e-12


def test_axis_from_spec_and_errors():
    assert isinstance(axis_from_spec("fourier", 8, 0, 1), Fourier)
    assert axis_from_spec("legendre", 8, 0, 1, rule="gauss").rule == "gauss"
    with pytest.raises(ValueError, match="unknown basis family"):
        axis_from_spec("hermite", 8, 0, 1)
    with pytest.raises(ValueError, match="b > a"):
        Chebyshev(8, 1.0, 1.0)


# ── tensor grid ─────────────────────────────────────────────────────────────────────────────────────────


def _channel(nx=16, ny=13):
    return Grid([Fourier(nx, 0.0, 2 * np.pi), Chebyshev(ny, -1.0, 1.0)])


class TestGrid:
    def test_points_layout_and_faces(self):
        g = _channel()
        P = g.points
        assert P.shape == (16 * 13, 2)
        assert np.allclose(P[g.face(1, 0), 1], -1.0) and np.allclose(P[g.face(1, 1), 1], 1.0)
        assert np.allclose(P[g.face(0, 0), 0], 0.0)
        with pytest.raises(ValueError, match="periodic"):
            g.face(0, 1)
        assert set(g.boundary) == set(g.face(1, 0)) | set(g.face(1, 1))

    def test_mixed_derivatives_are_exact(self):
        g = _channel()
        x, y = g.points.T
        u = np.sin(2 * x) * y**4 + np.cos(x) * y
        assert _err(g.diff(u, (1, 0)), 2 * np.cos(2 * x) * y**4 - np.sin(x) * y) < 1e-12
        assert _err(g.diff(u, (0, 2)), 12 * np.sin(2 * x) * y**2) < 1e-10
        assert _err(g.diff(u, (1, 1)), 8 * np.cos(2 * x) * y**3 - np.sin(x)) < 1e-11

    def test_vector_fields_keep_their_components(self):
        g = _channel()
        x, y = g.points.T
        u = np.stack([np.sin(x) * y, np.cos(x) * y**2], axis=1)
        d = g.diff(u, (0, 1))
        assert d.shape == u.shape
        assert _err(d, np.stack([np.sin(x), 2 * np.cos(x) * y], axis=1)) < 1e-12

    def test_transfer_between_grids_and_scattered_points(self):
        g = _channel()
        h = Grid([Fourier(24, 0.0, 2 * np.pi), Legendre(9, -1.0, 1.0, rule="gauss")])
        f = lambda x, y: np.sin(x) * (1 + y**3) + np.cos(3 * x) * y
        x, y = g.points.T
        xh, yh = h.points.T
        assert _err(g.to(f(x, y), h), f(xh, yh)) < 1e-12
        rng = np.random.default_rng(0)
        q = np.stack([rng.uniform(0, 2 * np.pi, 300), rng.uniform(-1, 1, 300)], axis=1)
        assert _err(g.at(f(x, y), q, batch=128), f(q[:, 0], q[:, 1])) < 1e-12

    def test_integration_and_transforms(self):
        g = _channel()
        x, y = g.points.T
        assert abs(float(g.integrate(1 + np.cos(x) * y**2 + y**4)) - 2 * np.pi * (2 + 2 / 5)) < 1e-12
        u = np.sin(x) * y**3 + 1.0
        c = g.forward(u)
        assert c.shape == g.shape
        assert _err(g.backward(c), u) < 1e-13


class TestSpectralSpace:
    def test_ties_choose_fourier_and_conflicts_raise(self):
        sp = SpectralSpace([(0, 1), (0, 2)], (8, 9))
        assert not sp.resolved
        assert sp.provisional_grid.shape == (8, 9)
        g = sp.resolve({0})
        assert isinstance(g.axes[0], Fourier) and isinstance(g.axes[1], Chebyshev)
        sp.resolve({0})  # idempotent
        with pytest.raises(ValueError, match="already solved"):
            sp.resolve(set())
        named = SpectralSpace([(0, 1)], 8, families=("fourier",))
        with pytest.raises(ValueError, match="no periodic tie"):
            named.resolve(set())
        leg = SpectralSpace([(0, 1)], 8, families=("legendre",))
        with pytest.raises(ValueError, match="a tied axis is Fourier"):
            leg.resolve({0})

    def test_unresolved_families_raise_with_the_fix(self):
        with pytest.raises(RuntimeError, match="fixed by the problem"):
            SpectralSpace([(0, 1)], 8).families

    def test_user_facing_calls(self):
        sp = SpectralSpace([(0, 2 * np.pi), (-1, 1)], (16, 11))
        sp.resolve({0})
        x, y = sp.nodes.T
        u = np.cos(x) * y**2
        c = sp.coeffs(u)
        assert c.shape == (16, 11)
        assert _err(sp.values(c), u) < 1e-13
        traj = np.stack([u, 2 * u, 3 * u])  # leading axes batch
        assert sp.coeffs(traj).shape == (3, 16, 11)
        assert _err(sp.values(sp.coeffs(traj)), traj) < 1e-13
        assert _err(sp(u, 0.3, 0.5), np.cos(0.3) * 0.25) < 1e-13
        assert sp(u, np.zeros((4, 5)), np.ones((4, 5))).shape == (4, 5)
        fine = sp.refine(u, (32, 21))
        g2 = sp.grid.with_orders((32, 21))
        assert _err(fine, np.cos(g2.points[:, 0]) * g2.points[:, 1] ** 2) < 1e-12
        assert abs(float(sp.mean(1 + 0 * u)) - 1.0) < 1e-14
        assert sp.wavenumbers(0)[1] == pytest.approx(1.0)

    def test_parse_space(self):
        assert parse_space("spectral", 2) == (None, None)
        assert parse_space(("fourier", "legendre"), 2) == ("fourier", "legendre")
        assert parse_space("Lagrange", 2) is None
        with pytest.raises(ValueError, match="names 1 axes"):
            parse_space(("fourier",), 2)
