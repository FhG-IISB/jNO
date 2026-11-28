"""Design gradients of ``jno.rcwa`` against central finite differences, for every Fourier factorization.

The vector factorizations build a tangent vector field from the permittivity grid and use it to assemble the
transverse permittivity matrix. That field follows the interfaces, so it depends on the design: a gradient that
treats it as a constant misses a term whenever the design moves an edge (measured before the fix: 37 % off for
``JONES_DIRECT_FOURIER``, 4-12 % for ``POL``/``NORMAL``, while ``FFT``, which has no tangent field, was right).
The degenerate inputs of the tangent-field construction (a grid without any edge, a 1-D grating) must give finite
gradients too.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """These tests run in float64. The session default is x64-off (see tests/conftest.py), and this
    flag is process-wide -- save/restore keeps it from leaking to whatever module runs next."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


from jno.rcwa import Rcwa  # noqa: E402

FORMULATIONS = [
    "FFT",
    "JONES_DIRECT",
    "JONES_DIRECT_FOURIER",
    "JONES",
    "JONES_FOURIER",
    "NORMAL",
    "NORMAL_FOURIER",
    "POL",
    "POL_FOURIER",
]
N = 32
_X = (np.arange(N) + 0.5) / N - 0.5
GX, GY = np.meshgrid(_X, _X, indexing="ij")
D_FILM, EPS_SUB = 0.3, 2.25


def _transmission(formulation, grid0, orders=25, k_in=(0.4, 0.1)):
    """``T(grid)`` for one patterned film on a substrate, at an oblique, non-symmetric incidence so that no
    derivative vanishes by symmetry."""
    rc = Rcwa(
        [(np.inf, 1.0), (D_FILM, np.asarray(grid0)), (np.inf, EPS_SUB)],
        period=(0.8, 0.8),
        orders=orders,
        wavelength=1.0,
        k_in=k_in,
        formulation=formulation,
        assume_periodic=True,
        polarization="x",
    )

    def T(grid):
        return rc.solve(layers=[(np.inf, 1.0), (D_FILM, grid), (np.inf, EPS_SUB)]).efficiency("T")

    return T


def _disk(r, w=0.04):
    """A disk of eps 4 in eps 1 with a smooth edge of width ``w``: the radius moves the interface."""
    return 1.0 + 3.0 * jax.nn.sigmoid((r - np.sqrt(GX**2 + GY**2)) / w)


def _bar(a, w=0.04):
    """A 1-D grating (a bar of half-width ``a`` along y): the grid varies in x only."""
    return 1.0 + 3.0 * jax.nn.sigmoid((a - np.abs(GX)) / w)


@pytest.mark.parametrize("formulation", FORMULATIONS)
def test_gradient_in_a_moving_edge_matches_finite_differences(formulation):
    """dT/dr of the disk: autodiff equals a central difference to the truncation error of the difference."""
    T = _transmission(formulation, _disk(0.3))
    f = lambda r: T(_disk(r))  # noqa: E731
    g = float(jax.grad(f)(0.3))
    h = 1e-4
    fd = (float(f(0.3 + h)) - float(f(0.3 - h))) / (2 * h)
    assert abs(fd) > 1e-3, "the case must have a gradient worth comparing"
    assert g == pytest.approx(fd, rel=5e-4), f"{formulation}: autodiff {g} vs finite difference {fd}"


@pytest.mark.parametrize("formulation", ["FFT", "JONES_DIRECT_FOURIER", "POL", "NORMAL_FOURIER"])
def test_gradient_in_the_pixels_matches_finite_differences(formulation):
    """The topology-optimization gradient: dT along a random smooth pixel direction, autodiff against a central
    difference, on a grey design (every pixel is a design variable).

    An off-centre ellipse; the symmetric case has its own test below."""
    rng = np.random.default_rng(0)
    base = np.asarray(1.0 + 3.0 * jax.nn.sigmoid((0.3 - np.sqrt(((GX - 0.05) / 1.2) ** 2 + (GY + 0.03) ** 2)) / 0.08))

    def smooth(a):  # a periodic low-pass, so the direction is resolved by the grid
        f = np.fft.fft2(a)
        k = np.fft.fftfreq(N) * N
        return np.real(np.fft.ifft2(f * np.exp(-(k[:, None] ** 2 + k[None, :] ** 2) / 18.0)))

    direction = smooth(rng.standard_normal((N, N)))
    direction /= np.abs(direction).max()
    T = _transmission(formulation, base)
    grad = np.asarray(jax.grad(T)(jnp.asarray(base)))
    assert np.all(np.isfinite(grad))
    g = float(np.sum(grad * direction))
    h = 1e-4
    fd = (float(T(jnp.asarray(base + h * direction))) - float(T(jnp.asarray(base - h * direction)))) / (2 * h)
    assert abs(fd) > 1e-4, "the case must have a gradient worth comparing"
    assert g == pytest.approx(fd, rel=5e-4), f"{formulation}: autodiff {g} vs finite difference {fd}"


@pytest.mark.parametrize("formulation", FORMULATIONS)
def test_gradient_of_a_1d_grating_is_finite_and_matches_finite_differences(formulation):
    """A grating that varies in x only takes the 1-D branch of the tangent-field construction (the field is set
    by hand there). Its gradient in the bar width must be finite and right."""
    T = _transmission(formulation, _bar(0.2))
    f = lambda a: T(_bar(a))  # noqa: E731
    g = float(jax.grad(f)(0.2))
    h = 1e-4
    fd = (float(f(0.2 + h)) - float(f(0.2 - h))) / (2 * h)
    assert np.isfinite(g)
    assert abs(fd) > 1e-3, "the case must have a gradient worth comparing"
    assert g == pytest.approx(fd, rel=5e-4), f"{formulation}: autodiff {g} vs finite difference {fd}"


@pytest.mark.parametrize("formulation", FORMULATIONS)
def test_gradient_of_a_grid_without_edges_is_finite(formulation):
    """A constant grid has no interface: the gradient field that seeds the tangent field is zero everywhere.
    The pixel gradient must still be finite (never NaN), and its sum is the derivative in the film's
    permittivity, which a central difference on the uniform value gives."""
    base = np.full((N, N), 2.5)
    T = _transmission(formulation, base)
    grad = np.asarray(jax.grad(T)(jnp.asarray(base)))
    assert np.all(np.isfinite(grad)), f"{formulation}: NaN/inf in the pixel gradient of a uniform grid"
    h = 1e-5
    fd = (float(T(jnp.asarray(base + h))) - float(T(jnp.asarray(base - h)))) / (2 * h)
    assert float(grad.sum()) == pytest.approx(fd, rel=1e-4, abs=1e-9)


def _three_layer_T(formulation="JONES_DIRECT_FOURIER"):
    """``T(d1, d2, d3)`` of a uniform film, a patterned film and a tensor film, all thicknesses traced."""
    film = np.asarray(_disk(0.3))
    tensor = (2.5, 0.2, 0.2, 3.0, 2.8)
    rc = Rcwa(
        [(np.inf, 1.0), (0.25, 2.1), (D_FILM, film), (0.2, tensor), (np.inf, EPS_SUB)],
        period=(0.8, 0.8),
        orders=25,
        wavelength=1.0,
        k_in=(0.4, 0.1),
        formulation=formulation,
        assume_periodic=True,
        polarization="x",
    )

    def T(d):
        layers = [(np.inf, 1.0), (d[0], 2.1), (d[1], jnp.asarray(film)), (d[2], tensor), (np.inf, EPS_SUB)]
        return rc.solve(layers=layers).efficiency("T")

    return T, jnp.asarray([0.25, D_FILM, 0.2])


@pytest.mark.parametrize("formulation", ["FFT", "JONES_DIRECT_FOURIER"])
def test_gradient_in_every_layer_thickness_matches_finite_differences(formulation):
    """dT/dd for a uniform, a patterned and a tensor film at once, autodiff under jit against central
    differences: the thickness is an input of the compiled core, the slice and chunk counts are static."""
    T, d0 = _three_layer_T(formulation)
    g = np.asarray(jax.jit(jax.grad(T))(d0))
    h = 1e-5
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        fd = (float(T(d0 + e)) - float(T(d0 - e))) / (2 * h)
        assert abs(fd) > 1e-3, f"layer {i}: the case must have a gradient worth comparing"
        assert g[i] == pytest.approx(fd, rel=1e-6), f"layer {i}: autodiff {g[i]} vs finite difference {fd}"


def test_a_thickness_inside_the_margin_equals_a_fresh_build():
    """Solving a stack at 1.8x its construction thickness gives what a stack built at that thickness gives."""
    T, d0 = _three_layer_T()
    d1 = d0 * jnp.asarray([1.8, 1.8, 0.5])
    film = np.asarray(_disk(0.3))
    fresh = Rcwa(
        [
            (np.inf, 1.0),
            (float(d1[0]), 2.1),
            (float(d1[1]), film),
            (float(d1[2]), (2.5, 0.2, 0.2, 3.0, 2.8)),
            (np.inf, EPS_SUB),
        ],
        period=(0.8, 0.8),
        orders=25,
        wavelength=1.0,
        k_in=(0.4, 0.1),
        assume_periodic=True,
        polarization="x",
    )
    assert float(T(d1)) == pytest.approx(float(fresh.solve().efficiency("T")), rel=1e-11)


@pytest.mark.parametrize("formulation", ["POL", "JONES_DIRECT_FOURIER", "JONES"])
def test_gradient_of_a_symmetric_design_matches_finite_differences(formulation):
    """A centred disk: many pixels tie for the largest permittivity gradient. When the tangent field was scaled by
    that maximum, T had a kink there and autodiff disagreed with central differences by 1e-4 to 1.4e-3 at every
    step size (git-bug a3cd6ef). With the smooth p-norm scale the two agree like on any other design."""
    rng = np.random.default_rng(1)
    base = np.asarray(1.0 + 3.0 * jax.nn.sigmoid((0.3 - np.sqrt(GX**2 + GY**2)) / 0.08))

    def smooth(a):
        f = np.fft.fft2(a)
        k = np.fft.fftfreq(N) * N
        return np.real(np.fft.ifft2(f * np.exp(-(k[:, None] ** 2 + k[None, :] ** 2) / 18.0)))

    direction = smooth(rng.standard_normal((N, N)))
    direction /= np.abs(direction).max()
    T = _transmission(formulation, base)
    g = float(np.sum(np.asarray(jax.grad(T)(jnp.asarray(base))) * direction))
    for h in (1e-4, 1e-5):
        fd = (float(T(jnp.asarray(base + h * direction))) - float(T(jnp.asarray(base - h * direction)))) / (2 * h)
        assert abs(fd) > 1e-4
        assert g == pytest.approx(fd, rel=2e-6), f"{formulation} h={h}: autodiff {g} vs finite difference {fd}"
