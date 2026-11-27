"""The never-silent guards of ``Rcwa.solve`` also speak inside ``jax.jit``.

Eagerly, a solve whose efficiencies break energy conservation or are not finite raises ``RcwaError``. Under
``jit`` (every design loop) the efficiencies are tracers; the guard used to step aside there, so an unconverged
truncation returned T > 1 without a word. Inside a trace the check now runs on the host through a debug
callback and warns with the same message (raising from inside a compiled program would surface only as an
opaque runtime error, and would end the optimisation that should see the warning).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings  # noqa: E402

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


from jno.rcwa import Rcwa, RcwaError, RcwaWarning  # noqa: E402

N = 64
_X = (np.arange(N) + 0.5) / N - 0.5
GX, GY = np.meshgrid(_X, _X, indexing="ij")
DISK = np.where(GX**2 + GY**2 < 0.3**2, 4.0, 1.0)  # eps 4 disk of radius 0.3 in a unit cell


def _rc(orders):
    """A thick high-contrast disk layer: at 21 terms (orders=25) the default factorization is far from
    converged and gives T + R = 1.04; at orders=121 it conserves energy."""
    return Rcwa(
        [(np.inf, 1.0), (1.0, DISK), (np.inf, 2.25)],
        period=(1.0, 1.0),
        orders=orders,
        wavelength=1.07,
        assume_periodic=True,
    )


def _jit_T(rc):
    return jax.jit(lambda g: rc.solve(layers=[(np.inf, 1.0), (1.0, g), (np.inf, 2.25)]).efficiency("T"))


def test_eager_solve_raises_when_energy_is_not_conserved():
    with pytest.raises(RcwaError, match="energy not conserved"):
        _rc(25).solve()


def test_jitted_solve_warns_when_energy_is_not_conserved():
    T = _jit_T(_rc(25))
    with pytest.warns(RcwaWarning, match="energy not conserved"):
        t = float(jax.block_until_ready(T(jnp.asarray(DISK))))
    assert t > 1.0  # the value itself is returned unchanged


def test_gradient_of_a_jitted_solve_warns_and_stays_finite():
    rc = _rc(25)
    loss = jax.jit(jax.grad(lambda g: rc.solve(layers=[(np.inf, 1.0), (1.0, g), (np.inf, 2.25)]).efficiency("T")))
    with pytest.warns(RcwaWarning, match="energy not conserved"):
        g = np.asarray(jax.block_until_ready(loss(jnp.asarray(DISK))))
    assert np.all(np.isfinite(g))


def test_converged_jitted_solve_does_not_warn():
    T = _jit_T(_rc(121))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RcwaWarning)
        t = float(jax.block_until_ready(T(jnp.asarray(DISK))))
    assert 0.0 < t < 1.0


def test_vmapped_solve_warns_once_for_the_batch():
    """Under vmap the callback sees the whole batch; one unconverged member is enough to warn."""
    rc = _rc(25)
    grids = jnp.stack([jnp.asarray(DISK), jnp.ones((N, N)) * 2.0])
    T = jax.jit(jax.vmap(lambda g: rc.solve(layers=[(np.inf, 1.0), (1.0, g), (np.inf, 2.25)]).efficiency("T")))
    with pytest.warns(RcwaWarning, match="energy not conserved"):
        jax.block_until_ready(T(grids))


def _solved_square_post(period=2.1):
    g = np.ones((16, 16))
    g[4:12, 4:12] = 3.0
    stack = [(0.5, 2.13), (0.36, g), (0.5, 1.0)]
    return Rcwa(stack, period=(period, period), orders=25, wavelength=1.0, assume_periodic=True).solve()


def test_a_nan_field_with_finite_efficiencies_raises(monkeypatch):
    """#313's case: T and R finite but the reconstructed field NaN. The field readout raises instead of
    returning the NaNs (forced here by a NaN intensity from the reconstruction)."""
    from jno._rcwa import fields as fields_mod

    sol = _solved_square_post()
    real = fields_mod.volume_intensity

    def nan_intensity(*a, **k):
        inten, x, y = real(*a, **k)
        return inten.at[0, 0, 0].set(jnp.nan), x, y

    monkeypatch.setattr(fields_mod, "volume_intensity", nan_intensity)
    for readout in ("field", "field3d"):
        with pytest.raises(RcwaError, match="grazing"):
            getattr(sol, readout)(nx=16, density=20.0)


def test_the_grazing_message_names_the_order():
    """The message names the order whose kz vanishes: with k_in = k0 - 2 pi / P along x, the (1, 0) order lands
    exactly on the light cone of the air substrate (layer 2), kz = 0 there."""
    sol = _solved_square_post()
    eng = sol._eng
    k0 = 2 * np.pi
    eng._prepared_kin = jnp.asarray([k0 - 2 * np.pi / 2.1, 0.0])  # (1, 0): kx = k0 exactly in the air layer
    msg = sol._nonfinite_field_message()
    assert "order (1, 0) in layer 2" in msg, msg


def _film_T(rc, grid):
    return lambda d: rc.solve(layers=[(np.inf, 1.0), (d, grid), (np.inf, 2.25)]).efficiency("T")


def test_a_thickness_beyond_the_build_margin_raises():
    """The slice and chunk counts are sized for up to 2x the construction thickness: beyond, an eager solve
    raises instead of returning an answer of unknown accuracy."""
    rc = _rc(121)
    _film_T(rc, DISK)(1.9)  # inside the margin: fine
    with pytest.raises(RcwaError, match="left the range"):
        _film_T(rc, DISK)(2.5)


def test_a_traced_thickness_beyond_the_build_margin_warns():
    rc = _rc(121)
    T = jax.jit(_film_T(rc, jnp.asarray(DISK)))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RcwaWarning)
        jax.block_until_ready(T(1.5))
    with pytest.warns(RcwaWarning, match="left the range"):
        jax.block_until_ready(T(2.5))
