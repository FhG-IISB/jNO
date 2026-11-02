"""Material laws: a property attached as a function of a field and evaluated where it is used, ``d.k(T)``.

Oracle: steady conduction through a slab 0 < x < 1, T = 0 and 1 at its ends, with k = 1 + b T in the left half.
The Kirchhoff transform makes the integral of k over T linear in x inside that half; with a constant k2 in the right
half the flux q is the same through both, which fixes the interface temperature Tm from
    Tm + b Tm^2 / 2 = q / 2   and   k2 (1 - Tm) = q / 2,
so the profile is exact: (-1 + sqrt(1 + 2 b q x)) / b on the left, linear on the right.
"""

import numpy as np
import pytest
from jax import config

import jno

config.update("jax_enable_x64", True)

B, K2 = 2.0, 3.0


def _slab(k_left, k_right=K2, names=("T", "v")):
    R = jno.shape.rect
    d = (
        (R(0, 0, 0.5, 0.1).name("hot_half").attach(k=k_left) + R(0, 0, 1, 0.1).name("cold_half").attach(k=k_right))
        .sized(0.02)
        .domain()
    )
    T, v = d.fem_symbols(names=names)
    x, y, _ = d.variable("interior", split=True)
    return d, T, v, x, y


def _solve(d, T, v, x, y, k):
    Ti, vi = T.bind(x=x, y=y), v.bind(x=x, y=y)
    fem = jno.fem(
        [
            k * (Ti.d(x) * vi.d(x) + Ti.d(y) * vi.d(y)),
            T(*d.variable("left", split=True)[:2]) - 0.0,
            T(*d.variable("right", split=True)[:2]) - 1.0,
        ]
    )
    if fem._mode != "nonlinear":  # numbers / functions of position only: a linear problem
        return np.asarray(fem.solve(linear=jno.solve.lu())), fem
    return np.asarray(
        fem.solve(x0=np.full(len(d.mesh.points), 0.5), nonlinear=jno.solve.newton(direct=True, atol=1e-10, rtol=0.0))
    ), fem


def _exact(x, k_right=K2):
    # Tm + B Tm^2 / 2 = k_right (1 - Tm)  ->  B/2 Tm^2 + (1 + k_right) Tm - k_right = 0
    a, b_, c = B / 2, 1 + k_right, -k_right
    Tm = (-b_ + np.sqrt(b_**2 - 4 * a * c)) / (2 * a)
    q = 2 * k_right * (1 - Tm)
    return np.where(x <= 0.5, (-1 + np.sqrt(1 + 2 * B * q * x)) / B, Tm + (1 - Tm) * (x - 0.5) / 0.5)


def test_a_law_and_a_number_in_two_regions_match_the_kirchhoff_profile():
    d, T, v, x, y = _slab(lambda T: 1.0 + B * T)
    sol, fem = _solve(d, T, v, x, y, d.k(T))
    assert fem._mode == "nonlinear"  # k depends on T: Newton, with dk/dT in the tangent
    p = np.asarray(d.mesh.points)[:, 0]
    assert np.abs(sol - _exact(p)).max() < 2e-4


def test_a_datasheet_object_is_a_law():
    """A callable object (a table read from a datasheet) works like a function; here it holds the same line."""

    class Table:
        def __init__(self, t, k):
            self.t, self.k = t, k

        def __call__(self, T):  # piecewise linear, constant beyond the ends, written with jno.np
            out = self.k[0]
            for i in range(1, len(self.t)):
                ramp = jno.np.minimum(jno.np.maximum((T - self.t[i - 1]) / (self.t[i] - self.t[i - 1]), 0.0), 1.0)
                out = out + (self.k[i] - self.k[i - 1]) * ramp
            return out

    d, T, v, x, y = _slab(Table([0.0, 0.5, 1.0], [1.0, 1.0 + 0.5 * B, 1.0 + B]))
    sol, _ = _solve(d, T, v, x, y, d.k(T))
    p = np.asarray(d.mesh.points)[:, 0]
    assert np.abs(sol - _exact(p)).max() < 2e-4


def test_constant_properties_can_be_called_too():
    """d.k(T) on numbers only is d.k: one spelling whether or not a region has a law."""
    d, T, v, x, y = _slab(K2)
    sol, _ = _solve(d, T, v, x, y, d.k(T))
    p = np.asarray(d.mesh.points)[:, 0]
    np.testing.assert_allclose(sol, p, atol=1e-9)  # uniform k: linear profile


def test_a_law_used_bare_raises_with_the_fix():
    d, T, v, x, y = _slab(lambda T: 1.0 + B * T)
    with pytest.raises(TypeError, match=r"d\.k\(T\)"):
        _ = d.k * 2.0


def test_parameters_named_after_fields_make_a_law_even_with_coordinate_arity():
    """lambda T, c: ... has two parameters, like (x, y) on a 2-D domain. Because T is a declared field it is a
    law (bare use raises), not silently a function of position."""
    d, T, v, x, y = _slab(lambda T, c: 1.0 + B * T)
    with pytest.raises(TypeError, match="material law"):
        _ = d.k + 0.0
    with pytest.raises(TypeError, match="could not be called with 1 argument"):
        d.k(T)


def test_a_function_of_position_keeps_its_meaning():
    """lambda x, y: ... (not field names) is still a function of position, read bare as before."""
    d, T, v, x, y = _slab(lambda x_, y_: 1.0 + 0.0 * x_, k_right=1.0)
    sol, _ = _solve(d, T, v, x, y, d.k)
    p = np.asarray(d.mesh.points)[:, 0]
    np.testing.assert_allclose(sol, p, atol=1e-9)


def test_a_trainable_coefficient_in_a_law_is_fitted_through_the_solve():
    """k(T) = a (1 + b T) with [a, b] a jno.np.parameter, fitted to the exact Kirchhoff profile of b = 2 (uniform
    law on the whole slab). b is recovered; a is not identifiable from T with both ends held (a uniform scale of k
    leaves T unchanged), so it must stay where it started."""
    import jax
    import jax.numpy as jnp
    import optax

    from jno.utils.adaptive.callbacks import Callback

    theta = jno.np.parameter((2,), name="theta_law")
    theta.dtype(jnp.float64)
    theta.initialize(lambda key, shape, dtype=jnp.float64: jnp.asarray([1.5, 0.5], dtype))
    theta.optimizer(optax.lbfgs())
    law = lambda T: theta[0] * (1.0 + theta[1] * T)  # noqa: E731
    R = jno.shape.rect
    d = (
        (R(0, 0, 0.5, 0.1).name("hot_half").attach(k=law) + R(0, 0, 1, 0.1).name("cold_half").attach(k=law))
        .sized(0.03)
        .domain()
    )
    T, v = d.fem_symbols(names=("T", "v"))
    x, y, _ = d.variable("interior", split=True)
    Ti, vi = T.bind(x=x, y=y), v.bind(x=x, y=y)
    p = np.asarray(d.mesh.points)[:, 0]
    T_true = (-1 + np.sqrt(1 + 2 * B * p * (1 + B / 2))) / B
    fem = jno.fem(
        [
            d.k(T) * (Ti.d(x) * vi.d(x) + Ti.d(y) * vi.d(y)),
            T(*d.variable("left", split=True)[:2]) - 0.0,
            T(*d.variable("right", split=True)[:2]) - 1.0,
        ]
    )
    T_h = fem.solve(x0=np.full(len(p), 0.5), nonlinear=jno.solve.newton(direct=True, atol=1e-12, rtol=0.0))

    class Last(Callback):
        def on_epoch_end(self, trainable, total_loss, **_):
            self.value = np.concatenate([np.ravel(leaf) for leaf in jax.tree_util.tree_leaves(trainable)])
            return False

    last = Last()
    jno.core([(T_h - T_true).mse]).solve(30, callbacks=[last])
    a, b = last.value
    assert b == pytest.approx(B, rel=1e-3)
    assert a == pytest.approx(1.5, rel=1e-9)
