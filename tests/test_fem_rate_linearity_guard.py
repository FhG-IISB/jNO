"""A transient term must be LINEAR in ``u_t``; one that is not is refused, not marched wrongly.

A first-order march writes each term as a mass action ``M(u) u_t``: the constant-mass path takes ``M`` from
the term's derivative at ``u_t = 0``, the state-dependent path replaces ``u_t`` by ``u - u_prev`` and
divides the whole action by the step once. A term quadratic in ``u_t`` -- the ``u_t⊗u_t`` piece of a
residual-based VMS Reynolds stress -- came out scaled by ``dt`` instead of ``dt²``. Measured on
``u_t + a u_t² + u = 0``: the march matched the mis-scaled recursion to every digit, not backward Euler.

Oracles: the quadratic and the wrapped (``sin``) rate are refused by name; terms linear in the rate --
a state-dependent coefficient, a rate inside a contraction -- still build and march.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _square():
    d = jno.shape.rect(0, 0, 1, 1).structured(n=4).domain(time=(0.0, 0.3, 4))
    d.tag("wall", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    return d


def _scalar(term_of):
    d = _square()
    u, v = d.fem_symbols(names=("u", "v"))
    x, y, t = d.variable("interior", split=True)
    xw, yw, _ = d.variable("wall", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), v.bind(x=x, y=y, t=t)
    c = d.variable("initial", split=True)
    return jno.fem([term_of(ui, vi) + ui * vi, u(xw, yw) - 0.0, u(*c) - 0.3])


@pytest.mark.parametrize(
    "term, kind",
    [
        (lambda u, v: u.t * v + 0.5 * u.t * u.t * v, "quadratic"),
        (lambda u, v: jno.np.sin(u.t) * v, "not linear"),
        (lambda u, v: (u.t * u.t / (1.0 + u * u)) * v, "quadratic"),
    ],
    ids=["u_t squared", "u_t inside sin", "u_t squared over a coefficient"],
)
def test_a_term_not_linear_in_the_rate_is_refused(term, kind):
    with pytest.raises(NotImplementedError, match=kind):
        _scalar(term)


def test_terms_linear_in_the_rate_still_march():
    fem = _scalar(lambda u, v: (1.0 + u * u) * u.t * v)  # a state-dependent mass coefficient
    traj = np.asarray(fem.solve().fn())
    assert np.isfinite(traj).all() and np.abs(traj).max() > 0.05

    d = _square()
    w, z = d.fem_symbols(value_shape=(2,), names=("w", "z"))
    x, y, t = d.variable("interior", split=True)
    xw, yw, _ = d.variable("wall", split=True)
    wi, zi = w.bind(x=x, y=y, t=t), z.bind(x=x, y=y, t=t)
    gz = jno.np.grad(z, [x, y])
    dot = lambda a, b: jno.np.inner(a, b, n_contract=1)  # noqa: E731
    c = d.variable("initial", split=True)
    fem = jno.fem(
        [
            dot(wi.t, zi) + 0.1 * dot(dot(gz, wi), wi.t) + dot(wi, zi),  # the rate inside a contraction
            w(xw, yw) - 0.0,
            w(*c)[0] - 0.2,
            w(*c)[1] - 0.1,
        ]
    )
    assert np.isfinite(np.asarray(fem.solve().fn())).all()
