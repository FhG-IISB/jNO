"""A box ``u.bounds(lo, hi)`` together with a periodic tie ``u(A) - u(B)``.

The two used to meet in a bare JAX shape error (``max got incompatible shapes for broadcasting: (289,),
(274,)``): the box was resolved over the full DOF vector while the tied solve root-finds on the reduced
unknowns ``u = P ũ``. The box is now stated on ``ũ`` -- the full box restricted to the kept DOFs. That is
exact when ``P`` is a weight-1 selection and the bound is the same on both sides of the tie, since every
eliminated DOF then IS a kept one; both conditions are checked when the form is built, and a form that
breaks either is refused by name.

Oracles:

* the classic obstacle problem, periodic in x and clamped in y, against its ANALYTIC solution and free
  boundary (``-u'' = -1``, ``u >= -c`` detaches tangentially at ``a = sqrt(2c)``, see test_fem_bounds.py);
* a periodic obstacle that varies in x: the KKT conditions of the REDUCED problem (``Pᵀ R = 0`` off the
  contact set, ``Pᵀ R >= 0`` on it, with ``R`` assembled un-eliminated by ``fem.eval``), and the same data
  without the tie giving a different answer, so the tie is not vacuous.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

import jno

grad, inner = jno.np.grad, jno.np.inner
C = 1.0 / 18.0  # obstacle depth -> free boundary at a = sqrt(2c) = 1/3
A_FREE = np.sqrt(2.0 * C)
EPS = 1e-9


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _obstacle_exact(y, c=C):
    """``-u'' = -1``, ``u(0) = u(1) = 0``, ``u >= -c``: the parabola meeting ``-c`` tangentially at ``a``."""
    a = np.sqrt(2.0 * c)
    y = np.asarray(y)
    return np.where(y < a, 0.5 * y**2 - a * y, np.where(y > 1 - a, 0.5 * (1 - y) ** 2 - a * (1 - y), -c))


def _obstacle(lo, *, tie=True, n=24):
    """``-Δu = -1`` on the unit square, ``u = 0`` on y = 0 and y = 1, periodic in x (``tie``) or natural
    there, ``u >= lo(x, y)``. Returns ``(fem, equilibrium, u)``."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=n).domain()
    d.tag("l", lambda x, y: x < EPS)
    d.tag("r", lambda x, y: x > 1 - EPS)
    d.tag("b", lambda x, y: y < EPS)
    d.tag("t", lambda x, y: y > 1 - EPS)
    on = lambda tag: d.variable(tag, split=True)[:2]  # noqa: E731
    u, phi = d.fem_symbols()
    co = d.variable("interior", split=True)
    X = [co[0], co[1]]
    equilibrium = inner(grad(u, X), grad(phi, X), 1) + 1.0 * phi
    terms = [equilibrium, u(*on("b")) - 0.0, u(*on("t")) - 0.0, u.bounds(lo(*X), None)]
    if tie:
        terms.append(u(*on("l")) - u(*on("r")))
    fem = jno.fem(terms)
    s = fem.solve()
    return fem, equilibrium, np.asarray(s.fn() if hasattr(s, "fn") else s).reshape(-1)


def _seam(X):
    left, right = np.flatnonzero(X[:, 0] < 1e-6), np.flatnonzero(X[:, 0] > 1 - 1e-6)
    return left[np.argsort(X[left, 1])], right[np.argsort(X[right, 1])]


def test_a_tied_obstacle_problem_matches_the_analytic_free_boundary():
    fem, _eq, u = _obstacle(lambda x, y: -C + 0.0 * x)
    X = np.asarray(fem.points)
    assert u.min() > -C - 1e-12, f"the obstacle was violated by {-C - u.min():.3e}"
    err = np.abs(u - _obstacle_exact(X[:, 1])).max()
    assert err < 2e-3, f"max |u - u_exact| = {err:.3e}"
    on = np.abs(u + C) < 1e-9
    assert on.any(), "nothing made contact -- the bound never activated"
    assert abs(X[on, 1].min() - A_FREE) < 0.03 and abs(X[on, 1].max() - (1 - A_FREE)) < 0.03
    left, right = _seam(X)
    assert np.array_equal(u[left], u[right]), "the tie is not exact under the box"


def test_a_periodic_obstacle_satisfies_the_reduced_kkt_conditions():
    """``lo = -C (1 + 0.6 sin 2πx)``: periodic, but not symmetric about the tied faces, so the natural
    (untied) answer differs. At the solution the REDUCED residual -- each eliminated left-face row summed
    into the right-face row it is tied to -- vanishes off the contact set and pushes back on it."""
    lo = lambda x, y: -C * (1.0 + 0.6 * jno.np.sin(2 * np.pi * x))  # noqa: E731
    fem, equilibrium, u = _obstacle(lo)
    X = np.asarray(fem.points)
    psi = -C * (1.0 + 0.6 * np.sin(2 * np.pi * X[:, 0]))
    assert (u - psi).min() > -1e-12, f"the obstacle was violated by {-(u - psi).min():.3e}"
    left, right = _seam(X)
    assert np.array_equal(u[left], u[right]), "the tie is not exact under the box"

    R = np.asarray(fem.eval(equilibrium, u)).reshape(-1)  # the same term, un-eliminated
    R_red = R.copy()
    R_red[right] += R[left]  # Pᵀ: an eliminated row is summed into the row it is tied to
    kept = np.ones(len(u), dtype=bool)
    kept[left] = False
    walls = (X[:, 1] < 1e-6) | (X[:, 1] > 1 - 1e-6)  # Dirichlet rows carry their own reaction
    on = np.abs(u - psi) < 1e-9
    assert on[kept & ~walls].any(), "nothing made contact -- the bound never activated"
    free = kept & ~walls & ~on
    scale = np.abs(R_red[kept & ~walls]).max()
    assert np.abs(R_red[free]).max() < 1e-9 * max(scale, 1.0), "equilibrium is violated OFF the contact set"
    assert (R_red[kept & ~walls & on] > -1e-9).all(), "the contact reaction changed sign"

    _fem, _eq, u_natural = _obstacle(lo, tie=False)
    assert np.abs(u_natural - u).max() > 1e-3, "the tie changes nothing here; the check would be vacuous"


def test_a_bound_that_differs_across_the_tie_is_refused_by_name():
    """The tie makes the two faces ONE unknown; ``lo = -C (0.5 + x)`` asks for two different bounds on it."""
    with pytest.raises(ValueError, match="lower bound .* differs across the periodic tie"):
        _obstacle(lambda x, y: -C * (0.5 + x))


def test_a_bound_on_a_weighted_tie_is_refused_by_name():
    """A non-matching interface ties an eliminated DOF to a WEIGHTED combination of kept ones, which a box
    on the kept ones does not bound. Refused at build, naming the combination."""
    d = jno.shape.regions(  # each body at its own size, so the interface meshes do not match
        lower=jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3),
        upper=jno.shape.rect(0.0, 1.0, 1.0, 2.0, size=0.21),
        conforming=False,
    ).domain()
    sec, main = sorted(t for t in d.built_mesh.cell_sets if "|" in t)
    u, phi = d.fem_symbols()
    co = d.variable("interior", split=True)
    X = [co[0], co[1]]
    ob = d.variable("outer", where=lambda x, y: (y < EPS) | (y > 2 - EPS), split=True)
    sv, mv = d.variable(sec, split=True), d.variable(main, split=True)
    with pytest.raises(NotImplementedError, match="bounds.*WEIGHTED"):
        jno.fem(
            [
                inner(grad(u, X), grad(phi, X), 1) + 1.0 * phi,
                u(ob[0], ob[1]) - 0.0,
                u(sv[0], sv[1]) - u(mv[0], mv[1]),
                u.bounds(-C, None),
            ]
        )
