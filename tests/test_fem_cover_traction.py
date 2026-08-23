"""A cover element under an APPLIED TRACTION — the case the cover suite never exercised.

``tests/test_fem_cover.py`` covers the algebra, the null space, Dirichlet pinning, convergence and
the transient path, in twenty tests. Not one of them applies a surface load. That gap is not
incidental: the enrichment was expanded on the volume path and not on the facet path, so a surface
term emitted ``n_local`` test rows while the assembler's static pattern had already allocated
``n_local * blk``. Volume terms and Dirichlet conditions happen to avoid it; a traction does not.

The mismatch surfaces far from its cause, inside the Jacobian's scatter:

    ValueError: Incompatible types for broadcasting:
                input type=float64[67968] and requested type=float64[271872]

with ``271872 = 4 x 67968`` and 4 the cover block for a 3-D vector field. So ``space="cover"`` was
unusable with any applied load, and the way that was found was a 3-D topology-optimisation run
failing on another machine after the coarse cases had all passed.

What is deliberately NOT asserted here: a displacement value or a convergence rate. Those belong to
the tests that already exist. This one asserts that the problem assembles and solves at all, which
is precisely what it did not do.
"""

from __future__ import annotations

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


def _loaded_bar(space, *, dim=3, size=0.5):
    """Linear elasticity on a bar: clamped at ``x = 0``, uniform traction on ``x = L``.

    The smallest configuration carrying both a Dirichlet condition and a Neumann one, because it is
    the COMBINATION that failed -- either alone assembles.
    """
    inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
    L = 2.0
    E0, nu = 1.0, 0.3
    lam, mu = E0 * nu / ((1 + nu) * (1 - 2 * nu)), E0 / (2 * (1 + nu))
    tol = 1e-9

    shp = (
        jno.Shape.box(0, 0, 0, L, 1.0, 1.0, size=size)
        if dim == 3
        else jno.Shape.rect(0.0, 0.0, L, 1.0, size=size)
    )
    d = shp.domain()
    u, phi = d.fem_symbols(value_shape=(dim,), **({"space": "cover"} if space == "cover" else {}))
    co = d.variable("interior", split=True)[:dim]
    root = d.variable("root", where=lambda *c: c[0] < tol, split=True)[:dim]
    tip = d.variable("tip", where=lambda *c: c[0] > L - tol, split=True)[:dim]

    eps = lambda w: symgrad(w, list(co))  # noqa: E731
    a = lambda p, q: lam * trace(p) * trace(q) + 2 * mu * inner(p, q, n_contract=2)  # noqa: E731
    pull = np.zeros(dim)
    pull[-1] = -1.0  # pull the free end down

    fem = jno.fem(
        [
            a(eps(u), eps(phi)),
            u(*root) - tuple(0.0 for _ in range(dim)),
            -1.0 * inner(jax.numpy.asarray(pull), phi.bind(**dict(zip("xyz", tip))), 1),
        ],
        quad_degree=2,
    )
    return d, fem, u, eps, a


def _solve_dense(fem):
    """The displacement vector, via a dense solve -- the route `test_fem_cover.py` already uses."""
    import jax.numpy as jnp

    dense = lambda a: jnp.asarray(a.todense() if hasattr(a, "todense") else a)  # noqa: E731
    return np.asarray(
        fem.solve(lambda a, b: jnp.linalg.solve(dense(a), jnp.asarray(b).reshape(-1)))
    ).reshape(-1)


class TestCoverUnderTraction:
    """The enriched space must assemble and solve with a surface load applied."""

    @pytest.mark.parametrize("dim", [2, 3])
    def test_it_assembles_and_solves(self, dim):
        """Before the facet path expanded the cover, this raised inside the Jacobian scatter."""
        _d, fem, _u, _eps, _a = _loaded_bar("cover", dim=dim, size=0.6 if dim == 3 else 0.4)
        sol = _solve_dense(fem)
        assert np.all(np.isfinite(sol)), "the enriched solve returned non-finite values"
        assert np.abs(sol).max() > 0.0, (
            "a loaded bar must deflect; an all-zero solution means the traction never assembled"
        )

    def test_the_enriched_bar_deflects_like_the_p1_one(self):
        """Sanity, not accuracy: the two formulations must agree on the sign and the order.

        A cover element is softer than P1 on the same mesh, so it deflects at least as far. Asserting
        the ORDER rather than a value keeps this a smoke test of the assembly and leaves the accuracy
        claims to `test_fem_cover.py`, which owns them.
        """
        _dl, fem_l, _ul, _e, _a = _loaded_bar("Lagrange", dim=3, size=0.6)
        _dc, fem_c, _uc, _e2, _a2 = _loaded_bar("cover", dim=3, size=0.6)
        tip_l = float(np.abs(_solve_dense(fem_l)).max())
        tip_c = float(np.abs(_solve_dense(fem_c)).max())
        assert tip_l > 0.0 and tip_c > 0.0, f"both must deflect; got P1 {tip_l}, cover {tip_c}"
        assert tip_c >= 0.5 * tip_l, (
            f"the enriched solve deflected {tip_c} against P1's {tip_l} -- an order of magnitude "
            f"apart means the traction is being scaled by the cover block, not merely resolved better"
        )
