"""Density-based topology optimisation end to end: SIMP + a differentiable FEM solve + MMA.

This is the integration test for the pieces `jno.le`, `jno.optimizers.mma` and the differentiable
`fem.solve()` only make sense together. Compliance minimisation under a volume constraint is the
canonical problem of the field (Bendsoe & Sigmund, *Topology Optimization*, Springer 2004), and
every sensitivity here comes from differentiating the assembled solve — none is hand-derived.

What is deliberately NOT asserted: a specific compliance value. The mesh is unstructured triangles
with a nodal density and no filter, so there is no published number this can be held against. What
*is* pinned is the behaviour that would break if any link in the chain were wrong — the objective
must fall a lot, the volume constraint must end exactly active, the box must hold, and the design
must leave the uniform field it started from.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

E0, EMIN, NU, PENAL, VOLFRAC = 1.0, 1e-9, 0.3, 3.0, 0.4
LAM, MU = E0 * NU / (1 - NU**2), E0 / (2 * (1 + NU))


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _cantilever(size=0.16, move=0.15):
    """Clamped left edge, downward traction on the right. Returns (crux, rho, n_nodes)."""
    inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace

    d = jno.Shape.rect(0, 0, 2, 1, size=size).domain()
    u, phi = d.fem_symbols(value_shape=(2,))
    _r, s = d.fem_symbols(names=("r", "s"))
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)

    rho = jno.np.parameter(s, name="rho")
    rho.dtype(jnp.float64)
    rho.initialize(jax.nn.initializers.constant(VOLFRAC))
    rho.optimizer(jno.optimizers.mma(move=move, lower=1e-3, upper=1.0))

    eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
    fem = jno.fem(
        [
            (EMIN + rho**PENAL * (E0 - EMIN))
            * (LAM * trace(eu) * trace(ep) + 2 * MU * inner(eu, ep, n_contract=2)),
            u(xl, yl) - (0.0, 0.0),
            -1.0 * inner(jnp.array([0.0, -1.0]), phi.bind(x=xr, y=yr), n_contract=1),
        ],
        quad_degree=2,
    )
    n_nodes = int(np.asarray(d.built_mesh.points).shape[0])
    # The load vector does not depend on rho, so capture it once; compliance is then f.u exactly.
    _A, b = fem.operator.evaluate({"rho": jnp.full(n_nodes, VOLFRAC)})
    f_vec = np.asarray(jnp.asarray(b).reshape(-1))

    compliance = jno.fn(lambda uu: jnp.sum(uu * jnp.asarray(f_vec)), [fem.solve()], name="C")
    crux = jno.core(
        [compliance, jno.le(rho.mean, VOLFRAC)],
        domain=jno.domain.from_array({"_": np.zeros((1, 1))}),
    )
    return crux, rho, n_nodes


class TestComplianceMinimisation:
    def test_the_whole_chain_runs_and_optimises(self):
        crux, rho, _ = _cantilever()
        stats = crux.solve(40)
        hist = stats.total_loss_history
        rho_f = np.asarray(crux.eval([rho])).reshape(-1)

        # 1. The objective falls, and keeps falling. MMA is a descent method here; a rise means the
        #    asymptote update or the dual is wrong.
        assert stats.total_loss < 0.5 * hist[0], f"compliance barely moved: {hist[0]} -> {stats.total_loss}"
        assert np.all(np.diff(hist) <= 1e-8), "compliance must not rise"

        # 2. The volume constraint ends ACTIVE — material is worth spending, so an optimum spends
        #    all of it — and is never violated.
        assert rho_f.mean() <= VOLFRAC + 1e-8, "the volume constraint must hold"
        assert rho_f.mean() == pytest.approx(VOLFRAC, abs=1e-6), "and should end exactly active"

        # 3. The box holds at both ends, and both ends are reached: a design that never touches
        #    its bounds has not really been penalised into a black-and-white layout.
        assert rho_f.min() >= 1e-3 - 1e-12 and rho_f.max() <= 1.0 + 1e-12
        assert rho_f.max() > 0.95 and rho_f.min() < 0.05

        # 4. It left the uniform field it started from — the actual point of the exercise.
        assert rho_f.std() > 0.3, "the density should be strongly non-uniform"

    def test_the_design_is_close_to_binary(self):
        """SIMP with penal=3 should push intermediate densities out; M_nd measures how far.

        No filter is applied here, so this converges to a *checkerboarded* layout — the classic
        instability of density methods on an unfiltered mesh. That is expected at this stage and is
        exactly what a filter (or the patch-based scheme of Jung et al. 2026) exists to suppress;
        the assertion below only pins the binarisation, not the connectivity.
        """
        crux, rho, _ = _cantilever()
        crux.solve(40)
        rho_f = np.asarray(crux.eval([rho])).reshape(-1)
        grey = float(((rho_f > 0.1) & (rho_f < 0.9)).mean())
        m_nd = float(4.0 * np.mean(rho_f * (1.0 - rho_f)))  # Sigmund's grey-level indicator
        assert grey < 0.25, f"too much intermediate density: {grey:.3f}"
        assert m_nd < 0.20, f"grey-level indicator too high: {m_nd:.3f}"

    def test_a_tighter_volume_budget_costs_compliance(self):
        """The constraint must actually bind: less material has to mean a worse optimum.

        This is the sharpest cheap check that the constraint is being enforced rather than
        decorated — a constraint that were quietly ignored would give the same answer either way.
        """
        results = {}
        for budget in (0.25, 0.5):
            inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
            d = jno.Shape.rect(0, 0, 2, 1, size=0.2).domain()
            u, phi = d.fem_symbols(value_shape=(2,))
            _r, s = d.fem_symbols(names=("r", "s"))
            xi, yi, _ = d.variable("interior", split=True)
            xl, yl, _ = d.variable("left", split=True)
            xr, yr, _ = d.variable("right", split=True)
            rho = jno.np.parameter(s, name=f"rho{budget}")
            rho.dtype(jnp.float64)
            rho.initialize(jax.nn.initializers.constant(budget))
            rho.optimizer(jno.optimizers.mma(move=0.15, lower=1e-3, upper=1.0))
            eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
            fem = jno.fem(
                [
                    (EMIN + rho**PENAL * (E0 - EMIN))
                    * (LAM * trace(eu) * trace(ep) + 2 * MU * inner(eu, ep, n_contract=2)),
                    u(xl, yl) - (0.0, 0.0),
                    -1.0 * inner(jnp.array([0.0, -1.0]), phi.bind(x=xr, y=yr), n_contract=1),
                ],
                quad_degree=2,
            )
            n_nodes = int(np.asarray(d.built_mesh.points).shape[0])
            _A, b = fem.operator.evaluate({f"rho{budget}": jnp.full(n_nodes, budget)})
            f_vec = np.asarray(jnp.asarray(b).reshape(-1))
            C = jno.fn(lambda uu, _f=f_vec: jnp.sum(uu * jnp.asarray(_f)), [fem.solve()], name="C")
            crux = jno.core(
                [C, jno.le(rho.mean, budget)],
                domain=jno.domain.from_array({"_": np.zeros((1, 1))}),
            )
            crux.solve(30)
            results[budget] = float(np.asarray(crux.eval([C])).reshape(-1)[0])

        assert results[0.25] > results[0.5], (
            f"a tighter budget must give higher compliance: "
            f"C(0.25)={results[0.25]:.4f} vs C(0.5)={results[0.5]:.4f}"
        )
