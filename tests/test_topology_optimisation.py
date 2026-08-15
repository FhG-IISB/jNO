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

# EMIN = 1e-6, not the 1e-9 of the original 88-line-code lineage: at a 1e9 stiffness contrast
# cuSolver's QR spsolve -- the GPU backend behind the default differentiable solve -- falsely
# reports the (regular, merely ill-conditioned) SIMP stiffness SINGULAR and the whole chain dies
# inside the gradient. Measured: every compliance test fails on GPU at 1e-9 and passes at 1e-6,
# which is itself a standard void-stiffness choice; the layouts are indistinguishable.
E0, EMIN, NU, PENAL, VOLFRAC = 1.0, 1e-6, 0.3, 3.0, 0.4
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
            (EMIN + rho**PENAL * (E0 - EMIN)) * (LAM * trace(eu) * trace(ep) + 2 * MU * inner(eu, ep, n_contract=2)),
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
            f"a tighter budget must give higher compliance: C(0.25)={results[0.25]:.4f} vs C(0.5)={results[0.5]:.4f}"
        )


class TestP0DensityParameter:
    """``jno.np.parameter(<P0 symbol>)`` — one design value per ELEMENT.

    This is the density variable of the method (Jung, Yun & Kim, *Computers & Structures* **331**
    (2026) 108403, eq. 12; Bendsoe & Sigmund, *Topology Optimization*, Springer 2004). A P0 symbol
    reports ``order == 1``, so the space -- not the order -- is what distinguishes it, and without
    that branch a P0 symbol silently produced a NODE-sized parameter: 18 values for 22 cells on the
    mesh below, with the design smeared across element boundaries by the P1 interpolation.
    """

    @staticmethod
    def _build(size=0.5):
        inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
        d = jno.Shape.rect(0, 0, 2, 1, size=size).domain()
        u, phi = d.fem_symbols(value_shape=(2,))
        _r, s = d.fem_symbols(space="P0", names=("r", "s"))
        xi, yi, _ = d.variable("interior", split=True)
        xl, yl, _ = d.variable("left", split=True)
        rho = jno.np.parameter(s, name="rho")
        eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
        # Linear in rho on purpose: a one-hot design then assembles exactly one element matrix,
        # which is what makes the two checks below exact rather than approximate.
        fem = jno.fem(
            [
                rho * (LAM * trace(eu) * trace(ep) + 2 * MU * inner(eu, ep, n_contract=2)),
                u(xl, yl) - (0.0, 0.0),
            ],
            quad_degree=2,
        )
        return d, np.asarray(d._cells_p1()), fem

    def test_the_parameter_is_sized_by_cells_not_nodes(self):
        d = jno.Shape.rect(0, 0, 2, 1, size=0.5).domain()
        _r, s = d.fem_symbols(space="P0", names=("r", "s"))
        n_cells = int(np.asarray(d._cells_p1()).shape[0])
        n_nodes = int(np.asarray(d.built_mesh.points).shape[0])
        assert n_cells != n_nodes, "the mesh must distinguish the two for this to test anything"

        rho = jno.np.parameter(s, name="rho_p0")
        assert rho.model._fem_field == "cell", "a P0 symbol must mark the parameter as a cell field"
        shapes = [lf.shape for lf in jax.tree_util.tree_leaves(rho.model.module)]
        assert shapes == [(n_cells,)], f"expected [({n_cells},)], got {shapes}"

    def test_each_value_lands_on_its_own_element_and_nowhere_else(self):
        d, cells, fem = self._build()
        n_cells = cells.shape[0]

        def K(vals):
            a, _ = fem.operator.evaluate({"rho": jnp.asarray(vals, dtype=jnp.float64)})
            return np.asarray(jnp.asarray(a.todense()))

        k_all = K(np.ones(n_cells))
        eye = np.eye(k_all.shape[0])
        # The Dirichlet rows are replaced by the identity in EVERY assembly, so they would
        # accumulate across the sum below; compare on the free block.
        dirichlet = np.where(np.all(np.isclose(k_all, eye), axis=1))[0]
        free = np.setdiff1d(np.arange(k_all.shape[0]), dirichlet)

        # 1. Linearity: with a coefficient linear in rho, one-hot designs must sum to the whole.
        acc = sum(K(np.eye(n_cells)[j])[np.ix_(free, free)] for j in range(n_cells))
        np.testing.assert_allclose(acc, k_all[np.ix_(free, free)], atol=1e-12)

        # 2. Support: a one-hot design assembles ONLY that element's block. A node-sized
        #    parameter would leak into every element touching those nodes.
        j = 5
        k_j = K(np.eye(n_cells)[j])[np.ix_(free, free)]
        cell_dofs = set(np.concatenate([2 * cells[j], 2 * cells[j] + 1]).tolist())
        elsewhere = np.array([i for i, dof in enumerate(free) if dof not in cell_dofs])
        assert np.abs(k_j[np.ix_(elsewhere, elsewhere)]).max() == 0.0
        here = np.array([i for i, dof in enumerate(free) if dof in cell_dofs])
        assert np.abs(k_j[np.ix_(here, here)]).max() > 0.0

    def test_a_uniform_p0_density_matches_a_uniform_p1_density(self):
        """The two spaces must agree exactly where they can — on a constant field."""
        inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace

        def assemble(space):
            d = jno.Shape.rect(0, 0, 2, 1, size=0.5).domain()
            u, phi = d.fem_symbols(value_shape=(2,))
            _r, s = d.fem_symbols(names=("r", "s")) if space == "P1" else d.fem_symbols(space="P0", names=("r", "s"))
            xi, yi, _ = d.variable("interior", split=True)
            xl, yl, _ = d.variable("left", split=True)
            rho = jno.np.parameter(s, name="rho")
            eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
            fem = jno.fem(
                [
                    rho * (LAM * trace(eu) * trace(ep) + 2 * MU * inner(eu, ep, n_contract=2)),
                    u(xl, yl) - (0.0, 0.0),
                ],
                quad_degree=2,
            )
            n = int(np.asarray(d._cells_p1()).shape[0]) if space == "P0" else int(np.asarray(d.built_mesh.points).shape[0])
            a, _ = fem.operator.evaluate({"rho": jnp.full(n, 0.7, dtype=jnp.float64)})
            return np.asarray(jnp.asarray(a.todense()))

        np.testing.assert_allclose(assemble("P0"), assemble("P1"), atol=1e-12)


def _f_patch_reference(rk, others, boundary):
    """eq. (18) written out literally, one patch at a time, from the paper.

    Deliberately a slow transcription with an explicit loop: the vectorised implementation in
    ``domain.patch_filter`` has to agree with THIS, and a shared helper would let one bug hide
    in both.
    """
    n = len(others) + 1
    if n < 3:
        return 0.0
    prod = 1.0
    for i in range(2, n - 1):
        pre = sum(others[: i - 1]) / (i - 1)
        suf = sum(others[i : n - 1]) / (n - i - 1)
        prod *= 1 - others[i - 1] * (1 - pre) * (1 - suf)
    last = 1.0 if boundary else 1 - rk * (1 - (others[0] + others[n - 2]) / 2)
    return (prod * last) ** (1.0 / (n - 2))


class TestPatchFilter:
    """The patch filter, eq. (17)-(19) — Jung, Yun & Kim, *Comput. Struct.* **331** (2026) 108403."""

    def test_it_reproduces_the_configurations_the_paper_names(self):
        """Fig. 2b, a five-element patch with the reference element dense.

        The paper states the rule in words: "when rho_{k,2} = 1, a valid connection requires either
        rho_{k,1} = 1, or both rho_{k,3} = 1 and rho_{k,4} = 1". Both alternatives must survive and
        everything else in that family must be suppressed — this is the behaviour the formula is
        FOR, so it is checked directly rather than through the algebra.
        """
        f = _f_patch_reference
        assert f(1.0, [0, 0, 0, 0], False) == pytest.approx(0.0), "a lone dense element must go"
        assert f(1.0, [0, 1, 0, 0], False) == pytest.approx(0.0), "a one-node connection must go"
        assert f(1.0, [1, 1, 0, 0], False) > 0.5, "valid: adjacent, must survive"
        assert f(1.0, [0, 1, 1, 1], False) > 0.5, "valid: the other alternative, must survive"
        assert f(1.0, [1, 1, 1, 1], False) == pytest.approx(1.0), "a full patch is untouched"
        # N = 3 collapses to the last term alone, as the paper says it does.
        assert f(1.0, [0, 0], False) == pytest.approx(0.0)
        assert f(1.0, [1, 1], False) == pytest.approx(1.0)

    def test_the_vectorised_filter_matches_the_literal_formula(self):
        d = jno.Shape.rect(0, 0, 2, 1, size=0.25).domain()
        topo = d.patch_topology()
        n_cells = int(d._cells_p1().shape[0])
        assert topo["size"].max() >= 5, "the mesh must contain patches big enough to exercise eq. (18)"
        assert topo["boundary"].any() and (~topo["boundary"]).any(), "both branches must be covered"

        filt = d.patch_filter()
        rng = np.random.default_rng(0)
        for r in (
            np.ones(n_cells),
            np.full(n_cells, 0.4),
            rng.uniform(0.0, 1.0, n_cells),
            (rng.random(n_cells) > 0.6).astype(float),
        ):
            expected = np.empty_like(r)
            for k in range(n_cells):
                fs = []
                for v in range(3):
                    n = int(topo["size"][k, v])
                    if n < 3:
                        continue
                    fs.append(
                        _f_patch_reference(
                            float(r[k]),
                            [float(r[j]) for j in topo["others"][k, v, : n - 1]],
                            bool(topo["boundary"][k, v]),
                        )
                    )
                expected[k] = r[k] * (sum(fs) / len(fs) if fs else 1.0)
            np.testing.assert_allclose(np.asarray(filt(jnp.asarray(r))), expected, atol=1e-7)

    def test_a_full_density_field_passes_through_untouched(self):
        """rho = 1 everywhere has no bad configuration anywhere, so the filter must be the identity.

        The sharpest single check that the walk and the padding are right: any mis-indexed
        neighbour would read a padded zero and pull the product below one.
        """
        d = jno.Shape.rect(0, 0, 2, 1, size=0.2).domain()
        n_cells = int(d._cells_p1().shape[0])
        out = np.asarray(d.patch_filter()(jnp.ones(n_cells, dtype=jnp.float64)))
        np.testing.assert_allclose(out, 1.0, atol=1e-12)

    def test_it_suppresses_an_isolated_element_on_a_real_mesh(self):
        """One dense element in a void field is unbuildable; the filter must knock it down.

        Note what "suppress" actually means numerically. The ``1 / (N - 2)`` exponent softens the
        near-zero product: on a six-element patch a fully isolated element lands near
        ``0.001 ** 0.25 ~ 0.18``, not at zero. That is the formula's own behaviour, not a defect —
        **SIMP finishes the job**, since the stiffness carries ``rho_bar ** penal`` and 0.18 cubed
        is under 1 % of solid, which is why the paper also raises ``penal`` as it converges.
        """
        d = jno.Shape.rect(0, 0, 2, 1, size=0.2).domain()
        topo = d.patch_topology()
        n_cells = int(d._cells_p1().shape[0])
        # Pick an element all of whose patches are interior, so no vertex takes the boundary rule.
        k = int(np.where(~topo["boundary"].any(axis=1))[0][0])
        r = np.full(n_cells, 1e-3)
        r[k] = 1.0
        out = np.asarray(d.patch_filter()(jnp.asarray(r)))
        assert out[k] < 0.35, f"an isolated dense element barely moved: rho_bar = {out[k]:.4f}"
        assert out[k] ** PENAL < 0.01, "and its SIMP stiffness must be under 1 % of solid"
        # Its void neighbours stay essentially void: the filter scales each element by its OWN
        # patches, so a neighbour only feels this one through its own (already near-zero) density.
        assert np.abs(np.delete(out, k) - np.delete(r, k)).max() < 1e-3

    def test_the_node_and_the_filter_agree_and_the_node_carries_a_gradient(self):
        d = jno.Shape.rect(0, 0, 2, 1, size=0.25).domain()
        _r, s = d.fem_symbols(space="P0", names=("r", "s"))
        rho = jno.np.parameter(s, name="rho_patch")
        n_cells = int(d._cells_p1().shape[0])
        r = jnp.asarray(np.random.default_rng(2).uniform(0, 1, n_cells))

        np.testing.assert_allclose(np.asarray(rho.patch().fn(r)), np.asarray(d.patch_filter()(r)), atol=0.0)
        g = jax.grad(lambda z: jnp.sum(d.patch_filter()(z)))(r)
        assert np.all(np.isfinite(np.asarray(g))) and np.any(np.asarray(g) != 0.0)

    def test_a_nodal_density_is_refused(self):
        """The reference element of a patch is an ELEMENT; a nodal field has none."""
        d = jno.Shape.rect(0, 0, 2, 1, size=0.4).domain()
        _r, s = d.fem_symbols(names=("r", "s"))
        with pytest.raises(TypeError, match="P0"):
            jno.np.parameter(s, name="rho_nodal").patch()


class TestPerimeter:
    """Smoothed structural perimeter — eq. (38), after Haber, Jog & Bendsoe (1996)."""

    @staticmethod
    def _setup(size=2.0):
        d = jno.Shape.rect(0, 0, 60, 30, size=size).domain()
        _r, s = d.fem_symbols(space="P0", names=("r", "s"))
        rho = jno.np.parameter(s, name="rho_perim")
        cells = np.asarray(d._cells_p1())
        centroids = np.asarray(d.mesh.points)[:, :2][cells].mean(axis=1)
        return d, rho.perimeter(zeta=0.1).fn, cells.shape[0], centroids

    def test_a_uniform_density_has_no_perimeter(self):
        """No jump anywhere means no material boundary — the smoothing must vanish exactly."""
        _d, p, n_cells, _c = self._setup()
        for value in (0.0, 0.4, 1.0):
            assert float(p(jnp.full(n_cells, value))) == pytest.approx(0.0, abs=1e-12)

    def test_a_bar_measures_its_own_boundary_length(self):
        """A bar spanning the domain has two interfaces of length 60, so P must be ~120.

        Slightly ABOVE 120 is correct, not an error: the interface follows the mesh edges, and a
        zig-zag through triangles is longer than the straight line it approximates.
        """
        _d, p, n_cells, centroids = self._setup()
        bar = np.where(np.abs(centroids[:, 1] - 15.0) < 6.0, 1.0, 0.0)
        got = float(p(jnp.asarray(bar)))
        assert 120.0 <= got < 132.0, f"a full-width bar should measure ~120, got {got:.2f}"

    def test_a_fragmented_layout_costs_more_perimeter_than_a_compact_one(self):
        """The whole point: same volume, more pieces, more boundary. This is the feature-scale
        signal a barrier on P acts on."""
        _d, p, n_cells, centroids = self._setup()
        rng = np.random.default_rng(0)
        one_bar = np.where(np.abs(centroids[:, 1] - 15.0) < 6.0, 1.0, 0.0)
        scattered = np.zeros(n_cells)
        scattered[rng.choice(n_cells, int(one_bar.sum()), replace=False)] = 1.0
        assert float(p(jnp.asarray(scattered))) > 3.0 * float(p(jnp.asarray(one_bar)))

    def test_the_smoothing_is_exact_at_both_ends(self):
        """eq. (38)'s bracket is 0 at no jump and exactly 1 at a full one, for any zeta."""
        for zeta in (0.05, 0.1, 0.5):
            f = lambda j, z=zeta: np.sqrt((1 + 2 * z) * j**2 + z * z) - z  # noqa: E731
            assert f(0.0) == pytest.approx(0.0, abs=1e-15)
            assert f(1.0) == pytest.approx(1.0, abs=1e-12)

    def test_a_nodal_density_is_refused(self):
        d = jno.Shape.rect(0, 0, 2, 1, size=0.4).domain()
        _r, s = d.fem_symbols(names=("r", "s"))
        with pytest.raises(TypeError, match="P0"):
            jno.np.parameter(s, name="rho_nodal_p").perimeter()


class TestCrossMeshTransfer:
    """``transfer_cell_field`` — the machinery a reanalysis needs.

    An optimisation whose mesh coordinates are design variables can lower its objective by
    distorting elements into under-integrating strain energy, and cannot tell that apart from
    genuine stiffness. Re-solving the converged density on a clean mesh is the only check that
    separates them: measured once at +163 % on a design that looked entirely correct.
    """

    def test_a_constant_field_survives_transfer(self):
        coarse = jno.Shape.rect(0, 0, 60, 30, size=4.0).domain()
        fine = jno.Shape.rect(0, 0, 60, 30, size=1.5).domain()
        vals = np.full(int(coarse._cells_p1().shape[0]), 0.7)
        out = coarse.transfer_cell_field(vals, fine)
        assert out.shape == (int(fine._cells_p1().shape[0]),)
        np.testing.assert_allclose(out, 0.7, atol=1e-12), "a constant must be carried exactly"

    def test_a_region_lands_in_the_right_place(self):
        """A bar transfers to a bar: the geometry, not just the values, must survive."""
        coarse = jno.Shape.rect(0, 0, 60, 30, size=2.0).domain()
        fine = jno.Shape.rect(0, 0, 60, 30, size=1.0).domain()
        c_cen = np.asarray(coarse.mesh.points)[:, :2][coarse._cells_p1()].mean(axis=1)
        bar = np.where(np.abs(c_cen[:, 1] - 15.0) < 6.0, 1.0, 0.0)

        out = coarse.transfer_cell_field(bar, fine)
        f_cen = np.asarray(fine.mesh.points)[:, :2][fine._cells_p1()].mean(axis=1)
        # Every target centroid well inside the bar must be solid, and well outside must be void.
        deep_in = np.abs(f_cen[:, 1] - 15.0) < 4.0
        deep_out = np.abs(f_cen[:, 1] - 15.0) > 8.0
        assert out[deep_in].min() == 1.0 and out[deep_out].max() == 0.0
        # Area is preserved to the resolution of the coarser mesh.
        assert out.mean() == pytest.approx(bar.mean(), abs=0.05)

    def test_deformed_source_coordinates_are_honoured(self):
        """The source domain still holds the positions it was BUILT with, so a mesh moved by
        `.trainable()` has to pass its deformed coordinates explicitly — otherwise the transfer
        reads the design off the wrong geometry and silently reports the wrong answer."""
        coarse = jno.Shape.rect(0, 0, 60, 30, size=3.0).domain()
        fine = jno.Shape.rect(0, 0, 60, 30, size=1.5).domain()
        pts = np.asarray(coarse.mesh.points)[:, :2]
        cells = coarse._cells_p1()
        bar = np.where(np.abs(pts[cells].mean(axis=1)[:, 1] - 15.0) < 4.0, 1.0, 0.0)

        # A gentle shift of the interior nodes only, small enough not to tangle the mesh.
        shifted = pts.copy()
        interior = (pts[:, 1] > 1e-9) & (pts[:, 1] < 30.0 - 1e-9)
        shifted[interior, 1] += 2.0

        same = coarse.transfer_cell_field(bar, fine)
        moved = coarse.transfer_cell_field(bar, fine, points=shifted)
        assert not np.allclose(same, moved), "moving the source nodes must move the field"
        # Same amount of material, in a different place — the geometry moved, not the design.
        assert moved.sum() == pytest.approx(same.sum(), rel=0.25)

    def test_a_mis_sized_field_is_refused(self):
        coarse = jno.Shape.rect(0, 0, 60, 30, size=4.0).domain()
        fine = jno.Shape.rect(0, 0, 60, 30, size=2.0).domain()
        with pytest.raises(ValueError, match="entries but this mesh has"):
            coarse.transfer_cell_field(np.ones(3), fine)


class TestFacetTractionTotal:
    """A traction band must apply the resultant it was asked for, at every mesh size.

    This is a regression test for a defect that produced two wrong results before it was found. A
    boundary term integrates over FACETS, and a facet is selected only when all of its nodes satisfy
    the region predicate. Selecting a band of width ``span`` with a strict ``y < span`` therefore
    drops the facet whose upper node sits exactly at ``y == span``, and the applied resultant
    silently becomes ``(span - h) / span`` instead of 1 -- zero on a mesh with ``h == span``.

    Nothing raises, nothing looks wrong, and compliance depends on the load QUADRATICALLY, so the
    error is 4x at ``h = span / 2``. Measured on a 60x30 domain with ``span = 2``: the resultant
    came out 0.000 / -0.500 / -0.750 / -0.875 at ``h = 2 / 1 / 0.5 / 0.25``.
    """

    @staticmethod
    def _resultant(size, pad, span=2.0, L=60.0, H=30.0):
        tol = 1e-6 * L
        d = jno.Shape.rect(0, 0, L, H, size=size).domain()
        u, phi = d.fem_symbols(value_shape=(2,))
        _r, s = d.fem_symbols(space="P0", names=("r", "s"))
        xi, yi, _ = d.variable("interior", split=True)
        xl, yl, _ = d.variable("left", split=True)
        xt, yt, _ = d.variable("tip", where=lambda x, y: (x > L - tol) & (y < span + pad), split=True)
        rho = jno.np.parameter(s, name="rho_traction")
        rho.dtype(jnp.float64)
        eu, ep = jno.np.symgrad(u, [xi, yi]), jno.np.symgrad(phi, [xi, yi])
        fem = jno.fem(
            [
                (EMIN + rho**PENAL * (E0 - EMIN))
                * (LAM * jno.np.trace(eu) * jno.np.trace(ep) + 2 * MU * jno.np.inner(eu, ep, n_contract=2)),
                u(xl, yl) - (0.0, 0.0),
                -1.0 * jno.np.inner(jnp.array([0.0, -1.0 / span]), phi.bind(x=xt, y=yt), n_contract=1),
            ],
            quad_degree=2,
        )
        n_cells = np.asarray(d._cells_p1()).shape[0]
        _a, b = fem.operator.evaluate({"rho_traction": jnp.full(n_cells, VOLFRAC)})
        return float(np.asarray(jnp.asarray(b).reshape(-1, 2))[:, 1].sum())

    @pytest.mark.parametrize("size", [2.0, 1.0])
    def test_padded_predicate_applies_the_full_resultant(self, size):
        """The inclusive band integrates to exactly -1 regardless of the mesh."""
        got = self._resultant(size, pad=1e-6 * 60.0)
        assert got == pytest.approx(-1.0, abs=1e-9), f"h={size}: traction band integrated to {got:.6f}, not -1"

    def test_strict_predicate_loses_load_and_is_mesh_dependent(self):
        """Pins WHY the tolerance is needed: without it the load depends on the discretisation."""
        coarse, fine = self._resultant(2.0, pad=0.0), self._resultant(1.0, pad=0.0)
        assert coarse == pytest.approx(0.0, abs=1e-9), f"h == span should lose the only facet, got {coarse:.6f}"
        assert fine == pytest.approx(-0.5, abs=1e-9), f"expected (span-h)/span, got {fine:.6f}"
