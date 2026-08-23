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
        topo = d._patch_topology()
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
        topo = d._patch_topology()
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

    # --- tetrahedra. In 3-D the interior facets are triangles, so the same functional measures the
    # --- material boundary's AREA; the formula, the smoothing and the barrier are unchanged.

    @staticmethod
    def _setup_3d(size=0.5, lx=4.0, ly=2.0, lz=4.0):
        d = jno.Shape.box(0, 0, 0, lx, ly, lz, size=size).domain()
        _r, s = d.fem_symbols(space="P0", names=("r", "s"))
        rho = jno.np.parameter(s, name="rho_perim_3d")
        cells = np.asarray(d._cells_p1())
        centroids = np.asarray(d.mesh.points)[:, :3][cells].mean(axis=1)
        return d, rho.perimeter(zeta=0.1).fn, cells.shape[0], centroids

    def test_a_uniform_density_has_no_perimeter_on_tets(self):
        _d, p, n_cells, _c = self._setup_3d()
        for value in (0.0, 0.4, 1.0):
            assert float(p(jnp.full(n_cells, value))) == pytest.approx(0.0, abs=1e-12)

    def test_a_slab_measures_its_own_interface_area(self):
        """A slab spanning the full x-y extent has two interfaces of area ``lx * ly``, so the flat
        answer is ``2 * 4 * 2 = 16``.

        The measured value is ~1.6x that, and **the mesh-independence is the assertion**, not the
        band. A triangulated interface stepping between tets is genuinely larger than the plane it
        approximates -- the 3-D version of a 2-D bar measuring above 120 -- and it does not converge
        to the flat area under refinement, it converges to a constant multiple of it. What WOULD
        move with the mesh is a miscount: including boundary facets, or double-counting shared ones,
        makes P scale with the facet count. Measured 1.60 / 1.64 / 1.51 / 1.62 at h = 0.8 / 0.5 /
        0.35 / 0.25, across a 22x range in tet count (410 -> 9035), so the ratio is pinned and the
        count is not.
        """
        ratios = []
        for h in (0.8, 0.35):
            _d, p, _n, centroids = self._setup_3d(size=h)
            slab = np.where(np.abs(centroids[:, 2] - 2.0) < 1.0, 1.0, 0.0)
            assert 0.1 < slab.mean() < 0.9, "the slab must be a real subset for this to measure anything"
            ratios.append(float(p(jnp.asarray(slab))) / 16.0)
        assert all(1.2 < r < 2.2 for r in ratios), f"a full-width slab should measure ~1.6 x 16, got {ratios}"
        assert abs(ratios[0] - ratios[1]) < 0.35, (
            f"the interface area must not track the mesh: {ratios[0]:.3f} vs {ratios[1]:.3f} at a "
            "8.5x change in tet count — a facet miscount is what would scale"
        )

    def test_a_fragmented_layout_costs_more_area_than_a_compact_one_on_tets(self):
        _d, p, n_cells, centroids = self._setup_3d()
        rng = np.random.default_rng(0)
        slab = np.where(np.abs(centroids[:, 2] - 2.0) < 1.0, 1.0, 0.0)
        scattered = np.zeros(n_cells)
        scattered[rng.choice(n_cells, int(slab.sum()), replace=False)] = 1.0
        assert float(p(jnp.asarray(scattered))) > 3.0 * float(p(jnp.asarray(slab)))


def _summed(fn):
    """The per-ridge node as a scalar. ``curvature()`` returns one value per ridge so the aggregate
    stays the caller's choice, and every assertion below is about the total."""
    return lambda *a, **k: jnp.sum(fn(*a, **k))


class TestCurvature:
    """Discrete bending energy of the material boundary — the LOCAL smoothness handle.

    ``perimeter()`` is a global budget: a design can meet it exactly and still be locally serrated,
    paying for a spike here with flatness elsewhere. Measured on a converged 3-D bracket with the
    perimeter barrier active and satisfied, the angle between adjacent surface facets had a median
    of 33.8 degrees and a 99th percentile of 88.9 — a boundary nobody would machine, scoring as
    well on eq. (38) as a smooth one. What is pinned here is the term that does see it: the
    discrete bending energy of Grinspun, Hirani, Desbrun & Schroeder, *Discrete Shells*, SCA 2003,
    Sec. 3, with the density jump selecting which facets are boundary at all.

    A structured quadrilateral grid carries most of these, because on one the answers are exact
    integers rather than mesh artefacts — a right angle is a right angle, so ``1 - cos t`` is
    exactly 1, and the assertions can be equalities.
    """

    @staticmethod
    def _grid(n=8, size=1.0):
        d = jno.Shape.rect(0, 0, n, n, size=size).structured().domain(cell="quad")
        _r, s = d.fem_symbols(space="P0", names=("r", "s"))
        rho = jno.np.parameter(s, name=f"rho_curv_{n}_{size}")
        cells = np.asarray(d._cells_topo()[0])
        cen = np.asarray(d.mesh.points)[:, :2][cells].mean(axis=1)
        return d, _summed(rho.curvature(zeta=0.1).fn), rho.perimeter(zeta=0.1).fn, cen

    def test_a_uniform_density_has_no_boundary_to_bend(self):
        """Not "small": exactly zero, for any density and any zeta, because eq. (38)'s bracket is
        exactly zero at no jump and it multiplies every pair."""
        _d, s, _p, cen = self._grid()
        for value in (0.0, 0.37, 1.0):
            assert float(s(jnp.full(cen.shape[0], value))) == 0.0

    def test_a_straight_interface_costs_exactly_nothing(self):
        """The floor has to be exact, or the term is a perimeter in disguise.

        On a structured grid a horizontal interface is genuinely collinear, so there is no mesh
        zig-zag to hide behind: S must be 0 while P measures the interface's full length of 8.
        """
        _d, s, p, cen = self._grid()
        flat = np.where(cen[:, 1] > 4.0, 1.0, 0.0)
        assert float(p(jnp.asarray(flat))) == pytest.approx(8.0, abs=1e-9), "the interface must be there at all"
        assert float(s(jnp.asarray(flat))) == pytest.approx(0.0, abs=1e-12)

    def test_a_right_angle_costs_exactly_one(self):
        """``|D_i| |D_j| (1 - cos t)`` at a full jump and ``t = 90`` degrees is exactly 1, so the
        energy counts corners. An L-shape turns once; a notch cut into a straight interface turns
        four times."""
        _d, s, _p, cen = self._grid()
        corner = np.where((cen[:, 1] > 4.0) | (cen[:, 0] > 6.0), 1.0, 0.0)
        assert float(s(jnp.asarray(corner))) == pytest.approx(1.0, abs=1e-9)
        notch = np.where(cen[:, 1] > 4.0, 1.0, 0.0)
        notch[(np.abs(cen[:, 0] - 4.0) < 1.0) & (np.abs(cen[:, 1] - 4.5) < 0.6)] = 0.0
        assert float(s(jnp.asarray(notch))) == pytest.approx(4.0, abs=1e-9)

    def test_it_separates_designs_the_perimeter_cannot_tell_apart(self):
        """**The test that justifies the term existing.**

        Under a monotone staircase boundary the perimeter telescopes: one horizontal unit per
        column plus a total rise of ``h_first - h_last``, whatever route the staircase takes
        between them. So these three designs have the same material volume AND the same perimeter
        to every decimal place, and differ only in how many times the boundary turns. Measured
        P = 14.0000 for all three; S = 2, 10, 12 against corner counts of 2, 10, 12.
        """
        n = 8
        _d, s, p, cen = self._grid(n=n)
        col = np.clip(cen[:, 0].astype(int), 0, n - 1)
        designs = {
            "one big step": [7, 7, 7, 7, 1, 1, 1, 1],
            "five small ones": [7, 7, 6, 5, 3, 2, 1, 1],
            "a staircase": [7, 6, 5, 4, 4, 3, 2, 1],
        }
        got = {}
        for name, h in designs.items():
            v = jnp.asarray((cen[:, 1] < np.asarray(h)[col]).astype(float))
            got[name] = (float(p(v)), float(s(v)), float(np.asarray(v).sum()), 2 * int(np.count_nonzero(np.diff(h))))

        vols = {round(g[2], 9) for g in got.values()}
        perims = {round(g[0], 9) for g in got.values()}
        assert len(vols) == 1, f"the designs must hold volume fixed for this to mean anything: {got}"
        assert len(perims) == 1, f"the whole construction is that P is identical: {got}"
        for name, (_pp, ss, _vv, turns) in got.items():
            assert ss == pytest.approx(float(turns), abs=1e-9), f"{name}: S={ss} against {turns} right angles"
        assert got["a staircase"][1] > 5.0 * got["one big step"][1], (
            "at identical perimeter the turn-heavy design must cost several times more bending, "
            f"got {got['a staircase'][1]} vs {got['one big step'][1]}"
        )

    def test_the_aggregate_equals_a_literal_pair_enumeration(self):
        """The double sum over facet pairs is evaluated as ``2 sum_{i<j} x_i x_j = (sum x)^2 - sum
        x^2`` so it never materialises a pair — an O(N) scatter where the fan around a 3-D edge
        would otherwise be O(N^2). The identity is exact, so this is an equality.

        Written out separately rather than through a shared helper, for the reason
        ``test_the_vectorised_filter_matches_the_formula`` gives: a shared one would let a single
        bug hide in both sides.
        """
        z = 0.1
        d = jno.Shape.box(0, 0, 0, 2, 1, 2, size=0.5).domain()
        _r, sym = d.fem_symbols(space="P0", names=("r", "s"))
        fast = _summed(jno.np.parameter(sym, name="rho_pairs").curvature(zeta=z).fn)
        topo = d._facet_ridges()
        pts = np.asarray(d.mesh.points)[:, :3]
        cells = np.asarray(d._cells_p1())
        cen = pts[cells].mean(axis=1)
        nodes, owners, fr, rn = topo["nodes"], topo["cells"], topo["facet_ridge"], topo["ridge_nodes"]

        raw = np.cross(pts[nodes[:, 1]] - pts[nodes[:, 0]], pts[nodes[:, 2]] - pts[nodes[:, 0]])
        nrm = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
        nrm *= np.where(np.sum(nrm * (cen[owners[:, 1]] - cen[owners[:, 0]]), -1) >= 0.0, 1.0, -1.0)[:, None]
        by_ridge: dict = {}
        for f, row in enumerate(fr):
            for r in row:
                by_ridge.setdefault(int(r), []).append(f)

        rng = np.random.default_rng(3)
        for rv in (rng.random(cells.shape[0]), (rng.random(cells.shape[0]) > 0.5).astype(float)):
            jump = rv[owners[:, 0]] - rv[owners[:, 1]]
            amp = np.sqrt((1 + 2 * z) * jump**2 + z * z) - z
            vec = nrm * (jump * np.sqrt(1 + z * z) / np.sqrt(jump**2 + z * z))[:, None]
            want = 0.0
            for r, fs in by_ridge.items():
                w = float(np.linalg.norm(pts[rn[r, 1]] - pts[rn[r, 0]]))
                for i in range(len(fs)):
                    for j in range(i + 1, len(fs)):
                        want += w * amp[fs[i]] * amp[fs[j]] * (1.0 - float(vec[fs[i]] @ vec[fs[j]]))
            assert want > 1.0, "a random design must have real bending in it, or this compares 0 to 0"
            assert float(fast(jnp.asarray(rv))) == pytest.approx(want, rel=1e-6)

    def test_it_is_never_negative_even_on_the_grayest_designs(self):
        """A bending energy that could dip below zero would be *exploitable*: the optimiser would
        manufacture folds to pay for compliance. Non-negativity is structural — ``|v_i . v_j| <= 1``
        with ``|v| <= 1`` — and gray is where a formulation that merely looks non-negative on
        black-and-white designs would give itself away, since that is where the smoothed magnitude
        and the true one disagree most."""
        d = jno.Shape.box(0, 0, 0, 2, 1, 2, size=0.6).domain()
        _r, sym = d.fem_symbols(space="P0", names=("r", "s"))
        s = _summed(jno.np.parameter(sym, name="rho_nonneg").curvature(zeta=0.1).fn)
        n = np.asarray(d._cells_p1()).shape[0]
        rng = np.random.default_rng(11)
        for _ in range(8):
            assert float(s(jnp.asarray(rng.random(n)))) >= 0.0
        for value in (0.5, 0.5 + 1e-9):  # the exact half-jump, where a sign error would surface
            checker = np.where(np.arange(n) % 2 == 0, value, 1.0 - value)
            assert float(s(jnp.asarray(checker))) >= 0.0

    def test_a_folded_interface_costs_more_than_a_flat_one_on_tets(self):
        """In 3-D the discrimination is real but milder than on a grid, and the honest number is
        the *ratio between the two functionals*: a square corrugation at fixed volume raises P by
        1.59x and S by 1.90x, and scrambling the same material raises P by 14.6x and S by 48.2x.
        So bending is roughly 1.2x sharper on a fold and 3.3x sharper on noise — worth a term, and
        not a replacement for the perimeter.
        """
        d = jno.Shape.box(0, 0, 0, 4, 2, 4, size=0.5).domain()
        _r, sym = d.fem_symbols(space="P0", names=("r", "s"))
        rho = jno.np.parameter(sym, name="rho_fold")
        s, p = _summed(rho.curvature(zeta=0.1).fn), rho.perimeter(zeta=0.1).fn
        cells = np.asarray(d._cells_p1())
        cen = np.asarray(d.mesh.points)[:, :3][cells].mean(axis=1)

        flat = np.where(cen[:, 2] > 2.0, 1.0, 0.0)
        fold = np.where(cen[:, 2] > 2.0 + 0.5 * np.where((cen[:, 0] // 1.0) % 2 < 0.5, 1.0, -1.0), 1.0, 0.0)
        assert abs(fold.mean() - flat.mean()) < 0.02, "the corrugation must not change the volume"
        s_ratio = float(s(jnp.asarray(fold))) / float(s(jnp.asarray(flat)))
        p_ratio = float(p(jnp.asarray(fold))) / float(p(jnp.asarray(flat)))
        assert s_ratio > p_ratio > 1.0, f"folding must cost bending faster than perimeter: {s_ratio=} {p_ratio=}"

    def test_the_flat_floor_does_not_track_the_mesh(self):
        """A geometrically flat interface still costs something on an unstructured mesh, because
        its facets zig-zag between tetrahedra — the 3-D counterpart of a straight bar measuring
        above its length. **That floor must be a property of the mesh's shape, not of its size**,
        or the term would grow under refinement and its weight would have to be re-tuned per mesh.
        Measured 50.7 at h=0.5 and 52.4 at h=0.35, across a 2.6x change in tet count; what would
        scale is a facet miscount or a double-counted ridge.
        """
        floors = []
        for h in (0.5, 0.35):
            d = jno.Shape.box(0, 0, 0, 4, 2, 4, size=h).domain()
            _r, sym = d.fem_symbols(space="P0", names=("r", "s"))
            s = _summed(jno.np.parameter(sym, name=f"rho_floor_{h}").curvature(zeta=0.1).fn)
            cells = np.asarray(d._cells_p1())
            cen = np.asarray(d.mesh.points)[:, :3][cells].mean(axis=1)
            floors.append(float(s(jnp.asarray(np.where(cen[:, 2] > 2.0, 1.0, 0.0)))))
        assert all(f > 1.0 for f in floors), "an unstructured interface is not coplanar; the floor is real"
        assert abs(floors[0] - floors[1]) < 0.25 * max(floors), f"the floor tracked the mesh: {floors}"

    def test_it_is_differentiable_in_the_mesh_and_points_downhill(self):
        """Under a deformable mesh this is the term that gives node migration a reason to ALIGN the
        boundary rather than merely shorten it, so the mesh gradient is not an incidental extra —
        it is half the point. Only the nodes near the boundary may feel it."""
        d = jno.Shape.rect(0, 0, 8, 4, size=0.6).domain()
        xs, ys, _ = d.variable("mv", where=lambda x, y: (x > 0.1) & (x < 7.9), split=True)
        for c, name in ((xs, "curv_cx"), (ys, "curv_cy")):
            c.trainable(name=name)
        _r, sym = d.fem_symbols(space="P0", names=("r", "s"))
        node = jno.np.parameter(sym, name="rho_move").curvature(zeta=0.1)
        cells = np.asarray(d._cells_p1())
        cen = np.asarray(d.mesh.points)[:, :2][cells].mean(axis=1)
        bar = jnp.asarray(np.where(np.abs(cen[:, 1] - 2.0) < 0.9, 1.0, 0.0))
        coords = [jnp.asarray(np.asarray(d.mesh.points)[np.asarray(sp["ids"]), sp["axis"]]) for sp in d._trainable_coords]
        assert len(coords) == 2, "both in-plane coordinates must be design variables here"

        value = float(jnp.sum(node.fn(bar, *coords)))
        gx = jax.grad(lambda c0: jnp.sum(node.fn(bar, c0, coords[1])))(coords[0])
        gy = jax.grad(lambda c1: jnp.sum(node.fn(bar, coords[0], c1)))(coords[1])
        assert np.all(np.isfinite(np.asarray(gx))) and np.all(np.isfinite(np.asarray(gy)))
        moving = int(np.count_nonzero(np.asarray(gx)) + np.count_nonzero(np.asarray(gy)))
        assert 0 < moving < gx.size + gy.size, f"only the boundary's own nodes should feel it, got {moving}"
        assert float(jnp.sum(node.fn(bar, coords[0] - 1e-2 * gx, coords[1] - 1e-2 * gy))) < value, (
            "a step down the mesh gradient must lower the bending — a sign error would raise it"
        )

    def test_it_returns_one_value_per_ridge_so_the_aggregate_stays_the_callers(self):
        """The sum is an L1 aggregate: it buys down the BULK of the boundary and lets a few sharp
        folds survive. Measured on a converged 3-D bracket, a design penalised on the sum still had
        9% of its iso-surface edges folded past 60 degrees against 0% for a smooth sphere on the
        same mesh. Targeting that tail needs a p-norm, and a p-norm needs the ridges.

        Which makes the shape and the sign the contract: one non-negative value per ridge. The sign
        is not free -- the O(N) identity is exact but its evaluation cancels, so a ridge carrying no
        boundary lands a few ulp below zero and a p-norm over it would raise a negative to a power.
        """
        d = jno.Shape.box(0, 0, 0, 2, 1, 1, size=0.5).domain()
        _r, sym = d.fem_symbols(space="P0", names=("r", "s"))
        node = jno.np.parameter(sym, name="rho_perridge").curvature(zeta=0.1)
        n_ridges = d._facet_ridges()["ridge_nodes"].shape[0]
        cells = np.asarray(d._cells_p1())
        cen = np.asarray(d.mesh.points)[:, :3][cells].mean(axis=1)
        flat = np.asarray(node.fn(jnp.asarray(np.where(cen[:, 2] > 0.5, 1.0, 0.0))))
        assert flat.shape == (n_ridges,), f"expected one value per ridge, got {flat.shape}"
        assert (flat >= 0.0).all(), "a negative ridge would break any p-norm aggregate"
        assert flat.max() > 0.0, "a flat cut still bends where it steps between tetrahedra"

    def test_the_ridge_distribution_is_heavy_tailed_exactly_when_the_boundary_is_rough(self):
        """Why a p-norm is worth offering at all. On a boundary that is already smooth the bending
        is spread evenly and the sum is the only sensible aggregate; on a rough one a few ridges
        carry the folds and a p-norm can reach them while the sum averages them away.

        Measured on this mesh: max/median over the ridges that carry any boundary is 2.8 for a flat
        cut and 23.9 for a scrambled design (5.1 and 17.7 at half the mesh size).
        """
        d = jno.Shape.box(0, 0, 0, 2, 1, 1, size=0.5).domain()
        _r, sym = d.fem_symbols(space="P0", names=("r", "s"))
        node = jno.np.parameter(sym, name="rho_tail").curvature(zeta=0.1)
        cells = np.asarray(d._cells_p1())
        cen = np.asarray(d.mesh.points)[:, :3][cells].mean(axis=1)
        rng = np.random.default_rng(0)

        ratios = {}
        for name, v in (
            ("flat", np.where(cen[:, 2] > 0.5, 1.0, 0.0)),
            ("scrambled", (rng.random(cells.shape[0]) > 0.5).astype(float)),
        ):
            out = np.asarray(node.fn(jnp.asarray(v)))
            live = out[out > 1e-12]
            assert live.size > 5, f"{name}: too few loaded ridges to speak of a distribution"
            ratios[name] = float(out.max() / np.median(live))
        assert ratios["flat"] < 8.0, f"a smooth boundary should spread its bending evenly: {ratios}"
        assert ratios["scrambled"] > 10.0, f"a rough boundary should concentrate it: {ratios}"
        assert ratios["scrambled"] > 3.0 * ratios["flat"], (
            f"roughness must be what creates the tail, not the mesh: {ratios}"
        )

    def test_a_nodal_density_is_refused(self):
        d = jno.Shape.rect(0, 0, 2, 1, size=0.4).domain()
        _r, s = d.fem_symbols(names=("r", "s"))
        with pytest.raises(TypeError, match="P0"):
            jno.np.parameter(s, name="rho_nodal_c").curvature()


class TestInteriorFacets:
    """``_interior_facets`` — edges in 2-D, triangles in 3-D, each shared by exactly two cells."""

    @pytest.mark.parametrize(
        "shape, n_face_nodes",
        [(jno.Shape.rect(0, 0, 2, 1, size=0.4), 2), (jno.Shape.box(0, 0, 0, 2, 1, 1, size=0.5), 3)],
        ids=["triangles", "tets"],
    )
    def test_every_facet_is_shared_by_exactly_two_distinct_cells(self, shape, n_face_nodes):
        d = shape.domain()
        f = d._interior_facets()
        cells = np.asarray(d._cells_p1())
        assert f["nodes"].shape[1] == n_face_nodes
        assert f["cells"].shape == (f["nodes"].shape[0], 2)
        assert f["nodes"].shape[0] > 0, "a mesh this size has interior facets"
        assert (f["cells"][:, 0] != f["cells"][:, 1]).all(), "a facet cannot be shared with itself"
        # Every listed facet's nodes must actually belong to BOTH of its cells — the property that
        # would break first if the owner pairing were mis-assembled by the sort.
        for side in (0, 1):
            owner_nodes = cells[f["cells"][:, side]]
            assert np.all([np.isin(f["nodes"][i], owner_nodes[i]).all() for i in range(len(f["nodes"]))])

    def test_boundary_facets_are_excluded(self):
        """Counted against the boundary connectivity jNO already builds, so the two agree on what a
        boundary facet is rather than this test defining it a second time."""
        from jno.utils.solver.fem_facets import build_facet_connectivity

        for shape, key, n_local in (
            (jno.Shape.rect(0, 0, 2, 1, size=0.4), "triangle", 3),
            (jno.Shape.box(0, 0, 0, 2, 1, 1, size=0.5), "tetrahedron", 4),
        ):
            d = shape.domain()
            cells = np.asarray(d._cells_p1())
            n_interior = d._interior_facets()["nodes"].shape[0]
            n_boundary = int(build_facet_connectivity(cells, key).n_bfaces)
            # Each cell has n_local facets; an interior one is counted twice, a boundary one once.
            assert 2 * n_interior + n_boundary == n_local * cells.shape[0], (
                f"{key}: {n_interior} interior + {n_boundary} boundary does not tile "
                f"{n_local * cells.shape[0]} cell-facet slots"
            )

    def test_a_one_dimensional_domain_is_refused(self):
        """An interval's facets are its vertices, which carry no measure for a perimeter to sum."""
        d = jno.domain(constructor=jno.domain.line(mesh_size=0.2))
        assert d.dimension == 1
        with pytest.raises(NotImplementedError, match="triangles, quadrilaterals or tetrahedra"):
            d._interior_facets()

    # --- ridges: the facets OF the interior facets, which is what a bending functional walks ---

    @pytest.mark.parametrize(
        "shape, n_local, n_ridge_nodes",
        [(jno.Shape.rect(0, 0, 2, 1, size=0.4), 2, 1), (jno.Shape.box(0, 0, 0, 2, 1, 1, size=0.5), 3, 2)],
        ids=["triangles", "tets"],
    )
    def test_every_facet_reports_its_own_facets(self, shape, n_local, n_ridge_nodes):
        """An interior edge has two endpoints; an interior triangle has three edges. Both tables
        must index into a ridge list whose entries really are shared sub-facets of those facets."""
        d = shape.domain()
        t = d._facet_ridges()
        n_facets = t["nodes"].shape[0]
        assert t["facet_ridge"].shape == (n_facets, n_local)
        assert t["ridge_nodes"].shape[1] == n_ridge_nodes
        assert t["facet_ridge"].min() >= 0 and t["facet_ridge"].max() < t["ridge_nodes"].shape[0]
        # A facet's own corners must contain every node of each ridge it claims.
        for f in range(0, n_facets, max(n_facets // 40, 1)):
            for r in t["facet_ridge"][f]:
                assert np.isin(t["ridge_nodes"][r], t["nodes"][f]).all()
        # No facet may list the same ridge twice, or a pair sum would count it against itself.
        assert all(len(set(row.tolist())) == n_local for row in t["facet_ridge"])

    def test_a_three_dimensional_ridge_is_shared_by_a_fan_not_by_two_facets(self):
        """The reason the bending term weights *pairs* rather than picking "the" neighbour: around
        a mesh edge in 3-D the interior faces form a fan, and which of them the material boundary
        runs through is a property of the density, not of the mesh. Measured ~6 on a tet mesh
        against exactly 2 for the 2-D case, where a ridge is a point on a boundary curve.
        """
        d3 = jno.Shape.box(0, 0, 0, 2, 1, 1, size=0.4).domain()
        t3 = d3._facet_ridges()
        counts = np.bincount(t3["facet_ridge"].reshape(-1), minlength=t3["ridge_nodes"].shape[0])
        assert counts.max() > 2, "a tet mesh edge carries a fan of faces, not a pair"
        assert 3.0 < counts.mean() < 9.0, f"an interior edge fan averages ~6 faces, got {counts.mean():.2f}"

    def test_the_ridge_table_is_consistent_with_the_facet_table(self):
        """``_facet_ridges`` must extend ``_interior_facets`` rather than recompute it — a second
        traversal that disagreed about which facets are interior would silently bend the wrong
        boundary."""
        for shape in (jno.Shape.rect(0, 0, 2, 1, size=0.4), jno.Shape.box(0, 0, 0, 2, 1, 1, size=0.5)):
            d = shape.domain()
            base, ext = d._interior_facets(), d._facet_ridges()
            assert np.array_equal(base["cells"], ext["cells"])
            assert np.array_equal(base["nodes"], ext["nodes"])


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

    # --- tetrahedra. The reanalysis is what keeps a deformable-mesh run honest, so it has to
    # --- exist in 3-D before a 3-D design does; `_locate_in_cells` is already dimension-generic
    # --- on a simplex, so these pin that the wrapper agrees rather than that a new algorithm works.

    def test_a_constant_field_survives_transfer_on_tets(self):
        """Also the regression test for the candidate-search width, which is why the sizes are exact.

        Both meshes tile the same box, so EVERY target centroid lies in some source tet and a
        constant must come back constant -- any element taking ``outside`` is a point location that
        missed. On this configuration one of 767 did: the target centroid at (1.545, 0.996, 1.015)
        had its containing tet at rank exactly 32 in the centroid ordering, and the k=32 search
        looks at ranks 0-31. A tet is pointier than a triangle, so its centroid sits further from
        parts of it and the ranking is looser than the 2-D default assumes.
        """
        coarse = jno.Shape.box(0, 0, 0, 4, 2, 2, size=1.0).domain()
        fine = jno.Shape.box(0, 0, 0, 4, 2, 2, size=0.5).domain()
        n_c, n_f = int(coarse._cells_p1().shape[0]), int(fine._cells_p1().shape[0])
        assert coarse._cells_p1().shape[1] == 4, "this must be a tetrahedral mesh"
        assert n_f > 2 * n_c, f"the target must be genuinely finer ({n_c} -> {n_f})"
        out = coarse.transfer_cell_field(np.full(n_c, 0.7), fine)
        assert out.shape == (n_f,)
        np.testing.assert_allclose(out, 0.7, atol=1e-12)

    def test_a_slab_lands_in_the_right_place_on_tets(self):
        """A slab transfers to a slab: the geometry has to survive, not merely the values.

        The margins are one source cell wide on purpose. The slab is defined by which SOURCE
        centroid falls inside it, so its real boundary is a staircase of amplitude ``h_src``; a
        target centroid closer than that to the nominal interface can legitimately land in a source
        cell on the other side. Asserting inside a tighter band would be asserting that the coarse
        mesh resolves the plane exactly, which it does not.
        """
        h = 0.5
        coarse = jno.Shape.box(0, 0, 0, 4, 2, 6, size=h).domain()
        fine = jno.Shape.box(0, 0, 0, 4, 2, 6, size=h / 2).domain()
        c_cen = np.asarray(coarse.mesh.points)[:, :3][coarse._cells_p1()].mean(axis=1)
        slab = np.where(np.abs(c_cen[:, 2] - 3.0) < 1.5, 1.0, 0.0)
        assert 0.1 < slab.mean() < 0.9, "the slab must be a real subset for this to test anything"

        out = coarse.transfer_cell_field(slab, fine)
        f_cen = np.asarray(fine.mesh.points)[:, :3][fine._cells_p1()].mean(axis=1)
        deep_in = np.abs(f_cen[:, 2] - 3.0) < 1.5 - 1.5 * h
        deep_out = np.abs(f_cen[:, 2] - 3.0) > 1.5 + 1.5 * h
        assert deep_in.any() and deep_out.any(), "both bands must be populated or this asserts nothing"
        assert out[deep_in].min() == 1.0, "material vanished inside the slab"
        assert out[deep_out].max() == 0.0, "material appeared outside it"
        assert out.mean() == pytest.approx(slab.mean(), abs=0.05)

    def test_deformed_source_coordinates_are_honoured_on_tets(self):
        """The 3-D half of the reanalysis contract: a mesh moved by `.trainable()` must be read on
        its DEFORMED coordinates, or the density is sampled off the geometry it was never on."""
        coarse = jno.Shape.box(0, 0, 0, 4, 2, 2, size=0.8).domain()
        fine = jno.Shape.box(0, 0, 0, 4, 2, 2, size=0.4).domain()
        pts = np.asarray(coarse.mesh.points)[:, :3]
        slab = np.where(np.abs(pts[coarse._cells_p1()].mean(axis=1)[:, 2] - 1.0) < 0.4, 1.0, 0.0)

        shifted = pts.copy()
        interior = (pts[:, 2] > 1e-9) & (pts[:, 2] < 2.0 - 1e-9)
        shifted[interior, 2] += 0.3

        same = coarse.transfer_cell_field(slab, fine)
        moved = coarse.transfer_cell_field(slab, fine, points=shifted)
        assert not np.allclose(same, moved), "moving the source nodes must move the field"
        assert moved.sum() == pytest.approx(same.sum(), rel=0.30)

    def test_a_target_of_a_different_dimension_is_refused(self):
        """Both meshes carry `(n_cells,)` fields, so a dimension mismatch would otherwise reach the
        point locator as a shape error from three frames down."""
        box = jno.Shape.box(0, 0, 0, 2, 2, 2, size=1.0).domain()
        rect = jno.Shape.rect(0, 0, 2, 2, size=1.0).domain()
        with pytest.raises(ValueError, match="cannot cross dimensions"):
            box.transfer_cell_field(np.ones(int(box._cells_p1().shape[0])), rect)


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


class TestPatchFilterOnTets:
    """The patch criterion on tetrahedra — eq. (17)-(19) walked over EDGE fans.

    Around an interior edge a tet has exactly two faces containing both endpoints, so the fan's
    dual graph is 2-regular and the walk eq. (18) needs exists without any angular sort. The vertex
    patch, which is what 2-D uses, has a 3-regular dual in 3-D and no total order at all -- and at
    ``4T/V ~ 27`` elements the criterion has no contrast left anyway
    (``tests/test_patch_filter_scaling.py``). The edge fan is ``6T/E ~ 5.2``, the regime where the
    formula is sharpest, so it transfers verbatim: the same kernel, a different index map.
    """

    @staticmethod
    def _box(size=0.5):
        d = jno.Shape.box(0, 0, 0, 3, 2, 2, size=size).domain()
        return d, np.asarray(d._cells_p1()), d._patch_topology()

    def test_a_tet_belongs_to_six_edge_fans(self):
        d, cells, topo = self._box()
        assert cells.shape[1] == 4, "this must be a tetrahedral mesh"
        assert topo["others"].shape[:2] == (cells.shape[0], 6)
        assert topo["size"].shape == topo["boundary"].shape == (cells.shape[0], 6)

    def test_the_fan_is_the_size_the_element_count_predicts(self):
        """``6T/E`` with ``T ~ 6.8V`` and ``E ~ 7.8V`` puts an interior fan near 5.2 elements — the
        same regime as the paper's 2-D vertex patch, which is why nothing needs recalibrating."""
        _d, _cells, topo = self._box()
        interior = topo["size"][~topo["boundary"]]
        assert interior.size > 0, "the mesh must contain interior edges"
        assert 4.0 < interior.mean() < 7.0, f"interior fans average {interior.mean():.2f} elements"

    def test_every_member_of_a_fan_actually_contains_the_edge(self):
        """The defining property: a fan is the elements around ONE edge, so every member must carry
        both of its endpoints. A mis-keyed fan would still produce plausible-looking arrays."""
        _d, cells, topo = self._box()
        others, size = topo["others"], topo["size"]
        for k in range(0, cells.shape[0], 7):  # stride: the property is per-slot, not statistical
            for slot, (i, j) in enumerate(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))):
                n = int(size[k, slot])
                if n < 2:
                    continue
                edge = {int(cells[k][i]), int(cells[k][j])}
                for nb in others[k, slot, : n - 1]:
                    assert edge <= set(cells[int(nb)]), f"cell {nb} is in edge {edge}'s fan but omits it"

    def test_an_interior_fan_starts_and_ends_on_a_face_neighbour(self):
        """``rho_{k,1}`` and ``rho_{k,N-1}`` are the two elements sharing a FACET with ``k`` -- that
        is what eq. (18)'s first and last terms mean, and it holds only if the walk is ordered.

        Interior fans only. A boundary fan is an open chain, so its wrap-around pair is not a face
        neighbour -- which is exactly the case ``boundary=True`` marks for the Fig. 2c-d rule, and
        the shipped 2-D builder behaves the same way.
        """
        _d, cells, topo = self._box()
        others, size, bnd = topo["others"], topo["size"], topo["boundary"]
        checked = 0
        for k in range(cells.shape[0]):
            for slot in range(6):
                n = int(size[k, slot])
                if n < 2 or bnd[k, slot]:
                    continue
                for nb in (others[k, slot, 0], others[k, slot, n - 2]):
                    assert len(set(cells[k]) & set(cells[int(nb)])) == 3, (
                        f"cell {k}'s fan neighbour {nb} shares only "
                        f"{len(set(cells[k]) & set(cells[int(nb)]))} vertices, so not a face"
                    )
                    checked += 1
        assert checked > 500, f"only {checked} interior fan ends checked; the test would be weak"

    def test_no_element_appears_in_its_own_fan(self):
        _d, cells, topo = self._box()
        assert not (topo["others"] == np.arange(cells.shape[0])[:, None, None]).any()

    def test_the_size_counts_the_reference_element_too(self):
        """``N`` includes ``k``; ``others`` does not. eq. (18)'s exponent is ``1/(N-2)``, so an
        off-by-one here changes every value the filter produces."""
        _d, _cells, topo = self._box()
        listed = (topo["others"] >= 0).sum(axis=-1)
        assert np.array_equal(topo["size"], listed + 1)

    def test_the_vectorised_filter_matches_the_literal_formula_on_tets(self):
        """The same eq. (18) transcription the 2-D tests use, driven over edge fans."""
        d, cells, topo = self._box(size=0.7)
        n_cells = cells.shape[0]
        filt = d.patch_filter()
        rng = np.random.default_rng(0)
        for r in (np.full(n_cells, 0.4), rng.uniform(0.0, 1.0, n_cells), (rng.random(n_cells) > 0.6).astype(float)):
            expected = np.empty_like(r)
            for k in range(n_cells):
                fs = []
                for slot in range(6):
                    n = int(topo["size"][k, slot])
                    if n < 3:
                        continue
                    fs.append(
                        _f_patch_reference(
                            float(r[k]),
                            [float(r[j]) for j in topo["others"][k, slot, : n - 1]],
                            bool(topo["boundary"][k, slot]),
                        )
                    )
                expected[k] = r[k] * (sum(fs) / len(fs) if fs else 1.0)
            np.testing.assert_allclose(np.asarray(filt(jnp.asarray(r))), expected, atol=1e-7)

    def test_a_full_density_field_passes_through_untouched_on_tets(self):
        """rho = 1 everywhere has no bad configuration, so the filter must be the identity -- the
        sharpest single check that the fan walk and the -1 padding are right, since any mis-indexed
        neighbour reads a padded zero and pulls the product below one."""
        d, cells, _topo = self._box()
        out = np.asarray(d.patch_filter()(jnp.ones(cells.shape[0], dtype=jnp.float64)))
        np.testing.assert_allclose(out, 1.0, atol=1e-12)

    def test_it_suppresses_an_isolated_element_on_a_real_tet_mesh(self):
        """The behaviour the filter exists for, in 3-D: one dense tet in a void field is
        unbuildable and must be knocked down far enough that SIMP finishes it off."""
        d, cells, topo = self._box()
        k = int(np.where(~topo["boundary"].any(axis=1))[0][0])  # all six fans interior
        r = np.full(cells.shape[0], 1e-3)
        r[k] = 1.0
        out = np.asarray(d.patch_filter()(jnp.asarray(r)))
        assert out[k] < 0.35, f"an isolated dense tet barely moved: rho_bar = {out[k]:.4f}"
        assert out[k] ** PENAL < 0.01, "and its SIMP stiffness must be under 1 % of solid"

    def test_the_node_and_the_filter_agree_on_tets(self):
        d, cells, _topo = self._box(size=0.7)
        _r, s = d.fem_symbols(space="P0", names=("r", "s"))
        rho = jno.np.parameter(s, name="rho_patch_3d")
        r = jnp.asarray(np.random.default_rng(2).uniform(0, 1, cells.shape[0]))
        np.testing.assert_allclose(np.asarray(rho.patch().fn(r)), np.asarray(d.patch_filter()(r)), atol=0.0)
        g = jax.grad(lambda z: jnp.sum(d.patch_filter()(z)))(r)
        assert np.all(np.isfinite(np.asarray(g))) and np.any(np.asarray(g) != 0.0)
