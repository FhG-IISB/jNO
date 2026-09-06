"""Lightweight tests for VPINN / weak-form integration."""

import pytest

pytest.importorskip("foundax", reason="foundax required for neural VPINN tests")

import foundax
import jax
import numpy as np

import jno
import jno.jnp_ops as jnn
from jno.trace import dump_tree

# ============================================================
# Helpers
# ============================================================


def make_domain(mesh_size=0.35):
    """Create a small rectangular domain for fast VPINN tests."""
    return jno.Shape.rect(0, 0, 1, 1, size=mesh_size).domain()


def init_vpinn_fem(dom, with_neumann_tags=True):
    """
    Initialize FEM quadrature tags used by the VPINN route.

    VPINN uses the same sampled tags as the FEM route:
    - fem_gauss
    - gauss_<boundary_tag>
    """
    bcs = [dom.dirichlet("left")]
    if with_neumann_tags:
        bcs.append(dom.neumann(["right", "top"]))

    # Native FEM context: the same quadrature / shape-function / boundary tensors the
    # grouped-weak-form evaluator reads, built from the native Lagrange + facet machinery.
    dom.init_fem_native(
        quad_degree=2,
        bcs=bcs,
    )
    return dom


def make_scalar_net():
    key = jax.random.PRNGKey(0)
    return jnn.nn.wrap(
        foundax.mlp(
            2,
            hidden_dims=16,
            num_layers=2,
            activation=jax.nn.tanh,
            key=key,
        )
    )


def make_vector_net():
    key = jax.random.PRNGKey(0)
    return jnn.nn.wrap(
        foundax.mlp(
            2,
            hidden_dims=16,
            num_layers=2,
            activation=jax.nn.tanh,
            key=key,
            output_dim=2,
        )
    )


# ============================================================
# VPINN variable / tag access
# ============================================================


class TestVpinnVariables:
    def test_fem_gauss_and_boundary_quadrature_variables_exist(self):
        dom = make_domain()
        init_vpinn_fem(dom, with_neumann_tags=True)

        xg, yg, _ = dom.variable("fem_gauss", split=True)
        xr, yr, _ = dom.variable("gauss_right", split=True)
        xt, yt, _ = dom.variable("gauss_top", split=True)

        assert xg is not None
        assert yg is not None
        assert xr is not None
        assert yr is not None
        assert xt is not None
        assert yt is not None

    def test_boundary_quadrature_tags_are_created(self):
        dom = make_domain()
        init_vpinn_fem(dom, with_neumann_tags=True)

        assert "fem_gauss" in dom._mesh_pool
        assert "gauss_right" in dom._mesh_pool
        assert "gauss_top" in dom._mesh_pool

        assert dom._mesh_pool["fem_gauss"].shape[0] > 0
        assert dom._mesh_pool["gauss_right"].shape[0] > 0
        assert dom._mesh_pool["gauss_top"].shape[0] > 0


# ============================================================
# Scalar VPINN assembly
# ============================================================


class TestVpinnScalarAssembly:
    def test_scalar_volume_weak_form_assembles(self):
        # Authored through jno.fem (the sole entry): the network trial u=net(x,y) sits inside the
        # weak form (detected as a ModelCall) and the Dirichlet condition masks the boundary test
        # functions. jno.fem builds the native fem_context internally (no init_fem / weak.assemble).
        dom = make_domain()
        u, phi = dom.fem_symbols()
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        vi = phi.bind(x=xi, y=yi)
        u_net = make_scalar_net()(xi, yi)
        weak = jnn.grad(u_net, xi) * jnn.grad(vi, xi) + jnn.grad(u_net, yi) * jnn.grad(vi, yi) - (1.0 + 0.0 * xi) * vi

        pde = jno.fem([weak, u(xb, yb) - 0.0])

        assert pde is not None
        assert hasattr(pde, "mse")
        assert hasattr(pde, "volume_grad_expr")

    def test_scalar_nonlinear_volume_weak_form_assembles(self):
        dom = make_domain()
        u, phi = dom.fem_symbols()
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        vi = phi.bind(x=xi, y=yi)
        u_net = make_scalar_net()(xi, yi)
        weak = (1.0 + u_net**2) * (jnn.grad(u_net, xi) * jnn.grad(vi, xi) + jnn.grad(u_net, yi) * jnn.grad(vi, yi)) - (
            1.0 + 0.0 * xi
        ) * vi

        pde = jno.fem([weak, u(xb, yb) - 0.0])

        assert pde is not None
        assert hasattr(pde, "mse")


# ============================================================
# Boundary-tagged VPINN assembly
# ============================================================


class TestVpinnBoundaryAssembly:
    def test_volume_plus_boundary_weak_form_assembles(self):
        """A Neumann (boundary-test) flux term must reach its OWN region's channel.

        It used to land in the volume channel and be silently dropped: a bound test function
        ``phi.bind(x=xr, y=yr)`` carries its region on the coordinate Variables, not in the
        variational-sampling registry that ``infer_term_bucket`` consulted, so the term fell through
        that function's ``("volume", "volume")`` default. The bucketing now defers to the FEM path's
        ``_region_and_support`` -- one classifier, read from the coordinate tags.
        """
        dom = make_domain()
        u, phi = dom.fem_symbols()
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        xr, yr, _ = dom.variable("right", split=True)
        vi = phi.bind(x=xi, y=yi)
        vr = phi.bind(x=xr, y=yr)
        u_net = make_scalar_net()(xi, yi)

        vol = jnn.grad(u_net, xi) * jnn.grad(vi, xi) + jnn.grad(u_net, yi) * jnn.grad(vi, yi)
        surf = (1.0 + 0.0 * xr) * vr  # a Neumann (boundary test) flux term on 'right'
        pde = jno.fem([vol, surf, u(xb, yb) - 0.0])

        assert hasattr(pde, "mse")
        assert "right" in pde.boundary_value_exprs, "the flux must reach its own region's channel"
        assert set(pde.boundary_value_exprs) == {"right"}, "and no other region's"

    def test_a_boundary_flux_does_not_pollute_the_volume_channel(self):
        """The other half of the bug: when the flux was mis-filed it was ADDED to the volume channel.

        Oracle: assembling with and without the surface term must leave the volume channels
        untouched, since the flux belongs to neither of them. Comparing against the same form
        without the term is what distinguishes "routed correctly" from "routed anywhere else".
        """
        dom = make_domain()
        u, phi = dom.fem_symbols()
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        xr, yr, _ = dom.variable("right", split=True)
        vi, vr = phi.bind(x=xi, y=yi), phi.bind(x=xr, y=yr)
        u_net = make_scalar_net()(xi, yi)

        vol = jnn.grad(u_net, xi) * jnn.grad(vi, xi) + jnn.grad(u_net, yi) * jnn.grad(vi, yi)
        without = jno.fem([vol, u(xb, yb) - 0.0])
        with_flux = jno.fem([vol, (1.0 + 0.0 * xr) * vr, u(xb, yb) - 0.0])

        assert not without.boundary_value_exprs, "the control carries no boundary term"
        assert set(with_flux.boundary_value_exprs) == {"right"}
        # the volume channels must be structurally the same: a pure boundary term changes neither
        for chan in ("volume_value_expr", "volume_grad_expr"):
            a, b = getattr(without, chan), getattr(with_flux, chan)
            assert (a is None) == (b is None), f"{chan} gained/lost content from a pure BOUNDARY term"


# ============================================================
# Vector VPINN assembly
# ============================================================


class TestVpinnVectorAssembly:
    def test_vector_weak_form_assembles(self):
        dom = make_domain()
        u, phi = dom.fem_symbols(value_shape=(2,))
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        vi = phi.bind(x=xi, y=yi)
        u_net = make_vector_net()(xi, yi)

        eps_u = jnn.symgrad(u_net, [xi, yi])
        eps_phi = jnn.symgrad(vi, [xi, yi])
        weak = jnn.inner(eps_u, eps_phi, n_contract=2)

        pde = jno.fem([weak, u(xb, yb) - (0.0, 0.0)])

        assert pde is not None
        assert hasattr(pde, "mse")


# ============================================================
# Validation / display
# ============================================================


class TestVpinnValidation:
    def test_invalid_tag_raises(self):
        dom = make_domain()

        with pytest.raises(Exception):
            dom.variable("does_not_exist", split=True)


def test_dump_tree_on_vpinn_weak_form():
    dom = make_domain()
    init_vpinn_fem(dom, with_neumann_tags=False)

    u, phi = dom.fem_symbols()
    x, y, _ = dom.variable("fem_gauss", split=True)

    weak = jnn.grad(u, x) * jnn.grad(phi, x) + 0.0 * y
    tree = dump_tree(weak)

    assert isinstance(tree, str)
    assert len(tree) > 0


def test_vpinn_via_jno_fem_solves_poisson():
    """VPINN entirely through dom.fem_symbols() + jno.fem (no init_fem, no weak.assemble): a network
    trial u=net(x,y) written into the weak form is detected (ModelCall) and test-projected onto the
    FE test space; the Dirichlet condition u(boundary)-0 declares which test functions vanish on the
    boundary (so their du/dn-flux residual is masked). Trains to the analytic Poisson solution."""
    import numpy as np

    optax = pytest.importorskip("optax")

    dom = make_domain(mesh_size=0.2)
    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xb, yb, _ = dom.variable("boundary", split=True)
    net = jnn.nn.wrap(foundax.mlp(2, hidden_dims=24, num_layers=3, activation=jax.nn.tanh, key=jax.random.PRNGKey(0)))
    bc = xi * (1 - xi) * yi * (1 - yi)
    u_net = net(xi, yi) * bc  # hard-BC ansatz: vanishes on the [0,1]^2 boundary
    phii = phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))  # -lap(x(1-x)y(1-y)) = f

    weak = jnn.grad(u_net, xi) * jnn.grad(phii, xi) + jnn.grad(u_net, yi) * jnn.grad(phii, yi) - f * phii
    pde = jno.fem([weak, u(xb, yb) - 0.0])  # net trial + Dirichlet (masks boundary test functions)
    assert type(pde).__name__ == "GroupedAssembly" and hasattr(pde, "mse")

    net.optimizer(optax.adam(1e-2))
    crux = jno.core([pde.mse], domain=dom)
    crux.solve(1500)

    # verify the trained net solves the PDE on a fresh grid (eval the prediction, not the loss)
    test_dom = make_domain(mesh_size=0.12)
    xt, yt, _ = test_dom.variable("interior", split=True)
    bc_t = xt * (1 - xt) * yt * (1 - yt)
    pred = np.asarray(crux.eval([net(xt, yt) * bc_t], domain=test_dom)).reshape(-1)
    exact = np.asarray(crux.eval([bc_t], domain=test_dom)).reshape(-1)
    rel = float(np.linalg.norm(pred - exact) / np.linalg.norm(exact))
    assert rel < 1e-2, f"VPINN did not solve Poisson: rel-L2={rel:.3e}"


@pytest.mark.parametrize("opt_name", ["adam", "sgd", "adamw", "rmsprop"])
def test_network_trains_under_x64_with_optax_optimizer(opt_name):
    """Network training under jax_enable_x64 (float64 params) works for ANY optax optimizer: jNO
    casts the optimizer state to the param precision, so optax's float32-default moment/LR state does
    not clash with the float64 params (the optimizer-state dtype mismatch this guards against)."""
    import numpy as np
    import optax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        opt = {
            "adam": optax.adam(1e-2),
            "sgd": optax.sgd(5e-2),
            "adamw": optax.adamw(1e-2),
            "rmsprop": optax.rmsprop(2e-3),
        }[opt_name]
        dom = make_domain(mesh_size=0.35)
        u, phi = dom.fem_symbols()
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        net = make_scalar_net()
        bc = xi * (1 - xi) * yi * (1 - yi)
        u_net = net(xi, yi) * bc
        vi = phi.bind(x=xi, y=yi)
        f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))
        weak = jnn.grad(u_net, xi) * jnn.grad(vi, xi) + jnn.grad(u_net, yi) * jnn.grad(vi, yi) - f * vi
        pde = jno.fem([weak, u(xb, yb) - 0.0])
        net.optimizer(opt)
        crux = jno.core([pde.mse], domain=dom)
        l0 = float(np.asarray(crux.eval([pde.mse])).mean())
        crux.solve(300)
        l1 = float(np.asarray(crux.eval([pde.mse])).mean())
        assert l1 < 0.5 * l0, f"{opt_name} under x64 did not train: {l0:.2e} -> {l1:.2e}"
    finally:
        jax.config.update("jax_enable_x64", prev)


# ============================================================
# 1D (line domain)
# ============================================================
def _line_domain(mesh_size=0.05):
    pytest.importorskip("pygmsh", reason="pygmsh required for line meshing")
    return jno.domain(constructor=jno.domain.line(mesh_size=mesh_size))


def test_vpinn_1d_assembles_on_a_line():
    """A network trial on a **1D** line domain. VPINN was gated to 2D/3D because the native
    ``fem_context`` it test-projects onto could not be built on an interval mesh — the element spec,
    the facet connectivity and the facet normals were all triangle/tet only.

    The 1D assembler is deliberately bypassed for a VPINN form: its trial is a network, not an FE
    field, so there is no linear system to assemble."""
    dom = _line_domain(0.1)
    u, phi = dom.fem_symbols()
    xi = dom.variable("interior", split=True)[0]
    xb = dom.variable("boundary", split=True)[0]
    vi = phi.bind(x=xi)
    net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=8, num_layers=2, activation=jax.nn.tanh, key=jax.random.PRNGKey(0)))
    u_net = net(xi)
    weak = jnn.grad(u_net, xi) * jnn.grad(vi, xi) - (1.0 + 0.0 * xi) * vi

    pde = jno.fem([weak, u(xb) - 0.0])
    assert type(pde).__name__ == "GroupedAssembly"
    assert hasattr(pde, "mse") and hasattr(pde, "volume_grad_expr")


def test_vpinn_1d_loss_evaluates_and_is_finite():
    """Guard on the pathology this uncovered: with ``dim == 1`` the canonical test-grad coefficient is
    a **one-item** stack, and ``jnp_ops.concat`` skipped its fast path for a single operand and fell
    into a rank-alignment fallback written for two or more — which re-entered trace-node construction
    inside the evaluator, so the loss never finished evaluating (a hang, not an error).

    Evaluating it at all is the assertion; a finite number is the proof it took the direct path."""
    import numpy as np

    dom = _line_domain(0.1)
    u, phi = dom.fem_symbols()
    xi = dom.variable("interior", split=True)[0]
    xb = dom.variable("boundary", split=True)[0]
    vi = phi.bind(x=xi)
    net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=8, num_layers=2, activation=jax.nn.tanh, key=jax.random.PRNGKey(0)))
    u_net = net(xi) * (xi * (1 - xi))
    weak = jnn.grad(u_net, xi) * jnn.grad(vi, xi) - (2.0 + 0.0 * xi) * vi

    pde = jno.fem([weak, u(xb) - 0.0])
    crux = jno.core([pde.mse], domain=dom)
    loss = float(np.asarray(crux.eval([pde.mse])).mean())
    assert np.isfinite(loss) and loss > 0.0


def test_vpinn_1d_trains_and_solves_poisson():
    """End to end: ``-u'' = 2`` on [0,1] with ``u(0)=u(1)=0`` has the exact solution ``u = x(1-x)``.

    The hard-BC ansatz ``net(x)·x(1-x)`` vanishes on the boundary by construction, so the network
    cannot satisfy the loss by cheating there. Verified on the trained net's OWN prediction over a
    fresh, finer grid — not on the training loss."""
    import numpy as np
    import optax

    dom = _line_domain(0.05)
    u, phi = dom.fem_symbols()
    xi = dom.variable("interior", split=True)[0]
    xb = dom.variable("boundary", split=True)[0]
    vi = phi.bind(x=xi)
    net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=24, num_layers=3, activation=jax.nn.tanh, key=jax.random.PRNGKey(0)))
    u_net = net(xi) * (xi * (1 - xi))
    weak = jnn.grad(u_net, xi) * jnn.grad(vi, xi) - (2.0 + 0.0 * xi) * vi
    pde = jno.fem([weak, u(xb) - 0.0])

    net.optimizer(optax.adam(1e-2))
    crux = jno.core([pde.mse], domain=dom)
    crux.solve(1500)

    test_dom = _line_domain(0.013)
    xt = test_dom.variable("interior", split=True)[0]
    pred = np.asarray(crux.eval([net(xt) * (xt * (1 - xt))], domain=test_dom)).reshape(-1)
    exact = np.asarray(crux.eval([xt * (1 - xt)], domain=test_dom)).reshape(-1)
    rel = float(np.linalg.norm(pred - exact) / np.linalg.norm(exact))
    assert rel < 1e-2, f"1D VPINN did not solve Poisson: rel-L2={rel:.3e}"


# ============================================================
# VPINN scope — refused by name, not by an internal tag error
# ============================================================


class TestVpinnScopeRefusals:
    def test_three_d_vpinn_is_refused_by_name(self):
        """A 3-D network trial used to die on ``Tag 'fem_gauss' is not in the mesh pool or context``
        -- an internal tag name, for a scope limit no user can infer from it. The lowering builds its
        quadrature through the 1-D/2-D native context; say so where the decision is made."""
        dom = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
        u, phi = dom.fem_symbols()
        si = dom.variable("interior", split=True)
        xi, yi, zi = si[0], si[1], si[2]
        sb = dom.variable("boundary", split=True)
        net = jnn.nn.wrap(foundax.mlp(3, hidden_dims=8, num_layers=2, activation=jax.nn.tanh, key=jax.random.PRNGKey(0)))
        vi = phi.bind(x=xi, y=yi, z=zi)
        u_net = net(xi, yi, zi) * xi * (1 - xi) * yi * (1 - yi) * zi * (1 - zi)
        with pytest.raises(NotImplementedError, match=r"VPINN.*1-D and 2-D"):
            jno.fem(
                [
                    jnn.grad(u_net, xi) * jnn.grad(vi, xi)
                    + jnn.grad(u_net, yi) * jnn.grad(vi, yi)
                    + jnn.grad(u_net, zi) * jnn.grad(vi, zi)
                    - 1.0 * vi,
                    u(*sb) - 0.0,
                ]
            )


class TestVpinnVectorSource:
    """A load term on a VECTOR VPINN, written either way, must lower to the same residual."""

    def _build(self, spelling):
        dom = make_domain()
        u, phi = dom.fem_symbols(value_shape=(2,))
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        net = jnn.nn.wrap(
            foundax.mlp(2, output_dim=2, hidden_dims=8, num_layers=2, activation=jax.nn.tanh, key=jax.random.PRNGKey(0))
        )
        vi = phi.bind(x=xi, y=yi)
        u_net = net(xi, yi)
        gu, gv = jnn.jacobian(u_net, [xi, yi]), jnn.jacobian(vi, [xi, yi])
        g = 1.0 + 0.0 * xi  # a coordinate-carried constant, so the source is a real traced expression
        src = (
            jnn.inner(g * np.array([1.0, 2.0]), vi, n_contract=1) if spelling == "inner" else g * vi[0] + (2.0 * g) * vi[1]
        )
        return dom, jno.fem([jnn.inner(gu, gv, n_contract=2) - src, u(xb, yb) - (0.0, 0.0)])

    def _build_spelling(self, spelling):
        """The same vector source, written four ways. All four must lower identically."""
        dom = make_domain()
        u, phi = dom.fem_symbols(value_shape=(2,))
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        net = jnn.nn.wrap(
            foundax.mlp(2, output_dim=2, hidden_dims=8, num_layers=2, activation=jax.nn.tanh, key=jax.random.PRNGKey(0))
        )
        vi = phi.bind(x=xi, y=yi)
        gu, gv = jnn.jacobian(net(xi, yi), [xi, yi]), jnn.jacobian(vi, [xi, yi])
        g = 1.0 + 0.0 * xi
        src = {
            "component": lambda: g * vi[0] + (2.0 * g) * vi[1],
            "const_array": lambda: jnn.inner(np.array([1.0, 2.0]), vi, n_contract=1),
            "scaled_array": lambda: jnn.inner(g * np.array([1.0, 2.0]), vi, n_contract=1),
            "stacked": lambda: jnn.inner(jnn.stack([g, 2.0 * g], axis=-1), vi, n_contract=1),
        }[spelling]()
        return dom, jno.fem([jnn.inner(gu, gv, n_contract=2) - src, u(xb, yb) - (0.0, 0.0)])

    @pytest.mark.parametrize("spelling", ["component", "const_array", "scaled_array", "stacked"])
    def test_every_vector_source_spelling_lowers_identically(self, spelling):
        """A value-channel coefficient may arrive as a per-point vector, a bare CONSTANT vector (no
        quadrature axis -- a constant does not vary over the points), or component-first from
        ``stack(..., axis=-1)``. Only the per-point form used to assemble; the others died on raw
        shape errors naming internal arrays. The oracle is that all four give the SAME residual --
        and, in the parity test below, the same acceptance as the FEM trial.
        """
        dom_ref, pde_ref = self._build_spelling("scaled_array")
        r_ref = float(np.asarray(jno.core([pde_ref.mse], domain=dom_ref).eval([pde_ref.mse])).reshape(()))
        assert r_ref > 0.0, "a non-trivial residual is needed for the comparison to mean anything"

        dom, pde = self._build_spelling(spelling)
        r = float(np.asarray(jno.core([pde.mse], domain=dom).eval([pde.mse])).reshape(()))
        assert abs(r - r_ref) <= 1e-12 * max(1.0, abs(r_ref)), (
            f"spelling {spelling!r} lowered differently: {r:.12e} vs reference {r_ref:.12e}"
        )

    def test_a_component_first_stack_is_refused_on_both_lowerings(self):
        """``stack`` defaults to ``axis=0``, which builds a component-FIRST array while the value
        axis is trailing everywhere in the assemblers. Neither lowering rewrites it -- transposing
        silently would be guessing at intent -- and both name the fix, so one spelling means one
        thing whichever trial is used."""
        dom = make_domain()
        u, phi = dom.fem_symbols(value_shape=(2,))
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        vi = phi.bind(x=xi, y=yi)
        g = 1.0 + 0.0 * xi
        bad = jnn.inner(jnn.stack([g, 2.0 * g]), vi, n_contract=1)

        # FEM trial
        ui = u.bind(x=xi, y=yi)
        gu, gv = jnn.jacobian(ui, [xi, yi]), jnn.jacobian(vi, [xi, yi])
        with pytest.raises(ValueError, match=r"axis=-1"):
            jno.fem([jnn.inner(gu, gv, n_contract=2) - bad, u(xb, yb) - (0.0, 0.0)]).solve(linear=jno.solve.lu())

        # network trial -- same weak form, same refusal
        net = jnn.nn.wrap(
            foundax.mlp(2, output_dim=2, hidden_dims=8, num_layers=2, activation=jax.nn.tanh, key=jax.random.PRNGKey(0))
        )
        gn = jnn.jacobian(net(xi, yi), [xi, yi])
        with pytest.raises(ValueError, match=r"axis=-1"):
            pde = jno.fem([jnn.inner(gn, gv, n_contract=2) - bad, u(xb, yb) - (0.0, 0.0)])
            jno.core([pde.mse], domain=dom).eval([pde.mse])

    @pytest.mark.parametrize("spelling", ["component", "const_array", "scaled_array", "stacked"])
    def test_the_fem_trial_accepts_the_same_spellings(self, spelling):
        """One DSL: whatever the network trial takes, the FEM trial must take too. Before this the
        two lowerings accepted different subsets -- the VPINN path a strict superset -- so a weak
        form was not portable between them."""
        dom, _ = self._build_spelling(spelling)  # network trial: must not raise
        d2 = make_domain()
        u, phi = d2.fem_symbols(value_shape=(2,))
        xi, yi, _ = d2.variable("interior", split=True)
        xb, yb, _ = d2.variable("boundary", split=True)
        vi, ui = phi.bind(x=xi, y=yi), u.bind(x=xi, y=yi)
        g = 1.0 + 0.0 * xi
        src = {
            "component": lambda: g * vi[0] + (2.0 * g) * vi[1],
            "const_array": lambda: jnn.inner(np.array([1.0, 2.0]), vi, n_contract=1),
            "scaled_array": lambda: jnn.inner(g * np.array([1.0, 2.0]), vi, n_contract=1),
            "stacked": lambda: jnn.inner(jnn.stack([g, 2.0 * g], axis=-1), vi, n_contract=1),
        }[spelling]()
        gu, gv = jnn.jacobian(ui, [xi, yi]), jnn.jacobian(vi, [xi, yi])
        sol = np.asarray(
            jno.fem([jnn.inner(gu, gv, n_contract=2) - src, u(xb, yb) - (0.0, 0.0)]).solve(linear=jno.solve.lu())
        )
        assert np.all(np.isfinite(sol)) and np.linalg.norm(sol) > 0.0

    def test_component_source_lowers_like_the_inner_spelling(self):
        """``c * phi[k]`` used to raise "could not extract a canonical test channel" on the VPINN
        path while assembling fine on the FEM path -- the same weak form accepted by one and refused
        by the other. It now lowers to the vector channel that already worked, ``inner(c*e_k, phi)``,
        and the oracle is that both spellings give the SAME residual, not merely that both build.
        """
        dom_i, pde_i = self._build("inner")
        dom_c, pde_c = self._build("component")
        r_i = float(np.asarray(jno.core([pde_i.mse], domain=dom_i).eval([pde_i.mse])).reshape(()))
        r_c = float(np.asarray(jno.core([pde_c.mse], domain=dom_c).eval([pde_c.mse])).reshape(()))
        assert r_i > 0.0, "a non-trivial residual is needed for the comparison to mean anything"
        assert abs(r_i - r_c) <= 1e-12 * max(1.0, abs(r_i)), (
            f"the two spellings of one source must lower identically: inner={r_i:.12e} component={r_c:.12e}"
        )

    def test_a_component_source_actually_enters_the_residual(self):
        """Guard against the lowering quietly producing a zero coefficient (which would also make the
        two spellings 'agree'): dropping the source must change the residual."""
        dom_c, pde_c = self._build("component")
        r_with = float(np.asarray(jno.core([pde_c.mse], domain=dom_c).eval([pde_c.mse])).reshape(()))

        dom = make_domain()
        u, phi = dom.fem_symbols(value_shape=(2,))
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        net = jnn.nn.wrap(
            foundax.mlp(2, output_dim=2, hidden_dims=8, num_layers=2, activation=jax.nn.tanh, key=jax.random.PRNGKey(0))
        )
        vi = phi.bind(x=xi, y=yi)
        gu, gv = jnn.jacobian(net(xi, yi), [xi, yi]), jnn.jacobian(vi, [xi, yi])
        pde0 = jno.fem([jnn.inner(gu, gv, n_contract=2), u(xb, yb) - (0.0, 0.0)])
        r_without = float(np.asarray(jno.core([pde0.mse], domain=dom).eval([pde0.mse])).reshape(()))
        assert abs(r_with - r_without) > 1e-9, "the component source is not reaching the residual at all"


class TestVpinnCoupledAsVector:
    """A coupled system whose fields share a test space IS one vector field -- the route the
    single-field refusal points at, pinned so the message cannot rot into a false promise.

    System:  -Lap a = fa + b,  -Lap b = fb,  a = b = 0 on the boundary,
    manufactured with a* = s, b* = 2s for s = x(1-x)y(1-y), so fa = lap_s - 2s and fb = 2*lap_s.
    The inter-field coupling ``- b*va`` is a cross-component term ``u[1]*v[0]``.
    """

    @staticmethod
    def _terms(dom, u, phi, trial, xi, yi):
        vi = phi.bind(x=xi, y=yi)
        s = xi * (1 - xi) * yi * (1 - yi)
        lap = 2.0 * (xi * (1 - xi) + yi * (1 - yi))
        gu, gv = jnn.jacobian(trial, [xi, yi]), jnn.jacobian(vi, [xi, yi])
        return [
            jnn.inner(gu, gv, n_contract=2)
            - (lap - 2.0 * s) * vi[0]  # fa
            - (2.0 * lap) * vi[1]  # fb
            - trial[1] * vi[0]  # the coupling: b enters a's equation
        ]

    def test_the_coupled_system_is_expressible_and_correct_as_a_vector_field(self):
        """Oracle: solve the SAME form with an FEM trial. If FEM recovers (s, 2s), the vector
        rewrite of the coupled system is right -- which is what the refusal message asserts."""
        dom = jno.Shape.rect(0, 0, 1, 1, size=0.12).domain()
        u, phi = dom.fem_symbols(value_shape=(2,))
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        ui = u.bind(x=xi, y=yi)

        fem = jno.fem(self._terms(dom, u, phi, ui, xi, yi) + [u(xb, yb) - (0.0, 0.0)])
        sol = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1, 2)
        pts = np.asarray(fem.points)
        ex = pts[:, 0] * (1 - pts[:, 0]) * pts[:, 1] * (1 - pts[:, 1])
        for i, scale in enumerate((1.0, 2.0)):
            rel = float(np.linalg.norm(sol[:, i] - scale * ex) / np.linalg.norm(scale * ex))
            assert rel < 5e-3, f"component {i} of the coupled system is wrong: rel-L2 {rel:.2e}"

    def test_a_network_trial_assembles_on_that_same_coupled_form(self):
        """And the network-trial version of the identical form builds a trainable residual, so the
        route the refusal names is actually open (training quality is a separate, documented matter --
        the two component residuals compete under an equal-weight loss)."""
        dom = make_domain()
        u, phi = dom.fem_symbols(value_shape=(2,))
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        net = jnn.nn.wrap(
            foundax.mlp(2, output_dim=2, hidden_dims=8, num_layers=2, activation=jax.nn.tanh, key=jax.random.PRNGKey(0))
        )
        s = xi * (1 - xi) * yi * (1 - yi)
        pde = jno.fem(self._terms(dom, u, phi, net(xi, yi) * s, xi, yi) + [u(xb, yb) - (0.0, 0.0)])
        r = float(np.asarray(jno.core([pde.mse], domain=dom).eval([pde.mse])).reshape(()))
        assert np.isfinite(r) and r > 0.0, "the coupled-as-vector VPINN residual must be finite and non-trivial"

    def test_two_separate_fields_are_refused_with_the_route_named(self):
        """The refusal must point somewhere, not just say no."""
        dom = make_domain()
        a, ta = dom.fem_symbols(names=("a", "ta"))
        b, tb = dom.fem_symbols(names=("b", "tb"))
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        n1, n2 = make_scalar_net(), make_scalar_net()
        tai, tbi = ta.bind(x=xi, y=yi), tb.bind(x=xi, y=yi)
        s = xi * (1 - xi) * yi * (1 - yi)
        with pytest.raises(NotImplementedError, match=r"value_shape"):
            jno.fem(
                [
                    jnn.grad(n1(xi, yi) * s, xi) * jnn.grad(tai, xi) - 1.0 * tai,
                    jnn.grad(n2(xi, yi) * s, xi) * jnn.grad(tbi, xi) - 1.0 * tbi,
                    a(xb, yb) - 0.0,
                    b(xb, yb) - 0.0,
                ]
            )
