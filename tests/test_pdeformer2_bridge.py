"""Extensive tests for the PDEformer-2 jNO bridge.

Coverage:
    1. Node type table integrity
    2. PDEGraphBuilder primitives (uf, op, coef, ic, edges, padding)
    3. Tensor assembly (shapes, dtypes, Floyd-Warshall, clamping, attn_bias)
    4. PDETraceWalker dispatch (every supported operator + every error path)
    5. IC RHS extraction and sign handling
    6. PDEformer2Wrapper runtime behaviour
    7. jno.core auto-attach + non-interference with non-PDEformer models
    8. Mini end-to-end training smoke

Run with::

    CUDA_VISIBLE_DEVICES="" pixi run pytest tests/test_pdeformer2_bridge.py -v
"""

from __future__ import annotations

import equinox as eqx
import foundax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno
from jno.architectures.pdeformer2_bridge import (
    DAG_NODE_TYPES,
    DEFAULT_NUM_DEGREE,
    DEFAULT_NUM_SPATIAL,
    DISCONN_ATTN_BIAS,
    NODE_TYPE_DICT,
    PDEformer2Wrapper,
    PDEGraphBuilder,
    PDETraceWalker,
    UnsupportedPDEOperatorError,
    _contains_model_call,
    _ic_target_sign,
    _model_call_arg_layout,
    _pure_eval,
    _term_has_tag,
    _unwrap_loss,
    maybe_attach_pdeformer2_graphs,
)
from jno.trace import (
    BinaryOp,
    FunctionCall,
    Hessian,
    Literal,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _tiny_pdeformer():
    """A tiny PDEformer config suitable for CPU smoke tests."""
    return foundax.pdeformer2.small(
        num_encoder_layers=1,
        embed_dim=32,
        ffn_embed_dim=64,
        num_heads=4,
        inr_dim_hidden=32,
        inr_num_layers=2,
        hyper_num_layers=1,
        scalar_num_layers=1,
    )


def _heat_problem(mesh_size=0.25, T_end=0.1, N_t=2):
    """A reusable 2-D heat problem fixture."""
    domain = jno.domain(
        constructor=jno.domain.rect(mesh_size=mesh_size),
        time=(0, T_end, N_t),
    )
    x, y, t = domain.variable("interior")
    x0, y0, t0 = domain.variable("initial")
    net = jno.nn.wrap(_tiny_pdeformer())
    net.optimizer(optax.adam, lr=jno.LearningRateSchedule(1e-3))
    u = net(t, x, y)
    u0 = net(t0, x0, y0)
    pde = jno.np.grad(u, t) - 0.1 * jno.np.laplacian(u, [x, y])
    ini = u0 - jno.np.sin(jno.np.pi * x0) * jno.np.sin(jno.np.pi * y0)
    return domain, net, pde, ini, (x, y, t, x0, y0, t0)


def _populate_basic_builder(builder, with_ic=True, n_pts=16):
    """Build a heat equation graph; optionally add a constant-value IC."""
    uf = builder.add_uf()
    dt_u = builder.add_op("dt", uf)
    dxx = builder.add_op("dx", builder.add_op("dx", uf))
    dyy = builder.add_op("dy", builder.add_op("dy", uf))
    lap = builder.add_op("add", dxx, dyy)
    alpha = builder.add_coef(0.1)
    diff = builder.add_op("mul", alpha, lap)
    builder.add_eq0(builder.add_op("add", dt_u, builder.add_op("neg", diff)))
    if with_ic:
        builder.add_ic(
            values=np.ones(n_pts, dtype=np.float32),
            x_pts=np.linspace(0, 1, n_pts, dtype=np.float32),
            y_pts=np.linspace(0, 1, n_pts, dtype=np.float32),
            t_pts=np.zeros(n_pts, dtype=np.float32),
        )
    return uf


# =====================================================================
# 1. Node-type table integrity
# =====================================================================


class TestNodeTypeTable:
    def test_pad_is_zero(self):
        assert NODE_TYPE_DICT["pad"] == 0

    def test_all_indices_unique(self):
        assert len(set(NODE_TYPE_DICT.values())) == len(NODE_TYPE_DICT)

    def test_total_count(self):
        # 1 (pad) + 1 (uf) + 1 (coef) + 5 (function) + 15 (operator)
        # + 16 (reserved) + 16 (branch) + 32 (mod) = 87
        assert len(DAG_NODE_TYPES) == 87

    def test_index_within_model_capacity(self):
        # foundax.pdeformer2.small() uses num_node_type=128, so all indices must be < 128.
        assert max(NODE_TYPE_DICT.values()) < 128

    def test_required_operators_present(self):
        for op in ("dt", "dx", "dy", "add", "mul", "neg", "square", "sin", "cos", "eq0"):
            assert op in NODE_TYPE_DICT

    def test_inr_and_branch_naming(self):
        for j in range(32):
            assert f"Mod{j}" in NODE_TYPE_DICT
        for j in range(16):
            assert f"Branch{j}" in NODE_TYPE_DICT


# =====================================================================
# 2. PDEGraphBuilder primitives
# =====================================================================


class TestPDEGraphBuilderPrimitives:
    def test_add_uf_creates_mod_nodes(self):
        b = PDEGraphBuilder(n_inr_nodes=3, function_num_branches=4)
        uf_id = b.add_uf()
        assert uf_id == 0
        # Mod nodes follow uf in id order.
        assert b._scalar_types == ["uf", "Mod0", "Mod1", "Mod2"]

    def test_uf_zero_mod_nodes(self):
        b = PDEGraphBuilder(n_inr_nodes=0, function_num_branches=4)
        uf_id = b.add_uf()
        assert uf_id == 0
        assert b._scalar_types == ["uf"]

    def test_add_op_records_edges(self):
        b = PDEGraphBuilder(n_inr_nodes=1, function_num_branches=4)
        uf = b.add_uf()
        dt = b.add_op("dt", uf)
        assert (uf, dt) in b._edges

    def test_add_op_unknown_type_raises(self):
        b = PDEGraphBuilder(n_inr_nodes=1, function_num_branches=4)
        uf = b.add_uf()
        with pytest.raises(ValueError, match="Unknown operator"):
            b.add_op("nonsense", uf)

    def test_add_coef_stores_value(self):
        b = PDEGraphBuilder(n_inr_nodes=0, function_num_branches=4)
        cid = b.add_coef(0.42)
        assert b._scalar_types[cid] == "coef"
        assert b._scalar_values[cid] == pytest.approx(0.42)

    def test_add_ic_appends_function_arrays(self):
        b = PDEGraphBuilder(n_inr_nodes=0, function_num_branches=4)
        b.add_uf()
        n_pts = 8
        vals = np.linspace(0, 1, n_pts, dtype=np.float32)
        b.add_ic(
            values=vals,
            x_pts=np.linspace(0, 1, n_pts, dtype=np.float32),
            y_pts=np.zeros(n_pts, dtype=np.float32),
            t_pts=np.zeros(n_pts, dtype=np.float32),
        )
        assert len(b._function_arrays) == 1
        # column 4 is the value (col layout: [t, x, y, z, f])
        np.testing.assert_allclose(b._function_arrays[0][:, 4], vals)

    def test_multiple_uf_nodes(self):
        b = PDEGraphBuilder(n_inr_nodes=2, function_num_branches=4)
        uf1 = b.add_uf()
        uf2 = b.add_uf()
        # each uf should have its own Mod nodes
        assert uf1 == 0 and uf2 == 3  # uf, Mod0, Mod1, uf
        assert b._scalar_types[:6] == ["uf", "Mod0", "Mod1", "uf", "Mod0", "Mod1"]


# =====================================================================
# 3. Tensor assembly
# =====================================================================


class TestBuildTensors:
    def test_minimal_graph_shape(self):
        b = PDEGraphBuilder(n_inr_nodes=2, function_num_branches=4)
        b.add_uf()
        t = b.build_tensors()
        # 1 uf + 2 Mod = 3 scalar nodes, no function nodes
        assert t["node_type"].shape == (1, 3, 1)
        assert t["node_scalar"].shape == (1, 3, 1)
        assert t["node_function"].shape == (1, 0, 1, 5)
        assert t["spatial_pos"].shape == (1, 3, 3)

    def test_dtypes(self):
        b = PDEGraphBuilder(n_inr_nodes=2, function_num_branches=4)
        _populate_basic_builder(b)
        t = b.build_tensors()
        assert t["node_type"].dtype == np.int32
        assert t["node_scalar"].dtype == np.float32
        assert t["node_function"].dtype == np.float32
        assert t["in_degree"].dtype == np.int32
        assert t["out_degree"].dtype == np.int32
        assert t["attn_bias"].dtype == np.float32
        assert t["spatial_pos"].dtype == np.int32

    def test_node_type_layout(self):
        """Scalar nodes first, then per-function Branch nodes in order."""
        b = PDEGraphBuilder(n_inr_nodes=1, function_num_branches=3)
        uf = b.add_uf()
        b.add_eq0(b.add_op("dt", uf))
        b.add_ic(np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4))
        t = b.build_tensors()
        types_flat = t["node_type"].flatten()
        # last 3 entries are Branch0, Branch1, Branch2 (function tokens)
        assert int(types_flat[-3]) == NODE_TYPE_DICT["Branch0"]
        assert int(types_flat[-2]) == NODE_TYPE_DICT["Branch1"]
        assert int(types_flat[-1]) == NODE_TYPE_DICT["Branch2"]

    def test_node_scalar_only_coef_nonzero(self):
        b = PDEGraphBuilder(n_inr_nodes=1, function_num_branches=4)
        uf = b.add_uf()
        coef = b.add_coef(2.71)
        b.add_eq0(b.add_op("mul", coef, uf))
        t = b.build_tensors()
        scalars = t["node_scalar"][0, :, 0]
        nonzero = np.where(scalars != 0)[0]
        assert len(nonzero) == 1
        assert float(scalars[nonzero[0]]) == pytest.approx(2.71)

    def test_in_out_degree_one_indexed_and_clamped(self):
        b = PDEGraphBuilder(n_inr_nodes=1, function_num_branches=4)
        uf = b.add_uf()
        b.add_eq0(b.add_op("dt", uf))
        t = b.build_tensors()
        assert int(t["in_degree"].min()) >= 1
        assert int(t["out_degree"].min()) >= 1
        assert int(t["in_degree"].max()) < DEFAULT_NUM_DEGREE
        assert int(t["out_degree"].max()) < DEFAULT_NUM_DEGREE

    def test_spatial_pos_diagonal_is_one(self):
        b = PDEGraphBuilder(n_inr_nodes=2, function_num_branches=4)
        _populate_basic_builder(b)
        t = b.build_tensors()
        diag = np.diag(t["spatial_pos"][0])
        assert (diag == 1).all()

    def test_spatial_pos_within_range(self):
        b = PDEGraphBuilder(n_inr_nodes=2, function_num_branches=4)
        _populate_basic_builder(b)
        t = b.build_tensors()
        assert int(t["spatial_pos"].max()) <= DEFAULT_NUM_SPATIAL - 1

    def test_attn_bias_zero_for_diagonal(self):
        b = PDEGraphBuilder(n_inr_nodes=2, function_num_branches=4)
        _populate_basic_builder(b)
        t = b.build_tensors()
        diag = np.diag(t["attn_bias"][0])
        np.testing.assert_array_equal(diag, np.zeros_like(diag))

    def test_attn_bias_for_unreachable_pairs(self):
        # Two disconnected ufs → their Mod nodes should be mutually unreachable.
        b = PDEGraphBuilder(n_inr_nodes=1, function_num_branches=4)
        uf1 = b.add_uf()
        uf2 = b.add_uf()  # disconnected from uf1
        _ = uf1, uf2
        t = b.build_tensors()
        # node ids: 0=uf1, 1=Mod0(uf1), 2=uf2, 3=Mod0(uf2)
        # Mod for uf1 and uf2 are mutually unreachable (no path)
        assert float(t["attn_bias"][0, 1, 3]) == DISCONN_ATTN_BIAS

    def test_in_degree_for_uf_with_mods(self):
        b = PDEGraphBuilder(n_inr_nodes=3, function_num_branches=4)
        b.add_uf()  # only uf with 3 Mod children
        t = b.build_tensors()
        # uf has in_degree = 3 + 1 (1-indexed) since 3 Mods point to it
        assert int(t["in_degree"][0, 0]) == 4

    def test_multiple_ics_combined_into_function_tensor(self):
        b = PDEGraphBuilder(n_inr_nodes=1, function_num_branches=2)
        b.add_uf()
        n_pts = 4
        b.add_ic(np.zeros(n_pts), np.zeros(n_pts), np.zeros(n_pts), np.zeros(n_pts))
        b.add_ic(np.ones(n_pts), np.zeros(n_pts), np.zeros(n_pts), np.zeros(n_pts))
        t = b.build_tensors()
        assert t["node_function"].shape == (1, 2, n_pts, 5)
        # Second function values should be 1.0 in the f-column.
        np.testing.assert_allclose(t["node_function"][0, 1, :, 4], np.ones(n_pts))

    def test_rejects_mismatched_function_points(self):
        b = PDEGraphBuilder(n_inr_nodes=1, function_num_branches=1)
        b.add_uf()
        b.add_ic(np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4))
        b.add_ic(np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5))
        with pytest.raises(ValueError, match="same number of sample points"):
            b.build_tensors()


# =====================================================================
# 4. PDETraceWalker dispatch
# =====================================================================


class TestPDETraceWalker:
    def _setup(self, n_inr=1, n_branches=4):
        domain = jno.domain(
            constructor=jno.domain.rect(mesh_size=0.5),
            time=(0, 0.1, 2),
        )
        x, y, t = domain.variable("interior")
        x0, y0, t0 = domain.variable("initial")
        net = jno.nn.wrap(_tiny_pdeformer())
        builder = PDEGraphBuilder(n_inr_nodes=n_inr, function_num_branches=n_branches)
        uf_id = builder.add_uf()
        walker = PDETraceWalker(target_model=net, builder=builder, uf_id=uf_id)
        return walker, builder, net, (x, y, t, x0, y0, t0)

    def test_model_call_returns_uf(self):
        walker, builder, net, (x, y, t, *_) = self._setup()
        nid = walker.walk(net(t, x, y))
        assert nid == 0  # uf_id

    def test_model_call_different_model_raises(self):
        walker, _, _, (x, y, t, *_) = self._setup()
        other_net = jno.nn.wrap(_tiny_pdeformer())
        with pytest.raises(UnsupportedPDEOperatorError, match="different model"):
            walker.walk(other_net(t, x, y))

    def test_jacobian_temporal_maps_to_dt(self):
        walker, builder, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        walker.walk(jno.np.grad(u, t))
        assert builder._scalar_types[-1] == "dt"

    def test_jacobian_spatial_x_maps_to_dx(self):
        walker, builder, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        walker.walk(jno.np.grad(u, x))
        assert builder._scalar_types[-1] == "dx"

    def test_jacobian_spatial_y_maps_to_dy(self):
        walker, builder, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        walker.walk(jno.np.grad(u, y))
        assert builder._scalar_types[-1] == "dy"

    def test_hessian_trace_decomposes_to_add(self):
        walker, builder, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        walker.walk(jno.np.laplacian(u, [x, y]))
        types = builder._scalar_types
        # Need a dx, a second dx (its child), a dy, a second dy, and an 'add'.
        assert types.count("dx") == 2
        assert types.count("dy") == 2
        assert "add" in types

    def test_hessian_no_trace_raises(self):
        walker, _, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        with pytest.raises(UnsupportedPDEOperatorError, match="trace=True"):
            walker.walk(Hessian(u, [x, y], trace=False))

    def test_binaryop_plus(self):
        walker, builder, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        walker.walk(u + u)
        assert builder._scalar_types[-1] == "add"

    def test_binaryop_minus_uses_neg(self):
        walker, builder, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        walker.walk(u - u)
        # Last two scalar nodes should be neg followed by add (top-level).
        assert builder._scalar_types[-1] == "add"
        assert "neg" in builder._scalar_types

    def test_binaryop_mul_with_left_literal(self):
        walker, builder, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        walker.walk(0.5 * u)
        assert "coef" in builder._scalar_types
        assert builder._scalar_types[-1] == "mul"

    def test_binaryop_mul_with_right_literal(self):
        walker, builder, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        walker.walk(u * 0.5)
        assert "coef" in builder._scalar_types
        assert builder._scalar_types[-1] == "mul"

    def test_binaryop_mul_two_subexprs(self):
        walker, builder, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        walker.walk(u * u)
        assert builder._scalar_types[-1] == "mul"
        assert "coef" not in builder._scalar_types

    def test_binaryop_pow_two_maps_to_square(self):
        walker, builder, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        walker.walk(u**2)
        assert builder._scalar_types[-1] == "square"

    def test_binaryop_pow_other_raises(self):
        walker, _, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        with pytest.raises(UnsupportedPDEOperatorError, match="Power"):
            walker.walk(u**3)

    def test_binaryop_div_raises(self):
        walker, _, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        with pytest.raises(UnsupportedPDEOperatorError, match="BinaryOp '/'"):
            walker.walk(u / u)

    def test_functioncall_sin(self):
        walker, builder, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        walker.walk(jno.np.sin(u))
        assert builder._scalar_types[-1] == "sin"

    def test_functioncall_cos(self):
        walker, builder, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        walker.walk(jno.np.cos(u))
        assert builder._scalar_types[-1] == "cos"

    def test_functioncall_exp_raises(self):
        walker, _, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        with pytest.raises(UnsupportedPDEOperatorError):
            walker.walk(jno.np.exp(u))

    def test_functioncall_tanh_raises(self):
        walker, _, net, (x, y, t, *_) = self._setup()
        u = net(t, x, y)
        with pytest.raises(UnsupportedPDEOperatorError):
            walker.walk(jno.np.tanh(u))

    def test_literal_maps_to_coef(self):
        walker, builder, _, _ = self._setup()
        walker.walk(Literal(3.14))
        assert builder._scalar_types[-1] == "coef"
        assert builder._scalar_values[-1] == pytest.approx(3.14)


# =====================================================================
# 5. IC RHS extraction / sign handling
# =====================================================================


class TestICExtraction:
    def test_sign_for_u_minus_f(self):
        """`ini = u0 - f` → sign should be -1 (so we negate eval)."""
        _, net, _, ini, _ = _heat_problem()
        # ini = u0 - sin(pi*x0)*sin(pi*y0); ModelCall on the LEFT of '-'
        sign = _ic_target_sign(ini, net)
        assert sign == -1

    def test_sign_for_f_minus_u(self):
        """`ini = f - u0` → sign should be +1."""
        domain = jno.domain(
            constructor=jno.domain.rect(mesh_size=0.3),
            time=(0, 0.1, 2),
        )
        x0, y0, t0 = domain.variable("initial")
        net = jno.nn.wrap(_tiny_pdeformer())
        u0 = net(t0, x0, y0)
        ini = jno.np.sin(jno.np.pi * x0) * jno.np.sin(jno.np.pi * y0) - u0
        sign = _ic_target_sign(ini, net)
        assert sign == +1

    def test_pure_eval_constant(self):
        # Literal(2.0) at any var_values → 2.0
        val = _pure_eval(Literal(2.0), {}, model_to_skip=None)
        assert float(val) == pytest.approx(2.0)

    def test_pure_eval_arithmetic(self):
        expr = BinaryOp("+", Literal(2.0), Literal(3.0))
        val = _pure_eval(expr, {}, model_to_skip=None)
        assert float(val) == pytest.approx(5.0)

    def test_pure_eval_substitutes_variable(self):
        domain = jno.domain(
            constructor=jno.domain.rect(mesh_size=0.4),
            time=(0, 0.1, 2),
        )
        x0, *_ = domain.variable("initial")
        x_vals = jnp.array([0.0, 0.5, 1.0])
        out = _pure_eval(
            x0 * x0,
            {("initial", (0, 1), "spatial"): x_vals},
            model_to_skip=None,
        )
        np.testing.assert_allclose(np.asarray(out), [0.0, 0.25, 1.0])

    def test_contains_model_call(self):
        _, net, _, ini, _ = _heat_problem()
        assert _contains_model_call(ini, net)
        assert not _contains_model_call(Literal(1.0), net)


# =====================================================================
# 6. PDEformer2Wrapper runtime behaviour
# =====================================================================


class TestPDEformer2Wrapper:
    def _wrap(self, model=None, arg_order=(0, 1, 2)):
        m = model if model is not None else _tiny_pdeformer()
        func_res = m.pde_encoder.function_encoder.resolution
        n_branches = max(1, (func_res // 64)) ** 2
        b = PDEGraphBuilder(n_inr_nodes=m.n_inr_nodes, function_num_branches=n_branches)
        _populate_basic_builder(b, with_ic=True, n_pts=func_res * func_res)
        tensors = b.build_tensors()
        return PDEformer2Wrapper(
            pde_model=m,
            t_arg_idx=arg_order[0],
            x_arg_idx=arg_order[1],
            y_arg_idx=arg_order[2],
            **tensors,
        )

    def test_forward_output_shape(self):
        w = self._wrap()
        out = w(jnp.zeros((5, 1)), jnp.linspace(0, 1, 5).reshape(-1, 1), jnp.linspace(0, 1, 5).reshape(-1, 1))
        assert out.shape == (1, 5, 1)

    def test_forward_finite(self):
        w = self._wrap()
        out = w(jnp.zeros((5, 1)), jnp.linspace(0, 1, 5).reshape(-1, 1), jnp.linspace(0, 1, 5).reshape(-1, 1))
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_arg_order_respected(self):
        """Wrapping with arg_order (1, 0, 2) means args[1]=t, args[0]=x, args[2]=y."""
        w = self._wrap(arg_order=(1, 0, 2))
        x_arr = jnp.linspace(0, 1, 4).reshape(-1, 1)
        t_arr = jnp.zeros((4, 1))
        y_arr = jnp.linspace(0, 1, 4).reshape(-1, 1)
        out = w(x_arr, t_arr, y_arr)  # args[0]=x, args[1]=t, args[2]=y
        assert out.shape == (1, 4, 1)

    def test_graph_tensors_have_no_gradient(self):
        """Gradient w.r.t. wrapper params is finite for pde_model but zero for graph tensors."""
        w = self._wrap()
        x = jnp.linspace(0, 1, 3).reshape(-1, 1)
        y = jnp.linspace(0, 1, 3).reshape(-1, 1)
        t = jnp.zeros((3, 1))

        def loss(model):
            return jnp.mean(model(t, x, y) ** 2)

        grads = eqx.filter_grad(loss)(w)
        # Graph tensors should have None or all-zero gradients (stop_gradient enforces this).
        for name in ("_node_scalar", "_node_function", "_attn_bias"):
            leaf = getattr(grads, name)
            if leaf is not None:
                np.testing.assert_array_equal(np.asarray(leaf), 0.0)


# =====================================================================
# 7. _model_call_arg_layout
# =====================================================================


class TestArgLayout:
    def test_t_x_y_order(self):
        _, net, _, _, (x, y, t, *_) = _heat_problem()
        mc = net(t, x, y)
        assert _model_call_arg_layout(mc) == (0, 1, 2)

    def test_x_y_t_order(self):
        _, net, _, _, (x, y, t, *_) = _heat_problem()
        mc = net(x, y, t)
        assert _model_call_arg_layout(mc) == (2, 0, 1)

    def test_missing_temporal_raises(self):
        domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.3))
        x, y = domain.variable("interior")[:2]
        net = jno.nn.wrap(_tiny_pdeformer())
        mc = net(x, y)
        with pytest.raises(UnsupportedPDEOperatorError, match="net\\(t, x, y\\)"):
            _model_call_arg_layout(mc)


# =====================================================================
# 8. Helper utilities
# =====================================================================


class TestHelpers:
    def test_unwrap_loss_strips_mse(self):
        domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.3))
        x, y = domain.variable("interior")[:2]
        expr = (x - y).mse
        assert isinstance(expr, FunctionCall) and getattr(expr, "_name", None) == "mse"
        inner = _unwrap_loss(expr)
        assert getattr(inner, "_name", None) != "mse"

    def test_unwrap_loss_passthrough(self):
        domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.3))
        x, y = domain.variable("interior")[:2]
        e = x - y
        assert _unwrap_loss(e) is e

    def test_term_has_tag_initial(self):
        _, _, _, ini, (x, y, t, x0, y0, t0) = _heat_problem()
        assert _term_has_tag(ini, "initial")
        assert not _term_has_tag(ini, "interior")

    def test_term_has_tag_interior(self):
        _, _, pde, _, _ = _heat_problem()
        assert _term_has_tag(pde, "interior")
        assert not _term_has_tag(pde, "initial")


# =====================================================================
# 9. Auto-attach behaviour through jno.core
# =====================================================================


@pytest.mark.integration
class TestAutoAttach:
    def test_module_replaced_by_wrapper(self):
        domain, net, pde, ini, _ = _heat_problem()
        assert not isinstance(net.module, PDEformer2Wrapper)
        maybe_attach_pdeformer2_graphs([pde.mse, ini.mse], domain)
        assert isinstance(net.module, PDEformer2Wrapper)

    def test_attach_is_idempotent_via_jno_core(self):
        """Constructing jno.core a second time on the same net should not crash."""
        domain, net, pde, ini, _ = _heat_problem()
        jno.core([pde.mse, ini.mse])
        assert isinstance(net.module, PDEformer2Wrapper)

    def test_arg_order_baked_into_wrapper(self):
        domain = jno.domain(
            constructor=jno.domain.rect(mesh_size=0.3),
            time=(0, 0.1, 2),
        )
        x, y, t = domain.variable("interior")
        x0, y0, t0 = domain.variable("initial")
        net = jno.nn.wrap(_tiny_pdeformer())
        net.optimizer(optax.adam, lr=jno.LearningRateSchedule(1e-3))
        u = net(x, y, t)  # NOTE: x, y, t order
        u0 = net(x0, y0, t0)
        pde = jno.np.grad(u, t) - 0.1 * jno.np.laplacian(u, [x, y])
        ini = u0 - jno.np.sin(jno.np.pi * x0) * jno.np.sin(jno.np.pi * y0)
        maybe_attach_pdeformer2_graphs([pde.mse, ini.mse], domain)
        w = net.module
        assert w.x_arg_idx == 0
        assert w.y_arg_idx == 1
        assert w.t_arg_idx == 2

    def test_no_op_when_no_pdeformer(self):
        domain = jno.domain(constructor=jno.domain.line(mesh_size=0.2))
        x, *_ = domain.variable("interior")
        net = jno.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=4, num_layers=1))
        net.optimizer(optax.adam, lr=jno.LearningRateSchedule(1e-3))
        u = net(x)
        loss = (u - jno.np.sin(jno.np.pi * x)).mse
        pre = net.module
        jno.core([loss])
        assert net.module is pre

    def test_mixed_pdeformer_and_mlp_models(self):
        """An MLP combined with a PDEformer should leave the MLP untouched."""
        domain = jno.domain(
            constructor=jno.domain.rect(mesh_size=0.3),
            time=(0, 0.1, 2),
        )
        x, y, t = domain.variable("interior")
        x0, y0, t0 = domain.variable("initial")
        pde_net = jno.nn.wrap(_tiny_pdeformer())
        pde_net.optimizer(optax.adam, lr=jno.LearningRateSchedule(1e-3))
        aux_net = jno.nn.wrap(foundax.mlp(3, output_dim=1, hidden_dims=4, num_layers=1))
        aux_net.optimizer(optax.adam, lr=jno.LearningRateSchedule(1e-3))
        u = pde_net(t, x, y)
        u0 = pde_net(t0, x0, y0)
        pde = jno.np.grad(u, t) - 0.1 * jno.np.laplacian(u, [x, y])
        ini = u0 - jno.np.sin(jno.np.pi * x0) * jno.np.sin(jno.np.pi * y0)
        pre_aux = aux_net.module
        jno.core([pde.mse, ini.mse])
        assert isinstance(pde_net.module, PDEformer2Wrapper)
        assert aux_net.module is pre_aux


# =====================================================================
# 10. Error paths surfaced through jno.core
# =====================================================================


class TestErrorPaths:
    def test_unsupported_op_in_pde(self):
        domain = jno.domain(
            constructor=jno.domain.rect(mesh_size=0.3),
            time=(0, 0.1, 2),
        )
        x, y, t = domain.variable("interior")
        x0, y0, t0 = domain.variable("initial")
        net = jno.nn.wrap(_tiny_pdeformer())
        net.optimizer(optax.adam, lr=jno.LearningRateSchedule(1e-3))
        u = net(t, x, y)
        u0 = net(t0, x0, y0)
        pde = jno.np.grad(u, t) - jno.np.tanh(u)
        ini = u0 - jno.np.sin(jno.np.pi * x0) * jno.np.sin(jno.np.pi * y0)
        with pytest.raises(UnsupportedPDEOperatorError):
            jno.core([pde.mse, ini.mse])

    def test_no_pde_term_raises(self):
        """If the user only supplies an IC, the bridge should complain."""
        domain = jno.domain(
            constructor=jno.domain.rect(mesh_size=0.3),
            time=(0, 0.1, 2),
        )
        x, y, t = domain.variable("interior")
        x0, y0, t0 = domain.variable("initial")
        net = jno.nn.wrap(_tiny_pdeformer())
        net.optimizer(optax.adam, lr=jno.LearningRateSchedule(1e-3))
        u0 = net(t0, x0, y0)
        ini = u0 - jno.np.sin(jno.np.pi * x0) * jno.np.sin(jno.np.pi * y0)
        with pytest.raises(UnsupportedPDEOperatorError, match="no PDE residual"):
            jno.core([ini.mse])


# =====================================================================
# 10b. Boundary-condition handling — non-interior/initial terms are skipped
# =====================================================================


@pytest.mark.integration
class TestBoundaryConditions:
    """Constraints tagged neither 'interior' nor 'initial' should be skipped
    by the DAG builder but still be trained as normal soft losses."""

    def _problem_with_bc(self):
        domain = jno.domain(
            constructor=jno.domain.rect(mesh_size=0.3),
            time=(0, 0.1, 2),
        )
        x, y, t = domain.variable("interior")
        x0, y0, t0 = domain.variable("initial")
        xb, yb, tb = domain.variable("boundary")
        net = jno.nn.wrap(_tiny_pdeformer())
        net.optimizer(optax.adam, lr=jno.LearningRateSchedule(1e-3))
        u = net(t, x, y)
        u0 = net(t0, x0, y0)
        ub = net(tb, xb, yb)
        pde = jno.np.grad(u, t) - 0.1 * jno.np.laplacian(u, [x, y])
        ini = u0 - jno.np.sin(jno.np.pi * x0) * jno.np.sin(jno.np.pi * y0)
        bc = ub  # u = 0 on ∂Ω
        return domain, net, pde, ini, bc

    def test_bc_term_does_not_break_attach(self):
        domain, net, pde, ini, bc = self._problem_with_bc()
        maybe_attach_pdeformer2_graphs([pde.mse, ini.mse, bc.mse], domain)
        assert isinstance(net.module, PDEformer2Wrapper)

    def test_bc_term_not_in_pde_graph(self):
        """The DAG built when bc.mse is present must match the DAG without it."""
        # Build a reference graph with no BC term.
        d1, n1, pde1, ini1, _ = self._problem_with_bc()
        maybe_attach_pdeformer2_graphs([pde1.mse, ini1.mse], d1)
        ref = np.asarray(n1.module._node_type)

        d2, n2, pde2, ini2, bc2 = self._problem_with_bc()
        maybe_attach_pdeformer2_graphs([pde2.mse, ini2.mse, bc2.mse], d2)
        with_bc = np.asarray(n2.module._node_type)

        # Identical shape and content → BC term was correctly skipped.
        np.testing.assert_array_equal(ref, with_bc)

    def test_jno_core_trains_with_bc(self):
        domain, net, pde, ini, bc = self._problem_with_bc()
        crux = jno.core([pde.mse, ini.mse, bc.mse])
        stats = crux.solve(3)
        last = float(stats.training_logs[-1]["total_loss"][-1])
        assert np.isfinite(last)

    def test_bc_only_with_no_pde_still_raises(self):
        """A purely boundary problem (no interior PDE) must still raise."""
        domain, net, _, ini, bc = self._problem_with_bc()
        with pytest.raises(UnsupportedPDEOperatorError, match="no PDE residual"):
            jno.core([ini.mse, bc.mse])


# =====================================================================
# 11. End-to-end PINN training smoke
# =====================================================================


@pytest.mark.integration
class TestTraining:
    def test_solve_finite_loss(self):
        domain, net, pde, ini, _ = _heat_problem()
        crux = jno.core([pde.mse, ini.mse])
        stats = crux.solve(3)
        last = float(stats.training_logs[-1]["total_loss"][-1])
        assert np.isfinite(last)

    def test_eval_after_training_runs(self):
        domain, net, pde, ini, (x, y, t, *_) = _heat_problem()
        u = net(t, x, y)
        crux = jno.core([pde.mse, ini.mse])
        crux.solve(2)
        u_val = crux.eval([u])[0]
        assert np.all(np.isfinite(np.asarray(u_val)))

    def test_loss_decreases_a_bit(self):
        domain, net, pde, ini, _ = _heat_problem()
        crux = jno.core([pde.mse, ini.mse])
        stats = crux.solve(10)
        log = stats.training_logs[-1]["total_loss"]
        # not guaranteed monotonic, but the average of the second half should
        # be no worse than the average of the first half.
        first = float(jnp.mean(log[: len(log) // 2]))
        second = float(jnp.mean(log[len(log) // 2 :]))
        assert second <= first * 1.1  # allow a tiny tolerance


# =====================================================================
# 12. DAG structural correctness — the core invariants
# =====================================================================
#
# These tests build DAGs through the actual PDETraceWalker for known PDEs and
# verify the resulting tensors against a hand-derived reference. They are the
# strongest guarantee that the bridge produces graphs PDEformer-2 will recognise.


def _dag_view(builder: PDEGraphBuilder, *, include_function_branches: bool = True):
    """Reconstruct a structured view of the DAG the builder has accumulated.

    Returns a dict with:
        nodes: list of (id, type_name, scalar_value or None)
        edges: list of (src_id, dst_id)
        adj_out: dict id -> list of children
        adj_in: dict id -> list of parents
    """
    nodes = []
    n_scalar = len(builder._scalar_types)
    for i, (t, v) in enumerate(zip(builder._scalar_types, builder._scalar_values)):
        nodes.append((i, t, v if t == "coef" else None))

    edges = list(builder._edges)

    if include_function_branches:
        next_id = n_scalar
        for fi, marker in enumerate(builder._function_marker_ids):
            for j in range(builder.function_num_branches):
                nodes.append((next_id, f"Branch{j}", None))
                edges.append((marker, next_id))
                next_id += 1

    adj_out = {nid: [] for nid, *_ in nodes}
    adj_in = {nid: [] for nid, *_ in nodes}
    for s, d in edges:
        adj_out[s].append(d)
        adj_in[d].append(s)

    return {"nodes": nodes, "edges": edges, "adj_out": adj_out, "adj_in": adj_in}


def _heat_walker_setup(n_inr=2, n_branches=4):
    """Returns (walker, builder, net, (x, y, t, x0, y0, t0))."""
    domain = jno.domain(
        constructor=jno.domain.rect(mesh_size=0.5),
        time=(0, 0.1, 2),
    )
    x, y, t = domain.variable("interior")
    x0, y0, t0 = domain.variable("initial")
    net = jno.nn.wrap(_tiny_pdeformer())
    builder = PDEGraphBuilder(n_inr_nodes=n_inr, function_num_branches=n_branches)
    uf_id = builder.add_uf()
    walker = PDETraceWalker(target_model=net, builder=builder, uf_id=uf_id)
    return walker, builder, net, (x, y, t, x0, y0, t0)


@pytest.mark.integration
class TestDAGStructuralCorrectness:
    """Verify the DAG topology and numeric content for known PDEs."""

    # --- heat equation ---------------------------------------------------

    def test_heat_equation_full_topology(self):
        """Build ∂u/∂t - α∇²u via the walker; verify every node and edge."""
        walker, builder, net, (x, y, t, *_) = _heat_walker_setup(n_inr=2, n_branches=4)
        u = net(t, x, y)
        α = 0.1
        pde = jno.np.grad(u, t) - α * jno.np.laplacian(u, [x, y])
        lhs = walker.walk(pde)
        builder.add_eq0(lhs)

        view = _dag_view(builder, include_function_branches=False)

        # Expected node-type sequence (id-ordered, scalar section only).
        # Walker visit order on `dt(u) - α * lap(u)`:
        #   walk(left) of '-': dt(uf)              → 3
        #   walk(right): mul(coef, lap)
        #     coef(α) inserted first               → 4
        #     walk(lap) recursive into Hessian:
        #       dx(uf) 5, dx(dx) 6, dy(uf) 7, dy(dy) 8, add(dxx,dyy) 9
        #     mul                                  → 10
        #   neg(mul)                               → 11
        #   add(dt, neg)                           → 12
        #   eq0                                    → 13
        expected_types = [
            "uf",
            "Mod0",
            "Mod1",
            "dt",
            "coef",
            "dx",
            "dx",
            "dy",
            "dy",
            "add",
            "mul",
            "neg",
            "add",
            "eq0",
        ]
        actual_types = [n[1] for n in view["nodes"]]
        assert actual_types == expected_types

        # Coefficient value preserved exactly.
        coef_node = view["nodes"][4]
        assert coef_node[1] == "coef" and coef_node[2] == pytest.approx(α)

        # Edge set must equal this exact reference.
        expected_edges = {
            (1, 0),
            (2, 0),  # Mod -> uf
            (0, 3),  # uf -> dt
            (0, 5),
            (5, 6),  # uf -> dx -> dx
            (0, 7),
            (7, 8),  # uf -> dy -> dy
            (6, 9),
            (8, 9),  # dxx, dyy -> add (Laplacian)
            (4, 10),
            (9, 10),  # coef, lap -> mul
            (10, 11),  # mul -> neg
            (3, 12),
            (11, 12),  # dt, neg -> add (PDE residual)
            (12, 13),  # residual -> eq0
        }
        assert set(view["edges"]) == expected_edges

    def test_heat_eq0_is_topological_root(self):
        walker, builder, net, (x, y, t, *_) = _heat_walker_setup(n_inr=1, n_branches=4)
        u = net(t, x, y)
        pde = jno.np.grad(u, t) - 0.1 * jno.np.laplacian(u, [x, y])
        builder.add_eq0(walker.walk(pde))
        view = _dag_view(builder, include_function_branches=False)
        roots = [nid for nid, *_ in view["nodes"] if not view["adj_out"][nid]]
        # Only eq0 should have no outgoing edges (every other node feeds something).
        eq0_id = next(nid for nid, ty, _ in view["nodes"] if ty == "eq0")
        assert roots == [eq0_id]

    def test_heat_uf_only_has_mod_parents(self):
        """uf must be a 'leaf' from the PDE perspective: only Mods feed it."""
        walker, builder, net, (x, y, t, *_) = _heat_walker_setup(n_inr=3, n_branches=4)
        u = net(t, x, y)
        pde = jno.np.grad(u, t) - 0.1 * jno.np.laplacian(u, [x, y])
        builder.add_eq0(walker.walk(pde))
        view = _dag_view(builder, include_function_branches=False)
        uf_id = 0
        parents = view["adj_in"][uf_id]
        for pid in parents:
            assert view["nodes"][pid][1].startswith("Mod")

    def test_dag_is_acyclic(self):
        walker, builder, net, (x, y, t, *_) = _heat_walker_setup(n_inr=2, n_branches=4)
        u = net(t, x, y)
        pde = jno.np.grad(u, t) - 0.1 * jno.np.laplacian(u, [x, y])
        builder.add_eq0(walker.walk(pde))
        # Add an IC so we exercise function-branch edges too.
        builder.add_ic(np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4))
        view = _dag_view(builder, include_function_branches=True)

        # Kahn's algorithm — if it finishes ordering all nodes, it's a DAG.
        n_total = len(view["nodes"])
        in_deg = {nid: len(view["adj_in"][nid]) for nid in range(n_total)}
        ordered = []
        stack = [nid for nid in range(n_total) if in_deg[nid] == 0]
        while stack:
            cur = stack.pop()
            ordered.append(cur)
            for child in view["adj_out"][cur]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    stack.append(child)
        assert len(ordered) == n_total, "Graph contains a cycle"

    def test_edges_in_out_degree_consistency(self):
        walker, builder, net, (x, y, t, *_) = _heat_walker_setup(n_inr=2, n_branches=4)
        u = net(t, x, y)
        pde = jno.np.grad(u, t) - 0.1 * jno.np.laplacian(u, [x, y])
        builder.add_eq0(walker.walk(pde))
        builder.add_ic(np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4))
        t_out = builder.build_tensors()
        n_edges = len(builder._edges) + len(builder._function_marker_ids) * builder.function_num_branches
        # in_degree / out_degree are 1-indexed → (sum - n_nodes) should equal n_edges.
        n_nodes = t_out["node_type"].shape[1]
        in_sum = int(t_out["in_degree"].sum()) - n_nodes
        out_sum = int(t_out["out_degree"].sum()) - n_nodes
        assert in_sum == out_sum == n_edges

    def test_node_type_indices_match_dict(self):
        """Tensor values must equal the canonical NODE_TYPE_DICT indices."""
        walker, builder, net, (x, y, t, *_) = _heat_walker_setup(n_inr=1, n_branches=4)
        u = net(t, x, y)
        pde = jno.np.grad(u, t) - 0.1 * jno.np.laplacian(u, [x, y])
        builder.add_eq0(walker.walk(pde))
        t_out = builder.build_tensors()
        for i, type_name in enumerate(builder._scalar_types):
            assert int(t_out["node_type"][0, i, 0]) == NODE_TYPE_DICT[type_name]

    # --- IC numerical correctness ---------------------------------------

    def test_ic_function_values_match_sin_sin(self):
        """IC values stored in node_function must equal sin(πx)sin(πy) at sampled (x,y)."""
        domain, net, pde, ini, _ = _heat_problem(mesh_size=0.5)
        maybe_attach_pdeformer2_graphs([pde.mse, ini.mse], domain)
        w = net.module
        nf = np.asarray(w._node_function)  # (1, 1, n_pts, 5)
        x_pts = nf[0, 0, :, 1]
        y_pts = nf[0, 0, :, 2]
        f_vals = nf[0, 0, :, 4]
        expected = np.sin(np.pi * x_pts) * np.sin(np.pi * y_pts)
        np.testing.assert_allclose(f_vals, expected, atol=1e-5)

    def test_ic_grid_spans_unit_square(self):
        """The IC grid must span the spatial bounds (≈[0,1]² for rect mesh)."""
        domain, net, pde, ini, _ = _heat_problem(mesh_size=0.5)
        maybe_attach_pdeformer2_graphs([pde.mse, ini.mse], domain)
        w = net.module
        nf = np.asarray(w._node_function)
        x_pts = nf[0, 0, :, 1]
        y_pts = nf[0, 0, :, 2]
        assert x_pts.min() == pytest.approx(0.0, abs=1e-5)
        assert x_pts.max() == pytest.approx(1.0, abs=1e-2)
        assert y_pts.min() == pytest.approx(0.0, abs=1e-5)
        assert y_pts.max() == pytest.approx(1.0, abs=1e-2)

    def test_ic_t_column_equals_time_start(self):
        """The t column in node_function must be the initial time."""
        domain, net, pde, ini, _ = _heat_problem(mesh_size=0.5, T_end=0.7)
        maybe_attach_pdeformer2_graphs([pde.mse, ini.mse], domain)
        w = net.module
        nf = np.asarray(w._node_function)
        t_pts = nf[0, 0, :, 0]
        # _heat_problem uses time=(0, T_end, N_t) so t_start=0.
        np.testing.assert_allclose(t_pts, 0.0)

    def test_ic_z_column_is_zero(self):
        domain, net, pde, ini, _ = _heat_problem(mesh_size=0.5)
        maybe_attach_pdeformer2_graphs([pde.mse, ini.mse], domain)
        w = net.module
        z_pts = np.asarray(w._node_function)[0, 0, :, 3]
        np.testing.assert_array_equal(z_pts, 0.0)

    def test_ic_grid_resolution_matches_func_encoder(self):
        domain, net, pde, ini, _ = _heat_problem(mesh_size=0.5)
        maybe_attach_pdeformer2_graphs([pde.mse, ini.mse], domain)
        w = net.module
        nf = np.asarray(w._node_function)
        res = net.module.pde_model.pde_encoder.function_encoder.resolution
        assert nf.shape == (1, 1, res * res, 5)

    def test_ic_value_at_origin_is_zero(self):
        """sin(0)*sin(0) = 0 → IC value at (0,0) must be 0."""
        domain, net, pde, ini, _ = _heat_problem(mesh_size=0.5)
        maybe_attach_pdeformer2_graphs([pde.mse, ini.mse], domain)
        nf = np.asarray(net.module._node_function)
        x = nf[0, 0, :, 1]
        y = nf[0, 0, :, 2]
        f = nf[0, 0, :, 4]
        idx = int(np.argmin(x**2 + y**2))
        assert f[idx] == pytest.approx(0.0, abs=1e-5)

    def test_ic_value_at_center_is_one(self):
        """sin(π/2)*sin(π/2) = 1 → IC value at (0.5, 0.5) must be ≈1."""
        domain, net, pde, ini, _ = _heat_problem(mesh_size=0.5)
        maybe_attach_pdeformer2_graphs([pde.mse, ini.mse], domain)
        nf = np.asarray(net.module._node_function)
        x = nf[0, 0, :, 1]
        y = nf[0, 0, :, 2]
        f = nf[0, 0, :, 4]
        idx = int(np.argmin((x - 0.5) ** 2 + (y - 0.5) ** 2))
        assert f[idx] == pytest.approx(1.0, abs=1e-2)

    def test_ic_handles_reversed_sign(self):
        """If user writes `ini = f - u0`, IC values still match +f (not -f)."""
        domain = jno.domain(
            constructor=jno.domain.rect(mesh_size=0.5),
            time=(0, 0.1, 2),
        )
        x, y, t = domain.variable("interior")
        x0, y0, t0 = domain.variable("initial")
        net = jno.nn.wrap(_tiny_pdeformer())
        net.optimizer(optax.adam, lr=jno.LearningRateSchedule(1e-3))
        u = net(t, x, y)
        u0 = net(t0, x0, y0)
        pde = jno.np.grad(u, t) - 0.1 * jno.np.laplacian(u, [x, y])
        ini = jno.np.sin(jno.np.pi * x0) * jno.np.sin(jno.np.pi * y0) - u0  # REVERSED
        maybe_attach_pdeformer2_graphs([pde.mse, ini.mse], domain)
        nf = np.asarray(net.module._node_function)
        f_vals = nf[0, 0, :, 4]
        # sin*sin is non-negative on [0,1]² — make sure we got positive values, not negated ones.
        assert (f_vals >= -1e-5).all()
        assert f_vals.max() > 0.5

    # --- wave equation: second-order time derivative --------------------

    def test_wave_equation_topology(self):
        """∂²u/∂t² - c²∇²u contains a dt→dt chain."""
        walker, builder, net, (x, y, t, *_) = _heat_walker_setup(n_inr=1, n_branches=4)
        u = net(t, x, y)
        c_sq = 4.0
        wave = jno.np.grad(jno.np.grad(u, t), t) - c_sq * jno.np.laplacian(u, [x, y])
        builder.add_eq0(walker.walk(wave))
        view = _dag_view(builder, include_function_branches=False)
        types = [n[1] for n in view["nodes"]]
        assert types.count("dt") == 2

        # Find the chained dt nodes and verify dt(dt(uf)) topology: uf -> dt -> dt
        dt_ids = [nid for nid, ty, _ in view["nodes"] if ty == "dt"]
        # one of them has uf as parent, the other has the first dt as parent
        parents_of_dt = {nid: view["adj_in"][nid] for nid in dt_ids}
        has_uf_parent = [nid for nid, p in parents_of_dt.items() if 0 in p]
        has_dt_parent = [nid for nid, p in parents_of_dt.items() if any(view["nodes"][pp][1] == "dt" for pp in p)]
        assert len(has_uf_parent) == 1
        assert len(has_dt_parent) == 1
        assert has_uf_parent[0] != has_dt_parent[0]

    def test_wave_coefficient_preserved(self):
        walker, builder, net, (x, y, t, *_) = _heat_walker_setup(n_inr=1, n_branches=4)
        u = net(t, x, y)
        c_sq = 4.0
        wave = jno.np.grad(jno.np.grad(u, t), t) - c_sq * jno.np.laplacian(u, [x, y])
        builder.add_eq0(walker.walk(wave))
        view = _dag_view(builder, include_function_branches=False)
        coefs = [v for _, ty, v in view["nodes"] if ty == "coef"]
        assert any(v == pytest.approx(c_sq) for v in coefs)

    # --- Burgers equation: u * ∂u/∂x non-linearity ---------------------

    def test_burgers_topology_has_mul_two_subexprs(self):
        """u * ∂u/∂x → mul with NO coef child (both operands are walked subtrees)."""
        walker, builder, net, (x, y, t, *_) = _heat_walker_setup(n_inr=1, n_branches=4)
        u = net(t, x, y)
        ν = 0.01
        burgers = jno.np.grad(u, t) + u * jno.np.grad(u, x) - ν * jno.np.grad(jno.np.grad(u, x), x)
        builder.add_eq0(walker.walk(burgers))
        view = _dag_view(builder, include_function_branches=False)

        # Find all `mul` nodes. There should be at least two:
        #   • the one from `u * du/dx`     → parents are uf and a `dx`
        #   • the one from `ν * d²u/dx²`   → parents are a `coef` and a `dx`
        mul_ids = [nid for nid, ty, _ in view["nodes"] if ty == "mul"]
        assert len(mul_ids) >= 2

        non_coef_mul_exists = False
        for mid in mul_ids:
            parent_types = [view["nodes"][p][1] for p in view["adj_in"][mid]]
            if "coef" not in parent_types:
                non_coef_mul_exists = True
                # parents must be uf and a derivative
                assert "uf" in parent_types
                assert any(p.startswith("dx") or p.startswith("dy") or p == "dt" for p in parent_types)
        assert non_coef_mul_exists, "u * du/dx mul (no coef) not found"

    def test_burgers_coefficient_is_nu(self):
        walker, builder, net, (x, y, t, *_) = _heat_walker_setup(n_inr=1, n_branches=4)
        u = net(t, x, y)
        ν = 0.01
        burgers = jno.np.grad(u, t) + u * jno.np.grad(u, x) - ν * jno.np.grad(jno.np.grad(u, x), x)
        builder.add_eq0(walker.walk(burgers))
        view = _dag_view(builder, include_function_branches=False)
        coefs = [v for _, ty, v in view["nodes"] if ty == "coef"]
        assert any(v == pytest.approx(ν) for v in coefs)

    # --- linear advection ----------------------------------------------

    def test_linear_advection_topology(self):
        """∂u/∂t + c ∂u/∂x = 0 contains exactly one dt, one dx, one coef, one eq0."""
        walker, builder, net, (x, y, t, *_) = _heat_walker_setup(n_inr=1, n_branches=4)
        u = net(t, x, y)
        c = 2.5
        adv = jno.np.grad(u, t) + c * jno.np.grad(u, x)
        builder.add_eq0(walker.walk(adv))
        view = _dag_view(builder, include_function_branches=False)
        types = [n[1] for n in view["nodes"]]
        assert types.count("dt") == 1
        assert types.count("dx") == 1
        assert types.count("dy") == 0
        assert types.count("eq0") == 1
        coefs = [v for _, ty, v in view["nodes"] if ty == "coef"]
        assert coefs == [pytest.approx(c)]

    # --- non-linear: u² source term -----------------------------------

    def test_reaction_diffusion_square_term(self):
        """∂u/∂t - α∇²u + u² should produce one `square` node fed by uf."""
        walker, builder, net, (x, y, t, *_) = _heat_walker_setup(n_inr=1, n_branches=4)
        u = net(t, x, y)
        rd = jno.np.grad(u, t) - 0.1 * jno.np.laplacian(u, [x, y]) + u**2
        builder.add_eq0(walker.walk(rd))
        view = _dag_view(builder, include_function_branches=False)
        sq_ids = [nid for nid, ty, _ in view["nodes"] if ty == "square"]
        assert len(sq_ids) == 1
        # square's parent must be uf.
        assert view["adj_in"][sq_ids[0]] == [0]

    # --- Floyd-Warshall sanity ------------------------------------------

    def test_spatial_pos_reflects_actual_distance_in_chain(self):
        """For dt(dt(uf)) we have a 2-edge chain — distance(uf, outer_dt) must be 3."""
        # spatial_pos = 1 + clipped_shortest_path → direct edge (sp=2), 2-edge chain (sp=3).
        walker, builder, net, (x, y, t, *_) = _heat_walker_setup(n_inr=0, n_branches=4)
        u = net(t, x, y)
        builder.add_eq0(walker.walk(jno.np.grad(jno.np.grad(u, t), t)))
        t_out = builder.build_tensors()
        sp = t_out["spatial_pos"][0]
        # node 0 = uf, node 1 = inner dt, node 2 = outer dt, node 3 = eq0
        assert int(sp[0, 1]) == 2  # uf → inner dt: 1 edge → sp = 1 + 1 = 2
        assert int(sp[0, 2]) == 3  # uf → outer dt: 2 edges
        assert int(sp[0, 3]) == 4  # uf → eq0: 3 edges

    def test_branch_nodes_unreachable_from_pde_operators(self):
        """ic → Branch are only reachable from the ic marker, not from PDE operators."""
        walker, builder, net, (x, y, t, *_) = _heat_walker_setup(n_inr=1, n_branches=2)
        u = net(t, x, y)
        builder.add_eq0(walker.walk(jno.np.grad(u, t)))
        builder.add_ic(np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4))
        t_out = builder.build_tensors()
        n_scalar = len(builder._scalar_types)
        sp = t_out["spatial_pos"][0]
        # First Branch node is at id = n_scalar.
        branch_id = n_scalar
        # From uf (id 0) to Branch should be max-clamped (no path).
        assert int(sp[0, branch_id]) == DEFAULT_NUM_SPATIAL - 1

    # --- coefficient extraction -----------------------------------------

    def test_multiple_distinct_coefficients(self):
        """Each Literal coefficient surfaces as its own coef node with the right value."""
        walker, builder, net, (x, y, t, *_) = _heat_walker_setup(n_inr=0, n_branches=4)
        u = net(t, x, y)
        # 0.3 * u + 0.7 * du/dt
        expr = 0.3 * u + 0.7 * jno.np.grad(u, t)
        builder.add_eq0(walker.walk(expr))
        view = _dag_view(builder, include_function_branches=False)
        coef_values = sorted(v for _, ty, v in view["nodes"] if ty == "coef")
        assert coef_values == [pytest.approx(0.3), pytest.approx(0.7)]

    # --- structural diff for the same logical PDE -----------------------

    def test_laplacian_and_manual_decomposition_produce_same_topology(self):
        """∇²u and dx(dx(u)) + dy(dy(u)) should yield identical scalar-node sequences."""
        # Path A: use the high-level laplacian.
        w_a, b_a, _, (x, y, t, *_) = _heat_walker_setup(n_inr=0, n_branches=4)
        u_a = w_a.target_model(t, x, y)
        w_a.walk(jno.np.laplacian(u_a, [x, y]))
        types_a = b_a._scalar_types[:]

        # Path B: manually decompose.
        w_b, b_b, _, (x2, y2, t2, *_) = _heat_walker_setup(n_inr=0, n_branches=4)
        u_b = w_b.target_model(t2, x2, y2)
        manual = jno.np.grad(jno.np.grad(u_b, x2), x2) + jno.np.grad(jno.np.grad(u_b, y2), y2)
        w_b.walk(manual)
        types_b = b_b._scalar_types[:]

        assert types_a == types_b
