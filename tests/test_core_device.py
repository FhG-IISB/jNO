"""Tests for core.py device placement, sharding, and mesh configuration.

These tests exercise the low-level infrastructure in ``jno.core`` that
controls how parameters and data are placed on devices:

* ``_setup_parallelism`` / mesh creation
* ``_shard_params``        — parameter device placement
* ``_shard_data``          — data device placement
* ``_replicate_for_devices`` — batch-dimension tiling

All tests are intentionally compatible with a single-device (CPU) host so
they pass in CI without requiring multiple GPUs/TPUs.

GPU-specific placement tests live in ``TestGPUPlacement`` and are
automatically skipped when no CUDA device is available.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import jno
import jno.jnp_ops as jnn

# ---------------------------------------------------------------------------
# GPU skip marker — all TestGPUPlacement tests are guarded by this
# ---------------------------------------------------------------------------

_has_gpu = any(d.platform == "gpu" for d in jax.devices())
requires_gpu = pytest.mark.skipif(not _has_gpu, reason="No CUDA GPU available")


# ---------------------------------------------------------------------------
# Module-level fixture: one solver shared across all test classes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def solver():
    """Minimal 1-D Poisson solver — compiled once per test module."""
    dom = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05))
    x, *_ = dom.variable("interior")
    key = jax.random.PRNGKey(0)
    u_net = jnn.nn.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=key)
    u = u_net(x) * x * (1 - x)
    pde = jnn.laplacian(u, [x])
    return jno.core([pde.mse], dom)


# ---------------------------------------------------------------------------
# _setup_parallelism / mesh
# ---------------------------------------------------------------------------


class TestSetupParallelism:
    def test_mesh_is_mesh_instance(self, solver):
        assert isinstance(solver.mesh, Mesh)

    def test_mesh_axis_names(self, solver):
        assert solver.mesh.axis_names == ("batch", "model")

    def test_mesh_covers_all_devices(self, solver):
        """batch * model must equal total device count."""
        n = len(jax.devices())
        shape = solver.mesh.shape
        assert shape["batch"] * shape["model"] == n

    def test_param_sharding_is_named_sharding(self, solver):
        assert isinstance(solver.param_sharding, NamedSharding)

    def test_data_sharding_is_named_sharding(self, solver):
        assert isinstance(solver.data_sharding, NamedSharding)

    def test_param_sharding_uses_mesh(self, solver):
        assert solver.param_sharding.mesh is solver.mesh

    def test_data_sharding_uses_mesh(self, solver):
        assert solver.data_sharding.mesh is solver.mesh

    def test_devices_list_non_empty(self, solver):
        assert len(solver.devices) > 0

    def test_devices_match_jax_devices(self, solver):
        assert solver.devices == jax.devices()

    def test_invalid_mesh_falls_back(self):
        """A mesh_shape whose product ≠ n_devices is silently corrected."""
        n = len(jax.devices())
        # Pick a shape that cannot match unless n==6
        if n == 6:
            pytest.skip("6 devices — skip this specific mismatch test")
        dom = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        key = jax.random.PRNGKey(1)
        u_net = jnn.nn.mlp(1, output_dim=1, hidden_dims=8, num_layers=1, key=key)
        u = u_net(x)
        pde = jnn.laplacian(u, [x])
        # (2, 3) = 6 devices but we almost certainly don't have 6 → fallback
        solver_bad = jno.core([pde.mse], dom, mesh=(2, 3))
        # After fallback the mesh must still cover all devices
        shape = solver_bad.mesh.shape
        assert shape["batch"] * shape["model"] == n


# ---------------------------------------------------------------------------
# _shard_params
# ---------------------------------------------------------------------------


class TestShardParams:
    def test_output_is_jax_array(self, solver):
        params = {"w": jnp.ones((4, 2)), "b": jnp.zeros((2,))}
        out = solver._shard_params(params)
        assert isinstance(out["w"], jax.Array)
        assert isinstance(out["b"], jax.Array)

    def test_arrays_placed_on_device(self, solver):
        params = {"w": jnp.ones((4, 2))}
        out = solver._shard_params(params)
        assert len(out["w"].devices()) > 0

    def test_shape_preserved(self, solver):
        params = {"w": jnp.ones((7, 5)), "b": jnp.zeros((5,))}
        out = solver._shard_params(params)
        assert out["w"].shape == (7, 5)
        assert out["b"].shape == (5,)

    def test_replicated_when_model_dim_is_one(self, solver):
        """With a 1-D model mesh, all params must be fully replicated P()."""
        if solver.mesh.shape["model"] != 1:
            pytest.skip("Test only valid when model_dim == 1")
        params = {"w": jnp.ones((4, 2)), "b": jnp.zeros((2,))}
        out = solver._shard_params(params)
        assert out["w"].sharding.spec == P()
        assert out["b"].sharding.spec == P()

    def test_numpy_array_converted_to_jax(self, solver):
        """numpy.ndarray inputs must be promoted to jax.Array."""
        params = {"w": np.ones((3, 3), dtype=np.float32)}
        out = solver._shard_params(params)
        assert isinstance(out["w"], jax.Array)

    def test_numpy_shape_preserved(self, solver):
        params = {"w": np.ones((5, 4), dtype=np.float32)}
        out = solver._shard_params(params)
        assert out["w"].shape == (5, 4)

    def test_non_array_leaves_pass_through(self, solver):
        """Non-array leaves (bool, str) must be returned unchanged."""
        params = {"flag": True, "name": "layer0"}
        out = solver._shard_params(params)
        assert out["flag"] is True
        assert out["name"] == "layer0"

    def test_values_unchanged_after_sharding(self, solver):
        """Sharding must not alter numerical values."""
        w = jnp.arange(6, dtype=jnp.float32).reshape((2, 3))
        out = solver._shard_params({"w": w})
        assert jnp.allclose(out["w"], w)

    def test_nested_pytree(self, solver):
        """Works on nested dicts (equinox-style pytrees)."""
        params = {"layer0": {"weight": jnp.ones((4, 4)), "bias": jnp.zeros((4,))}}
        out = solver._shard_params(params)
        assert isinstance(out["layer0"]["weight"], jax.Array)


# ---------------------------------------------------------------------------
# _shard_data
# ---------------------------------------------------------------------------


class TestShardData:
    def test_time_key_fully_replicated(self, solver):
        """__time__ arrays must always be fully replicated (all dims None)."""
        data = {"__time__": jnp.ones((10, 1))}
        out = solver._shard_data(data)
        spec = out["__time__"].sharding.spec
        assert all(s is None for s in spec), f"Expected all-None spec, got {spec}"

    def test_spatial_2d_batch_sharded(self, solver):
        """2-D spatial arrays (B, D) must be sharded on the leading batch axis."""
        data = {"interior": jnp.ones((8, 2))}
        out = solver._shard_data(data)
        spec = out["interior"].sharding.spec
        assert spec[0] == "batch"

    def test_spatial_3d_batch_sharded(self, solver):
        """3-D arrays (B, T, D) must also be sharded on the leading batch axis."""
        data = {"volume": jnp.ones((8, 4, 3))}
        out = solver._shard_data(data)
        spec = out["volume"].sharding.spec
        assert spec[0] == "batch"

    def test_1d_array_batch_sharded(self, solver):
        """1-D arrays must be sharded on the single (batch) axis."""
        data = {"weights": jnp.ones((8,))}
        out = solver._shard_data(data)
        spec = out["weights"].sharding.spec
        assert spec[0] == "batch"

    def test_scalar_not_sharded(self, solver):
        """Scalar arrays (ndim == 0) must be returned as-is (no device_put)."""
        eps = jnp.float32(0.01)
        data = {"eps": eps}
        out = solver._shard_data(data)
        assert out["eps"].ndim == 0

    def test_non_jax_array_pass_through(self, solver):
        """Non-JAX values (strings, ints) must pass through unchanged."""
        data = {"label": "interior", "count": 5}
        out = solver._shard_data(data)
        assert out["label"] == "interior"
        assert out["count"] == 5

    def test_shape_preserved(self, solver):
        data = {"pts": jnp.ones((12, 3))}
        out = solver._shard_data(data)
        assert out["pts"].shape == (12, 3)

    def test_values_unchanged(self, solver):
        """Values must be numerically identical after sharding."""
        x = jnp.arange(8, dtype=jnp.float32)
        out = solver._shard_data({"x": x})
        assert jnp.allclose(out["x"], x)

    def test_mixed_dict(self, solver):
        """A dict with both __time__ and spatial arrays is handled correctly."""
        data = {
            "__time__": jnp.zeros((5, 1)),
            "interior": jnp.ones((8, 2)),
        }
        out = solver._shard_data(data)
        time_spec = out["__time__"].sharding.spec
        interior_spec = out["interior"].sharding.spec
        assert all(s is None for s in time_spec)
        assert interior_spec[0] == "batch"


# ---------------------------------------------------------------------------
# _replicate_for_devices
# ---------------------------------------------------------------------------


class TestReplicateForDevices:
    def test_tiles_leading_dim_when_too_small(self, solver):
        """An array with batch-dim 1 must be tiled to fill n_devices=4."""
        x = jnp.ones((1, 4))
        out = solver._replicate_for_devices({"x": x}, n_devices=4)
        assert out["x"].shape[0] == 4

    def test_no_tile_when_batch_already_large(self, solver):
        """Arrays whose batch-dim >= n_devices must not be modified."""
        x = jnp.ones((8, 4))
        out = solver._replicate_for_devices({"x": x}, n_devices=4)
        assert out["x"].shape[0] == 8

    def test_exact_match_not_tiled(self, solver):
        """Batch-dim == n_devices: no extra tiling."""
        x = jnp.ones((4, 4))
        out = solver._replicate_for_devices({"x": x}, n_devices=4)
        assert out["x"].shape[0] == 4

    def test_tiled_values_are_copies(self, solver):
        """Each tiled row must equal the original row."""
        x = jnp.arange(4, dtype=jnp.float32).reshape((1, 4))
        out = solver._replicate_for_devices({"x": x}, n_devices=3)
        tiled = out["x"]
        for i in range(tiled.shape[0]):
            assert jnp.allclose(tiled[i], x[0])

    def test_1d_array_tiled(self, solver):
        x = jnp.ones((1,))
        out = solver._replicate_for_devices({"x": x}, n_devices=2)
        assert out["x"].shape[0] == 2

    def test_scalar_array_not_tiled(self, solver):
        """Scalar arrays (ndim == 0) must pass through untouched."""
        s = jnp.float32(3.14)
        out = solver._replicate_for_devices({"s": s}, n_devices=4)
        assert out["s"].ndim == 0

    def test_non_array_pass_through(self, solver):
        out = solver._replicate_for_devices({"label": "batch_data"}, n_devices=2)
        assert out["label"] == "batch_data"

    def test_n_devices_one_no_tile(self, solver):
        """With n_devices=1 nothing is ever tiled."""
        x = jnp.ones((1, 5))
        out = solver._replicate_for_devices({"x": x}, n_devices=1)
        assert out["x"].shape[0] == 1


# ---------------------------------------------------------------------------
# core.__init__ post-construction state
# ---------------------------------------------------------------------------


class TestCoreInit:
    def test_models_dict_non_empty(self, solver):
        """compile() must populate solver.models with at least one entry."""
        assert len(solver.models) > 0

    def test_rng_is_jax_array(self, solver):
        assert isinstance(solver.rng, jax.Array)

    def test_rng_shape(self, solver):
        """JAX PRNG keys must have shape () or (2,) depending on backend."""
        assert solver.rng.ndim in (0, 1)

    def test_constraints_stored(self, solver):
        assert len(solver.constraints) > 0

    def test_mesh_set_after_compile(self, solver):
        assert solver.mesh is not None

    def test_devices_set_after_compile(self, solver):
        assert solver.devices is not None

    def test_default_seed_is_42_without_config(self, monkeypatch):
        """With no config seed, core uses default seed 42."""
        import jno.utils.config as cfg_module

        # Force config to empty so get_seed() returns default 42
        monkeypatch.setattr(cfg_module, "_CONFIG", {})

        dom = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        key = jax.random.PRNGKey(0)
        u_net = jnn.nn.mlp(1, output_dim=1, hidden_dims=8, num_layers=1, key=key)
        u = u_net(x)
        pde = jnn.laplacian(u, [x])
        s = jno.core([pde.mse], dom)
        assert s.seed == 42

    def test_seed_read_from_config(self, monkeypatch):
        """core uses the seed value from [jno] config when present."""
        import jno.utils.config as cfg_module

        monkeypatch.setattr(cfg_module, "_CONFIG", {"jno": {"seed": 77}})

        dom = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        key = jax.random.PRNGKey(0)
        u_net = jnn.nn.mlp(1, output_dim=1, hidden_dims=8, num_layers=1, key=key)
        u = u_net(x)
        pde = jnn.laplacian(u, [x])
        s = jno.core([pde.mse], dom)
        assert s.seed == 77


# ---------------------------------------------------------------------------
# GPU placement — skipped when no CUDA device is present
# ---------------------------------------------------------------------------


class TestGPUPlacement:
    """Verify that parameters and data are actually placed on the GPU.

    All tests in this class are skipped automatically when no CUDA-capable
    GPU is available to JAX.
    """

    @requires_gpu
    def test_default_backend_is_gpu(self):
        """When a CUDA GPU is present the JAX default backend is 'gpu'."""
        assert jax.default_backend() == "gpu"

    @requires_gpu
    def test_params_land_on_gpu_after_solve(self):
        """After solve(), all trainable parameter leaves must live on the GPU.

        This exercises the full path: _shard_params is called in compile(),
        the JIT step updates trainable in-place on the GPU, and after solve()
        core reconstructs self.models from those updated leaves.
        """
        import optax
        from jno import LearningRateSchedule as lrs

        dom = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        key = jax.random.PRNGKey(0)
        u_net = jnn.nn.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=key)
        u = u_net(x) * x * (1 - x)
        pde = jnn.laplacian(u, [x])
        s = jno.core([pde.mse], dom)
        u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))
        s.solve(5)

        gpu_leaves = [leaf for leaf in jax.tree_util.tree_leaves(s.models) if isinstance(leaf, jax.Array)]
        assert len(gpu_leaves) > 0, "No jax.Array leaves found in s.models"
        for leaf in gpu_leaves:
            assert all(d.platform == "gpu" for d in leaf.devices()), f"Model param not on GPU after solve: {leaf.devices()}"

    @requires_gpu
    def test_data_sharded_on_gpu_during_solve(self, monkeypatch):
        """_shard_data is called with GPU-placed arrays during a standard solve.

        We spy on core._shard_data to capture the dict it receives on the
        first call made from within solve().  The arrays must already be on
        the GPU (placed there by jnp.zeros / domain initialisation) and the
        outputs must also be on the GPU.
        """
        import optax
        from jno import LearningRateSchedule as lrs

        dom = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        key = jax.random.PRNGKey(1)
        u_net = jnn.nn.mlp(1, output_dim=1, hidden_dims=8, num_layers=1, key=key)
        u = u_net(x) * x * (1 - x)
        pde = jnn.laplacian(u, [x])
        s = jno.core([pde.mse], dom)
        u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))

        shard_data_outputs = []
        original_shard_data = s._shard_data

        def spy_shard_data(data):
            result = original_shard_data(data)
            shard_data_outputs.append(result)
            return result

        monkeypatch.setattr(s, "_shard_data", spy_shard_data)
        s.solve(2)

        assert len(shard_data_outputs) > 0, "_shard_data was never called during solve()"
        first_output = shard_data_outputs[0]
        for key_name, v in first_output.items():
            if isinstance(v, jax.Array) and v.ndim > 0:
                assert all(d.platform == "gpu" for d in v.devices()), f"Data '{key_name}' not on GPU after _shard_data: {v.devices()}"

    @requires_gpu
    def test_offload_data_full_dataset_stays_on_host(self, monkeypatch):
        """With offload_data=True, core.solve() stores the full dataset as plain
        NumPy arrays on the host instead of loading everything onto the GPU.

        We spy on ``np.asarray`` inside the ``jno.core`` module to intercept
        the conversion call made inside solve() and verify it receives JAX GPU
        arrays (the domain context lives on GPU) and converts them to
        host-resident numpy arrays.

        ``20 * domain`` sets ``total_samples=20``; ``batchsize=5`` is less than
        20, so the guard in solve() does not suppress the offload path.
        """
        import sys
        import optax
        from jno import LearningRateSchedule as lrs

        # jno/__init__.py re-exports the class as jno.core, so grab the
        # underlying module (jno/core.py) via sys.modules.
        core_mod = sys.modules["jno.core"]

        dom = 20 * jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        key = jax.random.PRNGKey(2)
        u_net = jnn.nn.mlp(1, output_dim=1, hidden_dims=8, num_layers=1, key=key)
        u = u_net(x) * x * (1 - x)
        pde = jnn.laplacian(u, [x])
        s = jno.core([pde.mse], dom)
        u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))

        converted_types = {}
        original_np_asarray = core_mod.np.asarray

        def spy_asarray(v, *args, **kwargs):
            result = original_np_asarray(v, *args, **kwargs)
            if isinstance(v, jax.Array) and v.ndim > 0 and isinstance(result, np.ndarray):
                converted_types[id(v)] = {
                    "input_platform": [d.platform for d in v.devices()],
                    "output_type": type(result).__name__,
                }
            return result

        monkeypatch.setattr(core_mod.np, "asarray", spy_asarray)
        s.solve(3, offload_data=True, batchsize=5)

        assert len(converted_types) > 0, "np.asarray was never called with a jax.Array during offload_data solve; " "the host-offload path may not have been reached"
        for arr_id, info in converted_types.items():
            assert "gpu" in info["input_platform"], f"Expected input to np.asarray to be a GPU array, got {info['input_platform']}"
            assert info["output_type"] == "ndarray", f"Expected np.asarray to return ndarray, got {info['output_type']}"

    @requires_gpu
    def test_offload_batch_transferred_to_gpu_each_step(self, monkeypatch):
        """During offload_data=True training, each mini-batch starts as a host
        numpy slice and is transferred to the GPU before being passed to
        _shard_data.

        We spy on _shard_data to capture its inputs during the training loop
        and verify that by the time the batch reaches _shard_data it is a
        JAX GPU array (result of jax.device_put(batch_np) inside solve()).

        ``20 * domain`` sets ``total_samples=20``; ``batchsize=5`` is less
        than 20, so the offload path is actually entered.
        """
        import optax
        from jno import LearningRateSchedule as lrs

        dom = 20 * jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        key = jax.random.PRNGKey(3)
        u_net = jnn.nn.mlp(1, output_dim=1, hidden_dims=8, num_layers=1, key=key)
        u = u_net(x) * x * (1 - x)
        pde = jnn.laplacian(u, [x])
        s = jno.core([pde.mse], dom)
        u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))

        shard_data_inputs = []
        original_shard_data = s._shard_data

        def spy_shard_data(data):
            shard_data_inputs.append(data)
            return original_shard_data(data)

        monkeypatch.setattr(s, "_shard_data", spy_shard_data)
        s.solve(3, offload_data=True, batchsize=5)

        # The first shard_data call during the offload loop receives the
        # device_put'd batch — it should be a JAX array on the GPU.
        offload_calls = shard_data_inputs[1:]  # index 0 is the trace_context call
        assert len(offload_calls) > 0, "_shard_data was never called in the training loop"
        for call_data in offload_calls[:3]:  # check first 3 loop iterations
            for key_name, v in call_data.items():
                if isinstance(v, jax.Array) and v.ndim > 0:
                    assert all(d.platform == "gpu" for d in v.devices()), f"Batch '{key_name}' not on GPU when entering _shard_data: {v.devices()}"

    @requires_gpu
    def test_solve_one_step_runs_on_gpu(self):
        """A single training step executes on the GPU and produces a finite loss."""
        import optax
        from jno import LearningRateSchedule as lrs

        dom = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        key = jax.random.PRNGKey(42)
        u_net = jnn.nn.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=key)
        u = u_net(x) * x * (1 - x)
        pde = jnn.laplacian(u, [x])
        s = jno.core([pde.mse], dom)
        u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))
        stats = s.solve(1)

        loss = stats.training_logs[-1]["total_loss"][-1]
        assert jnp.isfinite(loss), f"GPU solve produced non-finite loss: {loss}"
        # The RNG should now live on a GPU device after the step
        assert all(d.platform == "gpu" for d in s.rng.devices()), f"Expected rng on GPU after solve, got {s.rng.devices()}"
