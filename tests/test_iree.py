"""Tests for IREEModel: compile, pickle round-trip, jno.save/load, and jno.core integration."""

import pickle
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Skip the entire module if iree.runtime or iree-compile isn't installed.
ireert = pytest.importorskip("iree.runtime", reason="iree.runtime not installed")
if shutil.which("iree-compile") is None:
    pytest.skip("iree-compile not on PATH", allow_module_level=True)


import jno  # noqa: E402
import jno.numpy as jnn  # noqa: E402
from jno.utils.iree import IREEModel  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _linear(x: jnp.ndarray) -> jnp.ndarray:
    """Simple 1-D linear function — easy to verify by hand."""
    return 2.0 * x + 1.0


def _multiarg(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Two-input function to ensure *args forwarding works."""
    return x * y


@pytest.fixture(scope="module")
def linear_model():
    """Compile _linear once and reuse across tests in this module."""
    sample = (jnp.ones((4,), dtype=jnp.float32),)
    return IREEModel.compile(_linear, sample)


@pytest.fixture
def sample_input():
    return jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------


class TestCompile:
    def test_returns_iree_model(self, linear_model):
        assert isinstance(linear_model, IREEModel)

    def test_module_name_matches_function(self, linear_model):
        assert linear_model.module_name == "jit__linear"

    def test_vmfb_bytes_non_empty(self, linear_model):
        assert len(linear_model.vmfb_bytes) > 0

    def test_correct_output(self, linear_model, sample_input):
        expected = np.array(2.0 * sample_input + 1.0)
        result = np.asarray(linear_model(sample_input))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_multiarg(self):
        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        y = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32)
        model = IREEModel.compile(_multiarg, (x, y))
        expected = np.array(x * y)
        np.testing.assert_allclose(np.asarray(model(x, y)), expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Pickle round-trip
# ---------------------------------------------------------------------------


class TestPickleRoundTrip:
    def test_output_identical_after_pickle(self, linear_model, sample_input):
        before = np.asarray(linear_model(sample_input))

        blob = pickle.dumps(linear_model)
        restored = pickle.loads(blob)

        after = np.asarray(restored(sample_input))
        np.testing.assert_array_equal(before, after)

    def test_vmfb_bytes_preserved(self, linear_model):
        restored = pickle.loads(pickle.dumps(linear_model))
        assert restored.vmfb_bytes == linear_model.vmfb_bytes

    def test_module_name_preserved(self, linear_model):
        restored = pickle.loads(pickle.dumps(linear_model))
        assert restored.module_name == linear_model.module_name

    def test_device_preserved(self, linear_model):
        restored = pickle.loads(pickle.dumps(linear_model))
        assert restored.device == linear_model.device

    def test_runtime_is_rebuilt(self, linear_model):
        """Unpickled model must have live IREE runtime handles."""
        restored = pickle.loads(pickle.dumps(linear_model))
        assert restored._module is not None
        assert restored._ctx is not None

    def test_multiple_roundtrips(self, linear_model, sample_input):
        """Output stays identical over several pickle round-trips."""
        expected = np.asarray(linear_model(sample_input))
        model = linear_model
        for _ in range(3):
            model = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(np.asarray(model(sample_input)), expected)


# ---------------------------------------------------------------------------
# jno.save / jno.load round-trip
# ---------------------------------------------------------------------------


class TestJnoSaveLoad:
    def test_output_identical_after_jno_save_load(self, linear_model, sample_input, tmp_path):
        path = str(tmp_path / "model.pkl")
        before = np.asarray(linear_model(sample_input))

        jno.save(linear_model, path)
        loaded = jno.load(path)

        after = np.asarray(loaded(sample_input))
        np.testing.assert_array_equal(before, after)

    def test_loaded_is_iree_model(self, linear_model, tmp_path):
        path = str(tmp_path / "model.pkl")
        jno.save(linear_model, path)
        loaded = jno.load(path)
        assert isinstance(loaded, IREEModel)

    def test_jno_save_creates_file(self, linear_model, tmp_path):
        path = str(tmp_path / "model.pkl")
        assert not (tmp_path / "model.pkl").exists()
        jno.save(linear_model, path)
        assert (tmp_path / "model.pkl").exists()


# ---------------------------------------------------------------------------
# jno.core model integration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def laplace1d_solver():
    """Build and briefly train a 1-D Laplace PINN — compiled once per module."""
    import optax
    from jno import LearningRateSchedule as lrs

    domain = jno.domain(constructor=jno.domain.line(mesh_size=0.05))
    x, *_ = domain.variable("interior")

    u_net = jnn.nn.mlp(
        in_features=1,
        hidden_dims=16,
        num_layers=2,
        key=jax.random.PRNGKey(0),
    ).optimizer(optax.adam(1e-3))

    u = u_net(x) * x * (1 - x)
    pde = jnn.grad(jnn.grad(u, x), x) + jnn.sin(jnn.pi * x)

    solver = jno.core([pde.mse], domain)
    solver.solve(300)
    return solver, u


@pytest.fixture(scope="module")
def laplace1d_iree(laplace1d_solver):
    """Compile the trained 1-D Laplace PINN to IREEModel."""
    from jno.trace_compiler import TraceCompiler

    solver, u = laplace1d_solver

    # Snapshot trained weights and context so the closure is pure.
    models = solver.models
    context = solver.domain_data.context  # {"interior": (1, 1, N, 1)}
    rng = solver.rng

    # Build a pure array-in / array-out inference function.
    fn = TraceCompiler.compile_traced_expression(u, solver.all_ops)

    def infer(x_pts):  # x_pts: (1, 1, N, 1)
        return fn(models, {"interior": x_pts}, batchsize=None, key=rng)

    x_sample = context["interior"]  # shape (1, 1, N, 1)
    iree_model = IREEModel.compile(infer, (x_sample,))
    return iree_model, solver, u, x_sample


class TestJnoCoreIntegration:
    def test_iree_model_compiles(self, laplace1d_iree):
        iree_model, *_ = laplace1d_iree
        assert isinstance(iree_model, IREEModel)

    def test_jax_and_iree_outputs_match(self, laplace1d_iree):
        """IREE inference must match jno.core eval within float32 cross-backend tolerance."""
        iree_model, solver, u, x_sample = laplace1d_iree

        jax_out = np.asarray(solver.eval(u))  # (1, 1, N, 1)
        iree_out = np.asarray(iree_model(x_sample))  # same shape

        # IREE's LLVM-CPU backend may use fused multiply-add and different
        # vectorisation lanes, giving ~1e-5 absolute deviation on float32.
        np.testing.assert_allclose(iree_out, jax_out, rtol=1e-3, atol=1e-4)

    def test_outputs_match_after_pickle(self, laplace1d_iree):
        """IREE + pickle round-trip must still match jno.core eval."""
        iree_model, solver, u, x_sample = laplace1d_iree

        jax_out = np.asarray(solver.eval(u))
        restored = pickle.loads(pickle.dumps(iree_model))
        iree_out = np.asarray(restored(x_sample))

        np.testing.assert_allclose(iree_out, jax_out, rtol=1e-3, atol=1e-4)

    def test_outputs_match_after_jno_save_load(self, laplace1d_iree, tmp_path):
        """jno.save → jno.load round-trip must still match jno.core eval."""
        iree_model, solver, u, x_sample = laplace1d_iree

        jax_out = np.asarray(solver.eval(u))

        path = str(tmp_path / "laplace.pkl")
        jno.save(iree_model, path)
        loaded = jno.load(path)

        iree_out = np.asarray(loaded(x_sample))
        np.testing.assert_allclose(iree_out, jax_out, rtol=1e-3, atol=1e-4)
