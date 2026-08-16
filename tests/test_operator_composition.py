"""The composition proof: the branch's primitives used together in real training runs.

Each of C3-C6 has its own unit tests. This file exists to check the thing unit tests cannot — that
they compose, on both of the two operator-learning paths:

* **generated** — `jno.noise.grf` produces a fresh input function every step, reshaped to the grid
  and reduced over *named* axes. No dataset at all.
* **stored** — a grid tensor attached in its natural `(B, H, W, C)` shape (the time axis inserted
  for you), optionally as a **lazy** handle streamed per batch, reduced over the same named axes.

Both are marked slow: they train.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno

G, B = 8, 12


class _Lazy:
    """A lazy source: `.shape` + `__getitem__`, nothing else."""

    def __init__(self, a):
        self._a = np.asarray(a)
        self.shape = self._a.shape
        self.dtype = self._a.dtype
        self.reads = 0

    def __getitem__(self, k):
        self.reads += 1
        return self._a[k]


def _grid_domain(batch=None):
    base = jno.domain(constructor=jno.domain.poseidon(nx=G, ny=G), compute_mesh_connectivity=True)
    dom = batch * base if batch else base
    return dom


def _net(in_features=1, key=0):
    import foundax

    n = jno.nn(foundax.mlp(in_features, output_dim=1, hidden_dims=32, num_layers=2, key=jax.random.PRNGKey(key)))
    n.optimizer(optax.adam(3e-3))
    return n


@pytest.mark.slow
class TestGeneratedPath:
    """C5 (grf) + C3 (named axes) — an operator trained with no dataset."""

    def test_grf_input_reshaped_to_grid_and_reduced_over_named_axes(self):
        dom = _grid_domain()
        x, y, _ = dom.variable("interior")

        f = jno.noise.grf(x, y, length_scale=0.4, modes=128)  # C5: fresh field every step
        u = _net()(f)
        # C3: reduce the GRID axes by coordinate, not by index. `reshape` bridges the point cloud
        # the evaluator sees to the grid the axes name.
        resid = jno.np.reshape(u - jno.fn.tanh(f), (G, G, 1))
        loss = jno.np.mean(resid**2, axis=(x, y))

        crux = jno.core([loss])
        stats = crux.solve(120)
        hist = np.asarray(stats.training_logs[-1]["total_loss"])
        assert np.isfinite(hist).all()
        assert hist[-1] < 0.5 * hist[0], f"did not train: {hist[0]:.4f} -> {hist[-1]:.4f}"

    def test_named_axis_reduction_equals_the_integer_form_in_a_real_run(self):
        """The composition must not change the numbers — same loss, both spellings.

        Compared inside ONE run on ONE grf node. Two separate runs would not be comparable: each
        `grf` node folds its own id into the step key, so a second node is a different realisation
        by design (that is what makes two fields in one expression independent).
        """
        dom = _grid_domain()
        x, y, _ = dom.variable("interior")
        f = jno.noise.grf(x, y, length_scale=0.4, modes=128)
        r = jno.np.reshape(_net()(f) - jno.fn.tanh(f), (G, G, 1))

        crux = jno.core([jno.np.mean(r**2, axis=(x, y)), jno.np.mean(r**2, axis=(0, 1))])
        per_constraint = np.asarray(crux.solve(20).training_logs[-1]["losses"])
        named, integer = per_constraint[:, 0], per_constraint[:, 1]
        np.testing.assert_allclose(named, integer, rtol=1e-6)


@pytest.mark.slow
class TestStoredPath:
    """C4 (natural attach) + C6 (lazy streaming) + C3 (named axes)."""

    def _data(self, rng):
        f = rng.normal(size=(B, G, G, 1)).astype(np.float32)
        return f, np.tanh(f).astype(np.float32)

    def test_eager_attach_in_natural_shape_trains(self):
        rng = np.random.default_rng(0)
        f, t = self._data(rng)
        dom = _grid_domain(B)
        x, y, _ = dom.variable("interior")
        dom.variable("_f", f)  # C4: (B, H, W, C) — the time axis is inserted
        dom.variable("_t", t)
        assert dom.context["_f"].shape == (B, 1, G, G, 1)

        _f, _t = dom.variable("_f"), dom.variable("_t")
        loss = jno.np.mean((_net()(_f) - _t) ** 2, axis=(x, y))  # C3
        stats = jno.core([loss], domain=dom).solve(120, batchsize=4)
        hist = np.asarray(stats.training_logs[-1]["total_loss"])
        assert np.isfinite(hist).all()
        assert hist[-1] < 0.5 * hist[0], f"did not train: {hist[0]:.4f} -> {hist[-1]:.4f}"

    def test_lazy_source_streams_and_matches_the_eager_run(self):
        """C6 on top: the same run, the dataset never materialized."""
        rng = np.random.default_rng(0)
        f, t = self._data(rng)
        f5, t5 = f[:, None, ...], t[:, None, ...]  # a lazy source is validated, not reshaped

        out = {}
        for mode in ("eager", "lazy"):
            dom = _grid_domain(B)
            x, y, _ = dom.variable("interior")
            src = _Lazy(f5) if mode == "lazy" else f5
            dom.variable("_f", src)
            dom.variable("_t", t5)
            loss = jno.np.mean((_net()(dom.variable("_f")) - dom.variable("_t")) ** 2, axis=(x, y))
            stats = jno.core([loss], domain=dom).solve(30, batchsize=4, offload_data=True)
            out[mode] = float(np.asarray(stats.training_logs[-1]["total_loss"])[-1])
            if mode == "lazy":
                assert src.reads > 0, "the lazy source was never read"

        assert out["lazy"] == pytest.approx(out["eager"], rel=1e-5)
