"""A tensor tag may be a **lazy** array-like — anything with ``.shape`` and ``__getitem__``.

h5py, zarr, tensorstore and ``np.memmap`` all qualify, and none of them is imported by jNO: the
contract is duck-typed, so there is no new dependency. The handle is stored unread and sliced one
batch at a time by ``solve(offload_data=True)``, which is the only path that can stream it.

Before this, the attach gate tested ``isinstance(sample, (np.ndarray, jnp.ndarray))`` and a
duck-typed object fell through **silently**, surfacing later as an unrelated
``tag not in self.context`` error.
"""

import numpy as np
import optax
import pytest

import jno

B, N, D = 12, 8, 1


class Counting:
    """A minimal lazy source: ``.shape`` + ``__getitem__``, and a log of what was asked for."""

    def __init__(self, arr):
        self._a = np.asarray(arr)
        self.shape = self._a.shape
        self.dtype = self._a.dtype
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return self._a[key]

    @property
    def gathers(self):
        return [k for k in self.reads if isinstance(k, np.ndarray)]


def _dom(batch=B, mesh_size=0.1):
    d = batch * jno.domain(constructor=jno.domain.line(mesh_size=mesh_size))
    x, *_ = d.variable("interior")
    return d, x


def _arr(*shape):
    return np.arange(int(np.prod(shape)), dtype=np.float32).reshape(*shape)


def _train(dom, x, steps=3, batchsize=4, **kw):
    import foundax
    import jax

    net = jno.nn(foundax.mlp(1, output_dim=1, key=jax.random.PRNGKey(0)))
    net.optimizer(optax.adam(1e-3))
    crux = jno.core([(net(x) - 0.0).mse])
    return crux.solve(steps, batchsize=batchsize, **kw)


class TestAttach:
    def test_a_duck_typed_source_is_stored_unread(self):
        dom, _ = _dom()
        src = Counting(_arr(B, N, D))
        dom.variable("_f", src)
        assert dom.context["_f"] is src, "the handle itself must be stored, not a copy"
        assert src.reads == [], "attaching must not read the source"

    def test_np_memmap_attaches(self, tmp_path):
        path = tmp_path / "f.dat"
        mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(B, N, D))
        mm[:] = _arr(B, N, D)
        mm.flush()
        dom, _ = _dom()
        dom.variable("_f", np.memmap(path, dtype=np.float32, mode="r", shape=(B, N, D)))
        assert dom.context["_f"].shape == (B, N, D)

    def test_eager_arrays_still_take_the_eager_path(self):
        """np.ndarray satisfies the duck-type too — it must NOT be treated as lazy."""
        dom, _ = _dom()
        dom.variable("_f", _arr(B, N, D))
        import jax.numpy as jnp

        assert isinstance(dom.context["_f"], jnp.ndarray)

    def test_shape_without_getitem_raises_naming_both_requirements(self):
        class Half:
            shape = (B, N, D)

        dom, _ = _dom()
        with pytest.raises(TypeError, match="__getitem__"):
            dom.variable("_f", Half())

    def test_nonsense_raises(self):
        dom, _ = _dom()
        with pytest.raises(TypeError, match="sampling spec"):
            dom.variable("_f", object())


class TestStreaming:
    def test_trains_without_ever_reading_the_whole_source(self):
        dom, x = _dom()
        src = Counting(_arr(B, N, D))
        dom.variable("_f", src)
        _train(dom, x, steps=3, offload_data=True)
        assert len(src.gathers) == 3, f"one gather per step, got {len(src.gathers)}"
        assert not any(isinstance(k, slice) and k == slice(None) for k in src.reads), "read whole"

    def test_indices_are_strictly_increasing(self):
        """h5py/zarr fancy indexing REQUIRES increasing indices — the on-device path already sorts."""
        dom, x = _dom()
        src = Counting(_arr(B, N, D))
        dom.variable("_f", src)
        _train(dom, x, steps=4, offload_data=True)
        for g in src.gathers:
            assert np.all(np.diff(g) > 0), f"unsorted gather {g}"

    def test_loss_matches_the_eager_equivalent(self):
        a = _arr(B, N, D)
        d1, x1 = _dom()
        d1.variable("_f", a)
        s1 = _train(d1, x1, steps=3, offload_data=True)
        d2, x2 = _dom()
        d2.variable("_f", Counting(a))
        s2 = _train(d2, x2, steps=3, offload_data=True)
        l1 = float(s1.training_logs[-1]["total_loss"][-1])
        l2 = float(s2.training_logs[-1]["total_loss"][-1])
        assert l1 == pytest.approx(l2, rel=1e-6)

    def test_broadcast_row_is_read_as_one_row_not_the_whole_array(self):
        dom, x = _dom()
        src = Counting(_arr(1, N, D))  # leading dim 1 -> broadcast across the batch
        dom.variable("_f", src)
        _train(dom, x, steps=2, offload_data=True)
        assert all(isinstance(k, slice) and k == slice(0, 1) for k in src.reads), src.reads

    def test_a_lazy_and_an_eager_tag_coexist(self):
        dom, x = _dom()
        dom.variable("_f", Counting(_arr(B, N, D)))
        dom.variable("_g", _arr(B, N, D))
        _train(dom, x, steps=2, offload_data=True)


class TestRefusals:
    def test_on_device_path_refuses_and_names_the_fix(self):
        dom, x = _dom()
        dom.variable("_f", Counting(_arr(B, N, D)))
        with pytest.raises(ValueError, match="offload_data=True"):
            _train(dom, x, steps=2)  # no offload_data -> would read the whole dataset

    def test_missing_time_axis_raises_instead_of_being_rewritten(self):
        """The eager path inserts the (B, T, ...) time axis; a lazy source cannot be rewritten
        without reading it, so the same layout rule is enforced as an error."""
        d = B * jno.Shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=(5, 4)).domain(compute_mesh_connectivity=True)
        d.variable("interior")
        with pytest.raises(ValueError, match="time axis"):
            d.variable("_f", Counting(_arr(B, 6, 5, 1)))

    def test_correct_layout_is_accepted(self):
        d = B * jno.Shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=(5, 4)).domain(compute_mesh_connectivity=True)
        d.variable("interior")
        d.variable("_f", Counting(_arr(B, 1, 6, 5, 1)))
        assert d.context["_f"].shape == (B, 1, 6, 5, 1)


class TestExtremes:
    def test_batchsize_equal_to_total(self):
        dom, x = _dom()
        src = Counting(_arr(B, N, D))
        dom.variable("_f", src)
        _train(dom, x, steps=2, batchsize=B, offload_data=True)

    def test_low_rank_tag_untouched_by_the_layout_check(self):
        dom, _ = _dom()
        src = Counting(_arr(B, 3))
        dom.variable("p", src)
        assert dom.context["p"] is src

    def test_shared_tag_leading_dim_is_left_alone(self):
        dom, _ = _dom()
        src = Counting(_arr(B + 5, 6, 5, 1))  # neither B nor 1 -> "shared", never vmapped
        dom.variable("table", src)
        assert dom.context["table"] is src
