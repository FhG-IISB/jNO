"""A grid-valued tensor tag gets the time axis the (B, T, ...) layout requires.

Context tensors are ``(B, T, ...)``. The compiler peels ``B`` with a vmap, then infers the time
extent as ``max(v.shape[0])`` over the remaining values with ``ndim >= 3`` — so a tensor attached as
``(B, H, W, C)``, the shape a user actually has, had ``H`` read as the timestep count. One "step"
reached the expression and the rest was silently dropped.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import jno

B, H, W, C = 4, 8, 5, 1  # H != W so a mix-up is visible


def _steady(nx=H, ny=W, batch=B):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=(nx - 1, ny - 1)).domain(compute_mesh_connectivity=True)
    dom = batch * d
    dom.variable("interior")
    return dom


def _arr(*shape):
    return np.arange(int(np.prod(shape)), dtype=np.float32).reshape(*shape)


class TestSteadyDomain:
    def test_natural_shape_reaches_the_evaluator_whole(self):
        """The regression: (B, H, W, C) used to arrive as (W, C) — 7/8 of the field gone."""
        dom = _steady()
        dom.variable("_f", _arr(B, H, W, C))
        out = jno.core([], domain=dom).eval([dom.variable("_f")], domain=dom)
        assert np.asarray(out[0]).shape[-3:] == (H, W, C)

    def test_it_matches_the_hand_written_form(self):
        a = _arr(B, H, W, C)
        d1, d2 = _steady(), _steady()
        d1.variable("_f", a)
        d2.variable("_f", a[:, None, ...])  # what users had to write
        np.testing.assert_array_equal(np.asarray(d1.context["_f"]), np.asarray(d2.context["_f"]))

    def test_already_normalized_is_left_alone_and_is_idempotent(self):
        dom = _steady()
        dom.variable("_f", _arr(B, 1, H, W, C))
        assert dom.context["_f"].shape == (B, 1, H, W, C)

    def test_the_six_axis_tutorial_form_still_works(self):
        dom = _steady()
        dom.variable("_f", _arr(B, 1, 1, H, W, C))
        assert dom.context["_f"].shape == (B, 1, 1, H, W, C)

    def test_broadcast_leading_dim_is_normalized_too(self):
        dom = _steady()
        dom.variable("_f", _arr(1, H, W, C))
        assert dom.context["_f"].shape == (1, 1, H, W, C)

    def test_square_grid(self):
        dom = _steady(nx=6, ny=6)
        dom.variable("_f", _arr(B, 6, 6, C))
        assert dom.context["_f"].shape == (B, 1, 6, 6, C)


class TestLeftAlone:
    def test_low_rank_parameter_untouched(self):
        """(B, 1, 1) is the DeepONet branch input — rank 3, never reaches the time inference."""
        dom = _steady()
        dom.variable("k", _arr(B, 1, 1))
        assert dom.context["k"].shape == (B, 1, 1)

    def test_two_axis_parameter_untouched(self):
        dom = _steady()
        dom.variable("p", _arr(B, 7))
        assert dom.context["p"].shape == (B, 7)

    def test_shared_tag_untouched(self):
        """shape[0] is neither B nor 1 — the compiler never vmaps it, so there is no T to insert."""
        dom = _steady()
        dom.variable("table", _arr(B + 3, H, W, C))
        assert dom.context["table"].shape == (B + 3, H, W, C)


class TestTimeDependent:
    def _dom(self, n_t=3):
        d = (
            jno.shape.rect(0.0, 0.0, 1.0, 1.0)
            .structured(n=(H - 1, W - 1))
            .domain(
                compute_mesh_connectivity=True,
                time=(0.0, 1.0, n_t),
            )
        )
        dom = B * d
        dom.variable("interior")
        return dom

    def test_correct_time_axis_is_left_alone(self):
        dom = self._dom(3)
        dom.variable("_f", _arr(B, 3, H, W, C))
        assert dom.context["_f"].shape == (B, 3, H, W, C)

    def test_broadcast_time_axis_is_left_alone(self):
        dom = self._dom(3)
        dom.variable("_f", _arr(B, 1, H, W, C))
        assert dom.context["_f"].shape == (B, 1, H, W, C)

    def test_missing_time_axis_raises_rather_than_guessing(self):
        """On a time-dependent domain axis 1 is genuinely ambiguous — refuse, don't insert."""
        dom = self._dom(3)
        with pytest.raises(ValueError, match="timesteps"):
            dom.variable("_f", _arr(B, H, W, C))


class TestPerNode:
    """Rank 3, ``(B, n, k)``: one value per node per sample, the operator-learning target. The
    single-window path took axis 1 as time (``arr[0]``), so it arrived as ``(B, k)`` -- node 0 only."""

    def _dom(self, batch=B, n=6):
        dom = batch * jno.domain.from_array({"nodes": np.random.default_rng(0).random((n, 2))})
        dom.variable("nodes")
        return dom

    def test_per_node_data_gets_the_time_axis(self):
        dom = self._dom()
        dom.variable("u", _arr(B, 6, 3))
        assert dom.context["u"].shape == (B, 1, 6, 3)

    def test_attached_before_the_nodes_are_sampled(self):
        dom = B * jno.domain.from_array({"nodes": np.zeros((6, 2))})
        dom.variable("u", _arr(B, 6, 1))
        assert dom.context["u"].shape == (B, 1, 6, 1)

    def test_no_matching_point_set_raises_naming_the_layout(self):
        dom = self._dom()
        with pytest.raises(ValueError, match=r"node count.*\(4, 1, 7, 1\)"):
            dom.variable("u", _arr(B, 7, 1))

    def test_n_equal_to_B_raises(self):
        dom = self._dom(batch=6)
        with pytest.raises(ValueError, match="batch count"):
            dom.variable("u", _arr(6, 6, 1))

    def test_per_step_values_get_a_node_axis(self):
        """(B, T, k), one value per sample and step, is stored as (B, T, 1, k) -- the layout the compiler reads."""
        dom = TestTimeDependent()._dom(3)
        dom.variable("u", _arr(B, 3, 2))
        assert dom.context["u"].shape == (B, 3, 1, 2)

    @pytest.mark.parametrize("window", [1, None])
    def test_every_step_receives_its_own_per_step_value(self, window):
        """data[b, s] = 100 b + s attached as (B, T, 1): sample b at step s must see 100 b + s. Left as (B, T, k),
        the default one-step window handed every step step 0's value, and the full window handed every step the
        whole series."""
        import jax

        nb, nt = 2, 4
        dom = nb * jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=(2, 2)).domain(time=(0.0, 3.0, nt))
        x, _y, t = dom.variable("interior")
        data = np.array([[[100.0 * b + s] for s in range(nt)] for b in range(nb)])
        g = dom.variable("g", data)
        crux = jno.core([], domain=dom)
        if window is None:
            got = np.asarray(crux.eval([g], domain=dom, min_consecutive=None))
            np.testing.assert_array_equal(got.reshape(nb, nt), data[..., 0])
            return
        for seed in (0, 3, 7):
            tt, gv = crux.eval([t + 0.0 * x, g], domain=dom, min_consecutive=1, key=jax.random.PRNGKey(seed))
            tt, gv = np.asarray(tt), np.asarray(gv)
            for b in range(nb):
                s = int(round(tt[b, 0, 0]))
                assert gv[b].ravel()[0] == data[b, s, 0], (seed, b, s, gv[b].ravel()[0])

    def test_a_lazy_per_step_source_is_refused_naming_the_layout(self):
        class Lazy:
            def __init__(self, a):
                self._a, self.shape, self.dtype = a, a.shape, a.dtype

            def __getitem__(self, key):
                return self._a[key]

        dom = TestTimeDependent()._dom(3)
        with pytest.raises(ValueError, match=r"\(4, 3, 1, 2\)"):
            dom.variable("u", Lazy(_arr(B, 3, 2)))

    def test_per_node_on_a_time_grid_is_shared_across_steps(self):
        dom = TestTimeDependent()._dom(3)
        dom.variable("u", _arr(B, H * W, 2))
        assert dom.context["u"].shape == (B, 1, H * W, 2)

    def test_timestep_count_equal_to_node_count_raises(self):
        dom = TestTimeDependent()._dom(H * W)
        with pytest.raises(ValueError, match="timestep count"):
            dom.variable("u", _arr(B, H * W, 1))


class TestExtremes:
    def test_zero_sized_grid_axis(self):
        dom = _steady()
        dom.variable("_f", jnp.zeros((B, 0, W, C)))
        assert dom.context["_f"].shape == (B, 1, 0, W, C)

    def test_rank_four_with_singleton_axis_one_is_ambiguous_but_safe(self):
        """(B, 1, W, C) already looks normalized; leaving it alone is the only safe read."""
        dom = _steady()
        dom.variable("_f", _arr(B, 1, W, C))
        assert dom.context["_f"].shape == (B, 1, W, C)

    def test_high_rank_field(self):
        dom = _steady()
        dom.variable("_f", _arr(B, H, W, 2, 3))
        assert dom.context["_f"].shape == (B, 1, H, W, 2, 3)
