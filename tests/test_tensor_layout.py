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
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=(nx - 1, ny - 1)).domain(compute_mesh_connectivity=True)
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
            jno.Shape.rect(0.0, 0.0, 1.0, 1.0)
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
