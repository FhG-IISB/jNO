"""`field.regularize(...)` — the unified field regularizer.

`.regularize(kind, ...)` is now the single surface (the `jno.fn.regularize` namespace was
removed). For a FEM nodal-parameter field it assembles the FEM-exact penalty (covered in
test_fem_inverse); for a coordinate network it uses autodiff. This file covers the
coordinate-field branch.
"""

from __future__ import annotations

import jax
import numpy as np
import optax
import pytest

import jno
import jno.jnp_ops as jnn

pytest.importorskip("foundax")
import foundax  # noqa: E402


def _coord_field():
    d = jno.domain(constructor=jno.domain.rect(mesh_size=0.5))
    x, y, _ = d.variable("interior")
    k_net = jnn.nn.wrap(foundax.mlp(in_features=2, output_dim=1, hidden_dims=8, num_layers=2, key=jax.random.PRNGKey(0)))
    k_net.optimizer(optax.adam(1e-3))
    return x, y, k_net(x, y)


def test_smooth_matches_manual_h1seminorm():
    """regularize('smooth', x, y) == sum of squared partials (the autodiff H1 seminorm)."""
    x, y, k = _coord_field()
    reg = k.regularize("smooth", x, y)
    manual = k.d(x) ** 2 + k.d(y) ** 2
    r, m = jno.core([reg.mean]).eval([reg, manual])
    assert np.allclose(np.asarray(r), np.asarray(m), atol=1e-6)


def test_nonneg_and_bounded_build_terms():
    x, y, k = _coord_field()
    assert k.regularize("nonneg") is not None
    assert k.regularize("nonneg", strength=2.0) is not None
    assert k.regularize("bounded", lo=0.1, hi=2.0) is not None
    assert k.regularize("tv", x, y) is not None


def test_smooth_without_variables_raises():
    x, y, k = _coord_field()
    with pytest.raises(ValueError, match="spatial variables"):
        k.regularize("smooth")


def test_l2_on_coordinate_field_raises():
    """'l2'/'tikhonov' is FEM-only (needs the mass matrix); a coordinate field rejects it."""
    x, y, k = _coord_field()
    with pytest.raises(ValueError, match="FEM-only"):
        k.regularize("l2")
