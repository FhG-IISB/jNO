"""Tests for loading per-tag coordinate arrays from .npz into jno.domain."""

from __future__ import annotations

import numpy as np
import pytest

import jno


@pytest.fixture
def npz_tag_file(tmp_path):
    path = tmp_path / "tagged_points.npz"
    np.savez(
        path,
        Air=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64),
        Gas=np.array([[0.25, 0.25], [0.75, 0.75]], dtype=np.float64),
        **{"Quartz.1": np.array([[0.1, 0.9]], dtype=np.float64)},
    )
    return str(path)


def test_domain_loads_npz_tags(npz_tag_file):
    dom = jno.domain(constructor=npz_tag_file, compute_mesh_connectivity=False)

    assert dom.mesh is None
    assert dom.dimension == 2
    assert "Air" in dom.avaiable_mesh_tags
    assert "Gas" in dom.avaiable_mesh_tags
    assert "Quartz.1" in dom.avaiable_mesh_tags

    assert dom._mesh_pool["Air"].shape == (3, 2)
    assert dom._mesh_pool["Gas"].shape == (2, 2)
    assert dom.context["Air"].shape == (1, 1, 3, 2)


def test_variable_works_without_sampling_for_npz_tag(npz_tag_file):
    dom = jno.domain(constructor=npz_tag_file, compute_mesh_connectivity=False)

    x, y, t = dom.variable("Air")

    assert x.tag == "Air"
    assert y.tag == "Air"
    assert t.tag in {"__time_Air__", "__time__"}


def test_missing_npz_file_raises(tmp_path):
    missing = tmp_path / "missing_points.npz"
    with pytest.raises(FileNotFoundError):
        jno.domain(constructor=str(missing), compute_mesh_connectivity=False)
