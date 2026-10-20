"""A mesh-free domain's reference clouds are drawn when first read, not by the constructor.

Drawing every tag's cloud in the constructor made it the most expensive step of a PEEC build: on a
welded network of 57 solids and wires, 67 of 70 s, rejection-sampling each small named face against
the whole union -- and the solver never reads those clouds. The tags must still all be listed at once,
and a cloud read later must be the same whenever and in whatever order it is read.
"""

import pickle

import numpy as np
import pytest

import jno


@pytest.fixture
def counted(monkeypatch):
    calls = []
    cls = type(jno.shape.box(0, 0, 0, 1, 1, 1).domain())
    orig = cls._draw_geometry_points

    def spy(self, kind, name, *a, **k):
        calls.append((kind, name))
        return orig(self, kind, name, *a, **k)

    monkeypatch.setattr(cls, "_draw_geometry_points", spy)
    return calls


def two_boxes():
    a = jno.shape.box(0, 0, 0, 1.0, 1.0, 0.2).name("a")
    b = jno.shape.box(0, 0, 0.5, 1.0, 1.0, 0.7).name("b")
    return (a + b).domain()


def test_the_constructor_draws_no_reference_cloud(counted):
    two_boxes()
    assert counted == []


def test_every_tag_is_listed_without_drawing(counted):
    d = two_boxes()
    assert {"interior", "boundary"} <= set(d._mesh_pool)
    assert len(d.boundary_tags()) > 1  # the named faces, not only "boundary"
    assert all(t in d._mesh_pool for t in d.boundary_tags())
    assert counted == []


def test_reading_one_tag_draws_only_that_tag(counted):
    d = two_boxes()
    pts = d._mesh_pool["interior"]
    assert counted == [("interior", None)]
    assert pts.shape == (d._REFERENCE_INTERIOR, 3)
    inside = ((pts[:, 2] >= -1e-12) & (pts[:, 2] <= 0.2 + 1e-12)) | (
        (pts[:, 2] >= 0.5 - 1e-12) & (pts[:, 2] <= 0.7 + 1e-12)
    )
    assert inside.all()


def test_a_cloud_is_the_same_whatever_was_read_before():
    d1, d2 = two_boxes(), two_boxes()
    tags = d1.boundary_tags()
    first = d1._mesh_pool[tags[-1]]
    for t in tags[:-1]:
        d2._mesh_pool.get(t)
    assert np.array_equal(first, d2._mesh_pool[tags[-1]])
    assert np.array_equal(first, d1._mesh_pool[tags[-1]])  # and on a second read


def test_a_value_set_before_the_draw_is_kept():
    d = two_boxes()
    mine = np.zeros((3, 3))
    d._mesh_pool["interior"] = mine
    assert d._mesh_pool["interior"] is mine


def test_it_pickles_as_the_full_dict():
    d = two_boxes()
    back = pickle.loads(pickle.dumps(d._mesh_pool))
    assert type(back) is dict
    assert set(back) == set(d._mesh_pool)
    assert all(isinstance(v, np.ndarray) and len(v) for v in back.values())
