"""The radiating faces of d.irradiation, found from the transparent medium (kappa == 0): one enclosure per
connected medium, the same view factors as the enclosure of those surfaces built by name, cached on disk, and a
medium open to the domain boundary refused. Also d.tag by a property predicate."""

import numpy as np
import pytest
from jax import config

import jno
from jno.domain import enclosure as E

config.update("jax_enable_x64", True)
R1, R2, R3 = 0.3, 0.6, 0.8


def _cylinders(eps1=0.6, eps2=0.8, h=0.04):
    disk = jno.shape.disk
    d = (
        (
            disk(0, 0, R1).name("core").attach(eps=eps1, kappa=1e9)
            + (disk(0, 0, R3) - disk(0, 0, R2)).name("shell").attach(eps=eps2, kappa=1e9)
            + disk(0, 0, R2).name("gas").attach(kappa=0.0)
        )
        .sized(h)
        .domain()
    )
    return d


def test_property_tag_selects_the_transparent_regions():
    d = _cylinders()
    d.tag("transparent", lambda kappa: kappa == 0)
    assert d.__dict__["_property_tag_regions"]["transparent"] == ["gas"]


def test_a_bool_where_raises_with_the_fix():
    d = _cylinders()
    with pytest.raises(TypeError, match=r"lambda kappa: kappa == 0"):
        d.tag("transparent", d.kappa == 0)  # identity comparison: a plain False


def _faces(d):
    from jno.domain.radiation import radiating_faces

    d.__dict__.pop("_radiation_faces", None)  # built once per domain; rebuild (from the disk cache) when asked
    return radiating_faces(d)


def test_the_found_faces_are_the_named_enclosure(tmp_path, monkeypatch):
    """The faces d.irradiation integrates over: every face between the kappa == 0 medium and its neighbours,
    with the same view factors as the enclosure of those surfaces built by name."""
    monkeypatch.setenv("JNO_CACHE", str(tmp_path))
    d = _cylinders()
    rad = _faces(d)
    named = E.build_enclosure(d, ["core", "shell"], medium_tags=["gas"], enforce_closure=True)
    assert len(rad.parts) == 1 and rad.media == [["gas"]]
    np.testing.assert_allclose(np.asarray(rad.parts[0].view_factor), np.asarray(named.view_factor), atol=1e-12)


def test_view_factors_come_from_the_cache_the_second_time(tmp_path, monkeypatch):
    monkeypatch.setenv("JNO_CACHE", str(tmp_path))
    calls = []
    real = E.build_enclosure
    monkeypatch.setattr(E, "build_enclosure", lambda *a, **k: calls.append(1) or real(*a, **k))
    d = _cylinders()
    F1 = np.asarray(_faces(d).parts[0].view_factor)
    F2 = np.asarray(_faces(d).parts[0].view_factor)
    assert len(calls) == 1
    np.testing.assert_array_equal(F1, F2)


def test_each_connected_medium_is_its_own_enclosure(tmp_path, monkeypatch):
    monkeypatch.setenv("JNO_CACHE", str(tmp_path))
    disk, rect = jno.shape.disk, jno.shape.rect
    d = (
        (
            disk(-0.5, 0, 0.3).name("pocket_a").attach(kappa=0.0)
            + disk(0.5, 0, 0.3).name("pocket_b").attach(kappa=0.0)
            + rect(-1, -0.5, 1, 0.5).name("block").attach(kappa=1e9, eps=0.7)
        )
        .sized(0.05)
        .domain()
    )
    rad = _faces(d)
    assert sorted(map(tuple, rad.media)) == [("pocket_a",), ("pocket_b",)]
    assert all(p.tags == ["block"] for p in rad.parts)


def test_a_medium_that_reaches_the_domain_boundary_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("JNO_CACHE", str(tmp_path))
    rect = jno.shape.rect
    d = (
        (rect(0, 0, 1, 1).name("solid").attach(eps=0.5, kappa=1e9) + rect(0, 0, 2, 1).name("gas").attach(kappa=0.0))
        .sized(0.1)
        .domain()
    )
    with pytest.raises(ValueError, match="reaches the domain boundary"):
        _faces(d)
