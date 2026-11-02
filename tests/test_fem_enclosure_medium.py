"""Radiation enclosures found from a transparent medium: d.tag by a property, d.enclosure(medium=...), the
per-face emissivity d.eps(gap.field(T)), and the irradiation operator.

Oracle: two long concentric grey cylinders (radii r1 < r2) exchange, per unit length,
    Q = 2 pi r1 sigma (T1^4 - T2^4) / (1/eps1 + (r1/r2)(1/eps2 - 1))
(Modest, Radiative Heat Transfer, 3rd ed., Ch. 5, two-surface enclosure)."""

import numpy as np
import pytest
from jax import config

import jno
from jno.domain import enclosure as E

config.update("jax_enable_x64", True)
SIGMA = E.STEFAN_BOLTZMANN
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


def test_found_enclosure_equals_the_named_one_and_matches_two_cylinders(tmp_path, monkeypatch):
    monkeypatch.setenv("JNO_CACHE", str(tmp_path))
    eps1, eps2, T1, T2 = 0.6, 0.8, 1500.0, 900.0
    d = _cylinders(eps1, eps2)
    d.tag("transparent", lambda kappa: kappa == 0)
    rad = d.enclosure(medium="transparent", enforce_closure=True)
    named = d.enclosure(["core", "shell"], medium_tags=["gas"], enforce_closure=True)
    assert len(rad.parts) == 1 and rad.media == [["gas"]]
    np.testing.assert_allclose(np.asarray(rad.parts[0].view_factor), np.asarray(named.view_factor), atol=1e-12)

    T, v = d.fem_symbols(names=("T", "v"))
    Te = rad.field(T)
    e = d.eps(Te)
    load = rad.load(e * (SIGMA * Te**4 - rad.irradiation(Te, eps=e)))
    pts = np.asarray(d.mesh.points)
    rho = np.hypot(pts[:, 0], pts[:, 1])
    u = np.where(rho < 0.5 * (R1 + R2), T1, T2)
    q = np.asarray(load.residual_fn(u))
    on_core = np.isclose(rho, R1, atol=1e-9)
    exact = 2 * np.pi * R1 * SIGMA * (T1**4 - T2**4) / (1 / eps1 + (R1 / R2) * (1 / eps2 - 1))
    assert q[on_core].sum() == pytest.approx(exact, rel=1e-2)
    assert q.sum() == pytest.approx(0.0, abs=1e-6 * exact)  # what one surface loses, the other gains


def test_view_factors_come_from_the_cache_the_second_time(tmp_path, monkeypatch):
    monkeypatch.setenv("JNO_CACHE", str(tmp_path))
    calls = []
    real = E.build_enclosure
    monkeypatch.setattr(E, "build_enclosure", lambda *a, **k: calls.append(1) or real(*a, **k))
    d = _cylinders()
    F1 = np.asarray(d.enclosure(medium="gas").parts[0].view_factor)
    F2 = np.asarray(d.enclosure(medium="gas").parts[0].view_factor)
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
    d.tag("gaps", lambda kappa: kappa == 0)
    rad = d.enclosure(medium="gaps")
    assert sorted(map(tuple, rad.media)) == [("pocket_a",), ("pocket_b",)]
    assert all(p.tags == ["block"] for p in rad.parts)


def test_a_medium_that_reaches_the_domain_boundary_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("JNO_CACHE", str(tmp_path))
    rect = jno.shape.rect
    d = (rect(0, 0, 1, 1).name("solid").attach(eps=0.5) + rect(0, 0, 2, 1).name("gas")).sized(0.1).domain()
    with pytest.raises(ValueError, match="reaches the domain boundary"):
        d.enclosure(medium="gas")


def test_an_emissivity_law_is_evaluated_per_face(tmp_path, monkeypatch):
    monkeypatch.setenv("JNO_CACHE", str(tmp_path))
    d = _cylinders()
    d.attach("core", eps=lambda T: 0.3 + 2e-4 * T)
    rad = d.enclosure(medium="gas")
    T, v = d.fem_symbols(names=("T", "v"))
    Te = rad.field(T)
    pts = np.asarray(d.mesh.points)
    u = 600.0 + 400.0 * np.hypot(pts[:, 0], pts[:, 1])
    got = np.asarray(d.eps(Te)(u))
    t = np.asarray(rad.field(u))
    tags = rad.element_tags.astype(str)
    want = np.where(tags == "core", 0.3 + 2e-4 * t, 0.8)
    np.testing.assert_allclose(got, want, rtol=1e-12)
