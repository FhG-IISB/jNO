"""Tests for domain export helpers across 1D/2D/3D domains."""

from pathlib import Path

import pytest

import jno


def _assert_nonempty(path: Path):
    assert path.exists(), f"Expected export file to exist: {path}"
    assert path.stat().st_size > 0, f"Expected non-empty export file: {path}"


@pytest.mark.parametrize(
    "constructor",
    [
        jno.domain.line(mesh_size=0.2),
        jno.domain.equi_distant_rect(nx=6, ny=6),
        jno.domain.cube(mesh_size=0.6),
    ],
)
def test_export_png_vtk_msh(constructor, tmp_path):
    dom = jno.domain(constructor=constructor, compute_mesh_connectivity=True)

    png_path = tmp_path / "mesh.png"
    vtk_path = tmp_path / "mesh.vtk"
    msh_path = tmp_path / "mesh.msh"

    dom.export(str(png_path), show_sampled=False)
    dom.export(str(vtk_path))
    dom.export(str(msh_path))

    _assert_nonempty(png_path)
    _assert_nonempty(vtk_path)
    _assert_nonempty(msh_path)


def test_export_dispatch_with_explicit_format(tmp_path):
    dom = jno.domain(constructor=jno.domain.equi_distant_rect(nx=4, ny=4), compute_mesh_connectivity=True)

    out = tmp_path / "mesh_any.ext"
    dom.export(str(out), fmt="vtk")
    _assert_nonempty(out)


@pytest.mark.parametrize(
    "constructor",
    [
        jno.domain.equi_distant_rect(nx=6, ny=6),
        jno.domain.cube(mesh_size=0.8),
    ],
)
def test_export_html_if_plotly_available(constructor, tmp_path):
    pytest.importorskip("plotly")

    dom = jno.domain(constructor=constructor, compute_mesh_connectivity=True)
    html_path = tmp_path / "mesh.html"
    dom.export(str(html_path), show_sampled=False)

    _assert_nonempty(html_path)


def test_export_unsupported_format_raises(tmp_path):
    dom = jno.domain(constructor=jno.domain.line(mesh_size=0.3), compute_mesh_connectivity=True)
    with pytest.raises(ValueError, match="Unsupported export format"):
        dom.export(str(tmp_path / "mesh.xyz"))
