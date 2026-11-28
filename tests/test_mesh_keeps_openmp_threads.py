"""Meshing leaves the calling thread's OpenMP thread count as it found it.

gmsh applies its ``General.NumThreads`` option (jNO sets 1 for reproducible meshes) through
``omp_set_num_threads``, which changes the CALLING thread's setting and is never undone. Every un-compiled
OpenMP routine called from that thread afterwards -- a PARDISO symbolic analysis, a BLAS call -- then ran on
one thread for the rest of the process.
"""

from __future__ import annotations

import ctypes
import sys

import pytest

import jno

pytest.importorskip("gmsh", reason="gmsh meshes the shapes")

_WANT = 3  # a count that is neither the default nor jNO's meshing thread count (1)


def _openmp_runtimes():
    """The OpenMP runtimes loaded into this process (Linux: read from the memory map)."""
    if not sys.platform.startswith("linux"):
        return []
    paths = set()
    with open("/proc/self/maps") as f:
        for line in f:
            if any(k in line for k in ("libomp", "libgomp", "libiomp")):
                paths.add(line.split()[-1])
    return [ctypes.CDLL(p) for p in sorted(paths)]


@pytest.fixture
def omp_runtimes():
    libs = _openmp_runtimes()
    if not libs:
        pytest.skip("no OpenMP runtime is loaded in this process")
    before = [lib.omp_get_max_threads() for lib in libs]
    for lib in libs:
        lib.omp_set_num_threads(_WANT)
    try:
        yield libs
    finally:
        for lib, n in zip(libs, before):
            lib.omp_set_num_threads(n)


def _counts(libs):
    return [lib.omp_get_max_threads() for lib in libs]


def test_building_a_3d_domain_keeps_the_callers_openmp_thread_count(omp_runtimes):
    d = jno.shape.box(0, 0, 0, 2, 1, 1, size=0.5).domain()
    d.fem_symbols(space="P0", names=("r", "s"))  # the first call that needs the mesh
    assert _counts(omp_runtimes) == [_WANT] * len(omp_runtimes)


def test_building_a_2d_domain_keeps_the_callers_openmp_thread_count(omp_runtimes):
    jno.shape.rect(0, 0, 1, 1, size=0.2).domain().fem_symbols()
    assert _counts(omp_runtimes) == [_WANT] * len(omp_runtimes)


def test_the_polygon_mesher_keeps_the_callers_openmp_thread_count(omp_runtimes):
    pytest.importorskip("shapely")
    from shapely.geometry import box

    jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    assert _counts(omp_runtimes) == [_WANT] * len(omp_runtimes)
