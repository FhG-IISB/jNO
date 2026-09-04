"""Assembly belongs on the HOST, even when the default backend is a GPU.

jNO's element loop allocates temporaries far larger than the matrix it produces, so assembling on
the device runs out of memory at roughly a third of the size the finished operator occupies happily:
a 527k-DOF mixed N1E x Lagrange operator is 0.77 GB and fits comfortably, while assembling it on an
8 GB card dies. Host RAM is the plentiful resource (~6 GB per million DOFs) and the device's job is
the solve.

Measured, on the two forms that matter, with the persistent compile cache off:

    Poisson  43k DOFs   assemble  4.02 s on CPU vs 4.98 s on GPU   (CPU faster at every size)
    A-V      43k DOFs   assemble  9.28 s on CPU vs 11.00 s on GPU  (first call)

The one case the device wins is REPEATED assembly of a heavy form once compilation has amortised
(A-V at 43k DOFs: 5.60 s on CPU vs 3.69 s on GPU), which is why the override below exists and is
tested. An out-of-memory failure is fatal; a 1.5x slowdown on a Newton loop is not.

There is deliberately NO new argument for this. ``jax.default_device`` is the standard JAX mechanism
for saying where work goes, so an explicit one wins and the default applies only when the user has
expressed no preference.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")
import jax  # noqa: E402

import jno  # noqa: E402

_HAS_GPU = any(d.platform != "cpu" for d in jax.devices())
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="needs a non-CPU backend to place anything")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson(size=0.14):
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(names=("u", "v"))
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    b = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    return jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - 1.0 * vi, u.bind(x=b[0], y=b[1], z=b[2]) - 0.0])


def _operator_device(fem):
    from jno.precond import _fem_concrete_operator

    A = _fem_concrete_operator(fem)
    bc = A.bcoo if hasattr(A, "bcoo") and A.bcoo is not None else A
    return list(bc.data.devices())[0]


@_gpu_only
def test_assembly_lands_on_the_host_by_default():
    """The whole point: with a GPU present and no preference expressed, assembly is still host-side."""
    assert _operator_device(_poisson()).platform == "cpu"


@_gpu_only
def test_assembly_allocates_no_device_memory_that_scales():
    """Placement is the mechanism; not touching the card is the ACTUAL requirement.

    The bar is not literally zero bytes. A handful of scalars still land on the device -- measured at
    a flat **8 bytes** from n=242 through n=3,797, i.e. 16x the DOFs and 50x the nonzeros for exactly
    the same residue. Constant means the element loop is on the host; the failure this guards against
    is per-element work on the device, which scales and cost 104 MB for a 9,970-DOF Poisson before
    the change. So assert O(1), not 0, and let a 16x problem be the discriminator.

    ``peak_bytes_in_use`` is monotonic for the life of the process, hence measuring GROWTH, and a
    warm-up first to absorb one-time device setup.
    """
    dev = jax.devices()[0]
    peak = lambda: dev.memory_stats()["peak_bytes_in_use"]  # noqa: E731

    _poisson(size=0.22)  # warm-up: one-time setup must not be charged to the first measurement
    b0 = peak()
    _poisson(size=0.22)
    small = peak() - b0
    b1 = peak()
    _poisson(size=0.065)  # 16x the DOFs, 50x the nonzeros
    large = peak() - b1

    assert large <= 4096, f"assembly put {large} bytes on the device; the element loop is not host-side"
    assert large <= small + 64, (
        f"device allocation scaled with the problem ({small} -> {large} bytes): per-element work is "
        "still running on the card"
    )


@_gpu_only
def test_an_explicit_default_device_still_wins():
    """Repeated assembly of a heavy form IS faster on the device, so the escape hatch has to work --
    and it is the standard JAX one, not a jNO argument."""
    gpu = next(d for d in jax.devices() if d.platform != "cpu")
    with jax.default_device(gpu):
        assert _operator_device(_poisson()).platform != "cpu"


def test_the_solution_is_unchanged():
    """A placement change must be numerically inert. int_0^1 of the unit-cube Poisson solve, against
    the same problem assembled under an explicit host default."""
    got = np.asarray(jno.np.asarray(_poisson().solve()))
    with jax.default_device(jax.devices("cpu")[0]):
        ref = np.asarray(jno.np.asarray(_poisson().solve()))
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-14)
    assert np.isfinite(got).all() and np.abs(got).max() > 0.0
