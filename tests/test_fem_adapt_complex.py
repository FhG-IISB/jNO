"""Anisotropic (Hessian-metric) adaptation on a COMPLEX field.

The metric is a scalar estimator, so a complex solution has to be reduced to one real field before
the Hessian is recovered. That reduction was missing and the driver refused complex outright --
which rules out anisotropic adaptation for every eddy-current, Helmholtz and RCWA problem, i.e.
exactly the ones whose solutions have the directional features anisotropy is for.

The reduction is ``|u|``, matching what the transient driver already does (it forms the modulus
from the stacked real/imaginary blocks) and what the isotropic ZZ indicator already does.

It must happen at the DRIVER, not inside ``hessian_metric``: ``recover_hessian`` deliberately
preserves the complex dtype, and the resulting ``H`` is complex *symmetric* but NOT Hermitian, while
``hessian_metric`` diagonalises with ``np.linalg.eigh`` -- which reads one triangle and conjugates
it, silently answering about a different matrix (measured: ``[-1, 6]`` against a true
``[-0.85+2.48j, 5.85+0.02j]``). So ``hessian_metric`` refuses complex input rather than accepting it.
"""

import numpy as np
import pytest

import jno  # noqa: E402
import jno.jnp_ops as J  # noqa: E402


def _imag_layer_fem(d, eps=0.05):
    """-lap u = 1j*f with u = 0 on the boundary, f the sharp dipole along x + y = 1 that produces an
    oblique internal layer. The whole solution is therefore PURELY IMAGINARY: Re(u) is identically
    zero, so a metric driven by anything that drops the imaginary part sees a flat field and refines
    nowhere in particular.

    (The complex part rides the source rather than the boundary value: a complex form's Re/Im legs
    share one Dirichlet row set, so a prescribed Im u is not expressible.)"""
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    t = J.tanh((xi + yi - 1.0) / eps)
    f = 1j * (4.0 / eps**2) * (1.0 - t * t) * t
    return jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0])


def test_anisotropic_adaptation_runs_on_a_complex_field():
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.12).domain()
    n0 = len(d.mesh.points)
    _imag_layer_fem(d).solve(adapt=jno.solve.remesh(anisotropic=True, max_iters=4, refine_factor=1.6, max_dofs=2500))
    assert len(d.mesh.points) > n0, "the mesh did not refine at all"


def test_the_metric_follows_the_imaginary_feature():
    """The layer is at x + y = 1 and exists only in Im(u). If the reduction dropped the imaginary
    part the field would be flat and refinement would spread out, so this is what makes the test
    about the reduction rather than about adaptation running."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.12).domain()
    _imag_layer_fem(d).solve(adapt=jno.solve.remesh(anisotropic=True, max_iters=4, refine_factor=1.6, max_dofs=2500))

    p = np.asarray(d.mesh.points)
    dist = np.abs(p[:, 0] + p[:, 1] - 1.0) / np.sqrt(2.0)
    near = float((dist < 0.08).mean())
    # A uniform mesh would put ~16% of its vertices in a band of half-width 0.08 about the diagonal.
    assert near > 0.30, f"only {near:.1%} of vertices landed near the layer; the metric did not find it"


def test_hessian_metric_still_refuses_complex_input():
    """Called directly it must raise, not diagonalise a complex-symmetric Hessian with `eigh`."""
    from jno.utils.solver.fem_adapt import hessian_metric

    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain()
    n = len(d.mesh.points)
    u = np.ones(n) * (1.0 + 1.0j)
    with pytest.raises((NotImplementedError, ValueError), match="complex"):
        hessian_metric(d, u, target_complexity=100.0, hmin=0.01, hmax=1.0)


def test_the_real_path_is_unchanged():
    """A real field must take exactly the path it did before."""
    from jno.utils.solver.fem_adapt import hessian_metric

    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    p = np.asarray(d.mesh.points)
    u = np.tanh((p[:, 0] + p[:, 1] - 1.0) / 0.1)
    M = hessian_metric(d, u, target_complexity=200.0, hmin=0.01, hmax=1.0)
    assert M.shape == (len(p), 3) and np.isfinite(M).all()
