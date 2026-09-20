"""``remesh(anisotropic=True)`` on a MOVING mesh: stretch the elements ALONG the feature.

A directional feature -- a thermal boundary layer, a recoil crater, a front -- is resolved far more
cheaply by elements stretched along it than by shrinking elements in every direction. That is what a
Hessian metric buys: its eigenvectors set the DIRECTION and its eigenvalues the size along each,
where an isotropic size field can only say "smaller here".

This was refused on a geometry-term march. The refusal was plumbing: it is the same ``mmg`` rebuild
the criterion path already takes, with :func:`hessian_metric` in place of the scalar size field, so
it costs what an isotropic remesh costs.

The oracle is ALIGNMENT, not the usual shape quality. An isotropic measure such as
``4 sqrt(3) A / sum l^2`` calls every stretched cell bad by construction, and so cannot tell a
correctly-aligned boundary-layer element from a degenerate sliver -- it would mark the right answer
wrong. What distinguishes them is whether the long axis lies ALONG the level sets: for a cell with
principal stretch direction ``e`` and solution gradient ``g``, ``|e . g/|g||`` near 0 means the
element is stretched along the feature (right) and near 1 means across it (wrong).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

pytest.importorskip("mmgpy", reason="mmgpy required for metric-based remeshing")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _front(adapt, *, n=11, T=0.5):
    """A sharp front in x, on a mesh squeezed in x -- so the aspect criterion genuinely trips."""
    d = jno.shape.disk(0.0, 0.0, 0.5, size=0.09).domain(time=(0.0, T, n))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + 0.02 * (ui.x * vi.x + ui.y * vi.y),
            u(ci[0], ci[1]) - jno.np.tanh(12.0 * ci[0]),
            xi.d(ti) + 0.35 * jno.np.tanh(6.0 * xi),
        ]
    )
    return fem, fem.solve(adapt=adapt)


def _alignment(tr, k=8):
    """``(n_nodes, mean aspect of the k most stretched cells, mean |long axis . grad u| for those)``."""
    P, C = (np.asarray(x) for x in tr.meshes[-1])
    u = np.asarray(tr.states[-1])[: len(P)]
    ar, al = [], []
    for i, j, l in C:
        J = np.array([P[j] - P[i], P[l] - P[i]]).T
        if abs(np.linalg.det(J)) < 1e-30:
            continue
        U, sv, _ = np.linalg.svd(J)
        g = np.linalg.solve(J.T, np.array([u[j] - u[i], u[l] - u[i]]))
        ng = float(np.linalg.norm(g))
        if ng < 1e-12:
            continue
        ar.append(sv[0] / max(sv[1], 1e-30))
        al.append(abs(float(U[:, 0] @ (g / ng))))
    ar, al = np.array(ar), np.array(al)
    s = np.argsort(-ar)[:k]
    return len(P), float(ar[s].mean()), float(al[s].mean())


def _criterion(dd):
    return jno.le(dd.cell_aspect(), 2.0)


def test_the_metric_stretches_elements_along_the_feature():
    """Both remesh; only the metric one aligns the stretch with the level sets."""
    fem_i, iso = _front(jno.solve.remesh(criterion=_criterion, every=2))
    fem_a, ani = _front(jno.solve.remesh(criterion=_criterion, every=2, anisotropic=True))

    # not vacuous: a run that never remeshed would compare two identical marches
    for nm, fm in (("isotropic", fem_i), ("anisotropic", fem_a)):
        n_re = sum(1 for h in (getattr(fm, "adapt_history", []) or []) if h.get("remeshed"))
        assert n_re > 0, f"{nm} never remeshed, so this test compares nothing"

    n_i, ar_i, al_i = _alignment(iso)
    n_a, ar_a, al_a = _alignment(ani)

    assert ar_a > 1.5 * ar_i, f"metric did not stretch: aspect {ar_a:.2f} vs isotropic {ar_i:.2f}"
    assert al_a < 0.5 * al_i, f"stretch is not aligned with the feature: {al_a:.3f} vs isotropic {al_i:.3f}"
    assert al_a < 0.25, f"the long axis should lie ALONG the level sets, got |e.g| = {al_a:.3f}"


def test_anisotropic_still_needs_a_criterion_to_say_when():
    """The metric says HOW to remesh; a moving mesh still needs a condition saying WHEN."""
    with pytest.raises(NotImplementedError, match="criterion"):
        _front(jno.solve.remesh(anisotropic=True, every=2))
