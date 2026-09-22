"""Carrying a state across a REMESH must conserve it, not smear it.

A remesh changes the node set, so the state has to be moved between two different meshes. The obvious
route -- evaluate the old field at the new nodes -- is neither conservative nor optimal, and
``_l2_transfer_jax``'s own docstring measures what it costs: a translating Gaussian lost **27.6 % of its
peak over 2 steps**. That routine is the conservative alternative, but it requires both meshes to share
one cell array (its point location is a one-ring search), so a remesh could not use it.

Measured consequence, on a laser-heated drop whose field is a 300 -> 3000 K layer a few cells thick: one
remesh cost **540 K of peak and 6 % of the thermal energy**, and the melt pool visibly collapsed.

:func:`_l2_project_across_meshes` closes that gap -- the same Galerkin transfer, with the general
KD-tree point location that already existed for the h-adaptive path. This pins the property that makes
it worth having: ``int u`` is preserved.
"""

import numpy as np
from scipy.spatial import Delaunay

from jno.utils.solver.fem_adapt import _eval_fe_fields_at_points, _l2_project_across_meshes


def _disk(n_ring, seed, n_bnd=64):
    """A triangulated unit disk whose BOUNDARY is fixed and whose interior differs with ``seed``.

    The boundary has to be shared: two jittered boundaries are two different polygonal DOMAINS, and
    then `int u` over one is not comparable with `int u` over the other -- the comparison would measure
    the domains disagreeing, not the transfer. (The solver's own remesh preserves the surface to
    `hausd`, so a shared boundary is the faithful case.)"""
    rng = np.random.default_rng(seed)
    a = 2 * np.pi * np.arange(n_bnd) / n_bnd
    bnd = np.c_[np.cos(a), np.sin(a)]
    pts = [bnd]
    for i in range(1, n_ring):
        r = i / n_ring
        m = max(6, int(n_bnd * r))
        th = 2 * np.pi * (np.arange(m) + rng.random()) / m
        jit = 0.35 / n_ring * rng.standard_normal((m, 2))
        pts.append(np.c_[r * np.cos(th), r * np.sin(th)] + jit)
    pts.append(np.zeros((1, 2)))
    P = np.vstack(pts)
    P = P[np.hypot(P[:, 0], P[:, 1]) <= 1.0 + 1e-12]
    return P, np.asarray(Delaunay(P).simplices, dtype=np.int64)


def _integral(P, C, u):
    V = P[C]
    a = 0.5 * np.abs(
        (V[:, 1, 0] - V[:, 0, 0]) * (V[:, 2, 1] - V[:, 0, 1]) - (V[:, 2, 0] - V[:, 0, 0]) * (V[:, 1, 1] - V[:, 0, 1])
    )
    return float((u[C].mean(axis=1) * a).sum())


def _sharp(P):
    """A steep boundary-layer-like field: this is the case nodal interpolation destroys."""
    return np.exp(-((P[:, 1] - 0.55) ** 2 + P[:, 0] ** 2) / (2 * 0.16**2))


def test_a_remesh_transfer_conserves_the_integral():
    PA, CA = _disk(12, 0)
    PB, CB = _disk(14, 1)
    uA = _sharp(PA)
    lay = {"offsets": np.array([0, len(PA)]), "vecs": [1], "orders": [1]}
    layB = {"offsets": np.array([0, len(PB)]), "vecs": [1], "orders": [1]}
    uB = _l2_project_across_meshes(PA, CA, uA, lay, PB, CB, layB, 2, total_dst=len(PB))

    IA, IB = _integral(PA, CA, uA), _integral(PB, CB, np.asarray(uB))
    rel = abs(IB - IA) / abs(IA)
    assert rel < 0.02, f"integral not conserved across the remesh: {IA:.6f} -> {IB:.6f} ({rel:.2%})"
    # and the peak must survive: smearing a boundary layer is the failure this exists to prevent
    assert uB.max() > 0.80 * uA.max(), f"peak collapsed {uA.max():.3f} -> {uB.max():.3f}"


def test_the_conservative_transfer_beats_pointwise_interpolation():
    """Not just 'good enough' -- measurably better than the route it replaces, on the same pair."""
    import jax.numpy as jnp

    PA, CA = _disk(12, 0)
    PB, CB = _disk(14, 1)
    uA = _sharp(PA)
    lay = {"offsets": np.array([0, len(PA)]), "vecs": [1], "orders": [1]}
    layB = {"offsets": np.array([0, len(PB)]), "vecs": [1], "orders": [1]}
    IA = _integral(PA, CA, uA)

    u_l2 = np.asarray(_l2_project_across_meshes(PA, CA, uA, lay, PB, CB, layB, 2, total_dst=len(PB)))
    u_pw = np.asarray(
        _eval_fe_fields_at_points(PA, CA, jnp.asarray(uA), lay["offsets"], lay["orders"], [CA], lay["vecs"], [PB], dim=2)[0]
    ).reshape(-1)

    e_l2 = abs(_integral(PB, CB, u_l2) - IA) / abs(IA)
    e_pw = abs(_integral(PB, CB, u_pw) - IA) / abs(IA)
    assert e_l2 < e_pw, f"projection ({e_l2:.3%}) should beat interpolation ({e_pw:.3%})"
