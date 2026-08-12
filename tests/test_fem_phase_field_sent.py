"""Phase-field fracture, end to end — the capstone the six FEM additions were built for.

The study that motivated them: a single-edge-notched specimen pulled in tension, with the crack
regularized into a damage field. Every piece that used to need hand-rolled machinery around `jno.fem`
is now a term or a slot:

    sigma = jno.np.diff(psi, eps(u))                 # the stress IS the energy derivative
    Hs.evolves(maximum(Hs.i(-1), psi_p))             # irreversible history, on a COUPLED system
    dm.bounds(0.0, 1.0)                              # an inequality, in the term list
    fem.solve(nonlinear=jno.solve.staggered([u, dm]),  # alternate minimization
              tau=jno.solve.adaptive(limit=...))       # adaptive load path
    fem.eval(momentum, u_k)[grip]                    # the reaction force

Oracles here are physical, not restatements of the output: damage is bounded and monotone (never
clipped), it initiates *ahead of the notch tip* and localizes on the crack plane, the reaction rises
to a peak and then softens, and the 3-D case reproduces the 2-D one — a through-thickness notch in a
thin slab is a plane-strain problem, so the mid-plane damage must match.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jno

n = jno.np
E, NU = 210.0, 0.3
LAM, MU = E * NU / ((1 + NU) * (1 - 2 * NU)), E / (2 * (1 + NU))
GC, ELL, ETA = 2.7e-3, 0.06, 1e-6


def _sent(dim, *, h=0.03, thick=0.06, delta=1.4e-2, nout=8, limit=0.5):
    """A SENT specimen as a term list. ``dim=2`` plane strain, ``dim=3`` a thin slab with a
    through-thickness notch (the same problem, so the mid-plane answer must agree).

    ``nout`` is the reported step count AND the adaptive controller's first step (``span/(nout-1)``),
    so it does change the path: a coarser first step reaches further past the critical load before the
    controller cuts, and this specimen goes unstable there. 8 keeps the run inside the stable branch."""
    inner, sym, grad, trace = n.inner, n.sym, n.grad, n.trace
    maximum, minimum, diff, ident = n.maximum, n.minimum, n.diff, n.identity
    w = 0.010
    if dim == 2:
        shape = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=h) - jno.Shape.rect(-0.01, 0.5 - w, 0.5, 0.5 + w, size=h)
        top_pred, bot_pred = (lambda x, y: y > 1 - 1e-9), (lambda x, y: y < 1e-9)
        kbulk = LAM + MU  # plane strain
    else:
        shape = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, thick, size=h) - jno.Shape.box(
            -0.01, 0.5 - w, -0.01, 0.5, 0.5 + w, thick + 0.01, size=h
        )
        top_pred, bot_pred = (lambda x, y, z: y > 1 - 1e-9), (lambda x, y, z: y < 1e-9)
        kbulk = LAM + 2 * MU / 3  # 3-D bulk modulus
    dom = shape.domain(tau=(0.0, 1.0, nout))
    dom.tag("bot", bot_pred)
    dom.tag("top", top_pred)
    co, cb, ct = (dom.variable(r, split=True) for r in ("interior", "bot", "top"))
    X = list(co[:dim])
    u, phi = dom.fem_symbols(value_shape=(dim,))
    dm, q = dom.fem_symbols()
    Hs, _ = dom.fem_symbols(value_shape=())

    Id = ident(dim)
    eps = lambda v: sym(grad(v, X))  # noqa: E731
    e_u = eps(u)
    tr = trace(e_u)
    dev = e_u - tr / dim * Id
    psi_p = 0.5 * kbulk * maximum(tr, 0.0) ** 2 + MU * inner(dev, dev, 2)
    psi_m = 0.5 * kbulk * minimum(tr, 0.0) ** 2
    sigma = diff(((1.0 - dm) ** 2 + ETA) * psi_p + psi_m, e_u)
    momentum = inner(sigma, eps(phi), 2)
    fem = jno.fem(
        [
            momentum,
            (GC / ELL) * dm * q + GC * ELL * inner(grad(dm, X), grad(q, X), 1) - 2.0 * (1.0 - dm) * Hs.i(-1) * q,
            Hs.evolves(maximum(Hs.i(-1), psi_p)),
            dm.bounds(0.0, 1.0),
            *[u(*cb)[i] - 0.0 for i in range(dim)],
            *[u(*ct)[i] - 0.0 for i in range(dim) if i != 1],
            u(*ct)[1] - delta * ct[-1],  # displacement control, ramped in tau
        ]
    )
    alt = jno.solve.staggered([u, dm], max_sweeps=60, rtol=1e-7, atol=1e-9)
    traj = np.asarray(fem.solve(nonlinear=alt, tau=jno.solve.adaptive(limit=[(dm, limit)], max_steps=120)))
    return fem, traj, momentum, u, dm


def _check_physics(fem, traj, momentum, u, dm, dim):
    dmg = traj[:, fem.blocks[fem.block_index(dm)]]
    pts = np.asarray(fem.field_points[fem.block_index(dm)])

    # The bound holds exactly — a solve, not a clip.
    assert dmg.min() > -1e-12 and dmg.max() < 1.0 + 1e-12, f"damage left [0,1]: [{dmg.min():.2e}, {dmg.max():.4f}]"
    assert dmg.max() > 0.3, f"no damage developed (max {dmg.max():.3f}) — nothing to check"
    # The history field is a running max, so damage can only grow.
    assert np.all(np.diff(dmg.max(axis=1)) > -1e-9), "damage is not monotone — irreversibility broke"
    # It initiates AHEAD of the notch tip, on the crack plane.
    apex = pts[int(np.argmax(dmg[-1]))]
    assert apex[0] > 0.5 - ELL, f"damage must peak ahead of the notch tip x=0.5, got x={apex[0]:.3f}"
    assert abs(apex[1] - 0.5) < ELL, f"damage must peak on the crack plane y=0.5, got y={apex[1]:.3f}"
    # The reaction rises to a peak and softens (the grip corners are excluded: a fully clamped edge
    # meeting a free lateral edge is singular there, a property of the BCs rather than the model).
    grip = fem.region_dofs("top", field=u, component=1)
    R = np.abs([float(np.asarray(fem.eval(momentum, traj[k]))[grip].sum()) for k in range(traj.shape[0])])
    kp = int(np.argmax(R))
    assert 0 < kp < len(R) - 1, f"the reaction must peak inside the path, peaked at step {kp} of {len(R)}"
    assert R[-1] < 0.95 * R[kp], f"the specimen must soften after the peak: {R[-1]:.3e} vs {R[kp]:.3e}"
    # The adaptive controller actually adapted.
    sched = fem.tau_schedule
    assert len(sched) >= 3 and np.ptp(np.diff(sched)) > 1e-3, f"the load path did not adapt: {sched}"
    return dmg, pts, R


def test_sent_2d_initiates_localizes_and_softens():
    fem, traj, momentum, u, dm = _sent(2)
    dmg, pts, _R = _check_physics(fem, traj, momentum, u, dm, 2)
    # Localization is a statement about SUBSTANTIAL damage, and deliberately so: AT2 has no elastic
    # threshold, so d > 0 everywhere from the first increment (measured: a field-wide damage-weighted
    # spread is 0.24, i.e. dominated by that diffuse tail, and says nothing about the crack). Nodes past
    # the middle of the [0,1] range are the band. The few stragglers are the GRIP CORNERS, where a fully
    # clamped edge meets a free lateral edge and the elastic field is singular — a property of the
    # boundary conditions, not of the model.
    hot = dmg[-1] > 0.5
    assert hot.sum() > 5, f"too few substantially damaged nodes ({int(hot.sum())}) to speak of a band"
    inband = np.abs(pts[hot, 1] - 0.5) < 4 * ELL
    assert inband.mean() > 0.8, f"damage must localize on the crack plane ({100 * inband.mean():.0f}% in band)"


@pytest.mark.slow
def test_sent_3d_reproduces_the_plane_strain_answer():
    """A through-thickness notch in a thin slab IS the plane-strain problem, so the 3-D mid-plane damage
    must track the 2-D result. This is the first 3-D run of the coupled march + bounds + staggered
    stack, and the oracle is the 2-D answer rather than the 3-D run's own output."""
    fem3, traj3, mom3, u3, dm3 = _sent(3, h=0.06, thick=0.06, nout=5)
    dmg3, pts3, _R3 = _check_physics(fem3, traj3, mom3, u3, dm3, 3)

    fem2, traj2, mom2, u2, dm2 = _sent(2, h=0.06, nout=5)
    dmg2 = traj2[:, fem2.blocks[fem2.block_index(dm2)]]
    # Compare the through-thickness variation: a plane-strain field is z-independent, so the damage at
    # the two faces must agree with the mid-plane to a few percent of the peak.
    z = pts3[:, 2]
    lo, hi = z < z.min() + 1e-9, z > z.max() - 1e-9
    peak3 = dmg3[-1].max()
    assert peak3 > 0.3
    order = np.lexsort((np.round(pts3[lo, 1], 6), np.round(pts3[lo, 0], 6)))
    order_h = np.lexsort((np.round(pts3[hi, 1], 6), np.round(pts3[hi, 0], 6)))
    face_gap = np.abs(dmg3[-1][lo][order] - dmg3[-1][hi][order_h]).max() / peak3
    assert face_gap < 0.05, f"the 3-D field is not through-thickness uniform ({100 * face_gap:.1f}%)"
    # ...and the two dimensions agree on how much damage there is at the end of the same load path.
    assert abs(peak3 - dmg2[-1].max()) / max(peak3, dmg2[-1].max()) < 0.15, (
        f"3-D peak damage {peak3:.3f} vs 2-D {dmg2[-1].max():.3f}"
    )
