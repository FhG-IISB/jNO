"""A lattice element is a CUBE, and the quadrature has to sample it like one.

The sub-point rule placed points only ALONG an element's axis, so its cross-section was a point. For
a wire that is right -- a filament is long and thin by construction. For a bar lattice it is not:
neighbouring cells sit one pitch apart while extending a full pitch transversely, so every
near-neighbour mutual was over-counted. Measured against the volume integral it is +1.2 % when the
element is 8x longer than it is thick, +15.3 % when it is a cube, and +48.8 % when it is 8x shorter.

The consequence was the worst kind: not an obvious error but a WRONG TREND. On the copper bar of
Romano et al. (IEEE TEMC 65(2), 2023, sec. V-A), where Q3D and three PEEC variants agree at 2.85 nH,
refining the lattice laterally made the answer move away from it.

The oracle here is a volume Monte-Carlo of the Neumann double integral -- crude, but wrong in a
completely different way from a Gauss rule, which is the point of it.
"""

import jax
import numpy as np
import pytest

import jno

jax.config.update("jax_enable_x64", True)
from jno.utils.solver.peec import bar_filaments

MU0 = 4e-7 * np.pi


def mc_mutual(l, w, t, d, n=3_000_000, seed=0):
    """The Neumann double integral between two parallel bars, by brute force."""
    rng = np.random.default_rng(seed)
    a = rng.random((n, 3)) * np.array([w, l, t]) + np.array([-w / 2, -l / 2, -t / 2])
    b = rng.random((n, 3)) * np.array([w, l, t]) + np.array([d - w / 2, -l / 2, -t / 2])
    return MU0 / (4 * np.pi) * l**2 * np.mean(1.0 / np.linalg.norm(a - b, axis=1))


def _two_bar_mutual(l, w, t, d, quad=3, quad_t=2):
    """M between two parallel lattice-shaped bars, through jNO's own sub-point machinery."""
    import jax.numpy as jnp
    from jno.utils.solver.kernel import pair_matrix

    gl, wl = np.polynomial.legendre.leggauss(quad)
    gt, wt = np.polynomial.legendre.leggauss(quad_t)
    ox, oy, oz = (w / 2) * gt, (l / 2) * gl, (t / 2) * gt
    P = np.stack(np.meshgrid(ox, oy, oz, indexing="ij"), -1).reshape(-1, 3)
    W = ((wt[:, None, None] / 2) * (wl[None, :, None] / 2) * (wt[None, None, :] / 2)).reshape(-1) * l
    pos = np.concatenate([P, P + np.array([d, 0.0, 0.0])])
    mom = np.concatenate([np.stack([np.zeros_like(W), W, np.zeros_like(W)], -1)] * 2)
    grp = np.concatenate([np.zeros(len(P), int), np.ones(len(P), int)])
    M = np.asarray(pair_matrix(jnp.asarray(pos), jnp.asarray(mom), lambda r: 1.0 / r,
                               jnp.asarray([1.0, 1.0]), group=grp)) * MU0 / (4 * np.pi)
    return M[0, 1]


@pytest.mark.parametrize("l,t,tol", [(0.5e-3, 0.0625e-3, 0.03), (0.25e-3, 0.25e-3, 0.06),
                                     (0.0625e-3, 0.5e-3, 0.12)])
def test_the_transverse_rule_fixes_the_near_neighbour_mutual(l, t, tol):
    """Across the aspect ratios a lattice actually produces, including the pathological one."""
    w = 0.0625e-3
    d = 2 * w
    ref = mc_mutual(l, w, t, d)
    got = _two_bar_mutual(l, w, t, d, quad=3, quad_t=2)
    assert abs(got / ref - 1) < tol, f"l/t={l/t:.2f}: {got * 1e12:.4f} pH vs {ref * 1e12:.4f} pH"


@pytest.mark.parametrize("l,t", [(0.25e-3, 0.25e-3), (0.0625e-3, 0.5e-3)])
def test_the_transverse_rule_is_strictly_better_than_the_line_rule(l, t):
    """The regression guard: whatever the tolerance, sampling the cross-section must help."""
    w = 0.0625e-3
    d = 2 * w
    ref = mc_mutual(l, w, t, d)
    line = abs(_two_bar_mutual(l, w, t, d, quad=3, quad_t=1) / ref - 1)
    vol = abs(_two_bar_mutual(l, w, t, d, quad=3, quad_t=2) / ref - 1)
    assert vol < 0.5 * line, f"line {100*line:.1f} % -> volume {100*vol:.1f} %"


def test_a_lattice_elements_moments_still_sum_to_its_length():
    """Whatever the rule, an element's total current moment is length x tangent. It is the one
    invariant the whole partial-inductance machinery rests on."""
    f = bar_filaments(jno.Shape.box(0, 0, 0, 4e-3, 2e-3, 1e-3), size=(1e-3, 1e-3, 1e-3))
    mom, grp = np.asarray(f.mom), np.asarray(f.group)
    ln = np.asarray(f.length)
    ne = int(grp.max()) + 1
    tot = np.zeros((ne, 3))
    np.add.at(tot, grp, mom)
    assert np.allclose(np.linalg.norm(tot, axis=1), ln, rtol=1e-12)


def test_the_published_bar_converges_instead_of_diverging():
    """Romano et al. sec. V-A: 2.85 nH. Refining laterally with the thickness at one cell used to
    give 3.01 -> 3.23 -> 3.41. It must now move toward the answer, not away from it."""
    W = T = 0.5e-3
    L0 = 5.0e-3
    got = []
    for p in (0.25e-3, 0.125e-3):
        bar = jno.Shape.box(0, 0, 0, W, L0, T, size=(p, p, T)).attach(sigma=5.8e7).name("bar")
        d = bar.domain()
        d.tag("A", lambda x, y, z, p=p: y < p * 1.01)
        d.tag("B", lambda x, y, z, p=p: y > L0 - p * 1.01)
        _i, v = d.peec_symbols()
        at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
        s = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e3).build().solve()
        got.append(float(np.real(s.L)) * 1e9)
    # it converges from BELOW (the lattice conducts centre-to-centre, so a coarse grid is short),
    # so the test is that refining gets CLOSER, not that it decreases
    assert abs(got[1] - 2.847) < abs(got[0] - 2.847), f"not converging: {got}"
    assert abs(got[1] - 2.847) < 0.15, f"{got} nH against 2.847"


def _bar_L(cells, quad_t, hz=1.0):
    """Partial self-inductance of a 38 x 4 x 2 mm bar, sliced into ``cells`` through the thickness.

    At 1 Hz the skin depth is 66 mm against a 2 mm thickness, so the current is uniform and L is a
    pure geometry-and-quadrature quantity: it cannot depend on the slicing.
    """
    from jno.utils.solver.peec import solve_network, terminal_nodes

    sig, t = 5.8e7, 0.002
    f = bar_filaments(jno.Shape.box(0, 0, 0, 0.040, 0.004, t), size=(0.002, 0.004, t / cells), quad_t=quad_t)
    p = np.asarray(f.nodes)
    a = terminal_nodes(f, lambda q: q[:, 0] < p[:, 0].min() + 1e-9)
    b = terminal_nodes(f, lambda q: q[:, 0] > p[:, 0].max() - 1e-9)
    _c, _phi, inj = solve_network(
        f, sig, {"A": a, "B": b}, [("A", "B", 1.0 + 0j)], omega=2 * np.pi * hz, matrix_free=False
    )
    return complex(1.0 / inj["A"]).imag / (2 * np.pi * hz)


GROVER = 2e-7 * 0.038 * (np.log(2 * 0.038 / 0.006) + 0.5 + 0.2235 * 0.006 / 0.038)


def test_the_assembled_bar_inductance_moves_onto_grovers_formula():
    """An analytic oracle for the whole assembled solve, not just one element pair.

    Grover, *Inductance Calculations* (1946), rectangular bar:
    L = (mu0 l / 2pi) [ln(2l/(w+t)) + 1/2 + 0.2235 (w+t)/l] = 23.364 nH here, good to a couple of
    percent at this bar's l/(w+t) = 6.3. The line rule sat 25 % above it; the volume rule halves
    that and keeps closing as the transverse order rises (23.92 nH, +2.4 %, at 8 cells / order 4).

        order 1    29.320 nH   +25.5 %
        order 2    25.630 nH    +9.7 %
    """
    line, volume = _bar_L(1, quad_t=1), _bar_L(1, quad_t=2)
    assert line / GROVER - 1 > 0.20  # the rule this replaced really was that far out
    assert 0.0 < volume / GROVER - 1 < 0.15  # closed from above, and not overshot
    assert volume < line


def test_slicing_a_bar_through_its_thickness_does_not_change_its_inductance():
    """The defect's clearest signature: a partial inductance that depended on the SLICING.

    Every added cell brought more near-neighbour mutuals, and the line rule over-counted each one,
    so L climbed with refinement -- a wrong TREND, which is worse than a wrong number because it
    survives a convergence study. Over 1, 8 and 16 cells at 1 Hz, where L cannot legitimately move:

        order 1    29.320  29.476  31.138 nH    +6.2 %
        order 2    25.630  24.823  25.564 nH    -0.3 %
    """
    volume = [_bar_L(n, quad_t=2) for n in (1, 8, 16)]
    assert abs(volume[-1] / volume[0] - 1) < 0.02

    line = [_bar_L(n, quad_t=1) for n in (1, 8, 16)]
    assert line[-1] / line[0] - 1 > 0.04  # the drift the volume rule removes
