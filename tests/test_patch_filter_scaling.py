"""How the eq. (18) patch criterion behaves as the patch GROWS -- and where it stops working.

Jung, Yun & Kim, *Computers & Structures* **331** (2026) 108403. Sec. 2.3.2 leaves the
three-dimensional extension open, and these tests measure *why* it is open.

The criterion is a **geometric mean** over ``N - 2`` factors: the product runs ``i = 2 ... N-2``
(``N - 3`` factors) plus the single-dense-element term, under a ``1 / (N - 2)`` exponent. A mean
dilutes an outlier by ``1 / N``, so everything depends on how big the patch actually is:

===========================  ==================  =========
patch                        count               size
===========================  ==================  =========
2-D vertex (the paper)       ``3T / V``          ``~ 6``
3-D edge fan                 ``6T / E``          ``~ 5.2``
3-D vertex                   ``4T / V``          ``~ 27``
===========================  ==================  =========

on a Delaunay tetrahedral mesh, where ``T ~ 6.8V`` and ``E ~ 7.8V``. So a 3-D **edge fan** lands in
the same regime the formula was designed for -- in fact slightly better -- while a 3-D **vertex
patch** is four to five times larger than anything the paper evaluates.

What is measured below: the criterion works at ``N = 6``, and by ``N = 12`` the hinge signal is
already three-quarters gone. That is a property of any mean (a density-weighted geometric mean with
exponent ``1 / sum(rho)`` behaves the same), so it cannot be recovered by reweighting -- it is the
reason a 3-D vertex criterion needs an order statistic rather than eq. (18).

No real triangulation reaches ``N = 27``, so the patch topology is **synthesised** and handed to the
real :meth:`jno.Domain.patch_filter` kernel. ``test_the_synthetic_patch_matches_a_real_mesh``
pins that substitution against an actual mesh, so the numbers here are the shipped code's, not a
re-transcription's.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

# The reference element's own density in every probe below. The criterion scales rho_k by f, so
# taking rho_k = 1 makes the filter's output BE f and keeps the tables readable.
RK = 1.0


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _ring_topology(n: int) -> dict:
    """One interior vertex patch of exactly ``n`` elements, in :meth:`patch_topology`'s layout.

    ``others[k]`` is the ring rotated to start at the element adjacent to ``k`` and excluding ``k``
    -- the same counterclockwise walk the real builder emits (``ring[q+1:] + ring[:q]``). ``size``
    counts the patch INCLUDING ``k``, and every vertex is interior, so the single-dense-element term
    stays live rather than taking the Fig. 2d boundary branch.
    """
    others = np.full((n, 1, n - 1), -1, dtype=np.int64)
    for k in range(n):
        others[k, 0, :] = [(k + 1 + j) % n for j in range(n - 1)]
    return {
        "others": others,
        "size": np.full((n, 1), n, dtype=np.int64),
        "boundary": np.zeros((n, 1), dtype=bool),
    }


def _f_at(n: int, design) -> float:
    """Drive the REAL patch_filter kernel with a synthetic patch of size ``n``; return ``f``.

    With a single patch slot the filter reduces to ``rho_k * f``, so dividing by ``rho_k`` recovers
    the criterion itself.
    """
    d = jno.Shape.rect(0, 0, 2, 1, size=1.0).domain()  # tiny; its own topology is replaced below
    d.patch_topology = lambda: _ring_topology(n)
    r = np.asarray(design(n), dtype=float)
    assert r.shape == (n,), f"the probe must supply one density per patch element, got {r.shape}"
    return float(np.asarray(d.patch_filter()(r))[0] / max(r[0], 1e-300))


# --- the four configurations probed at every size -------------------------------------------
# Index 0 is always the reference element k, since _ring_topology walks outward from it.

RHO_MIN = 1e-3  # the SIMP floor the tutorial uses; a true zero is never reachable in a real run


def _lone_dense(n):
    """k dense, the whole rest of the patch void -- the single-dense-element defect."""
    r = np.full(n, RHO_MIN)
    r[0] = 1.0
    return r


def _hinge(n):
    """Two dense arcs meeting only at this vertex: a one-node connection."""
    r = np.full(n, RHO_MIN)
    arc = max(n // 6, 1)
    r[:arc] = 1.0
    r[n // 2 : n // 2 + arc] = 1.0
    return r


def _solid(n):
    return np.ones(n)


def _gray(n):
    return np.full(n, 0.5)


class TestThePaperRegime:
    """At the size the paper evaluates, the criterion does what it claims."""

    def test_it_reproduces_the_published_six_element_values(self):
        """N = 6 is ``3T/V`` on a 2-D triangulation -- the size eq. (18) was calibrated on.

        Both numbers are the shipped kernel's own, and they are what every larger patch below is
        compared against.
        """
        assert _f_at(6, _lone_dense) == pytest.approx(0.178, abs=0.005)
        assert _f_at(6, _gray) == pytest.approx(0.842, abs=0.005)

    def test_a_hinge_is_strongly_suppressed(self):
        f_hinge, f_solid = _f_at(6, _hinge), _f_at(6, _solid)
        assert f_solid == pytest.approx(1.0, abs=1e-12), "a full patch must pass through untouched"
        assert f_hinge < 0.1, f"a one-node connection must be caught at N=6, got f = {f_hinge:.4f}"

    def test_an_edge_fan_sized_patch_is_in_the_same_working_regime(self):
        """``6T/E ~ 5.2`` -- a 3-D edge fan. The criterion transfers to it verbatim.

        This is the load-bearing positive result: it is why the edge criterion needs no
        recalibration of the exponent or the aggregator, only an index remapping.
        """
        assert _f_at(5, _hinge) < 0.1, "the edge-fan regime must still catch a hinge"
        assert _f_at(5, _lone_dense) < 0.25
        assert _f_at(5, _solid) == pytest.approx(1.0, abs=1e-12)


class TestTheContrastCollapses:
    """The measurement that rules eq. (18) out for 3-D vertex patches."""

    def test_the_hinge_signal_is_three_quarters_gone_by_twelve_elements(self):
        """Detection is not lost gradually at ``N = 27``; it is mostly lost by ``N = 12``.

        Measured ``f_hinge / f_solid``: 0.04 at N=6, 0.11 at N=8, 0.76 at N=12. A valence-12 vertex
        is ordinary on an unstructured 2-D mesh, so this bounds the paper's own method too.
        """
        assert _f_at(8, _hinge) / _f_at(8, _solid) < 0.25, "N=8 must still discriminate"
        assert _f_at(12, _hinge) / _f_at(12, _solid) > 0.7, "N=12 must have lost the signal"

    def test_a_three_dimensional_vertex_patch_has_no_usable_contrast(self):
        """``4T/V ~ 27``. A genuine defect and a uniformly grey patch become indistinguishable.

        Measured at N=27: lone-dense 0.758, hinge 0.845, uniform-grey 0.870, solid 1.0. Both defects
        still sit *below* grey, but the separation that was 0.178-vs-0.842 at N=6 -- a factor 4.7 --
        is a factor 1.15 for a lone dense element and 1.03 for a hinge. Nothing downstream can act
        on a 3 % margin, and SIMP cannot finish the job either: 0.758 cubed is 0.44 of solid,
        against 0.006 for the 0.178 of a six-element patch.
        """
        f_lone, f_hinge, f_gray = _f_at(27, _lone_dense), _f_at(27, _hinge), _f_at(27, _gray)
        assert f_lone > 0.7, f"the defect is no longer suppressed at all: f = {f_lone:.4f}"
        assert abs(f_lone - f_gray) < 0.15, (
            f"a defect ({f_lone:.4f}) and uniform grey ({f_gray:.4f}) must be within 15 % -- if they "
            "separate again, the dilution argument for replacing eq. (18) in 3-D is void"
        )
        assert f_gray - f_hinge < 0.05, (
            f"a hinge ({f_hinge:.4f}) must be within 5 % of uniform grey ({f_gray:.4f}); it is the "
            "configuration eq. (18) exists to catch and at this size it is not distinguishable"
        )
        assert f_lone**3.0 > 0.4, "and SIMP at penal=3 no longer removes it"

    def test_the_contrast_degrades_monotonically_with_patch_size(self):
        """No threshold, no cliff -- so no patch size is a safe place to keep using eq. (18)."""
        sizes = (5, 6, 8, 12, 16, 20, 27)
        ratio = [_f_at(n, _hinge) / _f_at(n, _solid) for n in sizes]
        assert ratio[0] < 0.15 and ratio[-1] > 0.8, f"the sweep must span the transition, got {ratio}"
        assert all(b >= a - 1e-9 for a, b in zip(ratio, ratio[1:])), (
            f"contrast must degrade monotonically in N, got {[round(x, 3) for x in ratio]}"
        )


class TestTheHarnessIsFaithful:
    """The synthetic topology must be the real thing, or every number above is about nothing."""

    def test_the_synthetic_patch_matches_a_real_mesh(self):
        """A lone dense element on a real triangulation, against the synthetic ring.

        The comparison is exact for this configuration and only this one: an element sits in three
        vertex patches and the filter averages ``f`` over them, but when the whole neighbourhood is
        void each patch sees the same lone-dense case, so the average is the per-size value. That
        makes it the one configuration where a real mesh and a single synthetic ring must agree to
        machine precision -- which is what pins the substitution.
        """
        d = jno.Shape.rect(0, 0, 2, 1, size=0.2).domain()
        topo = d.patch_topology()
        n_cells = int(d._cells_p1().shape[0])
        k = int(np.where(~topo["boundary"].any(axis=1))[0][0])  # all three patches interior

        r = np.full(n_cells, RHO_MIN)
        r[k] = 1.0
        measured = float(np.asarray(d.patch_filter()(r))[k])

        valences = [int(topo["size"][k, v]) for v in range(3)]
        assert min(valences) >= 3, "the reference element needs three real patches"
        expected = float(np.mean([_f_at(n, _lone_dense) for n in valences]))
        assert measured == pytest.approx(expected, abs=1e-9), (
            f"synthetic {expected:.6f} vs real mesh {measured:.6f} at valences {valences}"
        )

    def test_no_real_two_dimensional_mesh_reaches_the_three_dimensional_patch_size(self):
        """Why the topology has to be synthesised at all -- and why N=27 is a 3-D-only regime."""
        biggest = 0
        for size in (0.5, 0.25, 0.12):
            biggest = max(biggest, int(jno.Shape.rect(0, 0, 2, 1, size=size).domain().patch_topology()["size"].max()))
        assert biggest < 15, f"a 2-D triangulation reached a patch of {biggest}; the framing needs revisiting"
