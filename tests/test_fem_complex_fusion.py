"""The complex fusion layer must not undo what the two-pass assembly bought.

The legs of a complex problem assemble with a host-planned unique-sorted pattern, but the layer
ABOVE them lagged on the old design, in two places:

  * ``_complex_block_bcoo`` concatenated four leg copies UNSORTED, so every downstream CSR
    conversion -- each sparse solve, each preconditioner build -- re-sorted the full ``2n`` pattern
    on device, per solve;
  * ``_complex_operator`` (the complex-native AMS/eddy route) collapsed the concatenated legs with
    an in-trace ``sum_duplicates()`` -- a device argsort over the raw triplets, the exact workspace
    spike ``compress_plan`` exists to avoid.

Both patterns are host-static (fixed by mesh and terms), so the order is decided once in numpy and
the device work is one O(nnz) scatter. Traced indices (a jit-captured re-form) keep the plain
concatenation -- correctness never depends on the plan, which is what the fallback test pins.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")
import jax  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import jno  # noqa: E402

inner, vec = jno.np.inner, jno.np.vector


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def complex_fem():
    """A small complex N1E curl-curl problem: gauged by a complex mass term, PEC boundary."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
        u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
        ci = d.variable("interior", split=True)
        x, y, z = ci[0], ci[1], ci[2]
        yield jno.fem(
            [
                inner(u.vector.curl(x, y, z), v.vector.curl(x, y, z))
                + (1e-3 + 0.5j) * inner(u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z))
                - inner(vec(1.0 + 0.0 * x, 0.0 * x, 0.0 * x), v.bind(x=x, y=y, z=z)),
                u.vector.cross(d.variable("boundary", normals=True)),
            ]
        )
    finally:
        jax.config.update("jax_enable_x64", prev)


def _coo(A):
    return sp.coo_matrix(
        (np.asarray(A.data), (np.asarray(A.indices[:, 0]), np.asarray(A.indices[:, 1]))), shape=A.shape
    ).tocsr()


def test_fused_block_is_sorted_and_unique(complex_fem):
    """Downstream CSR conversions must find the pattern already canonical -- both flags, no device sort."""
    A = complex_fem._A
    assert hasattr(A, "indices"), "the fused 2n system must stay sparse"
    assert bool(A.indices_sorted), "fused block emitted unsorted: every solve re-sorts 4x the leg nnz"
    assert bool(A.unique_indices), "the four sub-blocks are disjoint by construction; say so to XLA"
    keys = np.asarray(A.indices[:, 0]).astype(np.int64) * A.shape[1] + np.asarray(A.indices[:, 1])
    assert np.all(np.diff(keys) > 0), "indices_sorted=True must be TRUE, not asserted"


def test_fused_block_values_match_the_legs(complex_fem):
    """The plan is a reordering, not an arithmetic change: [[Ar,-Ai],[Ai,Ar]] entry-for-entry."""
    (Ar, _br), (Ai, _bi) = complex_fem._complex_legs
    ref = sp.bmat([[_coo(Ar), -_coo(Ai)], [_coo(Ai), _coo(Ar)]], format="csr")
    got = _coo(complex_fem._A)
    err = abs(ref - got).max()
    assert err == 0.0, f"fused block drifted from the legs by {err}"


def test_complex_operator_is_planned_not_sorted_on_device(complex_fem, monkeypatch):
    """The AMS/eddy route's A_r + i*A_i must come out canonical WITHOUT sum_duplicates (device sort)."""
    from jax.experimental import sparse as jsp

    from jno._fem import _complex_operator

    def _boom(self, *a, **k):
        raise AssertionError("sum_duplicates ran: the complex operator still sorts on device")

    monkeypatch.setattr(jsp.BCOO, "sum_duplicates", _boom)
    (Ar, _), (Ai, _) = complex_fem._complex_legs
    Ac = _complex_operator(Ar, Ai)
    assert bool(Ac.indices_sorted) and bool(Ac.unique_indices)
    keys = np.asarray(Ac.indices[:, 0]).astype(np.int64) * Ac.shape[1] + np.asarray(Ac.indices[:, 1])
    assert np.all(np.diff(keys) > 0)
    ref = (_coo(Ar) + 1j * _coo(Ai)).tocsr()
    err = abs(ref - _coo(Ac)).max()
    assert err == 0.0, f"complex operator drifted from the legs by {err}"


def _strip_flags(A):
    """The same operator with the canonical flags dropped — the shape a non-planned producer hands in."""
    from jax.experimental import sparse as jsp

    return jsp.BCOO((A.data, A.indices), shape=A.shape)


def test_unflagged_legs_take_the_compress_plan_and_stay_canonical(complex_fem):
    """A concrete but unflagged leg (a non-planned producer) cannot use the counting merge; the
    compress-plan route must still emit a canonical block with unchanged values."""
    from jno._fem import _complex_block_bcoo

    (Ar, _), (Ai, _) = complex_fem._complex_legs
    blk = _complex_block_bcoo(_strip_flags(Ar), _strip_flags(Ai), complex_fem._complex_n)
    assert bool(blk.indices_sorted) and bool(blk.unique_indices)
    ref = sp.bmat([[_coo(Ar), -_coo(Ai)], [_coo(Ai), _coo(Ar)]], format="csr")
    assert abs(ref - _coo(blk)).max() == 0.0


def test_traced_legs_fall_back_to_the_concatenation(complex_fem, monkeypatch):
    """No host plan at all (traced indices) -> the old concatenation, still correct. Both plans are
    optimisations; a jit-captured re-form must never crash or change values."""
    from jno._fem import _complex_block_bcoo, _complex_operator
    from jno.utils.solver import fem_utils

    def _traced(indices):
        raise RuntimeError("simulated: no host plan available under trace")

    monkeypatch.setattr(fem_utils, "compress_plan", _traced)
    (Ar, _), (Ai, _) = complex_fem._complex_legs
    blk = _complex_block_bcoo(_strip_flags(Ar), _strip_flags(Ai), complex_fem._complex_n)
    ref = sp.bmat([[_coo(Ar), -_coo(Ai)], [_coo(Ai), _coo(Ar)]], format="csr")
    assert abs(ref - _coo(blk)).max() == 0.0
    Ac = _complex_operator(_strip_flags(Ar), _strip_flags(Ai))
    ref_c = (_coo(Ar) + 1j * _coo(Ai)).tocsr()
    assert abs(ref_c - _coo(Ac)).max() == 0.0


def test_the_solution_is_unchanged(complex_fem):
    """The reorder must be invisible to the physics: solve and compare against the scipy reference."""
    (Ar, _br), (Ai, _bi) = complex_fem._complex_legs
    import scipy.sparse.linalg as spla

    Ac = (_coo(Ar) + 1j * _coo(Ai)).tocsc()
    b_c = np.asarray(_br) + 1j * np.asarray(_bi)
    ref = spla.spsolve(Ac, b_c)
    sol = np.asarray(jno.np.asarray(complex_fem.solve()))
    assert np.allclose(sol.reshape(-1), ref.reshape(-1), rtol=1e-8, atol=1e-12)
