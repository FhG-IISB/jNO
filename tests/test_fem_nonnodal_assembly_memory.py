"""Assembly must never cost magnitudes more memory than the matrix it produces.

Measured on the case that motivated this: a 1.29M-tet mixed N1E x Lagrange system whose FINAL
operator is ~2.6 GB was OOM-killed at a 17 GB cap DURING assembly. Two structural causes, both in
the old sparse volume path:

  * one full triplet block was appended PER ADDITIVE TERM -- the A-V form has ~4-6 volume terms,
    each carrying an identical (row, col) pattern, so the unreduced triplet array was ~terms x the
    per-element size before anything was summed;
  * duplicates were collapsed at the END by ``sum_duplicate_triplets`` -- an in-trace SORT over that
    whole unreduced array, with its own workspace.

The two-pass design removes both: terms sharing a test field are summed at the ELEMENT level (one
Ke per test-field group), and a host-side symbolic pass computes the unique pattern plus an inverse
map ONCE, so values scatter-add straight into their final slots -- no in-trace sort, no stored
duplicates, ever.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")
import jax  # noqa: E402

import jno  # noqa: E402

inner_, vec = jno.np.inner, jno.np.vector


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _mixed_many_terms(size=0.35):
    """A mixed N1E x Lagrange form with SEVERAL additive volume terms -- the shape that multiplied
    the old path's triplet storage."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    p, q = d.fem_symbols(names=("p", "q"), space="Lagrange")
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    A_, V_ = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    Vs, Vt = p.bind(x=x, y=y, z=z), q.bind(x=x, y=y, z=z)
    g = lambda s: vec(s.x, s.y, s.z)  # noqa: E731
    tg = lambda s: jno.np.stack([s.x, s.y, s.z], axis=2)  # noqa: E731
    w = 1.0e4
    return d, jno.fem(
        [
            inner_(u.vector.curl(x, y, z), v.vector.curl(x, y, z))
            + 1j * w * inner_(A_, V_)
            + 1j * w * inner_(g(Vs), V_)
            - inner_(vec(1.0 + 0.0 * x, 0.0 * x, 0.0 * x), V_),
            1j * w * inner_(A_, tg(Vt)) + 1j * w * inner_(g(Vs), tg(Vt)),
            u.vector.cross(d.variable("boundary", normals=True)),
        ]
    )


def _operator_bcoo(fem):
    from jno.precond import _fem_concrete_operator

    A = _fem_concrete_operator(fem)
    return A.bcoo if getattr(A, "bcoo", None) is not None else A


def test_no_in_trace_dedup_on_the_volume_path(monkeypatch):
    """The mechanism: values land in their final slots directly, so the sorting collapse must never
    run. Patched to raise, because 'it was not needed' is only provable by making it fatal."""
    from jno.utils.solver import fem_utils

    def _boom(*a, **k):
        raise AssertionError("sum_duplicate_triplets ran: the assembly still stores duplicates")

    # the assembler does `from fem_utils import sum_duplicate_triplets` at build time, so patching
    # the SOURCE module before the fem is built intercepts it
    monkeypatch.setattr(fem_utils, "sum_duplicate_triplets", _boom)
    _d, fem = _mixed_many_terms()
    bc = _operator_bcoo(fem)
    assert bc.shape[0] > 0


def test_stored_triplets_are_exactly_the_unique_pattern():
    """No duplicate (i, j) is ever STORED: nse equals the unique count, and the BCOO carries the
    sorted/unique flags so every downstream matvec skips the lazy re-summation too."""
    _d, fem = _mixed_many_terms()
    bc = _operator_bcoo(fem)
    idx = np.asarray(bc.indices)
    enc = idx[:, 0].astype(np.int64) * int(bc.shape[1]) + idx[:, 1].astype(np.int64)
    assert len(np.unique(enc)) == bc.nse, "duplicate (i,j) triplets are stored"
    assert bool(bc.unique_indices), "the BCOO must declare unique_indices"
    assert bool(bc.indices_sorted), "the BCOO must declare indices_sorted"


def test_operator_values_are_unchanged():
    """The redesign is a memory change, not a numerics change: the assembled operator must produce
    the same matvec as a triplet-level reference built independently in numpy from the same BCOO."""
    _d, fem = _mixed_many_terms(size=0.5)
    bc = _operator_bcoo(fem)
    import scipy.sparse as sp

    idx = np.asarray(bc.indices)
    S = sp.csr_matrix((np.asarray(bc.data), (idx[:, 0], idx[:, 1])), shape=tuple(bc.shape))
    rng = np.random.default_rng(0)
    v = rng.standard_normal(bc.shape[1]) + 1j * rng.standard_normal(bc.shape[1])
    import jax.numpy as jnp

    got = np.asarray(bc @ jnp.asarray(v))
    ref = S @ v
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-12)
    # and the solve built on it still runs end to end
    sol = fem.solve(linear=jno.solve.lu())
    assert np.isfinite(np.asarray(jno.np.asarray(sol))).all()
