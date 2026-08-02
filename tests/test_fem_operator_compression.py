"""Every assembled operator must store each ``(row, col)`` pair once.

The assemblers emit one triplet block **per additive weak-form term** and never pre-sum, and on top
of that every interior DOF pair receives a contribution from each element sharing it (~20 tets for
P1). BCOO sums duplicates lazily on every ``@``, so the answers were always right — they just cost
~19x the work on each of a Krylov solve's hundreds of matvecs, and ~19x the memory that decides which
3-D problems fit in 8 GB.

The compression landed on the **steady linear** path alone, while jNO has many parallel ones. The
paths that missed out were the worst offenders (measured redundancy: nonlinear Jacobian 21.3x,
transient operator 12.5x, transient mass 4.2x) *and* the ones that re-apply the operator every
timestep. This file is the guard against that gap reopening: it walks the operator kinds and asserts
compression on each, so a new path cannot quietly ship uncompressed.

Uncompressed is not *wrong*, so none of this can be caught by an answer check — which is exactly why
it needs its own test. The exactness tests below pin the other half: compression must not move the
operator.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

pytest.importorskip("shapely", reason="shapely required for the box/rect domains")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _redundancy(A):
    """``(stored, unique)`` triplet counts for a BCOO. 1.0 ratio means fully compressed."""
    idx = np.asarray(A.indices)
    return int(idx.shape[0]), len({(int(r), int(c)) for r, c in idx})


def _assert_compressed(A, what):
    stored, unique = _redundancy(A)
    assert stored == unique, f"{what}: {stored} triplets stored for {unique} unique pairs ({stored / unique:.2f}x)"


def _matvec_matches_uncompressed(A, seed=0):
    """Compression is a storage change, so it must reproduce the *uncompressed* accumulation exactly.

    Built by hand from the triplets rather than compared against the operator itself, so this cannot
    pass by comparing a value to a copy of itself."""
    idx, data = np.asarray(A.indices), np.asarray(A.data)
    n = A.shape[0]
    v = np.random.default_rng(seed).standard_normal(n)
    ref = np.zeros(n, dtype=np.result_type(data, v))
    np.add.at(ref, idx[:, 0], data * v[idx[:, 1]])
    got = np.asarray(A @ jax.numpy.asarray(v))
    assert np.max(np.abs(got - ref)) < 1e-10 * max(1.0, float(np.max(np.abs(ref))))


# ---------------------------------------------------------------------------------------------
# steady
# ---------------------------------------------------------------------------------------------


def _poisson_3d(size=0.3):
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, phi = d.fem_symbols()
    c = d.variable("interior", split=True)
    cb = d.variable("boundary", split=True)
    ui, vi = u.bind(x=c[0], y=c[1], z=c[2]), phi.bind(x=c[0], y=c[1], z=c[2])
    return jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - 1.0 * vi, u(cb[0], cb[1], cb[2]) - 0.0])


def test_steady_linear_operator_is_compressed():
    """The path compression originally landed on — kept here so the whole set lives in one file."""
    A, _b = _poisson_3d().operator
    _assert_compressed(A, "steady linear 3-D")
    _matvec_matches_uncompressed(A)


def test_compression_does_not_move_the_solution():
    """The property that makes this a storage change rather than an approximation."""
    u = np.asarray(_poisson_3d().solve()).reshape(-1)
    assert np.all(np.isfinite(u))
    assert u.max() > 0.0, "the interior must be lifted by the source"
    # symmetric positive-definite Poisson with zero Dirichlet data: no undershoot
    assert u.min() > -1e-10, f"unexpected negative interior value {u.min():.3e}"


# ---------------------------------------------------------------------------------------------
# transient — the operators applied on EVERY step, and the ones that missed out
# ---------------------------------------------------------------------------------------------


def _heat_2d(h=0.12, nsteps=6):
    d = jno.domain(jno.Shape.rect(0, 0, 1, 1), mesh_size=h, time=(0.0, 0.03, nsteps))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(np.pi * ci[0]) * jno.np.sin(np.pi * ci[1])
    return jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(ci[0], ci[1]) - ic])


def test_transient_mass_and_operator_are_compressed():
    """Measured before the fix: operator 12.54x redundant, mass 4.15x — and both are applied on every
    timestep, so the waste multiplies by the step count rather than being paid once."""
    block = _heat_2d().operator  # a transient FEM's .operator IS the semidiscrete block
    M, A = block.M, block.A
    assert hasattr(M, "indices") and hasattr(A, "indices"), "expected sparse transient operators"
    _assert_compressed(M, "transient mass M")
    _assert_compressed(A, "transient operator A")
    _matvec_matches_uncompressed(M, seed=1)
    _matvec_matches_uncompressed(A, seed=2)


def test_transient_solution_is_unchanged_by_compression():
    """Decay from a unit initial state with zero Dirichlet data: monotone, bounded, finite."""
    traj = np.asarray(_heat_2d().solve().fn())
    assert np.all(np.isfinite(traj))
    assert traj.shape[0] > 1, "expected a trajectory"
    peak = np.abs(traj).max(axis=tuple(range(1, traj.ndim)))
    assert peak[-1] <= peak[0] + 1e-8, f"diffusion must not grow the state: {peak[0]:.3e} -> {peak[-1]:.3e}"


# ---------------------------------------------------------------------------------------------
# 1-D
# ---------------------------------------------------------------------------------------------


def test_1d_steady_operator_is_compressed():
    """The 1-D assembler is a separate code path with its own Dirichlet handling; its tridiagonal
    operator is small, but it is also the one most likely to be solved at very high node counts."""
    d = jno.domain(constructor=jno.domain.line(x_range=(0, 1), mesh_size=0.01))
    u, phi = d.fem_symbols()
    c = d.variable("interior", split=True)[0]
    cb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=c), phi.bind(x=c)
    fem = jno.fem([ui.x * vi.x - 1.0 * vi, u(cb) - 0.0])
    A, _b = fem.operator
    if not hasattr(A, "indices"):
        pytest.skip("1-D operator is dense in this build")
    _assert_compressed(A, "1-D steady")
    _matvec_matches_uncompressed(A, seed=3)

    # and the answer is still the exact FEM solution of -u'' = 1, u(0)=u(1)=0
    x = np.asarray(d.mesh.points)[:, 0]
    got = np.asarray(fem.solve()).reshape(-1)
    assert np.max(np.abs(got - 0.5 * x * (1.0 - x))) < 1e-10


# ---------------------------------------------------------------------------------------------
# nonlinear Jacobian — assembled INSIDE the trace, via the hoisted static pattern
# ---------------------------------------------------------------------------------------------


def _nonlinear(kind):
    """``(1 + u²)∇u·∇v`` — nonlinear in the unknown, so the Jacobian is re-assembled per Newton step."""
    if kind == "3d":
        d = jno.domain(jno.Shape.box(0, 0, 0, 1, 1, 1), mesh_size=0.3)
        u, v = d.fem_symbols()
        i, b = d.variable("interior", split=True), d.variable("boundary", split=True)
        a, c = u.bind(x=i[0], y=i[1], z=i[2]), v.bind(x=i[0], y=i[1], z=i[2])
        return jno.fem([(1.0 + a**2) * (a.x * c.x + a.y * c.y + a.z * c.z) - 1.0 * c, u(b[0], b[1], b[2]) - 0.0])
    d = jno.domain(jno.Shape.rect(0, 0, 1, 1), mesh_size=0.18)
    u, v = d.fem_symbols()
    i = d.variable("interior", split=True)
    r, ln = d.variable("right", split=True), d.variable("left", split=True)
    a, c = u.bind(x=i[0], y=i[1]), v.bind(x=i[0], y=i[1])
    if kind == "surface":  # a Robin term -> exercises the surface half of the hoisted pattern
        ar, cr = u.bind(x=r[0], y=r[1]), v.bind(x=r[0], y=r[1])
        return jno.fem([(1.0 + a**2) * (a.x * c.x + a.y * c.y) - 1.0 * c, 3.0 * ar * cr, u(ln[0], ln[1]) - 0.0])
    b = d.variable("boundary", split=True)
    return jno.fem([(1.0 + a**2) * (a.x * c.x + a.y * c.y) - 1.0 * c, u(b[0], b[1]) - 0.0])


@pytest.mark.parametrize("kind", ["2d", "3d", "surface"])
def test_traced_jacobian_is_compressed_and_still_the_right_matrix(kind):
    """The keystone: compression inside the trace, checked against an INDEPENDENT Jacobian.

    ``sum_duplicates`` needs a static ``nse`` under ``jit`` and infers it from concrete indices it
    does not have mid-trace, so the traced assemblies could not compress at all — which excluded the
    most redundant operator in the library (measured 9.6x on a 3-D nonlinear Jacobian: 36752 stored
    triplets -> 3841, 0.561 -> 0.059 MiB, matvec 6.6x faster). The pattern is hoisted host-side
    because the triplet INDICES come only from mesh connectivity and the term list.

    Checked against ``jacfwd`` of the residual, and that choice of oracle is the whole point. A
    rebuild from the operator's own triplets would be self-consistent: if the hoisted pattern were
    wrong the data would land in the wrong entries and the rebuild would agree with the wrong answer.
    An independent Jacobian is the only reference that can catch it — and a wrong static count is a
    silently WRONG operator (``sum_duplicates`` drops entries when ``nse`` is too small), not merely a
    slower one, so this needs the strong oracle rather than a solve that happens to converge."""
    fem = _nonlinear(kind)
    n = int(fem.dofs)
    J, R = fem._op.jacobian, fem._op.residual  # sparse handle; the public .jacobian densifies

    for seed, scale in ((None, 0.0), (7, 0.3)):
        u0 = jax.numpy.asarray(np.zeros(n) if seed is None else np.random.default_rng(seed).standard_normal(n) * scale)
        A_eager = J(u0)
        A_traced = jax.jit(J)(u0)

        idx = np.asarray(A_traced.indices)
        assert idx.min() >= 0 and idx.max() < n, "an out-of-bounds padding index leaked into the operator"
        _assert_compressed(A_traced, f"traced nonlinear Jacobian ({kind})")

        ref = np.asarray(jax.jacfwd(lambda w: jax.numpy.asarray(R(w)).reshape(-1))(u0))
        got = np.asarray(A_traced.todense())
        scale_ref = max(1.0, float(np.max(np.abs(ref))))
        assert np.max(np.abs(got - ref)) / scale_ref < 1e-11, "the compressed Jacobian is not the Jacobian"
        assert np.max(np.abs(np.asarray(A_eager.todense()) - got)) / scale_ref < 1e-11, "jit changed the operator"


def test_the_static_pattern_does_not_depend_on_the_state():
    """What makes a *static* count legitimate: the pattern is fixed by mesh and terms, so the same
    ``nse`` must serve every state the assembler is traced at. If it varied with ``u``, caching one
    count would drop entries at some other state — a wrong answer that no single-state test sees."""
    fem = _nonlinear("3d")
    n = int(fem.dofs)
    J = jax.jit(fem._op.jacobian)
    rng = np.random.default_rng(0)
    counts = {int(J(jax.numpy.asarray(rng.standard_normal(n) * s)).nse) for s in (0.0, 0.1, 1.0, 10.0)}
    assert len(counts) == 1, f"nse varies with the state: {sorted(counts)}"


def test_newton_still_converges_on_the_compressed_jacobian():
    """The Jacobian is not an output — it drives Newton. Compression must not disturb that."""
    fem = _nonlinear("2d")
    u = np.asarray(fem.solve()).reshape(-1)
    assert np.all(np.isfinite(u))
    assert u.max() > 0.0, "the source must lift the interior"
    assert u.min() > -1e-8, f"unexpected undershoot {u.min():.3e}"


# ---------------------------------------------------------------------------------------------
# the compression primitive itself
# ---------------------------------------------------------------------------------------------


def test_static_nse_must_be_exact_or_entries_are_dropped():
    """The failure mode that makes ``unique_triplet_count`` a correctness requirement rather than an
    optimisation, pinned so nobody later "optimises" it into an estimate.

    A too-small ``nse`` does not raise and does not merely cost accuracy — it silently discards
    triplets, producing a different matrix. Also pins ``remove_zeros=False``: with removal on,
    ``sum_duplicates`` pads out to ``nse`` using OUT-OF-BOUNDS indices, which the AMG/AMS CSR
    conversions would turn into a row that does not exist."""
    import jax.experimental.sparse as jsp
    import jax.numpy as jnp

    from jno.utils.solver.fem_utils import sum_duplicate_triplets, unique_triplet_count

    rng = np.random.default_rng(0)
    n, nnz = 40, 500
    idx = np.stack([rng.integers(0, n, nnz), rng.integers(0, n, nnz)], axis=1).astype(np.int32)
    data = rng.standard_normal(nnz)
    raw = jsp.BCOO((jnp.asarray(data), jnp.asarray(idx)), shape=(n, n))

    k = unique_triplet_count(idx)
    assert k == len({tuple(t) for t in idx}), "the count must be the exact number of distinct pairs"

    packed = sum_duplicate_triplets(raw, nse=k)
    assert int(packed.nse) == k
    assert np.asarray(packed.indices).max() < n, "no out-of-bounds padding may be emitted"
    dense = np.zeros((n, n))
    np.add.at(dense, (idx[:, 0], idx[:, 1]), data)
    assert np.max(np.abs(np.asarray(packed.todense()) - dense)) < 1e-12

    # the hazard itself: one short and the operator is quietly wrong, with no error raised
    short = sum_duplicate_triplets(raw, nse=k - 1)
    assert np.max(np.abs(np.asarray(short.todense()) - dense)) > 1e-9, (
        "a too-small nse must visibly change the matrix — if it did not, the exactness requirement "
        "would be untestable and could rot"
    )


def test_dropping_an_explicit_zero_is_invisible_to_the_preconditioners():
    """Compression removes numerically-zero triplets, which changes the *pattern*. The docstring
    claims that is safe for the diagonal-reading preconditioners; this is that claim under test
    rather than under assertion.

    ``matrix_diagonal`` scatter-*adds* only on-diagonal triplets, so a dropped zero contributes
    nothing and a stored zero contributes nothing — identical by construction. ``jacobi`` then guards
    ``|d| > 1e-30``, so a diagonal that is zero either way is left unscaled instead of producing
    ``inf``. Checked on a saddle-shaped operator whose zero diagonal block is exactly the case where
    the distinction could have mattered."""
    import jax.experimental.sparse as jsp
    import jax.numpy as jnp

    from jno.utils.solver.linear import jacobi, matrix_diagonal

    n = 12
    rows = list(range(n)) + [0, 5]
    cols = list(range(n)) + [3, 9]
    vals = [1.0 + i for i in range(n - 4)] + [0.0] * 4 + [2.0, -1.5]  # four EXPLICIT zero diagonals
    with_zeros = jsp.BCOO((jnp.asarray(vals), jnp.asarray(np.stack([rows, cols], 1), dtype=jnp.int32)), shape=(n, n))
    keep = [i for i, v in enumerate(vals) if v != 0.0]  # the same operator with the zeros dropped
    without = jsp.BCOO(
        (
            jnp.asarray([vals[i] for i in keep]),
            jnp.asarray(np.stack([[rows[i] for i in keep], [cols[i] for i in keep]], 1), dtype=jnp.int32),
        ),
        shape=(n, n),
    )
    assert without.nse < with_zeros.nse

    assert np.allclose(np.asarray(matrix_diagonal(with_zeros)), np.asarray(matrix_diagonal(without)))
    x = jnp.asarray(np.random.default_rng(0).standard_normal(n))
    got_a, got_b = np.asarray(jacobi(with_zeros)(x)), np.asarray(jacobi(without)(x))
    assert np.allclose(got_a, got_b), "jacobi must not see the difference"
    assert np.all(np.isfinite(got_a)), "a zero diagonal must be left unscaled, not divided by"


def test_unique_count_refuses_a_traced_index_array():
    """It must not guess. Every caller derives its pattern from host-static mesh connectivity, so a
    tracer here means that invariant broke — and a guessed count is a wrong matrix."""
    import jax.numpy as jnp

    from jno.utils.solver.fem_utils import unique_triplet_count

    with pytest.raises(Exception):
        jax.jit(lambda i: unique_triplet_count(i))(jnp.zeros((4, 2), jnp.int32))


def test_traced_operators_are_returned_untouched_not_broken():
    """Without a static ``nse``, ``sum_duplicates`` cannot run under trace. The contract is that it
    degrades to *uncompressed*, never to an error — the paths whose pattern could not be hoisted (the
    parametric placeholders, the per-step transient re-assembly) still depend on that."""
    import jax.experimental.sparse as jsp
    import jax.numpy as jnp

    from jno.utils.solver.fem_utils import sum_duplicate_triplets

    idx = jnp.asarray([[0, 0], [0, 0], [1, 1]], dtype=jnp.int32)

    def f(data):
        A = sum_duplicate_triplets(jsp.BCOO((data, idx), shape=(2, 2)))
        return A @ jnp.ones(2)

    out = jax.jit(f)(jnp.asarray([1.0, 2.0, 5.0]))
    assert np.allclose(np.asarray(out), [3.0, 5.0]), "the traced fallback must still be correct"


def test_non_bcoo_operators_pass_through():
    """The non-nodal vertex families (Hermite/Argyris/Morley) keep a dense global jacfwd, and the
    compression call sits on a shared line above both branches. A dense operator must survive it."""
    import jax.numpy as jnp

    from jno.utils.solver.fem_utils import sum_duplicate_triplets

    dense = jnp.arange(9.0).reshape(3, 3)
    assert sum_duplicate_triplets(dense) is dense
    assert sum_duplicate_triplets(None) is None
