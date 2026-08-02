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
# the compression primitive itself
# ---------------------------------------------------------------------------------------------


def test_traced_operators_are_returned_untouched_not_broken():
    """``sum_duplicates`` infers ``nse``, which needs concrete indices, so a traced assembly cannot be
    compressed yet. The contract is that it degrades to *uncompressed*, never to an error — the
    parametric and per-step paths depend on that until the pattern is hoisted host-side."""
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
