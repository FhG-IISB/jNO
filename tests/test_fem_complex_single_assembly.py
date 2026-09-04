"""A steady complex non-nodal form assembles ONCE, with complex element data.

The historic route split every term into ``.real``/``.imag`` legs and ran the FULL assembler twice
-- two symbolic passes, two data passes, two residual evaluations -- because the real assembler
would have silently cast the imaginary part away. The basis is real and only the COEFFICIENT is
complex, so one data pass with a real linearisation point gives the complex element blocks directly
(``jacfwd`` of a real->complex linear residual needs no holomorphic special case), and the legs the
downstream consumers read (``_complex_legs``) are derived from the one complex operator.

Measured motivation (275k-tet complex mixed N1E x Lagrange build): the second leg's assembly was
+1.3 GB of peak for information the first pass already touched.

The split is NOT gone: a parametric form (runtime parameters / trainable coordinates), a complex
literal in a BOUNDARY term (impedance / incident), and flux/rotation BCs keep the two-leg route
until each is verified complex-clean. The gate must fail CLOSED: anything unrecognised assembles
the old way, and a complex form whose single-pass operator comes back real raises rather than
returning a silently half-cast system.
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


def _terms(d, extra_coeff=None):
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    c = (1e-3 + 0.5j) if extra_coeff is None else extra_coeff
    return [
        inner(u.vector.curl(x, y, z), v.vector.curl(x, y, z))
        + c * inner(u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z))
        - inner(vec(1.0 + 0.0 * x, 0.0 * x, 0.0 * x), v.bind(x=x, y=y, z=z)),
        u.vector.cross(d.variable("boundary", normals=True)),
    ]


def _count_assemblies(monkeypatch):
    from jno.utils.solver import fem_nonnodal

    calls = []
    orig = fem_nonnodal.assemble_fem_nonnodal

    def _counted(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(fem_nonnodal, "assemble_fem_nonnodal", _counted)
    return calls


def _coo(A):
    return sp.coo_matrix(
        (np.asarray(A.data), (np.asarray(A.indices[:, 0]), np.asarray(A.indices[:, 1]))), shape=A.shape
    ).tocsr()


def test_complex_volume_form_assembles_once(monkeypatch):
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    calls = _count_assemblies(monkeypatch)
    fem = jno.fem(_terms(d))
    assert len(calls) == 1, f"the steady complex volume form ran {len(calls)} assemblies; one pass suffices"
    assert fem._complex_legs is not None, "the derived legs must still exist for their consumers"


def test_single_pass_matches_the_two_leg_reference(monkeypatch):
    """The derived legs must equal what the historic Re/Im split assembled -- operator AND load.
    The reference comes from the split itself, via the module escape hatch."""
    import jno._fem as F

    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    fem = jno.fem(_terms(d))
    (Ar, br), (Ai, bi) = fem._complex_legs

    monkeypatch.setattr(F, "_COMPLEX_SINGLE_ASSEMBLY", False)
    d2 = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    ref = jno.fem(_terms(d2))
    (Rr, rr), (Ri, ri) = ref._complex_legs

    # The split's legs must stay REAL dtype -- a complex-dtype leg doubles the data for a zero
    # imaginary part and lands in consumers that expect real legs.
    assert not np.iscomplexobj(np.asarray(Rr.data)) and not np.iscomplexobj(np.asarray(Ri.data))
    assert not np.iscomplexobj(np.asarray(Ar.data)) and not np.iscomplexobj(np.asarray(Ai.data))

    # The two eliminations scale their pinned unit diagonals differently (complex-magnitude vs
    # per-leg), so raw entries on pinned rows may differ while the SYSTEMS are equivalent. Assert
    # equivalence: identical solutions, and each system is consistent with the other's solution.
    import scipy.sparse.linalg as spla

    A1, b1 = (_coo(Ar) + 1j * _coo(Ai)).tocsc(), np.asarray(br) + 1j * np.asarray(bi)
    A2, b2 = (_coo(Rr) + 1j * _coo(Ri)).tocsc(), np.asarray(rr) + 1j * np.asarray(ri)
    x1, x2 = spla.spsolve(A1, b1), spla.spsolve(A2, b2)
    scale = max(np.abs(x2).max(), 1e-300)
    assert np.abs(x1 - x2).max() / scale < 1e-10, "the two assembly routes disagree on the solution"
    assert np.abs(A1 @ x2 - b1).max() / max(np.abs(b1).max(), 1e-300) < 1e-10


def test_the_solve_is_right(monkeypatch):
    import scipy.sparse.linalg as spla

    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    fem = jno.fem(_terms(d))
    (Ar, br), (Ai, bi) = fem._complex_legs
    ref = spla.spsolve((_coo(Ar) + 1j * _coo(Ai)).tocsc(), np.asarray(br) + 1j * np.asarray(bi))
    sol = np.asarray(jno.np.asarray(fem.solve()))
    assert np.allclose(sol.reshape(-1), ref.reshape(-1), rtol=1e-8, atol=1e-12)


def test_parametric_complex_keeps_the_leg_split(monkeypatch):
    """A runtime parameter rides the parametric legs (tested elsewhere); the gate must not claim it."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    calls = _count_assemblies(monkeypatch)
    eps = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="eps_r")
    fem = jno.fem(_terms(d, extra_coeff=(1e-3 + 0.5j) * (1.0 + eps)))
    assert len(calls) == 2, "a parametric complex form must keep the two-leg split"
    assert fem._complex_legs is not None


def test_a_half_cast_operator_is_refused(monkeypatch):
    """If the single pass ever returns a REAL operator for a complex form, something silently cast
    the imaginary part away -- the one failure this design must never convert into a wrong answer."""
    from jax.experimental import sparse as jsp

    from jno.utils.solver import fem_nonnodal

    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    orig = fem_nonnodal.assemble_fem_nonnodal

    def _cast_away(*a, **k):
        (A, b), mode, offs = orig(*a, **k)
        A = jsp.BCOO((A.data.real, A.indices), shape=A.shape)
        return (A, np.asarray(b).real), mode, offs

    monkeypatch.setattr(fem_nonnodal, "assemble_fem_nonnodal", _cast_away)
    with pytest.raises(RuntimeError, match="imaginary"):
        jno.fem(_terms(d))
