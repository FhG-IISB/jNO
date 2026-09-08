"""Vector (displacement) ties across a non-matching interface — `u(A) - u(B)` on a vector field.

A tie identifies the DOFs of two boundary faces by algebraic elimination: a prolongation `P` removes
the secondary's DOFs, and where the meshes do not match the rows come from the dual-mortar integral
(Bernardi/Maday/Patera 1994; Wohlmuth 2000). That machinery was never scalar-specific — the mortar rows
are node-pair weights and `prolongation_from_ties` already expands them by `kron(P_node, I_vec)`. Only
the *routing* refused a vector field, so this is a lifted restriction rather than new numerics.

The oracle is the classical **mortar patch test** — a linear displacement field is an exact solution of
elasticity (constant strain, zero body force), so prescribing it on the whole outer boundary must
reproduce it everywhere, including across the tie — but compared against the **scalar** tie rather than
against machine zero, because neither is exact here.

The reason, diagnosed separately (see `test_fem_tie_dirichlet_conflict.py`): it is NOT mortar accuracy.
The prolongation is linearly complete (weights sum to 1, `P @ X_kept == X_full` to 9e-16). What fails is
that a node which is both **tie-eliminated and Dirichlet-prescribed** loses its prescribed value — the
elimination wins, silently. Prescribed nodes off the interface are exact. That defect is pre-existing and
orthogonal to routing a vector field, which is why the comparison here is vector-against-scalar: a wrong
component expansion would be an O(1) error, not a matched one, so matching the scalar path is what pins
`kron(P_node, I_vec)`.

It also unblocks the reason both mechanisms exist: with a vector tie AND `u.gap`, one interface can be
bonded and another in contact **on the same field** — the ordinary two-body setup, which used to raise
at build time because the gap requires a vector field and the tie rejected one.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

n = jno.np


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _split_box(size=0.5, conforming=False, zmid=1.0, ztop=2.0):
    return (
        jno.shape.regions(
            lower=jno.shape.box(0, 0, 0, 1, 1, zmid),
            upper=jno.shape.box(0, 0, zmid, 1, 1, ztop),
            conforming=conforming,
        )
        .sized(size)
        .domain()
    )


def _sides(d):
    return sorted(t for t in d.built_mesh.cell_sets if "|" in t)


def _elastic(d, u, phi):
    co = d.variable("interior", split=True)
    X = [co[0], co[1], co[2]]
    ui, vi = u.bind(x=co[0], y=co[1], z=co[2]), phi.bind(x=co[0], y=co[1], z=co[2])
    return n.inner(n.symgrad(ui, X), n.symgrad(vi, X), 2)


def _tie_term(d, u):
    sec, main = _sides(d)
    sv, mv = d.variable(sec, split=True), d.variable(main, split=True)
    return u(sv[0], sv[1], sv[2]) - u(mv[0], mv[1], mv[2])


# the linear field the patch test imposes: u = A x + c, a general (non-symmetric) gradient
_A = np.array([[0.011, -0.004, 0.007], [0.003, 0.009, -0.006], [-0.005, 0.002, 0.013]])
_C = np.array([0.002, -0.001, 0.004])


def _linear_field(pts):
    return pts @ _A.T + _C


def _patch_test(*, conforming, size=0.5):
    """Prescribe the linear field on the whole OUTER boundary (never on the interface) and return the
    max nodal deviation from it, and the field's scale."""
    d = _split_box(size, conforming)
    u, phi = d.fem_symbols(value_shape=(3,))
    terms = [_elastic(d, u, phi)]
    if not conforming:
        terms.append(_tie_term(d, u))

    ob = d.variable(
        "outer",
        where=lambda x, y, z: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9) | (z < 1e-9) | (z > 2 - 1e-9),
        split=True,
    )
    for k in range(3):
        gk = lambda p, k=k: float(_A[k] @ np.asarray(p)[:3] + _C[k])
        terms.append(u(ob[0], ob[1], ob[2])[k] - (_A[k, 0] * ob[0] + _A[k, 1] * ob[1] + _A[k, 2] * ob[2] + _C[k]))
        del gk

    sol = np.asarray(jno.fem(terms).solve(linear=jno.solve.lu(backend="host"))).reshape(-1, 3)
    pts = np.asarray(d.built_mesh.points)
    return float(np.abs(sol - _linear_field(pts)).max()), float(np.abs(_linear_field(pts)).max())


def _scalar_patch_error(size=0.5):
    """The same patch test on a SCALAR field through the same mortar rows — the reference the vector
    path must match."""
    d = _split_box(size, False)
    u, phi = d.fem_symbols()
    co = d.variable("interior", split=True)
    ui, vi = u.bind(x=co[0], y=co[1], z=co[2]), phi.bind(x=co[0], y=co[1], z=co[2])
    sec, main = _sides(d)
    sv, mv = d.variable(sec, split=True), d.variable(main, split=True)
    ob = d.variable(
        "outer_s",
        where=lambda x, y, z: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9) | (z < 1e-9) | (z > 2 - 1e-9),
        split=True,
    )
    lin = lambda X: _A[0, 0] * X[0] + _A[0, 1] * X[1] + _A[0, 2] * X[2] + _C[0]  # noqa: E731
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y + ui.z * vi.z,
            u(sv[0], sv[1], sv[2]) - u(mv[0], mv[1], mv[2]),
            u(ob[0], ob[1], ob[2]) - lin(ob),
        ]
    )
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))
    pts = np.asarray(d.built_mesh.points)
    exact = _A[0, 0] * pts[:, 0] + _A[0, 1] * pts[:, 1] + _A[0, 2] * pts[:, 2] + _C[0]
    return float(np.abs(sol - exact).max())


# ----------------------------------------------------------------------------------------------
# The patch test
# ----------------------------------------------------------------------------------------------


def test_the_vector_tie_matches_the_scalar_ties_patch_accuracy():
    """The vector expansion adds nothing of its own.

    NOTE the oracle is the SCALAR mortar tie, not machine zero: neither reproduces a linear field exactly,
    because a node that is both tie-eliminated and Dirichlet-prescribed silently loses its prescribed
    value (pinned in `test_fem_tie_dirichlet_conflict.py`; the prolongation itself IS linearly complete).
    That is pre-existing and independent of this change, which is why the comparison is against the
    scalar path rather than against zero. What THIS test pins is that routing a vector field through the
    same rows — `kron(P_node, I_vec)` — is no worse: a wrong component expansion would be an O(1) error,
    not a matched one."""
    scalar_err = _scalar_patch_error()
    vec_err, _scale = _patch_test(conforming=False)
    assert scalar_err > 0.0
    assert vec_err < 5.0 * scalar_err, f"vector tie {vec_err:.3e} vs scalar tie {scalar_err:.3e}"


def test_the_conforming_tie_passes_the_same_patch_test():
    """The control: with matching meshes the tie is an exact node-to-node identification, so any failure
    above would be in the mortar rows rather than in the vector expansion."""
    err, scale = _patch_test(conforming=True)
    assert err < 1e-10 * max(scale, 1.0)


def test_the_vector_tie_tracks_the_scalar_one_on_a_finer_mesh_too():
    """One mesh could be a coincidence — the two happening to match, say. Refine and re-compare."""
    scalar_err = _scalar_patch_error(size=0.34)
    vec_err, _scale = _patch_test(conforming=False, size=0.34)
    assert vec_err < 5.0 * scalar_err, f"vector {vec_err:.3e} vs scalar {scalar_err:.3e}"


# ----------------------------------------------------------------------------------------------
# It behaves like the bonded body
# ----------------------------------------------------------------------------------------------


def test_a_tied_stack_shears_like_one_bonded_body():
    """Beyond linear completeness: under a shear the tied two-body stack must transmit load like the
    same geometry meshed as a single body. The residual difference is the two meshes."""

    def run(conforming):
        d = _split_box(0.5, conforming)
        u, phi = d.fem_symbols(value_shape=(3,))
        terms = [_elastic(d, u, phi)]
        if not conforming:
            terms.append(_tie_term(d, u))
        cl = d.variable("cb", where=lambda x, y, z: z < 1e-9, split=True)
        dr = d.variable("dt", where=lambda x, y, z: z > 2.0 - 1e-9, split=True)
        terms += [u(cl[0], cl[1], cl[2])[k] - 0.0 for k in range(3)]
        terms += [u(dr[0], dr[1], dr[2])[0] - 0.02]
        terms += [u(dr[0], dr[1], dr[2])[k] - 0.0 for k in (1, 2)]
        sol = np.asarray(jno.fem(terms).solve(linear=jno.solve.lu(backend="host"))).reshape(-1, 3)
        pts = np.asarray(d.built_mesh.points)
        return float(sol[pts[:, 2] < 1.0 - 1e-9, 0].mean())

    bonded, tied = run(True), run(False)
    assert bonded > 1e-4
    assert tied == pytest.approx(bonded, rel=0.06)


# ----------------------------------------------------------------------------------------------
# The capstone, and the scope that remains
# ----------------------------------------------------------------------------------------------


def test_one_interface_tied_and_another_in_contact_on_the_same_field():
    """The reason both mechanisms exist, and what a scalar-only tie made unwritable: `u.gap` REQUIRES a
    vector field while the tie REJECTED one, so 'bond this seam, contact that one' could not be stated at
    all — the term list raised at build time. Three stacked blocks: the lower seam tied, the upper one
    carrying a contact traction, on one displacement field.

    Scope this exposes, recorded rather than hidden: `u.gap` marks the form structurally NONLINEAR (the
    tag is opaque to the unknown-dependence walk, so it must), and a reduced nonlinear system returns a
    DEFERRED trace node instead of an array. So the combination builds and assembles, and reading a
    concrete field back from it needs the deferred path. The step this test pins is the one that moved:
    from "cannot be expressed" to "assembles".
    """
    d = (
        jno.shape.regions(
            a=jno.shape.box(0, 0, 0, 1, 1, 1),
            b=jno.shape.box(0, 0, 1, 1, 1, 2),
            c=jno.shape.box(0, 0, 2, 1, 1, 3),
            conforming=False,
        )
        .sized(0.5)
        .domain()
    )
    seams = sorted(t for t in d.built_mesh.cell_sets if "|" in t)
    tie_sec, tie_main = [t for t in seams if t.startswith("a|b")][:2]
    con_sec, con_main = [t for t in seams if t.startswith("b|c")][:2]

    u, phi = d.fem_symbols(value_shape=(3,))
    sv = d.variable(con_sec, split=True)
    vs = phi.bind(x=sv[0], y=sv[1], z=sv[2])
    nrm = d.variable(con_sec, normals=True)
    g = u.gap(con_sec, con_main, domain=d)
    tsv, tmv = d.variable(tie_sec, split=True), d.variable(tie_main, split=True)

    cl = d.variable("cb", where=lambda x, y, z: z < 1e-9, split=True)
    dr = d.variable("dt", where=lambda x, y, z: z > 3.0 - 1e-9, split=True)
    terms = [
        _elastic(d, u, phi),
        u(tsv[0], tsv[1], tsv[2]) - u(tmv[0], tmv[1], tmv[2]),  # seam 1: TIED
        n.maximum(0.0, -1.0e5 * g) * n.inner(nrm, vs, 1),  # seam 2: one-sided CONTACT
    ]
    terms += [u(cl[0], cl[1], cl[2])[k] - 0.0 for k in range(3)]
    terms += [u(dr[0], dr[1], dr[2])[k] - (0.0, 0.0, -0.01)[k] for k in range(3)]

    fem = jno.fem(terms)  # <- this is the line that used to raise
    assert fem._mode == "nonlinear", "the gap makes the form structurally nonlinear, by design"
    assert fem.dofs > 0
    # and the tie really did reduce the system rather than being ignored
    assert fem._periodic is not None and fem._periodic.get("n_red") < fem._periodic.get("n_full")


def test_a_transient_vector_tie_still_refuses_and_names_the_steady_case():
    """Scope preserved and stated: the transient route pre-builds its own reduction, which is still
    scalar. The message must point at what DOES work rather than repeating the old blanket refusal."""
    d = _split_box(0.5).domain_like() if False else _split_box(0.5)
    d = (
        jno.shape.regions(lower=jno.shape.box(0, 0, 0, 1, 1, 1), upper=jno.shape.box(0, 0, 1, 1, 1, 2), conforming=False)
        .sized(0.5)
        .domain(time=(0.0, 1.0, 4))
    )
    u, phi = d.fem_symbols(value_shape=(3,))
    co = d.variable("interior", split=True)
    ui, vi = u.bind(x=co[0], y=co[1], z=co[2], t=co[-1]), phi.bind(x=co[0], y=co[1], z=co[2], t=co[-1])
    X = [co[0], co[1], co[2]]
    sec, main = _sides(d)
    sv, mv = d.variable(sec, split=True), d.variable(main, split=True)
    with pytest.raises(NotImplementedError, match="TRANSIENT"):
        jno.fem(
            [
                n.inner(ui.t, vi, 1) + n.inner(n.symgrad(ui, X), n.symgrad(vi, X), 2),
                u(sv[0], sv[1], sv[2]) - u(mv[0], mv[1], mv[2]),
            ]
        )
