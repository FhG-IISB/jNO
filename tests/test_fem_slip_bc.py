"""Normal ("slip" / no-penetration) conditions ``n·u - g`` and which fields may carry them.

``_normal_flux_spec`` recognises ``n·u - g`` on *any* element family, but the ``flux_bcs`` bucket it
routes to is forwarded only to ``assemble_fem_nonnodal`` — the H(div)/H(curl) edge-DOF path. A
nodal-Lagrange field written with that spelling was therefore claimed by the parser and then dropped
on the floor: the constraint vanished from the term list, the boundary was left unconstrained, and
the solve returned a plausible wrong answer with nothing to indicate a BC had gone missing.

It is now imposed **exactly** on a nodal field instead, by eliminating one velocity component per
constrained node (a master-slave prolongation, so the reduced solve lives on the constraint manifold).
These tests pin both halves of the contract:

* on a nodal Lagrange field the condition holds to round-off, with no penalty parameter;
* on RT / N1curl the same spelling still routes to ``flux_bcs`` exactly as before.

The oracle throughout is the constraint residual measured on the solution's own nodal values, never a
restatement of the parser's bookkeeping.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    """FEM assembly is float64, so these tests opt into x64 per-test (session default is x64-off)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _disk_vector_poisson(size=0.35):
    """A well-posed vector Poisson on a disk, plus the pieces needed to write ``n·u`` on its boundary."""
    d = jno.Shape.disk(0.0, 0.0, 1.0, size=size).domain()
    u, phi = d.fem_symbols(value_shape=(2,))
    _ = phi
    xi, yi, _ = d.variable("interior", split=True)
    cb = d.variable("boundary", normals=True, split=True)
    xb, yb, nx, ny = cb[0], cb[1], cb[-2], cb[-1]

    grad, inner = jno.np.grad, jno.np.inner
    vi = phi.bind(x=xi, y=yi)
    weak = inner(grad(u, [xi, yi]), grad(phi, [xi, yi]), n_contract=2) - (1.0 * vi.component(0) + 0.5 * vi.component(1))
    ub = u(xb, yb)
    return d, u, phi, weak, (xb, yb), (nx, ny), nx * ub[0] + ny * ub[1] - 0.0


def _solve_and_measure(terms, fem_kwargs=None):
    """Solve and return max |u·n| on the unit-circle wall, from the solution's own nodal values."""
    fem = jno.fem(terms)
    sol = np.asarray(fem.solve()).reshape(-1)
    pts = np.asarray(fem.field_points[0])
    r = np.linalg.norm(pts, axis=1)
    on = r > 1.0 - 1e-6
    U = sol.reshape(-1, 2)[on]
    N = pts[on] / r[on][:, None]
    return float(np.abs((U * N).sum(1)).max()), fem


def test_slip_condition_holds_to_machine_precision():
    """The point of the whole feature: `n·u = 0` on a nodal field is EXACT, not penalty-approximate.

    This spelling used to be silently discarded (the constraint vanished and the wall was left free).
    Now it is imposed by eliminating one component per constrained node, so the residual is at round-off
    for any mesh and any solution — nothing to tune.
    """
    _, _, _, weak, _, _, slip = _disk_vector_poisson()
    err, fem = _solve_and_measure([weak, slip])
    assert err < 1e-12, f"slip wall violated by {err:.3e}"
    assert any("slip" in c for c in fem.classification), fem.classification


def test_slip_beats_the_penalty_it_replaces_by_orders_of_magnitude():
    """Fair baseline (house rule 2): the SAME wall imposed weakly, at a penalty stiff enough to be a
    serious competitor. The penalty error is O(1/c) and needs tuning; the elimination has no such knob.
    """
    d, u, phi, weak, (xb, yb), (nx, ny), slip = _disk_vector_poisson()
    ub, vb = u(xb, yb), phi.bind(x=xb, y=yb)
    pen = 1.0e6 * (nx * ub[0] + ny * ub[1]) * (nx * vb.component(0) + ny * vb.component(1))

    exact, _ = _solve_and_measure([weak, slip])
    weak_err, _ = _solve_and_measure([weak, pen])
    assert exact < 1e-12, f"elimination should be at round-off, got {exact:.3e}"
    assert weak_err > 1e-9, f"penalty baseline unexpectedly exact ({weak_err:.3e}) — test is not comparing"
    assert exact < weak_err / 1e5, f"elimination {exact:.3e} vs penalty {weak_err:.3e}"


def test_the_eliminated_dof_count_is_one_per_constrained_node():
    """One scalar condition per node removes exactly one component — the reduced system is SMALLER,
    not augmented (a Lagrange-multiplier formulation would grow it)."""
    _, _, _, weak, _, _, slip = _disk_vector_poisson()
    fem = jno.fem([weak, slip])
    pts = np.asarray(fem.field_points[0])
    n_wall = int((np.linalg.norm(pts, axis=1) > 1.0 - 1e-6).sum())
    assert fem._periodic is not None, "the slip reduction was not attached"
    assert int(fem._periodic["n_red"]) == fem.dofs - n_wall


def test_an_inhomogeneous_normal_condition_is_refused():
    """`n·u = g` with g != 0 needs an affine offset the prolongation cannot carry — refuse, don't
    silently impose the homogeneous condition instead (house rule 1)."""
    _, u, _, weak, (xb, yb), (nx, ny), _ = _disk_vector_poisson()
    ub = u(xb, yb)
    with pytest.raises(NotImplementedError, match=r"[Ii]nhomogeneous"):
        jno.fem([weak, nx * ub[0] + ny * ub[1] - 2.5])


def test_ordinary_dirichlet_conditions_are_untouched():
    """The slip route keys on the *spelling*, not on the presence of normals in the domain."""
    _, u, _, weak, (xb, yb), _, _ = _disk_vector_poisson()
    fem = jno.fem([weak, u(xb, yb) - 0.0])
    assert "dirichlet@boundary" in fem.classification
    fem_comp = jno.fem([weak, u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0])
    assert any(c.startswith("dirichlet@boundary[") for c in fem_comp.classification)


# ---------------------------------------------------------------------------------------------
# The elimination itself: `build_slip_prolongation` turns `C·u_node = 0` into a prolongation, so the
# reduced solve lives on the constraint manifold and the condition holds exactly (not to a penalty
# tolerance). These test the algebra directly — the oracle is the constraint residual itself.
# ---------------------------------------------------------------------------------------------


def test_prolongation_satisfies_the_constraint_exactly():
    """`n·u = 0` must hold to machine precision for ANY reduced vector, not merely for the solution."""
    from jno.utils.solver.fem_utils import build_slip_prolongation

    rng = np.random.default_rng(0)
    d, K = 3, 5
    node_dofs = np.arange(K * d).reshape(K, d)
    n = rng.normal(size=(K, 1, d))
    n /= np.linalg.norm(n, axis=2, keepdims=True)

    pro = build_slip_prolongation(K * d, node_dofs, n)
    assert pro["n_red"] == K * d - K  # one component eliminated per condition
    for _ in range(5):
        u = np.asarray(pro["P"] @ jnp.asarray(rng.normal(size=pro["n_red"])))
        residual = max(abs(float(n[a, 0] @ u[node_dofs[a]])) for a in range(K))
        assert residual < 1e-14, f"constraint violated by {residual:.3e}"


def test_an_axis_aligned_normal_reduces_to_a_plain_component_pin():
    """`n = e_y` must eliminate exactly u_y — the new machinery contains the old case."""
    from jno.utils.solver.fem_utils import build_slip_prolongation

    rng = np.random.default_rng(1)
    pro = build_slip_prolongation(12, np.array([[3, 4, 5]]), np.array([[0.0, 1.0, 0.0]]))
    assert pro["n_red"] == 11
    u = np.asarray(pro["P"] @ jnp.asarray(rng.normal(size=11)))
    assert abs(float(u[4])) == 0.0  # exactly zero, not "small"


def test_two_conditions_on_one_node_leave_one_free_direction():
    """A node where two slip surfaces meet (the roll/side edge) keeps only the intersection tangent."""
    from jno.utils.solver.fem_utils import build_slip_prolongation

    rng = np.random.default_rng(2)
    C = np.array([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])  # (K=1, m=2, d=3)
    pro = build_slip_prolongation(12, np.array([[3, 4, 5]]), C)
    assert pro["n_red"] == 10  # two components eliminated
    u = np.asarray(pro["P"] @ jnp.asarray(rng.normal(size=10)))
    assert float(u[4]) == 0.0 and float(u[5]) == 0.0  # motion only along x, the shared tangent


@pytest.mark.parametrize(
    "coeffs, match",
    [
        (np.zeros((1, 3)), r"numerically zero"),
        (np.array([[[0.0, 1.0, 0.0], [1e-13, 1.0, 0.0]]]), r"numerically parallel"),
        (np.zeros((1, 4, 3)), r"over-constrained|numerically zero"),
    ],
)
def test_degenerate_constraints_raise(coeffs, match):
    """House rule 1: a direction that cannot be eliminated must fail loudly, not silently pick a pivot."""
    from jno.utils.solver.fem_utils import build_slip_prolongation

    with pytest.raises(ValueError, match=match):
        build_slip_prolongation(12, np.array([[3, 4, 5]]), coeffs)


def test_mixed_block_reduction_stays_sparse_and_exact():
    """A slip block beside an identity block (Taylor-Hood: velocity + pressure) must not densify.

    Before this path existed the mixed case fell through to the dense branch and materialised
    n_full x n_full — fatal at 3-D sizes. Oracle: the dense PᵀAP on a small system.
    """
    from jax.experimental import sparse as jsparse

    from jno.utils.solver.fem_utils import build_slip_prolongation, reduce_matrix_periodic

    rng = np.random.default_rng(3)
    nu, npr = 12, 5
    node_dofs = np.arange(nu).reshape(4, 3)
    n = rng.normal(size=(4, 1, 3))
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    pu = build_slip_prolongation(nu, node_dofs, n)
    Ip = jsparse.BCOO.fromdense(jnp.eye(npr))
    per = {
        "blocks": [
            {"P": pu["P"], "kept": pu["kept_nodes"], "vec": 1, "is_selection": False},
            {"P": Ip, "kept": np.arange(npr), "vec": 1, "is_selection": True},
        ],
        "off_full": np.array([0, nu, nu + npr]),
        "off_red": np.array([0, pu["n_red"], pu["n_red"] + npr]),
    }
    N = nu + npr
    A = rng.normal(size=(N, N))
    A = A + A.T
    red = reduce_matrix_periodic(per, jsparse.BCOO.fromdense(jnp.asarray(A)))
    assert hasattr(red, "indices"), "the mixed-block reduction densified"

    Pd = np.zeros((N, pu["n_red"] + npr))
    Pd[:nu, : pu["n_red"]] = np.asarray(pu["P"].todense())
    Pd[nu:, pu["n_red"] :] = np.eye(npr)
    np.testing.assert_allclose(np.asarray(red.todense()), Pd.T @ A @ Pd, atol=1e-12)


def test_rt_normal_flux_still_routes_to_the_edge_dof_path():
    """The other half of the contract: on H(div) the same spelling is the intended one and must work.

    Oracle (the house pattern from ``test_fem_nonnodal_dsl.py``): with a mass system and ``u·n = g``
    constant, every boundary edge DOF is pinned to ``-sign_topo * g * |edge|``. Here we only need the
    coarser statement that the pinning is *live* — the gate added for nodal fields must not have
    stolen this route — so we check the solve moves when ``g`` moves, which it cannot do if the
    constraint were dropped. The exact per-DOF values are already asserted in the nonnodal suite.
    """
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain()
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    xi, yi, _ = d.variable("interior", split=True)
    cb = d.variable("boundary", normals=True, split=True)
    xb, yb, nx, ny = cb[0], cb[1], cb[-2], cb[-1]
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    ub = u.bind(x=xb, y=yb)

    sols = []
    for g in (1.5, 3.0):
        fem = jno.fem([jno.np.inner(ui, vi), ub[0] * nx + ub[1] * ny - g])
        # routed to flux_bcs, so it must NOT appear as a Cartesian Dirichlet condition
        assert not any("dirichlet" in c for c in fem.classification), fem.classification
        A = fem.A
        A = np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)
        sols.append(np.linalg.solve(A, np.asarray(fem.b).reshape(-1)))

    # doubling g must double the pinned boundary DOFs; a dropped constraint would leave them identical
    assert not np.allclose(sols[0], sols[1]), "the u·n = g constraint was not applied on RT"
    np.testing.assert_allclose(sols[1], 2.0 * sols[0], rtol=1e-10, atol=1e-12)


def _curved_channel_top(zcut, nx=12, ny=3, nz=6, L=2.0, H=1.0, W=1.0, A=0.35):
    """A curved surface tagged on only ONE side of a plane it continues across.

    The top of a structured box is bent into ``y = h(x)`` — curved along ``x``, **constant along
    ``z``** — and then tagged only for ``z < zcut``. That is the geometry of a half-model cut on a
    symmetry plane: the surface itself does not stop at the cut, only the region does, so the facet
    patch around a node on the cut is one-sided while the exact normal there is unchanged.

    Returns the recovered normals, their positions, and the exact surface normal at each.
    """
    from jno._fem import _region_node_normals

    h = lambda x: H - 0.5 * A * (1.0 - np.cos(2.0 * np.pi * x / L))  # noqa: E731
    dh = lambda x: -A * (np.pi / L) * np.sin(2.0 * np.pi * x / L)  # noqa: E731

    d = jno.Shape.box(0.0, 0.0, 0.0, L, H, W, size=0.3).structured(n=(nx, ny, nz)).domain()
    P = np.asarray(d.mesh.points).copy()
    P[:, 1] *= h(P[:, 0]) / H  # bend the top; the lattice topology is untouched
    d.mesh.points = P
    d.tag("top", lambda x, y, z: (y > h(x) - 1e-9) & (z < zcut + 1e-9))

    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), order=2)
    ci, ct = d.variable("interior", split=True), d.variable("top", normals=True, split=True)
    X = list(ci[:3])
    eps, dot2 = lambda w: jno.np.symgrad(w, X), lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    ut = u.bind(x=ct[0], y=ct[1], z=ct[2])
    fem = jno.fem([dot2(eps(u), eps(v)), ct[-3] * ut[0] + ct[-2] * ut[1] + ct[-1] * ut[2] - 0.0])

    pts = np.asarray((getattr(d, "_fem_native_dof_points_all", None) or [fem.points])[0])
    got = _region_node_normals(d, pts, np.asarray(d._fem_native_assembly_cells), 2, "top")
    ids = np.array(sorted(got))
    N, Q = np.stack([got[i] for i in ids]), pts[ids]
    keep = np.linalg.norm(N, axis=1) > 0  # P2 midsides carry the flux; vertices carry a direction
    N, Q = N[keep] / np.linalg.norm(N[keep], axis=1, keepdims=True), Q[keep]
    s = dh(Q[:, 0])
    exact = np.stack([-s, np.ones_like(s), np.zeros_like(s)], 1)
    return N, Q, exact / np.linalg.norm(exact, axis=1, keepdims=True)


def test_a_truncated_region_does_not_tilt_the_normal_it_recovers():
    """Cutting a region on a plane the surface passes straight through must not move the normal.

    A node on the cut sees only the facets on one side of it. Weighting those by **area** is
    triangulation-dependent: on a lattice each quad splits along one diagonal, so a node touches one
    facet on its left and two on its right in the row above and the reverse below. In the interior
    the imbalance cancels between the two rows; on the cut only one row survives and the average
    tilts, systematically and in-plane, by a fixed fraction of the local slope. Weighting by the
    angle each facet subtends at the node is triangulation-independent (Thürmer & Wüthrich,
    *J. Graphics Tools* **3**(1), 1998, §3) and does cancel — 180°/180° in the interior, 90°/90° on
    the cut, at any aspect ratio.

    The oracle is exact and needs no reference solution: ``h`` does not depend on ``z``, so two nodes
    at the same ``x`` must be given the *same* normal whether or not one of them sits on the cut.
    Facet-geometry error is common to both and cancels in the comparison, which is what lets this
    resolve a bias four times smaller than the O(h) error it hides under.
    """
    zcut = 0.5
    N, Q, exact = _curved_channel_top(zcut)
    on_cut = Q[:, 2] > zcut - 1e-6
    assert on_cut.any() and (~on_cut).any(), "the tag did not truncate the surface"

    xr, yr = np.round(Q[:, 0], 9), np.round(Q[:, 1], 9)
    drift = []
    for i in np.where(on_cut)[0]:
        twin = np.where((~on_cut) & (xr == xr[i]) & (yr == yr[i]))[0]
        if len(twin):  # the nearest node of the same (x, y), away from the cut
            k = twin[np.argmin(np.abs(Q[twin, 2] - Q[i, 2]))]
            drift.append(np.degrees(np.arccos(np.clip(abs(float(N[i] @ N[k])), -1.0, 1.0))))
    drift = np.asarray(drift)
    assert len(drift) >= 10, f"only {len(drift)} cut nodes had an interior twin"

    # area weighting gives 0.63 deg mean / 2.68 deg max here, above the 0.15 deg the facet geometry
    # itself is worth; the bound is set between the two so it cannot pass on a wash.
    assert drift.mean() < 0.15, f"normals tilt on the cut: mean drift {drift.mean():.4f} deg"
    assert drift.max() < 0.50, f"normals tilt on the cut: max drift {drift.max():.4f} deg"

    # the surface is straight along z, so the recovered normal must have no z-component at all
    assert np.abs(N[:, 2]).max() < 1e-12, "a z-invariant surface produced an out-of-plane normal"

    # and the whole patch must still track the true surface to the facet error, cut or not
    ang = np.degrees(np.arccos(np.clip(np.abs((N * exact).sum(1)), -1.0, 1.0)))
    assert ang[on_cut].mean() < 1.35 * ang[~on_cut].mean(), (
        f"cut nodes are worse than interior ones: {ang[on_cut].mean():.4f} vs {ang[~on_cut].mean():.4f} deg"
    )
