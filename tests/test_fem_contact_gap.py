"""``u.gap(secondary, main)`` — the signed contact gap symbol.

``g = g0 - n . (u_secondary - u_main . Phi)`` at the secondary face's quadrature points. It follows the
``domain.cell_size`` pattern: a placeholder ``Variable`` whose real per-quadrature-point value is packed
during assembly. What is covered here is the **symbol layer** — that a gap binds to two real boundary
regions, records the pairing for assembly, and refuses every way of getting it wrong.

The placeholder is deliberately *dropped* from the assembly context, so a gap that assembly has not
packed raises as an unresolved symbol rather than evaluating to zero — which would read as "everywhere
exactly in contact" and be believed.
"""

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    """x64: a penalty of 1e5 on the gap makes the system stiff enough that float32 cannot reach
    Newton's 1e-8 tolerance (measured: it stalls at a 5.7e-8 residual against a 1.3e-8 target)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _two_body_domain(size=0.4):
    """Two independently meshed blocks, so both sides of the interface are tagged separately."""
    return (
        jno.Shape.regions(
            lower=jno.Shape.box(0, 0, 0, 1, 1, 1),
            upper=jno.Shape.box(0, 0, 1, 1, 1, 2.5),
            conforming=False,
        )
        .sized(size)
        .domain()
    )


def _sides(d):
    return sorted(t for t in d.built_mesh.cell_sets if "|" in t)


def test_gap_binds_to_two_tagged_faces_and_records_the_pairing():
    d = _two_body_domain()
    u, _phi = d.fem_symbols(value_shape=(3,))
    secondary, main = _sides(d)
    g = u.gap(secondary, main, domain=d)

    assert f"gap_{secondary}" in d.context, "the placeholder must exist so the Variable constructs"
    assert d._contact_pairs[f"gap_{secondary}"][:2] == (secondary, main), "assembly needs the pairing"
    assert getattr(g, "tag", None) == f"gap_{secondary}"


def test_gap_requires_the_domain_explicitly():
    """A fem symbol carries no domain, and ``Placeholder`` turns attribute access into trace nodes —
    so a ``getattr(self, "_domain", None)`` fallback would return a *node*, not None, and quietly bind
    the Variable to nonsense. The argument is required and keyword-only to make that impossible."""
    d = _two_body_domain()
    u, _phi = d.fem_symbols(value_shape=(3,))
    secondary, main = _sides(d)
    with pytest.raises(TypeError):
        u.gap(secondary, main)  # positional/omitted -> keyword-only error
    with pytest.raises(TypeError, match="must be a jno domain"):
        u.gap(secondary, main, domain="not a domain")


def test_gap_refuses_a_tag_that_is_not_a_boundary_region():
    d = _two_body_domain()
    u, _phi = d.fem_symbols(value_shape=(3,))
    secondary, main = _sides(d)
    with pytest.raises(ValueError, match="not a boundary region"):
        u.gap("does_not_exist", main, domain=d)
    with pytest.raises(ValueError, match="not a boundary region"):
        u.gap(secondary, "does_not_exist", domain=d)


def test_gap_refuses_a_face_against_itself():
    d = _two_body_domain()
    u, _phi = d.fem_symbols(value_shape=(3,))
    secondary, _main = _sides(d)
    with pytest.raises(ValueError, match="must be different regions"):
        u.gap(secondary, secondary, domain=d)


def test_a_face_carries_at_most_one_gap():
    """Re-pairing the same secondary face against a different main would silently overwrite the first
    pairing, so the second contact would quietly use the wrong main surface."""
    d = _two_body_domain()
    u, _phi = d.fem_symbols(value_shape=(3,))
    secondary, main = _sides(d)
    u.gap(secondary, main, domain=d)
    u.gap(secondary, main, domain=d)  # idempotent: the same pairing again is fine
    with pytest.raises(ValueError, match="already the secondary face"):
        u.gap(secondary, "top", domain=d)


def test_a_normal_gap_needs_a_vector_field():
    """`n . (u_s - u_m)` is a vector contraction. With a SCALAR field `jnp.einsum("d,qd->q", n, jump)`
    silently broadcasts the size-1 component axis and computes `sum(n) * jump` — right only when the
    normal happens to sum to 1, wrong on any tilted interface, and never an error. Refuse instead."""
    d = _two_body_domain()
    secondary, main = _sides(d)
    scalar, _ = d.fem_symbols()
    with pytest.raises(ValueError, match="needs a vector field"):
        scalar.gap(secondary, main, domain=d)
    vec, _ = d.fem_symbols(value_shape=(3,))
    assert vec.gap(secondary, main, domain=d) is not None


def _vector_poisson(c=None, size=0.4, traction_sign=+1):
    """Vector Poisson on the two-body bar; ``c`` adds a penalty on the interface gap. Returns the
    largest jump in u_z across coincident interface node pairs, and the peak displacement.

    ``traction_sign=-1`` writes the penalty with the wrong sign on purpose; it exists so a test can
    show the correct one is not arbitrary."""
    d = _two_body_domain(size)
    u, v = d.fem_symbols(value_shape=(3,))
    secondary, main = _sides(d)
    ci = d.variable("interior", split=True)
    bb = d.variable("boundary", split=True)
    vi = v.bind(x=ci[0], y=ci[1], z=ci[2])
    gu = jno.np.grad(u, [ci[0], ci[1], ci[2]])
    gv = jno.np.grad(v, [ci[0], ci[1], ci[2]])
    terms = [jno.np.inner(gu, gv, n_contract=2) - 1.0 * vi[2]]
    if c is not None:
        sb = d.variable(secondary, split=True)
        n = d.variable(secondary, normals=True)
        g = u.gap(secondary, main, domain=d)
        p = -c * g  # g < 0 penetrating -> p > 0 pressure
        terms.append(traction_sign * p * jno.np.inner(n, v.bind(x=sb[0], y=sb[1], z=sb[2]), n_contract=1))
    terms += [u(bb[0], bb[1], bb[2])[i] - 0.0 for i in range(3)]

    sol = np.asarray(jno.fem(terms, element_type="TET4").solve()).reshape(-1, 3)
    pts = np.asarray(d.built_mesh.points)
    lo = np.asarray(d.tag_indices[secondary]).reshape(-1)
    up = np.asarray(d.tag_indices[main]).reshape(-1)
    key = {tuple(np.round(pts[i, :2], 9)): i for i in lo}
    pairs = [(key[k], j) for j in up if (k := tuple(np.round(pts[j, :2], 9))) in key]
    assert len(pairs) > 5
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return float(np.abs(sol[a, 2] - sol[b, 2]).max()), float(np.abs(sol).max())


def test_a_penalty_on_the_gap_closes_the_interface_like_one_over_c():
    """The end-to-end gate, and the one that proves the gap is *live*: penalising it must drive the
    interface jump to zero like 1/c. That can only happen if the gap measures the real jump AND the
    tangent carries the main-side coupling — the matrix-free path picks the latter up through
    `jax.linearize` of the residual; the assembled path builds the explicit nonlocal blocks
    (gated by `test_assembled_gap_tangent_matches_the_matrix_free_jvp`)."""
    free, _ = _vector_poisson(None)
    jumps = {c: _vector_poisson(c)[0] for c in (1e2, 1e3, 1e4, 1e5)}

    assert jumps[1e2] < 0.5 * free, "a penalty must reduce the jump at all"
    for lo, hi in ((1e2, 1e3), (1e3, 1e4), (1e4, 1e5)):
        ratio = jumps[lo] / jumps[hi]
        assert 3.0 < ratio < 30.0, f"expected ~10x per decade of c, got {ratio:.1f} ({lo}->{hi})"
    assert jumps[1e5] < 1e-3 * free


def test_assembled_gap_tangent_matches_the_matrix_free_jvp():
    """The assembled tangent now carries the gap's NONLOCAL blocks -- (s,m) from jacfwd w.r.t. the
    gathered main values chained through the frozen mortar weights, and the reaction rows' (m,s) and
    (m,m). The matrix-free JVP (jax.linearize of the residual) is exact by construction, so the two
    must agree on random probe vectors, in both the ACTIVE (pressed) and INACTIVE (separated)
    branches of a one-sided max(0,.) traction."""
    import jax as _jax
    import jax.numpy as jnp

    d = _two_body_domain()
    u, v = d.fem_symbols(value_shape=(3,))
    secondary, main = _sides(d)
    ci = d.variable("interior", split=True)
    sb = d.variable(secondary, split=True)
    n = d.variable(secondary, normals=True)
    gu = jno.np.grad(u, [ci[0], ci[1], ci[2]])
    gv = jno.np.grad(v, [ci[0], ci[1], ci[2]])
    g = u.gap(secondary, main, domain=d)
    terms = [
        jno.np.inner(gu, gv, n_contract=2) - 1.0 * v.bind(x=ci[0], y=ci[1], z=ci[2])[2],
        jno.np.maximum(0.0, -1e3 * g) * jno.np.inner(n, v.bind(x=sb[0], y=sb[1], z=sb[2]), n_contract=1),
    ]
    bb = d.variable("boundary", split=True)
    terms += [u(bb[0], bb[1], bb[2])[i] - 0.0 for i in range(3)]
    fem = jno.fem(terms, element_type="TET4")
    op = fem.operator
    ndofs = int(fem.dofs)
    key = _jax.random.PRNGKey(0)

    for state_scale, label in ((-1e-3, "active (pressed)"), (+1e-3, "inactive (separated)")):
        # a z-displacement state that closes / opens the gap uniformly
        u0 = jnp.zeros(ndofs).reshape(-1, 3).at[:, 2].set(state_scale).reshape(-1)
        r = lambda w: jnp.asarray(op.residual(w)).reshape(-1)  # noqa: E731
        J = op.jacobian(u0)
        _r0, jvp = _jax.linearize(r, u0)
        for i in range(3):
            probe = _jax.random.normal(_jax.random.fold_in(key, i), (ndofs,))
            a = np.asarray(J @ probe)
            b = np.asarray(jvp(probe))
            scale = max(np.abs(b).max(), 1e-8)
            assert np.abs(a - b).max() < 5e-4 * scale, (
                f"{label}: assembled J and matrix-free JVP disagree "
                f"(max diff {np.abs(a - b).max():.3e} vs scale {scale:.3e})"
            )


def test_direct_newton_one_sided_matches_matrix_free():
    """`newton(direct=True)` + host LU over a one-sided contact -- the path that refused before --
    must land on the matrix-free answer."""
    inner, sym, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
    d = _stacked_blocks()
    sides = sorted(t for t in d.built_mesh.cell_sets if "|" in t)
    secondary = next(t for t in sides if t.endswith(".cap"))
    main = next(t for t in sides if t.endswith(".base"))
    u, phi = d.fem_symbols(value_shape=(2,))
    X = d.variable("interior", split=True)[:2]
    eu, ep = sym(u, list(X)), sym(phi, list(X))
    sv = d.variable(secondary, split=True)
    n = d.variable(secondary, normals=True)
    g = u.gap(secondary, main, domain=d)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    terms = [
        LAM_ * trace(eu) * trace(ep) + 2 * MU_ * inner(eu, ep, n_contract=2),
        jno.np.maximum(0.0, -1e4 * g) * inner(n, phi.bind(x=sv[0], y=sv[1]), n_contract=1),
        u(xb, yb)[0] - 0.0,
        u(xb, yb)[1] - 0.0,
        u(xt, yt)[0] - 0.0,
        u(xt, yt)[1] - (-0.02),
    ]
    mf = np.asarray(jno.fem(terms).solve(nonlinear=jno.solve.newton(line_search=True, rtol=1e-6, atol=1e-6))).reshape(-1, 2)
    dr = np.asarray(
        jno.fem(terms).solve(
            nonlinear=jno.solve.newton(direct=True, line_search=True, rtol=1e-6, atol=1e-6),
            linear=jno.solve.lu(backend="host"),
        )
    ).reshape(-1, 2)
    assert np.abs(mf - dr).max() < 5e-5, f"direct vs matrix-free one-sided contact: {np.abs(mf - dr).max():.2e}"


# ---------------------------------------------------------------------------
# The reaction on the main body# ---------------------------------------------------------------------------
# The reaction on the main body -- Newton's third law across the interface.
#
# The interface traction is written on the secondary face only, so the main's share has to be added by
# the assembler (the same integrand tested against the mortar-projected main trace, negated). Before
# that existed the main body was loaded by NOTHING: measured max|u| = 0.000e+00 over the whole base,
# i.e. contact against a rigid obstacle rather than between two deformable bodies.
# See ``plans/contact-main-reaction.md``.
# ---------------------------------------------------------------------------

E_, NU_ = 1000.0, 0.3
LAM_, MU_ = E_ * NU_ / (1 - NU_**2), E_ / (2 * (1 + NU_))


def _stacked_blocks(cap_size=0.09, base_size=0.22):
    """Two independently meshed unit squares stacked at y = 1, so the interface is non-matching."""
    return jno.Shape.regions(
        base=jno.Shape.rect(0, 0, 1, 1, size=base_size),
        cap=jno.Shape.rect(0, 1, 1, 2, size=cap_size),
        conforming=False,
    ).domain()


def _pressed_stack(c, *, glue=True, squeeze=-0.02):
    """Elastic stack pressed by a flat platen, the two bodies joined by a penalty on the gap.

    ``glue`` uses the two-sided penalty ``c*g`` -- the bonded interface, which is smooth and therefore
    the clean place to test the reaction (one-sided contact adds a ``max(0, .)`` kink that is a solver
    question, not a force-balance one). Returns ``(domain, u, secondary_ids, main_ids)``.
    """
    inner, sym, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
    d = _stacked_blocks()
    sides = sorted(t for t in d.built_mesh.cell_sets if "|" in t)
    secondary = next(t for t in sides if t.endswith(".cap"))  # the finer body must be the secondary
    main = next(t for t in sides if t.endswith(".base"))

    u, phi = d.fem_symbols(value_shape=(2,))
    X = d.variable("interior", split=True)[:2]
    eu, ep = sym(u, list(X)), sym(phi, list(X))
    terms = [LAM_ * trace(eu) * trace(ep) + 2 * MU_ * inner(eu, ep, n_contract=2)]

    sv = d.variable(secondary, split=True)
    n = d.variable(secondary, normals=True)
    g = u.gap(secondary, main, domain=d)
    p = (-c * g) if glue else jno.np.maximum(0.0, -c * g)  # g<0 penetrating -> p>0 pressure
    # `+p * inner(n, phi)`, not `-p * ...`: since dg/du_s = -n, this is the sign that adds a
    # POSITIVE-definite +c (n.du)(n.phi) to the tangent. The opposite sign is anti-stabilising -- see
    # `test_the_traction_sign_is_the_stabilising_one`, which is the gate that discriminates.
    terms.append(p * inner(n, phi.bind(x=sv[0], y=sv[1]), n_contract=1))

    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    terms += [u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0, u(xt, yt)[0] - 0.0, u(xt, yt)[1] - squeeze]
    nl = None if glue else jno.solve.newton(line_search=True)  # the max() kink needs globalizing
    sol = np.asarray(jno.fem(terms).solve(**({} if nl is None else {"nonlinear": nl}))).reshape(-1, 2)
    return d, sol, np.asarray(d.tag_indices[secondary]).reshape(-1), np.asarray(d.tag_indices[main]).reshape(-1)


def test_main_body_is_loaded_at_all():
    """The regression that names the defect: the main body used to be identically zero."""
    d, sol, _secondary, _main = _pressed_stack(1e5)
    base = np.asarray(d.built_mesh.points)[:, 1] < 0.999
    assert np.abs(sol[base]).max() > 1e-4, (
        "the main body carries no load: the interface traction is applied to the secondary only, "
        f"so contact is against a rigid obstacle (max|u| over the base = {np.abs(sol[base]).max():.3e})"
    )


@pytest.mark.parametrize("c, tol", [(1e4, 2e-3), (1e5, 1e-3), (1e6, 2e-4)])
def test_two_bonded_bodies_reproduce_the_single_bar(c, tol):
    """The oracle. A uniform bar of height 2 squeezed by 0.02 has ``uy(y=1) = -0.01`` exactly.

    Two bonded bodies must reproduce it as the penalty stiffens, and can only do so if each carries
    its share of the load -- with the reaction missing the main does not move and the secondary takes
    the whole 0.02.
    """
    _d, sol, secondary, main = _pressed_stack(c)
    for side, ids in (("secondary", secondary), ("main", main)):
        assert abs(sol[ids, 1].mean() - (-0.01)) < tol, f"{side} interface uy = {sol[ids, 1].mean():+.6f}, want -0.01"


def test_penalty_convergence_is_monotone():
    """Stiffening the interface must move the answer toward the bonded limit, not just near it."""
    errs = []
    for c in (1e4, 1e5, 1e6):
        _d, sol, secondary, _m = _pressed_stack(c)
        errs.append(abs(sol[secondary, 1].mean() - (-0.01)))
    assert errs[0] > errs[1] > errs[2], f"not converging to the bonded limit: {errs}"


# The bonded oracle above gates that a main reaction EXISTS -- ablating `_contact_reaction` leaves the
# main identically zero -- but measurement shows it does NOT gate the traction's sign: both signs
# converge to -0.01 (err 4.3e-06 vs 1.4e-05 at c=1e6), because a dominant two-sided penalty drives
# g -> 0 either way. The sign gate is the weakly-penalised regime below.


def test_the_traction_sign_is_the_stabilising_one():
    """`+p * inner(n, phi)` with `p = max(0, -c*g)`, not `-p * ...`.

    Since `dg/du_s = -n`, only one of the two adds a positive-definite `+c (n.du)(n.phi)` to the
    tangent; the other is anti-stabilising. At large `c` the penalty dominates and both enforce the
    constraint, which is why this has to be measured where the penalty is a perturbation: at `c=1e2`
    the correct sign reduces the interface jump well below the unpenalised one, and the wrong sign
    makes it *worse* than not penalising at all.
    """
    free, _ = _vector_poisson(None)
    good, _ = _vector_poisson(1e2, traction_sign=+1)
    bad, _ = _vector_poisson(1e2, traction_sign=-1)
    assert good < 0.5 * free, f"the documented sign did not close the interface: {good:.3e} vs free {free:.3e}"
    assert bad > free, (
        f"the flipped sign was expected to be anti-stabilising but gave {bad:.3e} <= free {free:.3e} -- "
        "if this fires, the sign is no longer pinned by this test and the convention needs re-deriving"
    )


def test_a_surface_load_on_one_side_of_an_interface_leaves_the_other_body_alone():
    """A traction on one side of a NON-CONFORMING interface must load that body only.

    The regression for the tag-resolution defect: interface tags used to be resolved through a
    *coordinate* predicate, and the two sides of a non-conforming interface sit at identical
    coordinates, so the tag returned the union of both. A plain Neumann load written on the cap's
    interface face was therefore applied to the base's facets too, and the base moved by 4.2e-03 with
    no coupling in the form at all. The two bodies here are joined by nothing, so the base must be
    *exactly* zero -- its block of the system has a zero right-hand side.
    """
    inner, sym, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
    d = _stacked_blocks()
    sides = sorted(t for t in d.built_mesh.cell_sets if "|" in t)
    secondary = next(t for t in sides if t.endswith(".cap"))

    u, phi = d.fem_symbols(value_shape=(2,))
    X = d.variable("interior", split=True)[:2]
    eu, ep = sym(u, list(X)), sym(phi, list(X))
    sv = d.variable(secondary, split=True)
    n = d.variable(secondary, normals=True)
    terms = [
        LAM_ * trace(eu) * trace(ep) + 2 * MU_ * inner(eu, ep, n_contract=2),
        5.0 * inner(n, phi.bind(x=sv[0], y=sv[1]), n_contract=1),  # a plain Neumann load, no gap
    ]
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    terms += [u(xb, yb)[i] - 0.0 for i in range(2)] + [u(xt, yt)[i] - 0.0 for i in range(2)]
    sol = np.asarray(jno.fem(terms).solve()).reshape(-1, 2)

    base = np.asarray(d.built_mesh.points)[:, 1] < 0.999
    assert np.abs(sol[base]).max() == 0.0, (
        f"a load on the cap's interface face moved the base by {np.abs(sol[base]).max():.3e}. The two "
        "sides of the interface are not being told apart -- the tag is resolving to both bodies' facets."
    )
    assert np.abs(sol[~base]).max() > 1e-6, "the load did not reach the cap either; the test is vacuous"


def _tagged_sides(owner_a="cap", owner_b="base"):
    """The documented alternative to the mesh's own interface names: one coordinate predicate over the
    interface plane, disambiguated by ``region=`` (see ``domain.tag``). jax.numpy, not ``jno.np``: a tag
    predicate is evaluated under ``jax.vmap`` over the mesh points, so it must be an array expression."""
    d = _stacked_blocks()
    at_interface = lambda x, y: jax.numpy.abs(y - 1.0) < 1e-9  # noqa: E731
    d.tag("side_a", at_interface, region=owner_a)
    d.tag("side_b", at_interface, region=owner_b)
    return d


def test_region_tagged_sides_of_an_interface_are_told_apart():
    """``d.tag(pred, region=...)`` is the documented way to name ONE side of a coincident interface, and
    it has to reach the surface-term face list too -- not just Dirichlet nodes. Same oracle as the
    mesh-tag case: a load on the cap's side must leave the base exactly zero."""
    inner, sym, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
    d = _tagged_sides()
    u, phi = d.fem_symbols(value_shape=(2,))
    X = d.variable("interior", split=True)[:2]
    eu, ep = sym(u, list(X)), sym(phi, list(X))
    sa = d.variable("side_a", split=True)
    n = d.variable("side_a", normals=True)
    terms = [
        LAM_ * trace(eu) * trace(ep) + 2 * MU_ * inner(eu, ep, n_contract=2),
        5.0 * inner(n, phi.bind(x=sa[0], y=sa[1]), n_contract=1),
    ]
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    terms += [u(xb, yb)[i] - 0.0 for i in range(2)] + [u(xt, yt)[i] - 0.0 for i in range(2)]
    sol = np.asarray(jno.fem(terms).solve()).reshape(-1, 2)

    base = np.asarray(d.built_mesh.points)[:, 1] < 0.999
    assert np.abs(sol[base]).max() == 0.0, (
        f"a load on side_a (region='cap') moved the base by {np.abs(sol[base]).max():.3e} -- the "
        "`region=` owner is not reaching the surface-term face selection."
    )
    assert np.abs(sol[~base]).max() > 1e-6, "the load did not reach the cap either; the test is vacuous"


def test_a_gap_whose_two_sides_are_the_same_body_is_refused():
    """Both sides named on the same body is not two surfaces. Left alone the gap would project that
    face onto itself and read ``g0 == 0`` everywhere -- indistinguishable from a perfectly tied
    interface, and believed. It has to raise instead."""
    d = _tagged_sides(owner_a="cap", owner_b="cap")
    u, phi = d.fem_symbols(value_shape=(2,))
    X = d.variable("interior", split=True)[:2]
    sa = d.variable("side_a", split=True)
    n = d.variable("side_a", normals=True)
    g = u.gap("side_a", "side_b", domain=d)
    terms = [
        jno.np.inner(jno.np.symgrad(u, list(X)), jno.np.symgrad(phi, list(X)), n_contract=2),
        -1e3 * g * jno.np.inner(n, phi.bind(x=sa[0], y=sa[1]), n_contract=1),
    ]
    xb, yb, _ = d.variable("bottom", split=True)
    terms += [u(xb, yb)[i] - 0.0 for i in range(2)]
    with pytest.raises(ValueError, match="share boundary facets"):
        jno.fem(terms).solve()


# NOT YET GUARDED: that an external (non-gap) surface term on the secondary face is *not* mirrored onto the
# main when the SAME form also carries a gap traction. ``_gaps_in`` filters on whether the expression
# references the gap variable; the test above covers the no-gap case, but a form carrying both on one
# face is still unguarded. See ``plans/contact-main-reaction.md``.


def test_the_documented_contact_sign_engages_instead_of_finding_the_free_root():
    """``g < 0`` is penetrating, so ``max(0, -c*g)`` must build pressure when the bodies are pressed.

    This is the sign pin. With the convention inverted the traction is *never* active, and a cap that
    passes straight through the base is an EXACT root of the residual -- Newton converges to it and
    reports success. Measured before the fix: base ``max|u| = 0``, cap bottom at the full imposed
    -0.02. Both assertions below are satisfied only by the correct orientation.
    """
    d, sol, secondary, _main = _pressed_stack(1e2, glue=False)
    base = np.asarray(d.built_mesh.points)[:, 1] < 0.999
    assert np.abs(sol[base]).max() > 1e-4, (
        f"the contact traction never activated: base max|u| = {np.abs(sol[base]).max():.3e}. "
        "The gap sign is inverted -- penetration is reading as separation."
    )
    # NOT asserted: where the cap ends up. One-sided contact does not converge at these tolerances
    # (the max() kink chatters -- see item 2 in plans/contact-main-reaction.md), so the position is
    # not yet trustworthy enough to gate on. That the base is loaded at all is the sign question, and
    # it is the half that inverts.


# ---------------------------------------------------------------------------
# ONE-SIDED (separating, Signorini) contact -- the max(0,.) case, measured.
#
# The old note here deferred one-sided contact as "a solver question: the max() kink chatters". The
# measurement that closed it: the chatter was a FLOAT32 RESIDUAL FLOOR (~1.6e-5 on the pressed
# stack, line search or not), not active-set cycling -- under x64 the same iteration converges to
# 3.6e-10 at rtol=1e-8. `jax.linearize` through `max` selects the active branch, which IS the
# semismooth Jacobian (the same argument `u.bounds` documents), so the solver was never the problem;
# the tolerance expectation was. These tests pin the physics at honest float32 tolerances and the
# x64 discriminator that proves the iteration superlinear.
# ---------------------------------------------------------------------------


def _one_sided(c, squeeze, *, rtol=1e-6):
    """The pressed stack with the ONE-SIDED penalty `max(0, -c*g)`; squeeze>0 lifts the platen."""
    inner, sym, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
    d = _stacked_blocks()
    sides = sorted(t for t in d.built_mesh.cell_sets if "|" in t)
    secondary = next(t for t in sides if t.endswith(".cap"))
    main = next(t for t in sides if t.endswith(".base"))
    u, phi = d.fem_symbols(value_shape=(2,))
    X = d.variable("interior", split=True)[:2]
    eu, ep = sym(u, list(X)), sym(phi, list(X))
    terms = [LAM_ * trace(eu) * trace(ep) + 2 * MU_ * inner(eu, ep, n_contract=2)]
    sv = d.variable(secondary, split=True)
    n = d.variable(secondary, normals=True)
    g = u.gap(secondary, main, domain=d)
    terms.append(jno.np.maximum(0.0, -c * g) * inner(n, phi.bind(x=sv[0], y=sv[1]), n_contract=1))
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    terms += [u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0, u(xt, yt)[0] - 0.0, u(xt, yt)[1] - squeeze]
    fem = jno.fem(terms)
    sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(line_search=True, rtol=rtol, atol=rtol))).reshape(-1, 2)
    return d, sol, np.asarray(d.tag_indices[secondary]).reshape(-1), np.asarray(d.tag_indices[main]).reshape(-1)


def test_one_sided_press_converges_to_the_bonded_oracle_like_one_over_c():
    """Under compression the closed one-sided contact must agree with the bonded interface: the
    uniform-bar oracle has uy(y=1) = -0.01, approached as the penalty stiffens."""
    errs = {}
    for c in (1e3, 1e4, 1e5):
        _d, sol, _s, main = _one_sided(c, -0.02)
        errs[c] = abs(sol[main, 1].mean() - (-0.01))
    assert errs[1e3] > errs[1e4] > errs[1e5], f"not converging to the bonded limit: {errs}"
    assert errs[1e5] < 3e-4, f"closed one-sided contact should sit near the bonded answer: {errs[1e5]:.2e}"


def test_one_sided_release_separates_without_adhesion():
    """THE defining physics: lift the platen and the traction must be exactly zero -- the main body
    undisturbed, the cap carried up rigidly, no tension transmitted across the open gap."""
    d, sol, secondary, main = _one_sided(1e4, +0.02)
    base = np.asarray(d.built_mesh.points)[:, 1] < 0.999
    assert np.abs(sol[base]).max() < 1e-9, (
        f"the open gap transmitted load: base max|u| = {np.abs(sol[base]).max():.3e} -- adhesion where there must be none"
    )
    assert sol[secondary, 1].mean() > 0.01, "the cap must move up with the platen once free"


def test_one_sided_zero_load_is_the_unconstrained_solve():
    """Extreme: no squeeze at all -- the contact term must contribute nothing anywhere."""
    _d, sol, _s, _m = _one_sided(1e4, 0.0)
    assert np.abs(sol).max() < 1e-9


def test_one_sided_complementarity_via_the_interface_jump():
    """p.g = 0, observed through displacements: pressed, the interface jump is O(1/c) (contact
    active, gap ~ 0); released, the jump is the full opening with zero transmitted force (gap open,
    traction 0). The two regimes may never mix."""
    _d, press, s_ids, m_ids = _one_sided(1e5, -0.02)
    jump_pressed = abs(press[s_ids, 1].mean() - press[m_ids, 1].mean())
    assert jump_pressed < 5e-4, f"pressed: the gap must be ~closed, jump {jump_pressed:.2e}"
    d, rel, s_ids, m_ids = _one_sided(1e5, +0.02)
    jump_open = rel[s_ids, 1].mean() - rel[m_ids, 1].mean()
    assert jump_open > 0.015, f"released: the gap must be OPEN by ~the lift, got {jump_open:.2e}"


def test_one_sided_tight_tolerance_is_a_float32_floor_not_a_solver_stall():
    """The discriminator that re-classified the recorded 'max() kink stalls Newton': under x64 the
    SAME iteration meets rtol=1e-8 (measured residual 3.6e-10) -- superlinear semismooth Newton,
    exactly as the `u.bounds` machinery argues. The float32 failure at 1e-8 is a precision floor."""
    import jax as _jax

    prev = _jax.config.jax_enable_x64
    _jax.config.update("jax_enable_x64", True)
    try:
        _d, sol, _s, main = _one_sided(1e4, -0.02, rtol=1e-8)
        assert abs(sol[main, 1].mean() - (-0.01)) < 1e-3
    finally:
        _jax.config.update("jax_enable_x64", prev)


def test_gradient_through_closed_one_sided_contact_fd_checks():
    """Differentiability through the ACTIVE contact: the platen displacement is a runtime Dirichlet
    PARAMETER, and d(mean base uy)/d(squeeze) flows through custom_root across the max() branch
    selection. FD-checked; away from onset the subgradient is the gradient."""
    import jax as _jax

    inner, sym, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
    d = _stacked_blocks()
    sides = sorted(t for t in d.built_mesh.cell_sets if "|" in t)
    secondary = next(t for t in sides if t.endswith(".cap"))
    main = next(t for t in sides if t.endswith(".base"))
    u, phi = d.fem_symbols(value_shape=(2,))
    X = d.variable("interior", split=True)[:2]
    eu, ep = sym(u, list(X)), sym(phi, list(X))
    sv = d.variable(secondary, split=True)
    n = d.variable(secondary, normals=True)
    g = u.gap(secondary, main, domain=d)
    sq = jno.np.parameter((1,), name="sq", key=_jax.random.PRNGKey(0))
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    fem = jno.fem(
        [
            LAM_ * trace(eu) * trace(ep) + 2 * MU_ * inner(eu, ep, n_contract=2),
            jno.np.maximum(0.0, -1e4 * g) * inner(n, phi.bind(x=sv[0], y=sv[1]), n_contract=1),
            u(xb, yb)[0] - 0.0,
            u(xb, yb)[1] - 0.0,
            u(xt, yt)[0] - 0.0,
            u(xt, yt)[1] - sq,
        ]
    )
    base_ids = np.asarray(d.tag_indices[main]).reshape(-1)

    import jax.numpy as jnp

    op = fem.operator  # FemResidualOperator: residual(u, args, t) with the held value re-formed per call
    ndofs = int(fem.dofs)
    drv = jno.solve.newton(line_search=True, rtol=1e-6, atol=1e-6)

    def out(sqv):
        r = lambda w: jnp.asarray(op.residual(w, {"sq": jnp.asarray(sqv).reshape(-1)})).reshape(-1)  # noqa: E731
        w = drv(r, jnp.zeros(ndofs))
        return w.reshape(-1, 2)[base_ids, 1].mean()

    grad = float(_jax.grad(out)(jnp.asarray(-0.02)))
    fd = float((out(jnp.asarray(-0.022)) - out(jnp.asarray(-0.020))) / (-0.002))
    # jax.grad runs through the driver's custom_root on the branch-selected (semismooth) operator;
    # in the closed regime the response is smooth in the platen displacement and the two must agree.
    assert 0.1 < fd < 1.0, f"FD slope {fd:.3f} outside the physical (0,1) load-share range"
    assert abs(grad - fd) < 0.05 * abs(fd), f"gradient through closed one-sided contact: jax.grad {grad:.4f} vs FD {fd:.4f}"


def test_augmented_lagrangian_beats_the_penalty_error_at_the_same_c():
    """AL exactness. The pure penalty carries an O(1/c) penetration error by construction; the
    Uzawa multiplier update `lam.evolves(max(0, lam.i(-1) - c*g))` absorbs it, so at the SAME
    c the marched AL answer must land far closer to the bonded oracle. Measured while writing
    this test: penalty err 3.4e-3 at c=1e3; AL err 1.1e-5 after 8 updates -- monotone geometric
    convergence (-0.00663 -> -0.00886 -> ... -> -0.00999). The multiplier is an ordinary scalar
    SURFACE state on the secondary face: the machinery is `evolves` + the tau march, no new API."""
    inner, sym, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
    c, nsteps = 1e3, 8
    d = jno.Shape.regions(
        base=jno.Shape.rect(0, 0, 1, 1, size=0.22),
        cap=jno.Shape.rect(0, 1, 1, 2, size=0.09),
        conforming=False,
    ).domain(tau=(0.0, 1.0, nsteps))
    sides = sorted(t for t in d.built_mesh.cell_sets if "|" in t)
    secondary = next(t for t in sides if t.endswith(".cap"))
    main = next(t for t in sides if t.endswith(".base"))
    u, phi = d.fem_symbols(value_shape=(2,))
    lam, _ = d.fem_symbols(value_shape=())  # the AL multiplier: a scalar surface state
    X = d.variable("interior", split=True)[:2]
    eu, ep = sym(u, list(X)), sym(phi, list(X))
    sv = d.variable(secondary, split=True)
    n = d.variable(secondary, normals=True)
    g = u.gap(secondary, main, domain=d)
    p = jno.np.maximum(0.0, lam.i(-1) + c * (-g))
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    fem = jno.fem(
        [
            LAM_ * trace(eu) * trace(ep) + 2 * MU_ * inner(eu, ep, n_contract=2),
            p * inner(n, phi.bind(x=sv[0], y=sv[1]), n_contract=1),
            lam.evolves(p),
            u(xb, yb)[0] - 0.0,
            u(xb, yb)[1] - 0.0,
            u(xt, yt)[0] - 0.0,
            u(xt, yt)[1] - (-0.02),
        ]
    )
    # atol above the measured float32 residual floor (~1.7e-5): the march guard re-checks each step.
    traj = np.asarray(fem.solve(nonlinear=jno.solve.newton(line_search=True, rtol=1e-5, atol=3e-5)))
    m = np.asarray(d.tag_indices[main]).reshape(-1)
    uys = [traj[k].reshape(-1, 2)[m, 1].mean() for k in range(traj.shape[0])]
    errs = [abs(v - (-0.01)) for v in uys]
    assert errs[0] > 2e-3, "step 0 is the pure-penalty answer and must carry the 1/c error"
    assert all(e2 <= e1 * 1.05 for e1, e2 in zip(errs, errs[1:])), f"AL error must fall monotonically: {errs}"
    assert errs[-1] < 1e-4, f"AL must beat the penalty error by orders of magnitude: final err {errs[-1]:.2e}"
