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
    `jax.linearize` of the residual, which is why no assembled Jacobian block is built."""
    free, _ = _vector_poisson(None)
    jumps = {c: _vector_poisson(c)[0] for c in (1e2, 1e3, 1e4, 1e5)}

    assert jumps[1e2] < 0.5 * free, "a penalty must reduce the jump at all"
    for lo, hi in ((1e2, 1e3), (1e3, 1e4), (1e4, 1e5)):
        ratio = jumps[lo] / jumps[hi]
        assert 3.0 < ratio < 30.0, f"expected ~10x per decade of c, got {ratio:.1f} ({lo}->{hi})"
    assert jumps[1e5] < 1e-3 * free


def test_the_assembled_tangent_refuses_a_gap():
    """A gap is non-local, and the per-element Jacobian emits parent-cell columns only, so it would
    silently drop the secondary–main coupling. It must refuse rather than return a degraded tangent."""
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
        g * jno.np.inner(n, v.bind(x=sb[0], y=sb[1], z=sb[2]), n_contract=1),
    ]
    bb = d.variable("boundary", split=True)
    terms += [u(bb[0], bb[1], bb[2])[i] - 0.0 for i in range(3)]
    fem = jno.fem(terms, element_type="TET4")
    with pytest.raises(NotImplementedError, match="matrix-free tangent"):
        fem.jacobian(np.zeros(fem.n_dofs) if hasattr(fem, "n_dofs") else np.zeros(1))


# ---------------------------------------------------------------------------
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
