"""Reduced-order solves — ``fem.solve(basis=U)``.

A Galerkin/POD basis and a periodic prolongation are the SAME object: a tall ``(n_dofs, k)`` map ``P``
defining ``PᵀAP`` / ``Pᵀb`` / ``u = P x``. So this feature is not a new solver — it hands a user basis to
the reduction machinery the periodic ties already drive, and the answer comes back in the full space.

What makes it different from every other path here is that it returns an **approximation**. That is why
the full-system residual at the lifted solution is measured on every call and a basis that does not span
the solution is refused: a plausible-looking wrong field is the one outcome this stack never returns.

Run with x64 (the FEM assembly is float64).
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson(x0, ms=0.08):
    """``-Δu = f`` with a Gaussian source at ``(x0, 0.5)`` — a one-parameter family whose solutions
    live on a low-dimensional manifold, which is what makes a reduced basis worth anything."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=ms)
    u, phi = d.fem_symbols()
    si, sb = d.variable("interior", split=True), d.variable("boundary", split=True)
    xi, yi, xb, yb = si[0], si[1], sb[0], sb[1]
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 50.0 * jno.np.exp(-60.0 * ((xi - x0) ** 2 + (yi - 0.5) ** 2))
    return jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0])


def _pod(k=10, ms=0.08):
    """POD basis from a training sweep: ``(n_dofs, k)``, orthonormal by construction."""
    snaps = np.stack([np.asarray(_poisson(x0, ms).solve()).reshape(-1) for x0 in np.linspace(0.2, 0.8, 12)])
    U, _s, _ = np.linalg.svd(snaps.T, full_matrices=False)  # columns of U are the spatial modes
    return jnp.asarray(U[:, :k])


def _eval(x):
    """Solve results as a flat array. The reduced NONLINEAR path returns a deferred trace node (it rides
    the same route as a periodic nonlinear solve, which is lazy); the linear one is eager."""
    return np.asarray(x.fn() if hasattr(x, "fn") else x).reshape(-1)


def _rel(a, b):
    return float(np.linalg.norm(_eval(a) - np.asarray(b).reshape(-1)) / np.linalg.norm(b))


def test_reduced_solve_is_accurate_at_a_held_out_parameter():
    """The headline: 10 unknowns instead of the full DOF count, at a parameter the basis never saw,
    and the answer comes back in the FULL space so nothing downstream changes."""
    U = _pod(10)
    fem = _poisson(0.435)  # deliberately between training samples
    u_full = np.asarray(fem.solve()).reshape(-1)
    u_rom = np.asarray(fem.solve(basis=U)).reshape(-1)

    assert u_rom.shape == u_full.shape, "a reduced solve must still return the FULL field"
    assert _rel(u_rom, u_full) < 1e-3, f"reduced solve too far from the full one: {_rel(u_rom, u_full):.2e}"
    assert fem.basis_residual is not None and fem.basis_residual < 1e-2


def test_accuracy_improves_monotonically_with_rank():
    """More modes must mean less error — the property that makes a rank sweep meaningful. Also pins the
    certificate as a CONSERVATIVE proxy: it tracks the error and never under-reports it."""
    U = _pod(10)
    fem_ref = _poisson(0.435)
    u_full = np.asarray(fem_ref.solve()).reshape(-1)

    errs, certs = [], []
    for k in (2, 4, 6, 8, 10):
        fem = _poisson(0.435)
        fem.BASIS_RESIDUAL_LIMIT = 1e9  # sweeping ranks on purpose: opt out of the span guard
        errs.append(_rel(fem.solve(basis=U[:, :k]), u_full))
        certs.append(fem.basis_residual)

    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)), f"error must fall with rank: {errs}"
    assert all(c >= e for c, e in zip(certs, errs)), f"certificate must not under-report the error: {certs} vs {errs}"


def test_complete_basis_reproduces_the_full_solve_exactly():
    """Edge case that pins the algebra rather than the approximation: with k = n_dofs and an orthonormal
    basis the span IS the full space, so the reduced solve must return the full answer to round-off."""
    fem = _poisson(0.5, ms=0.25)  # small mesh — this builds a dense n x n basis
    n = fem.dofs
    U = jnp.asarray(np.linalg.qr(np.random.default_rng(0).standard_normal((n, n)))[0])
    assert U.shape == (n, n)
    # tolerance is kappa(A)*eps, not eps: UᵀAU is an orthogonal congruence, so it keeps A's conditioning
    # and the dense reduced solve carries the same round-off the full one does.
    assert _rel(fem.solve(basis=U), np.asarray(fem.solve()).reshape(-1)) < 1e-7


def test_matches_a_hand_rolled_reduction_exactly():
    """The slot must be exactly ``Uᵀ A U`` / ``Uᵀ b`` / ``U x`` and nothing else — pinned against the
    reduction primitives applied by hand, so the wiring cannot drift from the mathematics."""
    from jno.utils.solver.fem_utils import prolong, reduce_matrix, reduce_vector

    U = _pod(8)
    fem = _poisson(0.37)
    A, b = fem.operator
    x_r = jnp.linalg.solve(jnp.asarray(reduce_matrix(U, A)), jnp.asarray(reduce_vector(U, b)).reshape(-1))
    assert _rel(fem.solve(basis=U), np.asarray(prolong(U, x_r)).reshape(-1)) < 1e-10


def test_composes_with_solver_slots():
    """``basis=`` picks the SPACE; ``linear=``/``precond=`` pick how to move through it. They are
    orthogonal because the reduction happens before any solver sees the operator."""
    U = _pod(10)
    ref = np.asarray(_poisson(0.435).solve()).reshape(-1)
    for slots in ({"linear": jno.solve.cg(tol=1e-12)}, {"linear": jno.solve.bicgstab(tol=1e-12)}, {"precond": None}):
        assert _rel(_poisson(0.435).solve(basis=U, **slots), ref) < 1e-3, f"slot composition changed the answer: {slots}"


def test_nonlinear_reduced_solve():
    """A nonlinear form reduces too — Newton runs on ``Uᵀr(U x)``. Note this is a MEMORY win, not a
    speed one: the full-order residual is still evaluated every step (hyper-reduction is not wired)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.12)
    u, phi = d.fem_symbols()
    si, sb = d.variable("interior", split=True), d.variable("boundary", split=True)
    xi, yi, xb, yb = si[0], si[1], sb[0], sb[1]
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    form = lambda: jno.fem([ui.x * vi.x + ui.y * vi.y + (u * u * u) * vi - 20.0 * vi, u(xb, yb) - 0.0])  # noqa: E731

    u_full = np.asarray(form().solve()).reshape(-1)
    U = jnp.asarray(np.linalg.qr(np.stack([u_full, u_full**2, np.ones_like(u_full)], axis=1))[0])
    node = form().solve(basis=U)
    assert hasattr(node, "fn"), "the reduced nonlinear solve is deferred, like a periodic nonlinear one"
    u_rom = _eval(node)
    assert u_rom.shape == u_full.shape, "a reduced solve must still return the FULL field"
    assert _rel(u_rom, u_full) < 5e-2, f"nonlinear reduced solve is off: {_rel(u_rom, u_full):.2e}"


def test_basis_is_per_call_and_does_not_stick():
    """The reduction is installed for the duration of ONE solve. A later plain solve on the same object
    must be the full one — a basis leaking onto the object would silently approximate every later call."""
    U = _pod(6)
    fem = _poisson(0.435)
    fem.solve(basis=U)
    assert fem._periodic is None, "the basis must not persist on the FEM object"
    assert _rel(fem.solve(), np.asarray(_poisson(0.435).solve()).reshape(-1)) < 1e-12


def test_differentiable_in_the_basis():
    """The learned-subspace path: ``∂(solve)/∂U`` flows, so a network can produce the basis and be
    trained through the reduced solve. This is what the ``reduce_matrix`` dtype probe used to block —
    it materialised ``P`` with ``np.asarray`` purely to read a dtype, which throws on a tracer."""
    U = _pod(6)
    fem = _poisson(0.435)
    ref = jnp.asarray(np.asarray(fem.solve()).reshape(-1))

    def loss(Ub):
        return jnp.sum((jnp.asarray(fem.solve(basis=Ub)).reshape(-1) - ref) ** 2)

    g = jax.grad(loss)(U)
    assert g.shape == U.shape
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.max(jnp.abs(g))) > 0.0

    # NOT asserted, but worth recording: a plain gradient step does NOT descend here. An orthonormal
    # basis lives on the Stiefel manifold, so U - eps*g leaves the feasible set and re-orthonormalising
    # it (QR) rotates the span by an amount unrelated to eps -- enough to leave the solution's span
    # entirely. A learned basis therefore wants the orthonormalisation INSIDE the differentiated
    # function (net -> QR -> basis), not a post-hoc projection of the step.


def test_a_basis_that_does_not_span_the_solution_fails_loud():
    """The core safety property. A reduced solve is the one path here that returns an approximation, so
    a basis unrelated to the problem must RAISE rather than hand back a plausible wrong field."""
    fem = _poisson(0.435)
    junk = jnp.asarray(np.linalg.qr(np.random.default_rng(1).standard_normal((fem.dofs, 2)))[0])
    with pytest.raises(ValueError, match="does not satisfy the full system|does not span"):
        fem.solve(basis=junk)
    assert fem.basis_residual is not None, "the measured residual must be readable after the refusal"


def test_basis_shape_and_orthonormality_guards():
    """Each malformed basis refused with its own reason. The orientation one matters most: a snapshot
    matrix is (n_snapshots, n_dofs), the transpose of what a basis needs, and that is the mistake
    everyone makes first — so the message says which factor of an SVD to take."""
    fem = _poisson(0.435)
    n = fem.dofs
    U = _pod(4)

    with pytest.raises(ValueError, match="2-D"):
        fem.solve(basis=jnp.ones(n))
    with pytest.raises(ValueError, match="rows but this problem has|transpos"):
        fem.solve(basis=jnp.asarray(np.asarray(U).T))  # (k, n_dofs) — the classic orientation slip
    with pytest.raises(ValueError, match="ORTHONORMAL"):
        fem.solve(basis=U * 2.0)
    with pytest.raises(ValueError, match="ORTHONORMAL"):
        fem.solve(basis=jnp.asarray(np.random.default_rng(2).standard_normal((n, 3))))
    with pytest.raises(ValueError, match="1 <= k"):
        fem.solve(basis=jnp.zeros((n, 0)))


def test_non_float_bases_are_refused():
    """dtype guards, each for a reason the orthonormality check cannot catch.

    An INTEGER basis sails through orthonormality (an identity slice is exactly orthonormal) and then
    silently truncates the reduced solve to integers. A COMPLEX one is worse: the reduction here is
    ``UᵀAU``, not the Hermitian ``UᴴAU``, so it is the wrong projection *and* it hands back a complex
    field for a real problem — which is what it did before this guard."""
    fem = _poisson(0.435)
    n = fem.dofs
    with pytest.raises(ValueError, match="real floating-point"):
        fem.solve(basis=jnp.asarray(np.eye(n, 3, dtype=np.int32)))
    with pytest.raises(ValueError, match="real floating-point|Hermitian"):
        fem.solve(basis=jnp.asarray(np.asarray(_pod(4)).astype(np.complex128)))


def test_non_finite_basis_is_refused():
    """NaN/Inf in the basis must be caught at the door, not surface later as a NaN field."""
    fem = _poisson(0.435)
    for bad in (np.nan, np.inf):
        U = np.asarray(_pod(4)).copy()
        U[0, 0] = bad
        with pytest.raises(ValueError, match="NaN or Inf"):
            fem.solve(basis=jnp.asarray(U))


def test_accepts_numpy_and_list_bases():
    """A basis is a plain array, so the ordinary array-likes must work. A numpy basis used to CRASH:
    the reduction sniffed BCOO with ``hasattr(x, "data")``, and a numpy array has ``.data`` too — a
    memoryview, with no ``.dtype``."""
    U = np.asarray(_pod(8))
    ref = np.asarray(_poisson(0.435).solve()).reshape(-1)
    assert _rel(_poisson(0.435).solve(basis=U), ref) < 1e-3  # numpy
    assert _rel(_poisson(0.435).solve(basis=U.tolist()), ref) < 1e-3  # nested list


def test_basis_residual_is_cleared_by_a_later_full_solve():
    """Staleness guard. ``basis_residual`` left over from an earlier reduced solve would read as
    'this answer was certified' on an answer that was never reduced at all."""
    fem = _poisson(0.435)
    fem.solve(basis=_pod(8))
    assert fem.basis_residual is not None
    fem.solve()  # a FULL solve on the same object
    assert fem.basis_residual is None, "a full solve must not leave a stale certificate behind"


def test_remeshing_slots_are_refused():
    """``adapt=``/``move=`` rebuild or move the mesh, so the DOF count and layout the basis was built
    against change underneath it — the basis would be silently meaningless, not merely inaccurate."""
    fem = _poisson(0.435)
    U = _pod(4)
    with pytest.raises(NotImplementedError, match="adapt="):
        fem.solve(basis=U, adapt={"max_iters": 1})
    with pytest.raises(NotImplementedError, match="move="):
        fem.solve(basis=U, move=object())


def test_reduction_is_restored_even_when_the_solve_raises():
    """The reduction is installed on the object for the duration of one call. If an exception escapes
    mid-solve it must still come off, or every later solve on that object is silently reduced."""
    fem = _poisson(0.435)
    with pytest.raises(RuntimeError):
        fem.solve(basis=_pod(8), solve_fn=lambda A, b: (_ for _ in ()).throw(RuntimeError("boom")))
    assert fem._periodic is None, "the reduction must be removed even on the exception path"


def test_trainable_parameter_basis_is_refused_with_a_reason():
    """A ``jno.np.parameter`` basis is a trace node, not an array — threading it as a runtime parameter
    is not wired. Refused by name, and the message points at the ``jax.grad`` path that DOES work."""
    fem = _poisson(0.435)
    P = jno.np.parameter((fem.dofs, 4), name="rom_basis")
    P.initialize(jax.nn.initializers.constant(0.0))
    with pytest.raises(NotImplementedError, match="concrete array|jno.np.parameter"):
        fem.solve(basis=P)


def test_unsupported_modes_fail_loud():
    """Scope, each with its own reason rather than one blanket refusal: transient (the time block is
    reduced eagerly at assembly), complex (solves through an internal real-equivalent 2n layout), and
    a periodic tie (two prolongations whose composition has no decided convention yet)."""
    # transient
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2, time=(0.0, 0.05, 6))
    u, phi = d.fem_symbols()
    si = d.variable("interior", split=True)
    xi, yi, ti = si[0], si[1], si[-1]
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    u0 = jno.np.sin(np.pi * ci[0]) * jno.np.sin(np.pi * ci[1])
    fem_t = jno.fem([ui.t * vi + (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])
    with pytest.raises(NotImplementedError, match="transient"):
        fem_t.solve(basis=jnp.asarray(np.linalg.qr(np.random.default_rng(0).standard_normal((fem_t.dofs, 3)))[0]))

    # periodic tie
    dp = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    up, php = dp.fem_symbols()
    sp_i, sp_l, sp_r = (dp.variable(r, split=True) for r in ("interior", "left", "right"))
    xi2, yi2 = sp_i[0], sp_i[1]
    upi, vpi = up.bind(x=xi2, y=yi2), php.bind(x=xi2, y=yi2)
    fem_p = jno.fem([upi.x * vpi.x + upi.y * vpi.y - 1.0 * vpi, up(sp_l[0], sp_l[1]) - up(sp_r[0], sp_r[1])])
    with pytest.raises(NotImplementedError, match="periodic"):
        fem_p.solve(basis=jnp.asarray(np.linalg.qr(np.random.default_rng(0).standard_normal((fem_p.dofs, 3)))[0]))
