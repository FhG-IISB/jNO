"""``jno.precond`` -- preconditioner **specs** for ``fem.solve(precond=...)``.

A spec is declarative: it says *what* preconditioner to build, and jno materializes it at solve
time against a :class:`jno.utils.solver.solver_api.PrecondContext` (the assembled operator; later
per-field blocks and auxiliary weak-form assembly). The materialized applier is just
``v -> M^{-1} v`` and composes with any Krylov solver from ``jno.solve``.

Preconditioners change convergence *speed*, never the converged solution, so a spec needs no
gradient path -- arbitrary (even non-JAX, via a future callback tier) appliers stay compatible
with differentiable solves. A user spec is any object with ``materialize(ctx)`` or a bare
``ctx -> (v -> M^{-1} v)`` callable::

    def my_precond(ctx):
        inv = 1.0 / ctx.diag()
        return lambda v: inv * v

    fem.solve(linear=jno.solve.cg(), precond=my_precond)

Composition: :func:`block_diag` / :func:`triangular` build block preconditioners over the
per-field DOF blocks (``fem.blocks``); :func:`form` assembles auxiliary weak-form operators
("preconditioners as weak forms"); :func:`inner` turns any ``jno.solve`` solver into an
(inexact) ``M^{-1}`` application.
"""

from __future__ import annotations

import jax.numpy as jnp

from .utils.solver.solver_api import (  # noqa: F401  (PrecondContext re-exported for user specs)
    PrecondApplier,
    PrecondContext,
    materialize_precond,
)

__all__ = [
    "PrecondContext",
    "jacobi",
    "chebyshev",
    "form",
    "inner",
    "block_diag",
    "triangular",
    "amg",
    "jaxamg",
    "cached",
    "ams",
    "real_equivalent",
    "gmg",
    "nystrom",
]


class _Spec:
    """Base for ``jno.precond.*`` specs — gives every preconditioner a fluent ``.cached()``.

    ``complex_ok`` declares that a spec can be applied to a **complex** operator directly. It is read
    where a consumer would otherwise have to reformulate: AMS splits each complex auxiliary into the
    real-equivalent 2n block ``[[Re,-Im],[Im,Re]]`` because AmgX-style multigrid is real-only, and that
    block is skew-dominated by construction (measured ‖A-Aᵀ‖/‖A‖ = 2.0), which destroys algebraic
    multigrid — smoothed aggregation diverged to 1e+20 on it. The underlying complex operator is
    complex-SYMMETRIC (8e-17), where the same solver converges in 5-7 iterations. So a complex-capable
    aux must be able to say so and skip the reformulation.

    Two class-level declarations decide whether a spec can ride the *compiled* slot path (see
    :func:`jno.utils.solver.solver_api.compose_linear_solve_fn`); both default to the conservative
    answer, so a new preconditioner is correct-but-eager until it says otherwise.

    ``traceable`` — can :meth:`materialize` run **inside** a trace? That means building the applier
    from the traced operator alone: no scipy/pyamg assembly, no Python branching on values, and no
    use of ``ctx.fem`` (the compiled path has no concrete FEM to offer). Jacobi qualifies because it
    reads the diagonal and nothing else; ``amg``/``ams``/``form`` assemble host-side, and Chebyshev
    branches on measured spectrum bounds.

    ``key`` — the spec's value identity, for the same reason :class:`LinearSolver` has one: the
    compiled path passes the spec to ``jax.jit`` as a static argument, and identity hashing would
    recompile on every call. ``None`` means identity, which keeps the spec off the compiled path.
    """

    traceable = False
    key = None
    complex_ok = False  # conservative: a consumer reformulates unless the spec says it takes complex

    def cached(self, *, refresh=False):
        """Wrap this preconditioner so its setup is built **once** and reused across solves — the
        fluent form of :func:`cached`. E.g. ``jno.precond.amg().cached()``. ``refresh`` controls
        invalidation (``False`` frozen, ``True`` on shape/sparsity change, an ``int k`` to rebuild
        every k-th materialization, or a ``ctx -> key`` callable)."""
        return _Cached(self, refresh)

    def __eq__(self, other):
        if self.key is None or not isinstance(other, _Spec) or other.key is None:
            return self is other
        return self.key == other.key

    def __hash__(self):
        return id(self) if self.key is None else hash(self.key)


class _Jacobi(_Spec):
    """Spec for the diagonal (Jacobi) preconditioner; see :func:`jacobi`."""

    traceable = True  # the diagonal comes off the traced operator; nothing is assembled host-side
    key = ("jacobi",)  # no parameters, so every jacobi() is the same preconditioner
    complex_ok = True  # a diagonal scaling is dtype-agnostic

    def materialize(self, ctx: PrecondContext):
        d = ctx.diag()
        safe = jnp.where(jnp.abs(d) > 1e-30, d, 1.0)  # zero diagonals (saddle blocks) left unscaled
        inv = 1.0 / safe
        return PrecondApplier(lambda v: inv * v)  # diagonal => symmetric => M^T == M

    def __repr__(self):
        return "jno.precond.jacobi()"


class _RealEquivalent(_Spec):
    """Spec for preconditioning the fused real-equivalent block; see :func:`real_equivalent`."""

    # Deliberately NOT complex_native: this spec WANTS the fused 2n form, because its whole purpose is
    # to hand a REAL operator to a real inner solver. Declaring complex-native would route the solve to
    # the complex n-sized operator and defeat it.
    complex_ok = False

    def __init__(self, inner):
        self.inner = inner

    def prepare(self, fem):
        if hasattr(self.inner, "prepare"):
            self.inner.prepare(fem)

    def materialize(self, ctx: PrecondContext):
        from .utils.solver.solver_api import LinearOperator, _slice_bcoo, materialize_precond

        A = ctx.A
        n2 = int(A.shape[0])
        if A.bcoo is None:
            raise ValueError(
                "jno.precond.real_equivalent needs an ASSEMBLED (sparse) operator: it reads the K and M "
                "blocks off the fused real-equivalent system. It cannot run on a matrix-free operator."
            )
        if n2 % 2:
            raise ValueError(
                f"jno.precond.real_equivalent expects the fused real-equivalent block, whose size is "
                f"EVEN (2n for a complex n-sized system); got {n2}. Use it on a complex problem."
            )
        n = n2 // 2
        top, bot = slice(0, n), slice(n, n2)
        # The fused form is [[K, -M], [M, K]] over [x_r; x_i], so K is the (0,0) block and the (0,1)
        # block is -M. K + M -- real, and definite whenever K is and M >= 0 -- is the inner operator
        # the Axelsson/Kucherov and Benzi/Bertaccini families are built around, and the reason this is
        # worth doing at all: a REAL inner solver applies unchanged, so a real-only GPU multigrid works
        # on a complex problem.
        K = _slice_bcoo(A.bcoo, top, top)
        negM = _slice_bcoo(A.bcoo, top, bot)
        if K is None or negM is None:
            raise ValueError("jno.precond.real_equivalent: could not take the K / M blocks of the operator.")
        KM = LinearOperator((K - negM).sum_duplicates())

        inner = materialize_precond(self.inner, PrecondContext(KM, ctx.fem))
        inner_T = getattr(inner, "T", inner)

        def _apply(fn):
            def go(v):
                return jnp.concatenate([fn(v[:n]), fn(v[n:])])

            return go

        return PrecondApplier(_apply(inner), _apply(inner_T))

    def __repr__(self):
        return f"jno.precond.real_equivalent({self.inner!r})"


class _GMG(_Spec):
    """Spec for the geometric-multigrid V-cycle preconditioner; see :func:`gmg`."""

    def __init__(self, n_pre, n_post, omega, min_size):
        self.n_pre, self.n_post, self.omega, self.min_size = n_pre, n_post, omega, min_size

    def materialize(self, ctx: PrecondContext):
        grid = ctx.grid
        if grid is None:
            raise ValueError(
                "jno.precond.gmg() needs a structured grid — build the domain with "
                "Shape.rect(...).structured() / Shape.box(...).structured(). This operator has "
                "no grid descriptor, so there is no coarsening hierarchy to build."
            )
        from .utils.solver.geometric_mg import build_vcycle

        vcycle, n_levels = build_vcycle(
            grid["shape"],
            grid["spacing"],
            n_pre=self.n_pre,
            n_post=self.n_post,
            omega=self.omega,
            min_size=self.min_size,
        )
        if n_levels < 2:
            raise ValueError(
                "jno.precond.gmg(): the grid is too small to coarsen (a single level) — nothing to "
                "precondition. Use a finer grid, or jno.precond.jacobi() / amg()."
            )
        return PrecondApplier(vcycle)  # a V-cycle for -Δ is ~symmetric (SPD) → reuse M for the transpose

    def __repr__(self):
        return "jno.precond.gmg()"


class _Nystrom(_Spec):
    """Spec for the randomized-Nyström low-rank preconditioner; see :func:`nystrom`."""

    def __init__(self, rank, mu, seed):
        self.rank, self.mu, self.seed = rank, mu, seed

    def materialize(self, ctx: PrecondContext):
        import jax

        from .utils.solver.krylov import nystrom_apply, nystrom_sketch

        if ctx.A.shape is None:
            raise TypeError(
                "jno.precond.nystrom needs to know the operator size — this is a matvec-only "
                "operator with no shape. Wrap it with a shape, or use jno.precond.jacobi()."
            )
        n = int(ctx.A.shape[0])
        if self.rank >= n:
            raise ValueError(
                f"jno.precond.nystrom(rank={self.rank}) on an n={n} operator: the rank must be smaller "
                "than the system (a rank-n sketch costs n matvecs, i.e. more than a direct solve). "
                "Use a rank well below n — the point is to capture only the top of the spectrum."
            )
        U, lam = nystrom_sketch(ctx.A.mv, n, rank=self.rank, key=jax.random.PRNGKey(self.seed))
        # mu defaults to the smallest captured eigenvalue: the spectrum below the sketch is left
        # untouched (identity), so scaling by lam_min matches the two parts continuously.
        mu = self.mu if self.mu is not None else jnp.maximum(lam[-1], 1e-12)
        apply = nystrom_apply(U, lam, mu)
        return PrecondApplier(apply)  # U diag U^T + (I - U U^T) is symmetric => M^T == M

    def __repr__(self):
        return f"jno.precond.nystrom(rank={self.rank})"


class _Chebyshev(_Spec):
    """Spec for the fixed-degree Chebyshev polynomial preconditioner; see :func:`chebyshev`."""

    def __init__(self, degree, lmin, lmax, lmin_ratio, safety, bound_iters):
        self.degree = degree
        self.lmin, self.lmax = lmin, lmax
        self.lmin_ratio, self.safety, self.bound_iters = lmin_ratio, safety, bound_iters

    def materialize(self, ctx: PrecondContext):
        from .utils.solver.krylov import chebyshev_apply, spectrum_bounds

        if ctx.A.shape is None and self.lmax is None:
            raise TypeError(
                "jno.precond.chebyshev on a matvec-only operator needs explicit spectrum bounds "
                "(lmin=, lmax=) — there is no assembled matrix to estimate them from."
            )
        # Lanczos measures BOTH ends; power iteration + lmin_ratio is the fallback. See
        # `spectrum_bounds` for why a fabricated lmin can make the polynomial amplify.
        lo, hi = spectrum_bounds(
            ctx.A.mv,
            None if ctx.A.shape is None else ctx.A.shape[0],
            iters=self.bound_iters,
            lmin=self.lmin,
            lmax=self.lmax,
            safety=self.safety,
            lmin_ratio=self.lmin_ratio,
        )
        # A^T shares the spectrum of A, so the same [lo, hi] bounds the transpose recurrence.
        return PrecondApplier(
            lambda v: chebyshev_apply(ctx.A.mv, v, lmin=lo, lmax=hi, degree=self.degree),
            lambda v: chebyshev_apply(ctx.A.T.mv, v, lmin=lo, lmax=hi, degree=self.degree),
        )

    def __repr__(self):
        return f"jno.precond.chebyshev(degree={self.degree})"


def real_equivalent(inner) -> _RealEquivalent:
    """Precondition a COMPLEX system through its fused real-equivalent block, with a **real** inner
    solver — so a real-only preconditioner (an AmgX-style GPU multigrid, say) works on it.

    A complex symmetric ``A = K + iM`` becomes ``[[K, -M], [M, K]]`` over ``[x_r; x_i]``. That block is
    skew-dominated (measured ``||A - A^T|| / ||A|| = 2.0`` when the mass term dominates), which is why
    multigrid applied to it *directly* diverges. The classical fix is not to precondition the block
    itself but to use ``K + M``: real, symmetric, and definite whenever ``K`` is and ``M >= 0``. Here
    that is applied to each of the two diagonal halves, and ``inner`` is any ordinary real spec::

        fem.solve(linear=jno.solve.fgmres(),
                  precond=jno.precond.real_equivalent(jno.precond.amg()))

    Measured on a complex A-V (N1E x Lagrange) system, GMRES to 1e-8 with an AMG inner solve:
    **18 iterations** where the imaginary part is definite.

    **It is only as good as the gauge.** On an eddy-current problem ``M = w sigma`` VANISHES outside
    the conductors, which is exactly the assumption (definite imaginary part) most of this literature
    makes. Measured on such a system, regularised by a displacement-current mass term of size ``eps``:

        eps = 1e-6 -> 40000 its | 1e-3 -> 2946 | 1e-2 -> 248 | 1e-1 -> 54 | 1 -> 16

    i.e. the iteration count tracks the gauge, not the vanishing ``M`` — with a well-conditioned gauge
    it matches the definite case. An ``eps`` mass term buys that conditioning by changing the physics;
    a tree-cotree gauge (``domain._extra_dof_pins``) buys it for free, and is the right pairing.

    ``inner`` sees ``K + M`` and never sees a complex number, so anything real composes -- but it must
    MATCH THAT OPERATOR'S CHARACTER, which is the practical trap. Where ``M`` vanishes, ``K + M`` is a
    bare curl-curl operator, and smoothed aggregation is the wrong tool for H(curl); that is what AMS
    is for. Measured on a cube: with a mass term everywhere (eps-gauge 1) ``real_equivalent(amg())``
    converges in ~20 iterations, and at eps 1e-4 -- so ``K + M`` is curl-curl in the non-conducting
    region -- the same call fails. Conductor thickness and volume fraction are NOT the variable: 20
    iterations from 60 % of the volume down to 0.86 %, and 4 elements thick down to 1.

    So on an eddy problem with a non-conducting region, pair this with an H(curl) inner (``ams()``),
    not ``amg()``. ``K + M`` being real is what keeps a GPU route open -- it wants a real AMS.

    References: Day & Heroux, *Solving complex-valued linear systems via equivalent real formulations*,
    SIAM J. Sci. Comput. 23(2):480-498, 2001 (which equivalent real form to pick); Axelsson & Kucherov,
    *Real valued iterative methods for solving complex symmetric linear systems*, Numer. Linear Algebra
    Appl. 7:197-218, 2000; Benzi & Bertaccini, *Block preconditioning of real-valued iterative
    algorithms for complex linear systems*, IMA J. Numer. Anal. 28:598-618, 2008.
    """
    return _RealEquivalent(inner)


def jacobi() -> _Jacobi:
    """Diagonal (Jacobi) preconditioner ``M^{-1} v = v / diag(A)``.

    The cheapest useful preconditioner: one elementwise multiply per application, ``jit``- and
    ``vmap``-native, effective on diagonally-dominant (elliptic) systems -- heat, diffusion,
    elasticity. Zero/near-zero diagonals (e.g. the pressure block of a saddle-point system) are
    left unscaled so it never produces ``inf``/``NaN`` -- but it does not *rescue* saddle
    systems; use ``jno.solve.lu()`` or a :func:`triangular` block/Schur spec there.

    ``fem.solve(linear=jno.solve.bicgstab(), precond=jno.precond.jacobi())`` reproduces the
    historic steady-linear default exactly.
    """
    return _Jacobi()


def gmg(*, n_pre: int = 2, n_post: int = 2, omega: float | None = None, min_size: int = 5) -> _GMG:
    """Geometric-multigrid V-cycle preconditioner for a **structured grid**
    (``jno.Shape.rect(...).structured().domain()``).

    Builds a coarsen-by-2 grid hierarchy and applies one V-cycle as ``M⁻¹``: damped-Jacobi smoothing
    (``n_pre``/``n_post`` sweeps, ``omega`` damping — default the model-problem optimum ``2d/(2d+1)``),
    full-weighting restriction, multilinear prolongation, **rediscretised** coarse Laplacians, and a
    dense solve at the coarsest level (stops coarsening below ``min_size`` nodes/axis or at an odd cell
    count). Convergence is **grid-independent** — ~0.1 residual reduction per V-cycle, O(N) work — on
    Poisson / Helmholtz-type operators. Matrix-free and differentiable; the V-cycle is a *fixed* linear
    operator, so standard GMRES (not FGMRES) suffices.

    Use it as ``fem.solve(linear=jno.solve.gmres(), precond=jno.precond.gmg())`` on a structured domain;
    a structured ``jno.fdm`` solve already uses it automatically. Raises if the operator has no
    structured grid, or the grid is too small to coarsen. v1 is constant-coefficient (the rediscretised
    coarse operator); a Galerkin ``RAP`` coarse operator for variable coefficients is future work.

    Reference: A. Brandt, *Multi-Level Adaptive Solutions to Boundary-Value Problems*, Mathematics of
    Computation 31(138), 1977.
    """
    return _GMG(n_pre, n_post, omega, min_size)


class _Form(_Spec):
    """Spec assembling an auxiliary weak form as the preconditioner operator; see :func:`form`."""

    def __init__(self, terms, inner_solver, quad_degree):
        # A CALLABLE is the solution-dependent variant: ``fn(sol) -> term list``, re-assembled from
        # the current outer iterate (see :meth:`refresh_from`). A plain list stays what it was: one
        # parameter-independent assembly, cached forever.
        self._terms_fn = terms if callable(terms) and not isinstance(terms, (list, tuple)) else None
        self.terms = None if self._terms_fn is not None else list(terms)
        self.inner = inner_solver
        self.quad_degree = quad_degree
        self._op = None  # assembled once (static terms); rebuilt per refresh (callable terms)

    def prepare(self, fem):
        """Eager one-time assembly (called at compose time, OUTSIDE any trace — assembling a
        fresh ``jno.fem`` inside the Newton/Picard ``while_loop`` entangles with loop tracers).
        The callable variant skips this: it has no terms until a concrete iterate arrives."""
        if self._op is None and self._terms_fn is None:
            from .utils.solver.solver_api import PrecondContext as _Ctx

            self._op = _Ctx(None, fem).assemble(self.terms, quad_degree=self.quad_degree)

    def refresh_from(self, sol, fem):
        """Re-assemble the auxiliary operator from the CONCRETE outer iterate ``sol``.

        Called by the composed nonlinear solve once per invocation, with the solve's entry iterate
        (the warm start / previous march step) — the **Picard-lagged preconditioner**: the
        coefficient trails the solution by one outer solve, which changes convergence *speed* only,
        never the answer (Elman, Silvester & Wathen, *Finite Elements and Fast Iterative Solvers*,
        2nd ed., OUP 2014, §9.2 uses exactly this lag for the viscosity-weighted Schur mass).

        Eager by necessity: every Newton driver's loop is a ``lax.while_loop``, so the per-step
        iterate is a tracer and no host assembly can see it. The lag is therefore not a shortcut but
        the only place a solution-dependent auxiliary CAN be assembled."""
        if self._terms_fn is None:
            return
        import numpy as _np

        from .utils.solver.solver_api import PrecondContext as _Ctx

        self.terms = list(self._terms_fn(_np.asarray(sol).reshape(-1)))
        self._op = _Ctx(None, fem).assemble(self.terms, quad_degree=self.quad_degree)

    def materialize(self, ctx: PrecondContext):
        if self._op is None and self._terms_fn is not None:
            raise NotImplementedError(
                "jno.precond.form(<callable>): this solution-dependent auxiliary was never assembled "
                "-- no concrete iterate reached it. It refreshes from the outer solve's entry iterate "
                "on the composed nonlinear path (fem.solve(nonlinear=..., precond=...)); a fully "
                "traced context (jit/vmap over the whole solve, a lax.scan march) never has one. "
                "Run the solve eagerly, or use a static form([...]) with a frozen coefficient."
            )
        if self._op is None:
            self._op = ctx.assemble(self.terms, quad_degree=self.quad_degree)
        op = self._op
        if self.inner is None:
            from . import solve as _s

            self.inner = _s.lu()
        solver = self.inner
        # M^{-T} v ~ (Â^{-1})^T v = (Â^T)^{-1} v: run the same inner solver on the transposed
        # auxiliary operator (for a symmetric Â -- e.g. a mass matrix -- op.T behaves like op).
        return PrecondApplier(lambda v: solver(op, v), lambda v: solver(op.T, v))

    def __repr__(self):
        what = "<solution-dependent>" if self._terms_fn is not None else f"<{len(self.terms)} terms>"
        return f"jno.precond.form({what}, inner={self.inner})"


def form(terms, *, inner=None, quad_degree: int = 2) -> _Form:
    """**Preconditioners as weak forms**: assemble an auxiliary operator ``Â`` from ordinary
    traced ``jno.fem`` terms and apply ``M^{-1} v = Â^{-1} v`` with ``inner`` (default
    ``jno.solve.lu()``).

    This is how the classical physics-based preconditioners are written declaratively — in the
    *same language as the PDE*:

    * a (weighted) **mass matrix** — e.g. the pressure Schur-complement approximation of a
      Stokes-type saddle system: ``jno.precond.form([w * pi * qi], inner=jno.solve.cg(...))``;
    * a **local proxy of a nonlocal operator** — assemble only the conduction terms to
      precondition a conduction+radiation system (the dense view-factor coupling stays in the
      outer matvec);
    * a **shifted/damped twin** of an indefinite operator (shifted-Laplacian Helmholtz);
    * a **low-order proxy** preconditioning a high-order discretisation.

    The auxiliary system is assembled once with the ordinary ``jno.fem`` machinery (cached on
    the spec — it is parameter-independent) and must be steady linear. Its size must match the
    (sub-)operator this spec preconditions: a form over one field's symbols preconditions that
    field's diagonal block inside :func:`block_diag`/:func:`triangular`; a form over all fields
    preconditions the full system. With an *iterative* ``inner``, drive the outer solve with
    ``jno.solve.fgmres()`` (flexible preconditioning).
    """
    return _Form(terms, inner, quad_degree)


class _InnerSolve(_Spec):
    """Spec using a configured linear solver as the ``M^{-1}`` application; see :func:`inner`."""

    def __init__(self, solver):
        self.solver = solver

    def materialize(self, ctx: PrecondContext):
        solver, op = self.solver, ctx.A
        # transpose applier solves the transposed (sub-)operator, so a non-symmetric block gets a
        # correctly-preconditioned adjoint solve (else reverse-mode stalls -- see PrecondApplier).
        return PrecondApplier(lambda v: solver(op, v), lambda v: solver(op.T, v))

    def __repr__(self):
        return f"jno.precond.inner({self.solver})"


def inner(solver) -> _InnerSolve:
    """Use a ``jno.solve`` linear solver as the preconditioner application ``M^{-1} v ≈ A^{-1} v``
    on whatever operator it is materialized against — the natural way to give a diagonal block of
    :func:`block_diag`/:func:`triangular` an (inexact) block solve, e.g.
    ``jno.precond.inner(jno.solve.cg(tol=1e-2, maxiter=50))``. With an iterative solver here the
    outer Krylov must be flexible: ``jno.solve.fgmres()``."""
    return _InnerSolve(solver)


def _pairs_to_appliers(pairs, ctx: PrecondContext):
    """Resolve ``(field, spec)`` pairs → offsets-ordered ``(slice, applier)`` per diagonal block."""
    from .utils.solver.solver_api import materialize_precond

    if ctx.fem is None:
        raise TypeError("Block preconditioners need the owning FEM (use them via fem.solve(precond=...)).")
    resolved = {}
    for field, spec in pairs:
        idx = ctx.fem.block_index(field)
        if idx in resolved:
            raise ValueError(f"Block preconditioner: field block {idx} specified twice.")
        resolved[idx] = spec
    blocks = ctx.blocks
    if sorted(resolved) != list(range(len(blocks))):
        raise ValueError(
            f"Block preconditioner: got specs for blocks {sorted(resolved)} but the system has "
            f"{len(blocks)} field blocks — every field needs exactly one (field, spec) pair."
        )
    appliers = []
    for idx in range(len(blocks)):
        sub_ctx = PrecondContext(ctx.sub(idx), ctx.fem)
        m = materialize_precond(resolved[idx], sub_ctx)
        # Normalize to a PrecondApplier so the block transpose paths (apply_T) always have `.T`.
        # A bare-callable block precond (e.g. amg) has no structural transpose, so `.T` reuses M
        # (M^T := M) — correct (a preconditioner never changes the converged solution), just not
        # transpose-optimal, exactly the firewall's bare-callable fallback one level down.
        if not isinstance(m, PrecondApplier):
            m = PrecondApplier(m)
        appliers.append((blocks[idx], m))
    return appliers


def _prepare_pairs(pairs, fem):
    """Recurse the eager-preparation hook into a block composition's child specs."""
    for _field, spec in pairs:
        prep = getattr(spec, "prepare", None)
        if prep is not None:
            prep(fem)


class _BlockDiag(_Spec):
    def __init__(self, pairs):
        self.pairs = pairs

    @property
    def complex_native(self):
        """A block composition is complex-native iff any child is (i.e. contains AMS).

        Without this, ``triangular((u, ams()), (p, amg()))`` on a complex A-V system silently fell
        through to the fused real-equivalent 2n block, where two things break at once:
        ``fem.blocks`` describes the n-sized COMPLEX layout, so every slice covered the wrong half
        of the operator — and AMS was applied to the skew-dominated 2n block its own docs say it
        diverges on (measured symptom: fgmres returned x ~ 0, relative residual exactly 1.0).
        Declaring it routes the composition through ``_solve_complex_block``: the outer Krylov runs
        on the sparse COMPLEX operator ``A_r + i·A_i``, whose n-layout the block slices are correct
        for; ``ctx.sub(i)`` hands each child its assembled complex diagonal sub-block (AMS is
        complex-native by design; pyamg builds complex hierarchies natively, see ``_AMG.complex_ok``).
        Note ``_AMS.prepare``'s auto-freeze intentionally no-ops here — ``_fem_concrete_operator``
        returns the full MIXED operator, whose size does not match G, so the eager build raises and
        is swallowed; AMS then assembles from the correctly-sized sub-block at materialize time.
        """
        return any(getattr(spec, "complex_native", False) for _f, spec in self.pairs)

    def prepare(self, fem):
        _prepare_pairs(self.pairs, fem)

    def materialize(self, ctx: PrecondContext):
        appliers = _pairs_to_appliers(self.pairs, ctx)

        def apply(v):
            out = jnp.zeros_like(v)
            for s, M in appliers:
                out = out.at[s].set(M(v[s]))
            return out

        def apply_T(v):  # block-diagonal transpose = transpose each independent block
            out = jnp.zeros_like(v)
            for s, M in appliers:
                out = out.at[s].set(M.T(v[s]))
            return out

        return PrecondApplier(apply, apply_T)

    def __repr__(self):
        return f"jno.precond.block_diag(<{len(self.pairs)} blocks>)"


class _Triangular(_Spec):
    def __init__(self, pairs):
        self.pairs = pairs

    @property
    def complex_native(self):
        """See ``_BlockDiag.complex_native`` — identical reasoning; the triangular sweep additionally
        applies the off-diagonal couplings ``ctx.sub(i, j)``, which on this path are matvecs through
        the complex operator and therefore also correctly sized."""
        return any(getattr(spec, "complex_native", False) for _f, spec in self.pairs)

    def prepare(self, fem):
        _prepare_pairs(self.pairs, fem)

    def materialize(self, ctx: PrecondContext):
        appliers = _pairs_to_appliers(self.pairs, ctx)
        k = len(appliers)
        subs = {(i, j): ctx.sub(i, j) for i in range(k) for j in range(i + 1, k)}

        def apply(v):
            # block backward substitution: y_i = M_i^{-1} (v_i - sum_{j>i} A_ij y_j)
            ys = [None] * k
            for i in reversed(range(k)):
                s, M = appliers[i]
                r = v[s]
                for j in range(i + 1, k):
                    r = r - subs[(i, j)].mv(ys[j])
                ys[i] = M(r)
            out = jnp.zeros_like(v)
            for (s, _), y in zip(appliers, ys):
                out = out.at[s].set(y)
            return out

        def apply_T(v):
            # transpose of an upper-triangular P is lower-triangular P^T: FORWARD substitution
            # with transposed diagonal blocks (M_i^T) and transposed couplings ((A_ji)^T = sub(j,i)^T):
            #   y_i = M_i^{-T} (v_i - sum_{j<i} A_ji^T y_j)
            ys = [None] * k
            for i in range(k):
                s, M = appliers[i]
                r = v[s]
                for j in range(i):
                    r = r - subs[(j, i)].T.mv(ys[j])
                ys[i] = M.T(r)
            out = jnp.zeros_like(v)
            for (s, _), y in zip(appliers, ys):
                out = out.at[s].set(y)
            return out

        return PrecondApplier(apply, apply_T)

    def __repr__(self):
        return f"jno.precond.triangular(<{len(self.pairs)} blocks>)"


def block_diag(*pairs) -> _BlockDiag:
    """Block-**diagonal** preconditioner over the per-field DOF blocks: each ``(field, spec)``
    pair materializes ``spec`` against that field's diagonal sub-operator (``field`` is the
    trial symbol from ``d.fem_symbols()``, or the integer block index). Cheaper per application
    than :func:`triangular` but ignores the coupling blocks — prefer :func:`triangular` for
    saddle systems."""
    return _BlockDiag(list(pairs))


def triangular(*pairs) -> _Triangular:
    """Block **upper-triangular** preconditioner ``P = [[Â_1, A_12, …], [0, Â_2, …], …]`` over
    the per-field blocks — the standard shape for saddle-point systems (Stokes / Taylor–Hood,
    mixed Poisson, Biot): the last-listed block is solved first, then substituted back through
    the *actual* off-diagonal coupling matvecs of the assembled operator.

    Each ``(field, spec)`` pair supplies the approximate **diagonal-block inverse** ``Â_i^{-1}``:
    e.g. ``jno.precond.inner(jno.solve.cg(tol=1e-4))`` (inexact block solve — but not too inexact:
    ``tol=1e-2`` measured 11x SLOWER end-to-end than ``1e-4`` on Taylor–Hood Stokes, because the
    outer Krylov pays more extra iterations than the cheaper block solve saves),
    ``jno.precond.chebyshev(...)`` (polynomial), or ``jno.precond.form([...])`` (auxiliary
    operator — for Stokes the classic pressure choice is the viscosity-weighted **mass matrix**
    ``form([(1/mu) * pi * qi])`` as the Schur-complement approximation; Elman, Silvester & Wathen,
    *Finite Elements and Fast Iterative Solvers*, 2nd ed., OUP 2014, §9.2). With inexact
    (iterative) block solves the outer Krylov must be flexible: ``jno.solve.fgmres()``."""
    return _Triangular(list(pairs))


class _AMG(_Spec):
    """Spec for the hybrid (pyamg-setup / pure-JAX-apply) AMG preconditioner; see :func:`amg`.

    ``traceable`` is a *property* here rather than the class-level flag, because whether this spec can
    be materialised inside a trace depends on whether its hierarchy exists yet. Unbuilt, ``materialize``
    calls pyamg on the host and must stay eager. **Built** -- via :meth:`build` or ``.cached()`` -- the
    levels are frozen data and :func:`~jno.utils.solver.amg.vcycle_apply` is pure JAX, so the applier
    traces like any other.

    That distinction is worth a lot. With the whole solve left eager, every Krylov iteration dispatched
    a ~10-level V-cycle op by op from Python, which buried AMG's entire algorithmic advantage: at
    n=46677 a built-hierarchy AMG solve measured 625 ms against 29 ms for Jacobi-BiCGStab -- 21x
    slower, despite needing an order of magnitude fewer iterations.
    """

    #: pyamg builds and solves complex hierarchies natively -- unlike AmgX, which is real-only. That
    #: difference is what lets an AMS complex auxiliary skip the 2n reformulation; see :class:`_Spec`.
    complex_ok = True

    def __init__(self, cycles, max_levels, coarse_size, smoother_degree):
        self.cycles = cycles
        self.max_levels = max_levels
        self.coarse_size = coarse_size
        self.smoother_degree = smoother_degree
        self._levels = None

    @property
    def traceable(self):  # only once the host-side pyamg setup has happened -- see the class docstring
        return self._levels is not None

    @property
    def key(self):
        """Value identity for the compiled slot path. The hierarchy is the compilation: two specs
        share a program only if they apply the *same* levels the same number of times. ``self``
        holds ``_levels``, so its ``id`` cannot be recycled while this spec is alive."""
        return None if self._levels is None else (type(self), id(self._levels), self.cycles)

    def build(self, A) -> "_AMG":
        """Eager one-time setup from a **concrete** operator (BCOO / dense / ``fem.A`` /
        ``LinearOperator``). Required before use inside traced (jit / vmap / parametric-inverse)
        solves — pyamg cannot run under a trace; the built hierarchy is then frozen closure data.
        Returns ``self`` (chainable). Rebuild when the operator values drift far from this one."""
        from .utils.solver.amg import build_hierarchy
        from .utils.solver.solver_api import LinearOperator

        if isinstance(A, LinearOperator):
            A = A.bcoo if A.bcoo is not None else A.dense()
        self._levels = build_hierarchy(
            A,
            max_levels=self.max_levels,
            coarse_size=self.coarse_size,
            smoother_degree=self.smoother_degree,
        )
        return self

    def materialize(self, ctx: PrecondContext):
        from .utils.solver.amg import build_hierarchy, vcycle_apply

        levels = self._levels  # persisted ONLY by an explicit eager .build(); None → rebuilt THIS solve
        if levels is None:
            A = ctx.A.bcoo if ctx.A.bcoo is not None else ctx.A.dense()
            # Rebuild each solve (NOT persisted): the hierarchy depends on the operator *values*, so
            # silently reusing a stale one across solves would quietly cost iterations. Wrap in
            # ``.cached()`` (or call ``.build()`` eagerly) to reuse it explicitly. Raises on a tracer.
            levels = build_hierarchy(
                A, max_levels=self.max_levels, coarse_size=self.coarse_size, smoother_degree=self.smoother_degree
            )
        cycles, A_op = self.cycles, ctx.A

        def apply(r):
            x = vcycle_apply(levels, r)
            for _ in range(cycles - 1):
                x = x + vcycle_apply(levels, r - A_op.mv(x))
            return x

        return apply

    def __repr__(self):
        return f"jno.precond.amg(cycles={self.cycles}, built={self._levels is not None})"


def amg(*, cycles: int = 1, max_levels: int = 10, coarse_size: int = 100, smoother_degree: int = 3) -> _AMG:
    """Hybrid **algebraic multigrid**: smoothed-aggregation setup by the optional ``pyamg``
    (Vaněk, Mandel & Brezina, Computing 56, 1996; Bell et al., JOSS 8(87):5495, 2023), applied as
    a pure-JAX V-cycle with Chebyshev polynomial smoothing (Adams et al., JCP 188, 2003) — see
    :mod:`jno.utils.solver.amg`.

    The host-side setup builds fixed-pattern level operators; the per-application V-cycle is then
    ``jit``/``vmap``-native and a fixed *linear* map, so it may precondition ``cg``/``minres`` as
    well as ``bicgstab``/``fgmres``. The mesh-independent convergence of multigrid makes this *the*
    preconditioner for large elliptic blocks — heat, diffusion, elasticity, the (Picard-lagged)
    velocity block of a saddle system inside :func:`triangular`.

    **Caching is explicit.** The hierarchy is (re)built at each solve — it depends on the operator
    *values*, so silently reusing a stale one would quietly cost iterations. To amortise the setup
    over a sweep / Newton loop / inverse solve, say so: ``jno.precond.amg().cached()``. Inside a
    **traced** context (jit, vmap, a parametric inverse) pyamg cannot run under the trace, so build
    once eagerly first — ``spec.build(fem.A)`` — and the frozen hierarchy is reused (a legitimate
    preconditioner while values drift: speed degrades gracefully, correctness never). pyamg is
    imported lazily — without it a clear ``ImportError`` explains the install. On a matvec-only
    sub-block the matrix is recovered via the (dense) block view.
    """
    return _AMG(cycles, max_levels, coarse_size, smoother_degree)


def _fem_concrete_operator(fem):
    """Concrete assembled operator of a linear/complex ``fem`` (real: the BCOO/dense matrix; complex:
    ``A_r + i·A_i``), evaluated at the current parameters — for eager preconditioner setup (``.build``)."""

    def _leg(o):  # a raw (A, b) or a parametric FemLinearSystem → its concrete matrix A
        A, _ = o.evaluate(None) if hasattr(o, "evaluate") else o
        return A

    op = fem.operator
    # A complex fem is fused into a real 2n system at assembly, but a complex-native preconditioner
    # (AMS) wants ``A_r + i·A_i`` — built from the Re/Im legs the fusion retains. The bare
    # ``_mode == "complex"`` case is the un-fused Bloch remnant, whose operator IS the leg pair.
    legs = getattr(fem, "_complex_legs", None) or (op if getattr(fem, "_mode", None) == "complex" else None)
    if legs is not None:
        from ._fem import _complex_operator

        return _complex_operator(_leg(legs[0]), _leg(legs[1]))
    if hasattr(op, "evaluate"):  # a bare parametric FemLinearSystem
        return _leg(op)
    return _leg(op[0]) if hasattr(op[0], "evaluate") else op[0]  # (A, b) or (FemLinearSystem, …)


class _AMS(_Spec):
    """Spec for the H(curl) auxiliary-space Maxwell (AMS) preconditioner; see :func:`ams`."""

    complex_native = True  # solve the COMPLEX operator directly (not the real-equivalent 2n block)

    def __init__(self, aux):
        self.aux = aux  # nodal aux solver; None -> host SuperLU factored ONCE per block, reused per apply
        self._G = None  # discrete gradient (node->edge), built once from the mesh topology
        self._Pis = None  # (Π_x, Π_y, Π_z) nodal->edge vector interpolation
        self._frozen = None  # host-assembled auxiliary operators, set by .build() for traced solves

    @property
    def traceable(self):
        """Can :meth:`materialize` run inside a trace? Only once the host work is already done.

        Unbuilt, it calls :meth:`_assemble_aux`, which is scipy on the host. **Built** -- by
        :meth:`build`, or automatically by :meth:`prepare` whenever the operator is concrete at compose
        time -- every ingredient of the applier is traceable: ``dinv`` comes from the traced
        ``ctx.diag()``, ``G``/``Pis`` are frozen BCOO constants, and the default auxiliary applies its
        SuperLU factor through ``jax.pure_callback``, which ``jit`` supports (the preconditioner is
        never differentiated -- the outer ``custom_linear_solve`` takes gradients through ``A``, not
        through ``M⁻¹``).

        This matters more here than anywhere else the flag appears. AMS applies a multi-level auxiliary
        solve on **every Krylov iteration**, so leaving it on the eager path dispatches that whole
        structure from Python per iteration. The same class-level default made a built AMG hierarchy
        measure 21x slower than Jacobi before it was fixed.

        A user-supplied ``aux`` gates the answer: it is called per apply, so AMS is only traceable if
        that solver is. ``jno.solve`` specs declare this as the ``jit`` trait; anything that does not
        declare it keeps AMS eager, which costs speed and never correctness.
        """
        if self._frozen is None or self._G is None:
            return False
        if self.aux is None or isinstance(self.aux, _AMG):
            return True  # both are applied through OUR pure_callback wrapper, which jit supports
        return bool(getattr(self.aux, "traits", {}).get("jit", False))

    @property
    def key(self):
        """Value identity for the compiled slot path. The frozen auxiliaries and the transfer operators
        ARE the compilation, so their identity is the key; ``self`` holds both, so neither ``id`` can be
        recycled onto another object while this spec is alive."""
        if not self.traceable:
            return None
        return (type(self), id(self._frozen), id(self._G), id(self.aux))

    def prepare(self, fem):
        """Eager setup at compose time (outside any trace): the mesh transfer operators G, Π, **and** —
        whenever the operator is concrete here (the forward *and* native parametric-inverse solves) — the
        frozen auxiliaries too, so an AMS-preconditioned solve is differentiable **automatically**, no
        explicit ``.build`` required. If only a tracer is available at this point (a raw ``jit``/``vmap``
        of ``fem.solve``, where there is no concrete operator to freeze from), freezing is deferred and
        :meth:`build` is the escape hatch — :meth:`materialize` says so."""
        self._transfer(fem.domain)
        if self._frozen is None:
            try:
                self.build(fem)  # auto-freeze when the operator is concrete & complete here
            except Exception:  # noqa: BLE001 — a tracer / a parameter-incomplete operator: fall through,
                pass  # and materialize either assembles from the concrete ctx or asks for an explicit .build

    def build(self, fem, *, field=None) -> "_AMS":
        """Eager host setup from a **concrete** ``fem`` — required to use AMS inside a **traced** solve
        (``jit`` / ``vmap`` / a **parametric-inverse** design loop), where the host scipy assembly of the
        nodal auxiliaries cannot run under the trace. It freezes the auxiliary operators once (from the
        operator at the current parameters); the frozen preconditioner stays valid while ``A(θ)`` drifts
        (speed degrades gracefully, correctness never), so the solve — **and its gradient**, which flows
        through the traced operator by implicit differentiation, not through the preconditioner — runs
        differentiably. Returns ``self`` (chainable): ``fem.solve(precond=jno.precond.ams().build(fem))``.

        On a **mixed** system (an N1E x Lagrange A-V pair, say) the concrete operator covers every
        field, so ``field=`` names the H(curl) trial symbol whose block AMS preconditions:
        ``ams().build(fem, field=u)``. Without it the operator and the discrete gradient disagree and
        this raises rather than failing inside the assembly."""
        self._transfer(fem.domain)
        from .utils.solver.solver_api import LinearOperator, PrecondContext

        ctx = PrecondContext(LinearOperator(_fem_concrete_operator(fem)), fem)
        # On a MIXED system the concrete operator is the whole thing -- (n_edges + n_verts) for an
        # N1E x Lagrange pair -- while AMS's discrete gradient G is (n_edges, n_verts). Assembling the
        # auxiliaries against it produced a bare `matmul: dimension mismatch` from inside the assembly,
        # naming nothing the caller could act on, and that raise is exactly why `prepare`'s auto-freeze
        # is a no-op inside a block composition. AMS cannot know which block is its own, so it is told.
        n_edges = self._G.shape[0]
        if int(ctx.A.shape[0]) != n_edges:
            if field is None:
                raise ValueError(
                    f"jno.precond.ams().build: this operator is {int(ctx.A.shape[0])} x "
                    f"{int(ctx.A.shape[0])} but the edge (H(curl)) space has {n_edges} DOFs, so it is a "
                    "MIXED system and AMS cannot tell which block is its own. Name it: "
                    "ams().build(fem, field=<the N1E trial symbol>)."
                )
            ctx = PrecondContext(ctx.sub(field), fem)
            if int(ctx.A.shape[0]) != n_edges:
                raise ValueError(
                    f"jno.precond.ams().build: the block for the given field is {int(ctx.A.shape[0])} "
                    f"DOFs but the edge space has {n_edges}. `field=` must name the H(curl) (N1E) "
                    "trial symbol, not another field of the system."
                )
        self._frozen = self._assemble_aux(ctx)
        return self

    def _transfer(self, domain):
        from .utils.solver.ams import discrete_gradient, nodal_vector_interpolation

        topo = domain._fem_nonnodal_topology
        self._G = discrete_gradient(topo)
        self._Pis = nodal_vector_interpolation(topo)

    def _assemble_aux(self, ctx: PrecondContext) -> dict:
        """Host (scipy) assembly of the nodal auxiliary operators from a **concrete** matrix — the part
        that cannot run under a trace. Returns frozen jno ``LinearOperator``s; :meth:`materialize` builds
        the (pure-JAX, differentiable) applier from these plus the smoother diagonal."""
        import numpy as np
        import scipy.sparse as sp
        from jax.experimental import sparse as jsp

        from .utils.solver.solver_api import LinearOperator

        if self._G is None:
            if ctx.fem is None:
                raise TypeError(
                    "jno.precond.ams needs the owning FEM to read the N1E edge topology — "
                    "use it via fem.solve(precond=jno.precond.ams()), not on a bare operator."
                )
            self._transfer(ctx.fem.domain)

        A = ctx.A.bcoo if ctx.A.bcoo is not None else ctx.A.dense()

        def _csr(M):
            # BCOO -> scipy CSR through its COO triplets: O(nnz), never densify. `.todense()` here would
            # materialize the full edge x edge operator (~80 GB at 70k edges) before re-sparsifying.
            # Duplicate indices sum, matching BCOO semantics; a traced operator's tracer .data raises under
            # np.asarray and is caught below as "cannot host-assemble under a trace".
            if hasattr(M, "indices") and hasattr(M, "data"):
                idx = np.asarray(M.indices)
                return sp.csr_matrix((np.asarray(M.data), (idx[:, 0], idx[:, 1])), shape=tuple(M.shape))
            return sp.csr_matrix(np.asarray(M))

        try:
            A_sp = _csr(A)
        except Exception as e:  # a traced operator (parametric/complex-inverse) can't be host-assembled
            raise RuntimeError(
                "jno.precond.ams assembles the nodal auxiliaries on the host from a concrete matrix, so it "
                "cannot run under a trace (jit/vmap/parametric-inverse). Freeze it once from a CONCRETE "
                "reference first — spec = jno.precond.ams().build(fem0), with fem0 the (non-parametric) fem "
                "at your reference parameters — then reuse that spec; the frozen preconditioner then runs "
                "and differentiates through (the gradient flows through the operator, not the preconditioner)."
            ) from e
        G_sp = _csr(self._G)
        Pi_sp = [_csr(P) for P in self._Pis]

        A_G = (G_sp.T @ A_sp @ G_sp).tocsr()  # gradient-space auxiliary operator GᵀAG
        g_scale = float(np.abs(A_G.data).max()) if A_G.nnz else 0.0
        if g_scale < 1e-12 * (float(np.abs(A_sp.data).max()) or 1.0):
            raise ValueError(
                "jno.precond.ams: the gradient auxiliary GᵀAG is ~0 — a pure curl-curl operator has no "
                "coercivity on the gradient space. Add a mass term (σ/ε-gauge: jω·ε·⟨A,v⟩) so it is "
                "non-singular; see the AMS docs."
            )

        def pin0(M):  # pin node 0 → remove the constant nodal mode so an exact aux solve is non-singular
            L = M.tolil()
            L[0, :] = 0.0
            L[:, 0] = 0.0
            L[0, 0] = 1.0
            return L.tocsr()

        def to_op(M):  # a pre-pinned scipy matrix -> (LinearOperator over a BCOO, the scipy CSR itself)
            # The CSR is kept so ``materialize`` can factor it ONCE on the host (SuperLU) and reuse the
            # factor across every apply, instead of re-factoring per Krylov iteration.
            csr = M.tocsr()
            coo = M.tocoo()
            idx = jnp.asarray(np.stack([coo.row, coo.col], axis=1))
            return LinearOperator(jsp.BCOO((jnp.asarray(coo.data), idx), shape=coo.shape)), csr

        # G's constant null-space (G·1 = 0) is exact; each Π_α's constant mode is redundant with it
        # (Π_α·1 = G·coordₐ is a gradient), so pinning node 0 in every auxiliary operator is correct.
        if not np.iscomplexobj(A_sp.data):
            g_op, g_csr = to_op(pin0(A_G))
            p_pairs = [to_op(pin0((P.T @ A_sp @ P).tocsr())) for P in Pi_sp]
            return {
                "complex": False,
                "g_op": g_op,
                "g_csr": g_csr,
                "p_ops": [o for o, _ in p_pairs],
                "p_csrs": [c for _, c in p_pairs],
            }

        # Complex operator (eddy νK + jωσM, or time-harmonic Maxwell K − k₀²εM + absorption): AmgX-style
        # multigrid aux solvers are real-only, so EVERY complex auxiliary is reformulated as the
        # real-equivalent 2n block [[Re,-Im],[Im,Re]], which the aux solves *exactly*.
        #
        # The gradient block used to keep only `Im A_G`, on the eddy-case reasoning that GᵀKG = 0 makes
        # A_G = jω·R purely imaginary, so A_G⁻¹ = -j·(Im A_G)⁻¹. That assumption fails twice for a driven
        # wave problem: (i) Re A_G = -k₀²·GᵀεMG ≠ 0, and (ii) with SURFACE-ONLY absorption (an impedance /
        # first-order absorbing BC and no volume loss) Im A_G is a *boundary* mass — identically zero on
        # every interior node, hence singular — so the aux solve returned garbage and the outer Krylov
        # stalled at residual ~1 with no error. Inverting the FULL complex A_G fixes both; on the eddy case
        # it is algebraically identical to the old form (solving [[0,-R],[R,0]] gives exactly -j·R⁻¹).
        # ... UNLESS the aux takes complex directly, in which case the reformulation is not merely
        # unnecessary but actively harmful. The 2n block is skew-dominated by construction -- with the
        # mass term jω(σ+ε) dominating, ‖A-Aᵀ‖/‖A‖ measures 2.0 -- and algebraic multigrid falls over
        # on it: smoothed aggregation diverged to 1e+20, `air_solver` made no progress at all, and
        # Ruge-Stuben returned NaN, on the very blocks whose COMPLEX form is exactly complex-symmetric
        # (8e-17) and converges in 5-7 iterations. A complex-capable aux therefore gets the complex
        # blocks untouched; a real-only one (AmgX) still gets the 2n form.
        aux_takes_complex = bool(getattr(self.aux, "complex_ok", False))

        def _real_block(M):  # complex M -> (LinearOperator, scipy CSR) of its real-equivalent 2n block
            return to_op(sp.bmat([[M.real, -M.imag], [M.imag, M.real]]).tocsr())

        block = to_op if aux_takes_complex else _real_block
        b_ops, b_csrs, nvs = [], [], []
        for P in Pi_sp:
            Ap = pin0((P.T @ A_sp @ P).tocsr())
            op, csr = block(Ap)
            b_ops.append(op)
            b_csrs.append(csr)
            nvs.append(Ap.shape[0])
        A_Gp = pin0(A_G)
        rg_op, rg_csr = block(A_Gp)
        return {
            "complex": True,
            "aux_complex": aux_takes_complex,
            "rg_op": rg_op,
            "rg_csr": rg_csr,
            "rg_nv": int(A_Gp.shape[0]),
            "b_ops": b_ops,
            "b_csrs": b_csrs,
            "nvs": nvs,
        }

    def _aux_solve_builder(self):
        """Return ``make(op, csr) -> (rhs -> op⁻¹ rhs)``, called ONCE per aux block so its setup (a
        factorization / AMG hierarchy) is amortized across every preconditioner apply — a preconditioner's
        whole point. The previous default called a stateless ``lu()`` *per apply*, re-factoring each aux
        block on **every** Krylov iteration (a fresh cuSolver sparse-LU factorization each iteration — the
        AMS setup wall)."""
        aux = self.aux
        if aux is None:
            # Default: freeze a host SuperLU factorization of each (concrete, host-assembled) aux block ONCE
            # and apply it per iteration via a host callback — no refactor, and off the GPU so it avoids
            # cuSolver entirely. The preconditioner need not be differentiable: the outer solve's
            # ``custom_linear_solve`` takes the gradient through the operator A, never through M⁻¹, so it
            # only ever *calls* this (forward), which a ``pure_callback`` supports.
            import jax
            import numpy as np
            import scipy.sparse.linalg as _spla

            def make(op, csr):
                lu = _spla.splu(csr.tocsc())  # factor ONCE (SuperLU)
                n = int(csr.shape[0])

                def solve(rhs):
                    rhs = jnp.asarray(rhs)
                    return jax.pure_callback(
                        lambda b: np.asarray(lu.solve(np.asarray(b)), dtype=np.asarray(b).dtype),
                        jax.ShapeDtypeStruct((n,), rhs.dtype),
                        rhs,
                    )

                return solve

            return make

        if isinstance(aux, _AMG):
            # ``aux=jno.precond.amg()`` -- an AMG hierarchy AS the auxiliary solver. Built once per
            # block here (``make`` is called once per block) and applied per iteration, which is the
            # whole point: a one-shot solver would re-run setup on every Krylov iteration.
            #
            # This is the auxiliary that makes complex AMS cheap. A plain Krylov aux stalls -- cg at
            # 7.6e-4 and bicgstab at 5.8e-4 relative residual, IDENTICALLY at aux tolerances 1e-3 and
            # 1e-6, so the tolerance was never the limiter -- while AMG on the (complex, unreformulated)
            # blocks converges in 5-7 iterations. pyamg's own solve is used rather than the JAX V-cycle
            # in ``utils.solver.amg`` because that one smooths with Chebyshev, which needs real spectral
            # bounds and has no meaning on a complex-symmetric spectrum. MEASURED, on an A-V block with
            # sigma = 0 outside the conductor: the auxiliaries span the whole first quadrant --
            # arg(lambda) 0.1 to 89.7 degrees for G^T A G and 0.2 to 89.8 for each Pi^T A Pi -- so there
            # is no real interval to fit. (With sigma UNIFORM and w*sigma >> nu they collapse onto a ray,
            # ~1 degree of spread, which is why a uniform-sigma test says nothing about this.) Making the
            # V-cycle usable here needs an ellipse-based complex polynomial instead: Manteuffel, *The
            # Tchebychev iteration for nonsymmetric linear systems*, Numer. Math. 28 (1977). Applied through
            # ``pure_callback`` for the same reason the default SuperLU aux is: forward-only, and the
            # preconditioner is never differentiated.
            import jax
            import numpy as np

            def make(op, csr):
                import pyamg

                # A REAL auxiliary has a real spectrum, so the Chebyshev objection above does not
                # apply and the JAX V-cycle can carry it -- which keeps the whole AMS apply on device:
                # G and the Pi blocks are already BCOO, the hierarchy levels are BCOO, and nothing in
                # the Krylov loop has to reach the host. Only the one-off setup (aggregation, the
                # triple products) stays there, which is per-operator rather than per-iteration.
                #
                # An INEXACT auxiliary is enough, measured on a real curl-curl + mass problem: an
                # exact SuperLU auxiliary takes 27 CG iterations, two V-cycles take 27, and a SINGLE
                # V-cycle takes 27. (Inexact KRYLOV auxiliaries are the ones that stall -- cg at
                # 7.6e-4 and bicgstab at 5.8e-4 -- so this is not the same question.)
                if not np.iscomplexobj(csr.data):
                    from .utils.solver.amg import build_hierarchy, vcycle_apply

                    levels = build_hierarchy(csr.tocsr(), max_levels=aux.max_levels)

                    def solve_real(rhs):
                        x = vcycle_apply(levels, jnp.asarray(rhs))
                        for _ in range(max(int(aux.cycles) - 1, 0)):  # extra cycles: correct the residual
                            x = x + vcycle_apply(levels, jnp.asarray(rhs) - op @ x)
                        return x

                    return solve_real

                ml = pyamg.smoothed_aggregation_solver(csr.tocsr(), max_levels=aux.max_levels)
                n = int(csr.shape[0])

                def solve(rhs):
                    rhs = jnp.asarray(rhs)

                    def _host(b):
                        b = np.asarray(b)
                        x = ml.solve(b, tol=1e-10, maxiter=50, accel="bicgstab")
                        return np.asarray(x, dtype=b.dtype)

                    return jax.pure_callback(_host, jax.ShapeDtypeStruct((n,), rhs.dtype), rhs)

                return solve

            return make

        # Any other ``aux`` (e.g. ``jno.solve.amg()`` → jaxamg on the GPU, or a custom callable): call it
        # per apply. A one-shot solver re-runs its own setup each iteration; to amortize a jaxamg
        # hierarchy across the Krylov loop it needs a setup-once handle (a jaxamg AmgX setup/solve split) —
        # wire that here once exposed. ``csr`` is unused on this path.
        return lambda op, csr: lambda rhs: aux(op, rhs)

    def materialize(self, ctx: PrecondContext):
        f = self._frozen if self._frozen is not None else self._assemble_aux(ctx)  # frozen (built) or eager

        G, Pis = self._G, self._Pis
        GT, PisT = G.T, [P.T for P in Pis]
        dinv = 1.0 / ctx.diag()  # Jacobi smoother — the traced diagonal carries A(θ) under a trace
        make = self._aux_solve_builder()  # build each aux solver's setup ONCE, reuse across every apply

        if not f["complex"]:
            solve_g = make(f["g_op"], f["g_csr"])
            solve_ps = [make(op, csr) for op, csr in zip(f["p_ops"], f["p_csrs"])]

            def apply(r):
                x = dinv * r
                x = x + G @ solve_g((GT @ r).at[0].set(0.0))  # gradient-space correction
                for P, PT, solve_p in zip(Pis, PisT, solve_ps):
                    x = x + P @ solve_p((PT @ r).at[0].set(0.0))  # solenoidal correction (per component)
                return x
        else:
            rg_nv, nvs = f["rg_nv"], f["nvs"]
            solve_rg = make(f["rg_op"], f["rg_csr"])
            solve_bs = [make(op, csr) for op, csr in zip(f["b_ops"], f["b_csrs"])]

            if f.get("aux_complex"):
                # The aux takes complex, so the auxiliaries were never split -- no packing, and the
                # solve sees the complex-symmetric operator multigrid can actually coarsen.
                def apply(r):
                    x = dinv * r
                    x = x + G @ solve_rg((GT @ r).at[0].set(0.0))
                    for P, PT, solve_b in zip(Pis, PisT, solve_bs):
                        x = x + P @ solve_b((PT @ r).at[0].set(0.0))
                    return x

            else:

                def apply(r):
                    x = dinv * r
                    rg = (GT @ r).at[0].set(0.0)
                    sg = solve_rg(jnp.concatenate([jnp.real(rg), jnp.imag(rg)]))  # complex A_G as a real 2n block
                    x = x + G @ (sg[:rg_nv] + 1j * sg[rg_nv:])  # gradient-space correction
                    for P, PT, solve_b, nv in zip(Pis, PisT, solve_bs, nvs):
                        rp = (PT @ r).at[0].set(0.0)
                        s = solve_b(jnp.concatenate([jnp.real(rp), jnp.imag(rp)]))  # real 2n-block solve
                        x = x + P @ (s[:nv] + 1j * s[nv:])
                    return x

        return PrecondApplier(apply)

    def __repr__(self):
        return f"jno.precond.ams(aux={self.aux!r})"


def ams(*, aux=None) -> _AMS:
    """**AMS** — the auxiliary-space Maxwell preconditioner for H(curl) (Nédélec/N1E) curl-curl
    systems (Hiptmair & Xu, *SIAM J. Numer. Anal.* 45(6):2483, 2007; Kolev & Vassilevski,
    *J. Comput. Math.* 27(5):604, 2009).

    Plain point/AMG smoothing cannot damp the huge **gradient near-null-space** of a curl-curl
    operator (``curl∘grad = 0``), so its condition number leaks into the iteration count. AMS adds
    two corrections on cheaper **nodal** auxiliary problems — one on the discrete-gradient space
    ``G``, one on the vector-nodal space ``Π`` — restoring near mesh-independent convergence::

        M⁻¹ r = D⁻¹ r  +  G (GᵀAG)⁻¹ Gᵀ r  +  Σ_α Π_α (Π_αᵀAΠ_α)⁻¹ Π_αᵀ r

    ``G`` and ``Π`` come from the N1E edge topology (:mod:`jno.utils.solver.ams`); the auxiliary
    operators are assembled once on the host from the concrete matrix and solved with ``aux``.
    **Default** (``aux=None``): each auxiliary block is **factored once** (host SuperLU) and that factor
    is reused across every Krylov iteration via a host callback — the setup is amortized (the whole point
    of a preconditioner) and runs off the GPU. The previous default re-factored on *every* iteration (a
    fresh cuSolver sparse-LU per iteration — impractically slow at scale). Pass any ``jno.solve`` solver
    as ``aux`` to override. Because the auxiliary problems are ordinary **nodal Poisson-like** systems, an
    algebraic-multigrid ``aux`` makes the whole preconditioner scalable at ``O(n)``: on the GPU pass
    ``aux=jno.solve.amg()`` (NVIDIA AmgX via ``jaxamg``), which caches each auxiliary hierarchy per
    operator (``jaxamg.with_cache(A, is_symmetric=…)``) — the AMS applier calls ``aux`` with the same
    operator every iteration, so caching gives setup-once with a pure-JAX apply.

    **Outer solver.** An *exact* ``aux`` (``lu``) is a fixed linear map, so it pairs with ``cg``; an
    *inexact/iterative* ``aux`` (multigrid, an inexact ``cg``) is a **variable** preconditioner and
    needs a **flexible** outer solver — :func:`jno.solve.fgmres` (real) — or ``cg`` stalls.

    **Differentiable / traced solves.** The host aux-assembly cannot run under a trace. A **forward**
    (concrete) solve freezes the auxiliaries automatically at compose time, so it is already
    differentiable-ready — nothing to do. For a solve whose **operator itself is traced** — a ``jit`` /
    ``vmap``, or a **parametric-inverse** design loop where ``A(θ)`` carries a ``jno.np.parameter`` —
    freeze once from a **concrete reference** and reuse it::

        spec = jno.precond.ams().build(fem0)          # fem0 = the fem at your reference parameters θ₀
        node = fem_of(theta).solve(precond=spec)      # parametric solve; node is differentiable

    The frozen preconditioner stays valid as ``A(θ)`` drifts (speed degrades, correctness never), and
    ``∂/∂θ`` flows through the **operator** by implicit differentiation — never through the
    preconditioner (a preconditioner cannot change the solution, so differentiating its setup would be
    pure waste). You cannot auto-freeze from the parametric fem itself because ``θ₀`` is only resolved
    at solve time; the one-line concrete reference is that choice made explicit.

    The same spec handles the **real** curl-curl+mass, the **complex** eddy operator ``νK + jωσM``, and
    a **driven time-harmonic** ``K − k₀²εM`` with absorption — dtype follows the assembled matrix; pair
    it with :func:`jno.solve.gmres` (complex-correct) for the complex cases. For a complex operator
    **every** auxiliary is reformulated as its real-equivalent ``2n`` block ``[[Re,-Im],[Im,Re]]``, which
    a **real-only** ``aux`` (AmgX/multigrid) solves *exactly*: the gradient block ``GᵀA_cG`` and each
    solenoidal block ``ΠᵀA_cΠ`` alike (non-symmetric → the ``aux`` must be non-symmetric-capable, e.g.
    AMG with ``is_symmetric=False``).

    The gradient block once kept only ``Im A_G``, on the eddy-case reasoning that ``GᵀKG = 0`` makes
    ``A_G = jω·R`` pure imaginary, so ``A_G⁻¹ = -j·(Im A_G)⁻¹``. That fails for a **driven wave**
    problem twice over: ``Re A_G = -k₀²·GᵀεMG ≠ 0``, and with **surface-only** absorption (an impedance
    / first-order absorbing BC and no volume loss) ``Im A_G`` is a *boundary* mass — identically zero on
    every interior node, hence singular, so the aux solve returned garbage and the outer Krylov stalled
    at residual ~1 with **no error**. Inverting the full complex ``A_G`` fixes both, and on the eddy case
    is algebraically identical to the old form (solving ``[[0,-R],[R,0]]`` gives exactly ``-j·R⁻¹``).

    Complex GMRES is **not flexible**, so solve a complex ``aux`` **tightly** (near-exact) — a strong
    multigrid or ``lu``, not a single V-cycle, which is a *variable* preconditioner and will stall.

    Requirements & scope:

    * The operator must be **coercive on the gradient space** — a bare curl-curl is singular there;
      a mass term (conductivity, or the σ=0-in-air **ε-gauge** ``jω·ε·⟨A,v⟩``) is what makes
      ``GᵀAG`` invertible. The spec raises if that term is missing.
    * ``G``/``Π`` are built from the **full** edge topology, so this targets weak/penalty (PEC-style)
      boundary terms; Dirichlet-**eliminated** DOFs would need row-masking — out of scope here.
    """
    return _AMS(aux)


class _JaxAMG(_Spec):
    """Spec for the GPU AMG preconditioner via jaxamg (NVIDIA AmgX); see :func:`jaxamg`."""

    complex_ok = False  # AmgX is real-only -- a complex operator must take the 2n real-equivalent path

    def __init__(self, config, symmetric=True):
        self.config = config
        self.symmetric = bool(symmetric)

    def materialize(self, ctx: PrecondContext):
        from .solve import _require_jaxamg  # reuse the lazy import + install-requirements error

        A = ctx.A.bcoo
        if A is None:
            raise ValueError(
                "jno.precond.jaxamg needs an assembled (sparse) operator — it cannot build an AMG "
                "hierarchy from a matrix-free operator."
            )
        jax_amg = _require_jaxamg()
        cfg = dict(self.config) if self.config is not None else {"solver": "AMG"}
        apply = jax_amg.make_preconditioner(A, config=cfg)  # build-once M⁻¹ apply (single AMG cycle)
        if self.symmetric:
            return PrecondApplier(lambda v: apply(v))  # .T reuses the applier -- right for SPD
        # The reverse pass of a differentiable solve preconditions A^T and needs M^T (see
        # PrecondApplier): an AMG hierarchy is not structurally transposable, so build a SECOND
        # hierarchy on the transposed matrix. Costs one extra setup; without it the adjoint Krylov
        # solve of a non-symmetric operator runs effectively unpreconditioned.
        import jax.experimental.sparse as _js

        At = _js.BCOO((jnp.asarray(A.data), jnp.asarray(A.indices)[:, ::-1]), shape=(A.shape[1], A.shape[0]))
        apply_t = jax_amg.make_preconditioner(At, config=cfg)
        return PrecondApplier(lambda v: apply(v), lambda v: apply_t(v))

    def __repr__(self):
        return f"jno.precond.jaxamg(symmetric={self.symmetric})"


def jaxamg(*, config: "dict | None" = None, symmetric: bool = True) -> _JaxAMG:
    """GPU AMG **preconditioner** via jaxamg (NVIDIA AmgX wrapped as a JAX primitive) — the
    on-device counterpart of :func:`amg`, and the natural smoother for large elliptic blocks or the
    auxiliary nodal solves of an H(curl) AMS preconditioner on the GPU.

    Builds the AMG hierarchy with ``jaxamg.make_preconditioner`` and applies a single cycle as
    ``M⁻¹`` — a proper build-once/apply-many preconditioner (unlike ``jno.precond.inner(jno.solve.amg())``,
    which re-solves each application). Wrap in ``.cached()`` to reuse the hierarchy across solves::

        fem.solve(linear=jno.solve.fgmres(), precond=jno.precond.jaxamg().cached())

    ``config`` is a full AmgX-format dict (default ``{"solver": "AMG"}``). Needs an **assembled**
    operator. Optional dependency — jaxamg (AmgX 2.5+, CUDA 12+, mpi4py/mpi4jax) is imported lazily.

    ``symmetric=False`` builds a **second hierarchy on** ``A^T`` so the adjoint (reverse-mode) solve
    is preconditioned too — an AMG hierarchy is not structurally transposable, and without this the
    reverse pass of a differentiable non-symmetric solve runs effectively unpreconditioned (see
    ``PrecondApplier``). Leave the default for SPD operators, where one hierarchy serves both
    directions. Real-valued operators only — AmgX has no complex mode.

    Reference: Liu, Fan & Wang, arXiv:2606.09001 (2026), wrapping NVIDIA AmgX (Naumov et al.,
    *SIAM J. Sci. Comput.* 37(5), 2015).
    """
    return _JaxAMG(config, symmetric)


_MISS = object()  # sentinel so the first materialize always builds


class _Cached(_Spec):
    """Spec that memoises another spec's setup across solves; see :func:`cached`."""

    def __init__(self, spec, refresh):
        self.spec = spec
        # False → frozen | True → rebuild on sparsity change | int k → rebuild every k-th
        # materialization | callable ctx→key
        self.refresh = refresh
        self._applier = None
        self._key = _MISS
        self._count = 0  # materializations since construction, for the int-k policy

    # `_Cached` wraps another spec's SETUP, not its nature. These two say what the inner spec IS --
    # `complex_native` selects whether the solve runs on the genuinely complex n-sized operator or on
    # the fused real-equivalent 2n block, and `complex_ok` whether a spec can be applied to a complex
    # operator without reformulation -- so hiding them changes the ANSWER, not the speed. Left
    # undeclared, `ams().cached()` reported False and quietly took the 2n path, where the block slices
    # cover the wrong half of the operator and AMS is handed the skew-dominated block its own docs
    # record diverging to 1e+20 on. That made `.cached()` and complex-native mutually exclusive, which
    # is backwards: a complex block preconditioner is the one whose setup most wants reusing.
    #
    # Forwarded explicitly rather than through `__getattr__`, so only these descriptive flags pass and
    # the wrapper keeps its own behaviour (`traceable`, `key`, `prepare`, `materialize`).
    @property
    def complex_native(self):
        return bool(getattr(self.spec, "complex_native", False))

    @property
    def complex_ok(self):
        return bool(getattr(self.spec, "complex_ok", False))

    @property
    def traceable(self):
        """A **frozen** cache that has already built does no setup at all -- :meth:`materialize` hands
        back the stored applier -- so it can be materialised inside a trace, and the solve around it
        can be compiled.

        This is the case that matters for many-iteration work. ``.cached()`` is the recipe for a
        transient run or a Newton loop, where the same preconditioner serves every step; leaving it on
        the eager path meant dispatching the applier op by op on exactly the workloads with the most
        applications to dispatch. The first solve still runs eager (it has to, to build); every solve
        after it compiles.

        ``refresh=`` is excluded: those variants key on the operator and may rebuild through the inner
        spec, which for ``amg``/``ams``/``form`` is host-side work that no trace can run.
        """
        return self.refresh is False and self._applier is not None

    @property
    def key(self):
        """The stored applier IS the compilation, so its identity is the key. ``self`` holds it, so the
        ``id`` cannot be recycled while this spec is alive."""
        return (type(self), id(self._applier)) if self.traceable else None

    def cached(self, *, refresh=False):
        return self  # already cached — .cached() is idempotent (no double-wrapping)

    def prepare(self, fem):  # forward the eager (out-of-trace) build hook, if the inner spec has one
        prep = getattr(self.spec, "prepare", None)
        if callable(prep):
            prep(fem)

    def _key_of(self, ctx):
        if self.refresh is False:
            return "frozen"
        if isinstance(self.refresh, bool):  # True → key on the operator's shape + sparsity size
            A = ctx.A
            return (A.shape, int(A.bcoo.nse)) if A.bcoo is not None else (A.shape,)
        if isinstance(self.refresh, int):
            # Every k-th materialization: the operator's values drift step by step (a Newton loop, a
            # transient march driving one solve per step), so the hierarchy is rebuilt on a cadence
            # rather than frozen forever or rebuilt every time. In between, the stale setup is a
            # legitimate preconditioner: speed degrades gracefully, correctness never.
            return self._count // max(1, int(self.refresh))
        if callable(self.refresh):
            return self.refresh(ctx)
        raise TypeError(f"cached(refresh=...): expected bool, int, or callable, got {type(self.refresh).__name__}")

    def materialize(self, ctx: PrecondContext):
        key = self._key_of(ctx)
        self._count += 1
        if self._applier is None or key != self._key:
            self._applier = materialize_precond(self.spec, ctx)  # build the wrapped preconditioner once
            self._key = key
        return self._applier  # same applier object (and its .T) reused on later solves

    def __repr__(self):
        return f"jno.precond.cached({self.spec!r}, refresh={self.refresh}, built={self._applier is not None})"


def cached(spec, *, refresh=False):
    """Memoise any preconditioner's setup so it is built **once** and reused across solves — the
    plug-and-play way to amortise an expensive setup (a multigrid hierarchy, an assembled auxiliary
    operator, a jaxamg/AmgX coloring) over a frequency sweep, a Newton loop, or an inverse-problem
    optimisation, *regardless of which backend does the work*.

    Wraps any spec (``jacobi``, ``amg``, ``form``, a jaxamg-backed preconditioner, or a user
    ``ctx -> M⁻¹`` callable). ``refresh=False`` (default) freezes the setup from the first solve and
    reuses it forever — the standard frozen-preconditioner trade (a preconditioner only changes
    convergence *speed*, never the solution, so reusing a slightly-stale setup is always correct and
    usually cheap). ``refresh=True`` rebuilds when the operator's shape/sparsity changes (values may
    still drift under the frozen setup); an ``int k`` rebuilds every k-th materialization — the
    cadence policy for a Newton loop or transient march whose operator values drift step by step;
    pass a callable ``ctx -> hashable`` for a custom invalidation key. The wrapped spec's eager ``prepare(fem)`` hook (if any) is forwarded, so it composes with the
    ``jit``/``vmap``/parametric-inverse build-eagerly requirement unchanged.

    Reuse the SAME ``cached(...)`` object across the solves you want to share the setup::

        M = jno.precond.cached(jno.precond.amg())          # backend-agnostic — pyamg or jaxamg alike
        for f in freqs:
            u = build_fem(f).solve(linear=jno.solve.fgmres(), precond=M)   # hierarchy built once
    """
    return _Cached(spec, refresh)


def nystrom(*, rank: int = 20, mu: float | None = None, seed: int = 0) -> _Nystrom:
    r"""Randomized **Nyström** low-rank preconditioner for SPD operators — the rung between
    ``jacobi`` and a multilevel method.

    Frangella, Tropp & Udell, "Randomized Nyström Preconditioning", *SIAM J. Matrix Anal. Appl.*
    44(2), 2023 — Algorithm 2.1 (the stabilized sketch) and §3 / Definition 3.1 (the
    preconditioner ``P^{-1} = (lam_min+mu) U (diag(lam)+mu)^{-1} U^T + (I - U U^T)``).

    Sketches ``A`` against a random ``n x rank`` matrix — **exactly ``rank`` matvecs, no assembled
    matrix and no triangular solves** — and deflates the captured top of the spectrum. That is the
    part `jacobi` cannot reach: a diagonal preconditioner rescales, it cannot separate a few large
    outlying eigenvalues, which is precisely what stalls Krylov on FEM operators with a stiff
    coefficient contrast or a near-null-space. Unlike ILU it needs no factorization and no
    sequential sweep, so it is ``jit``/``vmap``-native and runs on GPU.

    ``rank`` is the number of eigenvalues captured (cost is linear in it); ``mu`` is the
    regularization, defaulting to the smallest captured eigenvalue so the low-rank and identity
    parts meet continuously. ``seed`` fixes the sketch, so a solve is reproducible.

    **SPD only.** The sketch takes a Cholesky of ``Omega^T A Omega``, so a non-symmetric or
    indefinite operator will produce NaNs rather than a wrong answer quietly — use ``jacobi`` or
    ``chebyshev`` for those.
    """
    return _Nystrom(rank, mu, seed)


def chebyshev(
    *,
    degree: int = 8,
    lmin: float | None = None,
    lmax: float | None = None,
    lmin_ratio: float = 1.0 / 30.0,
    safety: float = 1.05,
    bound_iters: int = 30,
) -> _Chebyshev:
    """Fixed-degree Chebyshev **polynomial** preconditioner ``M^{-1} = p_degree(A) ≈ A^{-1}``
    for SPD operators (Saad 2003, §12.3 / Golub & Varga 1961 — the same recurrence as
    ``jno.solve.chebyshev``, truncated at ``degree`` with no convergence test, which keeps the
    application a fixed *linear* map so it may precondition CG and MINRES).

    The GPU-era substitute for Gauss-Seidel/ILU smoothing: only matvecs and AXPYs — no
    reductions, no triangular solves — ``jit``- and ``vmap``-native.

    Spectrum bounds of ``A`` are taken from ``lmin``/``lmax`` when given, else **both** ends are
    measured by ``bound_iters`` steps of Lanczos (Lanczos 1950, §II — the extreme Ritz values of
    the tridiagonal), at the same one-matvec-per-step cost as the power iteration it replaces.
    This matters because the polynomial is a contraction only *inside* the interval it is fitted
    to: the historical ``lmin = lmin_ratio * lmax`` guess, when it lands above the true smallest
    eigenvalue, leaves the lowest modes outside that interval where the polynomial amplifies them
    instead of damping. Without the optional :mod:`matfree` package the guess is still the
    fallback (``lmin_ratio`` then applies).
    """
    return _Chebyshev(degree, lmin, lmax, lmin_ratio, safety, bound_iters)
