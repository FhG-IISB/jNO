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
)

__all__ = ["PrecondContext", "jacobi", "chebyshev", "form", "inner", "block_diag", "triangular", "amg", "ams"]


class _Jacobi:
    """Spec for the diagonal (Jacobi) preconditioner; see :func:`jacobi`."""

    def materialize(self, ctx: PrecondContext):
        d = ctx.diag()
        safe = jnp.where(jnp.abs(d) > 1e-30, d, 1.0)  # zero diagonals (saddle blocks) left unscaled
        inv = 1.0 / safe
        return PrecondApplier(lambda v: inv * v)  # diagonal => symmetric => M^T == M

    def __repr__(self):
        return "jno.precond.jacobi()"


class _Chebyshev:
    """Spec for the fixed-degree Chebyshev polynomial preconditioner; see :func:`chebyshev`."""

    def __init__(self, degree, lmin, lmax, lmin_ratio, safety, bound_iters):
        self.degree = degree
        self.lmin, self.lmax = lmin, lmax
        self.lmin_ratio, self.safety, self.bound_iters = lmin_ratio, safety, bound_iters

    def materialize(self, ctx: PrecondContext):
        from .utils.solver.krylov import chebyshev_apply, power_iteration_bound

        if ctx.A.shape is None and self.lmax is None:
            raise TypeError(
                "jno.precond.chebyshev on a matvec-only operator needs explicit spectrum bounds "
                "(lmin=, lmax=) — there is no assembled matrix to estimate them from."
            )
        hi = self.lmax
        if hi is None:
            n = ctx.A.shape[0]
            hi = self.safety * power_iteration_bound(ctx.A.mv, n, iters=self.bound_iters)
        lo = self.lmin if self.lmin is not None else self.lmin_ratio * hi
        # A^T shares the spectrum of A, so the same [lo, hi] bounds the transpose recurrence.
        return PrecondApplier(
            lambda v: chebyshev_apply(ctx.A.mv, v, lmin=lo, lmax=hi, degree=self.degree),
            lambda v: chebyshev_apply(ctx.A.T.mv, v, lmin=lo, lmax=hi, degree=self.degree),
        )

    def __repr__(self):
        return f"jno.precond.chebyshev(degree={self.degree})"


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


class _Form:
    """Spec assembling an auxiliary weak form as the preconditioner operator; see :func:`form`."""

    def __init__(self, terms, inner_solver, quad_degree):
        self.terms = list(terms)
        self.inner = inner_solver
        self.quad_degree = quad_degree
        self._op = None  # assembled once; the auxiliary operator is parameter-independent

    def prepare(self, fem):
        """Eager one-time assembly (called at compose time, OUTSIDE any trace — assembling a
        fresh ``jno.fem`` inside the Newton/Picard ``while_loop`` entangles with loop tracers)."""
        if self._op is None:
            from .utils.solver.solver_api import PrecondContext as _Ctx

            self._op = _Ctx(None, fem).assemble(self.terms, quad_degree=self.quad_degree)

    def materialize(self, ctx: PrecondContext):
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
        return f"jno.precond.form(<{len(self.terms)} terms>, inner={self.inner})"


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


class _InnerSolve:
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


class _BlockDiag:
    def __init__(self, pairs):
        self.pairs = pairs

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


class _Triangular:
    def __init__(self, pairs):
        self.pairs = pairs

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
    e.g. ``jno.precond.inner(jno.solve.cg(tol=1e-2))`` (inexact block solve),
    ``jno.precond.chebyshev(...)`` (polynomial), or ``jno.precond.form([...])`` (auxiliary
    operator — for Stokes the classic pressure choice is the viscosity-weighted **mass matrix**
    ``form([(1/mu) * pi * qi])`` as the Schur-complement approximation; Elman, Silvester & Wathen,
    *Finite Elements and Fast Iterative Solvers*, 2nd ed., OUP 2014, §9.2). With inexact
    (iterative) block solves the outer Krylov must be flexible: ``jno.solve.fgmres()``."""
    return _Triangular(list(pairs))


class _AMG:
    """Spec for the hybrid (pyamg-setup / pure-JAX-apply) AMG preconditioner; see :func:`amg`."""

    def __init__(self, cycles, max_levels, coarse_size, smoother_degree):
        self.cycles = cycles
        self.max_levels = max_levels
        self.coarse_size = coarse_size
        self.smoother_degree = smoother_degree
        self._levels = None

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
        from .utils.solver.amg import vcycle_apply

        if self._levels is None:
            A = ctx.A.bcoo if ctx.A.bcoo is not None else ctx.A.dense()
            self.build(A)  # raises with .build() guidance when A is traced
        levels, cycles = self._levels, self.cycles
        A_op = ctx.A

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

    The setup runs **once on the host** (eagerly) and freezes fixed-pattern level operators; the
    per-application V-cycle is then ``jit``/``vmap``-native and a fixed *linear* map, so it may
    precondition ``cg``/``minres`` as well as ``bicgstab``/``fgmres``. The mesh-independent
    convergence of multigrid makes this *the* preconditioner for large elliptic blocks — heat,
    diffusion, elasticity, the (Picard-lagged) velocity block of a saddle system inside
    :func:`triangular`.

    Inside traced contexts (jit, vmap, a parametric inverse solve) call ``spec.build(fem.A)``
    once, eagerly, first; the frozen hierarchy is a legitimate preconditioner while operator
    values change (speed degrades gracefully, correctness never). pyamg is imported lazily —
    without it, a clear ``ImportError`` explains the install. On a matvec-only sub-block the
    matrix is recovered via the (dense) block view — fine for moderate blocks; pass a pre-built
    spec for very large ones.
    """
    return _AMG(cycles, max_levels, coarse_size, smoother_degree)


class _AMS:
    """Spec for the H(curl) auxiliary-space Maxwell (AMS) preconditioner; see :func:`ams`."""

    def __init__(self, aux):
        self.aux = aux  # jno.solve LinearSolver for the nodal auxiliary solves (None -> lu())
        self._G = None  # discrete gradient (node->edge), built once from the mesh topology
        self._Pis = None  # (Π_x, Π_y, Π_z) nodal->edge vector interpolation

    def prepare(self, fem):
        """Eager one-time build of the mesh-only transfer operators G, Π (parameter-independent)."""
        self._transfer(fem.domain)

    def _transfer(self, domain):
        from .utils.solver.ams import discrete_gradient, nodal_vector_interpolation

        topo = domain._fem_nonnodal_topology
        self._G = discrete_gradient(topo)
        self._Pis = nodal_vector_interpolation(topo)

    def materialize(self, ctx: PrecondContext):
        import numpy as np
        import scipy.sparse as sp
        from jax.experimental import sparse as jsp

        from . import solve as _solve
        from .utils.solver.solver_api import LinearOperator

        if self._G is None:
            if ctx.fem is None:
                raise TypeError(
                    "jno.precond.ams needs the owning FEM to read the N1E edge topology — "
                    "use it via fem.solve(precond=jno.precond.ams()), not on a bare operator."
                )
            self._transfer(ctx.fem.domain)
        aux = self.aux if self.aux is not None else _solve.lu()

        A = ctx.A.bcoo if ctx.A.bcoo is not None else ctx.A.dense()
        try:
            A_sp = sp.csr_matrix(np.asarray(A.todense() if hasattr(A, "todense") else A))
        except Exception as e:  # a traced operator (parametric/complex-inverse) can't be host-assembled
            raise RuntimeError(
                "jno.precond.ams assembles the nodal auxiliary operators on the host from a concrete "
                "matrix; it does not run under a trace (jit/vmap/parametric-inverse solve)."
            ) from e
        G_sp = sp.csr_matrix(np.asarray(self._G.todense()))
        Pi_sp = [sp.csr_matrix(np.asarray(P.todense())) for P in self._Pis]

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

        def to_op(M):  # scipy -> jno LinearOperator over a BCOO
            coo = pin0(M).tocoo()
            idx = jnp.asarray(np.stack([coo.row, coo.col], axis=1))
            return LinearOperator(jsp.BCOO((jnp.asarray(coo.data), idx), shape=coo.shape))

        # G's constant null-space (G·1 = 0) is exact; each Π_α's constant mode is redundant with it
        # (Π_α·1 = G·coordₐ is a gradient), so pinning node 0 in every auxiliary operator is correct.
        g_op = to_op(A_G)
        p_ops = [to_op((P.T @ A_sp @ P).tocsr()) for P in Pi_sp]

        G, Pis = self._G, self._Pis
        GT, PisT = G.T, [P.T for P in Pis]
        dinv = 1.0 / ctx.diag()  # Jacobi smoother (complex diagonal on a complex operator)

        def apply(r):
            x = dinv * r
            x = x + G @ aux(g_op, (GT @ r).at[0].set(0.0))  # gradient-space correction
            for P, PT, p_op in zip(Pis, PisT, p_ops):
                x = x + P @ aux(p_op, (PT @ r).at[0].set(0.0))  # solenoidal correction (per component)
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
    operators are assembled once on the host from the concrete matrix and solved with ``aux`` —
    **any** ``jno.solve`` solver (default :func:`jno.solve.lu`). Passing a multigrid inner solver
    makes the whole preconditioner scalable (the auxiliary problems are ordinary nodal Poisson-like
    systems). The same spec handles the **real** curl-curl+mass and the **complex** eddy operator
    ``νK + jωσM`` — dtype follows the assembled matrix; pair it with :func:`jno.solve.gmres`
    (complex-correct) for the eddy case.

    Requirements & scope:

    * The operator must be **coercive on the gradient space** — a bare curl-curl is singular there;
      a mass term (conductivity, or the σ=0-in-air **ε-gauge** ``jω·ε·⟨A,v⟩``) is what makes
      ``GᵀAG`` invertible. The spec raises if that term is missing.
    * ``G``/``Π`` are built from the **full** edge topology, so this targets weak/penalty (PEC-style)
      boundary terms; Dirichlet-**eliminated** DOFs would need row-masking — out of scope here.
    """
    return _AMS(aux)


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
    reductions, no triangular solves — ``jit``- and ``vmap``-native. Spectrum bounds of ``A``
    are taken from ``lmin``/``lmax`` when given, else estimated by power iteration (``safety``
    inflation, ``lmin = lmin_ratio * lmax``).
    """
    return _Chebyshev(degree, lmin, lmax, lmin_ratio, safety, bound_iters)
