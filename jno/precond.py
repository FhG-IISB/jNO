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

__all__ = ["PrecondContext", "jacobi", "chebyshev", "form", "inner", "block_diag", "triangular", "amg", "cached"]


class _Spec:
    """Base for ``jno.precond.*`` specs — gives every preconditioner a fluent ``.cached()``."""

    def cached(self, *, refresh: bool = False):
        """Wrap this preconditioner so its setup is built **once** and reused across solves — the
        fluent form of :func:`cached`. E.g. ``jno.precond.amg().cached()``. ``refresh`` controls
        invalidation (``False`` frozen, ``True`` on shape/sparsity change, or a ``ctx -> key`` callable)."""
        return _Cached(self, refresh)


class _Jacobi(_Spec):
    """Spec for the diagonal (Jacobi) preconditioner; see :func:`jacobi`."""

    def materialize(self, ctx: PrecondContext):
        d = ctx.diag()
        safe = jnp.where(jnp.abs(d) > 1e-30, d, 1.0)  # zero diagonals (saddle blocks) left unscaled
        inv = 1.0 / safe
        return PrecondApplier(lambda v: inv * v)  # diagonal => symmetric => M^T == M

    def __repr__(self):
        return "jno.precond.jacobi()"


class _Chebyshev(_Spec):
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


class _Form(_Spec):
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


class _AMG(_Spec):
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


_MISS = object()  # sentinel so the first materialize always builds


class _Cached(_Spec):
    """Spec that memoises another spec's setup across solves; see :func:`cached`."""

    def __init__(self, spec, refresh):
        self.spec = spec
        self.refresh = refresh  # False → frozen | True → rebuild on sparsity change | callable ctx→key
        self._applier = None
        self._key = _MISS

    def cached(self, *, refresh: bool = False):
        return self  # already cached — .cached() is idempotent (no double-wrapping)

    def prepare(self, fem):  # forward the eager (out-of-trace) build hook, if the inner spec has one
        prep = getattr(self.spec, "prepare", None)
        if callable(prep):
            prep(fem)

    def _key_of(self, ctx):
        if self.refresh is False:
            return "frozen"
        if callable(self.refresh):
            return self.refresh(ctx)
        A = ctx.A  # refresh=True → key on the operator's shape + sparsity size
        return (A.shape, int(A.bcoo.nse)) if A.bcoo is not None else (A.shape,)

    def materialize(self, ctx: PrecondContext):
        key = self._key_of(ctx)
        if self._applier is None or key != self._key:
            self._applier = materialize_precond(self.spec, ctx)  # build the wrapped preconditioner once
            self._key = key
        return self._applier  # same applier object (and its .T) reused on later solves

    def __repr__(self):
        return f"jno.precond.cached({self.spec!r}, refresh={self.refresh}, built={self._applier is not None})"


def cached(spec, *, refresh: bool = False):
    """Memoise any preconditioner's setup so it is built **once** and reused across solves — the
    plug-and-play way to amortise an expensive setup (a multigrid hierarchy, an assembled auxiliary
    operator, a jaxamg/AmgX coloring) over a frequency sweep, a Newton loop, or an inverse-problem
    optimisation, *regardless of which backend does the work*.

    Wraps any spec (``jacobi``, ``amg``, ``form``, a jaxamg-backed preconditioner, or a user
    ``ctx -> M⁻¹`` callable). ``refresh=False`` (default) freezes the setup from the first solve and
    reuses it forever — the standard frozen-preconditioner trade (a preconditioner only changes
    convergence *speed*, never the solution, so reusing a slightly-stale setup is always correct and
    usually cheap). ``refresh=True`` rebuilds when the operator's shape/sparsity changes (values may
    still drift under the frozen setup); pass a callable ``ctx -> hashable`` for a custom invalidation
    key. The wrapped spec's eager ``prepare(fem)`` hook (if any) is forwarded, so it composes with the
    ``jit``/``vmap``/parametric-inverse build-eagerly requirement unchanged.

    Reuse the SAME ``cached(...)`` object across the solves you want to share the setup::

        M = jno.precond.cached(jno.precond.amg())          # backend-agnostic — pyamg or jaxamg alike
        for f in freqs:
            u = build_fem(f).solve(linear=jno.solve.fgmres(), precond=M)   # hierarchy built once
    """
    return _Cached(spec, refresh)


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
