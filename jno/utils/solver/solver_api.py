"""Slot-based solver API: the contracts behind ``fem.solve(x0=, nonlinear=, linear=, precond=)``.

Design (see ``plans/fem-solver-api.md``): the FEM solver space factorises into four orthogonal
slots -- warm start, linearization driver, inner linear solve, preconditioner. Every slot takes a
**callable** (no strings); the shipped defaults live in the ``jno.solve`` / ``jno.precond``
namespaces and are pure JAX (jit- + vmap-native), reusing ``jax.scipy.sparse.linalg`` and the
existing ``sparse_lu_solve`` rather than re-implementing solvers. ``fem.solve(solve_fn=...)``
remains the total override and is unchanged.

Contracts
---------
* linear solver: ``fn(A, b, *, M=None, x0=None) -> x`` with ``A`` a :class:`LinearOperator`
  (``.mv(v)``, ``.T.mv(v)``, ``.diag()``, ``.bcoo``, ``.dense()``). Pure-JAX implementations
  inherit ``jit``/``vmap``/AD; the ``jax.scipy`` Krylov wrappers are built on
  ``lax.custom_linear_solve`` upstream, so they are implicitly differentiable already.
* nonlinear driver: ``fn(residual_fn, u0, *, linear_solve=None) -> u`` where ``linear_solve`` is
  a matrix-free ``(matvec, rhs) -> x`` built from the ``linear`` slot.
* preconditioner spec: an object with ``materialize(ctx) -> (v -> M^{-1} v)`` (or a bare
  ``ctx -> apply`` callable). Specs are declarative -- jno materializes them at solve time with a
  :class:`PrecondContext` carrying the assembled operator. A preconditioner only changes
  convergence *speed*, never the converged solution, so it needs **no** gradient path
  (the differentiability firewall of ``custom_linear_solve`` / ``custom_root``).
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from .linear import matrix_diagonal

__all__ = [
    "LinearOperator",
    "LinearSolver",
    "NonlinearSolver",
    "PrecondApplier",
    "PrecondContext",
    "materialize_precond",
    "prepare_precond",
    "compose_linear_solve_fn",
    "compose_nonlinear_solve_fn",
    "compose_transient_step_solvers",
]


class PrecondApplier:
    """A preconditioner application ``v -> M^{-1} v`` that also carries its **transpose** applier
    ``.T`` (``v -> M^{-T} v``).

    The reverse pass of a differentiable solve preconditions ``A^T`` and must use ``M^T``, not
    ``M``: a preconditioner never changes the converged solution, so reusing ``M`` is *correct*,
    but for a non-symmetric ``M`` (block-triangular Schur, ILU, ...) ``M`` approximates ``A^{-1}``
    and is near-useless for ``A^T`` -- the adjoint Krylov solve then runs almost unpreconditioned
    and dominates reverse-mode cost. Preconditioner specs build the transpose applier structurally
    (transpose each block, swap the substitution direction, transpose the coupling matvecs), which
    ``jax.linear_transpose`` cannot do through inner iterative/direct sub-solves.

    ``fwd`` is the forward applier; ``t`` the transpose applier, or ``None`` for a **symmetric**
    preconditioner (Jacobi, an SPD auxiliary form) -- then ``.T`` is the applier itself. A bare
    callable preconditioner (no ``.T``) still works: callers fall back to reusing ``M``.
    """

    __slots__ = ("_fwd", "_t")

    def __init__(self, fwd, t=None):
        self._fwd = fwd
        self._t = t

    def __call__(self, v):
        return self._fwd(v)

    @property
    def T(self) -> "PrecondApplier":
        return self if self._t is None else PrecondApplier(self._t, self._fwd)


class LinearOperator:
    """Uniform handle over an assembled operator (BCOO, dense array, or bare matvec).

    Gives every linear solver one interface regardless of the storage the assembler produced:
    ``.mv(v)`` (also ``@``), lazy transpose ``.T``, ``.diag()``, ``.dense()``, and ``.bcoo``
    (``None`` when not sparse). A matvec-only operator (``from_matvec``) supports ``mv`` and a
    transpose via ``jax.linear_transpose``; ``diag``/``dense`` raise -- direct solvers and
    diagonal preconditioners need an assembled matrix.
    """

    def __init__(self, A: Any, *, _transposed: bool = False):
        if callable(A) and not hasattr(A, "shape"):
            raise TypeError("Wrap a bare matvec with LinearOperator.from_matvec(fn).")
        self._A = A
        self._mv = None
        self._t_mv = None
        self._diag_fn = None
        self._dense_fn = None
        self._shape = None
        self._transposed = _transposed

    @classmethod
    def from_matvec(
        cls,
        mv: Callable,
        *,
        t_mv: Optional[Callable] = None,
        diag_fn: Optional[Callable] = None,
        dense_fn: Optional[Callable] = None,
        shape: Optional[tuple] = None,
        _transposed: bool = False,
    ) -> "LinearOperator":
        """Wrap a bare matvec. Optional hooks upgrade it: ``t_mv`` (transposed matvec — else
        derived exactly via ``jax.linear_transpose``), ``diag_fn``/``dense_fn`` (else those
        accessors raise), ``shape`` (else ``None``)."""
        op = cls.__new__(cls)
        op._A = None
        op._mv = mv
        op._t_mv = t_mv
        op._diag_fn = diag_fn
        op._dense_fn = dense_fn
        op._shape = shape
        op._transposed = _transposed
        return op

    @property
    def shape(self):
        if self._A is None:
            s = self._shape
        else:
            s = self._A.shape
        if s is None:
            return None
        return (s[1], s[0]) if self._transposed else tuple(s)

    def mv(self, v):
        if self._mv is not None:
            if self._transposed:
                if self._t_mv is not None:
                    return self._t_mv(v)
                # transpose of a linear map, via JAX's transposition (exact, not an approximation)
                (out,) = jax.linear_transpose(self._mv, jnp.zeros_like(v))(v)
                return out
            return self._mv(v)
        # ``v @ A`` is the transposed matvec for both BCOO and dense -- no transpose materialised
        return (v @ self._A) if self._transposed else (self._A @ v)

    __matmul__ = mv
    __call__ = mv

    @property
    def T(self) -> "LinearOperator":
        if self._mv is not None:
            op = LinearOperator.from_matvec(
                self._mv,
                t_mv=self._t_mv,
                diag_fn=self._diag_fn,
                dense_fn=self._dense_fn,
                shape=self._shape,
                _transposed=not self._transposed,
            )
            return op
        return LinearOperator(self._A, _transposed=not self._transposed)

    def diag(self):
        if self._A is None:
            if self._diag_fn is not None:
                return self._diag_fn()  # a (square) block's diagonal is transpose-invariant
            raise TypeError("LinearOperator.diag(): a matvec-only operator has no assembled diagonal.")
        return matrix_diagonal(self._A)  # transpose shares the diagonal

    def dense(self):
        if self._A is None:
            if self._dense_fn is not None:
                M = self._dense_fn()
                return M.T if self._transposed else M
            raise TypeError("LinearOperator.dense(): a matvec-only operator cannot densify.")
        M = self._A.todense() if hasattr(self._A, "todense") else jnp.asarray(self._A)
        return M.T if self._transposed else M

    @property
    def bcoo(self):
        return self._A if hasattr(self._A, "todense") else None


def _maybe_residual_check(op: LinearOperator, b, x, who: str, *, rtol: float = 1e-4):
    """Eager-only convergence guard: raise on a garbage solution when values are concrete.

    Under ``jit``/``vmap``/``grad`` (tracers) it is a no-op -- the check would either fail to
    concretise or break the transform; there the solver's own iteration cap is the guard.
    """
    if any(isinstance(v, jax.core.Tracer) for v in (x, b)):
        return x
    import numpy as np

    rel = float(jnp.linalg.norm(b - op.mv(x)) / (jnp.linalg.norm(b) + 1e-30))
    if not np.isfinite(rel) or rel > rtol:
        raise RuntimeError(
            f"jno.solve.{who} did not solve the system (relative residual {rel:.1e}); the problem may be "
            "singular/ill-posed or need a preconditioner. Try jno.solve.lu(), a precond= spec, or your own solve_fn."
        )
    return x


class LinearSolver:
    """A configured linear solver: ``solver(A, b, *, M=None, x0=None) -> x``.

    ``fn`` receives ``(op: LinearOperator, b, M, x0)``. ``traits`` documents transform support
    (``vmap: "native" | "sequential" | "no"``, ``jit: True | False``) so composition layers (and,
    later, the auto policy) can pick honestly instead of silently host-looping. ``jit=False`` marks
    a solver whose iteration cannot run *inside* a trace -- Chebyshev measures its spectrum bounds
    and branches on the measured values, which a tracer has no answer for.

    ``key`` is the solver's **value identity**: the constructor arguments that change what it does.
    It exists because the compiled slot path (:func:`compose_linear_solve_fn`) hands the spec to
    ``jax.jit`` as a static argument, and ``jax`` keys its compilation cache on ``hash``. Hashing by
    identity would mean ``fem.solve(linear=jno.solve.cg())`` -- the spec written inline, as the docs
    themselves write it -- recompiling on every call: measured 0.4 ms against 83.5 ms on a 513-DOF
    Poisson solve, i.e. far worse than never compiling. With a key, two equivalently configured
    specs are one cache entry.

    A spec with **no** key falls back to identity, and the composer then declines to compile at all.
    That default is deliberate: a key that omits a parameter would serve a cached solve configured
    the *other* way, which is a wrong answer, while no key merely forgoes a speed-up. So a new
    solver is slow until its key is written, never silently wrong.
    """

    def __init__(self, fn: Callable, *, name: str, traits: Optional[dict] = None, direct: bool = False, key: Any = None):
        self._fn = fn
        self.name = name
        self.traits = {"vmap": "native", "jit": True, **(traits or {})}
        self.direct = direct  # a direct solver ignores x0 and takes no preconditioner
        self.key = None if key is None else (type(self), name, key)

    def __call__(self, A, b, *, M=None, x0=None):
        op = A if isinstance(A, LinearOperator) else LinearOperator(A)
        b = jnp.asarray(b).reshape(-1)
        if self.direct and M is not None:
            raise ValueError(f"jno.solve.{self.name} is a direct solver -- it takes no preconditioner (precond=).")
        return self._fn(op, b, M=M, x0=x0)

    def __eq__(self, other):
        if self.key is None or not isinstance(other, LinearSolver) or other.key is None:
            return self is other
        return self.key == other.key

    def __hash__(self):
        return id(self) if self.key is None else hash(self.key)

    def __repr__(self):
        return f"jno.solve.{self.name}({', '.join(f'{k}={v}' for k, v in self.traits.items())})"


class NonlinearSolver:
    """A configured nonlinear driver: ``driver(residual_fn, u0, *, linear_solve=None, jacobian=None) -> u``.

    ``direct=True`` marks an **assembled-Jacobian, sparse-direct** Newton: it factorizes the assembled
    tangent (``jacobian=`` — a callable ``u -> BCOO``) each step instead of the matrix-free Krylov inner
    solve, so it is robust on indefinite/ill-conditioned systems (Taylor-Hood saddles, stiff drag). It
    composes only where the assembler provides that Jacobian (native nonlinear FEM / the transient
    stepper), which threads it in via ``jacobian=``.
    """

    def __init__(self, fn: Callable, *, name: str, traits: Optional[dict] = None, direct: bool = False):
        self._fn = fn
        self.name = name
        self.direct = direct
        self.traits = {"vmap": "native", "jit": True, **(traits or {})}

    def __call__(self, residual_fn, u0, *, linear_solve=None, jacobian=None):
        return self._fn(residual_fn, u0, linear_solve=linear_solve, jacobian=jacobian)

    def __repr__(self):
        return f"jno.solve.{self.name}()"


class PrecondContext:
    """What a preconditioner spec sees at materialization time.

    ``ctx.A`` is the assembled :class:`LinearOperator` (matvec-only on the Jacobian-free
    nonlinear path), ``ctx.fem`` the owning :class:`jno.FEM` (``None`` outside ``fem.solve``),
    ``ctx.diag()`` the operator diagonal. For multifield systems ``ctx.blocks`` are the
    per-field DOF slices (from ``fem.offsets``), ``ctx.block_slice(field)`` resolves a trial
    symbol (or integer index) to its slice, and ``ctx.sub(i, j=None)`` is the ``(i, j)``
    sub-operator as a :class:`LinearOperator` — applied through the *full* operator's matvec
    (embed into block ``j``, extract block ``i``), so it stays sparse/matrix-free; ``diag`` and
    ``dense`` are exact views for ``i == j`` direct/diagonal inner solvers.

    ``ctx.assemble(terms, quad_degree=...)`` assembles an **auxiliary weak form** with the
    ordinary ``jno.fem`` machinery and returns its operator — the "preconditioners are weak
    forms" primitive (weighted mass matrices, low-order proxies, shifted operators).
    """

    def __init__(self, A: LinearOperator, fem: Any = None, grid: Any = None):
        self.A = A
        self.fem = fem
        self._grid = grid

    def diag(self):
        return self.A.diag()

    @property
    def grid(self):
        """Structured-grid descriptor ``{shape, spacing, origin}`` when the operator lives on a regular
        grid (``jno.domain(..., structured=True)``), else ``None`` — needed by geometric multigrid
        (:func:`jno.precond.gmg`). An explicit override if given, else derived from the owning FEM's
        domain (``ctx.fem.domain.mesh_connectivity["grid"]``)."""
        if self._grid is not None:
            return self._grid
        dom = getattr(self.fem, "domain", None)
        mc = getattr(dom, "mesh_connectivity", None)
        return mc.get("grid") if isinstance(mc, dict) else None

    @property
    def blocks(self):
        blocks = getattr(self.fem, "blocks", None)
        if blocks is None:
            raise TypeError("PrecondContext.blocks: no per-field block structure (single field, or no FEM attached).")
        return blocks

    def block_slice(self, field) -> slice:
        if isinstance(field, int):  # plain block index
            return self.blocks[field]
        if self.fem is None:
            raise TypeError("PrecondContext.block_slice: resolving a trial symbol needs the owning FEM.")
        return self.blocks[self.fem.block_index(field)]

    def sub(self, i, j=None) -> LinearOperator:
        si = self.block_slice(i)
        sj = si if j is None else self.block_slice(j)
        A = self.A
        n = A.shape[0]

        def _embed(v, s):
            return jnp.zeros((n,), v.dtype).at[s].set(v)

        mv = lambda v: A.mv(_embed(v, sj))[si]
        t_mv = lambda v: A.T.mv(_embed(v, si))[sj]
        diag_fn = (lambda: A.diag()[si]) if sj == si else None
        dense_fn = lambda: A.dense()[si, sj]
        ni = si.stop - si.start
        nj = sj.stop - sj.start
        return LinearOperator.from_matvec(mv, t_mv=t_mv, diag_fn=diag_fn, dense_fn=dense_fn, shape=(ni, nj))

    def assemble(self, terms, *, quad_degree: int = 2) -> LinearOperator:
        import jax.experimental.sparse as jsp

        from ... import fem as _fem_entry

        aux = _fem_entry(list(terms), quad_degree=quad_degree)
        # steady linear, real (mode "linear") OR complex (mode "complex"); transient/nonlinear rejected
        if aux.is_transient or not (aux.is_linear or aux.is_complex):
            raise ValueError(
                "PrecondContext.assemble: the auxiliary preconditioner form must be steady linear (real or complex)."
            )

        def _bcoo(A):
            # small systems may assemble dense; convert NOW (concrete) so a sparse-direct inner
            # solver stays jit-safe when the applier later runs inside a traced Krylov loop
            if hasattr(A, "indices"):
                return A
            return jsp.BCOO.fromdense(jnp.asarray(A.todense() if hasattr(A, "todense") else A))

        if aux.is_complex:
            # A COMPLEX auxiliary operator (e.g. the shifted-Laplacian twin of a complex Helmholtz)
            # must precondition the outer complex solve's 2n real-equivalent system, i.e. the block
            # ``[[Mr,-Mi],[Mi,Mr]]``. This used to rebuild that block by hand from the form's Re/Im legs.
            # It no longer has to: a complex form is FUSED into exactly that 2n block at assembly, so
            # ``aux.A`` already IS it (verified bit-identical to the hand-built version). The remaining
            # complex-specific concern is periodic ties, and only that.
            if getattr(aux, "_periodic", None) is not None:
                raise NotImplementedError(
                    "PrecondContext.assemble: a complex auxiliary form with periodic ties is not supported "
                    "(the outer P-reduction is not mirrored onto the preconditioner block)."
                )

        return LinearOperator(_bcoo(aux.A))


def materialize_precond(spec: Any, ctx: PrecondContext) -> Callable:
    """Turn a preconditioner spec into the ``v -> M^{-1} v`` applier for this solve."""
    if hasattr(spec, "materialize"):
        return spec.materialize(ctx)
    if callable(spec):  # duck-typed: a bare ``ctx -> apply`` factory
        return spec(ctx)
    raise TypeError(
        f"precond= expects a jno.precond.* spec or a callable (ctx) -> (v -> M^-1 v); got {type(spec).__name__}."
    )


def prepare_precond(spec: Any, fem: Any) -> None:
    """Run a spec's eager preparation (optional ``spec.prepare(fem)``) once, at compose time.

    Auxiliary-assembly work (``jno.precond.form``) must happen OUTSIDE any trace: on the
    matrix-free nonlinear path ``materialize`` runs inside the Newton/Picard ``while_loop``
    body, where a fresh ``jno.fem`` assembly entangles with the loop tracers (and would
    re-assemble per re-trace anyway). Specs without a ``prepare`` hook need no eager work.
    """
    prep = getattr(spec, "prepare", None)
    if prep is not None and fem is not None:
        prep(fem)


_KRYLOV_SCALE_THRESHOLD = 1e8  # normalize the system only when the operator magnitude is genuinely extreme


def _normalize_extreme_scale(op, b, precond):
    """Guard the Krylov path against extreme operator magnitude. A mass-dominated eddy operator
    (``|A| ~ jωσ ~ 1e12``) makes jax's GMRES Arnoldi break down and return ~0 *regardless of the
    preconditioner* (plain Jacobi stalls identically). Scale the operator and RHS by a concrete scalar
    ``α`` so the iteration runs on an ``O(1)`` system; the solution is invariant (``Â x = b̂ ⟺ A x = b``),
    so nothing downstream changes and no un-scaling is needed.

    Fires ONLY for a **concrete** BCOO operator (a forward solve) whose magnitude exceeds the threshold —
    a traced/parametric operator (its magnitude is a tracer, so ``float(...)`` raises) and a normally
    scaled one are returned untouched. A frozen AMS auxiliary (assembled from the *un-scaled* operator) is
    reset so the coming ``materialize`` re-freezes it from the scaled operator; that reset only happens on
    this concrete forward path, so a traced/parametric ``.build()`` is never disturbed."""
    import math

    import jax.experimental.sparse as jsp

    bc = getattr(op, "bcoo", None)
    if bc is None:
        return op, b
    try:
        alpha = float(jnp.max(jnp.abs(bc.data)))  # concrete forward operator only; a tracer raises here
    except Exception:  # noqa: BLE001 — traced operator: leave the frozen-aux path untouched
        return op, b
    if not math.isfinite(alpha) or alpha <= _KRYLOV_SCALE_THRESHOLD:
        return op, b
    op_scaled = LinearOperator(jsp.BCOO((bc.data / alpha, bc.indices), shape=bc.shape))
    if precond is not None and getattr(precond, "_frozen", None) is not None:
        precond._frozen = None  # re-freeze the AMS auxiliaries from the scaled operator (forward path only)
    return op_scaled, jnp.asarray(b) / alpha


def _shardable(op: LinearOperator, linear, precond) -> bool:
    """Can this slot-composed solve be partitioned across devices?

    The set is narrow, and narrow on purpose -- each exclusion is a case where sharding would either
    fail outright or quietly gather the whole operator back onto one device, which is the worst
    outcome (the memory saving disappears and only the collectives remain).

    * **No assembled matrix** -- a matvec-only or dense operator has no triplet axis to partition.
    * **A direct solver** -- ``lu``/``dense`` factorise the matrix themselves; ``spsolve`` is
      single-device with no batching rule, so there is nothing to distribute.
    * **A preconditioner other than Jacobi** -- the applier is materialised from the assembled
      operator and closes over it (Chebyshev needs matvecs with ``A``, ``form`` holds an auxiliary
      BCOO, ``amg``/``ams`` build host-side through scipy/pyamg). A closed-over array is baked in as
      a compile-time constant and replicated to every device, so sharding the main operator while the
      preconditioner drags a full copy along saves nothing. Jacobi is the exception because it needs
      only the diagonal, and the diagonal is the same scatter-add the matvec already performs -- so it
      is computed *from the sharded triplets* and never touches an assembled matrix.
    A **traced** operator is covered, by a different mechanism: ``device_put`` cannot place a tracer,
    so the parametric and differentiate-through paths take the ``with_sharding_constraint`` route in
    :func:`~.sharding.constrain_operator` instead of the ``device_put`` + ``in_shardings`` one.

    Anything outside the set falls back silently to the ordinary path. It must not raise: the user
    asked for a solver configuration, not for sharding, and automatic placement is not a request they
    can be held to.
    """
    from ...precond import _Jacobi

    A = op.bcoo
    if A is None:
        return False
    if getattr(linear, "direct", False):
        return False
    return precond is None or isinstance(precond, _Jacobi)


def _is_traced(A) -> bool:
    """Is this operator's data a tracer? Decides which of the two sharding mechanisms applies."""
    return isinstance(A.data, jax.core.Tracer) or isinstance(A.indices, jax.core.Tracer)


def _compilable(linear, precond) -> bool:
    """Can this slot pair run as ONE compiled function, cached across ``fem.solve()`` calls?

    Both halves must clear two independent bars, and both default to "no" so that an unrecognised or
    newly added slot is merely eager rather than quietly broken.

    * It must **trace**. ``jacobi`` materialises from the traced operator's diagonal;
      ``amg``/``ams``/``form`` assemble an auxiliary operator host-side through scipy/pyamg, and
      ``chebyshev`` (either as a solver or as a preconditioner) branches on measured spectrum bounds
      -- a tracer has no answer for ``if hi <= 0``. Solvers declare this as the ``jit`` trait,
      preconditioners as :attr:`_Spec.traceable`.
    * It must have a stable **value key**. ``jax.jit`` caches on the hash of its static arguments, so
      a spec that hashes by identity would compile once per call -- the very cost this is buying off.
      See :class:`LinearSolver` for the measurement.

    A wrong "yes" would surface as a trace error at solve time, or worse as a cache hit on a
    differently configured solver; a wrong "no" costs only the speed-up. Hence the asymmetry.
    """
    # Read both slots through `getattr`: the documented extension contract lets a **bare callable**
    # stand in for either spec, and a bare callable has neither declaration -- so it lands on the
    # eager path, which is the right answer for code jNO knows nothing about.
    if not getattr(linear, "traits", {}).get("jit", True) or getattr(linear, "key", None) is None:
        return False
    return precond is None or (getattr(precond, "traceable", False) and getattr(precond, "key", None) is not None)


@partial(jax.jit, static_argnames=("linear", "precond"))
def _composed_compiled(A, b, x0, *, linear, precond):
    """The slot-composed linear solve, COMPILED -- and cached by ``jax.jit`` across ``fem.solve()``
    calls rather than by hand.

    The slot path called its Krylov solver from eager Python, so every iteration paid dispatch. What
    removing that is worth depends on the DEVICE, and in a way worth stating because it is not
    obvious: eager cost is host-bound (the Python dispatch per iteration), so it barely moves between
    machines, while compiled cost is device-bound. The ratio therefore tracks how fast the GPU is.

    Measured on an RTX 3070 at n=13759 -- ``bicgstab+jacobi`` 114.1 -> 18.1 ms (6.3x), ``cg+jacobi``
    97.5 -> 14.5 (6.7x), ``minres+jacobi`` 115.2 -> 20.2 (5.7x), ``gmres+jacobi`` 398.4 -> 183.2
    (2.2x), ``fgmres+jacobi`` 536.3 -> 300.8 (1.8x). On CPU, 1.8-4.2x. On a faster GPU the same
    ``bicgstab+jacobi`` was 104.7 -> 6.3 ms (16.6x) at n=13861: nearly the same EAGER cost, a 3x
    quicker compiled one. ``fgmres`` gains least everywhere -- it is jNO's own restart loop, so more
    of its time was always arithmetic rather than dispatch. Answers are unchanged (max |diff| 1.4e-12
    down to 3e-16).

    This has to live at MODULE level. Wrapping the closure inside ``compose_linear_solve_fn`` looks
    equivalent and is not: ``FEM.solve`` re-composes on every call, so the wrapper would be a new
    function object each time, miss ``jax.jit``'s cache, and recompile -- measured 115 ms against
    5.9 ms for the same callable reused, i.e. slower than not compiling at all. One module-level
    function with the slots as STATIC arguments means jax's own cache spans calls, with no hand-rolled
    memo to keep coherent. That only works because the specs hash by VALUE (:class:`LinearSolver`);
    with identity hashing, module level or not, an inline ``linear=jno.solve.cg()`` recompiles.

    Staleness is handled by the same mechanism rather than by bookkeeping: the specs are static, so a
    different solver is a different cache entry, while ``A``/``b`` are traced, so a changed operator
    re-runs the compiled graph -- the Jacobi diagonal is recomputed from the new values, not reused.
    Only host-side preconditioner setup would break that, which is exactly what :func:`_compilable`
    excludes; ``prepare_precond`` stays eager in the composer.

    There is deliberately no ``fem`` argument. Nothing that gets here needs one -- a traceable
    preconditioner builds from the operator by definition -- and holding a FEM object as a static
    argument would key the cache on a problem that is rebuilt each sweep step, recompiling every
    time and pinning every FEM ever solved in ``jax``'s compilation cache.
    """
    op = LinearOperator(A)  # the triplets arrive as jit ARGUMENTS; the wrapper is rebuilt inside
    M = materialize_precond(precond, PrecondContext(op, None)) if precond is not None else None
    return linear(op, jnp.asarray(b).reshape(-1), M=M, x0=x0)


def compose_linear_solve_fn(linear, precond, x0, fem=None, shard=None) -> Callable:
    """Compose the linear-mode slots into the classic ``(A, b) -> x`` ``solve_fn`` contract.

    The composed callable is handed to the *existing* dispatch (plain / periodic-reduced /
    parametric ``FemLinearSystem`` / complex real-block), so every path keeps its current
    reduction and implicit-differentiation behaviour. It accepts the assembler's BCOO operator
    directly -- no densification.

    This is the single ``LinearOperator`` construction point for **every** slot-composed linear
    solve, so partitioning it here covers all the Krylov solvers at once rather than one at a time.
    It cannot be a decorator around ``linear``, though: ``composed`` closes over the operator, while
    sharding requires ``data``/``indices`` to arrive as ``jit`` *arguments* (a closed-over array is
    replicated to every device, giving right answers with zero collectives and zero memory saving).
    So the call structure is inverted -- the triplets are threaded in as arguments and the matvec is
    rebuilt as a ``segment_sum`` inside. See :func:`_shardable` for what is and is not covered.
    """
    if linear is None:
        from ... import solve as _solve_ns

        linear = _solve_ns.bicgstab()  # matches the historic matrix-free default
    if precond is not None:
        prepare_precond(precond, fem)  # eager auxiliary assembly (safe for traced/parametric solves)
    x0_flat = None if x0 is None else jnp.asarray(x0).reshape(-1)

    from .sharding import resolve_devices

    devices = resolve_devices(shard)
    # `None`/`True` mean "automatic"; anything else is an explicit request. Only an explicit one may
    # shard a TRACED operator -- see the comment at the traced branch below for why that distinction
    # is a safety property and not caution.
    shard_explicit = shard is not None and shard is not True

    def composed(A, b):
        op = A if isinstance(A, LinearOperator) else LinearOperator(A)
        # Extreme-magnitude systems (the physical eddy regime) break the Krylov Arnoldi; scale to O(1)
        # first — solution-invariant, and a no-op for normally scaled / traced operators.
        op, b = _normalize_extreme_scale(op, b, precond)
        rhs = jnp.asarray(b).reshape(-1)
        _traced = _shardable(op, linear, precond) and _is_traced(op.bcoo)
        if devices and _traced and shard_explicit:
            # Parametric / differentiate-through: we are already INSIDE a trace, so there is no jit
            # boundary of ours to hang `in_shardings` on -- and adding one would close over the
            # operator, which constant-folds and replicates it. `with_sharding_constraint` partitions
            # the same axis from inside the trace instead, and gradients flow through.
            #
            # OPT-IN ONLY, and that is a safety property rather than caution. Inside a trace we are a
            # guest in someone else's computation, and a sharding constraint must agree with the
            # device commitments of every other value in that jit. Under `crux` it does not: the
            # optimiser's parameters arrive committed to one device while our constraint spans all of
            # them, and JAX rejects the mix. That conflict cannot be detected in advance
            # (`get_abstract_mesh()` is empty there) and cannot be caught locally -- it surfaces when
            # the OUTER jit compiles, long after this function returned -- so there is no fallback to
            # write. Automatic placement must never be able to fail a user's compile, so `shard=None`
            # leaves traced operators alone; an explicit `shard=N` is a request we can honour and the
            # user can diagnose.
            from .sharding import constrain_operator

            n = int(op.shape[0])
            matvec, diag_fn = constrain_operator(op.bcoo, devices)
            mf = LinearOperator.from_matvec(matvec, diag_fn=diag_fn, shape=(n, n))
            M = materialize_precond(precond, PrecondContext(mf, fem)) if precond is not None else None
            return linear(mf, rhs, M=M, x0=x0_flat)
        if devices and not _traced and _shardable(op, linear, precond):
            from .sharding import jacobi_from_diagonal, sharded_solve

            n = int(op.shape[0])

            def _run(matvec, r, M, guess):
                return linear(LinearOperator.from_matvec(matvec, shape=(n, n)), r, M=M, x0=guess)

            return sharded_solve(
                op.bcoo,
                rhs,
                _run,
                devices,
                precond_fn=None if precond is None else jacobi_from_diagonal,
                x0=x0_flat,
            )
        M = materialize_precond(precond, PrecondContext(op, fem)) if precond is not None else None
        return linear(op, rhs, M=M, x0=x0_flat)

    if not _compilable(linear, precond):
        return composed

    def composed_jit(A, b):
        """The compiled path, with the cases it must hand back to ``composed`` untouched.

        A **matrix-free or dense-wrapped** operator arrives as a ``LinearOperator``, which is not a
        pytree and so cannot be a ``jit`` argument at all (its matvec is a Python closure).

        A **sharded** solve compiles itself, with ``in_shardings`` on the triplets; compiling it again
        from here would close over the operator and replicate it, which is how sharding silently
        becomes a full copy per device.

        **Extreme magnitude** is why the scaling guard runs out here rather than inside. It fires only
        on a *concrete* operator -- ``float(max|A|)`` on a tracer raises, and it reads that as "traced,
        leave alone". Inside the compiled function every operator is a tracer, so an eddy-regime
        ``|A| ~ 1e12`` would sail past the guard into the Arnoldi breakdown it exists to prevent. The
        scaling is host-side work on one scalar, so hoisting it costs nothing.

        The **convergence guard** is hoisted for the same reason, and it is the subtler of the two.
        Every Krylov solver ends in :func:`_maybe_residual_check`, which needs a concrete residual and
        steps aside on a tracer -- so compiling the solver does not disable the check loudly, it
        disables it silently, trading "raise rather than return garbage" for the speed. A solve that
        exhausts its iteration budget then returns an unconverged vector as if it had succeeded.
        Re-running it out here on the result costs one matvec against hundreds of iterations, and the
        split is the same one the default steady-linear path makes.
        """
        if isinstance(A, LinearOperator):
            return composed(A, b)
        op = LinearOperator(A)
        if devices and _shardable(op, linear, precond):
            return composed(A, b)
        op, rhs = _normalize_extreme_scale(op, b, precond)
        # Re-decide AFTER the scaling guard, because that guard MUTATES the preconditioner: on an
        # extreme-magnitude operator it drops a frozen AMS auxiliary so the coming materialize
        # re-freezes from the scaled matrix. That un-freezes the very state `_compilable` keyed on at
        # compose time, and materialising an unfrozen AMS inside the trace means scipy on a tracer.
        if not _compilable(linear, precond):
            return composed(A, b)
        rhs = jnp.asarray(rhs).reshape(-1)
        mat = op.bcoo if op.bcoo is not None else A
        x = _composed_compiled(mat, rhs, x0_flat, linear=linear, precond=precond)
        return _maybe_residual_check(op, rhs, x, getattr(linear, "name", "solve"))

    return composed_jit


def compose_nonlinear_solve_fn(nonlinear, linear, precond, fem=None) -> Callable:
    """Compose the nonlinear-mode slots into the ``(residual_fn, u0) -> u`` ``solve_fn`` contract.

    The inner Newton/Picard linear solve is **matrix-free** -- each outer iteration linearizes
    the residual into a JVP matvec ``J(u_k) v`` -- so a ``precond=`` spec is materialized *per
    linearization* against that matvec-only operator (wrapped with the system size, so block
    slicing and power-iteration bounds work). The same preconditioned solve serves the
    implicit-differentiation tangent/adjoint solve of ``custom_root``.

    What composes here: ``form`` (auxiliary operators assemble independently -- and are cached,
    so the assembly happens once even though materialization runs per iteration),
    ``inner(<krylov>)``, ``chebyshev`` (bounds by power iteration on the JVP), a **pre-built**
    ``amg`` (``spec.build(A_representative)``), and ``block_diag``/``triangular`` over those.
    What cannot: specs that need the assembled matrix -- ``jacobi`` (no diagonal on a matvec),
    an unbuilt ``amg``, ``lu``/``dense`` inner solvers on sub-blocks -- these raise their own
    targeted errors when materialized.
    """
    if nonlinear is None:
        from ... import solve as _solve_ns

        nonlinear = _solve_ns.newton()

    inner = None
    if linear is not None or precond is not None:
        if linear is None:
            from ... import solve as _solve_ns

            solver = _solve_ns.bicgstab()  # the historic matrix-free inner default
        else:
            solver = linear
        if precond is not None:
            prepare_precond(precond, fem)  # aux assembly now, NOT inside the traced Newton loop

        def inner(matvec, rhs):
            n = rhs.shape[0]
            op = LinearOperator.from_matvec(matvec, shape=(n, n))
            M = materialize_precond(precond, PrecondContext(op, fem)) if precond is not None else None
            return solver(op, rhs, M=M)

    def _composed(residual_fn, u0, *, jacobian=None):
        return nonlinear(residual_fn, u0, linear_solve=inner, jacobian=jacobian)

    # A direct (assembled-Jacobian) Newton needs the step Jacobian threaded in; flag it so the caller
    # (SemidiscreteTimeBlock.step) builds ``M/dt + jacobian`` and passes it via ``jacobian=``.
    _composed.wants_jacobian = bool(getattr(nonlinear, "direct", False))
    return _composed


def _add_step_operator(M, A, scale):
    """Form the theta-step operator ``M + scale * A`` once, eagerly.

    Both BCOO: concatenate triplets (duplicates are legal COO — every consumer sums them:
    matvec, ``matrix_diagonal``, ``sparse_lu_solve``, the AMG CSR conversion). Anything else:
    dense addition.

    Concatenation is itself a duplicate source: ``M`` and ``A`` overlap almost entirely, so the step
    operator carries ~2x the triplets of either even when both arrive compressed — and it is applied
    on every step of the march. ``sum_duplicate_triplets`` collapses that for the eager
    constant-operator path; for the per-step callers in ``backend_blocks`` the operands are traced, so
    it returns them untouched (correct, just uncompressed) until the pattern is hoisted host-side.
    """
    if hasattr(M, "todense") and hasattr(A, "todense"):
        import jax.experimental.sparse as jsp

        from .fem_utils import sum_duplicate_triplets

        data = jnp.concatenate([M.data, scale * A.data])
        indices = jnp.concatenate([M.indices, A.indices], axis=0)
        return sum_duplicate_triplets(jsp.BCOO((data, indices), shape=M.shape))
    dense = lambda x: x.todense() if hasattr(x, "todense") else jnp.asarray(x)
    return dense(M) + scale * dense(A)


def compose_transient_step_solvers(nonlinear, linear, precond, fem, block):
    """Compose the slots into per-step solvers for the transient integrator.

    Returns ``(linear_step_solve, nonlinear_step_solve)`` (one is ``None``), matching
    :meth:`SemidiscreteTimeBlock.step`'s injection points:

    * **nonlinear block** — the per-step implicit solve ``G(u_next) = 0`` gets the same
      composed driver as the steady nonlinear path (``nonlinear=`` slot + matrix-free inner
      ``linear``/``precond``), so ``picard(damping=…)`` with ``jno.lag`` coefficients works per
      time step.
    * **linear block** — the theta-step system ``(M + θ dt A) u_next = rhs`` is what the
      ``linear`` solver and ``precond`` spec see. When the operator is **time-independent**
      (``operator_fn is None``) the step matrix is formed once, eagerly, and the preconditioner
      is materialized **once before the scan** — the AMG hierarchy / auxiliary ``form`` operator
      is then reused by every step (the whole point of preconditioning a time loop). A
      time-dependent ``operator_fn(t, args)`` falls back to per-step materialization against a
      matvec-only operator that still exposes the exact step diagonal (so ``jacobi`` works).
    """
    if block.is_nonlinear():
        return None, compose_nonlinear_solve_fn(nonlinear, linear, precond, fem)
    if nonlinear is not None:
        raise ValueError("fem.solve: nonlinear= given, but this transient block is linear (no linearization).")

    if linear is None and precond is None:
        # No linear/precond slots: defer to SemidiscreteTimeBlock.step's built-in per-step solve, a
        # Jacobi-preconditioned BiCGStab at tol=1e-10. The historic default here was a *bare* bicgstab()
        # with NO preconditioner, which silently under-converged each step — invisible on homogeneous
        # decay (the warm-start is already near the answer) but corrupting any forced/growing solution.
        return None, None

    if linear is None:
        from ... import solve as _solve_ns

        solver = _solve_ns.bicgstab()
    else:
        solver = linear
    if precond is not None:
        prepare_precond(precond, fem)

    # constant-operator fast path: one step matrix, one preconditioner, reused by every step
    static_op = None
    static_M = None
    if block.operator_fn is None and block.A is not None and block.dt is not None:
        theta = float(block.metadata.get("theta", 1.0)) if block.metadata else 1.0
        static_op = LinearOperator(_add_step_operator(block.M, block.A, theta * float(block.dt)))
        if precond is not None:
            static_M = materialize_precond(precond, PrecondContext(static_op, fem))

    def step_solve(matvec, rhs, x0, diag_fn):
        if static_op is not None:
            op, M = static_op, static_M
        else:
            op = LinearOperator.from_matvec(matvec, diag_fn=diag_fn, shape=(rhs.shape[0], rhs.shape[0]))
            M = materialize_precond(precond, PrecondContext(op, fem)) if precond is not None else None
        if getattr(solver, "direct", False):
            # A DIRECT solver (lu/dense) factorizes the step operator itself — it takes no preconditioner,
            # so don't synthesize the Jacobi one below (which it would reject). Needs a materializable
            # operator: the constant-operator ``static_op`` (a real M+θdt·A matrix) provides it, letting a
            # direct solve compose with the transient stepper — e.g. a Taylor-Hood saddle under adapt=.
            return solver(op, rhs, x0=x0)
        if M is None:  # iterative: never run the per-step solve unpreconditioned -> Jacobi (the step diagonal)
            diag = op.diag() if hasattr(op, "diag") else diag_fn()
            inv = 1.0 / jnp.where(jnp.abs(diag) > 1e-30, diag, 1.0)
            M = lambda x: inv * x  # noqa: E731
        return solver(op, rhs, M=M, x0=x0)

    return step_solve, None
