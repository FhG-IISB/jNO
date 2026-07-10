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
    (``vmap: "native" | "sequential" | "no"``) so composition layers (and, later, the auto
    policy) can pick honestly instead of silently host-looping.
    """

    def __init__(self, fn: Callable, *, name: str, traits: Optional[dict] = None, direct: bool = False):
        self._fn = fn
        self.name = name
        self.traits = {"vmap": "native", "jit": True, **(traits or {})}
        self.direct = direct  # a direct solver ignores x0 and takes no preconditioner

    def __call__(self, A, b, *, M=None, x0=None):
        op = A if isinstance(A, LinearOperator) else LinearOperator(A)
        b = jnp.asarray(b).reshape(-1)
        if self.direct and M is not None:
            raise ValueError(f"jno.solve.{self.name} is a direct solver -- it takes no preconditioner (precond=).")
        return self._fn(op, b, M=M, x0=x0)

    def __repr__(self):
        return f"jno.solve.{self.name}({', '.join(f'{k}={v}' for k, v in self.traits.items())})"


class NonlinearSolver:
    """A configured nonlinear driver: ``driver(residual_fn, u0, *, linear_solve=None) -> u``."""

    def __init__(self, fn: Callable, *, name: str, traits: Optional[dict] = None):
        self._fn = fn
        self.name = name
        self.traits = {"vmap": "native", "jit": True, **(traits or {})}

    def __call__(self, residual_fn, u0, *, linear_solve=None):
        return self._fn(residual_fn, u0, linear_solve=linear_solve)

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

    def __init__(self, A: LinearOperator, fem: Any = None):
        self.A = A
        self.fem = fem

    def diag(self):
        return self.A.diag()

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
            # must precondition the outer complex solve's 2n real-equivalent system, so assemble it as
            # the same block ``[[Mr,-Mi],[Mi,Mr]]`` (2n x 2n) from the form's two real/imag legs.
            if getattr(aux, "_periodic", None) is not None:
                raise NotImplementedError(
                    "PrecondContext.assemble: a complex auxiliary form with periodic ties is not supported "
                    "(the outer P-reduction is not mirrored onto the preconditioner block)."
                )
            from ..._fem import _complex_block_bcoo

            legs = aux.operator  # ((A_r, b_r), (A_i, b_i)); a preconditioner form is parameter-independent
            if not (isinstance(legs, tuple) and len(legs) == 2 and all(isinstance(leg, tuple) for leg in legs)):
                raise NotImplementedError(
                    "PrecondContext.assemble: expected two eager (A, b) complex legs "
                    "(a parametric complex preconditioner form is not supported)."
                )
            A_r, A_i = _bcoo(legs[0][0]), _bcoo(legs[1][0])
            return LinearOperator(_complex_block_bcoo(A_r, A_i, A_r.shape[0]))

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


def compose_linear_solve_fn(linear, precond, x0, fem=None) -> Callable:
    """Compose the linear-mode slots into the classic ``(A, b) -> x`` ``solve_fn`` contract.

    The composed callable is handed to the *existing* dispatch (plain / periodic-reduced /
    parametric ``FemLinearSystem`` / complex real-block), so every path keeps its current
    reduction and implicit-differentiation behaviour. It accepts the assembler's BCOO operator
    directly -- no densification.
    """
    if linear is None:
        from ... import solve as _solve_ns

        linear = _solve_ns.bicgstab()  # matches the historic matrix-free default
    if precond is not None:
        prepare_precond(precond, fem)  # eager auxiliary assembly (safe for traced/parametric solves)
    x0_flat = None if x0 is None else jnp.asarray(x0).reshape(-1)

    def composed(A, b):
        op = A if isinstance(A, LinearOperator) else LinearOperator(A)
        M = materialize_precond(precond, PrecondContext(op, fem)) if precond is not None else None
        return linear(op, jnp.asarray(b).reshape(-1), M=M, x0=x0_flat)

    return composed


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

    return lambda residual_fn, u0: nonlinear(residual_fn, u0, linear_solve=inner)


def _add_step_operator(M, A, scale):
    """Form the theta-step operator ``M + scale * A`` once, eagerly.

    Both BCOO: concatenate triplets (duplicates are legal COO — every consumer sums them:
    matvec, ``matrix_diagonal``, ``sparse_lu_solve``, the AMG CSR conversion). Anything else:
    dense addition.
    """
    if hasattr(M, "todense") and hasattr(A, "todense"):
        import jax.experimental.sparse as jsp

        data = jnp.concatenate([M.data, scale * A.data])
        indices = jnp.concatenate([M.indices, A.indices], axis=0)
        return jsp.BCOO((data, indices), shape=M.shape)
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

    if linear is None:
        from ... import solve as _solve_ns

        solver = _solve_ns.bicgstab()  # the historic per-step default
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
        return solver(op, rhs, M=M, x0=x0)

    return step_solve, None
