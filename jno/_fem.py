"""``jno.fem`` — assemble a traced weak form into FEM matrices/operators.

Author the physics as ordinary jNO residual expressions and hand the flat list
to :func:`fem`. Each residual is classified by **role** (does it contain the
test function?) and **region** (carried by the bound coordinates), then
assembled by the native Lagrange assembler. The returned :class:`FEM` exposes
the assembled artefacts — ``A``/``b`` for a linear problem,
``residual``/``jacobian`` for a nonlinear one — plus the ``problem``, ``mesh``
and ``dofs``.

Classification rule
-------------------
* a residual that contains the **test function** is a weak term — integrated
  over its region (volume, or a boundary region for surface/Neumann/Robin
  terms);
* a residual with the **trial only** (no test function) on a **boundary**
  region is an essential (Dirichlet) condition ``u - g`` — its DOFs are pinned;
* a trial-only residual on a volume region is an error (a forgotten test
  function).

For a plain forward solve you can drive your own solver off the assembled artefacts
(``jnp.linalg.solve(fem.A, fem.b)``, ``scipy``). :meth:`FEM.solve` additionally provides a
**differentiable** forward solve as a trace node — the entry point for inverse problems — with
matrix-free defaults (BiCGStab / Newton–Krylov / backward-Euler) you can override via ``solve_fn``.
"""

from __future__ import annotations

import functools
import inspect
import warnings
from typing import Any, Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .trace import (
    FemLinearSystem,
    FemResidualOperator,
    FunctionCall,
    GaugePin,
    ModelCall,
    NormalDerivative,
    Placeholder,
    RegionMask,
    StateUpdate,
    TestFunction,
    TrialFunction,
    Variable,
    state_updates,
)

__all__ = ["fem", "FEM"]


def _as_dense(x):
    """Densify a sparse/BCOO matrix to a plain JAX array (no-op if already dense)."""
    return jnp.asarray(x.todense()) if hasattr(x, "todense") else jnp.asarray(x)


def _as_flat(x):
    """Flatten a nodal vector to a 1-D JAX array."""
    return jnp.asarray(x).reshape(-1)


# Component names for per-component (roller/symmetry) Dirichlet specs.
_COMPONENT_NAMES = {0: "x", 1: "y", 2: "z"}


def _residual_check(A, b, u, who):
    """Raise (eagerly) if ``A u = b`` is not solved -- a hard fail beats silently returning garbage."""
    matvec = (lambda v: A @ v) if hasattr(A, "__matmul__") else (lambda v: jnp.asarray(A) @ v)
    rel = float(jnp.linalg.norm(b - matvec(u)) / (jnp.linalg.norm(b) + 1e-30))
    if not np.isfinite(rel) or rel > 1e-4:
        raise RuntimeError(
            f"fem.solve default ({who}) did not solve the system (relative residual {rel:.1e}); the "
            "problem may be singular/ill-posed. Pass your own solver: fem.solve(solve_fn=lambda A, b: ...)."
        )
    return u


def _solve_linear_matrix_free(A, b, *, tol=1e-8, maxiter=20_000):
    """Default steady-linear solve: matrix-free **Jacobi-preconditioned** BiCGStab on the BCOO operator.

    Never forms the dense ``N x N`` matrix — each iteration is a couple of sparse matvecs ``A @ v`` (cost
    ``O(nnz)``), so memory stays ``O(nnz)`` and it runs at sizes where a factorisation would not fit;
    GPU-safe (unlike a sparse direct factorisation, which can exhaust cuSolver). BiCGStab handles
    **general** (non-symmetric) systems and a diagonal **Jacobi** preconditioner (cheap ``1/diag(A)``)
    accelerates diagonally-dominant (elliptic) problems. Raises if it fails to converge instead of
    returning garbage — for the indefinite saddle-point systems where Jacobi does not help, pass a
    direct solver via ``solve_fn`` (e.g. ``jno.utils.solver.linear.sparse_lu_solve`` on CPU, or
    ``jnp.linalg.solve``).
    """
    from .utils.solver.linear import jacobi

    matvec = lambda v: A @ v
    u, _info = jax.scipy.sparse.linalg.bicgstab(matvec, b, tol=tol, atol=0.0, maxiter=maxiter, M=jacobi(A))
    return _residual_check(A, b, u, "Jacobi-preconditioned BiCGStab")


def lag(expr: Any) -> Any:
    """Mark a weak-form coefficient as **lagged**: frozen within each linearization, updated
    between iterations — the Picard / fixed-point linearization as first-class API.

    The canonical use is a solution-dependent coefficient whose Newton tangent destroys the
    linearized system's structure — e.g. a shear-thinning viscosity ``mu_eff(u)`` in a
    rigid-plastic/non-Newtonian Stokes flow, where full Newton produces a strongly nonsymmetric
    velocity block that defeats AMG/block preconditioners, while the lagged (Picard) system is a
    plain symmetric Stokes solve per step::

        mu_eff = k_f / (3 * jno.np.sqrt(2/3 * rate2 + eps0**2))
        fem = jno.fem([2 * jno.lag(mu_eff) * inner(eps(ui), eps(vi)) - ...])
        fem.solve(nonlinear=jno.solve.picard(damping=0.7), linear=..., precond=...)

    Mechanically ``lag`` is ``stop_gradient`` on the traced expression, so ``jax.linearize`` of
    the residual yields the *lagged* operator: :func:`jno.solve.picard` (and plain Newton) then
    iterate with that linearization automatically. The converged solution is unchanged —
    ``R(u) = 0`` does not depend on gradient markers.

    **Inverse-problem caveat**: implicit differentiation (``custom_root``) also uses the lagged
    Jacobian for its tangent/adjoint solve, so gradients of ``fem.solve()`` w.r.t. parameters
    become the standard "Picard adjoint" approximation — widely used and usually descent-worthy,
    but not exact. Remove ``lag`` (full Newton) when exact parameter gradients matter more than
    per-step solvability.
    """
    # views and raw trace nodes expose ``.stop_gradient`` as a *property* (its value — a trace
    # node — is itself callable, so a callable() test cannot distinguish them from methods)
    if isinstance(getattr(type(expr), "stop_gradient", None), property):
        return expr.stop_gradient
    return jax.lax.stop_gradient(expr)  # plain arrays inside hand-written residuals


# ---------------------------------------------------------------------------
# expression helpers
# ---------------------------------------------------------------------------
def _is_view(obj: Any) -> bool:
    """A typed semantic view wraps its Placeholder in ``_expr``."""
    return hasattr(obj, "_expr")


def _bare(obj: Any):
    """Underlying Placeholder behind an optional typed view."""
    return obj.expr if _is_view(obj) else obj


def _walk(node: Any):
    """Yield every node in a Placeholder tree (deduplicated by id)."""
    from .utils.solver.solver_helper import iter_placeholder_children

    seen: set[int] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        yield n
        for child in iter_placeholder_children(n):
            stack.append(child)


def _spatial_coord_vars(constraint: Any) -> List[Variable]:
    """Spatial coordinate Variables relevant to a constraint.

    Includes both the tree coordinates and the bound-view ``_coord_vars`` (which
    survive arithmetic), so a plain-value term like ``u(xl, yl) - 0`` still
    exposes its region.
    """
    out: dict[int, Variable] = {}

    def _add_coords(cv):
        # `_coord_vars` ({name: Variable}) is carried by bound views and, after a
        # `jno.np` reduction, by the resulting FunctionCall — so a bound boundary
        # term keeps its region even through `inner(...)` / arithmetic wrapping.
        if cv:
            for v in cv.values():
                if isinstance(v, Variable) and getattr(v, "axis", None) != "temporal":
                    out[id(v)] = v

    _add_coords(getattr(constraint, "_coord_vars", None))
    for n in _walk(_bare(constraint)):
        _add_coords(getattr(n, "_coord_vars", None))
        if isinstance(n, Variable) and getattr(n, "axis", None) != "temporal":
            out[id(n)] = n
    return list(out.values())


def _contains(constraint: Any, cls) -> bool:
    from .utils.solver.solver_helper import contains_node_type

    return contains_node_type(_bare(constraint), cls)


def _reject_source_terms(fem_obj: Any, where: str) -> None:
    """Fail loud if the weak form carries a **load** — a term with a test function but no trial.

    ``K x = λ M x`` is homogeneous: there is nowhere for a source to go. The assembled operator is
    ``(A, b)`` and an eigensolve reads only ``A``, so a load would otherwise be **silently dropped**
    and the caller would get the *undriven* spectrum back with no warning — measured: adding
    ``-3.0 * v`` on a boundary returned eigenvalues bit-identical to the source-free problem.

    Detection mirrors the assembler's own classification (:func:`_bare` + :func:`_contains`, the pair
    used at the top of :func:`fem`): split each volume/surface constraint additively and look for a
    piece with no ``TrialFunction``. Bilinear surface terms (Robin / impedance ``α u v``) contain the
    trial and are correctly left alone — they belong in ``K`` and genuinely shift the spectrum.
    """
    from .trace import TrialFunction
    from .utils.solver.weak_form import _split_additive_terms

    dom = getattr(fem_obj, "domain", None)
    for c, cl in zip(getattr(fem_obj, "_constraints", None) or [], fem_obj.classification or []):
        if not (isinstance(cl, str) and (cl.startswith("volume") or cl.startswith("surface"))):
            continue
        for _sign, sub in _split_additive_terms(dom, _bare(c)):
            if not _contains(sub, TrialFunction):
                raise ValueError(
                    f"{where}: the weak form carries a source term on '{cl}' (a term with the test "
                    f"function but no trial function). An eigenproblem K x = λ M x is homogeneous — the "
                    f"load vector is not part of the pencil, so it would be silently ignored and you "
                    f"would get the spectrum of the *undriven* problem. Drop the source term."
                )


def _eigs_constraint_maps(fem_obj: Any, n_full: int):
    """``(restrict, prolong, n_red)`` for the **space-reducing** constraints of an eigenproblem.

    ``fem.solve()`` applies Dirichlet by *row replacement* and periodic ties by a *per-call*
    reduction (:meth:`FemLinearSystem.solve`'s ``periodic=``). Neither survives into
    ``fem.operator[0]``, which is why reading that matrix directly gets an eigenproblem wrong:

    * **Dirichlet** row-replacement leaves identity rows against a full mass row, which injects
      spurious pairs — measured on a 198-DOF Dirichlet square (48 constrained), the lowest spurious
      eigenvalue was 267.4 while the true spectrum starts at 19.9. Small ``k`` misses them by luck;
      a larger ``k`` or ``which="largest"`` does not.
    * **Periodic** ties are dropped entirely — a periodic-in-x unit square returned the *non-periodic*
      Neumann spectrum ``0, π², π², 2π²`` where the truth is ``0, π², 4π², 4π²``.

    Both are the same operation: a prolongation ``P`` (n_full × n_red). Returned as matvec maps rather
    than matrices so the reduced pencil ``PᵀKP`` is never assembled and the LOBPCG path stays
    matrix-free. ``(None, None, n_full)`` when the form is unconstrained.
    """
    per = getattr(fem_obj, "_periodic", None)
    pairs = getattr(getattr(fem_obj, "domain", None), "_fem_native_dirichlet_pairs", None) or []
    if pairs and any(abs(float(val)) > 1e-12 for _d, val in pairs):
        raise ValueError(
            "jno.fem eigs: an INHOMOGENEOUS Dirichlet value has no meaning in K x = λ M x — the "
            "constrained DOFs are eliminated, not driven. Use a homogeneous pin (u(region) - 0.0)."
        )
    dofs = sorted({int(d) for d, _v in pairs})
    if per is not None and dofs:
        raise NotImplementedError(
            "jno.fem eigs: combining periodic ties with a Dirichlet pin is not supported yet — the "
            "Dirichlet DOFs are numbered in the FULL space and would have to be re-indexed into the "
            "periodic master space to compose the two reductions. Use one or the other for now."
        )

    if per is not None:
        from .utils.solver.fem_utils import prolong_periodic, reduce_vector_periodic

        restrict = lambda w: jnp.asarray(reduce_vector_periodic(per, w))  # noqa: E731  Pᵀ w
        prolong = lambda v: jnp.asarray(prolong_periodic(per, v))  # noqa: E731         P v
        n_red = int(jnp.shape(restrict(jnp.zeros(n_full)))[0])
        return restrict, prolong, n_red, "periodic"
    if dofs:
        free = jnp.asarray(sorted(set(range(n_full)) - set(dofs)), dtype=jnp.int32)
        restrict = lambda w: w[free]  # noqa: E731                                      gather (= Pᵀ)
        prolong = lambda v: jnp.zeros(n_full, v.dtype).at[free].set(v)  # noqa: E731    scatter (= P)
        return restrict, prolong, int(free.shape[0]), "dirichlet"
    return None, None, n_full, None


def _contains_network_call(constraint: Any) -> bool:
    """True if a constraint embeds a *network* ModelCall (``jno.nn.wrap(net)(...)``).

    Excludes zero-arg runtime parameters (``jno.np.parameter(...)``, scalar or FEM field). A network
    in a weak form plays one of two roles, disambiguated at the constraint-*set* level (see the
    ``is_vpinn`` routing in :func:`fem`): it *replaces* the trial (``u = net(x, y)`` — the VPINN
    signal; no ``TrialFunction`` appears in any weak constraint) or it is a *coefficient*
    multiplying a genuine trial (``net(x, y) * u.dx * v.dx`` — an assembled, runtime-parametric FE
    system whose kernel re-evaluates the network at the quadrature points)."""
    from .utils.solver.parametric_helpers import _is_runtime_scalar_parameter

    return any(isinstance(n, ModelCall) and not _is_runtime_scalar_parameter(n) for n in _walk(_bare(constraint)))


def _discover_domain(constraints: List[Any]):
    for c in constraints:
        for v in _spatial_coord_vars(c):
            d = getattr(v, "_domain", None)
            if d is not None:
                return d
    raise ValueError(
        "jno.fem: could not discover a domain from the constraints. "
        "Author the weak form with coordinates from domain.variable(...)."
    )


def _lower_gauge_pin(pin: GaugePin) -> Any:
    """Lower a ``p.pin()`` marker to a single-node Dirichlet residual ``field(node) - value``.

    Picks a deterministic vertex (nearest the mesh min-corner) and reuses
    ``domain.point_region`` + ``domain.variable``, so the synthesized residual is identical to a
    hand-written ``p(xpn, ypn) - value`` -- the value side is the literal constant, so the
    transient route treats it as a plain (time-independent) Dirichlet at every step. The pin's
    coordinate Variables are cached per (domain, field) so a second ``jno.fem(...)`` on the same
    domain neither re-registers the region nor re-samples (the cached vars are Dirichlet-only and
    never retagged, so reuse is safe).
    """
    import numpy as _np

    field = pin.field
    domain = getattr(field, "_domain", None)
    if domain is None:
        raise ValueError(
            "jno.fem: p.pin() needs a field from domain.fem_symbols(...); the pinned symbol carries no domain."
        )
    dim = int(domain.dimension)
    # Single leading underscore (not "__...__"): the pin node is a genuine single-vertex boundary
    # region that `_region_and_support` must SEE, so its tag must not match the reserved
    # double-underscore filter that hides internal/temporal tags from region detection.
    tag = f"_gauge_pin_{field.field_key}"
    cache = domain.__dict__.setdefault("_gauge_pin_coords", {})
    if tag not in cache:
        pts = _np.asarray(domain.mesh.points)[:, :dim]
        target = pts.min(axis=0)  # deterministic gauge node: the mesh min-corner vertex
        domain.point_region(tag, target)
        cache[tag] = domain.variable(tag, split=True)
    spatial = cache[tag][:dim]
    return field(*spatial) - pin.value


def _infer_vec(constraints: List[Any]) -> int:
    """Infer the vector size from the trial's ``value_shape`` across the constraints.

    Scalar → 1, ``value_shape=(2,)`` → 2, etc. (reuses ``_infer_trial_metadata``).
    """
    from .utils.solver.fem_utils import _infer_trial_metadata

    for c in constraints:
        meta = _infer_trial_metadata(_bare(c))
        if meta.get("has_trial"):
            return max(1, int(meta.get("vec", 1)))
    return 1


# (dimension, polynomial order) -> simplex element type; 1D uses the native LINE2 path.
_ELEMENT_FOR = {(2, 1): "TRI3", (2, 2): "TRI6", (3, 1): "TET4", (3, 2): "TET10"}
_P2_ELEMENTS = {"TRI6", "TET10", "QUAD8"}


def _infer_order(constraints: List[Any]) -> int:
    """Max element polynomial order across the trial fields (P1=1, P2=2)."""
    orders = [int(getattr(n, "order", 1)) for c in constraints for n in _walk(_bare(c)) if isinstance(n, TrialFunction)]
    return max(orders) if orders else 1


def _trial_spaces(constraints: List[Any]) -> set:
    """Distinct element families declared across the trial fields (default ``"Lagrange"``)."""
    return {
        str(getattr(n, "space", "Lagrange")) for c in constraints for n in _walk(_bare(c)) if isinstance(n, TrialFunction)
    }


def _native_lagrange_ok(domain: Any, constraints: List[Any], weak_bares: List[Any], periodic_ties: List[Any]) -> bool:
    """Whether the native Lagrange assembler should handle this problem.

    Native covers scalar/vector Lagrange P1/P2 on 2D triangle and 3D tetrahedral meshes — single-
    and multi-field, linear/nonlinear, steady & transient, Dirichlet + Neumann/Robin (2D edge / 3D
    tet-face quadrature) + per-region/frozen-coefficient terms, and runtime-parametric *scalar*
    coefficients (steady inverse). This gate rules out what the native Lagrange path does not yet
    cover, which the specialized branches in :func:`fem` handle (or reject as unsupported):

    * FEM *field* (nodal ``k(x)``) parameters — native threads scalar parameters only.

    Note: a runtime-*scalar* parameter AND a single-field nodal FIELD parameter k(x) are allowed here
    (this gate only runs on single-field problems -- multifield returns earlier). The transient call
    sites add their own runtime-parameter exclusion (native transient-parametric is not wired yet).
    Periodic ties are allowed here for the steady scalar single-field case (the caller scopes out the
    transient / vector / parametric periodic sub-cases, which build the reduction in their own
    branches); the reduction (``_build_periodic_reduction``) is fed the native assembly cells in
    ``_finalize``.
    """
    if getattr(domain, "dimension", None) not in (2, 3):
        return False
    if _trial_spaces(constraints) - {"Lagrange"}:
        return False
    # complex=True fields: the real-equivalent form couples re/im test functions within one term,
    # which the native one-test-field-per-term classifier rejects -> route to the complex branch.
    if any(
        getattr(n, "_complex_field_member", False)
        for c in constraints
        for n in _walk(_bare(c))
        if isinstance(n, (TrialFunction, TestFunction))
    ):
        return False
    return True


def _element_for(dimension: int, order: int) -> str:
    """Simplex element-type label for a ``(dimension, order)`` pair (2D/3D simplex, any order >= 1).

    Legacy names ``TRI3``/``TRI6``/``TET4``/``TET10`` for orders 1-2; a generic ``TRI-P{k}`` / ``TET-P{k}``
    for higher order. The native Lagrange assembler keys off the integer ``order`` (the basis and the
    promoted P{k} node mesh both come from the same basix element), not this string -- the label is only
    consumed by the VPINN context builder."""
    key = (int(dimension), int(order))
    if key in _ELEMENT_FOR:
        return _ELEMENT_FOR[key]
    if int(dimension) in (2, 3) and int(order) >= 1:
        return f"{'TRI' if int(dimension) == 2 else 'TET'}-P{int(order)}"
    raise ValueError(
        f"jno.fem: no built-in element for dimension {dimension}, order {order} "
        "(supported: 2D/3D simplex at order >= 1; pass element_type=... to override)."
    )


def _order_of_element(element_type: str) -> int:
    if element_type in _P2_ELEMENTS:
        return 2
    if "-P" in element_type:
        return int(element_type.split("-P")[1])
    return 1


def _field_keys(constraints: List[Any]) -> List[Any]:
    """Ordered distinct trial ``field_key``s across constraints (first appearance).

    Each ``fem_symbols()`` call is one coupled field (its trial + test share a key).
    More than one key ⇒ a coupled / mixed multi-field problem."""
    keys: List[Any] = []
    seen: set = set()
    for c in constraints:
        for n in _walk(_bare(c)):
            if isinstance(n, TrialFunction):
                k = getattr(n, "field_key", n.op_id)
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
    return keys


def _field_key_of(constraint: Any) -> Any:
    """The trial ``field_key`` of an essential (Dirichlet) constraint."""
    for n in _walk(_bare(constraint)):
        if isinstance(n, TrialFunction):
            return getattr(n, "field_key", n.op_id)
    return None


def _normalize_quad_tag(tag: Any, boundary_regions: Any) -> Any:
    """Undo a quadrature retag this module produced, so region detection is robust to a
    coordinate Variable that was already retagged by an earlier (or interleaved) ``jno.fem``
    call (``_retag_coords_for_quadrature`` mutates ``Variable.tag`` in place; reusing a stored
    coord var across two ``fem()`` calls otherwise leaves its tag as ``"gauss_<region>"`` and
    silently misclassifies the term as volume). The retag is total + reserved
    (``boundary -> "gauss_{region}"``), so stripping ``gauss_`` recovers the region exactly.

    Defensive: only strip ``gauss_<r>`` when ``<r>`` is an actual boundary region, so a
    user region with an unusual name is never silently rewritten -- we only ever undo a tag
    we ourselves produced. ``"fem_gauss"`` (volume) is left as-is and falls through to volume.
    """
    if isinstance(tag, str) and tag.startswith("gauss_"):
        region = tag[len("gauss_") :]
        if region in (boundary_regions or {}):
            return region
    return tag


def _region_and_support(constraint: Any, domain: Any) -> Tuple[str, str]:
    """Return ``(support, region_id)`` for a constraint.

    ``support`` is ``"volume"`` or ``"boundary"``; ``region_id`` is ``"volume"``
    for volume terms and the boundary tag otherwise. Region comes from the tags
    of the constraint's coordinate Variables; a constraint must resolve to a
    single region.
    """
    _bregions = getattr(domain, "_boundary_regions", {})

    def _region_of(tag: str) -> str:
        # an outward-normal Variable `n_<region>` (from domain.variable(region, normals=True)) belongs
        # to its region, not a separate one -- so `g*(v·n)` on a boundary is a single-region term.
        if tag.startswith("n_") and tag[2:] in _bregions:
            tag = tag[2:]
        return _normalize_quad_tag(tag, _bregions)

    def _effective_tag(v) -> str:
        # A coord reused from an earlier jno.fem() call has its `.tag` already rebound to the quadrature
        # pool ("fem_gauss" / "gauss_<tag>"); recover its original region from `_jno_region_tag` so a
        # sub-region term is still detected on reuse (see `_retag_coords_for_quadrature`).
        t = v.tag
        if isinstance(t, str) and (t == "fem_gauss" or t.startswith("gauss_")):
            return getattr(v, "_jno_region_tag", t)
        return t

    tags = {
        _region_of(_effective_tag(v))
        for v in _spatial_coord_vars(constraint)
        if isinstance(_effective_tag(v), str) and not _effective_tag(v).startswith("__")
    }
    # The t=t0 slice is its own support; an IC residual lives here. A *velocity* IC `u.t(initial)-v0`
    # carries its region only on the temporal variable (the `.t` derivative drops the spatial bind),
    # tagged `__time_initial__` -- detect that so a second-order velocity IC classifies as 'initial'.
    initial_temporal = any(
        isinstance(n, Variable) and getattr(n, "axis", None) == "temporal" and getattr(n, "tag", None) == "__time_initial__"
        for n in _walk(_bare(constraint))
    )
    if "initial" in tags or initial_temporal:
        return "initial", "initial"
    # Interior sub-region (sub-domain) volume term: the coords carry a registered interior region --
    # a geometry part (`_source_regions`) or a `domain.tag` predicate that is NOT a boundary region.
    # The term integrates over that region's cells only (per-cell centroid mask, applied at assembly).
    # The default whole-domain interior tags normalize to "volume", so they never match here.
    src_regions = getattr(domain, "_source_regions", {}) or {}
    tag_preds = getattr(domain, "_tag_predicates", {}) or {}
    shape_regions = getattr(domain, "_shape_regions", {}) or {}

    def _subregion_id(t: str):
        # `from_regions` registers a geometry part's interior under the tag ``interior_<name>`` (see
        # PolygonDomain._register_interior_tag) while the part itself is keyed *bare* in
        # ``_source_regions``. Map the tag back to the bare region so the per-cell ``RegionMask``
        # (resolved via ``_cell_region_mask`` -> ``_source_regions[name]``) restricts integration to it.
        # Without this, a term on ``interior_<name>`` falls through to whole-domain "volume" and the
        # per-region material / source is silently integrated over the entire mesh.
        if t in src_regions:
            return t
        if t.startswith("interior_") and t[len("interior_") :] in src_regions:
            return t[len("interior_") :]
        if t in tag_preds and t not in _bregions:
            return t
        if t in shape_regions:
            return t
        return None

    # An interior sub-region tag takes precedence over a coincidental boundary-region collision:
    # `from_regions` may also register a fully-enclosed part's `interior_<name>` tag in
    # `_boundary_regions` (its mesh boundary is a closed internal interface). For a VOLUME term (test
    # function present) that part must integrate over its *cells*, not be misread as a (face-less)
    # boundary term -> b == 0. But a TRIAL-ONLY Dirichlet `u(interior_<name>) - g` legitimately pins
    # that region's *node set* (a volumetric hard constraint), so there the boundary/node-set
    # classification is exactly what we want -- keep it.
    has_test = _contains(constraint, TestFunction)
    if has_test:
        # The whole-domain interior ("interior") is never a boundary -- even if a coarse mesh left every
        # interior node lying on the boundary and so registered "interior" in `_boundary_regions`. A
        # VOLUME weak-form term must integrate over cells, not be misread as a (face-less) boundary term
        # (this mirrors the sub-region `_subregion_id` guard: both are interior regions that happen to
        # collide with a boundary-region node set). Without this, a coarse-mesh weak form with any surface
        # term classifies the volume term as boundary -> empty `volume_terms` -> "no trial fields".
        boundary_tags = {t for t in tags if t in _bregions and _subregion_id(t) is None and t != "interior"}
    else:
        boundary_tags = {t for t in tags if t in _bregions}
    interiorish = tags - boundary_tags

    if len(boundary_tags) > 1 or (boundary_tags and interiorish):
        raise ValueError(
            f"jno.fem: a residual spans multiple regions {sorted(tags)}; each residual must live on a single region."
        )
    if boundary_tags:
        return "boundary", next(iter(boundary_tags))

    subregions = {r for r in (_subregion_id(t) for t in interiorish) if r is not None}
    if len(subregions) > 1:
        raise ValueError(
            f"jno.fem: a volume residual spans multiple sub-regions {sorted(subregions)}; "
            "each residual must live on a single region."
        )
    if subregions:
        return "volume", next(iter(subregions))
    return "volume", "volume"


def _retag_coords_for_quadrature(constraint: Any, support: str, region_id: str) -> None:
    """Point a weak term's coordinate Variables at the FEM quadrature pool.

    The assembly kernels bind a coordinate Variable to the live quadrature points
    only when its tag is ``"fem_gauss"`` (volume) or ``"gauss_<tag>"`` (surface);
    Jacobians use the coordinate's ``dim`` (axis), so derivatives are unaffected.
    """
    target = "fem_gauss" if support == "volume" else f"gauss_{region_id}"
    for v in _spatial_coord_vars(constraint):
        # outward-normal Variables (`n_<region>`) and the element-size symbol (`cell_size`) are not
        # quadrature coordinates -- leave their tag so they stay resolvable from the domain context.
        if isinstance(v.tag, str) and v.tag not in ("fem_gauss", "cell_size") and not v.tag.startswith(("gauss_", "n_")):
            # Remember the region before rebinding to the quadrature pool. The retag must persist for
            # lazy operators (nonlinear/transient re-read `.tag` at call time), but the SAME coord object
            # is often reused in a later jno.fem() call, where region detection must still recover the
            # original region -- `_region_and_support` reads `_jno_region_tag` when `.tag` is a quad tag.
            if not hasattr(v, "_jno_region_tag"):
                v._jno_region_tag = v.tag
            v.tag = target


def _constant_of(node: Any) -> Optional[float]:
    """Best-effort scalar extraction from a constant/Literal node."""
    for attr in ("value", "val", "data", "constant"):
        if hasattr(node, attr):
            try:
                return float(getattr(node, attr))
            except (TypeError, ValueError):
                return None
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


def _scalar_const(node: Any) -> Optional[complex]:
    """Complex-aware scalar extraction from a constant/Literal node (returns None if not a scalar)."""
    for attr in ("value", "val", "data", "constant"):
        if hasattr(node, attr):
            v = getattr(node, attr)
            try:
                if np.ndim(v) != 0:
                    return None
                return complex(v)
            except (TypeError, ValueError):
                return None
    try:
        return complex(node)
    except (TypeError, ValueError):
        return None


def _tie_phase(bare: Any) -> Optional[complex]:
    """The Bloch phase of a tie ``u(A) - phase*u(B)``: ``1`` for plain periodic, a complex scalar for a
    quasi-periodic (Bloch) tie, or ``None`` when the relation is not a valid tie (anti-periodic, etc.).

    The slave side (left) must be a bare trial; the master side (right) is a bare trial (phase 1) or a
    constant-scalar times a bare trial (the Bloch factor ``e^{i k·L}``)."""
    if getattr(bare, "op", None) != "-":
        return None
    left, right = bare.left, bare.right
    if getattr(left, "op", None) in {"+", "-", "*", "/"}:  # slave side must be a bare trial
        return None
    rop = getattr(right, "op", None)
    if rop is None:
        return 1.0 + 0.0j  # plain periodic  u(A) - u(B)
    if rop == "*":  # Bloch  c*u(B) (either factor order), c a constant scalar
        for a, b in ((right.left, right.right), (right.right, right.left)):
            c = _scalar_const(a)
            if c is not None and getattr(b, "op", None) is None:
                return c
    return None


def _component_index_of(node: Any) -> Optional[int]:
    """If ``node`` is a single component of the trial (``u[..., i]``), return ``i``.

    Reads the index recorded on the ``getitem`` node (``Placeholder.__getitem__``),
    so a per-component (roller) BC ``u(region)[i] - g`` is recoverable. Returns
    ``None`` for the bare trial (all components).
    """
    if getattr(node, "_name", None) == "getitem" and hasattr(node, "getitem_key"):
        args = getattr(node, "args", None) or []
        if len(args) == 1 and _contains(args[0], TrialFunction):
            ints = [k for k in node.getitem_key if isinstance(k, int)]
            if len(ints) == 1:
                return ints[0]
    return None


def _essential_spec(bare: Any) -> Tuple[Optional[int], Any]:
    """``(component_index_or_None, value_node)`` for an essential residual.

    Handles ``u(region) - g`` (component ``None`` = all components) and
    ``u(region)[i] - g`` (component ``i``). The unknown side must be the bare
    trial or a single component of it — a nonlinear/scaled form (e.g. ``u**2 - 1``)
    raises rather than being silently read as ``u - 1``.
    """
    op = getattr(bare, "op", None)
    if op == "-" and hasattr(bare, "left") and hasattr(bare, "right"):
        left, right = bare.left, bare.right
        left_has_trial = _contains(left, TrialFunction)
        right_has_trial = _contains(right, TrialFunction)
        if left_has_trial and not right_has_trial:
            trial_side, value_node = left, right
        elif right_has_trial and not left_has_trial:
            trial_side, value_node = right, left
        else:
            trial_side = value_node = None
        if value_node is not None:
            comp = _component_index_of(trial_side)
            if not (isinstance(trial_side, TrialFunction) or comp is not None):
                raise ValueError(
                    "jno.fem: an essential boundary/initial condition must be affine in the unknown — "
                    f"write `u(region) - value` or `u(region)[i] - value`, not a nonlinear/scaled trial "
                    f"expression. Got: {trial_side!r}."
                )
            return comp, value_node
    if isinstance(bare, TrialFunction):
        return None, 0.0
    raise ValueError(
        "jno.fem: could not read an essential condition from the residual. "
        "Write it as `u(region) - value` (e.g. `u(xl, yl) - 0.0`) or `u(xl, yl)[i] - value`."
    )


def _eval_value_node_at(value_node: Any, points: Any, params: Any = None) -> Any:
    """Evaluate a coordinate value expression at ``points`` (1-D result).

    Reuses the existing :class:`~jno.trace_evaluator.TraceEvaluator` (the engine
    behind ``.eval``) — the coordinates are concrete (the mesh is built), so this
    is a single forward pass. No bespoke expression walker is introduced.

    ``params`` (a ``{name: value}`` map of runtime parameters) substitutes the referenced trainable
    parameters into the evaluation, so the result stays **differentiable** in them — the coefficient of a
    parametric natural / surface boundary term (an inverse-design impedance / incident source). When it is
    ``None`` (the default) the parameters keep their stored values (a plain forward pass).
    """
    from .trace_evaluator import TraceEvaluator

    tags = {v.tag for v in _walk(value_node) if isinstance(v, Variable)}
    pts = jnp.atleast_2d(jnp.asarray(points))
    # Register every ModelCall (parameter / network) so the evaluator resolves it. With ``params`` (the
    # runtime ``args``), substitute a trainable parameter's value OR a trainable network's live module from
    # it, so the coefficient stays differentiable in the design variable; else use the stored module
    # (a plain forward pass / a frozen net).
    table: dict = {}
    _mcs = [nd for nd in _walk(value_node) if type(nd).__name__ == "ModelCall"]
    if _mcs:
        import equinox as eqx

        from .utils.solver.parametric_helpers import _neural_coefficient_name

        for nd in _mcs:
            m = nd.model
            mod = m.module
            if params:
                if getattr(m, "_is_parameter", False):
                    pn = getattr(m, "_parameter_name", None)
                    if pn is not None and pn in params:  # trainable parameter: substitute its .value
                        mod = eqx.tree_at(lambda mm: mm.value, mod, jnp.asarray(params[pn]))
                elif (nn := _neural_coefficient_name(nd)) in params:  # trainable network: its live module
                    mod = params[nn]
            table[m.layer_id] = mod
    return jnp.reshape(TraceEvaluator(table).evaluate(value_node, context={t: pts for t in tags}), (-1,))


def _coord_value_fn(value_node: Any) -> Callable:
    """A ``value(point)`` callable for a coordinate Dirichlet value — the same
    per-point hook ``domain.dirichlet("left", lambda p: p[1])`` uses."""

    def value_fn(p):
        p_arr = jnp.asarray(p)
        out = _eval_value_node_at(value_node, jnp.atleast_2d(p_arr))
        return out[0] if p_arr.ndim == 1 else out

    return value_fn


def _value_from_node(node: Any) -> Any:
    """Constant or ``value(point)`` callable from an essential value node."""
    const = _constant_of(node)
    return const if const is not None else _coord_value_fn(node)


def _node_has_complex_literal(node: Any) -> bool:
    """True if a single trace node carries a complex-valued constant (a user-written ``1j``)."""
    for attr in ("value", "val", "data", "constant"):
        v = getattr(node, attr, None)
        if v is None:
            continue
        try:
            if jnp.iscomplexobj(jnp.asarray(v)):
                return True
        except Exception:  # not array-like; ignore
            continue
    return False


def _bares_have_complex_coeff(bares: Any) -> bool:
    """True if any raw weak-form bare contains a complex literal. The complex constant survives
    lowering unchanged, so walking the raw bares is equivalent to :func:`_is_complex_form` but works
    *before* the lowered ``ir`` is built (used to route the non-nodal path)."""
    return any(_node_has_complex_literal(n) for b in bares for n in _walk(b))


def _is_complex_form(domain: Any, ir: Any) -> bool:
    """True if any lowered term's expression contains a complex constant — the signal to route a
    (steady, linear) weak form through the real-equivalent complex solver instead of the plain real
    assembly. We walk the tree for a complex-valued literal (a user-written ``1j``) rather than
    evaluating, because the lowered coeff embeds the (non-evaluable) trial/test channel."""
    del domain
    return any(_node_has_complex_literal(node) for term in ir.terms for node in _walk(term.coeff))


def _complex_block_bcoo(A_r: Any, A_i: Any, n: int) -> Any:
    """Assemble the real-equivalent block ``[[A_r, -A_i], [A_i, A_r]]`` (shape ``2n x 2n``) as one
    BCOO from the two sparse legs, by placing each sub-block's triplets at its ``n`` offset -- no
    densification. The four sub-blocks occupy disjoint index ranges, so the result has no duplicate
    coordinates. ``A_r`` appears at (0,0) and (n,n); ``A_i`` at (n,0) and, negated, at (0,n)."""
    from jax.experimental import sparse as jsp

    ir, dr = A_r.indices, A_r.data
    ii, di = A_i.indices, A_i.data
    off = jnp.asarray([n, n], dtype=ir.dtype)
    col = jnp.asarray([0, n], dtype=ir.dtype)
    row = jnp.asarray([n, 0], dtype=ir.dtype)
    indices = jnp.concatenate([ir, ii + col, ii + row, ir + off], axis=0)  # (0,0),(0,n),(n,0),(n,n)
    data = jnp.concatenate([dr, -di, di, dr])
    return jsp.BCOO((data, indices), shape=(2 * n, 2 * n))


def _to_bcoo(A: Any) -> Any:
    """Coerce ``A`` to a BCOO (identity if already sparse) — the native assembler returns BCOO, so this is
    a defensive no-op that also lets a dense operator flow into the sparse block composers below."""
    from jax.experimental import sparse as jsp

    return A if hasattr(A, "indices") else jsp.BCOO.fromdense(jnp.asarray(A))


def _bcoo_empty(m: int, n: int, dtype: Any) -> Any:
    """An all-zero ``(m, n)`` BCOO (no stored entries)."""
    from jax.experimental import sparse as jsp

    return jsp.BCOO((jnp.zeros((0,), dtype), jnp.zeros((0, 2), jnp.int32)), shape=(m, n))


def _bcoo_block(subblocks: Any, shape: Any, dtype: Any) -> Any:
    """Assemble a block matrix of ``shape`` from ``(sub, row_off, col_off, scale)`` entries — each ``sub`` a
    BCOO placed (and scaled) at its ``(row_off, col_off)`` offset. Composes in ``O(Σ nnz)`` without ever
    forming a dense intermediate (BCOO sums any coincident coordinates on matvec / todense); a ``None`` sub
    is the zero block. This is the sparse analogue of ``jnp.block`` for the augmented / saddle systems."""
    from jax.experimental import sparse as jsp

    idx, dat = [], []
    for sub, roff, coff, scale in subblocks:
        if sub is None:
            continue
        idx.append(sub.indices + jnp.asarray([roff, coff], dtype=sub.indices.dtype))
        dat.append(scale * sub.data if scale != 1.0 else sub.data)
    if not idx:
        return jsp.BCOO((jnp.zeros((0,), dtype), jnp.zeros((0, 2), jnp.int32)), shape=shape)
    return jsp.BCOO((jnp.concatenate(dat), jnp.concatenate(idx)), shape=shape)


def _complex_operator(A_r: Any, A_i: Any) -> Any:
    """Fuse the real/imag legs into one complex operator ``A_r + i·A_i`` — as a **complex BCOO** when
    the legs are sparse (so an iterative complex solve never densifies), else a dense complex array.
    This is the operator a slot-composed complex solver (``linear=gmres, precond=ams``) runs on."""
    if hasattr(A_r, "indices") and hasattr(A_i, "indices"):
        from jax.experimental import sparse as jsp

        indices = jnp.concatenate([A_r.indices, A_i.indices], axis=0)
        data = jnp.concatenate([A_r.data + 0j, 1j * A_i.data])  # +0j promotes to the complex dtype
        return jsp.BCOO((data, indices), shape=A_r.shape).sum_duplicates()
    Ar = A_r.todense() if hasattr(A_r, "todense") else A_r
    Ai = A_i.todense() if hasattr(A_i, "todense") else A_i
    return jnp.asarray(Ar) + 1j * jnp.asarray(Ai)


def _solve_complex_block(
    ops: Any,
    solve_fn: Optional[Callable] = None,
    periodic: Optional[dict] = None,
    complex_solve: Optional[Callable] = None,
    from_slots: bool = False,
) -> Any:
    """Solve a complex linear FEM system via the real-equivalent block (everything was
    assembled as two *real* systems)::

        [[A_r, -A_i], [A_i, A_r]] [u_r; u_i] = [b_r; b_i],     u = u_r + i u_i.

    ``ops = (op_r, op_i)`` are the Re-coeff and Im-coeff real systems (each a raw ``(A, b)``).

    The real-equivalent block is itself **sparse** -- ``[[A_r,-A_i],[A_i,A_r]]`` is four sparse
    sub-blocks -- so by default it is assembled as one BCOO (concatenating the triplets at the right
    ``n`` offsets) and solved with the same sparse-direct path the real solves use, never densifying
    (memory ``O(nnz)``; robust on the indefinite Helmholtz block). Pass ``solve_fn=(A, b) -> u`` to
    use a dense/direct solver instead -- it receives the densified block.

    Each leg is a raw ``(A, b)`` (forward solve) or a parametric :class:`FemLinearSystem` (the complex
    *inverse*): when either leg is parametric this returns a differentiable :class:`FunctionCall` trace
    node instead of an array, so a real ``jno.np.parameter`` recovered through a complex forward solve
    trains through ``crux`` -- the same real-equivalent block, with ``A(θ), b(θ)`` re-formed and the
    block re-solved on each call.

    A periodic tie reduces *both* legs by the **same** prolongation ``P`` -- Re and Im live on one FE
    space, so one ``P`` -- before the block is formed: ``A -> P^T A P``, ``b -> P^T b`` on each leg.
    Because the same ``P`` hits both, ``blkdiag(P, P)^T [[A_r,-A_i],[A_i,A_r]] blkdiag(P, P)`` preserves
    the real-equivalent structure exactly; the reduced complex solution is prolonged back ``u = P u_red``."""
    from .trace import FemLinearSystem, FunctionCall
    from .utils.solver.fem_utils import prolong_periodic, reduce_matrix_periodic, reduce_vector_periodic

    bloch = bool(periodic is not None and periodic.get("is_bloch", False))

    def _ab(op, args, conj=False):
        A, b = op.evaluate(args) if isinstance(op, FemLinearSystem) else op
        b = jnp.asarray(b).reshape(-1)
        if periodic is not None:
            A = reduce_matrix_periodic(periodic, A, conj=conj)  # P^T A P (P^H when Bloch); stays sparse
            b = reduce_vector_periodic(periodic, b, conj=conj)  # P^T b (P^H when Bloch)
        return A, b  # keep A sparse (BCOO) -- do NOT densify

    def _bloch_solve(args):
        # Bloch/quasi-periodic: P is complex, so the reduced operator A_c = P^H (A_r + i A_i) P mixes
        # Re/Im and cannot be split into independent legs. Reduce Hermitian, form the complex reduced
        # operator, solve, and prolong with the complex P.
        (A_r, b_r), (A_i, b_i) = _ab(ops[0], args, conj=True), _ab(ops[1], args, conj=True)
        b_c = jnp.asarray(b_r) + 1j * jnp.asarray(b_i)
        n = b_c.shape[0]
        if hasattr(A_r, "indices") and hasattr(A_i, "indices"):
            # Stay SPARSE. `reduce_matrix_periodic` already keeps the reduction sparse, so densifying here
            # just to call `jnp.linalg.solve` reintroduced the O(n²) peak it exists to avoid — tolerable for
            # a small reduced cell, ruinous for a 3-D metasurface unit cell. Both reduced legs are
            # themselves COMPLEX (P^H is complex even for a real leg), so fuse them into A_c = A_r + i·A_i
            # over the concatenated index set, split into Re/Im parts sharing that index set, and
            # sparse-direct-solve the real-equivalent 2n block. Repeated (i, j) entries from the
            # concatenation are merged by `sparse_lu_solve`'s `sum_duplicates`.
            from jax.experimental import sparse as jsp

            from .utils.solver.linear import sparse_lu_solve

            idx = jnp.concatenate([A_r.indices, A_i.indices], axis=0)
            dat = jnp.concatenate([jnp.asarray(A_r.data), 1j * jnp.asarray(A_i.data)])
            re = jsp.BCOO((jnp.real(dat), idx), shape=(n, n))
            im = jsp.BCOO((jnp.imag(dat), idx), shape=(n, n))
            sol = sparse_lu_solve(_complex_block_bcoo(re, im, n), jnp.concatenate([jnp.real(b_c), jnp.imag(b_c)]))
            u_red = sol[:n] + 1j * sol[n:]
        else:  # dense fallback (1-D, or already-dense legs)
            to_d = lambda M: jnp.asarray(M.todense() if hasattr(M, "todense") else M)  # noqa: E731
            u_red = jnp.linalg.solve(to_d(A_r) + 1j * to_d(A_i), b_c)
        return prolong_periodic(periodic, u_red)

    def _block_solve(args):
        if bloch:
            return _bloch_solve(args)
        (A_r, b_r), (A_i, b_i) = _ab(ops[0], args), _ab(ops[1], args)
        n = b_r.shape[0]

        if complex_solve is not None:
            # Slot-composed iterative path (linear=/precond=): solve the COMPLEX system directly on the
            # sparse operator A_r + i·A_i (never the dense 2n block), e.g. gmres + jno.precond.ams.
            A_c = _complex_operator(A_r, A_i)
            u = complex_solve(A_c, b_r + 1j * b_i)
            return (
                u
                if periodic is None
                else prolong_periodic(periodic, jnp.real(u)) + 1j * prolong_periodic(periodic, jnp.imag(u))
            )

        rhs = jnp.concatenate([b_r, b_i])
        is_sparse = hasattr(A_r, "indices") and hasattr(A_i, "indices")

        if solve_fn is not None and from_slots and is_sparse:
            # Slot-composed solver on a non-complex-native precond (gmres + jacobi/form/amg): hand it the
            # SPARSE 2n BCOO block [[A_r,-A_i],[A_i,A_r]] rather than densifying — keeps it O(nnz) AND lets
            # the composed solver's extreme-magnitude normalization fire (a dense block has no .bcoo, so a
            # mass-dominated eddy system would otherwise stall the Krylov Arnoldi at real scale).
            sol = jnp.asarray(solve_fn(_complex_block_bcoo(A_r, A_i, n), rhs))
        elif solve_fn is not None:
            # raw user solver (or a dense-leg fallback) receives the densified block (dense/direct back-compat)
            Ar_d = jnp.asarray(A_r.todense() if hasattr(A_r, "todense") else A_r)
            Ai_d = jnp.asarray(A_i.todense() if hasattr(A_i, "todense") else A_i)
            sol = jnp.asarray(solve_fn(jnp.block([[Ar_d, -Ai_d], [Ai_d, Ar_d]]), rhs))
        elif is_sparse:
            from .utils.solver.linear import sparse_lu_solve

            sol = sparse_lu_solve(_complex_block_bcoo(A_r, A_i, n), rhs)  # sparse-direct on the 2n block
        else:
            Ar_d, Ai_d = jnp.asarray(A_r), jnp.asarray(A_i)  # dense fallback (e.g. 1D / already dense)
            sol = jnp.linalg.solve(jnp.block([[Ar_d, -Ai_d], [Ai_d, Ar_d]]), rhs)

        if periodic is None:
            return sol[:n] + 1j * sol[n:]
        # Prolong the real and imaginary reduced halves *separately* (real P @ real vector); prolonging
        # the complex vector directly would cast it to P's real dtype and silently drop the imaginary part.
        return prolong_periodic(periodic, sol[:n]) + 1j * prolong_periodic(periodic, sol[n:])

    # Forward (non-parametric): solve eagerly. Inverse: one or both legs are a parametric FemLinearSystem
    # -> return a trace node over the union of their runtime parameters so ∂u/∂θ flows through crux.
    rpe: dict = {}
    for op in ops:
        if isinstance(op, FemLinearSystem):
            rpe.update(op.runtime_parameter_exprs)
    if not rpe:
        return _block_solve(None)
    names = list(rpe)
    return FunctionCall(
        lambda *vals: _block_solve(dict(zip(names, vals))), [rpe[n] for n in names], name="fem_complex_solve"
    )


def _is_temporal_value_node(node: Any) -> bool:
    """True if an essential value carries a temporal Variable (time-varying ``g(x,t)``)."""
    return any(isinstance(v, Variable) and getattr(v, "axis", None) == "temporal" for v in _walk(node))


def _eval_value_node_at_time(value_node: Any, points: Any, t: Any) -> Any:
    """Evaluate a time-dependent coordinate value ``g(x, t)`` at ``points`` and time ``t``.

    Like :func:`_eval_value_node_at`, but the **temporal** Variable carries its own tag
    (``'__time__'``, separate from the spatial coordinate tag), so its context entry is a
    ``(n, 1)`` array filled with ``t`` while each spatial tag maps to the points. (Mapping
    every tag to the points, as the steady evaluator does, makes the time variable read a
    spatial column.) Reuses the existing ``TraceEvaluator``."""
    from .trace_evaluator import TraceEvaluator

    pts = jnp.atleast_2d(jnp.asarray(points))
    ctx: dict = {}
    for v in _walk(value_node):
        if isinstance(v, Variable):
            ctx[v.tag] = jnp.full((pts.shape[0], 1), t, dtype=pts.dtype) if getattr(v, "axis", None) == "temporal" else pts
    return jnp.reshape(TraceEvaluator({}).evaluate(value_node, context=ctx), (-1,))


def _dirichlet_spec(bare: Any) -> Tuple[Optional[int], Any, Any]:
    """``(component_index_or_None, value, value_node)`` for an essential Dirichlet residual.

    ``component`` is ``None`` for an all-component clamp (``u(region) - g``) or an integer
    for a per-component / roller BC. ``value`` is a constant or ``value(point)`` callable;
    ``value_node`` is the raw expression (kept so the transient route can detect/evaluate a
    time-varying ``g(x,t)``)."""
    comp, value_node = _essential_spec(bare)
    if _is_bare_neural_value_node(value_node):
        # A trainable BC profile ``u(region) - net(x)``: keep the net node raw and leave ``value`` as a
        # sentinel -- the native parametric path evaluates ``net(boundary_coords)`` from the runtime args
        # each solve (``_value_from_node`` would build a stored-weights closure, non-differentiable).
        return comp, None, value_node
    return comp, _value_from_node(value_node), value_node


def _is_bare_neural_value_node(value_node: Any) -> bool:
    """True if a Dirichlet value node is a *bare* neural coefficient ``net(x)`` (a trainable BC profile).

    Only a bare network call is supported as a trainable Dirichlet value; a compound expression such as
    ``1 + net(x)`` is rejected up front (it would need a general args-aware value evaluator)."""
    from .utils.solver.parametric_helpers import _is_neural_coefficient

    return _is_neural_coefficient(_bare(value_node)) if value_node is not None else False


def _normal_flux_spec(constraint: Any, domain: Any) -> Optional[Tuple[Any, str, Any]]:
    """Recognise an essential **normal-flux** BC ``u·n - g`` (H(div) RT) -> ``(field_key, region,
    value_node)``; ``None`` otherwise.

    The trial side is affine in ``u`` and references a boundary outward-normal Variable (tag
    ``n_<region>``, from ``domain.variable(region, normals=True)``); there is no test function (it is
    essential, not a weak Neumann term). The physical normal is recomputed per boundary edge at
    assembly, so only the region and the prescribed value ``g`` (``value_node``) are carried. Separated
    out before classification (like periodic ties) so it never reaches the Cartesian Dirichlet parser."""
    bare = _bare(constraint)
    if getattr(bare, "op", None) != "-" or _contains(bare, TestFunction):
        return None
    left, right = getattr(bare, "left", None), getattr(bare, "right", None)
    if left is None or right is None:
        return None
    left_trial, right_trial = _contains(left, TrialFunction), _contains(right, TrialFunction)
    if left_trial == right_trial:  # need exactly one side with the trial (the u·n expression)
        return None
    trial_side, value_node = (left, right) if left_trial else (right, left)
    normals = [
        n
        for n in _walk(trial_side)
        if isinstance(n, Variable) and isinstance(getattr(n, "tag", None), str) and n.tag.startswith("n_")
    ]
    if not normals:
        return None
    return _field_key_of(constraint), normals[0].tag[2:], value_node


def _tangential_bc_spec(constraint: Any, domain: Any) -> Optional[Tuple[Any, str, Any]]:
    """Recognise a homogeneous **PEC** tangential BC ``n × u = 0`` (H(curl) N1E, 3-D) -> ``(field_key,
    region, 0.0)``; ``None`` otherwise.

    The constraint is the cross product of the N1E trial with the region's boundary normal
    (``u.vector.cross(nvec)``, ``nvec`` built from ``domain.variable(region, normals=True)`` — tag
    ``n_<region>``), with no test function. Only the **homogeneous** perfect-electric-conductor case
    (tangential trace zero) is wired; an inhomogeneous ``n × u = g`` (nonzero rhs) is not (returns ``None``,
    so it falls through rather than silently zeroing). Peeled before classification like the RT normal-flux
    BC; routed into ``flux_bcs`` where the N1E-3D branch pins every boundary-face edge DOF in the region."""
    bare = _bare(constraint)
    if getattr(bare, "op", None) == "-":  # `cross(...) - rhs`: PEC only, so require rhs == 0
        left, right = getattr(bare, "left", None), getattr(bare, "right", None)
        if _constant_of(right) == 0.0:
            bare = _bare(left)
        elif _constant_of(left) == 0.0:
            bare = _bare(right)
        else:
            return None  # inhomogeneous n×u=g not wired
    if not (isinstance(bare, FunctionCall) and getattr(bare, "_name", None) == "cross"):
        return None
    if _contains(bare, TestFunction) or not _contains(bare, TrialFunction):
        return None
    normals = [
        n
        for n in _walk(bare)
        if isinstance(n, Variable) and isinstance(getattr(n, "tag", None), str) and n.tag.startswith("n_")
    ]
    if not normals:
        return None
    return _field_key_of(constraint), normals[0].tag[2:], 0.0


def _rotation_bc_spec(constraint: Any, domain: Any) -> Optional[Tuple[Any, str, Any]]:
    """Recognise an essential **rotation** BC ``u.dn(region) - h`` (``∂u/∂n = h`` on a C¹/Morley plate field)
    -> ``(field_key, region, value_node)``; ``None`` otherwise.

    The trial side is a :class:`~jno.trace.NormalDerivative` marker and there is no test function (it is
    essential, not a natural moment term). Peeled before classification (like the RT normal-flux BC): the
    physical outward normal is recomputed per boundary edge at assembly, so only the region and the
    prescribed value ``h`` are carried."""
    bare = _bare(constraint)
    if getattr(bare, "op", None) != "-" or _contains(bare, TestFunction):
        return None
    left, right = getattr(bare, "left", None), getattr(bare, "right", None)
    if left is None or right is None:
        return None
    left_nd, right_nd = isinstance(left, NormalDerivative), isinstance(right, NormalDerivative)
    if left_nd == right_nd:  # exactly one side is the ∂u/∂n marker
        return None
    value_node = right if left_nd else left
    if _contains(value_node, TrialFunction):  # the prescribed value must not contain the unknown
        return None
    support, region = _region_and_support(constraint, domain)
    if support != "boundary":
        return None
    return _field_key_of(constraint), region, value_node


# ---------------------------------------------------------------------------
# the FEM container
# ---------------------------------------------------------------------------
class FEM:
    """Assembled FEM artefacts produced by :func:`fem` (no solve).

    Attributes
    ----------
    domain, mesh, problem:
        the owning jNO domain, its meshio mesh, and the assembled problem object.
    dofs:
        total number of degrees of freedom.
    classification:
        human-readable summary of how each residual was bucketed.
    """

    def __init__(self, domain: Any, op: Any, classification: List[str], *, mode: str, offsets: Any = None):
        # mode: "linear" | "nonlinear" | "transient"
        self.domain = domain
        self._op = op
        self._mode = mode
        self.classification = classification
        self.mesh = getattr(domain, "mesh", None)
        self.problem = getattr(domain, "_fem_problem", None)
        self._periodic = None  # periodic-tie reduction (prolongation P), attached by fem()
        self._complex_n = None  # half-size n when _op is a fused complex real-equivalent 2n system
        # The unfused (re, im) legs. KNOWN CONSUMERS — check all of them before changing this or the
        # fused layout, since a stale reader does not crash (see the `operator` docstring: the fused
        # value is also a 2-tuple, so it keeps unpacking and silently means something else):
        #   * jno.precond            — `_complex_operator(A_r, A_i)` for the complex-native AMS route
        #   * jno.rcwa `_source_kin` — reads the incident load off the imaginary leg
        #   * FEM.solve              — the complex-native precond branch
        #   * tests/test_fem_nedelec_{complex,anisotropic,impedance,incident}.py,
        #     tests/test_fem_nonnodal_sparse_assembly.py — assert on the legs directly
        self._complex_legs = None
        self._periodic_2n = None  # blkdiag(P, P): the periodic reduction of that 2n system
        #: relative residual of the full system after the most recent ``solve(basis=...)`` — ``None``
        #: for a full solve, or when the reduced one was traced/parametric (nothing concrete to measure).
        self.basis_residual = None
        self._offsets = offsets  # per-field block offsets for the native non-nodal path (else problem.offset)
        self._term_source = None  # (domain, volume_terms); attached by fem() for the provisional term_kinds accessor
        self._constraints = None  # original constraint list; attached by fem() for the adaptive driver
        self._fem_kwargs = {}  # original fem() build options; attached by fem() for the adaptive driver

        self._A = self._b = None
        if mode == "linear":
            # the assembler returns either a raw (A, b) tuple (non-parametric)
            # or a FemLinearSystem (runtime-parametric).
            if isinstance(op, FemLinearSystem):
                self._A, self._b = op.A, op.b
            else:
                self._A, self._b = op[0], op[1]

    @property
    def is_transient(self) -> bool:
        return self._mode == "transient"

    @property
    def is_linear(self) -> bool:
        if self._mode == "transient":
            return bool(self._op.is_linear())
        return self._mode == "linear"

    @property
    def is_complex(self) -> bool:
        """A complex-valued problem (steady or transient), solved via the real-equivalent block.

        Both steady and transient complex problems are fused into one real 2n system at assembly, so
        they are ordinary ``"linear"`` / ``"transient"`` modes: a steady one announces itself through
        ``_complex_n`` (the half size), a transient one through its block's ``metadata["complex"]``.
        The surviving ``"complex"`` mode is the **Bloch** case alone, whose complex prolongation does
        not split into two real legs."""
        if self._mode == "complex" or self._complex_n is not None:
            return True
        return bool((getattr(self._op, "metadata", None) or {}).get("complex"))

    @property
    def operator(self) -> Any:
        """The raw assembled block — ``(A, b)`` / ``FemLinearSystem`` /
        ``FemResidualOperator`` (steady) or ``SemidiscreteTimeBlock`` (transient).

        **A COMPLEX problem returns the fused real ``2n`` block**, i.e. an ordinary ``(A, b)`` over
        ``x = [x_r; x_i]`` — NOT the ``(re, im)`` leg pair it used to. Read the unfused legs off
        :attr:`_complex_legs` instead.

        Beware when migrating code: the fused value is *also* a 2-tuple, so ``op_r, op_i =
        fem.operator`` keeps unpacking cleanly and silently binds ``op_i`` to the LOAD VECTOR. It
        does not raise; it just means something else. That is what broke ``jno.rcwa`` and made a
        test report a missing imaginary part that was there all along."""
        return self._op

    @property
    def term_kinds(self):
        """PROVISIONAL — structural classification of each additively-split volume (PDE) term.

        Returns ``list[TermKind]`` (see :mod:`jno.utils.solver.term_kind`) labelling each term
        local/global (``is_local``), its temporal order, trial/test spatial-gradient channel, and
        linearity — the basis for operator-splitting routing. ``None`` on assembly paths that do
        not expose their source terms. API may change once the routing pass lands.
        """
        if self._term_source is None:
            return None
        from .utils.solver.term_kind import classify_term
        from .utils.solver.weak_form import _split_additive_terms

        domain, vterms = self._term_source
        return [classify_term(domain, sub) for vt in vterms for _sign, sub in _split_additive_terms(domain, vt)]

    @property
    def offsets(self) -> Any:
        """Per-field block offsets into the flat solution: ``sol[offsets[i]:offsets[i+1]]`` is field ``i``.

        Set for the native non-nodal (RT/P0) path; falls back to the ``problem.offset`` for the
        Lagrange multi-field path (``None`` for a single field with no block structure)."""
        if self._offsets is not None:
            return list(self._offsets)
        return getattr(self.problem, "offset", None)

    @property
    def blocks(self):
        """Per-field DOF ``slice``s into the flat solution (``None`` without block structure).

        The structural handle block preconditioners build on: ``jno.precond.block_diag`` /
        ``triangular`` resolve their field arguments to these slices (see ``docs/fem.md``)."""
        off = self.offsets
        if off is None:
            return None
        return [slice(int(off[i]), int(off[i + 1])) for i in range(len(off) - 1)]

    def block_index(self, field) -> int:
        """Resolve a trial symbol (or plain index) to its position in :attr:`blocks` /
        :attr:`offsets` — the field order is first appearance in the ``jno.fem`` constraints."""
        if isinstance(field, int):
            return field
        # the native assembler records the keys in assembly (= offsets) order — snapshotted onto
        # this FEM at finalize time (the domain attribute is overwritten by any later assembly on
        # the same domain, e.g. an auxiliary jno.precond.form); the constraint-walk order is only
        # a fallback for paths that don't set it
        keys = getattr(self, "_block_field_keys", None) or getattr(self, "_trial_field_keys", None)
        fk = getattr(field, "field_key", None)
        if keys is None or fk is None:
            raise TypeError(
                "FEM.block_index: cannot resolve this object to a field block — pass the trial "
                "symbol from d.fem_symbols() (or the integer block index)."
            )
        if fk not in keys:
            raise KeyError(f"FEM.block_index: trial field {getattr(field, 'name', fk)!r} is not part of this system.")
        return keys.index(fk)

    def solve(
        self,
        solve_fn=None,
        *,
        adapt=None,
        move=None,
        x0=None,
        nonlinear=None,
        linear=None,
        precond=None,
        time=None,
        basis=None,
        profile=False,
        **kwargs,
    ) -> Any:
        """Differentiable forward solve as a trace node — the inverse-problem entry.

        Pass ``adapt=AdaptSpec(...)`` to run the **adaptive** loop
        (``solve -> estimate -> mark -> refine``): the domain is remeshed in place to
        equidistribute a Zienkiewicz–Zhu error estimate, and this returns the solution
        on the final adapted mesh. After it returns, this ``FEM`` and its ``domain`` refer
        to that final mesh, and ``fem.adapt_history`` records the per-round trace. The
        refinement step is non-differentiable (discrete remeshing); gradients are exact on
        the frozen final mesh, so a differentiable inverse problem is run *after* adapting.
        See :class:`jno.utils.solver.fem_adapt.AdaptSpec`.

        Pass ``move=MovingBoundary(velocity=...)`` on a **transient** problem to run the
        **moving-boundary** loop: the domain boundary moves each step by the prescribed velocity, the
        mesh deforms to follow it (harmonic interior extension), and the state is carried across each
        move — for free-surface / deforming-domain physics. Returns an ``AdaptiveTrajectory`` (each frame
        on its own moved mesh). See :class:`jno.utils.solver.fem_adapt.MovingBoundary` for the method and
        its scope (operator-split ALE; scalar-P1, real, prescribed velocity for now).

        Delegates to :meth:`FemLinearSystem.solve` (steady linear),
        :meth:`FemResidualOperator.solve` (steady nonlinear), or
        :meth:`SemidiscreteTimeBlock.solve` (transient). The result is a jNO field: compare it
        to data and train any ``jno.np.parameter`` in the weak form through
        ``crux.solve``::

            alpha = jno.np.parameter((1,), name="alpha")
            fem = jno.fem([alpha * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
            crux = jno.core([(fem.solve() - u_obs).mse])   # domain inferred from the solve node
            crux.solve(n)                      # recovers alpha

        ``solve_fn`` is **your** solver (jNO writes none):

        * steady linear: ``(A, b) -> u``. The **default** is a matrix-free **Jacobi-preconditioned**
          BiCGStab on the BCOO operator — never forms the ``N x N`` matrix (memory ``O(nnz)``),
          GPU-safe, handles general (non-symmetric) systems, ``jit`` + grad; it raises on
          non-convergence. For the indefinite saddle-point systems where Jacobi does not help, pass a
          **direct** ``solve_fn`` — ``jno.utils.solver.linear.sparse_lu_solve`` (JAX ``spsolve``, no
          dependency, robust on CPU) or ``jnp.linalg.solve`` (dense); it receives the densified
          ``(A, b)``;
        * steady nonlinear: ``(residual_fn, u0) -> u`` (default a matrix-free Jacobian-free
          Newton-Krylov, implicit-diff, no optimistix; pass ``u0=`` for the guess);
        * transient: ``(block, args, save_ts) -> ys`` returning a ``(len(save_ts),
          n_dofs)`` trajectory (default a backward-Euler ``lax.scan`` over the block's assembled
          ``dt``, each step solved by the same matrix-free Newton-Krylov; ``save_ts=`` overrides
          the sample times, default the domain's time grid). For a custom integrator build it
          from the block's ``M`` / ``A`` / ``state0`` and pass it as ``solve_fn``.

        Enable x64 — the assembly is float64.

        For a **complex** steady linear problem (complex coefficients in the weak form),
        ``solve()`` returns the complex solution ``u_r + i·u_i`` via the real-equivalent block
        ``[[A_r,-A_i],[A_i,A_r]]`` (the assembly produces only real systems); pass ``solve_fn=(A, b) -> u``
        to choose the real block solver.

        **Solver slots** (callables-only; alternative to ``solve_fn``, which stays the total
        override — passing both is an error). The solver space factorises into four orthogonal
        slots; each takes a configured callable from ``jno.solve`` / ``jno.precond`` (or your
        own with the same contract, see those modules), and every ``None`` keeps today's
        default::

            fem.solve(
                x0        = None,                   # warm start (initial guess / iterate)
                nonlinear = jno.solve.newton(),     # linearization driver (nonlinear mode)
                linear    = jno.solve.gmres(),      # inner linear solve
                precond   = jno.precond.jacobi(),   # v -> M^{-1} v spec, materialized per solve
            )

        The slots compose into a ``solve_fn`` internally, so each dispatch path below keeps its
        periodic reduction and implicit-differentiation behaviour; slot solvers receive the
        **BCOO** operator (no densification). On the (matrix-free) **nonlinear** path a
        ``precond`` spec is materialized per Newton/Picard linearization against the JVP
        operator — ``form`` / ``inner(...)`` / ``chebyshev`` / pre-built ``amg`` and their
        ``block_diag``/``triangular`` compositions work; ``jacobi`` (needs the assembled
        diagonal) does not. On a **transient** problem the slots configure the *per-step*
        solves of the default theta-stepper: ``linear``/``precond`` see the step operator
        ``M + θ·dt·A`` (materialized once, before the time loop, when the operator is
        time-independent — an AMG hierarchy / auxiliary form is then reused by every step),
        ``nonlinear`` drives each implicit step, each step warm-starts from the previous state
        (``x0`` is rejected — the ICs own the initial state). Not yet supported: slots on
        complex/complex-transient problems, and slots combined with ``adapt=`` (remeshing
        invalidates warm starts and cached preconditioner setups — pass ``solve_fn=`` there).

        ``basis=U`` solves in the span of an ``(n_dofs, k)`` **Galerkin basis** instead of the full space:
        the reduced system ``UᵀA U x = Uᵀb`` is ``k×k``, and the answer is lifted back to ``u = U x`` so
        nothing downstream changes. This is the reduced-order-model (POD) entry — build ``U`` from a few
        full solves with :func:`jno.solve.svd`, then every later solve in the family costs ``k`` unknowns::

            snapshots = jnp.stack([build(p).solve() for p in sweep])   # (n_snapshots, n_dofs)
            _, s, Vt  = jno.solve.svd(snapshots, k=10)
            u = build(p_new).solve(basis=Vt.T)                         # 10 unknowns, full field back

        Unlike every other way of solving here, this returns an **approximation**, so the relative residual
        of the FULL system at the lifted solution is measured (one matvec) and a basis that does not span
        the solution fails loud rather than returning a plausible wrong field — the measured value stays on
        ``fem.basis_residual``, and ``fem.BASIS_RESIDUAL_LIMIT`` relaxes the bar for deliberately coarse
        work. ``U`` must be orthonormal. It composes with ``linear=``/``precond=`` (which see the reduced
        operator) and is differentiable in the basis itself under ``jax.grad`` — the learned-subspace path.
        Steady only (linear and nonlinear); transient, complex and periodic-tied problems refuse with a
        reason. A reduced NONLINEAR solve returns a deferred trace node, as a periodic one does. Nonlinear
        reduction is a memory win, not a speed one: the full-order residual is still evaluated per Newton
        step (no hyper-reduction).

        ``profile=True`` runs the (eager, non-parametric) solve inside a JAX Perfetto trace, prints the DOF
        count + wall time, and writes the trace to ``./jno_traces`` — like ``jno.core.solve(profile=True)``.
        Profile a *concrete* forward solve; a parametric solve returns a deferred trace node with no numeric
        work to time.
        """
        # Cleared on EVERY solve, not only a reduced one: a leftover value from an earlier
        # ``basis=`` call would read as "this answer was certified" on an answer that never was.
        self.basis_residual = None
        reduction = None if basis is None else self._basis_reduction(basis, adapt=adapt, move=move)

        def _run():
            prev_periodic = self._periodic
            if reduction is not None:
                self._periodic = reduction
            try:
                result = self._solve_dispatch(
                    solve_fn,
                    adapt=adapt,
                    move=move,
                    x0=x0,
                    nonlinear=nonlinear,
                    linear=linear,
                    precond=precond,
                    time=time,
                    **kwargs,
                )
            finally:
                self._periodic = prev_periodic  # the basis is per-CALL; never sticks to the object
            if reduction is not None:
                self._check_basis_residual(result, reduction)
            if isinstance(result, Placeholder):
                # Tag the solve node with its domain so jno.core can infer the domain straight from the graph
                # (a data-misfit inverse `jno.core([(fem.solve() - u_obs).mse])` needs no explicit `domain=`).
                result._domain = self.domain
            return result

        if not profile:  # profile=True: run the (eager) solve inside a JAX Perfetto trace + print a summary
            return _run()
        from .utils.profiling import profile_solve

        return profile_solve(_run, label=f"fem profile · {self.dofs} DOFs · {self._mode}", warm=(adapt is None))

    #: relative residual of the FULL system above which a ``basis=`` solve is refused. Not a tuning
    #: knob: at this size the basis does not span the solution at all (a modelling error), rather than
    #: merely resolving it coarsely — which is the legitimate use of a reduced basis.
    BASIS_RESIDUAL_LIMIT = 0.1

    def _basis_reduction(self, basis, *, adapt=None, move=None):
        """Validate ``basis=`` against this problem and wrap it as a reduction dict."""
        if adapt is not None or move is not None:
            which = "adapt=" if adapt is not None else "move="
            raise NotImplementedError(
                f"jno.fem: fem.solve(basis=..., {which}...) is not supported — {which} rebuilds or moves "
                "the mesh, so the DOF count and layout the basis was built against change underneath it. "
                "The basis would then be silently meaningless. Adapt first, then build a basis on the "
                "final mesh."
            )
        if self.dofs is None:
            raise NotImplementedError(
                f"jno.fem: fem.solve(basis=...) needs a known DOF count to validate the basis against, "
                f"and this {self._mode} problem does not report one."
            )
        if self._periodic is not None:
            raise NotImplementedError(
                "jno.fem: fem.solve(basis=...) together with a periodic tie is not supported yet — both "
                "reduce the system by a prolongation, and composing the two needs a decided convention "
                "for which space the basis is expressed in (full, or already periodic-reduced). Build the "
                "basis from snapshots of the untied problem, or drop the tie."
            )
        # Ask `is_complex`, NOT the mode. A steady complex form is FUSED into a real 2n system at
        # assembly, so its mode is an ordinary "linear" and a mode-based test silently lets it through:
        # `dofs` then reports 2n, an n-row basis fails a shape check deep inside the reduction, and a
        # 2n-row one runs in the internal [Re; Im] layout with the imaginary part cast away.
        if self.is_complex:
            raise NotImplementedError(
                "jno.fem: fem.solve(basis=...) is not wired for a complex problem — a complex form solves "
                "through the real-equivalent 2n block over [Re; Im], so the basis would have to be "
                "expressed in that internal layout rather than in the n complex DOFs you author against. "
                "Real steady problems (linear and nonlinear) are supported."
            )
        if self._mode == "transient":
            raise NotImplementedError(
                "jno.fem: fem.solve(basis=...) is not wired for a transient problem yet — the time block "
                "is reduced eagerly at assembly (PᵀMP, PᵀAP, restricted state0), so a basis arriving at "
                "solve time has to re-reduce it. Steady (linear and nonlinear) is supported."
            )
        return _galerkin_reduction(basis, self.dofs)

    def _check_basis_residual(self, u, reduction):
        """Measure how well the reduced answer satisfies the FULL system, and refuse a hopeless basis.

        A reduced solve is the ONE path here that returns an approximation rather than the answer, which
        cuts against the rest of the stack. ``‖A u − b‖/‖b‖`` at the lifted ``u`` costs a single
        full-size matvec — negligible against the full solve it replaces — and is a real certificate: it
        cannot prove the answer is good, but it does catch a basis that does not span the solution.

        Only the concrete, non-parametric linear path is measurable; a parametric or nonlinear solve is
        a deferred trace node with no values yet, so ``basis_residual`` stays ``None`` there and the
        span check does NOT run — a real hole in the guarantee, stated here rather than papered over.
        The measured value is kept on ``self.basis_residual`` (cleared at the top of every solve).
        """
        if self._mode != "linear" or isinstance(self._op, FemLinearSystem):
            return
        uc = _concrete(u)
        if uc is None:
            return  # traced (jax.grad/jit through the basis) — nothing to check yet
        A, b = self._op
        b = jnp.asarray(b).reshape(-1)
        r = _concrete(jnp.asarray(A @ jnp.asarray(uc, b.dtype)).reshape(-1) - b)
        nb = float(np.linalg.norm(np.asarray(b)))
        rel = float(np.linalg.norm(r)) / (nb if nb > 0 else 1.0)
        self.basis_residual = rel
        if not np.isfinite(rel) or rel > self.BASIS_RESIDUAL_LIMIT:
            k = int(np.asarray(reduction["P"]).shape[1])
            raise ValueError(
                f"jno.fem: the reduced solve does not satisfy the full system (relative residual "
                f"{rel:.3e} > {self.BASIS_RESIDUAL_LIMIT:g}) — this {k}-column basis does not span the "
                "solution, so the returned field would be plausible and wrong. Add modes, or include this "
                "parameter's regime in the snapshots the basis was built from. The measured value is on "
                "`fem.basis_residual`; deliberately coarse work (a rank sweep, a rough design pass) can "
                "raise the bar with `fem.BASIS_RESIDUAL_LIMIT = ...`."
            )

    def _solve_dispatch(
        self,
        solve_fn=None,
        *,
        adapt=None,
        move=None,
        x0=None,
        nonlinear=None,
        linear=None,
        precond=None,
        time=None,
        **kwargs,
    ):
        """Mode dispatch for :meth:`solve` — returns the solution array or a differentiable trace node."""
        has_slots = (
            (x0 is not None)
            or (nonlinear is not None)
            or (linear is not None)
            or (precond is not None)
            or (time is not None)
        )
        if move is not None:
            if adapt is not None or has_slots:
                raise NotImplementedError(
                    "fem.solve(move=...) does not compose with adapt= or the solver slots "
                    "(x0/nonlinear/linear/precond/time) yet — the moving-boundary driver owns the march and "
                    "re-assembles each move. Use move= on its own (default θ-stepper)."
                )
            if self._mode != "transient":
                raise NotImplementedError(
                    f"fem.solve(move=...) needs a transient problem (a u.t term); this FEM is '{self._mode}'. "
                    "A moving boundary evolves in time — add the time derivative and a domain time grid."
                )
            from .utils.solver.fem_adapt import run_moving_boundary

            return run_moving_boundary(self, move, solve_fn=solve_fn, **kwargs)
        if adapt is not None:
            if getattr(adapt, "relocate", False):
                # r-adaptivity: relocate the .trainable() vertices (fixed connectivity), not h-refinement.
                from .utils.solver.fem_adapt import run_adaptive_relocate

                return run_adaptive_relocate(self, adapt, solve_fn=solve_fn, **kwargs)
            # x0= (warm start) and time= do not compose with a remesh — the DOF layout changes across it,
            # staling a warm start and any cached step operator. The nonlinear=/linear=/precond= slots DO
            # compose: they configure the per-step (Newton / theta) solve, which is layout-independent —
            # but only on the transient adaptive driver, which owns the march.
            _adapt_step_slots = (nonlinear is not None) or (linear is not None) or (precond is not None)
            if (x0 is not None) or (time is not None) or (_adapt_step_slots and self._mode != "transient"):
                raise NotImplementedError(
                    "fem.solve: adapt= composes with the nonlinear=/linear=/precond= slots only on a "
                    "transient problem; x0= (warm start) and time= do not compose with remeshing (the DOF "
                    "layout changes across a remesh). Drop them, or pass solve_fn= for a custom loop."
                )
            from .utils.solver.fem_adapt import run_adaptive_solve, run_adaptive_transient

            if self._mode == "transient":
                if self.is_complex:
                    # A complex transient is now an ordinary transient block, so it *routes* here — but the
                    # adaptive driver's cross-remesh state transfer interpolates a real nodal field, and this
                    # block's state is the stacked ``[u_r; u_i]`` pair on one mesh. Transferring it needs the
                    # two halves carried separately. Fail loud rather than interpolate half a complex field.
                    raise NotImplementedError(
                        "jno.fem: fem.solve(adapt=...) on a complex *transient* problem is not supported yet — "
                        "the adaptive state transfer is not complex-aware (the block state is the stacked "
                        "[Re; Im] pair). A complex *steady* problem adapts, and a real transient adapts."
                    )
                # Adapt the mesh AS the problem marches: remesh every `adapt.every` steps and carry the
                # state across (basis-aware transfer), tracking a moving feature. The nonlinear=/linear=/
                # precond= slots configure the per-step (Newton / theta) solve. Returns an AdaptiveTrajectory.
                return run_adaptive_transient(
                    self, adapt, solve_fn=solve_fn, nonlinear=nonlinear, linear=linear, precond=precond, **kwargs
                )
            # A steady COMPLEX problem adapts via the ISOTROPIC ZZ estimator, which uses the modulus of the
            # (complex) recovered-gradient gap; only the anisotropic Hessian metric is real-only (guarded in
            # run_adaptive_solve).
            return run_adaptive_solve(self, adapt, solve_fn=solve_fn, **kwargs)
        if has_slots:
            solve_fn, kwargs = self._compose_slots(
                solve_fn, x0=x0, nonlinear=nonlinear, linear=linear, precond=precond, time=time, kwargs=kwargs
            )
            from_slots = True
        else:
            from_slots = False

        # ---- pseudo-time HISTORY MARCH (path-dependent state, e.g. plasticity): the op carries step-
        # history buffers (``.i(k)`` in the form) and the domain a ``tau=`` pseudo-time grid. March the
        # grid, threading τ as the load coordinate and the per-QP internal state on ``args["__history__"]``;
        # each step solves equilibrium then advances the states via their ``.evolves`` readout. Triggered
        # with NOTHING passed — exactly as ``u.t`` triggers the transient stepper. ----
        if getattr(self._op, "history_specs", None) or getattr(self._op, "surface_history_specs", None):
            if not getattr(self.domain, "_is_pseudo_time", False):
                raise ValueError(
                    "jno.fem: this form reads step history (`.i(k)`) but the domain has no pseudo-time load "
                    "path, so `fem.solve()` has no steps to march over. Build the domain with "
                    "`domain(tau=(start, end, n))` (the load written as a function of τ in the form). "
                    "A plain steady solve cannot carry step history."
                )
            if getattr(self._op, "state_readout", None) is None:
                raise NotImplementedError(
                    "jno.fem: the history march is only wired on the real, steady, single-field native "
                    "Lagrange path; this form assembled through another route."
                )
            from .utils.solver.history_march import run_history_march

            return run_history_march(self, solve_fn if from_slots else solve_fn)
        if from_slots and getattr(precond, "complex_native", False) and self._complex_legs is not None:
            # A complex-native preconditioner (AMS) solves the sparse COMPLEX operator ``A_r + i·A_i``
            # directly, never the real-equivalent block — so it wants the Re/Im legs the fusion retained,
            # not the fused 2n system. Real-equivalent preconditioners (form/jacobi/…) stay on the block.
            return _solve_complex_block(self._complex_legs, periodic=self._periodic, complex_solve=solve_fn)
        if self._mode == "complex":
            # Bloch only (a complex prolongation does not split into two real legs; everything else was
            # fused into a real 2n system at assembly by `_fuse_complex_steady`).
            if from_slots and getattr(precond, "complex_native", False):
                return _solve_complex_block(self._op, periodic=self._periodic, complex_solve=solve_fn)
            return _solve_complex_block(self._op, solve_fn, periodic=self._periodic, from_slots=from_slots)
        if self._mode == "linear" and not isinstance(self._op, FemLinearSystem):
            # Non-parametric steady linear. Default: matrix-free Jacobi-preconditioned BiCGStab on the
            # BCOO operator (never densifies -> memory O(nnz); GPU-safe; solves general systems). Pass
            # your own ``solve_fn=(A, b) -> u`` to use a dense / direct solver instead -- it receives
            # the densified (A, b). (The runtime-parametric case is a FemLinearSystem below.)
            A, b = self._op
            b = jnp.asarray(b).reshape(-1)
            # A fused complex system solves as the real 2n block, so its periodic reduction is
            # blkdiag(P, P) and the result is recombined u = x[:n] + i·x[n:] on the way out.
            _per = self._periodic_2n if self._complex_n is not None else self._periodic
            _fin = _complex_recombine(self._complex_n) if self._complex_n is not None else (lambda x: x)
            if _per is not None:
                # periodic tie: eliminate slave DOFs via the prolongation P, solve the reduced
                # (P^T A P) u_red = P^T b, then prolong u = P u_red back to the full nodal layout.
                # The reduction stays sparse (BCOO triplet-remap) -- it never materialises the dense
                # full operator, so it is GPU-able at large N. The *_periodic helpers reduce block-wise
                # per field, so this serves coupled problems too.
                from .utils.solver.fem_utils import prolong_periodic, reduce_matrix_periodic, reduce_vector_periodic

                A_red = reduce_matrix_periodic(_per, A)
                b_red = reduce_vector_periodic(_per, b)
                if solve_fn is not None:
                    u_red = solve_fn(A_red, b_red)  # user solver receives the (BCOO) reduced operator
                elif hasattr(A_red, "todense"):
                    # sparse-direct on the reduced BCOO (robust on the indefinite/saddle reduced systems
                    # a coupled periodic problem can produce; never densifies the full operator)
                    from .utils.solver.linear import sparse_lu_solve

                    u_red = sparse_lu_solve(A_red, b_red)
                else:
                    u_red = jnp.linalg.solve(jnp.asarray(A_red), b_red)  # dense reduced (1D / dense fallback)
                return _fin(prolong_periodic(_per, u_red))
            if (
                solve_fn is None
                and not from_slots
                and getattr(self.domain, "dimension", None) == 1
                and hasattr(A, "todense")
            ):
                # A 1D (LINE2) operator is TRIDIAGONAL: a sparse-direct solve is O(N) — as cheap as
                # iterating — and exact to round-off, where Jacobi-BiCGStab converges only to its
                # tolerance and its error GROWS with N (measured 8.9e-16 vs 2.7e-11 at n=101, and
                # 4.6e-14 vs 2.6e-10 at n=1001). Iterating a tridiagonal system is strictly worse.
                from .utils.solver.linear import sparse_lu_solve

                return _fin(sparse_lu_solve(A, b))
            if solve_fn is None and self._complex_n is not None and hasattr(A, "todense"):
                # SPARSE-DIRECT is the complex default, and deliberately so: the real-equivalent block
                # is **indefinite** for Helmholtz / PML, where the Jacobi-preconditioned BiCGStab that
                # serves real elliptic systems does not converge at all (measured: relative residual
                # 1.4 on the PML benchmark). Carry the choice over rather than inherit the real default.
                from .utils.solver.linear import sparse_lu_solve

                return _fin(sparse_lu_solve(A, b))
            if solve_fn is None and hasattr(A, "todense"):
                return _fin(_solve_linear_matrix_free(A, b))
            if from_slots:
                return _fin(solve_fn(A, b))  # slot-composed solvers take the BCOO operator directly
            A = jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)
            return _fin((solve_fn or (lambda A_, b_: jnp.linalg.solve(A_, b_)))(A, b))
        if self._mode == "linear" and isinstance(self._op, FemLinearSystem):
            # Runtime-parametric steady linear: solve A(θ)x=b(θ) as a trace node (∂u/∂θ flows through
            # solve_fn). A periodic tie reduces per-call inside FemLinearSystem.solve, after A(θ) is
            # re-formed: u = P · solve(PᵀA(θ)P, Pᵀb(θ)); self._periodic is None for the untied case.
            _node = self._op.solve(solve_fn, periodic=self._periodic_2n if self._complex_n else self._periodic)
            if self._complex_n is None:
                return _node
            # Fused complex inverse: the trace node solves the real 2n block, so recombine INSIDE it —
            # the caller must receive a complex field, and ∂u/∂θ still flows through the wrapped fn.
            from .trace import FunctionCall

            _rc = _complex_recombine(self._complex_n)
            return FunctionCall(lambda *v, _f=_node.fn: _rc(_f(*v)), _node.args, name="fem_complex_solve")
        if self._mode == "nonlinear" and self._periodic is not None:
            # Periodic nonlinear: solve Newton in the reduced space -- r_red(u_red) = P^T r(P u_red) = 0,
            # then prolong u = P u_red (the tie is then satisfied exactly). Wraps the user's solve_fn
            # (or the operator's default Newton) so it operates on the reduced residual.
            from .utils.solver.fem_utils import prolong_periodic, reduce_vector_periodic, restrict_state_periodic

            periodic = self._periodic
            user_fn = solve_fn

            def _reduced(residual_fn, y0):
                def _base(rf, y):
                    if user_fn is not None:
                        return user_fn(rf, y)
                    # Matrix-free Newton-Krylov default (no optimistix); implicit-diff preserved.
                    from .utils.solver.newton_krylov import newton_krylov

                    return newton_krylov(rf, y)

                ur = _base(
                    lambda ur: reduce_vector_periodic(
                        periodic, jnp.asarray(residual_fn(prolong_periodic(periodic, ur))).reshape(-1)
                    ),
                    restrict_state_periodic(periodic, y0),
                )
                return prolong_periodic(periodic, ur)

            return self._op.solve(solve_fn=_reduced, **kwargs)
        if self._mode == "nonlinear" and not getattr(self._op, "is_parametric", False):
            # Non-parametric steady nonlinear: return the numeric solution eagerly (mirrors the linear
            # branch above). `fem.solve()` builds a FunctionCall trace node so a trainable parameter can
            # flow to crux; with no parameter it is just a forward solve, so evaluate it to an array.
            return self._op.solve(solve_fn, **kwargs).fn()
        if self._mode == "transient" and self.is_complex and not self._op.runtime_parameter_exprs:
            # Non-parametric COMPLEX transient: return the concrete complex trajectory eagerly, exactly as
            # the steady linear / nonlinear branches above do — and as the complex transient did before its
            # Re/Im legs were fused into one block. (A *real* transient stays lazy; that asymmetry predates
            # the fusion and is a separate call to make, not something a refactor should change silently.)
            return self._op.solve(solve_fn, **kwargs).fn()
        return self._op.solve(solve_fn, **kwargs)

    def _compose_slots(self, solve_fn, *, x0, nonlinear, linear, precond, time=None, kwargs):
        """Compose the solver slots into the mode-appropriate ``solve_fn`` (see :meth:`solve`)."""
        from .utils.solver.solver_api import compose_linear_solve_fn, compose_nonlinear_solve_fn

        if solve_fn is not None:
            raise ValueError(
                "fem.solve: pass either solve_fn= (the total override) or the solver slots "
                "(x0/nonlinear/linear/precond/time), not both."
            )
        if time is not None and self._mode != "transient":
            raise ValueError(
                f"fem.solve(time=...) picks a time-integration scheme, but this problem is {self._mode}, not transient."
            )
        if self._mode == "transient":
            # thread the slots into the default theta-stepper as per-step solvers: the linear
            # slot/precond see the step operator (M + theta dt A) -- materialized ONCE before the
            # scan when the operator is time-independent -- and the nonlinear slot drives each
            # implicit step. The bring-your-own (block, args, save_ts) contract is unchanged.
            if x0 is not None:
                raise ValueError(
                    "fem.solve: x0= on a transient problem -- the initial state comes from the initial "
                    "conditions, and each step already warm-starts from the previous state."
                )
            from .utils.solver.backend_blocks import _default_transient_integrate
            from .utils.solver.solver_api import compose_transient_step_solvers

            lin_s, nonlin_s = compose_transient_step_solvers(nonlinear, linear, precond, self, self._op)

            def _stepper(block, args, save_ts):
                if time is not None:  # jno.solve.theta(...) / jno.solve.exponential(...)
                    return time.integrate(block, args, save_ts, linear_solve=lin_s, nonlinear_solve=nonlin_s)
                return _default_transient_integrate(block, args, save_ts, linear_solve=lin_s, nonlinear_solve=nonlin_s)

            return _stepper, kwargs
        if self._mode == "nonlinear":
            fn = compose_nonlinear_solve_fn(nonlinear, linear, precond, self)
            if x0 is not None:
                if "u0" in kwargs:
                    raise ValueError("fem.solve: x0= and u0= are the same initial guess; pass one.")
                kwargs = {**kwargs, "u0": jnp.asarray(x0).reshape(-1)}
            return fn, kwargs
        if nonlinear is not None:
            raise ValueError(f"fem.solve: nonlinear= given, but this problem is {self._mode} (no linearization).")
        if self._mode == "complex" and x0 is not None:
            # Bloch only — the complex prolongation keeps this one off the real-equivalent block.
            raise NotImplementedError(
                "fem.solve: x0= on a *Bloch* (quasi-periodic) complex problem is not supported yet — "
                "that path still solves through a complex prolongation rather than the real-equivalent "
                "block. An ordinary complex problem accepts x0=."
            )
        if self._complex_n is not None and x0 is not None:
            # A complex guess enters the real-equivalent layout the solve runs in: x0 = [Re; Im].
            _x0 = jnp.asarray(x0).reshape(-1)
            x0 = jnp.concatenate([jnp.real(_x0), jnp.imag(_x0)]) if jnp.iscomplexobj(_x0) else _x0
        _per_x0 = self._periodic_2n if self._complex_n is not None else self._periodic
        if _per_x0 is not None and x0 is not None:
            # the solve runs in the periodic-reduced space; restrict the guess to match
            from .utils.solver.fem_utils import restrict_state_periodic

            x0 = restrict_state_periodic(_per_x0, jnp.asarray(x0).reshape(-1))
        return compose_linear_solve_fn(linear, precond, x0, self), kwargs

    @property
    def points(self):
        """Node coordinates the DOFs live on.

        For a higher-order (P2) element these are the assembly-mesh nodes
        (vertices + edge midpoints), which differ from the linear ``mesh`` the
        domain keeps — use these to interpret the solution vector."""
        prob = self.problem
        meshes = getattr(prob, "mesh", None) if prob is not None else None
        if meshes:
            return jnp.asarray(meshes[0].points)
        # Native Lagrange path (no problem object): the assembler records the DOF coordinates
        # (vertices + edge midpoints for P2) the flat solution lives on. Prefer the snapshot
        # captured at finalize time — the domain attribute is overwritten by any later assembly
        # on the same domain (e.g. a jno.precond.form auxiliary operator).
        native_pts = getattr(self, "_native_dof_points", None)
        if native_pts is None:
            native_pts = getattr(self.domain, "_fem_native_dof_points", None)
        if native_pts is not None:
            return jnp.asarray(native_pts)
        if self.mesh is not None:
            return jnp.asarray(self.mesh.points)[:, : self.domain.dimension]
        return None

    @property
    def field_points(self):
        """Per-field DOF coordinates: ``field_points[i]`` is the ``(n_nodes_i, dim)`` node array the
        block ``sol[offsets[i]:offsets[i+1]]`` lives on. For a coupled problem the fields may use
        different nodes (e.g. Taylor-Hood P2 velocity vs P1 pressure). Backed by the per-field meshes
        of the problem object, or the native assembler's recorded per-field DOF points."""
        prob = self.problem
        meshes = getattr(prob, "mesh", None) if prob is not None else None
        if meshes:
            return [jnp.asarray(m.points) for m in meshes]
        # snapshot first (see .points): the live domain attribute is clobbered by later assemblies
        native_all = getattr(self, "_native_dof_points_all", None)
        if native_all is None:
            native_all = getattr(self.domain, "_fem_native_dof_points_all", None)
        if native_all is not None:
            return [jnp.asarray(p) for p in native_all]
        pts = self.points
        return [pts] if pts is not None else []

    # -- steady linear --
    @property
    def A(self):
        """Dense ``(n_dofs, n_dofs)`` stiffness matrix of the steady linear system, ready for
        ``jnp.linalg.solve`` (use ``fem.operator`` for the raw sparse form on large problems)."""
        if self._mode != "linear":
            raise AttributeError(f"FEM is {self._mode}; .A is only for a steady linear problem (see .operator / .M).")
        return _as_dense(self._A)

    @property
    def b(self):
        """Load vector of the steady linear system as a flat ``(n_dofs,)`` JAX array."""
        if self._mode != "linear":
            raise AttributeError(f"FEM is {self._mode}; .b is only for a steady linear problem (see .operator / .M).")
        return _as_flat(self._b)

    def eigs(self, *, mass, k: int = 6, which: str = "smallest", precond=None, tol=None, maxiter=None):
        """Generalized eigenproblem ``K x = λ M x`` on this fem: ``K`` is this **source-less** fem's
        operator (its stiffness bilinear form) and ``M`` is the ``mass`` bilinear form assembled on the
        same space. Returns ``(λ, X)`` — the ``k`` eigenvalues at ``which`` (``'smallest'``/``'largest'``)
        and their **M-orthonormal** eigenvectors. Eigenvalues are differentiable (see :func:`jno.solve.eigs`).

        Modal analysis, buckling, EM cavity/waveguide resonances, photonic band structure::

            u, v = d.fem_symbols(); ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
            K = jno.fem([ui.x * vi.x + ui.y * vi.y])      # stiffness (no source term)
            lam, X = K.eigs(mass=[ui * vi], k=6)          # K x = λ M x  → λ = ω² (Neumann box)

        ``precond=`` switches from the dense reduction to **preconditioned LOBPCG**, which never
        densifies the operator and so scales past it — ``tol``/``maxiter`` tune that iteration and are
        rejected without it. See :func:`jno.solve.eigs` for the full contract::

            lam, X = K.eigs(mass=[ui * vi], k=6, precond=jno.precond.amg())
        """
        if self._mode != "linear":
            raise AttributeError(f"FEM.eigs needs a steady-linear (source-less) bilinear form; this fem is {self._mode}.")
        _reject_source_terms(self, "FEM.eigs")
        from . import solve as _solve
        from .utils.solver.solver_api import LinearOperator

        K = self.operator[0]
        n_full = int(jnp.shape(K)[0] if getattr(K, "shape", None) is not None else LinearOperator(K).shape[0])
        # Read the constraint sets BEFORE assembling the mass form. `_fem_native_dirichlet_pairs` is
        # stashed on the shared DOMAIN, so assembling the (Dirichlet-free) mass form overwrites it with
        # an empty list — the elimination would then silently not happen and the row-replaced spurious
        # modes would come back.
        restrict, prolong, n_red, _kind = _eigs_constraint_maps(self, n_full)
        M = fem(list(mass)).operator[0]  # mass matrix on the same FE space

        solver = _solve.eigs(k=k, which=which, precond=precond, tol=tol, maxiter=maxiter)
        if restrict is None:
            return solver(K, M)

        # Reduced pencil (PᵀKP) x̂ = λ (PᵀMP) x̂ as MATVECS — never assembled, so this composes with the
        # matrix-free LOBPCG path and never densifies a big operator just to drop some rows.
        Kop, Mop = LinearOperator(K), LinearOperator(M)

        def red(op):
            mv = lambda v: restrict(op.mv(prolong(v)))  # noqa: E731
            # `dense_fn` only fires on the small-problem dense path, which materializes anyway; the
            # LOBPCG path never calls it and so never builds the reduced matrix.
            return LinearOperator.from_matvec(
                mv,
                shape=(n_red, n_red),
                dense_fn=lambda: jax.vmap(mv, in_axes=1, out_axes=1)(jnp.eye(n_red)),
            )

        lam, Xr = solver(red(Kop), red(Mop))
        return lam, jax.vmap(prolong, in_axes=1, out_axes=1)(Xr)  # modes back on the full mesh

    # -- domain-decomposition coupling (`jno.core([...])`): the region this subdomain owns + a pinned solve --
    def _dd_region(self):
        # The weak-form coordinates are retagged for quadrature, so read the region from `classification`
        # (`"volume@A"`), which `_region_and_support` already resolved correctly.
        src = getattr(self.domain, "_source_regions", {}) or {}
        regions = {
            cl.split("@", 1)[1]
            for cl in (self.classification or [])
            if isinstance(cl, str) and "@" in cl and cl.split("@", 1)[1] in src
        }
        return (next(iter(regions)), src[next(iter(regions))]) if len(regions) == 1 else (None, None)

    @property
    def region(self):
        """The named sub-region this FEM problem owns (from its weak-form coordinates), or ``None``."""
        return self._dd_region()[0]

    @property
    def region_geometry(self):
        """The shapely geometry of :attr:`region`, or ``None`` — used by ``jno.core([...])`` coupling."""
        return self._dd_region()[1]

    def _as_whole_mesh(self):
        """Rebuild this region-tagged FEM with WHOLE-MESH assembly (no ``RegionMask``) while keeping its
        region label. A region-local matrix can't reconcile an overlapping-Schwarz band (its artificial
        boundary reaches no neighbour cells), so the overlap driver swaps in this whole-mesh rebuild. One
        extra assemble + factorization, reused across every iteration — cheap next to the Schwarz loop."""
        if not getattr(self, "_constraints", None):
            return self
        return fem(self._constraints, _dd_overlap=True, **(getattr(self, "_fem_kwargs", None) or {}))

    def pinned_solver(self, node_ids, *, nonlinear=None):
        """A reusable ``f(values) -> field`` that solves the linear system with ``node_ids`` pinned to
        ``values`` (row-replacement) — the interface Dirichlet data a coupled Schwarz step supplies. The
        (fixed) matrix is prefactored once; each iteration only re-solves against a new right-hand side.
        The dense LU runs on the host (robust: the GPU ``cuSolver`` dense path can be flaky)."""
        import numpy as _np
        import scipy.linalg as _sla

        a = _np.asarray(self.A).copy()
        b = _np.asarray(self.b).reshape(-1)
        pin = _np.asarray(node_ids, dtype=int)
        a[pin, :] = 0.0
        a[pin, pin] = 1.0  # row-replacement: pinned rows → identity (columns kept)
        lu = _sla.lu_factor(a)  # factor once; the interface pin only changes the RHS across iterations

        def solve(values):
            rhs = b.copy()
            rhs[pin] = _np.asarray(values).reshape(-1)
            return _sla.lu_solve(lu, rhs)

        return solve

    # -- nonlinear residual / Jacobian (steady and transient) --
    @property
    def residual(self):
        """Residual callable for a custom solver, returning a flat ``(n_dofs,)`` JAX array.

        Steady nonlinear: ``residual(u)``. Transient: ``residual(u, t)`` — the per-step
        semidiscrete residual (pass ``args=`` only for a runtime-parametric solve). Use
        ``fem.operator`` for the raw (unflattened) form."""
        if self._mode == "nonlinear":
            r = self._op.residual
            return lambda u: _as_flat(r(u))
        if self._mode == "transient":
            return lambda u, t, args=None: _as_flat(self._op.residual(u, t, args or {}))
        raise AttributeError(f"FEM is {self._mode}; .residual is for a steady-nonlinear or transient problem.")

    @property
    def jacobian(self):
        """Jacobian callable for a custom solver, returning a dense ``(n_dofs, n_dofs)`` JAX
        array — ``jacobian(u)`` (steady nonlinear) or ``jacobian(u, t)`` (transient). Use
        ``fem.operator`` for the raw sparse (BCOO) form on large problems."""
        if self._mode == "nonlinear":
            j = self._op.jacobian
            return lambda u: _as_dense(j(u))
        if self._mode == "transient":
            return lambda u, t, args=None: _as_dense(self._op.jacobian(u, t, args or {}))
        raise AttributeError(f"FEM is {self._mode}; .jacobian is for a steady-nonlinear or transient problem.")

    # -- transient (semidiscrete: M u_dot + ... ; integration window from the domain) --
    @property
    def M(self):
        """Dense ``(n_dofs, n_dofs)`` mass matrix of the semidiscrete transient system
        (use ``fem.operator`` for the raw sparse form on large problems)."""
        if self._mode != "transient":
            raise AttributeError("FEM is steady; .M (mass matrix) is only for a transient problem.")
        # the nonlinear-transient route carries the mass as a callable (the ``.M`` attribute is
        # unset); evaluate it once at t0 -- the mass is constant in the standard ``u_t * v`` form.
        M = self._op.M
        if M is None and getattr(self._op, "mass_residual", None) is not None:
            raise AttributeError(
                "FEM has a STATE-DEPENDENT (nonlinear) mass ``c(u)·u_t``: there is no single mass matrix "
                "(the mass M(u) varies with the solution). Its per-step action is assembled inside the "
                "solve; use ``fem.solve()`` to march it, not ``fem.M``."
            )
        if M is None:
            M = self._op.mass(self.t0, {})
        return _as_dense(M)

    @property
    def state0(self):
        """Initial nodal state as a flat ``(n_dofs,)`` JAX array (from the `u(initial) - u0`
        residual, else zeros)."""
        if self._mode != "transient":
            raise AttributeError("FEM is steady; .state0 is only for a transient problem.")
        return _as_flat(self._op.state0)

    def _time_block(self):
        """The (real) SemidiscreteTimeBlock carrying the integration window (a complex transient is
        fused into one such block over the stacked ``[Re; Im]`` state, so this needs no special case)."""
        return self._op

    @property
    def t0(self):
        return self._time_block().t0 if self.is_transient else None

    @property
    def t1(self):
        return self._time_block().t1 if self.is_transient else None

    @property
    def dt(self):
        return self._time_block().dt if self.is_transient else None

    @property
    def dofs(self) -> Optional[int]:
        if self._mode == "linear" and self._b is not None:
            return int(jnp.asarray(self._b).reshape(-1).shape[0])
        if self._mode == "nonlinear":
            size = getattr(self._op, "size", None)
            if size is not None:
                return int(size)
        if self.is_transient:
            tb = self._time_block()
            if getattr(tb, "state0", None) is not None:
                return int(jnp.asarray(tb.state0).reshape(-1).shape[0])
            if getattr(tb, "M", None) is not None:
                return int(tb.M.shape[0])  # BCOO or dense — both expose .shape
        prob = self.problem
        return int(prob.num_total_dofs_all_vars) if prob is not None else None

    def __repr__(self) -> str:
        kind = (
            self._mode if self._mode != "transient" else ("transient-linear" if self.is_linear else "transient-nonlinear")
        )
        return f"FEM({kind}, dofs={self.dofs}, terms={self.classification})"


# ---------------------------------------------------------------------------
# periodic ties:  u(A) - u(B)  (same trial, two boundary regions)
# ---------------------------------------------------------------------------
def _periodic_tie_spec(constraint: Any, domain: Any) -> Optional[Tuple[str, str, Optional[int], Any]]:
    """Recognise a **periodic tie** ``u(A) - u(B)`` and return ``(master_tag, slave_tag, comp,
    field_key)``; ``None`` for any non-tie constraint.

    The two regions are carried on the constraint's ``_periodic_tie`` attribute, stamped by the trace
    layer when it builds ``u(A) - u(B)`` (the only point where each side's region survives — the
    ``BinaryOp`` discards the per-side bound views). ``self`` (the left operand) is the eliminated
    slave; ``other`` (right) is the retained master — the relation ``u(A)=u(B)`` is symmetric.
    """
    tie = getattr(constraint, "_periodic_tie", None)
    if tie is None:
        return None
    # The trace stamps `_periodic_tie` on *any* trial-trial combination with clashing coords. A valid
    # tie is `u(A) - u(B)` (plain periodic) or `u(A) - c*u(B)` with a constant scalar `c` (Bloch /
    # quasi-periodic, `c = e^{i k·L}`). Reject the rest loudly (e.g. `u(A)+u(B)` anti-periodic).
    bare = _bare(constraint)
    phase = _tie_phase(bare)
    if phase is None:
        raise ValueError(
            "jno.fem: a periodic tie must be `u(A) - u(B)` (periodic) or `u(A) - c*u(B)` with a constant "
            "scalar `c` (Bloch/quasi-periodic). Anti-periodic (`+`) or non-scalar relations are not supported."
        )
    slave_tag, master_tag = tie
    breg = getattr(domain, "_boundary_regions", {})
    if not (isinstance(slave_tag, str) and isinstance(master_tag, str)):
        return None
    if slave_tag == master_tag or slave_tag not in breg or master_tag not in breg:
        raise ValueError(
            f"jno.fem: a periodic tie `u(A) - u(B)` must connect two distinct boundary regions; "
            f"got {slave_tag!r} and {master_tag!r} (known boundary tags: {sorted(breg)})."
        )
    return (master_tag, slave_tag, None, _field_key_of(constraint), phase)


def _ref_interior_facet_dofs(ref_pts: np.ndarray, fv: np.ndarray, dim: int, tol: float = 1e-9) -> List[int]:
    """Local DOF ids on the facet spanned by reference vertices ``fv`` (beyond the facet's own vertices),
    **ordered** to match the periodic interpolation convention: the interior nodes of each facet edge in
    cyclic order (each edge's nodes sorted by position), then the face-interior nodes (3D). For P1/P2 this
    reproduces the legacy facet layout exactly -- 2D ``[.. , mid_ab]``, 3D ``[.. , mid_ab, mid_bc, mid_ca]``
    -- which ``_periodic_facet_weights`` relies on; higher orders extend it (per-edge nodes by position)."""
    ncorner = dim + 1  # element vertex DOFs are 0..dim (basix lists vertices first)
    cand = list(range(ncorner, len(ref_pts)))
    edges = [(fv[0], fv[1])] if dim == 2 else [(fv[0], fv[1]), (fv[1], fv[2]), (fv[2], fv[0])]
    ordered: List[int] = []
    used: set = set()
    for a, b in edges:
        ab = b - a
        l2 = float(np.dot(ab, ab))
        on_edge = []
        for d in cand:
            ap = ref_pts[d] - a
            t = float(np.dot(ap, ab) / l2)
            perp = ap - t * ab
            if float(np.dot(perp, perp)) < tol * tol * l2 and tol < t < 1.0 - tol:
                on_edge.append((t, d))
        for _t, d in sorted(on_edge):
            ordered.append(d)
            used.add(d)
    if dim == 3:  # face-interior nodes: on the facet plane, strictly inside the triangle, not on an edge
        a, b, c = fv
        M = np.stack([b - a, c - a], axis=1)  # (3, 2)
        for d in cand:
            if d in used:
                continue
            st, *_ = np.linalg.lstsq(M, ref_pts[d] - a, rcond=None)
            s, t = float(st[0]), float(st[1])
            if float(np.linalg.norm((a + M @ st) - ref_pts[d])) < tol and min(s, t, 1.0 - s - t) >= -tol:
                ordered.append(d)
    return ordered


def _boundary_facets(points: Any, cells: Any, dim: int, order: int) -> Optional[np.ndarray]:
    """Boundary facets of the **assembly** mesh as global node-id rows, including higher-order nodes.

    A facet is a cell edge (2D) / triangular face (3D) of the vertex sub-connectivity (the first
    ``dim+1`` columns -- basix orders vertices first) that appears in exactly **one** cell. For
    ``order >= 2`` each facet row also carries the P{order} nodes lying on that facet (edge-interior in
    2D; edge + face-interior in 3D), found from the reference element -- the local DOFs whose reference
    interpolation point lies on the corresponding reference facet. Vertices come first in each row (the
    leading ``dim+1`` columns stay the facet vertices); the remaining columns are the higher-order facet
    nodes in arbitrary order (consumers use the node set or match by coordinate). Returns
    ``(n_facets, n_facet_dof)``, or ``None`` for an empty mesh."""
    points = np.asarray(points, dtype=float)
    cells = np.asarray(cells, dtype=int)
    if cells.ndim != 2 or cells.shape[0] == 0:
        return None
    n_cells = cells.shape[0]
    verts = cells[:, : dim + 1]
    combos = [(0, 1), (1, 2), (2, 0)] if dim == 2 else [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    allf = np.concatenate([verts[:, list(c)] for c in combos], axis=0)  # combo-major: row = combo*n_cells + cell
    _uniq, idx, counts = np.unique(np.sort(allf, axis=1), axis=0, return_index=True, return_counts=True)
    bidx = idx[counts == 1]  # allf row index of each boundary facet (a facet used by exactly one cell)
    if order < 2:
        return allf[bidx]
    from .utils.solver.fem_lagrange import lagrange_interp_points

    ref_pts = np.asarray(lagrange_interp_points(dim, order))
    ref_verts = (
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        if dim == 2
        else np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    facet_dofs = [list(c) + _ref_interior_facet_dofs(ref_pts, ref_verts[list(c)], dim) for c in combos]
    rows = [cells[int(r) % n_cells, facet_dofs[int(r) // n_cells]] for r in bidx]
    return np.asarray(rows, dtype=int)


def _face_nodes(domain: Any, points: Any, bnodes: Optional[np.ndarray], tag: str) -> Optional[np.ndarray]:
    """Global node ids on a periodic face, taken from the tag's **predicate** (the user's intent),
    evaluated over the **assembly** boundary nodes ``bnodes`` (so P2 midpoints and 3D face nodes are
    included). ``domain.tag`` partitions each boundary node into a single tag, so a corner shared by
    two faces lands in only one -- re-evaluating the predicate recovers it in every face it satisfies
    (multi-direction needs that), while a corner-excluding predicate still omits the Dirichlet
    corners. Tags without a stored predicate fall back to the (exclusive, P1-mesh) ``tag_indices``."""
    ti = getattr(domain, "tag_indices", {})
    pred = getattr(domain, "_tag_predicates", {}).get(tag)
    if pred is not None and bnodes is not None and len(bnodes):
        coords = np.asarray(points)[bnodes]
        mask = np.asarray(pred(*(coords[:, i] for i in range(coords.shape[1]))), dtype=bool).reshape(-1)
        if mask.any():
            return np.asarray(bnodes)[mask]
    return np.asarray(ti[tag], dtype=int).reshape(-1) if tag in ti else None


def _chain_facets(points: Any, ids: Any) -> Optional[np.ndarray]:
    """Flat-face P1 boundary edges (global node-id pairs) as a fallback when assembly cells are
    unavailable (e.g. the native 1D route): sort the face nodes along their tangent and pair
    consecutive ones. ``None`` for < 2 nodes (a 1D face is a single node -> node-to-node tie)."""
    ids = np.asarray(ids, dtype=int).reshape(-1)
    if ids.size < 2:
        return None
    pts = np.asarray(points)[ids]
    tdim = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
    chain = ids[np.argsort(pts[:, tdim])]
    return np.column_stack([chain[:-1], chain[1:]])


def _assembly_cells(prob: Any) -> Tuple[Optional[np.ndarray], int]:
    """``(cells, element_order)`` of a problem object's assembly mesh, or ``(None, 1)`` (e.g. native 1D)."""
    meshes = getattr(prob, "mesh", None) if prob is not None else None
    if not meshes:
        return None, 1
    am = meshes[0]
    order = 2 if str(getattr(am, "ele_type", "")).upper() in _P2_ELEMENTS else 1
    return np.asarray(am.cells, dtype=int), order


def _build_periodic_reduction(domain: Any, ties: List[Any], points: Any, cells: Any, ele_order: int, vec: int) -> dict:
    """Build the prolongation ``P`` for the collected ties on the **assembly** mesh (``points`` +
    ``cells``; ``cells=None`` for the native 1D route falls back to flat-chain facets)."""
    from .utils.solver.fem_utils import build_periodic_prolongation

    points = np.asarray(points)

    # Multidirectional periodicity needs each face to carry its shared corners (a corner is a slave in
    # several directions). A ``domain.tag`` predicate face includes corners by construction. An auto face
    # (from Shape/emit) does too now that a face chain keeps both endpoints (``_chain_edges_to_loop``) --
    # accept it when the mesh confirms corners are shared: every pair of perpendicular periodic faces
    # must share a node. Otherwise (older partitioned tagging) reject rather than silently mis-solve.
    if len(ties) > 1:
        preds = getattr(domain, "_tag_predicates", {}) or {}
        if not all(t in preds for (m, s, *_i) in ties for t in (m, s)):
            ti = getattr(domain, "tag_indices", {}) or {}
            face_nodes = {}
            for m, s, *_ignore in ties:
                for t in (m, s):
                    if t in ti and t not in face_nodes:
                        face_nodes[t] = set(int(i) for i in np.asarray(ti[t]).reshape(-1))
            faces = [t for tie in ties for t in tie[:2]]
            shared = all(any((face_nodes.get(a, set()) & face_nodes.get(b, set())) for b in faces if b != a) for a in faces)
            if not (shared and len(face_nodes) == len(set(faces))):
                raise NotImplementedError(
                    "jno.fem: multidirectional periodicity needs each periodic face to include its shared "
                    "corners. Define the faces via `domain.tag(name, predicate)` so a corner lies on every "
                    "face it touches (an axis-aligned predicate like `lambda x, y: x < tol` includes corners)."
                )
    dim = int(getattr(domain, "dimension", points.shape[1]) or points.shape[1])
    bfacets = _boundary_facets(points, cells, dim, ele_order) if cells is not None else None
    bnodes = np.unique(bfacets) if bfacets is not None and bfacets.size else None

    pairs = [(master, slave) for (master, slave, *_ignore) in ties]
    # Bloch phase per pair (``e^{i k·L}``); 1.0 = plain periodic (the ties may be 4-tuples pre-Bloch).
    phases = [complex(t[4]) if len(t) > 4 else 1.0 + 0.0j for t in ties]
    faces: dict = {}
    for master, slave, *_ignore in ties:
        for tag in (master, slave):
            if tag not in faces and (f := _face_nodes(domain, points, bnodes, tag)) is not None:
                faces[tag] = f

    facets: dict = {}
    if bfacets is not None and bfacets.size:
        for master, _slave, *_ignore in ties:
            fn = set(np.asarray(faces.get(master, np.empty(0, int))).tolist())
            keep = np.array([set(row.tolist()).issubset(fn) for row in bfacets], dtype=bool)
            if keep.any():
                facets[master] = bfacets[keep]
    else:  # native 1D / no assembly cells -> flat-chain fallback
        facets = {m: ff for (m, _s, *_ignore) in ties if (ff := _chain_facets(points, faces.get(m, ()))) is not None}

    return build_periodic_prolongation(points, pairs, faces, vec=vec, facets=facets, phases=phases)


def _build_periodic_reduction_n1e(domain: Any, ties: List[Any], offsets: Any) -> dict:
    """Periodic (Floquet/Bloch) reduction for a **Nédélec N1E** edge field — a DOF-level edge prolongation.
    Each DOF is one edge's tangential moment, so a tie matches boundary edges across the periodic faces (by
    midpoint) with an orientation sign and the Bloch phase. Uses the edge topology the non-nodal assembler
    stashed. Every field is the same N1E element, so one per-field ``P`` is built and block-concatenated."""
    from .utils.solver.fem_utils import build_periodic_prolongation_n1e

    topo = domain._fem_nonnodal_topology
    n_edges = int(topo["n_edges"])
    vpts = np.asarray(topo["vertex_points"])
    ev = np.asarray(topo["edge_vertices"])
    emid = np.asarray(topo["edge_midpoints"])
    edir = np.asarray(topo["edge_dirs"])

    pairs = [(master, slave) for (master, slave, *_rest) in ties]
    phases = [(t[4] if len(t) > 4 and t[4] is not None else 1.0) for t in ties]  # Bloch phase (e^{iφ}) per tie
    etags: dict = {}
    for tag in {t for tie in ties for t in tie[:2]}:
        f = _face_nodes(domain, vpts, None, tag)
        if f is None or np.asarray(f).size == 0:
            raise ValueError(f"jno.fem periodic (N1E): boundary tag {tag!r} has no mesh vertices.")
        vset = set(int(v) for v in np.asarray(f, dtype=int).reshape(-1))
        # a global edge is on the boundary tag iff BOTH its canonical vertices are on that tag
        etags[tag] = np.asarray([e for e in range(n_edges) if int(ev[e, 0]) in vset and int(ev[e, 1]) in vset], dtype=int)

    red = build_periodic_prolongation_n1e(n_edges, emid, edir, etags, pairs, phases)
    n_fields = len(offsets) - 1
    blocks, off_full, off_red = [], [0], [0]
    for _ in range(n_fields):
        blocks.append({"P": red["P"], "kept": red["kept_nodes"], "vec": 1, "is_selection": red["is_selection"]})
        off_full.append(off_full[-1] + red["n_full"])
        off_red.append(off_red[-1] + red["n_red"])
    return {
        "blocks": blocks,
        "off_full": off_full,
        "off_red": off_red,
        "n_full": off_full[-1],
        "n_red": off_red[-1],
        "is_bloch": red["is_bloch"],
    }


def _build_periodic_reduction_nonnodal(domain: Any, ties: List[Any], offsets: Any) -> dict:
    """Periodic reduction for **non-nodal C¹** fields (Morley) — a DOF-level prolongation the node-based
    builder can't produce. Uses the C¹ topology the non-nodal assembler stashed on the domain: value
    DOFs tie by vertex, edge normal-derivative DOFs tie by boundary edge with the geometric sign.
    Every field is the same Morley element (mixing is rejected at assembly), so one per-field ``P`` is
    built and block-concatenated. Raises loudly for families not yet supported."""
    from .utils.solver.fem_utils import build_periodic_prolongation_nonnodal

    topo = getattr(domain, "_fem_nonnodal_topology", None)
    if topo is None:
        raise NotImplementedError(
            "jno.fem: periodic ties on this non-nodal element are not supported — the assembler stashed "
            "no C¹ edge topology (only Morley/Argyris carry the edge-normal DOFs the periodic tie needs)."
        )
    if topo["family"] != "Morley":
        raise NotImplementedError(
            f"jno.fem: periodic ties are implemented for the **Morley** C¹ element; periodic "
            f"{topo['family']!r} (with extra per-vertex derivative DOFs whose periodic signs are not yet "
            "wired) is not supported. Use space='Morley', or open an issue."
        )
    n_verts, n_edges = int(topo["n_verts"]), int(topo["n_edges"])
    vpts = np.asarray(topo["vertex_points"])
    ev = np.asarray(topo["edge_vertices"])

    # A tie spec is ``(master, slave, comp, field_key, phase)``. Unpack it TOLERANTLY, as the nodal and
    # N1E builders do: this one hard-unpacked exactly four and so broke outright the moment the Bloch
    # ``phase`` was appended — every periodic Morley problem raised "too many values to unpack".
    phases = [(t[4] if len(t) > 4 and t[4] is not None else 1.0) for t in ties]
    if any(abs(complex(p) - 1.0) > 1e-12 for p in phases):
        raise NotImplementedError(
            "jno.fem: a Bloch/quasi-periodic tie `u(A) - c*u(B)` is not supported on the Morley C¹ element "
            "— the C¹ prolongation carries real edge-normal orientation signs, and a complex phase would "
            "need a complex prolongation (P^H A P mixes Re and Im). Use a plain periodic tie `u(A) - u(B)`, "
            "or a nodal Lagrange / N1E space, both of which do carry the phase."
        )

    pairs = [(master, slave) for (master, slave, *_rest) in ties]
    vtags: dict = {}
    etags: dict = {}
    for tag in {t for (m, s, *_rest) in ties for t in (m, s)}:
        f = _face_nodes(domain, vpts, None, tag)
        if f is None or np.asarray(f).size == 0:
            raise ValueError(f"jno.fem periodic (non-nodal C¹): boundary tag {tag!r} has no mesh vertices.")
        vids = np.asarray(f, dtype=int).reshape(-1)
        vtags[tag] = vids
        vset = set(int(v) for v in vids)
        # a global edge is on the boundary tag iff BOTH its canonical vertices are on that tag
        etags[tag] = np.asarray([e for e in range(n_edges) if int(ev[e, 0]) in vset and int(ev[e, 1]) in vset], dtype=int)

    red = build_periodic_prolongation_nonnodal(
        n_verts,
        n_edges,
        vpts,
        np.asarray(topo["edge_midpoints"]),
        np.asarray(topo["edge_normals"]),
        vtags,
        etags,
        pairs,
    )
    n_fields = len(offsets) - 1
    blocks, off_full, off_red = [], [0], [0]
    for _ in range(n_fields):
        blocks.append({"P": red["P"], "kept": red["kept_nodes"], "vec": 1, "is_selection": red["is_selection"]})
        off_full.append(off_full[-1] + red["n_full"])
        off_red.append(off_red[-1] + red["n_red"])
    return {"blocks": blocks, "off_full": off_full, "off_red": off_red, "n_full": off_full[-1], "n_red": off_red[-1]}


def _build_periodic_reduction_multifield(
    domain: Any, ties: List[Any], points: Any, cells: Any, ele_order: int, offsets: Any
) -> dict:
    """Periodic reduction for a coupled (multi-field) problem: one block per field. Each field's
    ``P_i`` is built from its own DOF nodes / element order / vec and its own ties (matched by
    ``field_key``), so heterogeneous-order couplings (e.g. Taylor-Hood: P2 velocity + P1 pressure)
    are supported. The Galerkin reduction stays block-wise (``P_i^T M[i,j] P_j``) via the
    ``fem_utils`` helpers — no block-diagonal ``P`` is materialised — and when all fields share a
    mesh + order a single node-``P`` is built once and shared (the common case)."""
    offs = [int(o) for o in offsets]
    n_fields = len(offs) - 1
    sizes = [offs[i + 1] - offs[i] for i in range(n_fields)]

    pts_all = getattr(domain, "_fem_native_dof_points_all", None) or [points] * n_fields
    cells_all = getattr(domain, "_fem_native_assembly_cells_all", None) or [cells] * n_fields
    orders = getattr(domain, "_fem_native_field_orders", None) or [ele_order] * n_fields
    field_keys = getattr(domain, "_fem_native_field_keys", None)

    def _tie_field_index(t):
        """The offset-order field index a tie belongs to (its trial ``field_key``); ``None`` if the
        key is unknown (then the tie is applied to every field — single-field-direction fallback)."""
        fk = t[3]
        return field_keys.index(fk) if (field_keys is not None and fk in field_keys) else None

    # Fast path: all fields share the same DOF-block size + element order -> one node-P, shared.
    if len(set(sizes)) == 1 and len(set(orders)) == 1:
        n_nodes = int(np.asarray(pts_all[0]).shape[0])
        vec = max(1, sizes[0] // n_nodes)
        seen: set = set()
        uniq = [t for t in ties if (t[0], t[1]) not in seen and not seen.add((t[0], t[1]))]
        red = _build_periodic_reduction(domain, uniq, pts_all[0], cells_all[0], int(orders[0]), int(vec))
        P, kept, v = red["P"], red["kept_nodes"], red["vec"]
        nrf = int(P.shape[1])
        blocks = [{"P": P, "kept": kept, "vec": v, "is_selection": red["is_selection"]} for _ in range(n_fields)]
        off_full = [i * sizes[0] for i in range(n_fields + 1)]
        off_red = [i * nrf for i in range(n_fields + 1)]
        return {"blocks": blocks, "off_full": off_full, "off_red": off_red, "n_full": off_full[-1], "n_red": off_red[-1]}

    # Heterogeneous: a distinct P_i per field, from its own nodes/order/vec and its own ties.
    blocks, off_full, off_red = [], [0], [0]
    for i in range(n_fields):
        pts_i = np.asarray(pts_all[i])
        vec_i = max(1, sizes[i] // int(pts_i.shape[0]))
        ties_i = [t for t in ties if _tie_field_index(t) in (i, None)]
        if ties_i:
            red_i = _build_periodic_reduction(domain, ties_i, pts_i, cells_all[i], int(orders[i]), int(vec_i))
            P_i, kept_i, v_i, sel_i = red_i["P"], red_i["kept_nodes"], red_i["vec"], red_i["is_selection"]
        else:  # a field with no periodic tie -> sparse identity (a selection: one master per row)
            import jax.experimental.sparse as jsparse

            ni = int(sizes[i])
            di = jnp.arange(ni)
            P_i = jsparse.BCOO((jnp.ones(ni), jnp.stack([di, di], axis=1)), shape=(ni, ni))
            kept_i, v_i, sel_i = np.arange(int(pts_i.shape[0])), vec_i, True
        blocks.append({"P": P_i, "kept": kept_i, "vec": v_i, "is_selection": sel_i})
        off_full.append(off_full[-1] + int(P_i.shape[0]))
        off_red.append(off_red[-1] + int(P_i.shape[1]))
    return {"blocks": blocks, "off_full": off_full, "off_red": off_red, "n_full": off_full[-1], "n_red": off_red[-1]}


def _complex_recombine(n: int):
    """``x -> x[..., :n] + i·x[..., n:]`` — the real-equivalent 2n solution read back as complex.

    Applied last, after any periodic prolongation: ``P`` is real and linear, so prolong-then-split is
    identical to split-then-prolong, and doing it here keeps every solver, slot and preconditioner on
    the real block, complex-unaware."""

    def recombine(x):
        x = jnp.asarray(x)
        return x[..., :n] + 1j * x[..., n:]

    return recombine


def _fuse_complex_steady(fem_obj: "FEM") -> "FEM":
    """Fuse a steady complex ``(re, im)`` leg pair into ONE real ``2n`` system, in place.

    The same move :func:`_assemble_second_order_time` makes for ``u_tt`` and the complex-transient
    assembly makes for its time block: build ``[[A_r, -A_i], [A_i, A_r]] x = [b_r; b_i]`` over
    ``x = [x_r; x_i]`` at assembly, so the result is an ordinary ``"linear"`` system. Everything that
    already works for a real linear FEM — the ``linear=``/``precond=`` slots, ``x0=``, the periodic
    reduction, the matrix-free default — then applies to a complex problem with no complex-specific
    branch, and the recombination ``u = x[:n] + i·x[n:]`` happens once on the way out.

    The Re/Im legs are retained on ``_complex_legs`` because they are still the right representation
    for one consumer: a **complex-native** preconditioner (AMS) solves ``A_r + i·A_i`` directly rather
    than the real-equivalent block, and needs the legs to build it.

    **Bloch is left alone.** A quasi-periodic tie has a *complex* ``P``, so ``P^H A P`` mixes Re and Im
    and does not reduce to two independent real legs — that needs ``P`` itself in real-equivalent form.
    Until then a Bloch problem keeps the ``"complex"`` mode and its dedicated block solve.
    """
    if fem_obj._mode != "complex":
        return fem_obj
    per = fem_obj._periodic
    if per is not None and per.get("is_bloch"):
        return fem_obj  # complex P — not expressible as two real legs; see the docstring
    from .trace import FemLinearSystem as _FLS

    op_r, op_i = fem_obj._op

    def _parts(leg):
        """``(A(args), b(args))`` for one leg — the static matrices, or re-formed at runtime args."""
        if isinstance(leg, _FLS):
            return (
                lambda a, _L=leg: _to_bcoo(_L.evaluate(a)[0]),
                lambda a, _L=leg: jnp.asarray(_L.evaluate(a)[1]).reshape(-1),
            )
        _A, _b = _to_bcoo(leg[0]), jnp.asarray(leg[1]).reshape(-1)
        return (lambda a, _A=_A: _A), (lambda a, _b=_b: _b)

    A_r_fn, b_r_fn = _parts(op_r)
    A_i_fn, b_i_fn = _parts(op_i)
    n = int(jnp.shape(b_r_fn(None))[0])
    rpe: dict = {}
    for leg in (op_r, op_i):
        if isinstance(leg, _FLS):
            rpe.update(getattr(leg, "runtime_parameter_exprs", None) or {})

    blk = lambda a: _complex_block_bcoo(A_r_fn(a), A_i_fn(a), n)  # noqa: E731
    rhs = lambda a: jnp.concatenate([b_r_fn(a), b_i_fn(a)])  # noqa: E731
    if rpe:  # parametric (the complex inverse): the 2n operator/load re-form from the runtime args
        fused = _FLS(
            blk(None),
            rhs(None),
            operator_fn=blk,
            rhs_fn=rhs,
            runtime_parameter_exprs=rpe,
            metadata={"complex": True},
        )
    else:
        fused = (blk(None), rhs(None))

    fem_obj._op = fused
    fem_obj._mode = "linear"
    fem_obj._complex_n = n
    fem_obj._complex_legs = (op_r, op_i)
    fem_obj._A, fem_obj._b = (fused.A, fused.b) if isinstance(fused, _FLS) else fused
    # the solve runs on the 2n block, so its periodic reduction is blkdiag(P, P)
    fem_obj._periodic_2n = None if per is None else _duplicate_periodic(per)
    return fem_obj


def _duplicate_periodic(periodic: dict) -> dict:
    """``blkdiag(P, P)`` — the periodic reduction for a state that is TWO stacked copies of the field.

    Three assemblies produce that shape, and the reduction is identical for all of them:

    * second-order in time — ``y = [u; v]`` (displacement, velocity)
    * complex transient    — ``y = [u_r; u_i]``
    * complex steady       — ``x = [x_r; x_i]``, the real-equivalent block system

    Duplicating the field's periodic blocks into a lower and an upper block is exactly ``blkdiag(P, P)``,
    which is what preserves the real-equivalent structure: ``blkdiag(P,P)ᵀ [[A_r,-A_i],[A_i,A_r]]
    blkdiag(P,P)`` is again of that form, with each sub-block reduced by the same ``P``."""
    from .utils.solver.fem_utils import _periodic_blocks

    b, of, orr = _periodic_blocks(periodic)
    of, orr = np.asarray(of), np.asarray(orr)
    nf, nr = int(of[-1]), int(orr[-1])
    return {
        "blocks": list(b) + list(b),
        "off_full": np.concatenate([of[:-1], of + nf]),
        "off_red": np.concatenate([orr[:-1], orr + nr]),
    }


def _reduce_transient_block_periodic(block: Any, periodic: dict) -> Any:
    """Reduce a native transient ``SemidiscreteTimeBlock`` by the periodic prolongation ``P`` and return a
    reduced block that carries ``P`` for prolongation (``u_full = P u_red``).

    Galerkin reduction is applied to every populated payload: ``M -> P^T M P``, a constant
    ``A -> P^T A P``, a runtime ``operator_fn(t, args) -> P^T A(t, args) P`` (so the reduction
    composes with re-assembly and stays differentiable in ``args``), the loads ``affine_bias`` and
    ``forcing_vector_fn(t, args) -> P^T(...)``, and the initial state is restricted to the master
    DOFs. A NONLINEAR block is reduced in the same spirit: ``mass`` / ``residual`` / ``jacobian`` are
    wrapped to act on the reduced state (prolong the input, reduce the output), so the integrator's
    matrix-free Newton then solves ``M_red(u_red⁺ - u_red)/dt + r_red(u_red⁺, t) = 0`` in the reduced
    space -- the reduced Jacobian ``P^T J P`` comes from autodiff through ``residual_red`` (mirrors the
    steady nonlinear periodic path)."""
    import dataclasses

    from .utils.solver.fem_utils import (
        _periodic_blocks,
        prolong_periodic,
        reduce_matrix_periodic,
        reduce_vector_periodic,
        restrict_state_periodic,
    )

    # A block whose state is TWO stacked copies of the field reduces by P on each half; see
    # :func:`_duplicate_periodic`. Both the second-order (``y=[u; v]``) and complex (``y=[u_r; u_i]``)
    # assemblies produce that shape, and every reduction below (M, A, operator_fn, forcing, state0)
    # then acts on the 2N system.
    _meta_in = getattr(block, "metadata", None) or {}
    if _meta_in.get("second_order") or _meta_in.get("complex"):
        periodic = _duplicate_periodic(periodic)

    blocks, off_f, off_r = _periodic_blocks(periodic)
    n_full, n_red = int(off_f[-1]), int(off_r[-1])
    meta = dict(getattr(block, "metadata", None) or {})
    meta.update(periodic=True, full_state_size=n_full, reduced_state_size=n_red)
    prol = blocks[0]["P"] if len(blocks) == 1 else periodic

    if block.is_nonlinear():
        _mass, _res, _jac = block.mass, block.residual, block.jacobian

        def mass_red(t, args=None, _p=periodic, _m=_mass):
            return reduce_matrix_periodic(_p, _m(t, args))

        def residual_red(u_red, t, args=None, _p=periodic, _r=_res):
            return reduce_vector_periodic(_p, jnp.asarray(_r(prolong_periodic(_p, u_red), t, args)).reshape(-1))

        jac_red = None
        if _jac is not None:

            def jac_red(u_red, t, args=None, _p=periodic, _j=_jac):
                return reduce_matrix_periodic(_p, _j(prolong_periodic(_p, u_red), t, args))

        return dataclasses.replace(
            block,
            mass=mass_red,
            residual=residual_red,
            jacobian=jac_red,
            state0=restrict_state_periodic(periodic, jnp.asarray(block.state0).reshape(-1)),
            prolongation=prol,
            metadata=meta,
        )

    op_red = None
    if block.operator_fn is not None:

        def op_red(t, args=None, _p=periodic, _op=block.operator_fn):
            return reduce_matrix_periodic(_p, _op(t, args))

    f_red = None
    if block.forcing_vector_fn is not None:

        def f_red(t, args=None, _p=periodic, _f=block.forcing_vector_fn):
            return reduce_vector_periodic(_p, jnp.asarray(_f(t, args)).reshape(-1))

    return dataclasses.replace(
        block,
        M=reduce_matrix_periodic(periodic, block.M),
        A=reduce_matrix_periodic(periodic, block.A) if block.A is not None else None,
        operator_fn=op_red,
        affine_bias=reduce_vector_periodic(periodic, jnp.asarray(block.affine_bias).reshape(-1))
        if block.affine_bias is not None
        else None,
        forcing_vector_fn=f_red,
        state0=restrict_state_periodic(periodic, jnp.asarray(block.state0).reshape(-1)),
        prolongation=prol,
        metadata=meta,
    )


def _concrete(x):
    """``np.asarray(x)`` if ``x`` carries values, else ``None`` (it is a JAX tracer).

    Used to run the *checkable* basis validation only when there is something to check — under
    ``jax.grad``/``jit`` w.r.t. the basis there are no values, and forcing them would raise. The
    two tracer errors are caught by NAME rather than with a bare ``except``: anything else going
    wrong here (a ragged array, a bad dtype) is a real problem and must not be swallowed into a
    silent "skip the validation"."""
    try:
        return np.asarray(x)
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        return None


def _galerkin_reduction(basis: Any, n_dofs: int, *, ortho_tol: float = 1e-8) -> dict:
    """Wrap a user-supplied Galerkin basis ``U`` as a reduction dict, in the SAME shape the periodic
    tie machinery already consumes -- ``{"P", "kept_nodes", "vec", "is_selection"}``.

    That is the whole trick behind ``fem.solve(basis=U)``: a periodic prolongation and a reduced-order
    basis are the same object (a tall ``n_full x k`` map defining ``P^T A P`` / ``P^T b`` / ``u = P x``),
    so the reduction, the per-mode combinator and the multifield block handling are reused verbatim.

    Two things differ from a periodic ``P`` and are recorded here:

    * ``kept_nodes=None`` -- the columns are not a subset of the full DOFs, so a state is restricted by
      PROJECTION (``P^T u``) rather than by gathering. See :func:`restrict_state`.
    * ``is_selection=False`` -- passed explicitly rather than sniffed, both because a dense basis never
      is one and because sniffing inspects values, which fails when the basis is traced.
    """
    if isinstance(basis, (Placeholder, ModelCall)):
        raise NotImplementedError(
            "jno.fem: fem.solve(basis=...) takes a concrete array. A basis built from jno.np.parameter / "
            "jno.nn.wrap is a trace node, and threading it as a runtime parameter is not wired yet. "
            "A basis differentiated through jax.grad/jit DOES work -- pass the array itself."
        )
    U = jnp.asarray(basis)  # normalise numpy / list / jnp up front; a tracer passes through unchanged
    if U.ndim != 2:
        raise ValueError(f"jno.fem: fem.solve(basis=U) needs a 2-D (n_dofs, k) basis; got shape {tuple(U.shape)}.")
    n, k = int(U.shape[0]), int(U.shape[1])
    if n != int(n_dofs):
        raise ValueError(
            f"jno.fem: the basis has {n} rows but this problem has {n_dofs} DOFs. Its COLUMNS are the modes "
            f"(shape (n_dofs, k)) -- a snapshot matrix is usually (n_snapshots, n_dofs), so it needs "
            "transposing. Note jno.solve.svd(snapshots, k) returns the spatial modes as `Vt.T` for a "
            "(n_snapshots, n_dofs) input, and as `U` for the transpose."
        )
    if not 1 <= k <= n:
        raise ValueError(f"jno.fem: the basis must have 1 <= k <= n_dofs columns; got k={k} against {n} DOFs.")
    if not jnp.issubdtype(U.dtype, jnp.floating):
        # An INTEGER basis passes the orthonormality check (an identity slice is exactly orthonormal)
        # and then silently truncates the reduced solve to integers. A COMPLEX basis is worse: the
        # reduction here is ``UᵀAU``, not the Hermitian ``UᴴAU``, so it is the wrong projection AND it
        # returns a complex field for a real problem. Neither may pass quietly.
        raise ValueError(
            f"jno.fem: fem.solve(basis=U) needs a real floating-point basis; got dtype {U.dtype}. "
            "A complex basis would need the Hermitian reduction UᴴAU (not wired), and an integer one "
            "silently truncates the reduced solve."
        )

    Uc = _concrete(U)
    if Uc is not None:  # these are checkable only when the basis carries values (not under trace)
        if not np.all(np.isfinite(Uc)):
            raise ValueError(
                "jno.fem: fem.solve(basis=U) got a basis containing NaN or Inf. A non-finite column "
                "poisons the whole reduced system, so it is refused here rather than surfacing later as "
                "a NaN solution."
            )
        gram = Uc.T @ Uc
        off = float(np.max(np.abs(gram - np.eye(k))))
        if not np.isfinite(off) or off > ortho_tol:
            raise ValueError(
                f"jno.fem: fem.solve(basis=U) needs an ORTHONORMAL basis (max|UᵀU - I| = {off:.2e}). A "
                "non-orthonormal basis is still a valid Galerkin projection, but restricting an initial "
                "state would then need (UᵀU)⁻¹ and would be silently wrong here. Orthonormalise it "
                "(np.linalg.qr(U)[0], or take it straight from jno.solve.svd, whose factors already are)."
            )
    return {"P": U, "kept_nodes": None, "vec": 1, "is_selection": False}


def reduce_op_periodic(op: Any, mode: str, periodic: dict) -> Any:
    """Apply the periodic Galerkin reduction ``P`` to a FEM operator, recursing into composite ops.

    One combinator over every :class:`FEM` ``_op`` representation, so a periodic tie composes with each
    feature without a per-combination branch:

    * ``"transient"`` -- eager reduction of the ``SemidiscreteTimeBlock`` (``P^T M P``, ``P^T A P``,
      runtime ``operator_fn``/``forcing_vector_fn``, restricted ``state0``) via
      :func:`_reduce_transient_block_periodic`.
    * ``"complex"`` (steady) -- ``op`` is a ``(re, im)`` tuple of real systems that share **one** FE
      space, hence **one** ``P``; each leg is reduced with the *same* ``periodic`` so the
      real-equivalent block ``[[A_r, -A_i], [A_i, A_r]]`` is preserved exactly. (A complex *transient*
      needs no case here: it is fused into a single 2n ``"transient"`` block at assembly, and
      :func:`_reduce_transient_block_periodic` duplicates ``P`` into ``blkdiag(P, P)`` for it.)
    * ``"linear"`` (a raw ``(A, b)``) and ``"nonlinear"`` (a residual operator) -- returned unchanged;
      these reduce lazily at solve time in :meth:`FEM.solve` (``P^T A P`` on the BCOO operator, or the
      ``P^T r(P·)`` residual wrap), which keeps the operator sparse and the residual matrix-free.
    """
    if mode == "complex":
        return tuple(reduce_op_periodic(leg, "linear", periodic) for leg in op)
    if mode == "transient":
        return _reduce_transient_block_periodic(op, periodic)
    return op


class Coupling:
    """A **nonlocal** residual term passed in the ``jno.fem([...])`` list.

    A weak term is *local* (a per-quadrature-point integrand, assembled element-by-element). Some physics
    is irreducibly *nonlocal* — enclosure radiation (every surface sees every other through the radiosity
    solve), integral / non-reflecting BCs, contact, peridynamics — and cannot be written as a local
    integrand. A ``Coupling`` carries a pure-JAX residual contribution ``residual_fn(u) -> (n_dofs,)`` that
    ``jno.fem`` **adds to the assembled residual**: ``R(u) = R_local(u) + sum_k coupling_k(u)``.

    This composes generally: a linear local form is *promoted* to a nonlinear residual operator
    (``R(u)=A u - b + sum_k coupling_k(u)``), a nonlinear one just gains the extra term; either way
    ``fem.solve()`` drives it with the matrix-free, ``custom_root``-differentiable ``newton_krylov`` (so it
    stays differentiable in any ``jno.np.parameter`` in the form, and trains through ``jno.core``). The
    contribution is zeroed on Dirichlet-pinned DOFs so it never corrupts a prescribed value.

    You write the nonlocal physics yourself as a pure-JAX residual of the DOFs and just **put the function
    in the** ``jno.fem([...])`` **list** -- no wrapper needed (a plain function/lambda is unambiguous next
    to the trace-node weak/Dirichlet terms). Grey-body enclosure radiation, for instance, is the radiosity
    solve on top of the enclosure geometry (``gap.field``/``view_factor``/``emissivity``/``load`` -- the
    geometry only; ``jno.fem`` never writes the physics for you)::

        F, eps = gap.view_factor, gap.emissivity({"hot": 0.8, "cold": 0.5})
        rho, eye, s_row = 1 - eps, jnp.eye(gap.size), F.sum(axis=1)

        def radiation(u):                                    # net grey-body surface load (n_dofs,)
            Tk = gap.field(u) + 273.15                       # absolute per-element temperature
            J = jnp.linalg.solve(eye - rho[:, None] * F, eps * SIGMA * Tk**4)   # radiosity
            return gap.load(s_row * J - F @ J)               # net flux -> consistent nodal load

        fem = jno.fem([conduction, radiation, u(xc, yc) - T_COOL])   # radiation is the bare function
        Tsol = fem.solve(u0=T_guess)                         # conduction + radiation, one implicit solve

    Wrapping it in ``jno.Coupling(fn, name=...)`` explicitly is still accepted -- needed only for a
    *callable object* (an instance with ``__call__``, which the bare-function detection deliberately skips),
    to give it a label, or to declare **trainable parameters** that live only inside the coupling. A
    ``jno.np.parameter`` in a *weak* term is found by walking the trace, but a coupling is an opaque
    pure-JAX function, so name its parameters explicitly with ``params=[...]``; the residual then takes a
    second argument, the ``{name: value}`` dict, so ``fem.solve()`` threads the trained values in and
    ``jno.core`` recovers them (e.g. a calibrated emissivity)::

        eps = jno.np.parameter((1,), name="eps")
        def radiation(u, p):                                 # p -> {"eps": value}
            J = jnp.linalg.solve(eye - (1 - p["eps"])[:, None] * F, p["eps"] * SIGMA * (gap.field(u) + 273.15)**4)
            return gap.load(s_row * J - F @ J)
        fem = jno.fem([conduction, jno.Coupling(radiation, params=[eps]), u(xc, yc) - T_COOL])

    In a **multifield** system, pass ``field_key=`` to act on one field's DOF sub-block: the residual then
    receives and returns *that field's* sub-vector (so the same ``gap.field``/``load`` scalar-T code works
    whether T stands alone or is one field of a heat+flow / thermo-mechanical system), and ``jno.fem``
    scatters it into the field's global block. Without ``field_key`` the coupling spans the whole vector
    (single field). The ``field_key`` is the trial field's key (its ``value_shape``/name, as elsewhere).

    Caveats (must be a *pure-JAX* function of the DOFs): a numpy/scipy-only coupling cannot go in-residual;
    and the matrix-free default solver may need a tailored ``fem.solve(solve_fn=...)`` for a stiff/dense
    coupling. The targeted field must be scalar P1 (its block is node-indexed, as ``gap.load`` assumes)."""

    def __init__(self, residual_fn: Callable, *, name: str = "coupling", field_key: Any = None, params=None):
        # residual_fn: (u_flat,) -> (n_dofs,), or (u_flat, {name: value}) -> (n_dofs,) when params are declared
        self.residual_fn = residual_fn
        self.name = str(name)
        self.field_key = field_key
        self.params = list(params or [])  # jno.np.parameter nodes used only inside this coupling

    def __repr__(self):
        return f"Coupling({self.name!r})"


def _wrap_couplings(domain: Any, fem_obj: "FEM", couplings: List["Coupling"]) -> "FEM":
    """Fold nonlocal :class:`Coupling` terms into ``fem_obj``'s residual: ``R(u) = R_local(u) + sum_k c_k(u)``,
    zeroed on the Dirichlet-pinned DOFs. A *linear* local form is promoted to a nonlinear residual operator
    (so ``fem.solve()`` uses ``newton_krylov``); a *nonlinear* one gains the extra term. A coupling with a
    ``field_key`` acts on that field's DOF sub-block (multifield, e.g. radiation on T in a heat+flow system);
    without one it acts on the whole vector (single field). For a **transient** form the coupling enters each
    implicit step: a nonlinear time block gains the term in its residual, a linear one is promoted to a
    nonlinear (backward-Euler) block -- so e.g. enclosure radiation over a heating cycle solves in-residual."""
    if fem_obj._mode not in ("linear", "nonlinear", "transient"):
        raise NotImplementedError(f"jno.fem: nonlocal Coupling terms are not supported for mode {fem_obj._mode!r}.")
    n = int(fem_obj.dofs)
    pairs = getattr(domain, "_fem_native_dirichlet_pairs", None) or []
    d_dofs = jnp.asarray([int(p[0]) for p in pairs], dtype=jnp.int32) if pairs else None

    # Resolve each coupling's target DOF slice. A `field_key` selects one field's block [off_k:off_{k+1}]
    # (authoritative order: domain._fem_native_field_keys; boundaries: fem_obj.offsets) -- the coupling then
    # sees and returns *that field's* sub-vector, so the same residual (e.g. gap.field/load on the scalar T
    # nodes) works whether T stands alone or is one field of a coupled system. No field_key -> whole vector.
    offsets = fem_obj.offsets
    field_keys = getattr(domain, "_fem_native_field_keys", None)

    def _slice_for(c):
        if c.field_key is None or not offsets or not field_keys:
            return 0, n
        if c.field_key not in field_keys:
            raise ValueError(f"jno.fem: Coupling field_key {c.field_key!r} is not among the fields {list(field_keys)}.")
        k = field_keys.index(c.field_key)
        return int(offsets[k]), int(offsets[k + 1])

    slices = [_slice_for(c) for c in couplings]

    def coupling_residual(u, args=None):  # sum of the per-coupling contributions, zeroed on pinned rows
        u = jnp.asarray(u).reshape(-1)
        total = jnp.zeros((n,), dtype=u.dtype)
        for c, (lo, hi) in zip(couplings, slices):
            u_c = u[lo:hi]  # the targeted field's sub-block (whole vector when no field_key)
            # a coupling that declared `params=[...]` reads them from the threaded {name: value} dict
            contrib = c.residual_fn(u_c, args or {}) if c.params else c.residual_fn(u_c)
            total = total.at[lo:hi].add(jnp.asarray(contrib, dtype=u.dtype).reshape(-1))
        return total if d_dofs is None else total.at[d_dofs].set(0.0)

    # Merge parameters declared on the couplings into the operator's runtime params so they appear in the
    # solve's runtime args (and thus as trainable inputs of the fem.solve() FunctionCall) -- the trace walk
    # that finds weak-form params never sees an opaque coupling function. `_collect_runtime_parameter_exprs`
    # keys by name and raises on a name reused for a different parameter, so a param shared with the weak
    # form is deduped.
    from .utils.solver.parametric_helpers import _collect_runtime_parameter_exprs

    def _merge_coupling_params(base):
        out = dict(base or {})
        for c in couplings:
            for p in c.params:
                _collect_runtime_parameter_exprs(p, out)
        return out

    if fem_obj._mode == "transient":
        # Inject the coupling into the implicit time step. The step prefers the nonlinear payload
        # (M(t)(u_next-u)/dt + R(u_next,t,args)=0, matrix-free Newton-Krylov), so a nonlinear block just
        # gains the term; a linear block (M u_dot + A u = c + f) is promoted to that nonlinear (backward-
        # Euler) form with R(u,t,args) = A(t,args) u - c - f(t,args) + coupling. Periodic transient runs in
        # a reduced DOF space the coupling cannot address, so it is refused.
        block = fem_obj._op
        if getattr(block, "prolongation", None) is not None:
            raise NotImplementedError("jno.fem: transient Coupling is not supported together with periodic ties.")
        block.runtime_parameter_exprs = _merge_coupling_params(getattr(block, "runtime_parameter_exprs", {}))
        if block.is_nonlinear():
            _base = block.residual

            def _t_residual(u, t, args=None, _b=_base):
                return jnp.asarray(_b(u, t, args)).reshape(-1) + coupling_residual(u, args)

            block.residual = _t_residual
        else:  # promote the linear payload -> nonlinear so the coupling enters the backward-Euler step
            _M, _A, _opfn = block.M, block.A, block.operator_fn
            _fvfn, _cbias = block.forcing_vector_fn, block.affine_bias

            def _operand(x):
                return x if hasattr(x, "todense") else jnp.asarray(x)

            def _t_mass(t, args=None, _m=_M):
                return _m

            def _t_residual(u, t, args=None):
                u = jnp.asarray(u).reshape(-1)
                A = _operand(_opfn(t, args or {}) if _opfn is not None else _A)
                c = 0.0 if _cbias is None else jnp.asarray(_cbias, u.dtype).reshape(-1)
                f = 0.0 if _fvfn is None else jnp.asarray(_fvfn(t, args or {}), u.dtype).reshape(-1)
                return (A @ u) - c - f + coupling_residual(u, args)

            block.mass, block.residual = _t_mass, _t_residual
        return fem_obj

    op = fem_obj._op
    rpe = _merge_coupling_params(getattr(op, "runtime_parameter_exprs", {}))
    if fem_obj._mode == "linear":
        # local residual R_local(u, args) = A(args) u - b(args)
        def residual(u, args=None, _op=op):
            u = jnp.asarray(u).reshape(-1)
            if isinstance(_op, FemLinearSystem):
                A, b = _op.evaluate(args)
            else:  # raw (A, b) tuple (non-parametric)
                A, b = _op
            return (A @ u) - jnp.asarray(b).reshape(-1) + coupling_residual(u, args)

        fem_obj._op = FemResidualOperator(residual, size=n, runtime_parameter_exprs=rpe)
        fem_obj._mode = "nonlinear"
        fem_obj._A = fem_obj._b = None
    else:  # already a nonlinear residual operator -> add the coupling
        base_residual = op.residual

        def residual(u, args=None, _base=base_residual):
            return jnp.asarray(_base(u, args)).reshape(-1) + coupling_residual(u, args)

        fem_obj._op = FemResidualOperator(
            residual, jacobian_fn=None, size=n, runtime_parameter_exprs=rpe, metadata=getattr(op, "metadata", None)
        )
    return fem_obj


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def fem(
    constraints: Any,
    *,
    quad_degree: int = 2,
    element_type: Optional[str] = None,
    vec: Optional[int] = None,
    _dd_overlap: bool = False,
) -> FEM:
    """Assemble a flat list of traced residuals into an :class:`FEM`.

    Parameters
    ----------
    constraints:
        A residual expression or list of them. Weak terms (containing the test
        function) and essential conditions (``u(region) - g``) are auto-classified.
    quad_degree, element_type:
        FEM discretisation options (forwarded to the quadrature setup).
    vec:
        Vector size of the unknown. ``None`` (default) infers it from the trial's
        ``value_shape`` (scalar → 1, ``(2,)`` → 2, …); pass an int to override.
    """
    from .utils.solver.fem_route import neumann
    from .utils.solver.weak_form import (
        LoweredChannelTerm,
        LoweredWeakForm,
        _apply_sign,
        _contains_temporal_derivative,
        _split_additive_terms,
    )

    if not isinstance(constraints, (list, tuple)):
        constraints = [constraints]
    if len(constraints) == 0:
        raise ValueError("jno.fem: no constraints provided.")

    # Keep the user's original constraint list + build options so the adaptive driver
    # (``FEM.solve(adapt=...)``) can re-assemble the *same* problem after the domain is
    # remeshed in place -- the constraints reference the domain (not a mesh snapshot),
    # so re-tracing them picks up the refined mesh automatically.
    _orig_constraints = list(constraints)
    _orig_fem_kwargs = {"quad_degree": quad_degree, "element_type": element_type, "vec": vec}

    # Gauge pins (`p.pin()`) remove a field's constant null space. Lower each to a single-node
    # Dirichlet `p(node) - value` *before* domain discovery and classification: a GaugePin is a
    # bare marker, not a walkable expression, so it must not reach `_discover_domain` (which walks
    # coordinate Variables) or the `_region_and_support` classifier.
    if any(isinstance(c, GaugePin) for c in constraints):
        pins = [c for c in constraints if isinstance(c, GaugePin)]
        constraints = [c for c in constraints if not isinstance(c, GaugePin)] + [_lower_gauge_pin(p) for p in pins]

    # Nonlocal coupling terms (radiation, integral/non-reflecting BCs, ...) are not local weak forms and
    # not walkable trace expressions: a plain pure-JAX residual function ``f(u) -> (n_dofs,)``. A bare
    # Python function/lambda/partial in the list is taken as one (weak-form and Dirichlet terms are trace
    # nodes, never plain callables), so no wrapper is needed; an explicit ``Coupling`` is still accepted.
    # Separate them up front (before domain discovery / classification) and fold them into the residual
    # after assembly (see _wrap_couplings / _finalize).
    couplings: List[Coupling] = []
    _rest: List[Any] = []
    for c in constraints:
        if isinstance(c, Coupling):
            couplings.append(c)
        elif inspect.isfunction(c) or inspect.ismethod(c) or isinstance(c, functools.partial):
            couplings.append(Coupling(c, name=getattr(c, "__name__", "coupling")))
        else:
            _rest.append(c)
    constraints = _rest

    # The bare-function shorthand only catches a *plain* function/lambda/partial -- it deliberately skips
    # callable trace nodes (a bare symbol, `jno.fn(...)`) so it can never swallow a real term. The flip side
    # is it also can't see a callable *object* meant as a coupling -- a `jax.jit(residual)` or an instance
    # with `__call__`. Rather than let that fall through to weak/Dirichlet classification and fail with an
    # opaque error, point the user at the explicit wrapper. (Trace expressions are not callable; the only
    # callable list-items are Placeholders, which keep their own handling.)
    for c in constraints:
        if callable(c) and not isinstance(c, Placeholder):
            raise TypeError(
                f"jno.fem: got a callable {type(c).__name__!s} in the constraint list. A *plain* function "
                "is taken as a nonlocal coupling residual automatically, but a jitted/object callable is "
                "not -- wrap it explicitly as jno.Coupling(fn) to use it as a coupling term."
            )

    domain = _discover_domain(constraints)

    # Nédélec (N1E) periodic ties need a CONFORMING periodic mesh — the per-edge DOFs must line up
    # one-to-one across the tied faces, which gmsh's default unstructured mesh does not guarantee. Infer
    # this from the constraint list: when periodic ties are present on an N1E field, re-mesh the (Shape-
    # backed) domain once with gmsh setPeriodic on the tied face pairs. No `periodic=` arg — driven purely
    # by the periodic conditions the user already authored. (Nodal fields tie by interpolation and need no
    # re-mesh, so this is gated on N1E.)
    if "N1E" in _trial_spaces(constraints) and hasattr(domain, "_remesh_periodic"):
        _pairs = [(s[0], s[1]) for c in constraints if (s := _periodic_tie_spec(c, domain)) is not None]
        if _pairs:
            domain._remesh_periodic(_pairs)

    # Periodic ties `u(A) - u(B)` are enforced by algebraic reduction (a prolongation P that
    # eliminates the slave-face DOFs), not by assembly. Separate them out *before* the weak/Dirichlet
    # classification (`_region_and_support` would otherwise reject a residual that spans two regions).
    periodic_ties: List[Any] = []
    core_constraints: List[Any] = []
    for c in constraints:
        spec = _periodic_tie_spec(c, domain)
        (periodic_ties.append(spec) if spec is not None else core_constraints.append(c))
    constraints = core_constraints
    if periodic_ties and not constraints:
        raise ValueError("jno.fem: only periodic ties were given — add the PDE weak form (and any other conditions).")

    # Essential normal-flux BCs `u·n - g` (H(div) RT) pin boundary-edge DOFs at assembly; like periodic
    # ties they must be separated before classification (the Cartesian Dirichlet parser would reject them).
    flux_bcs: List[Any] = []
    _core: List[Any] = []
    for c in constraints:
        spec = _normal_flux_spec(c, domain) or _tangential_bc_spec(c, domain)  # RT u·n / N1E u×n (incl. 3-D PEC)
        (flux_bcs.append(spec) if spec is not None else _core.append(c))
    constraints = _core

    # Essential rotation BCs `u.dn(region) - h` (∂u/∂n on a C¹/Morley plate field) pin the boundary
    # normal-derivative DOFs at assembly; separated before classification like the normal-flux BC.
    rotation_bcs: List[Any] = []
    _core_r: List[Any] = []
    for c in constraints:
        spec = _rotation_bc_spec(c, domain)
        (rotation_bcs.append(spec) if spec is not None else _core_r.append(c))
    constraints = _core_r

    # Internal-state EVOLUTION terms (`state.evolves(formula)`): pulled out BEFORE weak-form/Dirichlet
    # classification and field/space inference. A StateUpdate carries no test function and is NOT an
    # equation — but its formula references the trial (and its own past `state.i(-1)`), which would
    # otherwise mis-route it to the Dirichlet branch and mis-count its state field as a coupled unknown.
    # The `.i(k)` reads inside each formula are still walked by ``history_variables`` (in the assembler)
    # so they allocate the right per-quadrature-point buffer depth.
    _evolution = state_updates(constraints)  # {history_key: StateUpdate}
    constraints = [c for c in constraints if not isinstance(_bare(c), StateUpdate)]
    if rotation_bcs and not (_trial_spaces(constraints) - {"Lagrange"}):
        raise NotImplementedError(
            "jno.fem: a rotation BC `u.dn(region) - h` is a 4th-order plate essential BC — it requires a field "
            "on the Argyris or Morley element (`space='Argyris'`/`'Morley'`), not C⁰ Lagrange (which has no "
            "normal-derivative DOF)."
        )

    # A network ModelCall in the constraints plays one of two roles, decided at the SET level:
    #   * VPINN — the network *replaces* the trial (``u = net(x, y)`` written into the weak form):
    #     no weak (test-carrying) constraint contains a TrialFunction. jno.fem test-projects the
    #     weak form onto the FE test space -> a trainable residual loss, not an FE system.
    #   * neural COEFFICIENT — the network multiplies a genuine trial (``net(x,y)*u.dx*v.dx``):
    #     some weak constraint carries the TrialFunction. The system is assembled as usual and the
    #     kernel re-evaluates the network at the quadrature points (trainable via the runtime args).
    _has_network = any(_contains_network_call(c) for c in constraints)
    _weak_has_real_trial = any(_contains(c, TestFunction) and _contains(c, TrialFunction) for c in constraints)
    is_vpinn = _has_network and not _weak_has_real_trial
    has_neural_coeff = _has_network and not is_vpinn

    if has_neural_coeff:
        # A network in a trial-only (essential) constraint is a trainable *essential value*, NOT an
        # integrand coefficient: a *Dirichlet value* ``u(∂Ω) - net(x)`` (an unknown boundary profile) or an
        # *initial condition* ``u(initial) - net(x)`` (an unknown starting state, recovered from a
        # trajectory). Both are supported as a *bare* net, native Lagrange single-field — the parametric
        # path evaluates ``net(coords)`` from the runtime args each solve (a Dirichlet lift / the state0).
        # Everything else -- a compound value ``1+net(x)``, non-nodal or multifield -- is rejected.
        _net_trial_only = [
            c
            for c in constraints
            if _contains(c, TrialFunction) and not _contains(c, TestFunction) and _contains_network_call(c)
        ]
        for c in _net_trial_only:
            support, _rg = _region_and_support(c, domain)
            _comp, _val, _vnode = _dirichlet_spec(_bare(c))
            if support not in ("boundary", "initial") or not _is_bare_neural_value_node(_vnode):
                raise NotImplementedError(
                    "jno.fem: a network in an essential (trial-only) constraint must be a *bare* net(x) value "
                    "on a boundary region (a Dirichlet profile) or the 'initial' region (an initial "
                    "condition) — a compound expression (e.g. 1 + net(x)) is not supported."
                )
        if _net_trial_only:
            if _trial_spaces(constraints) - {"Lagrange"}:
                raise NotImplementedError("jno.fem: a net-valued essential value is supported on Lagrange elements only.")
            if len(_field_keys(constraints)) > 1:
                raise NotImplementedError("jno.fem: a net-valued essential value is single-field only.")
        if getattr(domain, "dimension", None) == 1 and len(_field_keys(constraints)) > 1:
            # the coupled 1D block assembler threads neither runtime parameters nor networks
            raise NotImplementedError(
                "jno.fem: a neural coefficient on a 1D domain is single-field only — the coupled 1D block "
                "assembler threads no runtime parameters. Use a single field, or a 2D/3D domain."
            )
        # Non-nodal: the scalar C¹ families (Argyris/Morley/Hermite) thread the network at the quad points
        # like their P1 field parameter. The vector edge families (RT/Nédélec) accept a *scalar coordinate*
        # net(x) coefficient (a spatially-varying permeability/permittivity multiplying a vector term), but
        # a *solution-dependent* net(u) there would feed the vector-valued trial into the network, which is
        # undefined — reject only that.
        _nonnodal_spaces = _trial_spaces(constraints) - {"Lagrange", "Argyris", "Morley", "Hermite"}
        if _nonnodal_spaces:
            from .utils.solver.weak_form import _is_obviously_nonlinear_in_unknown as _nlin_nn

            if any(_contains_network_call(c) and _nlin_nn(domain, _bare(c)) for c in constraints):
                raise NotImplementedError(
                    "jno.fem: a solution-dependent neural coefficient net(u) on the vector edge families "
                    f"(RT/Nédélec, {sorted(_nonnodal_spaces)}) is not supported — a vector-valued trial input to the "
                    "network is undefined. A scalar coordinate net(x) coefficient is supported."
                )

    multifield = len(_field_keys(constraints)) > 1
    if vec is None and not multifield:
        vec = _infer_vec(constraints)  # single-field only (coupled fields carry per-field vec)

    # Evolution terms ride the real, steady, single-field native-Lagrange path only (the plasticity
    # load-path march). Reject the structurally-incompatible routes up front — fail loud, never a
    # silently dropped update. (transient / complex are rejected below, once the IR reveals them.)
    if _evolution and (
        multifield
        or is_vpinn
        or getattr(domain, "dimension", None) == 1
        or (_trial_spaces(constraints) - {"Lagrange"})
        or periodic_ties
    ):
        raise NotImplementedError(
            "jno.fem: `state.evolves(...)` evolution terms are supported on the real, steady, single-field "
            "native-Lagrange path only (the plasticity load-path march over `domain(tau=...)`). Not yet: "
            "multifield, VPINN, 1D, non-nodal (Argyris/Morley/edge) elements, or periodic ties."
        )

    # The transient route reduces M/A from the assembly context at *assembly* time, so its P must be
    # built and injected before the time block is assembled (see the single-field transient branch below).
    # This holder carries that P so `_finalize` reuses it rather than rebuilding.
    periodic_holder: List[Any] = []

    def _finalize(fem_obj: "FEM") -> "FEM":
        """Attach the periodic reduction (if any): linear & nonlinear via ``FEM.solve``, transient via
        the time route's existing context-driven reduction. Single-field, vector, and coupled
        multi-field are all reduced block-wise (P_i^T M[i,j] P_j); complex / runtime-parametric remain out."""
        fem_obj._term_source = (domain, volume_terms)
        fem_obj._constraints = _orig_constraints
        fem_obj._fem_kwargs = _orig_fem_kwargs
        # Field-key snapshots for FEM.block_index: the assembler's list is offsets-ordered (and
        # must be captured NOW — a later assembly on the same domain overwrites the attribute);
        # the constraint-walk order is the fallback for paths that don't run the native assembler.
        fem_obj._block_field_keys = list(getattr(domain, "_fem_native_field_keys", None) or ())
        fem_obj._trial_field_keys = _field_keys(_orig_constraints)
        # Same snapshot treatment for the DOF coordinates behind .points / .field_points — an
        # auxiliary assembly (jno.precond.form) would otherwise clobber them mid-solve.
        fem_obj._native_dof_points = getattr(domain, "_fem_native_dof_points", None)
        _all = getattr(domain, "_fem_native_dof_points_all", None)
        fem_obj._native_dof_points_all = list(_all) if _all is not None else None
        if couplings and periodic_ties and fem_obj.is_transient:
            # The native periodic *transient* block reduces eagerly into a master-DOF space; the coupling
            # residual is written in the full nodal space, so the two cannot be composed on that path yet.
            raise NotImplementedError(
                "jno.fem: periodic ties combined with a *transient* nonlocal Coupling are not yet supported."
            )
        if couplings:
            # Fold the coupling into the residual FIRST -- a steady local form is promoted to a nonlinear
            # FemResidualOperator. The periodic reduction below then wraps the *coupled* residual through
            # the nonlinear Pᵀr(P·) solve path, so a periodic tie + Coupling compose with no extra branch.
            fem_obj = _wrap_couplings(domain, fem_obj, couplings)
        if not periodic_ties:
            return _fuse_complex_steady(fem_obj)
        # Single-field transient was already reduced by the dedicated native branch -> reuse its P.
        if periodic_holder:
            fem_obj._periodic = periodic_holder[0]
            return _fuse_complex_steady(fem_obj)
        # Build the reduction (single-field / vector, or coupled multifield). (1D has no assembly cells
        # -> flat-chain facets via points only.)
        prob = getattr(fem_obj, "problem", None)
        if prob is not None:
            cells, ele_order = _assembly_cells(prob)
        else:
            # Native path: no problem object -- read the assembly cells the native assembler stashed
            # (``None`` for the native 1D route, which falls back to flat-chain facets on points).
            cells = getattr(domain, "_fem_native_assembly_cells", None)
            ele_order = int(getattr(domain, "_fem_native_assembly_order", 1))
        _nonnodal_topo = getattr(domain, "_fem_nonnodal_topology", None)
        if _nonnodal_topo is not None and _nonnodal_topo.get("family") == "N1E":
            # Nédélec N1E (H(curl) edge): DOF-level edge prolongation (Floquet/Bloch, with orientation sign)
            periodic = _build_periodic_reduction_n1e(domain, periodic_ties, fem_obj.offsets)
        elif _nonnodal_topo is not None:
            # non-nodal C¹ (Morley): DOF-level reduction (value + signed edge-derivative ties)
            periodic = _build_periodic_reduction_nonnodal(domain, periodic_ties, fem_obj.offsets)
        elif multifield:
            periodic = _build_periodic_reduction_multifield(
                domain, periodic_ties, fem_obj.points, cells, ele_order, fem_obj.offsets
            )
        else:
            periodic = _build_periodic_reduction(domain, periodic_ties, fem_obj.points, cells, ele_order, vec or 1)
        # One op-level reduction combinator handles every representation: a transient block is reduced
        # now (P^T M P, ...) — including a fused complex transient, whose stacked [Re; Im] state reduces by
        # blkdiag(P, P) — and a steady complex (re, im) tuple is reduced leg-wise with the same P; steady
        # linear (A, b) and nonlinear residual ops are returned unchanged and reduce lazily in FEM.solve.
        fem_obj._op = reduce_op_periodic(fem_obj._op, fem_obj._mode, periodic)
        fem_obj._periodic = periodic
        return _fuse_complex_steady(fem_obj)

    volume_terms: List[Any] = []
    boundary_terms: dict[str, List[Any]] = {}
    dirichlet_values: dict[str, Any] = {}
    # All-component vs per-component Dirichlet is tracked per (field, region), not per region:
    # in a multi-field problem a vector field's per-component BC and a scalar field's
    # all-component BC may legitimately share a wall. (The actual coupled BCs come from the
    # field-keyed `dirichlet_raw` below; `dirichlet_values` only feeds the single-field path.)
    dirichlet_style: dict[tuple, str] = {}  # (field_key, region) -> "all" | "per_component"
    dirichlet_raw: List[Any] = []  # (field_key, region, comp, value) for the multi-field path
    ic_residuals: List[Any] = []
    classification: List[str] = []

    # Classify every constraint against its ORIGINAL coordinate tags before touching
    # any of them. `_retag_coords_for_quadrature` mutates the shared coordinate
    # Variables in place, so classifying up front prevents one constraint's retag from
    # leaking into another's region detection — e.g. two surface terms on the same
    # region (a Neumann load + a Robin term) that share the bound boundary coordinates.
    classified = [
        (c, _contains(c, TestFunction), _contains(c, TrialFunction), *_region_and_support(c, domain)) for c in constraints
    ]

    for c, has_test, has_trial, support, region in classified:
        if support == "initial":
            # initial condition: a trial-only residual `u(initial) - u0` on the t0 slice
            if has_test:
                raise ValueError("jno.fem: an initial-condition residual on 'initial' must not contain the test function.")
            ic_residuals.append(_bare(c))
            classification.append("initial")
        elif has_test:
            _retag_coords_for_quadrature(c, support, region)
            bare = _bare(c)
            if support == "volume":
                if region != "volume" and not _dd_overlap:
                    # Sub-domain term: multiply by the region's per-cell indicator so it integrates over
                    # that region's cells only. Stays a plain volume term (whole-mesh quadrature); the
                    # RegionMask zeroes the integrand outside the region (resolved in the assembly kernel).
                    # ``_dd_overlap`` skips this: a region-local matrix can't couple in an overlapping-Schwarz
                    # step (its artificial boundary reaches no neighbour cells), so the overlap driver rebuilds
                    # the FEM WHOLE-MESH while keeping the ``volume@{region}`` label below (still detectable).
                    bare = RegionMask(region) * bare
                volume_terms.append(bare)
                classification.append("volume" if region == "volume" else f"volume@{region}")
            else:
                boundary_terms.setdefault(region, []).append(bare)
                classification.append(f"surface@{region}")
        elif has_trial:
            # A trial-only residual is a Dirichlet pin `u(region) - g`. It lives on a boundary region,
            # the 'initial' region (IC), OR a **named interior sub-region** (`domain.region(name, poly)`,
            # keyed in `_source_regions`) — pinning that region's whole node set, a volumetric hard
            # constraint used by subdomain / domain-decomposition solves. The default whole-domain
            # `volume` is still rejected (that signals a forgotten test function).
            is_subregion_pin = support == "volume" and region in (getattr(domain, "_source_regions", {}) or {})
            if support != "boundary" and not is_subregion_pin:
                raise ValueError(
                    "jno.fem: a residual with the trial but no test function must live on a boundary "
                    "region (Dirichlet), the 'initial' region (IC), or a named interior sub-region "
                    "(domain.region(...)). Got the whole-domain volume — did you forget the test function?"
                )
            comp, value, value_node = _dirichlet_spec(_bare(c))
            fk = _field_key_of(c)
            dirichlet_raw.append((fk, region, comp, value, value_node))
            style_key = (fk, region)
            if comp is None:  # all components: u(region) - g
                if dirichlet_style.get(style_key) == "per_component":
                    raise ValueError(
                        f"jno.fem: the same field on region {region!r} mixes all-component "
                        f"(u({region})-g) and per-component (u({region})[i]-g) Dirichlet."
                    )
                dirichlet_style[style_key] = "all"
                dirichlet_values[region] = value
                classification.append(f"dirichlet@{region}")
            else:  # one component (roller/symmetry): u(region)[i] - g
                if comp not in _COMPONENT_NAMES:
                    raise ValueError(
                        f"jno.fem: Dirichlet component index {comp} out of range (vector components are 0..2)."
                    )
                if dirichlet_style.get(style_key) == "all":
                    raise ValueError(
                        f"jno.fem: the same field on region {region!r} mixes all-component "
                        f"(u({region})-g) and per-component (u({region})[i]-g) Dirichlet."
                    )
                dirichlet_style[style_key] = "per_component"
                current = dirichlet_values.get(region)
                current = dict(current) if isinstance(current, dict) else {}
                current[_COMPONENT_NAMES[comp]] = value
                dirichlet_values[region] = current
                classification.append(f"dirichlet@{region}[{_COMPONENT_NAMES[comp]}]")
        else:
            raise ValueError("jno.fem: a residual contains neither the trial nor the test function.")

    if is_vpinn and (getattr(domain, "dimension", None) == 1 or multifield):
        raise NotImplementedError("jno.fem VPINN (network trial) is currently single-field 2D/3D only.")

    # ---- second-order in time (`u_tt`): reduce to a first-order augmented (u, v=u_t) block ----
    # A weak term carrying a SECOND temporal derivative is lowered to the equivalent first-order
    # system in y=[u, v] (v=u_t) and integrated by the energy-conserving trapezoidal rule -- the
    # canonical wave / elastodynamics path the first-order route cannot express. Intercept here,
    # before the 1D / non-nodal / multifield branches, with explicit fail-loud scope guards.
    from .utils.solver.solver_helper import max_temporal_derivative_order as _max_temporal_order

    # Lagrange u_tt -> the native augmented [u, v] block here. A NON-NODAL (Argyris/Hermite) u_tt is NOT
    # intercepted: it falls through to the non-nodal branch below, which builds the same augmented block from
    # its own push-forward assembly / pins / IC-projection (so the C¹ element gets dynamic plates too).
    # 1D u_tt falls through to the native 1D branch (its assembler builds the augmented [u, v] block);
    # only the 2D/3D nodal-Lagrange route is intercepted here (a non-nodal / 1D u_tt has its own path).
    _second_order = any(_max_temporal_order(_bare(c)) >= 2 for c in constraints)
    if _second_order and getattr(domain, "dimension", None) != 1 and not (_trial_spaces(constraints) - {"Lagrange"}):
        if multifield:
            # Coupled second-order-in-time: all fields must be second-order (u_tt) — reduces to the
            # single-field formula with the coupled block M2/K. Mixed-order/damped/periodic/parametric
            # coupled 2nd-order fail loud inside the assembler / here.
            if periodic_ties:
                raise NotImplementedError(
                    "jno.fem: periodic ties on a coupled second-order-in-time form are not supported yet."
                )
            _so = _assemble_multifield_second_order(
                domain, volume_terms, boundary_terms, dirichlet_raw, ic_residuals, classification, quad_degree=quad_degree
            )
            _so._term_source = (domain, volume_terms)
            return _so
        if any(isinstance(n, RegionMask) for b in volume_terms for n in _walk(b)):
            raise NotImplementedError(
                "jno.fem: per-region (RegionMask) integration on a second-order-in-time problem is not "
                "wired yet — the region-grouping distributes the mask over the whole form, including the "
                "u_tt mass term. Use a jno.fn indicator coefficient instead, e.g. (1 + k*ind)*(u.x*v.x + "
                "u.y*v.y), which gives the same piecewise-material dynamics."
            )
        _so = _assemble_second_order_time(
            domain,
            volume_terms,
            boundary_terms,
            dirichlet_values,
            dirichlet_raw,
            ic_residuals,
            classification,
            order=_infer_order(constraints),
            vec=vec or 1,
            quad_degree=quad_degree,
        )
        _so._term_source = (domain, volume_terms)
        if periodic_ties:
            # Bloch / phononic in the time domain: reduce the augmented [u, v] block by the field
            # prolongation P (duplicated per block inside _reduce_transient_block_periodic).
            _cells = getattr(domain, "_fem_native_assembly_cells", None)
            _eo = int(getattr(domain, "_fem_native_assembly_order", 1))
            _periodic = _build_periodic_reduction(domain, periodic_ties, _so.points, _cells, _eo, vec or 1)
            _so._op = reduce_op_periodic(_so._op, "transient", _periodic)
            _so._periodic = _periodic
        return _so

    # ---- non-nodal element families (RT / Nedelec / Argyris): native push-forward assembler ----
    # These families need a basis push-forward, so -- like the 1D path -- assemble natively and reuse
    # the shared integrand evaluator (which carries space-guarded branches for the physical basis).
    if _trial_spaces(constraints) - {"Lagrange"}:
        from .utils.solver.fem_nonnodal import assemble_fem_nonnodal

        # ---- complex non-nodal (RT/Nédélec/Argyris): the Re/Im coefficient split, assembled by the
        # non-nodal push-forward assembler. The basis is real, so ``Re(c·T) = Re(c)·T``: wrapping each
        # term in ``.real``/``.imag`` gives two ordinary real systems A_r/A_i, and FEM.solve() forms
        # ``[[A_r,-A_i],[A_i,A_r]]`` (``mode="complex"`` → _solve_complex_block), recombining to a complex
        # ``u`` — needed for time-harmonic Maxwell (complex ε, the ``i k₀`` impedance BC). A runtime
        # parameter is allowed (the complex *inverse*): each ``.real``/``.imag`` leg carries the parameter,
        # so ``assemble_fem_nonnodal`` returns parametric ``FemLinearSystem`` legs and ``_solve_complex_block``
        # builds a differentiable trace node — exactly like the nodal complex inverse. Without this split the
        # plain real assembler would SILENTLY cast the imaginary part away, so the compositions that are still
        # unwired (transient / nonlinear / neural-coefficient) raise rather than mislead. ----
        _nn_bares = volume_terms + [e for exprs in boundary_terms.values() for e in exprs]
        if _bares_have_complex_coeff(_nn_bares):
            from .utils.solver.weak_form import _contains_temporal_derivative as _ctd_nn
            from .utils.solver.weak_form import _is_obviously_nonlinear_in_unknown as _nlin_nn

            if any(_ctd_nn(b) for b in _nn_bares):
                raise NotImplementedError(
                    "jno.fem: a complex non-nodal (RT/Nédélec/Argyris) *transient* form is not wired — only "
                    "the steady linear complex form is. (The real assembler would silently drop the imaginary "
                    "part, so this raises instead.)"
                )
            if has_neural_coeff:
                raise NotImplementedError(
                    "jno.fem: a *neural-coefficient* complex non-nodal (RT/Nédélec/Argyris) form is not wired "
                    "yet — the runtime-parameter complex non-nodal inverse IS (below). (Raises rather than "
                    "silently dropping the imaginary part.)"
                )
            if any(_nlin_nn(domain, b) for b in _nn_bares):
                raise NotImplementedError(
                    "jno.fem: a complex non-nodal *nonlinear* form is not wired — complex forms assemble as "
                    "linear real-equivalent blocks. (Raises rather than silently dropping the imaginary part.)"
                )
            real_bd = {tag: [e.real for e in exprs] for tag, exprs in boundary_terms.items()}
            imag_bd = {tag: [e.imag for e in exprs] for tag, exprs in boundary_terms.items()}
            domain._fem_problem = None
            op_r, _mr, offsets = assemble_fem_nonnodal(
                domain,
                [b.real for b in volume_terms],
                real_bd,
                dirichlet_raw,
                [],
                flux_bcs=flux_bcs,
                rotation_bcs=rotation_bcs,
                quad_degree=quad_degree,
            )
            op_i, _mi, _oi = assemble_fem_nonnodal(
                domain,
                [b.imag for b in volume_terms],
                imag_bd,
                dirichlet_raw,
                [],
                flux_bcs=flux_bcs,
                rotation_bcs=rotation_bcs,
                quad_degree=quad_degree,
            )
            return _finalize(
                FEM(domain=domain, op=(op_r, op_i), classification=classification, mode="complex", offsets=offsets)
            )

        op, mode, offsets = assemble_fem_nonnodal(
            domain,
            volume_terms,
            boundary_terms,
            dirichlet_raw,
            ic_residuals,
            flux_bcs=flux_bcs,
            rotation_bcs=rotation_bcs,
            quad_degree=quad_degree,
        )
        return _finalize(FEM(domain=domain, op=op, classification=classification, mode=mode, offsets=offsets))

    # ---- 1D (segment): a 1D line domain is assembled by the native 1D path ----
    # The native 1D assembler reuses the same integrand evaluator and returns the
    # same (op, mode) the FEM container expects; it needs no problem-object
    # scaffolding (coordinate vars resolve from the per-element quadrature points).
    if getattr(domain, "dimension", None) == 1:
        from .utils.solver.fem_1d import assemble_fem_1d, assemble_fem_1d_multifield

        _order_1d = _infer_order(constraints)
        if _order_1d > 2:
            raise NotImplementedError(
                f"jno.fem: 1D Lagrange elements are implemented for order 1 (LINE2) and 2 (LINE3); "
                f"got order={_order_1d}. Use order<=2, or refine the mesh for accuracy."
            )
        if _order_1d > 1 and multifield:
            raise NotImplementedError(
                "jno.fem: higher-order (P2 / LINE3) elements on a 1D domain are single-field only for "
                "now -- the coupled 1D block assembler is P1. Use order=1 for the coupled system."
            )
        if multifield:  # coupled 1D -> native block assembly
            if _second_order:
                raise NotImplementedError(
                    "jno.fem: second-order-in-time (u_tt) 1D is single-field only for now; write the coupled "
                    "problem as a first-order system (one velocity field per second-order field)."
                )
            op, mode = assemble_fem_1d_multifield(
                domain, volume_terms, boundary_terms, dirichlet_raw, ic_residuals, quad_degree=quad_degree
            )
        else:
            op, mode = assemble_fem_1d(
                domain,
                volume_terms,
                boundary_terms,
                dirichlet_values,
                ic_residuals,
                vec=vec,
                # a P2 mass term is degree 4, so the rule must be raised with the order (mirrors the
                # 2D/3D path); too few Gauss points would under-integrate the mass silently
                quad_degree=max(quad_degree, 2 * _order_1d),
                order=_order_1d,
            )
        # a second-order 1D block carries the augmented state y=[u; v] (size 2N) -> field offsets [0, N, 2N]
        offs_1d = None
        if _second_order and mode == "transient":
            _nh = int(np.asarray(op.state0).shape[0]) // 2
            offs_1d = [0, _nh, 2 * _nh]
        return _finalize(FEM(domain=domain, op=op, classification=classification, mode=mode, offsets=offs_1d))

    order = _infer_order(constraints)
    quad_degree = max(quad_degree, 2 * order)

    # ---- coupled / mixed multi-field -> block (multi-variable) assembly ----
    if multifield:
        return _finalize(
            _assemble_multifield(
                domain, volume_terms, boundary_terms, dirichlet_raw, ic_residuals, classification, quad_degree=quad_degree
            )
        )

    # ---- single field: element defaults to the field order (P1->TRI3/TET4,
    # P2->TRI6/TET10); a higher-order field bumps the quadrature for exactness. ----
    if element_type is None:
        element_type = _element_for(domain.dimension, order)
    quad_degree = max(quad_degree, 2 * order)  # factory uses 2*degree+1; bump to the field's order

    # ---- build IR with explicit regions, then detect transient vs steady. This is done so the
    # native path can route from the IR alone: split each weak constraint into additive sub-terms
    # (one LoweredChannelTerm each), matching lower_weak_form's granularity --
    # required so the transient route can separate the mass term (u_t * phi) from the spatial operator. ----
    terms: List[LoweredChannelTerm] = []

    def _emit_terms(bare: Any, support: str, region_id: str) -> None:
        for sign, sub in _split_additive_terms(domain, bare):
            terms.append(
                LoweredChannelTerm(
                    sign=sign,
                    support=support,
                    region_id=region_id,
                    channel="raw",
                    coeff=_apply_sign(domain, sign, sub),
                    variable_id=0,
                    value_shape=(),
                    original_expr=sub,
                )
            )

    for bare in volume_terms:
        _emit_terms(bare, "volume", "volume")
    for tag, exprs in boundary_terms.items():
        for bare in exprs:
            _emit_terms(bare, "boundary", tag)

    if not terms:
        raise ValueError("jno.fem: no weak-form (test-function) terms found to assemble.")

    ir = LoweredWeakForm(domain=domain, terms=terms)
    weak_bares = volume_terms + [e for exprs in boundary_terms.values() for e in exprs]
    is_transient = any(_contains_temporal_derivative(b) for b in weak_bares)

    # Evolution + real-transient (`u.t`) or complex is the last structural rejection (needs the IR): the
    # load path is the *pseudo-time* march (`domain(tau=...)`, no `u.t`), and the return map is real.
    if _evolution and (is_transient or _is_complex_form(domain, ir)):
        raise NotImplementedError(
            "jno.fem: `state.evolves(...)` cannot combine with a real time derivative (`u.t`) or a complex "
            "form — the load path is the *pseudo-time* march over `domain(tau=...)`, not a `u.t` transient, "
            "and the constitutive update is real. Drop `u.t`/complex, or drive time through the `tau` grid."
        )

    # ---- native Lagrange (single field): the standard fast path. Covers 2D triangle and 3D tet (incl.
    # Neumann/Robin surfaces), steady (incl. runtime-scalar-parametric) and transient (constant
    # Dirichlet + a time-dependent source). complex / VPINN / field-param / time-varying-Dirichlet /
    # transient-parametric / vector-or-parametric-periodic fall through to the specialized branches
    # below. ----
    from .utils.solver.parametric_helpers import _contains_runtime_parameter as _crp

    # Native periodic is wired for the steady, scalar single-field case that ``_finalize`` reduces --
    # both non-parametric (reduced eagerly at solve) and runtime-parametric (reduced per-call inside
    # FemLinearSystem.solve, after A(θ) is re-formed). Vector and the transient route pre-build the
    # reduction in their own branches, so they fall through here.
    _native_periodic_ok = not periodic_ties or (not is_transient and (vec or 1) == 1)
    if (
        not is_vpinn
        and _native_lagrange_ok(domain, constraints, weak_bares, periodic_ties)
        and not _is_complex_form(domain, ir)
        and _native_periodic_ok
    ):
        _native_now = True
        if is_transient and any(_is_temporal_value_node(vnode) for *_rest, vnode in dirichlet_raw):
            # native transient covers a runtime SCALAR parameter and a single-field nodal FIELD
            # parameter k(x). A time-varying Dirichlet g(x,t) routes native only for the LINEAR,
            # non-parametric transient (the row-replacement + per-step Dirichlet-lift forcing path);
            # combined with a runtime parameter or a nonlinear residual it is rejected below.
            from .utils.solver.parametric_helpers import _collect_neural_coefficient_exprs as _cnce
            from .utils.solver.weak_form import _is_obviously_nonlinear_in_unknown as _nlin

            _native_now = (
                not any(_crp(b) for b in weak_bares)
                and not any(_nlin(domain, b) for b in weak_bares)
                # tv-Dirichlet g(x,t) + a TRAINABLE net: rejected below (frozen nets stay native)
                and not any(_cnce(b) for b in weak_bares)
            )
        if _native_now:
            from .utils.solver.fem_native import assemble_fem_native

            domain._fem_problem = None  # native owns this domain's FE state
            op, mode, offs = assemble_fem_native(
                domain,
                volume_terms,
                boundary_terms,
                dirichlet_raw,
                ic_residuals,
                vec=vec or 1,
                quad_degree=quad_degree,
                evolution=_evolution,
            )
            return _finalize(FEM(domain=domain, op=op, classification=classification, mode=mode, offsets=offs))

    # ---- native complex (steady, single field): the Re/Im-coefficient split, assembled natively.
    # ``Re(c·T) = Re(c)·T`` for a real FE trial/test ``T``, so wrapping each term in ``.real`` /
    # ``.imag`` gives two ordinary real systems A_r/b_r and A_i/b_i; FEM.solve() forms
    # ``[[A_r, -A_i], [A_i, A_r]]`` and recombines to a complex ``u``. A complex *inverse* (runtime
    # parameter) and the complex *transient* (Schrodinger) path are handled separately below. ----
    if has_neural_coeff and _is_complex_form(domain, ir):
        # A real coordinate-input net in a steady complex form is fine — each Re/Im leg assembles as
        # a parametric FemLinearSystem and _solve_complex_block builds the differentiable trace node,
        # exactly like the scalar complex inverse. The two compositions that would mis-assemble stay
        # rejected: a trainable network's WEIGHTS are not routed into the complex-transient legs (a scalar
        # runtime parameter is), and a solution-dependent net makes the legs nonlinear (no complex Newton).
        if is_transient:
            raise NotImplementedError(
                "jno.fem: a neural coefficient in a complex *transient* form is not supported yet — a runtime "
                "PARAMETER now threads through the complex-transient path (parametric Re/Im legs + a "
                "differentiable trace node), but a trainable NETWORK's weights are not yet routed into those "
                "legs. Use a scalar/field jno.np.parameter coefficient, or a real transient form."
            )
        from .utils.solver.weak_form import _is_obviously_nonlinear_in_unknown as _nlin_cx

        if any(_nlin_cx(domain, b) for b in weak_bares):
            raise NotImplementedError(
                "jno.fem: a solution-dependent neural coefficient (net(u)/net(∇u)) in a complex form "
                "is not supported yet — complex forms assemble as linear real-equivalent blocks."
            )

    if (
        not is_vpinn
        and _is_complex_form(domain, ir)
        and not is_transient
        and _native_lagrange_ok(domain, constraints, weak_bares, periodic_ties)
    ):
        # A runtime parameter is allowed here (the complex *inverse*): assemble_fem_native then returns
        # parametric FemLinearSystem legs and _solve_complex_block builds a differentiable trace node.
        from .utils.solver.fem_native import assemble_fem_native

        real_bd = {tag: [e.real for e in exprs] for tag, exprs in boundary_terms.items()}
        imag_bd = {tag: [e.imag for e in exprs] for tag, exprs in boundary_terms.items()}
        domain._fem_problem = None
        op_r, _mode_r, offs = assemble_fem_native(
            domain, [b.real for b in volume_terms], real_bd, dirichlet_raw, [], vec=vec or 1, quad_degree=quad_degree
        )
        op_i, _mode_i, _offs_i = assemble_fem_native(
            domain, [b.imag for b in volume_terms], imag_bd, dirichlet_raw, [], vec=vec or 1, quad_degree=quad_degree
        )
        return _finalize(FEM(domain=domain, op=(op_r, op_i), classification=classification, mode="complex", offsets=offs))

    # ---- native complex transient (e.g. Schrodinger i u_t = H u): the real-equivalent block, assembled
    # natively. Split the volume terms into the mass (u_t) terms and the spatial terms FIRST -- the mass
    # is real (a real density), so a single steady assembly of the stripped mass gives M (and M_i = 0);
    # the spatial operator's Re/Im parts give A_r, A_i (with the Dirichlet rows handled). The complex IC
    # splits Re/Im across the two blocks. (A complex *mass* coefficient is rejected below.) ----
    if (
        not is_vpinn
        and is_transient
        and _is_complex_form(domain, ir)
        and not multifield
        and _native_lagrange_ok(domain, constraints, weak_bares, periodic_ties)
        and not any(_is_temporal_value_node(vnode) for *_rest, vnode in dirichlet_raw)
    ):
        from .trace import FemLinearSystem as _FLS
        from .utils.solver.backend_blocks import SemidiscreteTimeBlock as _FTB
        from .utils.solver.fem_native import assemble_fem_native
        from .utils.solver.time_route import _infer_time_window, _strip_temporal_trial_derivative
        from .utils.solver.weak_form import _apply_sign, _split_additive_terms

        if getattr(domain, "_trainable_coords", None):
            # A runtime PARAMETER now threads through this path (parametric Re/Im legs below), but a trainable
            # mesh COORDINATE additionally needs the relocation driver, which is real-only. Fail loud.
            raise NotImplementedError(
                "jno.fem: a complex-transient problem cannot carry a trainable mesh coordinate "
                "(Variable.trainable()) yet — the relocation driver is real-only. Relocate a complex *steady* "
                "problem (the real 2N linear path) or a real transient problem instead. (A non-coordinate "
                "runtime parameter in a complex-transient form IS now supported.)"
            )

        mass_stripped, spatial_raw = [], []
        for bare in volume_terms:
            for sign, sub in _split_additive_terms(domain, bare):
                coeff = _apply_sign(domain, sign, sub)
                if _contains_temporal_derivative(sub):
                    mass_stripped.append(_strip_temporal_trial_derivative(coeff))
                else:
                    spatial_raw.append(coeff)
        real_bd = {tag: [e.real for e in exprs] for tag, exprs in boundary_terms.items()}
        imag_bd = {tag: [e.imag for e in exprs] for tag, exprs in boundary_terms.items()}
        domain._fem_problem = None
        # Mass (real): a steady assembly of the stripped u_t * phi term -> M (raw, no Dirichlet).
        (M_raw, _bm), _mm, offs = assemble_fem_native(
            domain, mass_stripped, {}, [], [], vec=vec or 1, quad_degree=quad_degree
        )
        # Spatial Re/Im parts, with the Dirichlet conditions applied to each block. A runtime parameter makes
        # ``assemble_fem_native`` return a parametric ``FemLinearSystem`` leg (the complex-transient INVERSE)
        # instead of a raw ``(A, b)``; the block then carries operator_fn/forcing that re-form ``A(θ)``/``b(θ)``
        # at args, and the fused block wraps a differentiable trace node (mirrors the complex linear
        # inverse). A non-parametric leg keeps the static ``A``/``affine_bias`` block (byte-identical).
        _leg_r, _mr, _o1 = assemble_fem_native(
            domain, [s.real for s in spatial_raw], real_bd, dirichlet_raw, [], vec=vec or 1, quad_degree=quad_degree
        )
        _leg_i, _mi, _o2 = assemble_fem_native(
            domain, [s.imag for s in spatial_raw], imag_bd, dirichlet_raw, [], vec=vec or 1, quad_degree=quad_degree
        )
        from jax.experimental import sparse as _jsp

        from .utils.solver.fem_utils import bcoo_zero_rows as _bcoo_zero_rows

        M = M_raw  # keep the mass a BCOO; the marcher composes the real-equivalent block sparsely (no densify)
        n = int(M.shape[0])
        d_pairs = list(getattr(domain, "_fem_native_dirichlet_pairs", []) or [])
        d_dofs = jnp.asarray([p[0] for p in d_pairs], dtype=jnp.int32) if d_pairs else jnp.zeros((0,), jnp.int32)
        # a constrained dof carries no time derivative -> zero its mass row (BCOO-sparse, mirrors the steady path)
        M_bc = _bcoo_zero_rows(M, d_dofs) if hasattr(M, "indices") else M.at[d_dofs, :].set(0.0)
        pts = jnp.asarray(domain._fem_native_dof_points)
        s0 = (
            jnp.asarray(_ic_value_at_nodes(_bare(ic_residuals[0]), domain, pts, n, vec or 1))
            if ic_residuals
            else jnp.zeros((n,), jnp.complex128)
        )
        t0, t1, dt = _infer_time_window(domain)
        _common = dict(
            backend="transient",
            mode="implicit",
            time_order=1,
            spatial_kind="weak_form",
            t0=t0,
            t1=t1,
            dt=dt,
            eval_context={},
        )

        # ---- FUSE the Re/Im legs into ONE real 2n block, at ASSEMBLY time. This is the same move
        # ``_assemble_second_order_time`` makes for ``u_tt``: the real-equivalent block
        # ``[[A_r, -A_i], [A_i, A_r]]`` over the state ``y = [u_r; u_i]`` is an ordinary
        # :class:`SemidiscreteTimeBlock`, so the solver slots, the ``jno.solve`` time schemes and the
        # transient drivers all apply to a complex transient unchanged — there is no second marcher to
        # re-thread each of them into. Recombination ``u = y[:n] + i·y[n:]`` happens once, on the
        # finished trajectory, keyed off ``metadata["complex"]`` (after any periodic prolongation —
        # ``P`` is real and linear, so prolong-then-split equals split-then-prolong). ----
        def _leg_parts(leg):
            """``(A(args), c(args))`` for one real leg — the static matrices, or re-formed at runtime args."""
            if isinstance(leg, _FLS):  # parametric leg: operator/load re-form at args (the inverse)
                return (
                    lambda args, _L=leg: _to_bcoo(_L.evaluate(args)[0]),  # BCOO operator (not densified)
                    lambda args, _L=leg: jnp.asarray(_L.evaluate(args)[1]).reshape(-1),
                )
            _A, _c = _to_bcoo(leg[0]), jnp.asarray(leg[1]).reshape(-1)  # raw (A, b): static
            return (lambda args, _A=_A: _A), (lambda args, _c=_c: _c)

        A_r_fn, c_r_fn = _leg_parts(_leg_r)
        A_i_fn, c_i_fn = _leg_parts(_leg_i)
        rpe: dict = {}
        for _leg in (_leg_r, _leg_i):
            if isinstance(_leg, _FLS):
                rpe.update(getattr(_leg, "runtime_parameter_exprs", None) or {})

        # the imaginary mass M_i is 0 (a real density): an empty BCOO, so the block mass stays sparse
        _zero_Mi = _jsp.BCOO((jnp.zeros((0,), M.dtype), jnp.zeros((0, 2), jnp.int32)), shape=(n, n))
        # ``krylov="gmres"``: the real-equivalent block is genuinely NON-SYMMETRIC, and BiCGStab (the
        # stepper's default, right for the symmetric real blocks) can break down on it. The bespoke
        # marcher this fusion replaces chose GMRES for exactly that reason; carry the choice over rather
        # than silently regressing robustness on the indefinite / mass-dominated systems it was picked for.
        meta = {"complex": True, "krylov": "gmres"}
        y0 = jnp.concatenate([jnp.real(s0), jnp.imag(s0)])
        if rpe:  # parametric: the 2n operator/load are re-formed from the runtime args each step
            block = _FTB(
                M=_complex_block_bcoo(M_bc, _zero_Mi, n),
                operator_fn=lambda t, args: _complex_block_bcoo(A_r_fn(args), A_i_fn(args), n),
                forcing_vector_fn=lambda t, args: jnp.concatenate([c_r_fn(args), c_i_fn(args)]),
                runtime_parameter_exprs=rpe,
                state0=y0,
                metadata=meta,
                **_common,
            )
        else:  # non-parametric: one static 2n block, composed sparsely (never densified)
            block = _FTB(
                M=_complex_block_bcoo(M_bc, _zero_Mi, n),
                A=_complex_block_bcoo(A_r_fn(None), A_i_fn(None), n),
                affine_bias=jnp.concatenate([c_r_fn(None), c_i_fn(None)]),
                state0=y0,
                metadata=meta,
                **_common,
            )
        return _finalize(FEM(domain=domain, op=block, classification=classification, mode="transient"))

    # ---- native periodic transient (scalar single-field, linear, incl. runtime-parametric): assemble
    # the full native transient block, build the prolongation P from the native assembly mesh, then
    # reduce the block (P^T M P, P^T·operator_fn·P, ...). The reduced block carries P, so its trajectory
    # prolongs back with u = P u_red. This is the optimized scalar single-field fast path; vector and
    # coupled multi-field fall through to the general assembly + `_finalize` block-wise reduction. ----
    if (
        not is_vpinn
        and periodic_ties
        and is_transient
        and not multifield
        and (vec or 1) == 1
        and _native_lagrange_ok(domain, constraints, weak_bares, periodic_ties)
        and not _is_complex_form(domain, ir)
        and not any(_is_temporal_value_node(vnode) for *_rest, vnode in dirichlet_raw)
    ):
        from .utils.solver.fem_native import assemble_fem_native

        domain._fem_problem = None
        op, mode, offs = assemble_fem_native(
            domain, volume_terms, boundary_terms, dirichlet_raw, ic_residuals, vec=1, quad_degree=quad_degree
        )
        if mode != "transient":
            # fail loud rather than silently mis-reduce a steady/nonlinear op as a time block
            raise NotImplementedError(
                f"jno.fem: native periodic transient expected a transient block but assembled mode={mode!r}."
            )
        periodic = _build_periodic_reduction(
            domain,
            periodic_ties,
            domain._fem_native_dof_points,
            domain._fem_native_assembly_cells,
            int(getattr(domain, "_fem_native_assembly_order", 1)),
            1,
        )
        reduced = _reduce_transient_block_periodic(op, periodic)
        fem_obj = FEM(domain=domain, op=reduced, classification=classification, mode="transient", offsets=offs)
        # The time block is already reduced and carries P (transient solve() uses the block directly);
        # expose the reduction on the FEM too, mirroring the steady periodic path (P / n_red / n_full).
        fem_obj._periodic = periodic
        return fem_obj

    # ---- VPINN (network-trial) on a 2D Lagrange mesh: the only single-field path that reaches here.
    # It builds the native fem_context (init_fem_native) and test-projects the weak form. Every
    # standard single-field FEM problem has already returned natively above; anything else is rejected
    # explicitly below (fail loud -- never silently mis-assemble). ----
    if is_vpinn and getattr(domain, "dimension", None) == 2 and not periodic_ties:
        bcs = [domain.dirichlet(tag, value) for tag, value in dirichlet_values.items()]
        if boundary_terms:
            bcs.append(neumann(list(boundary_terms.keys())))
        domain._fem_problem = None
        domain.init_fem_native(element_type=element_type, quad_degree=quad_degree, bcs=bcs, vec=vec or 1)

    # ---- VPINN: test-project the (now fem_gauss-tagged) weak form onto the FE test space ----
    # The network trial already sits inside the weak terms; assemble_weak_form returns a
    # GroupedAssembly whose .mse is the trainable test-projected residual (for jno.core).
    # The Dirichlet condition (u(boundary) - g) declares which test functions vanish on the
    # boundary -> its nodes are masked from the residual (else the exact solution's du/dn flux is
    # an irreducible loss term and training diverges from the solution).
    if is_vpinn:
        from .utils.solver.weak_form import assemble_weak_form

        if not volume_terms:
            raise ValueError("jno.fem VPINN: no volume weak term (expected the test-projected PDE residual).")
        # The weak-term coords were retagged to the quadrature tags (fem_gauss / gauss_<region>);
        # trigger those variables' sampling so the test-projected residual resolves them when it is
        # re-evaluated each training step (crux.solve), not only during this assembly pass.
        domain.variable("fem_gauss")
        for region in boundary_terms:
            domain.variable(f"gauss_{region}")
        weak = volume_terms[0]
        for t in volume_terms[1:]:
            weak = weak + t
        for region_terms in boundary_terms.values():
            for t in region_terms:
                weak = weak + t
        return assemble_weak_form(domain, weak)

    if ic_residuals and not is_transient:
        raise ValueError("jno.fem: an initial condition was given but the weak form has no time derivative.")

    # A single-field weak form that matched none of the native branches above. The native assembler
    # covers every standard single-field problem (2D/3D Lagrange, steady/transient, linear/nonlinear,
    # complex, periodic, runtime/field parameters), so reaching here means an unsupported *combination*.
    # Reject it explicitly with the specific reason -- never silently mis-assemble.
    from .utils.solver.weak_form import _is_obviously_nonlinear_in_unknown as _nlin

    _parametric = any(_crp(b) for b in weak_bares)
    _nonlinear = any(_nlin(domain, b) for b in weak_bares)
    if periodic_ties:
        raise NotImplementedError(
            "jno.fem: a periodic tie is supported natively on a steady or transient SCALAR single field "
            f"(linear, with optional runtime parameters for the transient case). This form has "
            f"vec={vec or 1}, nonlinear={_nonlinear}, parametric+steady={_parametric and not is_transient}. "
            "Write the periodic field as a scalar, linearize it, or drop the periodic tie."
        )
    if is_transient and any(_is_temporal_value_node(vnode) for *_rest, vnode in dirichlet_raw):
        raise NotImplementedError(
            "jno.fem: a time-varying Dirichlet g(x, t) on a transient form is supported natively only for a "
            f"LINEAR, non-parametric problem (got nonlinear={_nonlinear}, parametric={_parametric}). "
            "Linearize the form, or remove the runtime parameter."
        )
    raise NotImplementedError(
        "jno.fem: this single-field weak form is not handled by the native assembler. Please report the "
        "form (and its dimension / element / boundary conditions) so the case can be supported."
    )


def _assemble_multifield(domain, volume_terms, boundary_terms, dirichlet_raw, ic_residuals, classification, *, quad_degree):
    """Assemble a coupled (multi-field) steady weak form into a block ``FEM``.

    Builds the same IR as the single-field path, buckets Dirichlet per field
    (field ordering taken from ``ir.volume_expr``), and hands off to the native
    assembler, which groups the per-tag surface terms into per-field surface
    kernels and differentiates the coupled block matrix."""
    from .utils.solver.fem_utils import _infer_fields
    from .utils.solver.parametric_helpers import _contains_runtime_parameter
    from .utils.solver.weak_form import (
        LoweredChannelTerm,
        LoweredWeakForm,
        _apply_sign,
        _contains_temporal_derivative,
        _is_obviously_nonlinear_in_unknown,
        _split_additive_terms,
    )

    weak_bares = volume_terms + [e for exprs in boundary_terms.values() for e in exprs]
    is_transient = bool(ic_residuals) or any(_contains_temporal_derivative(b) for b in weak_bares)

    domain._fem_quad_degree = quad_degree
    domain._variational_initialized = True

    # Same IR as the single-field path: one LoweredChannelTerm per additive sub-term.
    # Coupled surface (Neumann/Robin) terms are emitted on their boundary region_id;
    # the native assembler groups them per tag into per-field surface kernels.
    # (Coords were retagged to gauss_<region> by the classifier's _retag_coords_for_quadrature.)
    terms: List[Any] = []

    def _emit(bare: Any, support: str, region_id: str) -> None:
        for sign, sub in _split_additive_terms(domain, bare):
            terms.append(
                LoweredChannelTerm(
                    sign=sign,
                    support=support,
                    region_id=region_id,
                    channel="raw",
                    coeff=_apply_sign(domain, sign, sub),
                    variable_id=0,
                    value_shape=(),
                    original_expr=sub,
                )
            )

    for bare in volume_terms:
        _emit(bare, "volume", "volume")
    for tag, exprs in boundary_terms.items():
        for bare in exprs:
            _emit(bare, "boundary", tag)
    if not terms:
        raise ValueError("jno.fem: no weak-form (test-function) terms found to assemble.")
    ir = LoweredWeakForm(domain=domain, terms=terms)

    # Field ordering is taken from ir.volume_expr — the same source the native assembler
    # uses — so the Dirichlet field indices match the kernel's field order.
    fields, field_index = _infer_fields(ir.volume_expr)
    by_field: dict[int, dict[str, Any]] = {}
    dirichlet_tv: List[Any] = []  # (field_idx, region, comp, value_node) for time-varying g(x,t)
    for field_key, region, comp, value, value_node in dirichlet_raw:
        fidx = field_index.get(field_key)
        if fidx is None:
            continue
        region_values = by_field.setdefault(fidx, {})
        if comp is None:
            region_values[region] = value
        else:
            current = region_values.get(region)
            current = dict(current) if isinstance(current, dict) else {}
            current[_COMPONENT_NAMES[comp]] = value
            region_values[region] = current
        if _is_temporal_value_node(value_node):
            dirichlet_tv.append((fidx, region, comp, value_node))
    domain._fem_dirichlet_by_field = by_field

    # Native 2D Lagrange coverage gate (expressed over the inferred fields -- no `constraints` here):
    # 2D, all-Lagrange, non-complex, non-runtime-parametric. Periodic + coupled is rejected by
    # `_finalize` regardless, so it needs no separate guard here.
    # complex=True (re/im ComplexPair) lowers `weak.real` onto two coupled real fields; its terms can
    # weld both test functions in a product, which the native classifier distributes per test field,
    # so the coupled real form assembles directly here.
    _native_ok = (
        getattr(domain, "dimension", None) in (2, 3)
        and all(str(f.get("space", "Lagrange")) == "Lagrange" for f in fields)
        and not _is_complex_form(domain, ir)
        and not any(_contains_runtime_parameter(b) for b in weak_bares)
    )

    # Coupled transient (multi-field + time): block M + block spatial operator A. Native handles
    # constant (incl. non-homogeneous) Dirichlet + a time-dependent source, and a time-varying Dirichlet
    # g(x,t) for the LINEAR block (row-replacement + per-step Dirichlet-lift forcing); a nonlinear block
    # with a time-varying Dirichlet is rejected below (the native branch carries only constant Dirichlet).
    if is_transient:
        _tv_native = not dirichlet_tv or not any(_is_obviously_nonlinear_in_unknown(domain, b) for b in weak_bares)
        # The native coupled-transient assembler threads runtime SCALAR parameters through ``args``
        # (``fem_native._runtime_vals`` packs each parameter per cell, re-evaluated every step), so a
        # *parametric* coupled transient -- e.g. trainable rate constants recovered through the
        # differentiable solve in an inverse problem -- assembles natively too. It needs the same
        # 2D/3D-Lagrange-real gate as the non-parametric case, just WITHOUT the runtime-parameter
        # exclusion baked into ``_native_ok``. (A nodal FIELD parameter ``k(x)`` in a multi-field form
        # is still rejected -- by ``assemble_fem_native`` itself, with a clear single-field-only error.)
        _native_transient_ok = (
            getattr(domain, "dimension", None) in (2, 3)
            and all(str(f.get("space", "Lagrange")) == "Lagrange" for f in fields)
            and not _is_complex_form(domain, ir)
        )
        if _native_transient_ok and _tv_native:
            from .utils.solver.fem_native import assemble_fem_native

            domain._fem_problem = None
            op, mode, offs = assemble_fem_native(
                domain, volume_terms, boundary_terms, dirichlet_raw, ic_residuals, vec=1, quad_degree=quad_degree
            )
            return FEM(domain=domain, op=op, classification=classification, mode=mode, offsets=offs)
        # Coupled transient that the native block does not cover (a complex coefficient, or a time-varying
        # Dirichlet on a nonlinear block) -- reject explicitly rather than mis-assemble.
        raise NotImplementedError(
            "jno.fem: this coupled (multi-field) transient is not supported natively. The native coupled "
            "transient covers constant / time-dependent / runtime-parametric coefficients (incl. nonlinear) "
            "and a time-varying Dirichlet on a LINEAR block "
            f"(got nonlinear={any(_is_obviously_nonlinear_in_unknown(domain, b) for b in weak_bares)}, "
            f"complex={_is_complex_form(domain, ir)}, "
            f"time_varying_dirichlet={bool(dirichlet_tv)})."
        )

    if _native_ok:
        from .utils.solver.fem_native import assemble_fem_native

        domain._fem_problem = None  # the native assembler owns this domain's FE state
        op, mode, offs = assemble_fem_native(
            domain, volume_terms, boundary_terms, dirichlet_raw, ic_residuals, vec=1, quad_degree=quad_degree
        )
        return FEM(domain=domain, op=op, classification=classification, mode=mode, offsets=offs)

    # A coupled steady form `_native_ok` excluded -- a complex coefficient (vs the real-equivalent
    # complex=True), or a runtime parameter. The native coupled assembler covers linear and nonlinear
    # real forms (incl. complex=True); reject the rest explicitly rather than mis-assemble.
    raise NotImplementedError(
        "jno.fem: this coupled (multi-field) steady form is not supported natively -- it has a complex "
        f"coefficient ({_is_complex_form(domain, ir)}) or a runtime parameter "
        f"({any(_contains_runtime_parameter(b) for b in weak_bares)}). Use complex=True for a complex "
        "field, or recover the parameter on a single-field reduced form."
    )


def _sum_forcings(f1, f2):
    """Sum two optional forcing callables ``f(t, args) -> vector`` (None acts as 0)."""
    if f1 is None:
        return f2
    if f2 is None:
        return f1

    def summed(t, args=None):
        return jnp.asarray(f1(t, args)).reshape(-1) + jnp.asarray(f2(t, args)).reshape(-1)

    return summed


def _ic_value_at_nodes(bare: Any, domain: Any, pts: Any, n: int, vec: int = 1) -> Any:
    """Nodal value vector (``n`` dofs, node-major interleaved for ``vec>1``) from an IC residual
    ``<trial-expr>(initial) - g``.

    Tolerant of a temporal-derivative trial side, so it reads both the displacement IC
    (``u(initial) - u0``) and the velocity IC (``u.t(initial) - v0``) — the latter's trial side is
    ``Jacobian(u, [t])``, which :func:`_essential_spec` rejects. ``g`` is
    evaluated at the **assembly** nodes ``pts`` (P2 carries edge nodes, so they differ from the
    linear mesh); a vector ``g`` may be a constant per-component tuple (broadcast to every node), a
    full ``(n_nodes·vec,)`` field, or a single scalar (broadcast to all components)."""
    trial_side = value_node = None
    if getattr(bare, "op", None) == "-" and hasattr(bare, "left") and hasattr(bare, "right"):
        left, right = bare.left, bare.right
        if _contains(left, TrialFunction) and not _contains(right, TrialFunction):
            trial_side, value_node = left, right
        elif _contains(right, TrialFunction) and not _contains(left, TrialFunction):
            trial_side, value_node = right, left
    if value_node is None:
        return jnp.zeros((n,))
    from .utils.solver.parametric_helpers import _is_neural_coefficient

    if _is_neural_coefficient(_bare(value_node)):
        # A net-valued IC threads its weights only on the real, first-order-in-time transient path
        # (``state0_fn`` re-forms the initial state from ``args``). This routine bakes the IC at assembly
        # (complex-transient / second-order-in-time), so the weights would silently NOT train — reject it.
        raise NotImplementedError(
            "jno.fem: a net-valued initial condition u(initial) - net(x) is wired on a real, "
            "first-order-in-time transient form (its weights thread the initial state); this path "
            "(complex-transient or second-order-in-time) bakes the IC at assembly and cannot thread the "
            "weights. Use a real first-order transient form, or a constant/field initial condition here."
        )
    comp = _component_index_of(trial_side)  # one component (u(...)[i] - g) vs all (u(...) - g)
    n_nodes = int(jnp.asarray(pts).shape[0])
    const = _constant_of(value_node)
    if const is not None:
        if comp is None:
            return jnp.full((n,), float(const))  # scalar broadcast to every dof
        return jnp.zeros((n,)).at[comp::vec].set(float(const))  # one component only
    vals = jnp.asarray(_eval_value_node_at(value_node, jnp.asarray(pts))).reshape(-1)
    if comp is not None:  # one component: vals is a scalar field (or constant) over the nodes
        per = jnp.broadcast_to(vals, (n_nodes,)) if vals.shape[0] != n_nodes else vals
        return jnp.zeros((n,)).at[comp::vec].set(per)
    if vals.shape[0] == n:  # full (n_nodes·vec,) interleaved field
        return vals
    if vec > 1 and vals.shape[0] == vec:  # constant per-component vector -> broadcast to every node
        return jnp.tile(vals, n_nodes)
    if vec > 1 and vals.shape[0] == n_nodes:  # one scalar field shared by every component
        return jnp.repeat(vals, vec)
    if vals.shape[0] == 1:  # single scalar -> every dof
        return jnp.full((n,), vals[0])
    if vals.shape[0] == n_nodes:  # scalar problem, scalar field
        return vals
    raise NotImplementedError(
        f"jno.fem: could not place an initial condition of size {vals.shape[0]} into {n} dofs "
        f"(vec={vec}); write it as a constant, a per-component value, or a full nodal field."
    )


_SECOND_ORDER_FLOAT32_WARNED = False


def _warn_second_order_float32() -> None:
    """Warn once if a second-order-in-time solve is assembled without ``jax_enable_x64``.

    jNO does not own data precision (see :mod:`jno.utils.dtypes`), so we never force float64. But a
    wave / elastodynamics mass–stiffness pair has *soft* modes — a slender cantilever's fundamental
    bending frequency, say, whose modal stiffness is orders of magnitude below ``‖K‖`` — and float32
    assembly round-off (~1e-7 relative) can shift such a frequency by several percent. The
    trapezoidal rule still conserves energy on the (slightly wrong) block, so the error is *silent*:
    the vibration rings at the wrong speed while every conservation check passes. Enabling x64 makes
    it exact. We warn rather than raise, once per process, to keep the choice with the user.
    """
    global _SECOND_ORDER_FLOAT32_WARNED
    if _SECOND_ORDER_FLOAT32_WARNED or jax.config.jax_enable_x64:
        return
    _SECOND_ORDER_FLOAT32_WARNED = True
    warnings.warn(
        "jno.fem: assembling a second-order-in-time (u_tt) problem with jax_enable_x64 disabled "
        "(float32). Soft-mode frequencies (e.g. slender-beam bending) can be several percent wrong "
        "while energy is still conserved — a silent error. Enable "
        'jax.config.update("jax_enable_x64", True) for accurate wave / elastodynamics frequencies.',
        stacklevel=2,
    )


def _assemble_second_order_time(
    domain,
    volume_terms,
    boundary_terms,
    dirichlet_values,
    dirichlet_raw,
    ic_residuals,
    classification,
    *,
    order,
    vec,
    quad_degree,
):
    r"""Reduce a linear second-order-in-time weak form to a first-order augmented block.

    A second-order semidiscrete system :math:`M_2 \ddot u + C \dot u + K u = F` is rewritten with
    the velocity :math:`v = \dot u` as the first-order block
    :math:`M_\text{aug}\,\dot y + A_\text{aug}\,y = c_\text{aug}` in :math:`y = [u; v]`::

        [M2  0 ] [u']   [ 0   -M2] [u]   [0]
        [0   M2] [v'] + [ K    C ] [v] = [F]

    i.e. :math:`M_2\dot u = M_2 v` (the definition :math:`v=\dot u`) and
    :math:`M_2\dot v + Cv + Ku = F` (the PDE). The block integrates with the **trapezoidal rule**
    (the :math:`\theta=\tfrac12` default, equivalent to Newmark average-acceleration), which
    conserves energy for an undamped wave where backward Euler would spuriously damp it
    (Newmark 1959, "A Method of Computation for Structural Dynamics", §average-acceleration). It is a
    standard :class:`SemidiscreteTimeBlock`, so the differentiable :meth:`FEM.solve` and the flat ``fem.M`` /
    ``fem.state0`` accessors are unchanged; the state is ``y=[u; v]`` (size ``2N``), split via
    ``fem.offsets`` (``[0, N, 2N]``) — displacement ``y[:N]``, velocity ``y[N:]``.

    **Runtime / trainable parameters** (the differentiable inverse through ``u_tt`` — full-waveform
    inversion, elastography, source recovery) are supported: when a coefficient is a
    :func:`jno.np.parameter`, a trainable scalar, or a ``jno.nn.wrap`` field, each block ``M₂/C/K`` is
    assembled as a callable ``op(args)`` and the augmented ``M_aug``/``A_aug``/forcing are re-formed
    per step through ``operator_fn``/``mass_fn``/``forcing_vector_fn``; the θ=½ stepper differentiates
    through its own scan, so the gradient reaches the parameter with no stepper change. ``M₂`` feeds
    both ``M_aug`` and the ``−M₂`` coupling of ``A_aug``, so a parametric *mass* wires ``mass_fn`` as
    well as ``operator_fn``. The constant Dirichlet ``g`` rides ``affine_bias`` and the (possibly
    parametric) load ``F`` rides the forcing.

    Scope: linear, single field — **scalar or vector** (vector = elastodynamics, ``value_shape=(2,)``
    / ``(3,)``) — nodal Lagrange, 2D/3D, constant Dirichlet. Two initial conditions: displacement
    ``u(initial) - u0`` and (optional) velocity ``u.t(initial) - v0`` (default zero). Nonlinear,
    multi-field, time-varying-Dirichlet ``g(x,t)``, or *trainable-Dirichlet-value* second-order forms
    are rejected (fail-loud) rather than silently mis-assembled.
    """
    from .utils.solver.backend_blocks import SemidiscreteTimeBlock
    from .utils.solver.fem_native import assemble_fem_native
    from .utils.solver.fem_utils import bcoo_set_dirichlet_rows, bcoo_zero_rows
    from .utils.solver.parametric_helpers import _contains_runtime_parameter
    from .utils.solver.solver_helper import max_temporal_derivative_order as _mto
    from .utils.solver.time_route import _infer_time_window, _strip_temporal_trial_derivative
    from .utils.solver.weak_form import (
        _apply_sign,
        _is_obviously_nonlinear_in_unknown,
        _split_additive_terms,
    )

    vec = int(vec)

    # ---- fail-loud guards (a mis-assembled second-order solve is a silently wrong result) ----
    weak_bares = list(volume_terms) + [e for exprs in boundary_terms.values() for e in exprs]
    # A NONLINEAR spatial operator (sine-Gordon, cubic Klein–Gordon, large-deformation elastodynamics)
    # is supported via Newton on the augmented residual (below); only the mass/damping stay linear.
    is_nonlinear = any(_is_obviously_nonlinear_in_unknown(domain, b) for b in weak_bares)
    # Time-varying Dirichlet g(x,t) IS supported (driven boundaries: prescribed oscillation, seismic
    # input, a transducer feed): the displacement rows carry u[d]=g(x_d,t) and the velocity rows carry
    # the compatible v[d]=ġ(x_d,t), both written into the forcing per step (below).
    # Runtime/trainable parameters ARE supported (the differentiable inverse through `u_tt`): the
    # augmented block is re-formed from the runtime args each step (below). Only a parametric Dirichlet
    # *value* is rejected — the held boundary value is a constant snapshot, so a trainable g would
    # silently freeze at its initial value.
    has_param = any(_contains_runtime_parameter(b) for b in weak_bares)
    if has_param and any(_contains_runtime_parameter(vnode) for *_rest, vnode in dirichlet_raw):
        raise NotImplementedError(
            "jno.fem: a runtime/trainable Dirichlet value on a second-order-in-time problem is not "
            "supported (the held boundary value is a constant); keep the Dirichlet value fixed and "
            "recover the parameter through the operator or load instead."
        )

    # Wave / elastodynamics frequencies of soft modes are not resolvable in float32 (silent error).
    _warn_second_order_float32()

    # ---- quadrature setup (the native assembler owns the FE state; no problem object) ----
    quad_degree = max(quad_degree, 2 * order)  # factory uses 2*degree+1; bump to the field's order

    # ---- split each volume term by its temporal order: 2 -> mass M2, 1 -> damping C, 0 -> stiffness/load ----
    def _strip(coeff, times):
        for _ in range(times):
            coeff = _strip_temporal_trial_derivative(coeff)
        return coeff

    mass2_raw, damp_raw, stiff_raw = [], [], []
    for bare in volume_terms:
        for sign, sub in _split_additive_terms(domain, bare):
            coeff = _apply_sign(domain, sign, sub)
            o = _mto(sub)
            if o >= 2:
                mass2_raw.append(_strip(coeff, 2))  # u_tt * phi -> mass bilinear (u, phi)
            elif o == 1:
                damp_raw.append(_strip(coeff, 1))  # u_t * phi -> damping bilinear (u, phi)
            else:
                stiff_raw.append(coeff)  # spatial operator + load

    if not mass2_raw:
        raise ValueError("jno.fem: second-order route found no `u_tt * phi` mass term.")

    # Raw (no-Dirichlet) physics blocks assembled natively; Dirichlet is applied explicitly to the 2N
    # augmented system below (row replacement, columns kept). Each group is returned as a pair of
    # callables ``op(args)`` / ``rhs(args)``: a parameter-free group's callable just returns its
    # constant matrix, while a runtime-parametric group (a wave speed, a density, a ``k(x)``/nn field)
    # re-assembles at the runtime args each call, kept differentiable in args so the gradient reaches
    # the parameter through every step of the augmented march. ``A0``/``b0`` are the static
    # placeholders (parameters at 0 / stored weights) used for the ``.M``/``.A`` accessors and sizing.
    def _native_group(terms, bterms):
        op, _mode, _offs = assemble_fem_native(domain, terms, bterms, [], [], vec=vec, quad_degree=quad_degree)
        if isinstance(op, tuple):  # static (A, b): a parameter-free group
            A0, b0 = _to_bcoo(op[0]), jnp.asarray(op[1]).reshape(-1)  # BCOO — the 2N block is composed sparsely
            return {
                "op": (lambda a=None, _A=A0: _A),
                "rhs": (lambda a=None, _b=b0: _b),
                "A0": A0,
                "b0": b0,
                "rpe": {},
                "is_param": False,
            }
        # parametric FemLinearSystem: operator_fn(args)/rhs_fn(args) re-assemble at the runtime args.
        return {
            "op": (lambda a=None, _o=op: _to_bcoo(_o.operator_fn(a))),
            "rhs": (lambda a=None, _o=op: jnp.asarray(_o.rhs_fn(a)).reshape(-1)),
            "A0": _to_bcoo(op.A),
            "b0": jnp.asarray(op.b).reshape(-1),
            "rpe": dict(getattr(op, "runtime_parameter_exprs", {}) or {}),
            "is_param": True,
        }

    gm = _native_group(mass2_raw, {})  # u_tt -> M2 (mass)
    n = int(gm["A0"].shape[0])
    dtype = gm["A0"].dtype
    Z = _bcoo_empty(n, n, dtype)  # the zero block / zero damping — sparse, so the 2N block never densifies
    gc = _native_group(damp_raw, {}) if damp_raw else None  # u_t -> C (damping)
    # A nonlinear spatial operator is assembled as a residual/jacobian below, not a linear K matrix.
    gk = None if is_nonlinear else _native_group(stiff_raw, boundary_terms)  # spatial operator K + load F

    # Dirichlet from the native assembler (it stashes them): constant (dof, value) pairs → the held
    # value rides the affine bias; time-varying entries (dofs, g(x,t) node, coords) → the displacement
    # rows carry u[d]=g(x_d,t) and the velocity rows the compatible v[d]=ġ(x_d,t), written per step.
    assemble_fem_native(domain, stiff_raw, boundary_terms, dirichlet_raw, [], vec=vec, quad_degree=quad_degree)
    pairs = list(getattr(domain, "_fem_native_dirichlet_pairs", []) or [])
    rows = jnp.asarray([p[0] for p in pairs], dtype=int) if pairs else jnp.zeros((0,), dtype=int)
    g = jnp.asarray([p[1] for p in pairs], dtype=dtype) if pairs else jnp.zeros((0,), dtype=dtype)
    tv = list(getattr(domain, "_fem_native_dirichlet_tv", []) or [])  # driven boundaries g(x,t)
    has_tv = bool(tv)
    rows_tv = jnp.concatenate([jnp.asarray(e[0], dtype=int) for e in tv]) if tv else jnp.zeros((0,), dtype=int)
    rows_all = jnp.concatenate([rows, rows_tv])  # every Dirichlet displacement DOF (constant + driven)
    nrows = int(rows_all.shape[0])
    t0, t1, dt = _infer_time_window(domain)

    def _gval(vnode, coords, t):  # g(x_d, t) at the boundary DOF coordinates
        return jnp.asarray(_eval_value_node_at_time(vnode, coords, t)).reshape(-1)

    def _gdot(vnode, coords, t):  # ġ(x_d, t) = ∂g/∂t — the velocity compatible with a moving boundary
        _, gd = jax.jvp(lambda tt: _eval_value_node_at_time(vnode, coords, tt), (t,), (jnp.ones_like(t),))
        return jnp.asarray(gd).reshape(-1)

    def _tv_forcing(t):  # driven-boundary rows: g(t) on displacement, ġ(t) on velocity
        f = jnp.zeros((2 * n,), dtype)
        for dofs, vnode, coords in tv:
            dd = jnp.asarray(dofs, dtype=int)
            f = f.at[dd].set(_gval(vnode, coords, t)).at[dd + n].set(_gdot(vnode, coords, t))
        return f

    # ---- initial state y0 = [u0; v0] from the ICs, made Dirichlet-consistent (u[d]=g, v[d]=0 or ġ(t0)) ----
    # The IC is sampled at the native assembly DOF coordinates (vertices + P2 edge-midpoints), stashed by
    # the native assembler; ``n = N*vec`` flattens node-major, matching the block layout.
    pts = jnp.asarray(getattr(domain, "_fem_native_dof_points", domain.mesh.points))[:, : domain.dimension]
    u0 = jnp.zeros((n,), dtype)
    v0 = jnp.zeros((n,), dtype)
    for ic in ic_residuals:
        val = jnp.asarray(_ic_value_at_nodes(_bare(ic), domain, pts, n, vec), dtype)
        if _mto(_bare(ic)) >= 1:
            v0 = val
        else:
            u0 = val
    if int(rows.shape[0]):  # constant Dirichlet: u(0)=g, v(0)=0
        u0, v0 = u0.at[rows].set(g), v0.at[rows].set(0.0)
    for dofs, vnode, coords in tv:  # driven boundary: u(0)=g(x,t0), v(0)=ġ(x,t0)
        dd = jnp.asarray(dofs, dtype=int)
        u0, v0 = u0.at[dd].set(_gval(vnode, coords, t0)), v0.at[dd].set(_gdot(vnode, coords, t0))
    state0 = jnp.concatenate([u0, v0])
    domain._fem_problem = None  # native owns this domain's FE state -> FEM.points reads the native DOFs

    # ---- compose the 2N augmented system M_aug y' + A_aug y = c_aug, y = [u; v] ----
    #   [M2  0 ] [u']   [ 0   -M2] [u]   [0]
    #   [0   M2] [v'] + [ K    C ] [v] = [F]
    _aug_d = jnp.concatenate([rows_all, rows_all + n]) if nrows else rows_all  # Dirichlet rows in BOTH blocks

    def _dirichlet_A(A):  # every Dirichlet row (u-block d, v-block d+n) -> identity row (cols kept)
        if not nrows:
            return A
        if hasattr(A, "indices"):  # BCOO: row-replacement (zero rows + unit diagonal), never densify
            return bcoo_set_dirichlet_rows(A, _aug_d)
        A = A.at[rows_all, :].set(0.0).at[rows_all, rows_all].set(1.0)
        return A.at[rows_all + n, :].set(0.0).at[rows_all + n, rows_all + n].set(1.0)

    def _dirichlet_M(M):  # zero every Dirichlet row of both blocks (the constraint rows are algebraic)
        if not nrows:
            return M
        return (
            bcoo_zero_rows(M, _aug_d) if hasattr(M, "indices") else M.at[rows_all, :].set(0.0).at[rows_all + n, :].set(0.0)
        )

    common = dict(
        backend="transient",
        mode="implicit",
        time_order=2,
        spatial_kind="weak_form",
        state0=state0,
        t0=t0,
        t1=t1,
        dt=dt,
        eval_context={},
    )
    if is_nonlinear:
        # nonlinear spatial operator (sine-Gordon, cubic Klein–Gordon, large-deformation elastodynamics):
        # Newton on the augmented residual M_aug ẏ + R_aug(y) = 0 with R_aug(y) = [−M2 v ; N(u,args)+C v],
        # N(u)=S(u)−F the native nonlinear spatial residual. The θ=½ stepper (now θ-aware for nonlinear
        # blocks too) keeps the undamped wave from bleeding energy; args flow through N/J_N for the inverse.
        if has_tv:
            raise NotImplementedError(
                "jno.fem: time-varying Dirichlet g(x,t) on a *nonlinear* second-order-in-time form is not "
                "supported; use a constant Dirichlet value."
            )
        M2, C = gm["A0"], (gc["A0"] if gc else Z)
        sop, _sm, _so = assemble_fem_native(domain, stiff_raw, boundary_terms, [], [], vec=vec, quad_degree=quad_degree)
        M_aug = _dirichlet_M(_bcoo_block([(M2, 0, 0, 1.0), (M2, n, n, 1.0)], (2 * n, 2 * n), dtype))
        rpe = dict(getattr(sop, "runtime_parameter_exprs", {}) or {})

        def _residual_aug(y, t=0.0, args=None):
            y = jnp.asarray(y).reshape(-1)
            u_, v_ = y[:n], y[n:]
            r = jnp.concatenate([-(M2 @ v_), jnp.asarray(sop.residual(u_, args)).reshape(-1) + (C @ v_)])
            if int(rows.shape[0]):  # u[d]=g on displacement rows, v[d]=0 on velocity rows (constant g)
                r = r.at[rows].set(u_[rows] - g).at[rows + n].set(v_[rows])
            return r

        def _jacobian_aug(y, t=0.0, args=None):
            y = jnp.asarray(y).reshape(-1)
            jn = _to_bcoo(sop.jacobian(y[:n], args))  # ∂N/∂u (BCOO — composed into the augmented block sparsely)
            return _dirichlet_A(_bcoo_block([(M2, 0, n, -1.0), (jn, n, 0, 1.0), (C, n, n, 1.0)], (2 * n, 2 * n), dtype))

        meta = {"theta": 0.5, "second_order": True}
        if rpe:
            meta.update(runtime_parameter_names=list(rpe), nonaffine_operator=True)
        block = SemidiscreteTimeBlock(
            mass=lambda t, args=None, _M=M_aug: _M,
            residual=_residual_aug,
            jacobian=_jacobian_aug,
            runtime_parameter_exprs=rpe,
            metadata=meta,
            **common,
        )
    elif not has_param and not has_tv:
        # parameter-free, constant Dirichlet: assemble the augmented block once (the fast, common path)
        M2, C, K, F = gm["A0"], (gc["A0"] if gc else Z), gk["A0"], gk["b0"]
        M_aug = _dirichlet_M(_bcoo_block([(M2, 0, 0, 1.0), (M2, n, n, 1.0)], (2 * n, 2 * n), dtype))
        A_aug = _dirichlet_A(_bcoo_block([(M2, 0, n, -1.0), (K, n, 0, 1.0), (C, n, n, 1.0)], (2 * n, 2 * n), dtype))
        c_aug = jnp.concatenate([jnp.zeros((n,), dtype), F])
        if nrows:
            c_aug = c_aug.at[rows].set(g).at[rows + n].set(0.0)
        block = SemidiscreteTimeBlock(
            M=M_aug, A=A_aug, affine_bias=c_aug, metadata={"theta": 0.5, "second_order": True}, **common
        )
    else:
        # runtime parameters and/or driven boundaries: re-form what varies each step, so the gradient
        # flows through the whole march (the θ=½ stepper reads operator_fn/mass_fn/forcing_vector_fn and
        # differentiates through its own scan — no stepper change). M2 feeds both M_aug and the -M2
        # coupling of A_aug, so a parametric mass wires mass_fn *and* operator_fn. The constant Dirichlet
        # g rides affine_bias; the load F and the driven boundary g(t)/ġ(t) ride the forcing (zeroed on
        # the Dirichlet rows so the load never fights the held value).
        def _A_of(args):
            M2, C, K = gm["op"](args), (gc["op"](args) if gc else Z), gk["op"](args)
            return _dirichlet_A(_bcoo_block([(M2, 0, n, -1.0), (K, n, 0, 1.0), (C, n, n, 1.0)], (2 * n, 2 * n), dtype))

        def _M_of(args):
            M2 = gm["op"](args)
            return _dirichlet_M(_bcoo_block([(M2, 0, 0, 1.0), (M2, n, n, 1.0)], (2 * n, 2 * n), dtype))

        def _forcing(t, args):
            f = jnp.concatenate([jnp.zeros((n,), dtype), gk["rhs"](args)])
            if nrows:
                f = f.at[rows_all].set(0.0).at[rows_all + n].set(0.0)  # load off the Dirichlet rows
            return f + _tv_forcing(t) if has_tv else f  # driven-boundary g(t)/ġ(t) on the tv rows

        M2s, Cs, Ks = gm["A0"], (gc["A0"] if gc else Z), gk["A0"]
        M_aug0 = _dirichlet_M(_bcoo_block([(M2s, 0, 0, 1.0), (M2s, n, n, 1.0)], (2 * n, 2 * n), dtype))
        A_aug0 = _dirichlet_A(_bcoo_block([(M2s, 0, n, -1.0), (Ks, n, 0, 1.0), (Cs, n, n, 1.0)], (2 * n, 2 * n), dtype))
        c_dir = jnp.zeros((2 * n,), dtype).at[rows].set(g) if int(rows.shape[0]) else jnp.zeros((2 * n,), dtype)
        rpe = {**gm["rpe"], **(gc["rpe"] if gc else {}), **gk["rpe"]}
        block = SemidiscreteTimeBlock(
            M=M_aug0,
            A=A_aug0,
            affine_bias=c_dir,
            operator_fn=(lambda t, args=None: _A_of(args)) if has_param else None,
            mass_fn=((lambda t, args=None: _M_of(args)) if (has_param and gm["is_param"]) else None),
            forcing_vector_fn=lambda t, args=None: _forcing(t, args),
            runtime_parameter_exprs=rpe,
            metadata={"theta": 0.5, "second_order": True, "runtime_parameter_names": list(rpe), "nonaffine_operator": True},
            **common,
        )
    return FEM(domain=domain, op=block, classification=classification, mode="transient", offsets=[0, n, 2 * n])


def _assemble_multifield_second_order(
    domain, volume_terms, boundary_terms, dirichlet_raw, ic_residuals, classification, *, quad_degree
):
    r"""Coupled multifield second-order-in-time where **every** field carries ``u_tt`` (undamped,
    value-coupled). With all fields second-order the augmented state ``y=[u_all; v_all]`` reduces to
    the *single-field* formula with the **coupled block** ``M₂``/``K``::

        [M2  0 ] [u_all']   [ 0   -M2] [u_all]   [ 0 ]
        [0   M2] [v_all'] + [ K    0 ] [v_all] = [ F ]

    ``M₂`` is the block-diagonal mass over all fields; ``K`` carries the per-field spatial operators and
    the (value) couplings in its off-diagonal blocks. Two coupled membranes, coupled waves, a system of
    coupled oscillators — the clean case. **A mixed-order coupling** (a first-order field, i.e. a bare
    ``u_t`` term) needs a per-field velocity augmentation and is rejected: write it as a first-order
    system with an explicit velocity field per second-order field.
    """
    from .utils.solver.backend_blocks import SemidiscreteTimeBlock
    from .utils.solver.fem_native import assemble_fem_native
    from .utils.solver.fem_utils import bcoo_set_dirichlet_rows, bcoo_zero_rows
    from .utils.solver.parametric_helpers import _contains_runtime_parameter
    from .utils.solver.solver_helper import max_temporal_derivative_order as _mto
    from .utils.solver.time_route import _infer_time_window, _strip_temporal_trial_derivative
    from .utils.solver.weak_form import _apply_sign, _is_obviously_nonlinear_in_unknown, _split_additive_terms

    weak_bares = list(volume_terms) + [e for exprs in boundary_terms.values() for e in exprs]
    if any(_is_obviously_nonlinear_in_unknown(domain, b) for b in weak_bares):
        raise NotImplementedError("jno.fem: a nonlinear coupled second-order-in-time form is not supported.")
    if any(_contains_runtime_parameter(b) for b in weak_bares):
        raise NotImplementedError(
            "jno.fem: runtime parameters in a coupled second-order-in-time form are not supported yet."
        )
    if any(_is_temporal_value_node(vnode) for *_rest, vnode in dirichlet_raw):
        raise NotImplementedError(
            "jno.fem: time-varying Dirichlet on a coupled second-order-in-time form is not supported yet."
        )
    _warn_second_order_float32()
    quad_degree = max(quad_degree, 2)

    def _strip(coeff, times):
        for _ in range(times):
            coeff = _strip_temporal_trial_derivative(coeff)
        return coeff

    mass2_raw, spatial_raw = [], []
    for bare in volume_terms:
        for sign, sub in _split_additive_terms(domain, bare):
            coeff = _apply_sign(domain, sign, sub)
            o = _mto(sub)
            if o >= 2:
                mass2_raw.append(_strip(coeff, 2))
            elif o == 1:  # a bare u_t → damping / a first-order field: the mixed-order case
                raise NotImplementedError(
                    "jno.fem: a coupled second-order-in-time form with a u_t term (damping or a first-order "
                    "field) is not supported yet — every field must carry u_tt. Write the coupled problem as a "
                    "first-order system with an explicit velocity field per second-order field."
                )
            else:
                spatial_raw.append(coeff)
    if not mass2_raw:
        raise ValueError("jno.fem: coupled second-order route found no `u_tt * phi` mass term.")

    def _mat(terms, bterms):
        op, _mode, offs = assemble_fem_native(domain, terms, bterms, [], [], vec=1, quad_degree=quad_degree)
        A = _to_bcoo(op[0] if isinstance(op, tuple) else op.A)  # BCOO — the 2N block is composed sparsely
        b = jnp.asarray(op[1] if isinstance(op, tuple) else op.b).reshape(-1)
        return A, b, list(offs)

    M2, _mf, moffs = _mat(mass2_raw, {})  # block-diagonal coupled mass over all fields
    K, F, koffs = _mat(spatial_raw, boundary_terms)  # coupled spatial operator + load (couplings off-diagonal)
    if moffs != koffs:
        raise NotImplementedError(
            "jno.fem: coupled second-order-in-time requires every field to be second-order with a consistent "
            "block layout; a mixed-order coupling is not supported yet — write it as a first-order system."
        )
    n = int(M2.shape[0])
    dtype = M2.dtype

    # Dirichlet (dof, value) pairs from the native assembler stash (constant g only)
    assemble_fem_native(domain, spatial_raw, boundary_terms, dirichlet_raw, [], vec=1, quad_degree=quad_degree)
    pairs = list(getattr(domain, "_fem_native_dirichlet_pairs", []) or [])
    rows = jnp.asarray([p[0] for p in pairs], dtype=int) if pairs else jnp.zeros((0,), dtype=int)
    g = jnp.asarray([p[1] for p in pairs], dtype=dtype) if pairs else jnp.zeros((0,), dtype=dtype)
    nrows = int(rows.shape[0])

    # ---- initial state y0 = [u_all; v_all] — each IC placed into its field block, displacement vs velocity ----
    field_keys = list(getattr(domain, "_fem_native_field_keys", []) or [])
    field_index = {k: i for i, k in enumerate(field_keys)}
    pts_by_field = getattr(domain, "_fem_native_dof_points_all", None)
    u0 = jnp.zeros((n,), dtype)
    v0 = jnp.zeros((n,), dtype)
    for ic in ic_residuals:
        bare_ic = _bare(ic)
        fi = field_index.get(_field_key_of(ic))
        if fi is None:
            continue
        lo, hi = moffs[fi], moffs[fi + 1]
        pts_f = jnp.asarray(pts_by_field[fi] if pts_by_field is not None else domain.mesh.points)[:, : domain.dimension]
        val = jnp.asarray(_ic_value_at_nodes(bare_ic, domain, pts_f, hi - lo, 1), dtype)
        if _mto(bare_ic) >= 1:
            v0 = v0.at[lo:hi].set(val)  # velocity IC u̇(0)=v0 for this field
        else:
            u0 = u0.at[lo:hi].set(val)  # displacement IC u(0)=u0 for this field

    # ---- compose the 2N augmented block (single-field formula, coupled M2/K) — SPARSELY (BCOO), so a large
    #      coupled wave / membrane system never materialises the dense (2N × 2N) block ----
    M_aug = _bcoo_block([(M2, 0, 0, 1.0), (M2, n, n, 1.0)], (2 * n, 2 * n), dtype)  # [[M2, 0], [0, M2]]
    A_aug = _bcoo_block([(M2, 0, n, -1.0), (K, n, 0, 1.0)], (2 * n, 2 * n), dtype)  # [[0, -M2], [K, 0]]
    c_aug = jnp.concatenate([jnp.zeros((n,), dtype), F])
    if nrows:  # u[d]=g on displacement rows, v[d]=0 on velocity rows (identity rows, cols kept)
        _md = jnp.concatenate([rows, rows + n])
        M_aug = bcoo_zero_rows(M_aug, _md)
        A_aug = bcoo_set_dirichlet_rows(A_aug, _md)
        c_aug = c_aug.at[rows].set(g).at[rows + n].set(0.0)
        u0, v0 = u0.at[rows].set(g), v0.at[rows].set(0.0)
    domain._fem_problem = None
    t0, t1, dt = _infer_time_window(domain)
    block = SemidiscreteTimeBlock(
        backend="transient",
        mode="implicit",
        time_order=2,
        spatial_kind="weak_form",
        M=M_aug,
        A=A_aug,
        affine_bias=c_aug,
        state0=jnp.concatenate([u0, v0]),
        t0=t0,
        t1=t1,
        dt=dt,
        eval_context={},
        metadata={"theta": 0.5, "second_order": True},
    )
    # Field slicing on the augmented state [u1..uF, v1..vF]: displacement blocks then velocity blocks.
    aug_offsets = list(moffs[:-1]) + [n + o for o in moffs]
    return FEM(domain=domain, op=block, classification=classification, mode="transient", offsets=aug_offsets)


# ---------------------------------------------------------------------------
# Regularization for nodal field parameters (jno.np.parameter(phi))
# ---------------------------------------------------------------------------
def _fe_symbols_bound(domain: Any):
    """``(ui, vi, axes)`` -- P1 trial/test bound to the domain's interior coordinates."""
    u_sym, v_sym = domain.fem_symbols()
    coords = domain.variable("interior", split=True)
    axes = ("x", "y", "z")[: int(domain.dimension)]
    ui = u_sym.bind(**{ax: coords[i] for i, ax in enumerate(axes)})
    vi = v_sym.bind(**{ax: coords[i] for i, ax in enumerate(axes)})
    return ui, vi, axes


def _assemble_fe_gram(domain: Any, kind: str = "stiffness"):
    """Raw P1 Gram matrix on the domain's scalar space (no Dirichlet, no ``init_fem``,
    ``store_on_domain=False`` so it never clobbers a cached FEM problem):

    * ``'stiffness'`` -> ``L = integral grad(phi_i).grad(phi_j)``  (``k^T L k = integral |grad k|^2``)
    * ``'mass'``      -> ``M = integral phi_i phi_j``              (``k^T M k = integral k^2``)

    Assembled natively: the bilinear form is retagged to the quadrature pool and run through
    ``assemble_fem_native`` with no Dirichlet, giving the raw Gram matrix on the field's P1 nodes.
    """
    from .utils.solver.fem_native import assemble_fem_native

    ui, vi, axes = _fe_symbols_bound(domain)
    if kind == "stiffness":
        form = None
        for ax in axes:
            t = getattr(ui, ax) * getattr(vi, ax)
            form = t if form is None else form + t
    elif kind == "mass":
        form = ui * vi
    else:
        raise ValueError(f"_assemble_fe_gram: unknown kind {kind!r}")
    bare = _bare(form)
    _retag_coords_for_quadrature(bare, "volume", "volume")
    qd = int(getattr(domain, "_fem_quad_degree", 0) or 2)
    op, _mode, _offs = assemble_fem_native(domain, [bare], {}, [], [], vec=1, quad_degree=qd)
    A = op[0]
    return jnp.asarray(A.todense() if hasattr(A, "todense") else A)


def _assemble_h1_stiffness(domain: Any):  # back-compat alias
    return _assemble_fe_gram(domain, "stiffness")


def _fe_element_gradient_data(domain: Any):
    """``(shape_grads, JxW, cells)`` for the domain's P1 space -- per-element gradient
    geometry for total variation ``integral |grad k|``. Built natively from the native
    fem_context, which carries the physical shape gradients and ``JxW`` per element."""
    from .utils.solver.fem_native import build_native_fem_context

    dim = int(domain.dimension)
    cells = np.asarray(domain.mesh.cells_dict["triangle" if dim == 2 else "tetra"])
    n_cells, n_local = cells.shape
    qd = int(getattr(domain, "_fem_quad_degree", 0) or 2)
    ctx, _qp, _sq, _sn = build_native_fem_context(
        domain, element_type="TRI3" if dim == 2 else "TET4", quad_degree=qd, vec=1
    )
    dN = jnp.asarray(ctx["dN_dx_flat"])  # (n_cells*n_q, n_local, dim)
    n_q = int(dN.shape[0]) // n_cells
    return (
        dN.reshape(n_cells, n_q, n_local, dim),  # (n_cells, n_quad, n_local, dim)
        jnp.asarray(ctx["JxW"]),  # (n_cells, n_quad)
        jnp.asarray(cells),
    )


_REG_KINDS = ("h1seminorm", "tv", "l2", "nonneg", "bounded")


def _field_regularizer_term(param: Any, kind: str = "h1seminorm", **kwargs):
    """Build a regularization loss term (a :class:`FunctionCall` over ``param``) for a
    nodal field parameter. See :meth:`ModelCall.regularize` for the kinds + options."""
    from .trace import FunctionCall

    domain = getattr(param.model, "_fem_field_domain", None)
    if domain is None:
        raise ValueError("regularize(...) is only for a FEM field parameter (jno.np.parameter(<fem symbol>)).")
    k = str(kind).lower()

    if k in ("h1seminorm", "h1", "smooth"):  # integral |grad k|^2 = k^T L k
        L = _assemble_fe_gram(domain, "stiffness")

        def _h1(kv, _L=L):
            kf = jnp.asarray(kv).reshape(-1)
            return kf * (_L @ kf)

        return FunctionCall(_h1, [param], name="h1seminorm")

    if k in ("l2", "tikhonov", "ridge"):  # integral (k - ref)^2 = (k-ref)^T M (k-ref)
        M = _assemble_fe_gram(domain, "mass")
        ref = jnp.asarray(kwargs.get("ref", 0.0))

        def _l2(kv, _M=M, _ref=ref):
            kf = jnp.asarray(kv).reshape(-1)
            kf = kf - (_ref.reshape(-1) if _ref.ndim else _ref)
            return kf * (_M @ kf)

        return FunctionCall(_l2, [param], name="l2")

    if k in ("tv", "totalvariation"):  # integral |grad k|  (edge-preserving; eps-smoothed)
        sg, jxw, cells = _fe_element_gradient_data(domain)
        eps = float(kwargs.get("eps", 1.0e-8))

        def _tv(kv, _sg=sg, _jxw=jxw, _cells=cells, _eps=eps):
            kc = jnp.asarray(kv).reshape(-1)[_cells]  # (n_cells, n_local)
            gradk = jnp.einsum("cqld,cl->cqd", _sg, kc)  # (n_cells, n_quad, dim)
            mag = jnp.sqrt(jnp.sum(gradk**2, axis=-1) + _eps)  # (n_cells, n_quad)
            jxw = jnp.reshape(_jxw, mag.shape)  # JxW arrives (n_cells, 1, n_quad)
            return jnp.sum(mag * jxw, axis=1)  # (n_cells,) per-element TV

        return FunctionCall(_tv, [param], name="tv")

    if k == "nonneg":  # soft positivity barrier strength * relu(-k)
        strength = float(kwargs.get("strength", 1.0))

        def _nn(kv, _s=strength):
            return _s * jnp.maximum(0.0, -jnp.asarray(kv).reshape(-1))

        return FunctionCall(_nn, [param], name="nonneg")

    if k == "bounded":  # soft two-sided barrier outside [lo, hi]
        if "lo" not in kwargs or "hi" not in kwargs:
            raise ValueError("regularize('bounded', lo=..., hi=...) requires both lo and hi.")
        lo, hi = float(kwargs["lo"]), float(kwargs["hi"])

        def _bd(kv, _lo=lo, _hi=hi):
            kf = jnp.asarray(kv).reshape(-1)
            return jnp.maximum(0.0, kf - _hi) + jnp.maximum(0.0, _lo - kf)

        return FunctionCall(_bd, [param], name="bounded")

    raise ValueError(f"Unknown regularizer kind {kind!r}; supported: {_REG_KINDS}.")
