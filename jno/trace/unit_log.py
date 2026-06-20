"""Dimensional-analysis inference walk, warnings, and human-readable log.

This module is the engine behind ``jno.units.check(nodes, log=...)``.  It walks
the traced expression graph(s) in post-order, infers a :class:`~jno.trace.units.Unit`
for every node from its already-visited children, and records mismatches
(adding unlike units; passing a dimensioned argument to ``exp``/``log``/``sin``…).

Design notes
------------
* **Walk-based, not eager.**  Inference runs only when ``check()`` is called, so
  the hot graph-construction path (``BinaryOp.__init__`` etc.) is untouched and
  the feature costs nothing unless opted into.
* **Non-mutating.**  Inferred units for intermediate nodes live in a local
  ``id(node) -> Unit`` map for the duration of one ``check()`` call; nodes are
  never written to (a stored/shared graph can be checked repeatedly with no
  cross-talk).  Only *user-declared* leaf units live on the node, set explicitly
  via :meth:`Placeholder.unit`.
* **Honest about the unknown.**  A unit of ``None`` means "not derivable" (an
  undeclared leaf, or an operation on an undeclared operand) and never triggers
  a warning — only two *known* but incompatible units do.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .units import DIMENSIONLESS, Unit

# Functions whose argument must be dimensionless; result is dimensionless.
_TRANSCENDENTAL = {
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
}
# Functions that pass their (single) argument's unit through unchanged.
_UNIT_PRESERVING = {
    "abs",
    "absolute",
    "neg",
    "negative",
    "identity",
    "reshape",
    "getitem",
    "print",
    "stop_gradient",
    "copy",
    "real",
    "imag",
}


class UnitLogger:
    """Accumulates one ``(label, unit, warnings)`` entry per visited node."""

    def __init__(self):
        self.entries: List[Tuple[str, Optional[Unit], List[str]]] = []

    @property
    def warnings(self) -> List[str]:
        """Flat list of every warning string recorded across all nodes."""
        return [w for _, _, ws in self.entries for w in ws]

    def record(self, label: str, unit: Optional[Unit], warnings: Sequence[str] = ()) -> None:
        self.entries.append((label, unit, list(warnings)))

    def render(self) -> str:
        width = max((len(label) for label, _, _ in self.entries), default=0)
        lines = []
        for label, unit, warnings in self.entries:
            unit_str = "?" if unit is None else repr(unit)
            line = f"  {label.ljust(width)}  →  {unit_str}"
            for w in warnings:
                line += f"    [WARN: {w}]"
            lines.append(line)
        return "\n".join(lines)

    def write(self, path: str) -> None:
        header = "# jno dimensional-analysis log\n# node  →  inferred unit  [warnings]\n\n"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header + self.render() + "\n")


# ---------------------------------------------------------------------------
# Child enumeration — mirrors the attribute set used by the trace compiler's
# own graph walk so every node type is traversed uniformly.
# ---------------------------------------------------------------------------
_CHILD_ATTRS = ("target", "left", "right", "expr", "volume_expr", "time_var", "integration_var")
_CHILD_LIST_ATTRS = ("variables", "args", "boundary_exprs", "options")


def _children(node):
    from . import Placeholder

    out = []
    for attr in _CHILD_ATTRS:
        child = getattr(node, attr, None)
        if isinstance(child, Placeholder):
            out.append(child)
    for attr in _CHILD_LIST_ATTRS:
        children = getattr(node, attr, None)
        if isinstance(children, (list, tuple)):
            out.extend(c for c in children if isinstance(c, Placeholder))
    return out


def _fn_name(node) -> str:
    name = getattr(node, "_name", None) or getattr(getattr(node, "fn", None), "__name__", "")
    return str(name).lower()


def _label(node) -> str:
    text = repr(node)
    return text if len(text) <= 60 else text[:57] + "..."


def _infer_unit(node, umap: Dict[int, Optional[Unit]]) -> Tuple[Optional[Unit], List[str]]:
    """Infer a unit for *node* from already-inferred child units in *umap*.

    Returns ``(unit_or_None, warnings)``.  ``None`` = not derivable.
    """
    from . import (
        BinaryOp,
        FunctionCall,
        Hessian,
        Integral,
        IntegralTime,
        Jacobian,
        Literal,
        TemporalDerivative,
        TestFunction,
        Tracker,
    )

    warnings: List[str] = []

    def cu(child):  # child unit
        return umap.get(id(child))

    # --- leaves --------------------------------------------------------------
    if isinstance(node, Literal):
        return DIMENSIONLESS, warnings
    if isinstance(node, TestFunction):
        # Convention: the test function is dimensionless (the weak form divides
        # it out).  TrialFunction, by contrast, is a leaf that carries a
        # user-declared unit like any other unknown field.
        return DIMENSIONLESS, warnings
    if isinstance(node, Tracker):
        return cu(node.expr), warnings

    # --- arithmetic ----------------------------------------------------------
    if isinstance(node, BinaryOp):
        lu, ru = cu(node.left), cu(node.right)
        if node.op in ("+", "-"):
            if lu is not None and ru is not None and lu != ru:
                warnings.append(f"left={lu!r} right={ru!r} — units mismatch under '{node.op}'")
                return lu, warnings
            return (lu if lu is not None else ru), warnings
        if node.op == "*":
            return (lu * ru if lu is not None and ru is not None else None), warnings
        if node.op == "/":
            return (lu / ru if lu is not None and ru is not None else None), warnings
        if node.op == "**":
            exponent = _literal_scalar(node.right)
            if lu is None or exponent is None:
                return None, warnings
            return lu**exponent, warnings
        return None, warnings

    # --- derivatives ---------------------------------------------------------
    if isinstance(node, Jacobian):
        tu = cu(node.target)
        var_units = [cu(v) for v in node.variables]
        if tu is None or any(u is None for u in var_units):
            return None, warnings
        if len({u for u in var_units}) > 1:
            # heterogeneous gradient vector — no single unit
            return None, warnings
        return tu / var_units[0], warnings

    if isinstance(node, Hessian):
        tu = cu(node.target)
        var_units = [cu(v) for v in node.variables]
        if tu is None or any(u is None for u in var_units):
            return None, warnings
        distinct = {u for u in var_units}
        if node.trace:  # Laplacian: Σ ∂²/∂vᵢ² — all vᵢ must share a unit
            if len(distinct) > 1:
                warnings.append("Laplacian over variables with differing units")
            return tu / (var_units[0] ** 2), warnings
        # full Hessian matrix: single unit only for a single variable
        if len(distinct) == 1:
            return tu / (var_units[0] ** 2), warnings
        return None, warnings

    if isinstance(node, TemporalDerivative):
        tu, vu = cu(node.target), cu(getattr(node, "time_var", None))
        return (tu / vu if tu is not None and vu is not None else None), warnings

    # --- integrals (best-effort: needs a declared spatial coordinate) --------
    if isinstance(node, Integral):
        tu = cu(node.target)
        coord = _first_spatial_var(node.target)
        cu_coord = umap.get(id(coord)) if coord is not None else None
        if tu is None or cu_coord is None:
            return None, warnings
        ndim = int(getattr(coord, "size", 1) or 1)
        return tu * (cu_coord**ndim), warnings

    if isinstance(node, IntegralTime):
        tu, vu = cu(node.target), cu(getattr(node, "time_var", None))
        return (tu * vu if tu is not None and vu is not None else None), warnings

    # --- function calls ------------------------------------------------------
    if isinstance(node, FunctionCall):
        name = _fn_name(node)
        arg_units = [cu(a) for a in _children(node)]
        first = arg_units[0] if arg_units else None
        if name in _TRANSCENDENTAL:
            if first is not None and not first.is_dimensionless():
                warnings.append(f"argument of '{name}' has unit {first!r} — must be dimensionless")
            return DIMENSIONLESS, warnings
        if name in ("sqrt",):
            return (first**0.5 if first is not None else None), warnings
        if name in _UNIT_PRESERVING:
            return first, warnings
        return None, warnings

    # --- everything else (Model/ModelCall/Variable/TensorTag/TrialFunction) --
    # User-declared unit if present (set via Placeholder.unit()), else unknown.
    return getattr(node, "_unit", None), warnings


def _literal_scalar(node) -> Optional[float]:
    """Extract a Python float from a Literal exponent node, else None."""
    from . import Literal

    if not isinstance(node, Literal):
        return None
    try:
        value = node.value
        return float(value.item() if hasattr(value, "item") else value)
    except (TypeError, ValueError):
        return None


def _first_spatial_var(root):
    """Return the first spatial Variable found under *root* (pre-order), or None."""
    from . import Variable

    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Variable) and getattr(node, "axis", "spatial") == "spatial":
            return node
        stack.extend(_children(node))
    return None


def check(nodes, log: Optional[str] = None) -> UnitLogger:
    """Infer and audit physical units across one or more expression trees.

    Walks each node in *nodes* (a single :class:`~jno.trace.Placeholder` or a
    list/tuple of them) in post-order, inferring a unit for every sub-node and
    recording any dimensional inconsistencies.  Optionally writes a
    human-readable log.

    Parameters
    ----------
    nodes:
        A traced expression or a sequence of them (e.g. the residuals passed to
        ``jno.fem([...])``).
    log:
        If given, a filesystem path the per-node log is written to.

    Returns
    -------
    UnitLogger
        Inspect ``.entries`` (``(label, unit, warnings)`` triples) or
        ``.warnings`` (flat list) programmatically.
    """
    if not isinstance(nodes, (list, tuple)):
        nodes = [nodes]

    logger = UnitLogger()
    umap: Dict[int, Optional[Unit]] = {}
    visited = set()

    def visit(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        for child in _children(node):
            visit(child)
        unit, warnings = _infer_unit(node, umap)
        umap[id(node)] = unit
        logger.record(_label(node), unit, warnings)

    for root in nodes:
        visit(root)

    if log is not None:
        logger.write(log)
    return logger


def infer(node) -> Optional[Unit]:
    """Convenience: return the inferred unit of a single expression's root.

    Equivalent to ``check([node]).entries[-1][1]`` — runs a fresh, non-mutating
    walk and hands back the top node's unit (``None`` if not derivable).
    """
    return check([node]).entries[-1][1]


# ===========================================================================
# Non-dimensionalization (Phase A: analysis / dimensionless groups)
# ---------------------------------------------------------------------------
# Units give a node's *dimension*; ``.scale(value)`` gives a leaf's
# characteristic *magnitude*.  The magnitude walk below mirrors ``_infer_unit``
# rule-for-rule but propagates the per-leaf scales (floats) through each term's
# subtree.  The dimensionless group of an additive term is then its magnitude
# divided by a reference term's magnitude — e.g. the heat equation's diffusion
# term yields the Fourier number ``ατ/L²`` and advection–diffusion yields the
# Péclet number ``VL/α`` (Buckingham-π non-dimensionalization;
# see e.g. Barenblatt, *Scaling, Self-similarity, and Intermediate Asymptotics*,
# CUP 1996, Ch. 1).  Crucially the group must come from this subtree walk, NOT
# from a term's *net* unit: in a consistent residual every additive term shares
# the same net unit, so a net-unit formula collapses every group to 1.
# ===========================================================================


def _infer_scale(node, smap: Dict[int, Optional[float]], coeff_neutral: bool = False) -> Optional[float]:
    """Magnitude (float) of *node* from child magnitudes in *smap*.

    Sibling of :func:`_infer_unit` with identical dispatch and the same
    :func:`_children` traversal.  ``None`` = not derivable (an undeclared leaf,
    or an op on one).

    When *coeff_neutral* is set, coefficient leaves (``Constant``/``TensorTag``)
    contribute magnitude 1 instead of their declared scale — this yields the
    coordinate-plus-field magnitude ``Kᵢ`` used by the rescale transform, as
    opposed to the full magnitude ``Sᵢ``.
    """
    from . import (
        BinaryOp,
        Constant,
        FunctionCall,
        Hessian,
        Integral,
        IntegralTime,
        Jacobian,
        Literal,
        TemporalDerivative,
        TensorTag,
        TestFunction,
        Tracker,
        Variable,
    )

    def cs(child):  # child scale
        return smap.get(id(child))

    def vs(v):  # coordinate scale, read DIRECTLY (not from the possibly-neutralized smap)
        return getattr(v, "_scale", None)

    if coeff_neutral and isinstance(node, (Constant, TensorTag)):
        return 1.0

    # --- leaves --------------------------------------------------------------
    if isinstance(node, Literal):
        try:
            mag = abs(float(node.value.item() if hasattr(node.value, "item") else node.value))
        except (TypeError, ValueError):
            return None
        return mag or 1.0  # a bare 0/±1 carries no magnitude → neutral in products
    if isinstance(node, TestFunction):
        return 1.0  # dimensionless by convention; magnitude 1
    if isinstance(node, Tracker):
        return cs(node.expr)

    # --- arithmetic ----------------------------------------------------------
    if isinstance(node, BinaryOp):
        sl, sr = cs(node.left), cs(node.right)
        if node.op in ("+", "-"):
            return sl if sl is not None else sr  # terms share a magnitude
        if node.op == "*":
            return sl * sr if sl is not None and sr is not None else None
        if node.op == "/":
            return sl / sr if sl is not None and sr is not None else None
        if node.op == "**":
            exponent = _literal_scalar(node.right)
            if sl is None or exponent is None:
                return None
            return sl**exponent
        return None

    # --- derivatives (read coordinate scales DIRECTLY via vs(), see vs note) --
    if isinstance(node, Jacobian):
        st = cs(node.target)
        var_scales = [vs(v) for v in node.variables]
        if st is None or any(s is None for s in var_scales):
            return None
        if len(set(var_scales)) > 1:
            return None
        return st / var_scales[0]

    if isinstance(node, Hessian):
        st = cs(node.target)
        var_scales = [vs(v) for v in node.variables]
        if st is None or any(s is None for s in var_scales):
            return None
        if node.trace:  # Laplacian
            return st / (var_scales[0] ** 2)
        if len(set(var_scales)) == 1:
            return st / (var_scales[0] ** 2)
        return None

    if isinstance(node, TemporalDerivative):
        st, sv = cs(node.target), vs(getattr(node, "time_var", None))
        return st / sv if st is not None and sv is not None else None

    if isinstance(node, Integral):
        st = cs(node.target)
        coord = _first_spatial_var(node.target)
        sc = vs(coord) if coord is not None else None
        if st is None or sc is None:
            return None
        ndim = int(getattr(coord, "size", 1) or 1)
        return st * (sc**ndim)

    if isinstance(node, IntegralTime):
        st, sv = cs(node.target), vs(getattr(node, "time_var", None))
        return st * sv if st is not None and sv is not None else None

    # --- function calls ------------------------------------------------------
    if isinstance(node, FunctionCall):
        name = _fn_name(node)
        child_scales = [cs(a) for a in _children(node)]
        first = child_scales[0] if child_scales else None
        if name in _TRANSCENDENTAL:
            return 1.0
        if name == "sqrt":
            return first**0.5 if first is not None else None
        if name in _UNIT_PRESERVING:
            return first
        return None

    # --- bare-coordinate leaf -------------------------------------------------
    # A coordinate Variable appearing as a value (child of arithmetic/function)
    # contributes magnitude 1 in the coefficient-excluded K-walk: its scale L is
    # materialized in the graph by the bare-coordinate rewrite, so counting it
    # here too would double-apply L.  Derivative rules above read the real L via
    # vs() directly, independent of this neutralization.
    if isinstance(node, Variable):
        return 1.0 if coeff_neutral else getattr(node, "_scale", None)

    # --- other leaves carrying a user-declared scale (TrialFunction/Model/...) -
    return getattr(node, "_scale", None)


def _scale_of(node, coeff_neutral: bool = False) -> Optional[float]:
    """Run a fresh non-mutating magnitude walk and return *node*'s magnitude.

    With *coeff_neutral*, coefficient leaves count as magnitude 1 (returns the
    coordinate-plus-field magnitude ``Kᵢ`` rather than the full ``Sᵢ``).
    """
    smap: Dict[int, Optional[float]] = {}
    visited = set()

    def visit(n):
        if id(n) in visited:
            return
        visited.add(id(n))
        for child in _children(n):
            visit(child)
        smap[id(n)] = _infer_scale(n, smap, coeff_neutral)

    visit(node)
    return smap[id(node)]


def _additive_terms(node, sign: int = 1) -> List[Tuple[int, object]]:
    """Flatten the top-level ``+``/``-`` chain into signed additive terms."""
    from . import BinaryOp

    if isinstance(node, BinaryOp) and node.op in ("+", "-"):
        left = _additive_terms(node.left, sign)
        right_sign = sign if node.op == "+" else -sign
        right = _additive_terms(node.right, right_sign)
        return left + right
    return [(sign, node)]


class TermInfo:
    """One additive term of a residual: its node, sign, unit, magnitude, π."""

    def __init__(self, term, sign: int, unit: Optional[Unit], scale: Optional[float]):
        self.term = term
        self.sign = sign
        self.unit = unit
        self.scale = scale
        self.pi: Optional[float] = None

    def __repr__(self):
        return f"TermInfo({_label(self.term)}, unit={self.unit!r}, scale={self.scale}, pi={self.pi})"


class ResidualReport:
    """Per-residual non-dimensionalization result."""

    def __init__(self, root, terms: List[TermInfo], ref_index: Optional[int]):
        self.root = root
        self.terms = terms
        self.ref_index = ref_index

    @property
    def pis(self) -> List[Optional[float]]:
        return [t.pi for t in self.terms]


class NondimReport:
    """Result of :func:`nondimensionalize`: one :class:`ResidualReport` each."""

    def __init__(self, residuals: List[ResidualReport]):
        self.residuals = residuals

    def render(self) -> str:
        blocks = []
        for r in self.residuals:
            labels = [_label(t.term) for t in r.terms]
            width = max((len(s) for s in labels), default=0)
            lines = [f"residual: {_label(r.root)}"]
            for i, t in enumerate(r.terms):
                unit_str = "?" if t.unit is None else repr(t.unit)
                scale_str = "?" if t.scale is None else f"{t.scale:.6g}"
                pi_str = "?" if t.pi is None else f"{t.pi:.6g}"
                ref_mark = "  (ref)" if i == r.ref_index else ""
                lines.append(f"  {labels[i].ljust(width)}  unit={unit_str}  scale={scale_str}  π={pi_str}{ref_mark}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def write(self, path: str) -> None:
        header = "# jno non-dimensionalization report\n# per additive term: inferred unit, characteristic magnitude, dimensionless group π\n\n"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header + self.render() + "\n")


def nondimensionalize(nodes, ref: Optional[int] = None, report: Optional[str] = None) -> NondimReport:
    """Derive characteristic magnitudes and dimensionless groups for residual(s).

    For each residual in *nodes*, splits the top-level additive terms, infers
    each term's :class:`~jno.trace.units.Unit` (via :func:`infer`) and
    characteristic magnitude (via the per-leaf ``.scale`` walk), then forms the
    dimensionless group of each term as ``πᵢ = Sᵢ / S_ref``.

    Phase A only — this is pure analysis and does **not** change what the solver
    computes.

    Parameters
    ----------
    nodes:
        A residual expression or a sequence of them.
    ref:
        Index of the reference term whose magnitude normalises the groups.
        Defaults to the first term with both a known unit and a known scale.
    report:
        If given, a path the human-readable report is written to.

    Returns
    -------
    NondimReport
        ``.residuals[i].terms[j].pi`` holds each term's dimensionless group.
    """
    if not isinstance(nodes, (list, tuple)):
        nodes = [nodes]

    residuals: List[ResidualReport] = []
    for root in nodes:
        terms = [TermInfo(term, sign, infer(term), _scale_of(term)) for sign, term in _additive_terms(root)]

        if ref is not None:
            ref_index: Optional[int] = ref if 0 <= ref < len(terms) else None
        else:
            ref_index = next(
                (i for i, t in enumerate(terms) if t.unit is not None and t.scale is not None),
                None,
            )

        if ref_index is not None and terms[ref_index].scale:
            s_ref = terms[ref_index].scale
            for t in terms:
                t.pi = t.scale / s_ref if t.scale is not None else None

        residuals.append(ResidualReport(root, terms, ref_index))

    result = NondimReport(residuals)
    if report is not None:
        result.write(report)
    return result


# ===========================================================================
# Non-dimensionalization (Phase B: transform a problem to dimensionless form)
# ---------------------------------------------------------------------------
# The transform is Phase A applied constructively.  Each additive term is
# multiplied by ``gᵢ = Kᵢ / S_ref`` where ``Kᵢ`` is the term's *coordinate +
# field* magnitude (coefficients excluded) and ``S_ref`` the reference term's
# full magnitude.  On a coordinate context rescaled to O(1) — and with the
# network/trial output representing the O(1) dimensionless field — the
# transformed residual evaluates to ``Σ signᵢ·πᵢ·term̂ᵢ`` (the dimensionless
# residual whose leading coefficients are the Fourier/Péclet numbers).
#
# Note: the transformed graph is *not* symbolically dimensionless — the
# dimensional coordinates/coefficients remain in it; dimensionlessness is
# realized numerically once the rescaled (O(1)) context and field are supplied.
# Multiplying by ``gᵢ`` (rather than substituting coefficients) is the
# least-invasive correct mechanism: it never touches Variable identity, so the
# derivative semantics of Jacobian/Hessian are preserved.
# ===========================================================================


def _scale_columns(arr, d0: int, d1: Optional[int], factor: float):
    """Return *arr* with last-axis columns ``[d0:d1]`` multiplied by *factor*."""
    hi = arr.shape[-1] if d1 is None else d1
    if hasattr(arr, "at"):  # jax array — functional update
        return arr.at[..., d0:hi].multiply(factor)
    import numpy as _np

    out = _np.array(arr, copy=True)
    out[..., d0:hi] = out[..., d0:hi] * factor
    return out


class Rescaler:
    """Back-map between physical and dimensionless (O(1)) coordinates/fields.

    Produced by :func:`rescale`.  Holds the characteristic length(s) ``L`` of
    each coordinate and the field scale ``U`` so a rescaled problem can be set
    up and its solution recovered: ``u_physical(x) = U · û(x / L)``.
    """

    def __init__(self, coords, field_scale: Optional[float]):
        # coords: list of (tag, (d0, d1), L) — one per scaled coordinate column
        self._coords = coords
        self.field_scale = field_scale

    def rescaled_context(self, context: dict) -> dict:
        """Return a new context dict with each coordinate column scaled to O(1)."""
        out = dict(context)
        for tag, (d0, d1), L in self._coords:
            if tag in out and L:
                out[tag] = _scale_columns(out[tag], d0, d1, 1.0 / L)
        return out

    def rescaled_domain(self, domain):
        """Return a shallow copy of *domain* whose context is rescaled to O(1).

        The user's *domain* is never mutated.
        """
        import copy as _copy

        dom = _copy.copy(domain)
        dom.context = self.rescaled_context(domain.context)
        return dom

    def to_physical(self, field):
        """Map a dimensionless field value ``û`` back to physical units (``U·û``)."""
        return field if self.field_scale is None else field * self.field_scale

    def __repr__(self):
        return f"Rescaler(coords={len(self._coords)}, field_scale={self.field_scale})"


def _collect_scaled_leaves(constraints):
    """Find scaled coordinate Variables and the field scale across *constraints*."""
    from . import Model, ModelCall, TrialFunction, Variable

    coords: Dict[int, tuple] = {}
    field_scales: List[float] = []
    seen = set()
    stack = list(constraints)
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        s = getattr(n, "_scale", None)
        if s is not None:
            if isinstance(n, Variable):
                d = tuple(n.dim) if n.dim is not None else (0, None)
                coords[(n.tag, d)] = (n.tag, d, s)
            elif isinstance(n, (Model, ModelCall, TrialFunction)):
                field_scales.append(s)
        stack.extend(_children(n))
    return list(coords.values()), (field_scales[0] if field_scales else None)


def _unwrap_view(op):
    """Unwrap a typed semantic view (ScalarView/FieldView/…) to its Placeholder."""
    from .views import _VIEW_TYPES

    return op._expr if isinstance(op, _VIEW_TYPES) else op


def _scaled_coord_map(node):
    """``{id(var): (var, L)}`` for every scaled coordinate Variable under *node*."""
    from . import Variable

    out: Dict[int, tuple] = {}
    seen = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, Variable) and getattr(n, "_scale", None) is not None:
            out[id(n)] = (n, n._scale)
        stack.extend(_children(n))
    return out


def _rewrite_bare_coords(node, coord_map):
    """Return *node* with bare-coordinate leaves ``x`` replaced by ``L·x``.

    A coordinate is rewritten only where it appears as a *value* (child of an
    arithmetic/function node) — never as a network input (``ModelCall`` arg) or
    a differentiation variable (``Jacobian``/``Hessian``/``TemporalDerivative``
    ``.variables``/``.time_var``), which instead ride the rescaled context. This
    keeps analytic targets like ``sin(π·x/L)`` physical on the rescaled context
    while derivatives stay dimensionless. ``coord_map``: ``{id(var): (var, L)}``.
    """
    from . import (
        BinaryOp,
        FunctionCall,
        Hessian,
        Jacobian,
        Literal,
        ModelCall,
        Placeholder,
        TemporalDerivative,
    )

    def as_value(child):
        # *child* used as a value: wrap a scaled coordinate, else recurse.
        if not isinstance(child, Placeholder):
            return child
        info = coord_map.get(id(child))
        if info is not None:
            var, length = info
            return BinaryOp("*", Literal(length), var)
        return rewrite(child)

    def rewrite(n):
        if isinstance(n, BinaryOp):
            return BinaryOp(n.op, as_value(n.left), as_value(n.right))
        if isinstance(n, FunctionCall):
            return n.copy_with_args([as_value(a) for a in n.args])
        if isinstance(n, ModelCall):
            return n  # network inputs ride the rescaled context — leave args
        if isinstance(n, Jacobian):
            return Jacobian(rewrite(n.target), n.variables, n.scheme)
        if isinstance(n, Hessian):
            return Hessian(rewrite(n.target), n.variables, n.scheme, n.trace)
        if isinstance(n, TemporalDerivative):
            return TemporalDerivative(rewrite(n.target), n.time_var)
        return n  # leaves & other nodes unchanged

    return as_value(node)


def rescale(constraints, domain=None):
    """Transform residual constraint(s) into a solvable dimensionless form.

    Multiplies each additive term of every residual by ``gᵢ = Kᵢ / S_ref`` so
    that, evaluated on the :class:`Rescaler`'s O(1) coordinate context (and with
    the network/trial output representing the dimensionless field), the residual
    becomes ``Σ signᵢ·πᵢ·term̂ᵢ`` — O(1) and dimensionless, with the
    dimensionless groups (Fourier/Péclet) as the leading coefficients.

    Phase B — opt-in.  Returns ``(transformed_constraints, rescaler)``; nothing
    is solved and the input *domain* is never mutated.

    Parameters
    ----------
    constraints:
        A residual expression or a sequence of them.
    domain:
        Unused for the transform itself; accepted so callers can pair it with
        ``rescaler.rescaled_domain(domain)``.

    Returns
    -------
    (list, Rescaler)
        The transformed constraints (same length/order as the input) and the
        back-map :class:`Rescaler`.
    """
    from . import BinaryOp, Literal

    single = not isinstance(constraints, (list, tuple))
    items = [_unwrap_view(c) for c in ([constraints] if single else list(constraints))]

    transformed = []
    for root in items:
        coord_map = _scaled_coord_map(root)
        terms = _additive_terms(root)
        # gᵢ is computed on the ORIGINAL term (pre-rewrite) so the rewrite's L
        # factor is not counted twice; the rewrite then materializes it once.
        full = [_scale_of(t) for _, t in terms]
        kvals = [_scale_of(t, coeff_neutral=True) for _, t in terms]

        ref_index = next((i for i, s in enumerate(full) if s), None)
        s_ref = full[ref_index] if ref_index is not None else None

        node = None
        for (sign, term), k in zip(terms, kvals):
            g = (k / s_ref) if (k is not None and s_ref) else 1.0
            rewritten = _rewrite_bare_coords(term, coord_map)
            coeff = sign * g
            piece = rewritten if coeff == 1.0 else BinaryOp("*", Literal(coeff), rewritten)
            node = piece if node is None else BinaryOp("+", node, piece)
        transformed.append(node if node is not None else root)

    coords, field_scale = _collect_scaled_leaves(items)
    rescaler = Rescaler(coords, field_scale)
    return (transformed[0] if single else transformed), rescaler
