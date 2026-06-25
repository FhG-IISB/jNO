"""Structural classification of FEM weak-form terms — the substrate's term taxonomy.

Every weak-form term is described by a few independent structural axes: where it lives
(support), its temporal-derivative order, whether the trial/test fields appear under a
*spatial* gradient (the channel), and whether it is linear in the unknown. In particular
this detects whether a term is **local** — spatially pointwise, i.e. no spatial gradient on
either trial or test — which is a per-node reaction/mass contribution (diagonal under
lumping) rather than a neighbour-coupling global one. Operator-splitting / IMEX drivers use
``is_local`` to decide which terms to peel into a node ODE.

Representation note: front-end terms from ``dom.fem_symbols()`` are ``ScalarView`` wrappers
over the underlying ``Placeholder`` IR (reachable via ``term.expr``); classification runs on
that IR. A spatial derivative and a time derivative are *both* ``Jacobian(target, [var])`` —
the axis is decided by whether the ``Variable`` is temporal (``is_temporal_var``), so a mass
term ``u.t * v`` is correctly spatially local.

PROVISIONAL: the public surface here may change once the operator-splitting routing that
consumes it lands. See ``plans/fem_local_terms_substrate.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .solver_helper import (
    contains_node_type,
    is_temporal_var,
    iter_children,
    max_temporal_derivative_order,
)
from .weak_form_helpers import infer_term_bucket


def _trial_classes():
    from ...trace import StateField, TrialFunction

    return (TrialFunction, StateField)


def _test_classes():
    from ...trace import TestFunction

    return (TestFunction,)


@dataclass(frozen=True)
class TermKind:
    """Structural fingerprint of a single weak-form term.

    Attributes
    ----------
    support: ``"volume"`` | ``"boundary"`` | ``"constraint"`` (Dirichlet/IC/periodic tie — no test fn).
    region: variational region id, or ``None``.
    time_order: max temporal-derivative order (0 steady, 1 first-order, 2 second-order).
    trial_channel / test_channel: ``"none"`` (field absent) | ``"value"`` | ``"grad"`` (appears
        under a spatial gradient). A field appearing both ways is labelled ``"grad"``
        (conservative — correct for the ``is_local`` decision).
    linear: linear in the unknown (negated ``_is_obviously_nonlinear_in_unknown``).
    """

    support: str
    region: Any
    time_order: int
    trial_channel: str
    test_channel: str
    linear: bool

    @property
    def is_local(self) -> bool:
        """A volume term with no spatial gradient on trial or test → spatially pointwise."""
        return self.support == "volume" and self.trial_channel != "grad" and self.test_channel != "grad"


def _has_spatial_grad_over(node: Any, field_classes) -> bool:
    """True if ``node`` contains a Jacobian over a *spatial* variable whose target is the field."""
    from ...trace import Jacobian, Variable

    if isinstance(node, Jacobian):
        spatial = any(isinstance(v, Variable) and not is_temporal_var(v) for v in node.variables)
        if spatial and contains_node_type(node.target, field_classes):
            return True
    return any(_has_spatial_grad_over(c, field_classes) for c in (iter_children(node) or ()))


def _spatial_channel(ir: Any, field_classes) -> str:
    if not contains_node_type(ir, field_classes):
        return "none"
    return "grad" if _has_spatial_grad_over(ir, field_classes) else "value"


def classify_term(domain, term) -> TermKind:
    """Classify a single weak-form term (front-end ``ScalarView`` or raw IR) into a ``TermKind``."""
    from ...trace import TestFunction

    ir = getattr(term, "expr", term)  # unwrap ScalarView -> Placeholder IR

    if not contains_node_type(ir, TestFunction):
        support, region = "constraint", None  # Dirichlet / IC / periodic tie (no test function)
    else:
        try:
            support, region = infer_term_bucket(domain, ir)
        except Exception:
            support, region = "volume", None

    # lazy import to avoid an import cycle through weak_form
    from .weak_form import _is_obviously_nonlinear_in_unknown

    return TermKind(
        support=support,
        region=region,
        time_order=int(max_temporal_derivative_order(ir)),
        trial_channel=_spatial_channel(ir, _trial_classes()),
        test_channel=_spatial_channel(ir, _test_classes()),
        linear=not bool(_is_obviously_nonlinear_in_unknown(domain, ir)),
    )
