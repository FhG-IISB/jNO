"""The differentiation-scheme families, and the one place a scheme string is resolved.

A ``scheme=`` string is ``family`` or ``family:submethod`` — ``"automatic_differentiation:forward"``,
``"finite_difference:cotangent"``. The **family** picks which backend differentiates; the submethod is
that backend's business and is parsed by it (:func:`jno.utils.ad_mode.parse_ad_scheme`,
:meth:`jno.differential_operators.DifferentialOperators.parse_fd_scheme`).

This module exists because an unrecognised family used to be *silently reinterpreted*, three
different ways, none of them an error:

* ``parse_ad_scheme`` discards the family half outright (``_, sub = scheme.split(":", 1)``), so a
  scheme with no colon returned the global AD default whatever its name.
* ``parse_fd_scheme`` fell through to the finite-difference defaults, and a ``family:sub`` it did not
  know handed ``sub`` to the FD kernel as a ``method=``.
* The evaluator's spatial dispatch had no ``else``, so an unknown family returned ``None`` and
  surfaced much later as ``TypeError: 'NoneType' object is not subscriptable``.

Resolving the family here first makes all three fail loud at the point of use. Registering a new
family is adding an entry to :data:`SCHEME_FAMILIES` plus a backend — the evaluator does not change.
"""

from __future__ import annotations

#: Every differentiation family jNO knows, with a one-line description used in error messages.
#: A new backend adds itself here; nothing else enumerates the families.
SCHEME_FAMILIES: dict[str, str] = {
    "automatic_differentiation": "exact derivatives of the traced function (jacfwd / jacrev)",
    "finite_difference": "stencils over the mesh or structured grid",
}


def scheme_family(scheme: str) -> str:
    """The validated family half of ``scheme``.

    Raises rather than guessing: an unknown family is a typo or a backend that was never registered,
    and every historical way of tolerating it produced a wrong answer instead of an error.
    """
    if not isinstance(scheme, str):
        raise TypeError(f"scheme must be a string, got {type(scheme).__name__}: {scheme!r}")
    family = scheme.split(":", 1)[0].strip()
    if family not in SCHEME_FAMILIES:
        known = ", ".join(f"{k!r} ({v})" for k, v in SCHEME_FAMILIES.items())
        raise ValueError(f"Unknown differentiation scheme family {family!r} (from scheme={scheme!r}). Known: {known}.")
    return family


def require_family(scheme: str, expected: str) -> str:
    """The submethod of ``scheme``, after checking it belongs to ``expected``.

    Used by each family's own parser so it cannot be handed a scheme belonging to a different
    backend — which is how ``"spectral"`` used to end up producing an automatic-differentiation
    derivative, and how ``"spectral:fft"`` used to reach the finite-difference kernel as
    ``method="fft"``.
    """
    family = scheme_family(scheme)
    if family != expected:
        raise ValueError(
            f"scheme={scheme!r} belongs to the {family!r} family, but this is the {expected!r} parser. "
            "The scheme was routed to the wrong backend."
        )
    return scheme.split(":", 1)[1].strip() if ":" in scheme else ""
