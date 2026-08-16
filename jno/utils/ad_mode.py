"""AD mode (forward / reverse) selection for jNO operators.

Two layers, in order of precedence:

1. Per-call scheme suffix on the operator:

   - First-order   (``.d``, ``.diff``, ``d/dt``)::

       u.d(x, scheme="automatic_differentiation:forward")
       u.d(x, scheme="automatic_differentiation:reverse")

   - Second-order  (``.laplacian``, ``.hessian``, ``.d2``, ``.dd``)::

       u.laplacian(x, y, scheme="automatic_differentiation:fwd-over-rev")
       u.laplacian(x, y, scheme="automatic_differentiation:fwd-over-fwd")
       u.laplacian(x, y, scheme="automatic_differentiation:rev-over-rev")
       u.laplacian(x, y, scheme="automatic_differentiation:rev-over-fwd")

2. Global default — set via :func:`jno.setup` or via ``.jno.toml``::

       jno.setup(__file__, diff_type="forward", hessian_type="fwd-over-fwd")

   .. code-block:: toml

       [jno]
       diff_type    = "forward"        # first-order default
       hessian_type = "fwd-over-rev"   # second-order default

The plain string ``"automatic_differentiation"`` (no suffix) resolves to the
current global default. Defaults match historical behaviour: first-order
``reverse`` (was ``jax.jacobian`` = ``jacrev``); second-order ``fwd-over-rev``
(was ``jax.hessian`` = ``jacfwd ∘ jacrev``).
"""

from __future__ import annotations

import jax

from .schemes import require_family

_VALID_FIRST_ORDER = ("forward", "reverse")
_VALID_HESSIAN = ("fwd-over-rev", "fwd-over-fwd", "rev-over-rev", "rev-over-fwd")

_AD_MODE: str = "reverse"
_HESSIAN_MODE: str = "fwd-over-rev"


def set_ad_mode(mode: str) -> None:
    """Set the global default for first-order AD."""
    global _AD_MODE
    if mode not in _VALID_FIRST_ORDER:
        raise ValueError(f"Invalid AD mode {mode!r}; expected one of {_VALID_FIRST_ORDER}.")
    _AD_MODE = mode


def get_ad_mode() -> str:
    """Return the global default for first-order AD."""
    return _AD_MODE


def set_hessian_mode(mode: str) -> None:
    """Set the global default for second-order AD."""
    global _HESSIAN_MODE
    if mode not in _VALID_HESSIAN:
        raise ValueError(f"Invalid Hessian mode {mode!r}; expected one of {_VALID_HESSIAN}.")
    _HESSIAN_MODE = mode


def get_hessian_mode() -> str:
    """Return the global default for second-order AD."""
    return _HESSIAN_MODE


def parse_ad_scheme(scheme: str) -> str:
    """Resolve a first-order scheme string to ``"forward"`` or ``"reverse"``.

    Supported::

        "automatic_differentiation"          → global default (get_ad_mode())
        "automatic_differentiation:forward"  → "forward"
        "automatic_differentiation:reverse"  → "reverse"
    """
    # Check the FAMILY, not just the suffix: this used to discard the family half, so any scheme
    # without a colon -- including one belonging to another backend -- returned the AD default.
    sub = require_family(scheme, "automatic_differentiation")
    if not sub:
        return get_ad_mode()
    if sub in _VALID_FIRST_ORDER:
        return sub
    raise ValueError(f"Unknown AD scheme suffix {sub!r}; expected one of {_VALID_FIRST_ORDER}.")


def parse_hessian_scheme(scheme: str) -> tuple[str, str]:
    """Resolve a second-order scheme string to ``(outer, inner)`` AD modes.

    The result composes as ``outer(inner(f))``. E.g. ``("forward", "reverse")``
    means ``jax.jacfwd(jax.jacrev(f))`` — the historical ``jax.hessian`` path.

    Supported::

        "automatic_differentiation"                → global default (get_hessian_mode())
        "automatic_differentiation:fwd-over-rev"   → ("forward", "reverse")
        "automatic_differentiation:fwd-over-fwd"   → ("forward", "forward")
        "automatic_differentiation:rev-over-rev"   → ("reverse", "reverse")
        "automatic_differentiation:rev-over-fwd"   → ("reverse", "forward")

    First-order suffixes ``forward``/``reverse`` are accepted as shorthand for
    the matching same-mode composition (``forward`` → ``fwd-over-fwd``).
    """
    sub = require_family(scheme, "automatic_differentiation")
    if not sub:
        return _hessian_to_outer_inner(get_hessian_mode())
    if sub in _VALID_HESSIAN:
        return _hessian_to_outer_inner(sub)
    if sub in _VALID_FIRST_ORDER:
        return (sub, sub)
    raise ValueError(f"Unknown Hessian scheme suffix {sub!r}; expected one of {_VALID_HESSIAN}.")


def _hessian_to_outer_inner(mode: str) -> tuple[str, str]:
    mapping = {
        "fwd-over-rev": ("forward", "reverse"),
        "fwd-over-fwd": ("forward", "forward"),
        "rev-over-rev": ("reverse", "reverse"),
        "rev-over-fwd": ("reverse", "forward"),
    }
    if mode not in mapping:
        raise ValueError(mode)
    return mapping[mode]


def ad_fn(mode: str):
    """Return ``jax.jacfwd`` or ``jax.jacrev`` for the given mode."""
    if mode == "forward":
        return jax.jacfwd
    if mode == "reverse":
        return jax.jacrev
    raise ValueError(f"Invalid AD mode {mode!r}.")
