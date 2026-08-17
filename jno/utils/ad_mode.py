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
    if ":" not in scheme:
        return get_ad_mode()
    _, sub = scheme.split(":", 1)
    sub = sub.strip()
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
    if ":" not in scheme:
        return _hessian_to_outer_inner(get_hessian_mode())
    _, sub = scheme.split(":", 1)
    sub = sub.strip()
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


def rowwise_jacobian(f, x, rows):
    """``(len(rows), P)`` reverse-mode Jacobian of ``f: x -> (N,)``, with no ``vmap``.

    ``jax.jacrev`` takes one ``vjp`` and then **vmaps the pullback** across the rows of the
    identity basis. That vmap is the problem, not the differentiation: a differentiable FEM
    solve bottoms out in ``jax.experimental.sparse.linalg.spsolve``, which has no batching
    rule, so ``jacrev`` raises ``NotImplementedError: Batching rule for 'spsolve' not
    implemented`` on any trace containing ``fem.solve()`` -- even when the output is a single
    scalar, because the basis still carries a leading axis. Forward mode is no escape either
    (``jacfwd`` hits the same wall at ``csr_matvec``); plain ``jax.grad`` is what works.

    This does what ``jacrev`` does minus the vmap: ONE ``jax.vjp``, so the forward pass and
    its residuals are computed once and shared, followed by one pullback per requested row.
    The rows are unrolled at trace time.

    ``rows`` is an explicit sequence of output indices, so a caller that needs only some rows
    -- MMA needs the inequality-constraint rows and not the objective -- pays for those only.

    Args:
        f: Callable from the pytree ``x`` to a 1-D array of length ``N``.
        x: The pytree to differentiate with respect to.
        rows: Output indices to build rows for, in the returned order.

    Returns:
        ``(len(rows), P)`` array, where ``P`` is the total size of the array leaves of ``x``
        flattened in ``jax.tree_util.tree_leaves`` order -- the same layout ``jacrev``
        produces once its per-leaf blocks are raveled and concatenated.
    """
    import jax.numpy as jnp

    rows = list(rows)
    y, pullback = jax.vjp(f, x)
    out = []
    for i in rows:
        (grad_tree,) = pullback(jnp.zeros(y.shape, y.dtype).at[i].set(1.0))
        leaves = jax.tree_util.tree_leaves(grad_tree)
        out.append(jnp.concatenate([leaf.ravel() for leaf in leaves]))
    return jnp.stack(out)
