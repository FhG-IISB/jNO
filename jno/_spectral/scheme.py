"""``jno.sp(...)``: options of a spectral derivative, on a binding or one operator, and the node-set helpers.

``jno.sp`` rides on ``scheme=`` as ``jno.fd(...)`` does (a ``str``, family ``"spectral_basis"``)::

    w = W.bind(x=x, y=y, scheme=jno.sp(dealias=3/2))   # the equations built from w are evaluated on a padded grid
    xg, yg, _ = dom.variable("interior", nodes=jno.sp.gauss(y=31), split=True)

A spectral unknown's derivatives default to this family, so ``scheme=`` is only needed for options.

Based on:
  Orszag, S. A. (1971). On the elimination of aliasing in finite-difference schemes by filtering high-wavenumber
  components. J. Atmos. Sci. 28, 1074 -- the 3/2 rule.
"""

from __future__ import annotations

from .domain import NodeSpec

__all__ = ["SPScheme", "sp"]

FAMILY = "spectral_basis"


class SPScheme(str):
    """A spectral derivative scheme (see :func:`sp`). A ``str`` so it rides on ``scheme=`` like a stencil."""

    def __new__(cls, *, dealias=None):
        if dealias is not None:
            dealias = float(dealias)
            if not dealias >= 1.0:
                raise ValueError(f"jno.sp(dealias={dealias}): the padding factor is >= 1 (3/2 for quadratic terms).")
        parts = [f"dealias={dealias:g}"] if dealias is not None else []
        s = str.__new__(cls, FAMILY + (":" + ",".join(parts) if parts else ""))
        s.dealias = dealias
        return s


def sp(*, dealias=None) -> SPScheme:
    """The spectral scheme with options.

    Args:
        dealias: evaluate every equation built from this binding on a grid padded by this factor per axis, then
            project back: ``3/2`` removes the aliasing of quadratic products, ``2`` of cubic ones.
    """
    return SPScheme(dealias=dealias)


def _nodes(rule):
    def make(*, of=None, **orders):
        return NodeSpec(rule, of=of, **orders)

    make.__name__ = rule
    make.__doc__ = (
        f"The {rule} nodes on the named axes (``x=31``), the other axes as the base space has them; ``of=U`` picks "
        "the base (default: the domain's first spectral unknown)."
    )
    return make


sp.gauss = _nodes("gauss")
sp.lobatto = _nodes("lobatto")
sp.uniform = _nodes("uniform")
