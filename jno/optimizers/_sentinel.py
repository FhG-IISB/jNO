"""The common base of every optimizer sentinel whose step needs the per-point residuals.

Based on:
  Müller, J., & Zeinhofer, M. (2023).
  Achieving High Accuracy with PINNs via Energy Natural Gradient Descent.
  ICML 2023, arXiv:2302.13163 -- ENGD, the first of these methods in jNO, whose
  ``gram_terms`` auto-detection this base generalises.

An optax ``update()`` only ever sees the gradient of the summed loss, but Gauss-Newton-type methods
need the residual vector and its Jacobian. Such an optimizer is therefore a *sentinel*: ``jno.core``
sees it at ``solve()`` time, builds ``gram_terms`` (one ``(residual.grad(model), 1.0)`` pair per
loss term involving the model, unless given), asks the sentinel for the callback that computes the
step, and swaps the sentinel for ``optax.sgd(1.0)``, which applies that step unchanged.
"""

from __future__ import annotations


class ResidualOptimizer:
    """Base class: subclasses set ``_who`` and implement :meth:`build_callback`."""

    _who = "jno.optimizers.<residual optimizer>"

    def __init__(self, gram_terms=None):
        self._gram_terms = gram_terms

    def build_callback(self, gram_terms):
        """Return the ``Callback`` whose ``on_before_update`` replaces the gradient by the step."""
        raise NotImplementedError

    @property
    def _factory_name(self) -> str:
        return self._who.rsplit(".", 1)[-1]
