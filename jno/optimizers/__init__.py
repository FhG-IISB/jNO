"""``jno.optimizers`` — custom optax-compatible optimizers that aren't in optax.

Just the optimizers — for everything else (``chain``, learning-rate schedules, gradient clipping, …)
use ``optax`` directly. Each optimizer here is an ``optax`` ``GradientTransformation``, so it composes
with ``optax.chain`` and drops straight into ``model.optimizer(...)`` /
``jno.np.parameter(...).optimizer(...)``::

    import optax, jno
    k.optimizer(jno.optimizers.ssbroyden())                              # use it directly
    k.optimizer(optax.chain(optax.clip_by_global_norm(1.0),              # …or inside an optax chain
                            jno.optimizers.ssbfgs()))

- :func:`ssbroyden` / :func:`ssbfgs` — Self-Scaled Broyden / BFGS quasi-Newton with a line search
  (Urbán, Stefanou & Pons, *J. Comput. Phys.* **523** (2025) 113656); excellent on smooth
  PINN / inverse losses.
- :func:`soap` — SOAP, Shampoo with Adam in the preconditioner's eigenbasis
  (Vyas et al. 2024, arXiv:2409.11321).
- :func:`md` — Magnitude–Direction Decoupling: a generic wrapper that factorizes each weight
  matrix into a fixed-norm direction + learnable magnitude gains and steps any optax base optimizer
  on the direction (Hägele et al. 2026, arXiv:2606.25971). Pass it as a sentinel via
  ``net.optimizer(jno.optimizers.md(optax.adam(1e-3)))``; :func:`md_decouple` is the bare transform.

The ``scale_by_*`` variants are the bare transforms (no line search / learning-rate wrapper) for
building your own optax chains.

**Add a custom optimizer:** drop an optax-compatible ``GradientTransformation`` into a new module in
this package and re-export it below.
"""

from .md_decouple import MDOptimizer, md, md_decouple
from .soap import scale_by_soap, soap
from .ssbroyden import scale_by_ss_quasi_newton, ssbfgs, ssbroyden

__all__ = [
    "ssbroyden",
    "ssbfgs",
    "scale_by_ss_quasi_newton",
    "soap",
    "scale_by_soap",
    "md",
    "md_decouple",
    "MDOptimizer",
]
