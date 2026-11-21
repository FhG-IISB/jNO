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
- :func:`ss_esoap` — SS-ESOAP, SOAP with a secant-energy step scale and adaptive eigenbasis refreshes
  (Wang et al. 2026, arXiv:2608.29448); :func:`scale_by_ss_esoap` is the bare transform.
- :func:`score` — SCORE, self-scaled Broyden with a decrement-based step and a shifted secant pair
  (Si, An, Ma & Yan 2026, arXiv:2608.04206).
- :func:`nncg` — NysNewton-CG, damped Newton by Nyström-preconditioned CG with Armijo steps, a
  fine-tuning stage after Adam / L-BFGS (Rathore et al., ICML 2024, arXiv:2402.01868).
- :func:`mma` — Method of Moving Asymptotes (Svanberg, *IJNME* **24**(2), 1987), the standard
  optimiser for *constrained* structural design. Unlike everything else here it is a **sentinel**,
  not a ``GradientTransformation``: an optax ``update()`` never sees the individual constraint
  values and gradients MMA needs. Pair it with :func:`jno.le` / :func:`jno.ge`.
- :func:`gcmma` — globally convergent MMA (Svanberg, *SIAM J. Optim.* **12**(2), 2002): the same
  sentinel, but each step is re-solved until its approximation is conservative, so from a feasible
  start every design is feasible and the objective never rises.
- Not an optimiser but its partner: ``jno.le(g, b).auglag()`` (and ``jno.eq(g, b).auglag()``)
  enforces a constraint with ANY of these through an augmented Lagrangian
  (:mod:`jno.optimizers.auglag`).
- :func:`bounded` — box bounds for any optax optimiser: each step is projected back onto
  ``[lower, upper]`` (projected gradient). A sentinel, like :func:`md`, so the projection comes
  after jNO's learning-rate scale.
- :func:`md` — Magnitude–Direction Decoupling: a generic wrapper that factorizes each weight
  matrix into a fixed-norm direction + learnable magnitude gains and steps any optax base optimizer
  on the direction (Hägele et al. 2026, arXiv:2606.25971). Pass it as a sentinel via
  ``net.optimizer(jno.optimizers.md(optax.adam(1e-3)))``; :func:`md_decouple` is the bare transform.

Gauss-Newton family for PINN residuals: sentinels like :func:`engd`, which read the per-point
residuals of every loss term involving the model (``gram_terms``) instead of only the summed gradient:

- :func:`engd` — energy natural gradient (Müller & Zeinhofer, ICML 2023, arXiv:2302.13163), with
  SPRING momentum and fixed damping.
- :func:`lm` — Levenberg-Marquardt, damping adapted from the gain ratio (Levenberg 1944,
  Marquardt 1963; Madsen, Nielsen & Tingleff 2004).
- :func:`dngd` — Dual Natural Gradient Descent, the Gauss-Newton step solved over the ``m``
  residual rows, with geodesic acceleration (Jnini & Vella 2025, arXiv:2505.21404); dense or
  matrix-free with Nyström-preconditioned CG.
- :func:`nystrom_ngd` — matrix-free natural gradient, Nyström-preconditioned CG and Armijo steps
  (Bioli, Marcati & Sangalli, CMAME 2026, doi:10.1016/j.cma.2026.119456).
- :func:`dsgnar` — doubly-sketched Gauss-Newton, step length from a target gain ratio
  (Webb, Jerad & Cartis 2026, arXiv:2607.02194).

The ``scale_by_*`` variants are the bare transforms (no line search / learning-rate wrapper) for
building your own optax chains.

**Add a custom optimizer:** drop an optax-compatible ``GradientTransformation`` into a new module in
this package and re-export it below.
"""

from .bounded import BoundedOptimizer, bounded
from .continuation import (
    GeometricDecay,
    HeavisideContinuation,
    SIMPContinuation,
    geometric_decay,
    heaviside_continuation,
    simp_continuation,
)
from .dngd import DNGDOptimizer, dngd
from .dsgnar import DSGNAROptimizer, dsgnar
from .engd import ENGDOptimizer, engd
from .lm import LMOptimizer, lm
from .md_decouple import MDOptimizer, md, md_decouple
from .mma import GCMMAOptimizer, MMACallback, MMAOptimizer, gcmma, mma
from .nncg import nncg
from .nystrom_ngd import NystromNGDOptimizer, nystrom_ngd
from .score import score
from .soap import scale_by_soap, soap
from .ss_esoap import scale_by_ss_esoap, ss_esoap
from .ssbroyden import scale_by_ss_quasi_newton, ssbfgs, ssbroyden

__all__ = [
    "bounded",
    "BoundedOptimizer",
    "engd",
    "ENGDOptimizer",
    "dngd",
    "DNGDOptimizer",
    "dsgnar",
    "DSGNAROptimizer",
    "lm",
    "LMOptimizer",
    "nystrom_ngd",
    "NystromNGDOptimizer",
    "mma",
    "gcmma",
    "GCMMAOptimizer",
    "simp_continuation",
    "SIMPContinuation",
    "geometric_decay",
    "GeometricDecay",
    "heaviside_continuation",
    "HeavisideContinuation",
    "MMAOptimizer",
    "ssbroyden",
    "ssbfgs",
    "scale_by_ss_quasi_newton",
    "nncg",
    "score",
    "soap",
    "scale_by_soap",
    "ss_esoap",
    "scale_by_ss_esoap",
    "md",
    "md_decouple",
    "MDOptimizer",
]
