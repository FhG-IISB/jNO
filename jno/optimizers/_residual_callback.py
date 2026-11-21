"""The callback that runs a residual-based optimizer step inside ``jno.core.solve()``, and the two
step-length rules those optimizers use (the ENGD grid search and Armijo backtracking).

Based on:
  Müller, J., & Zeinhofer, M. (2023).
  Achieving High Accuracy with PINNs via Energy Natural Gradient Descent.
  ICML 2023, arXiv:2302.13163, Sec. 4.1 -- the grid line search over
  :math:`\\alpha \\in \\{0.5^0, \\ldots, 0.5^{30}\\}`.
  Armijo, L. (1966).
  Minimization of functions having Lipschitz continuous first partial derivatives.
  Pacific Journal of Mathematics, 16(1), 1-3 -- the sufficient-decrease backtracking.

A subclass implements ``init_state`` and ``direction``. ``direction`` sees the residual as a
function ``r_of(flat_p)`` of the model's flattened parameters and returns ``d`` such that the new
parameters are ``θ - d`` (``optax.sgd(1.0)`` applies it), the new state, and a dict of scalars
kept as ``last_info`` for inspection. The whole step is one ``jax.jit``-compiled program, compiled
once at ``on_solve_begin``.
"""

from __future__ import annotations

import jax
import jax.flatten_util
import jax.numpy as jnp

from jno.utils.adaptive.callbacks import Callback

from ._residuals import check_gram_terms, make_residual_fn


def half_sq_norm(r):
    """The Gauss-Newton objective :math:`\\tfrac12 \\|\\hat r\\|^2`."""
    return 0.5 * jnp.vdot(r, r)


def grid_line_search(obj, n: int = 31, dtype=jnp.float64):
    """``argmin`` of ``obj(alpha)`` over :math:`\\alpha \\in \\{0.5^0, \\ldots, 0.5^{n-1}\\}` and its value."""
    alphas = 0.5 ** jnp.arange(n, dtype=dtype)
    vals = jax.lax.map(obj, alphas)
    vals = jnp.where(jnp.isfinite(vals), vals, jnp.inf)
    i = jnp.argmin(vals)
    return alphas[i], vals[i]


def armijo(obj, f0, slope, *, c1: float = 1e-4, beta: float = 0.5, max_steps: int = 30, t0=1.0):
    """Backtrack ``t <- beta t`` from ``t0`` until ``obj(t) <= f0 + c1 t slope``.

    ``slope`` is the directional derivative along the search direction (negative for descent).
    Returns ``(t, obj(t))``; after ``max_steps`` halvings without sufficient decrease it returns
    ``t = 0`` (no step) rather than a step that increases the loss.
    """
    t0 = jnp.asarray(t0, dtype=jnp.result_type(f0))

    def cond(c):
        i, t, f = c
        return (i < max_steps) & ~(jnp.isfinite(f) & (f <= f0 + c1 * t * slope))

    def body(c):
        i, t, _ = c
        t = beta * t
        return i + 1, t, obj(t)

    i, t, f = jax.lax.while_loop(cond, body, (0, t0, obj(t0)))
    ok = jnp.isfinite(f) & (f <= f0 + c1 * t * slope)
    return jnp.where(ok, t, 0.0), jnp.where(ok, f, f0)


class ResidualStepCallback(Callback):
    """Replace the gradient of model ``lid`` by a residual-based step (see the module docstring)."""

    who = "residual optimizer"

    def __init__(self, gram_terms):
        self._lid = check_gram_terms(gram_terms, self.who)
        self._gram_terms = list(gram_terms)
        self._step_jit = None
        self._state = None
        self._unravel = None
        self.last_info: dict = {}

    # -- subclass interface ---------------------------------------------------------------------
    def init_state(self, flat_p, m: int):
        """Initial optimizer state (any pytree of arrays); reset at every ``solve()``."""
        return ()

    def check_sizes(self, m: int, P: int) -> None:
        """Raise when the method cannot handle ``m`` residual rows and ``P`` parameters."""

    def direction(self, r_of, flat_p, state):
        """Return ``(d, new_state, info)``: the new parameters are ``flat_p - d``."""
        raise NotImplementedError

    # -- Callback hooks -------------------------------------------------------------------------
    # docstring-from: jno.utils.adaptive.callbacks.Callback.on_solve_begin
    def on_solve_begin(self, **kwargs) -> None:
        """Called once after ``solve()`` finishes setup, before the training loop.

        Keyword Args:
            compiled_constraints_fn: Combined compiled JAX function for all constraints.
            n_constraints (int): Number of constraint terms.
            batchsize: Mini-batch size (``None`` for full-batch).
            frozen: Frozen parameter pytree (from ``eqx.partition``).
            static: Static (non-array) pytree.
            trainable: Initial trainable parameter pytree (use for pre-compilation only).
            context: Domain context (use for pre-compilation only).
            rng: PRNG key.
            min_consecutive (int): Minimum consecutive time steps per constraint call.
            constraint_exprs: List of the solver's raw constraint placeholders
                (the exact Python objects passed to ``jno.core([...])``).
                Callbacks that take a ``constraints=`` arg validate user input
                against this list by Python identity.
            inequality_idx (list[int]): Rows of ``individual_losses`` that are constraints
                (``jno.le`` / ``jno.ge``, feasible when ``<= 0``, and ``jno.eq``). Empty without
                constraints.
            equality_idx (list[int]): The ``jno.eq`` rows among them, feasible when ``== 0``.
            all_ops: List of OperationDef instances (the solver's compiled op set);
                callbacks may pass this to ``TraceCompiler.compile_multi_expression``
                to evaluate user-supplied placeholder expressions during training.
            domain: The solver's :class:`~jno.domain.domain` instance (read-only;
                use only for shape / metadata inspection during pre-compilation).
        """
        if "all_ops" not in kwargs:
            raise RuntimeError(f"{self.who} requires the solver to pass `all_ops` via on_solve_begin.")
        trainable = kwargs["trainable"]
        if self._lid not in trainable:
            raise ValueError(
                f"{self.who}: model with layer_id={self._lid} not found in trainable "
                f"(keys: {list(trainable.keys())}). The model in gram_terms must be trainable in this solve."
            )
        flat0, self._unravel = jax.flatten_util.ravel_pytree(trainable[self._lid])
        residual_fn = make_residual_fn(
            self._gram_terms,
            kwargs["all_ops"],
            kwargs["frozen"],
            kwargs["static"],
            self._lid,
            self._unravel,
            kwargs.get("min_consecutive", 1),
        )
        context, rng = kwargs["context"], kwargs["rng"]
        m = int(jax.eval_shape(residual_fn, flat0, trainable, context, rng).shape[0])
        self.m, self.P = m, int(flat0.shape[0])
        self.check_sizes(self.m, self.P)

        def step(flat_p, trainable, context, rng, state):
            def r_of(q):
                return residual_fn(q, trainable, context, rng)

            d, new_state, info = self.direction(r_of, flat_p, state)
            bad = ~jnp.all(jnp.isfinite(d))
            info = {**info, "nonfinite": bad}
            return jnp.where(bad, jnp.zeros_like(d), d), new_state, info

        self._step_jit = jax.jit(step)
        self._state = self.init_state(flat0, m)
        out = self._step_jit(flat0, trainable, context, rng, self._state)  # compile once, discard
        jax.block_until_ready(out)

    # docstring-from: jno.utils.adaptive.callbacks.Callback.on_before_update
    def on_before_update(self, *, grads, trainable, context, rng, epoch, **kwargs):
        """Called between gradient computation and the optimizer update.

        Subclasses may return a modified ``grads`` dict to redirect the
        optimizer (e.g. to apply a preconditioner), or ``None`` to leave the
        gradients unchanged.  The returned dict must have the same pytree
        structure as ``grads``.

        Called only when the solver detects that at least one active callback
        overrides this method (it is a no-op in the base class).  Requires
        ``inner_steps=1`` and no Bayesian models.

        Keyword Args:
            grads: Parameter gradient pytree ``{lid: model_params}``.
            trainable: Current trainable parameter pytree.
            context: Domain context (current batch).
            rng: Current JAX PRNG key (Python-side; not consumed here).
            epoch (int): Current outer epoch (Python integer, 0-indexed).
            total_loss: The scalar loss at this step, as descended.
            individual_losses: Per-entry values, one per ``jno.core`` constraint entry --
                **including** any ``jno.le``/``jno.ge`` inequality constraints, which are
                evaluated with everything else but held out of ``total_loss``. A constrained
                optimiser reads its ``g_j`` here rather than recomputing them.

        Returns:
            Modified ``grads`` dict, or ``None`` to leave unchanged.
        """
        if self._step_jit is None:
            return None
        flat_p, _ = jax.flatten_util.ravel_pytree(trainable[self._lid])
        d, self._state, self.last_info = self._step_jit(flat_p, trainable, context, rng, self._state)
        return {**grads, self._lid: self._unravel(d)}
