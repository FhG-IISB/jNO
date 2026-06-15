"""Optimizer-coverage tests for jno.core parameter identification.

Each test assigns a distinct optimizer to a trainable scalar parameter
and verifies convergence on an independent 1-D linear-regression problem:

    p_i * f_i(x) ≈ truth_i * f_i(x)

The residuals are decoupled by construction — parameter i only contributes
to residual i — so each optimizer trains in isolation even though all
parameters live in the same solve() call.
"""

from __future__ import annotations

import jax
import optax

import jno

π = jno.np.pi

# Truths and basis functions for 7 independent sub-problems.
_TRUTHS = [3.14, -2.71, 42.0, 1.5, -0.8, 7.0, -3.0]


def _basis_fns(x):
    return [
        jno.np.sin(π * x),
        jno.np.cos(π * x),
        x * (1 - x),
        jno.np.sin(2 * π * x),
        jno.np.cos(2 * π * x),
        x**2,
        x,
    ]


def _build_problem(optimizers, *, epochs=3000, mesh_size=0.05, seed=42):
    """Train one parameter per optimizer on an independent linear residual.

    Returns (recovered_values, truths) for the parameters that match the
    provided optimizer list (first len(optimizers) parameters).
    """
    n = len(optimizers)
    assert n <= len(_TRUTHS)

    domain = jno.domain.line(mesh_size=mesh_size)
    x, _ = domain.variable("interior")
    bases = _basis_fns(x)

    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    params = [jno.np.parameter((1,), key=k, name=f"p{i}") for i, k in enumerate(keys)]

    for param, opt in zip(params, optimizers):
        param.optimizer(opt)

    residuals = [(p * f - t * f).mse for p, f, t in zip(params, bases[:n], _TRUTHS[:n])]
    crux = jno.core(residuals)
    crux.solve(epochs)

    recovered = crux.eval(params)
    return [float(v[0]) for v in recovered], _TRUTHS[:n]


# ---------------------------------------------------------------------------
# Single-optimizer smoke tests — quick, one parameter each
# ---------------------------------------------------------------------------


class TestSingleOptimizerConvergence:
    """Each optimizer individually converges on a single-parameter problem."""

    def _single(self, opt, truth_idx=0, epochs=1000):
        domain = jno.domain.line(mesh_size=0.05)
        x, _ = domain.variable("interior")
        bases = _basis_fns(x)
        k = jax.random.PRNGKey(0)
        p = jno.np.parameter((1,), key=k, name="p")
        p.optimizer(opt)
        crux = jno.core([(p * bases[truth_idx] - _TRUTHS[truth_idx] * bases[truth_idx]).mse])
        crux.solve(epochs)
        # crux.eval returns the array directly (not a list) for a single expression.
        val = crux.eval([p])
        return float(val[0]), _TRUTHS[truth_idx]

    def test_adam(self):
        got, truth = self._single(optax.adam(1e-2))
        assert abs(got - truth) / abs(truth) < 0.05

    def test_adamw(self):
        got, truth = self._single(optax.adamw(1e-2))
        assert abs(got - truth) / abs(truth) < 0.05

    def test_sgd(self):
        got, truth = self._single(optax.sgd(1e-1), truth_idx=0)
        assert abs(got - truth) / abs(truth) < 0.05

    def test_rmsprop(self):
        got, truth = self._single(optax.rmsprop(1e-2))
        assert abs(got - truth) / abs(truth) < 0.05

    def test_adagrad(self):
        got, truth = self._single(optax.adagrad(1e-1))
        assert abs(got - truth) / abs(truth) < 0.05

    def test_lion(self):
        got, truth = self._single(optax.lion(5e-3), epochs=2000)
        assert abs(got - truth) / abs(truth) < 0.05

    def test_lbfgs(self):
        # L-BFGS converges in a handful of steps on a smooth quadratic.
        got, truth = self._single(optax.lbfgs, epochs=200)
        assert abs(got - truth) / abs(truth) < 0.01

    def test_adan(self):
        got, truth = self._single(optax.adan(1e-2))
        assert abs(got - truth) / abs(truth) < 0.05

    def test_nadam(self):
        got, truth = self._single(optax.nadam(1e-2))
        assert abs(got - truth) / abs(truth) < 0.05

    def test_radam(self):
        got, truth = self._single(optax.radam(1e-2), epochs=3000)
        assert abs(got - truth) / abs(truth) < 0.05


# ---------------------------------------------------------------------------
# Multi-optimizer solve — all 7 run in a single crux.solve() call
# ---------------------------------------------------------------------------


class TestMultiOptimizerInverseProblem:
    """Seven parameters, seven different optimizers, one crux.solve() call."""

    def test_seven_optimizers_converge(self):
        optimizers = [
            optax.adam(1e-2),
            optax.adamw(1e-2),
            optax.sgd(5e-1),
            optax.rmsprop(1e-2),
            optax.adagrad(1e-1),
            optax.lion(5e-3),
            optax.lbfgs,
        ]
        recovered, truths = _build_problem(optimizers, epochs=3000)
        for got, truth, opt in zip(recovered, truths, optimizers):
            rel_err = abs(got - truth) / (abs(truth) + 1e-8)
            assert rel_err < 0.1, f"optimizer {opt} failed: got {got:.4f}, truth {truth:.4f}, rel_err={rel_err:.3e}"
