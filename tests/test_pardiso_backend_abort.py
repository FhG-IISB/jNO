"""``lu(backend="pardiso")`` must not kill the process when the operator goes non-finite.

MKL PARDISO has no guard against a NaN/Inf system: handed one it corrupts its own heap and the
process dies with SIGABRT (glibc "double free or corruption", exit 134). Nothing can catch that, and
the crash surfaces far from its cause -- glibc notices at an unrelated `free`, inside JAX's callback
teardown, so the report names numpy and the NaN is nowhere in it.

Every OTHER backend already fails properly on the same system: `host` (SuperLU) raises out of
`scipy.splu`. PARDISO was the one that took the interpreter with it. `_pardiso_host_solve` now
checks the values it is about to hand to MKL and raises `FloatingPointError` naming the count of
non-finite entries -- the last point at which anything can still raise, since everything after it is
inside MKL.

HOW IT WAS FOUND, because the search was long and the trail is worth keeping: the abort needed BOTH
a filtered SIMP density and the ADJOINT, which made it look structural. It was not. Ruled out by
experiment: the phase-22 refactorization reuse (a full plan rebuild still aborts), sparsity drift
(nnz/indptr/indices identical at every refactorize), a GC double-free (`PyPardisoSolver` has no
`__del__`), the matrix type flipping (`mtype=-2` throughout, in both the crashing and the surviving
run), the transpose flag on a symmetric mtype, and dtype/layout mismatch. What finally showed it was
instrumenting the matrix itself: the diagonal read 2.0e-3, then 4.8e-1, then **NaN**, and the next
free aborted. The structure was always fine; the VALUES were not.

Note the two genuine defects found on the way and fixed separately, neither of which was this one:
MKL's handle was being driven from a different pool thread on nearly every call, and the solution
array handed back to JAX was a view onto pypardiso's reused buffer rather than a copy.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.slow

REPRO = textwrap.dedent(
    """
    import jax; jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp, numpy as np, jno
    inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
    E0, EMIN, NU, PENAL, VOLFRAC = 1.0, 1e-4, 0.3, 3.0, 0.3
    LAM, MU = E0*NU/((1+NU)*(1-2*NU)), E0/(2*(1+NU))
    USE_FILTER = __import__("os").environ["FILTER"] == "1"

    d = jno.Shape.box(0, 0, 0, 2, 1, 1, size=0.25).domain()
    u, phi = d.fem_symbols(value_shape=(3,))
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    xi, yi, zi = d.variable("interior", split=True)[:3]
    left = d.variable("left", where=lambda x, y, z: x < 1e-9, split=True)[:3]
    right = d.variable("right", where=lambda x, y, z: x > 2 - 1e-9, split=True)[:3]
    rho = jno.np.parameter(s, name="rho"); rho.dtype(jnp.float64)
    rho.initialize(jax.nn.initializers.constant(VOLFRAC))
    rho.optimizer(jno.optimizers.mma(move=0.2, lower=1e-3, upper=1.0))
    if USE_FILTER:
        rho.constrain(d.patch_filter())
    eu, ep = symgrad(u, [xi, yi, zi]), symgrad(phi, [xi, yi, zi])
    a = lambda p, q: LAM*trace(p)*trace(q) + 2*MU*inner(p, q, n_contract=2)
    E = lambda r: EMIN + r**PENAL*(E0-EMIN)
    fem = jno.fem([
        E(rho)*a(eu, ep),
        u(*left) - (0.0, 0.0, 0.0),
        -1.0*inner(jnp.array([0.0, -1.0, 0.0]), phi.bind(**dict(zip("xyz", right))), n_contract=1),
    ], quad_degree=2)
    C = (E(rho)*a(eu, eu)).integrate(fem, solver=jno.solve.lu(backend="pardiso"))
    crux = jno.core([C.name("C"), jno.le(rho.mean, VOLFRAC)],
                    domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    crux.solve(8)
    print("SURVIVED")
    """
)


def _run(filter_on: bool):
    env = dict(os.environ, FILTER="1" if filter_on else "0",
               CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    return subprocess.run([sys.executable, "-c", REPRO], env=env, capture_output=True, text=True, timeout=900)


def _skip_unless_available():
    from jno.utils.solver.linear import _pardiso_available

    if not _pardiso_available():
        pytest.skip("pypardiso (with the private phase hooks this backend drives) is not installed")


class TestTheBackendDoesNotKillTheProcess:
    def test_a_non_finite_operator_raises_instead_of_aborting(self):
        """The headline. Exit 134/-6 is SIGABRT and means the guard is gone."""
        _skip_unless_available()
        r = _run(filter_on=True)
        assert r.returncode not in (134, -6), (
            f"the process was ABORTED (exit {r.returncode}) rather than raising; MKL PARDISO was "
            f"handed a non-finite system again:\n{r.stderr[-600:]}"
        )
        assert "non-finite" in r.stderr, (
            f"expected the guard's FloatingPointError naming the non-finite entries, got:\n"
            f"{r.stderr[-800:]}"
        )

    def test_an_unfiltered_density_survives_repeated_adjoints(self):
        """The control. If this ever fails the reproduction has drifted and means nothing."""
        _skip_unless_available()
        r = _run(filter_on=False)
        assert r.returncode == 0, (
            f"the CONTROL aborted (exit {r.returncode}) -- the filter is no longer what "
            f"distinguishes the crash:\n{r.stderr[-600:]}"
        )
        assert "SURVIVED" in r.stdout
