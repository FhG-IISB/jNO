"""Releasing the factors of extra cuDSS slots between programs (the in-program LDL^T path).

A form solved twice in one program gets a second slot (each solve keeps its factors for its own adjoint). After
the program those factors are dead weight; `release_idle_ldl` frees them, the slot id stays valid, and the next
program that uses it re-runs the analysis and gives the same answer.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D box meshing")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402
import jno.utils.solver.cudss_sym as cs  # noqa: E402
import jno.utils.solver.linear as L  # noqa: E402

pytestmark = pytest.mark.skipif(not cs.available(), reason="needs a GPU with the cuDSS extension (nvcc, nvidia-cudss)")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _stats(plan):
    return plan.stats()  # {} when the slot holds no factors


def test_release_frees_the_extra_slots_and_they_still_work():
    from tests.test_solve_cudss_ldl import _form

    fem, nc = _form()
    lu = jno.solve.lu(backend="cudss")
    f = lambda k: jnp.sum(jnp.abs(fem.solve(linear=lu, k=k)) ** 2)  # noqa: E731
    two = jax.jit(lambda k: f(k) + f(0.7 * k))  # the same form twice: two slots
    k0 = jnp.asarray(np.random.default_rng(0).uniform(0.5, 1.5, nc))
    v1 = float(two(k0))
    pools = [p for p in L._LDL_PLANS.values() if len(p) >= 2]
    assert pools, "expected a second slot for the second solve in one program"
    extra = pools[0][1]["plan"]
    assert _stats(extra), "the extra slot should hold factors after the program"
    assert L.release_idle_ldl() >= 1
    assert not _stats(extra), "release left the factors in place"
    assert _stats(pools[0][0]["plan"]), "the first slot keeps its factors (it is refactored in place)"
    v2 = float(two(k0))  # the same compiled program, re-analysed
    np.testing.assert_allclose(v2, v1, rtol=1e-12)


def test_core_releases_between_programs():
    """jno.core.eval runs one program per expression and frees the extra slots after each"""
    from tests.test_solve_cudss_ldl import _form

    fem, nc = _form()
    A, k = fem._probe["A"], fem._probe["k"]
    k.initialize(jax.nn.initializers.constant(1.0))
    lu = jno.solve.lu(backend="cudss")
    a1 = jno.np.inner(A, A).integrate(fem, solver=lu)
    F = jno.fn(lambda i: jnp.real(i) + jnp.imag(i), [a1], name="F")
    crux = jno.core([F], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    crux.eval([F])
    held = [e["plan"] for p in L._LDL_PLANS.values() for e in p[1:] if _stats(e["plan"])]
    assert not held, f"{len(held)} extra slot(s) still hold factors after jno.core.eval"


def test_an_eager_solve_frees_its_factors():
    """an eager solve (concrete values, no program) frees its slot's factors once it has returned: nothing can come
    back for them, and kept they pile up, one factorisation per form solved; a later solve re-runs the analysis"""
    from tests.test_solve_cudss_ldl import _form

    fem, nc = _form()
    lu = jno.solve.lu(backend="cudss")
    k0 = jnp.asarray(np.random.default_rng(1).uniform(0.5, 1.5, nc))
    before = set(L._LDL_PLANS)
    x1 = np.asarray(fem.solve(linear=lu, k=k0))
    new = [L._LDL_PLANS[key] for key in L._LDL_PLANS if key not in before]
    if not new:
        pytest.skip("this solve ran inside a compiled program, not eagerly")
    assert all(not _stats(e["plan"]) for pool in new for e in pool), "an eager solve kept its factors"
    x2 = np.asarray(fem.solve(linear=lu, k=k0))  # the same slot, analysed again
    np.testing.assert_allclose(x2, x1, rtol=1e-12, atol=1e-14 * np.abs(x1).max())
    ref = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"), k=k0))
    np.testing.assert_allclose(x1, ref, rtol=1e-9, atol=1e-11 * np.abs(ref).max())
