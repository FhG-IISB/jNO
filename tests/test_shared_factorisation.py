"""Forms that differ only in their load share one factorisation (one block of right-hand sides) inside an
evaluation -- jNO recognises them by a structural key of the operator, with no API.

Oracles: the same quantities solved form by form (host LU, and the unshared cuDSS path) -- values and the
gradient of the evaluated expression; and a forced wrong grouping must be refused by the residual check.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D box meshing")
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402
import jno.utils.solver.cudss_sym as cs  # noqa: E402

pytestmark = pytest.mark.skipif(not cs.available(), reason="needs a GPU with the cuDSS extension (nvcc, nvidia-cudss)")
inner = jno.np.inner
LU = jno.solve.lu(backend="cudss")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _drives(other_operator=False):
    """two complex eddy forms on one box that differ only in their load (or, with `other_operator`, also in
    the operator); the readout is the cross energy of their frozen solutions"""
    d = jno.shape.box(0, 0, 0, 1, 1, 1).sized(0.35).domain()
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    k = jno.np.parameter(s, name="k")
    k.dtype(jnp.float64)
    k.initialize(lambda key, shape, dtype=None: jnp.asarray(np.random.default_rng(0).uniform(0.5, 1.5, shape)))
    A, V = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    curl = lambda f: f.vector.curl(x, y, z)  # noqa: E731
    bc = u.vector.cross(d.variable("boundary", normals=True))
    loads = [jno.np.vector(1.0 + 0.0 * x, y, 0.0 * x), jno.np.vector(z, 0.0 * x, 1.0 - x)]
    fems = []
    for i, J in enumerate(loads):
        mass = (0.5 + 2j) * k if not (other_operator and i == 1) else (0.7 + 2j) * k
        fems.append(jno.fem([inner(curl(u), curl(v)) + mass * inner(A, V) - inner(J, V), bc]))
    Ak = [u.bind(x=x, y=y, z=z).freeze(f.solve(linear=LU)) for f in fems]
    cross = jnp.real  # noqa: F841
    node = (k * inner(Ak[0], Ak[1])).integrate(d)
    F = jno.fn(lambda i: jnp.real(i) + 0.3 * jnp.imag(i), [node], name="F")
    crux = jno.core([F], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    return d, fems, F, crux, k


def _spy(monkeypatch):
    """record the right-hand-side shape of every in-program LDL^T solve"""
    import jno.utils.solver.linear as L

    calls = []
    orig = L.cudss_ldl_solve

    def spy(A, b, context=None):
        calls.append(tuple(np.shape(b)))
        return orig(A, b, context)

    monkeypatch.setattr(L, "cudss_ldl_solve", spy)
    return calls


def test_two_drives_share_one_block_solve(monkeypatch):
    d, fems, F, crux, k = _drives()
    calls = _spy(monkeypatch)
    got = float(np.asarray(crux.eval([F])).reshape(-1)[0])
    assert len(calls) == 1 and len(calls[0]) == 2 and calls[0][1] == 2, f"expected ONE block solve of both drives: {calls}"
    # oracle: host LU, form by form
    kv = jnp.asarray(np.random.default_rng(0).uniform(0.5, 1.5, len(d.mesh.cells_dict["tetra"])))
    a = [np.asarray(f.solve(linear=jno.solve.lu(backend="host"), k=kv)) for f in fems]
    c = d.variable("interior", split=True)
    u2, _v2 = d.fem_symbols(value_shape=(3,), names=("u2", "v2"), space="N1E")
    B = [u2.bind(x=c[0], y=c[1], z=c[2]).freeze(ai) for ai in a]
    k2 = jno.np.parameter(d.fem_symbols(space="P0", names=("r2", "s2"))[1], name="k2")
    k2.dtype(jnp.float64)
    k2.initialize(lambda key, shape, dtype=None: kv)
    n2 = (k2 * inner(B[0], B[1])).integrate(d)
    F2 = jno.fn(lambda i: jnp.real(i) + 0.3 * jnp.imag(i), [n2], name="F2")
    want = float(np.asarray(jno.core([F2], domain=jno.domain.from_array({"_": np.zeros((1, 1))})).eval([F2])).reshape(-1)[0])
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_the_gradient_is_that_of_the_separate_solves(monkeypatch):
    """d(F)/d(k) through the shared block solve equals the one through two separate solves"""
    from jno.trace_compiler import TraceCompiler
    import jno.utils.solver.shared_solve as S

    d, fems, F, crux, k = _drives()
    fn = TraceCompiler.compile_traced_expression(F, crux.all_ops)
    models = eqx.tree_inference(crux._unwrapped_models)
    ctx = crux.domain_data.context
    params, static = eqx.partition(models, eqx.is_inexact_array)
    grad = lambda: jax.grad(lambda p: jnp.sum(fn(eqx.combine(p, static), ctx)))(params)  # noqa: E731
    g_shared = jax.tree_util.tree_leaves(grad())
    monkeypatch.setattr(S, "solve_shared", lambda *a, **kw: False)
    g_alone = jax.tree_util.tree_leaves(grad())
    for gs, ga in zip(g_shared, g_alone):
        np.testing.assert_allclose(np.asarray(gs), np.asarray(ga), rtol=1e-8, atol=1e-12 * float(np.abs(ga).max() + 1e-300))
    assert any(float(np.abs(np.asarray(ga)).max()) > 0 for ga in g_alone)


def test_a_wrong_grouping_is_refused(monkeypatch):
    """forms whose operators differ, forced into one group: the member checked against its own matrix fails"""
    import jno._fem as F_

    d, fems, F, crux, k = _drives(other_operator=True)
    monkeypatch.setattr(F_.FEM, "_operator_key", lambda self: ("forced",))
    with pytest.raises(RuntimeError, match="shared factorisation"):
        crux.eval([F])
        from jno.utils.solver.solver_api import raise_if_gate_failed

        raise_if_gate_failed()


def test_different_operators_get_different_keys():
    d, fems, F, crux, k = _drives(other_operator=True)
    assert fems[0]._operator_key() != fems[1]._operator_key()
    d2, fems2, *_ = _drives()
    assert fems2[0]._operator_key() == fems2[1]._operator_key()
