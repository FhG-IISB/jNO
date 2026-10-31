"""What a direct solve means when nobody names a backend, and what the log says about it.

``jno.solve.lu()`` and a parametric form with no ``linear=`` are cuDSS on a GPU that has it, JAX's ``spsolve`` elsewhere;
the log line names what ran. Oracle for the GPU route: ``spsolve`` itself, by name, on value and every derivative mode,
on a symmetric form (the in-program LDL^T) and a non-symmetric one (the LU), with batches of matrices, inside and outside
a jit: one cuDSS slot holds one factorisation, so a batch must not read the last member's factors.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for meshing")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402
import jno.utils.solver.cudss_sym as cs  # noqa: E402
from jno._fem import _route_line  # noqa: E402

inner = jno.np.inner
gpu = pytest.mark.skipif(not cs.available(), reason="needs a GPU with the cuDSS extension")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _form(symmetric=True, size=0.2):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    x, y = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    nc = len(d.mesh.cells_dict["triangle"])
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    k = jno.np.parameter(s, name="k")
    k.dtype(jnp.float64)
    u, v = d.fem_symbols(names=("u", "v"))
    U, V = u.bind(x=x, y=y), v.bind(x=x, y=y)
    g = lambda f: jno.np.grad(f, [x, y])  # noqa: E731
    wind = 0.0 if symmetric else inner(jno.np.vector(0.7 + 0 * x, 0.3 + 0 * x), g(U)) * V
    return jno.fem([k * inner(g(U), g(V)) + wind - V, u(xb, yb) - 0.0]), nc


def test_the_log_names_the_solver_of_a_parametric_form():
    """it said Jacobi-BiCGStab while the solve ran a sparse-direct factorisation"""
    fem, _nc = _form()
    assert "bicgstab" not in _route_line(fem)
    assert "lu" in _route_line(fem)
    line = _route_line(fem, linear=jno.solve.lu(backend="cudss"))
    assert "lu-cudss" in line and "jacobi" not in line


def test_a_backend_named_is_kept():
    for name, spec in (("lu-host", "host"), ("lu", "device"), ("lu-cudss", "cudss")):
        assert jno.solve.lu(backend=spec).name == name
    with pytest.raises(ValueError, match="host="):
        jno.solve.lu(backend="device", host=True)


def test_no_cudss_on_the_host_scope_or_without_float64():
    with jax.default_device(jax.devices("cpu")[0]):
        assert jno.solve.lu().name == "lu"
    jax.config.update("jax_enable_x64", False)
    assert jno.solve.lu().name == "lu"


@gpu
def test_on_a_gpu_nobody_asking_means_cudss(monkeypatch):
    """the default route reaches the in-program factorisation, and spsolve is not called"""
    import jno.utils.solver.linear as L
    from jno.utils.solver.solver_api import at_solve_spec

    assert at_solve_spec(jno.solve.lu()).name == "lu-cudss"  # decided where the solve runs: here, on the GPU
    fem, nc = _form()
    assert "lu-cudss" in _route_line(fem, linear=jno._fem._default_direct(fem))
    calls = []
    orig = L._sparse_lu_solve
    monkeypatch.setattr(L, "_sparse_lu_solve", lambda *a: calls.append(1) or orig(*a))
    c = jnp.asarray(np.random.default_rng(0).uniform(0.7, 1.3, nc))
    x = fem.solve(k=c)
    y = fem.solve(linear=jno.solve.lu(backend="host"), k=c)
    assert not calls, "the default solve went to spsolve"
    np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-9, atol=1e-14)


@gpu
@pytest.mark.parametrize("symmetric", [True, False], ids=["symmetric", "non-symmetric"])
def test_every_derivative_mode_is_that_of_spsolve(symmetric):
    fem, nc = _form(symmetric)
    rng = np.random.default_rng(1)
    c = jnp.asarray(rng.uniform(0.7, 1.3, nc))
    w = jnp.asarray(rng.normal(size=nc))
    batch = jnp.stack([c, 1.3 * c, 0.6 * c])

    def modes(linear):
        F = lambda kv: jnp.sum(fem.solve(linear=linear, k=kv) ** 2)  # noqa: E731
        G = lambda kv: fem.solve(linear=linear, k=kv)[:3]  # noqa: E731
        return {
            "value": F(c),
            "jit": jax.jit(F)(c),
            "grad": jax.grad(F)(c),
            "jit grad": jax.jit(jax.grad(F))(c),
            "jvp": jax.jvp(F, (c,), (w,))[1],
            "jacrev": jax.jacrev(G)(c),
            "jacfwd": jax.jacfwd(G)(c),
            "one tangent": jax.jacfwd(lambda t: G(c * t))(1.0),
            "hvp": jax.jvp(jax.grad(F), (c,), (w,))[1],
            "vmap": jax.vmap(F)(batch),
            "jit vmap": jax.jit(jax.vmap(F))(batch),
            "vmap jit": jax.vmap(jax.jit(F))(batch),
            "vmap grad": jax.vmap(jax.grad(F))(batch),
            "vmap jit grad": jax.vmap(jax.jit(jax.grad(F)))(batch),
        }

    want, got = modes(jno.solve.lu(backend="device")), modes(jno.solve.lu(backend="cudss"))
    # the members of a batch differ: a batch read off one factorisation could not pass
    assert float(jnp.abs(want["vmap"][0] - want["vmap"][1])) > 1e-3 * float(jnp.abs(want["vmap"][0]))
    for name in want:
        scale = float(jnp.abs(want[name]).max())
        np.testing.assert_allclose(np.asarray(got[name]), np.asarray(want[name]), atol=1e-9 * scale, err_msg=name)


def test_lu_picks_its_backend_where_the_solve_runs(monkeypatch):
    """Built under one default device, solved under another: the solve's device decides (CPU has no cuDSS handler)."""
    import jno.solve as S

    spec = jno.solve.lu()
    assert spec.name == "lu" and spec.traits["host_kernel"] is None  # the device's own solve until composed
    monkeypatch.setattr(S, "_preferred_direct_backend", lambda: "cudss")
    assert spec.at_solve().name == "lu-cudss"
    monkeypatch.setattr(S, "_preferred_direct_backend", lambda: "device")
    assert spec.at_solve().name == "lu"
    assert jno.solve.lu(backend="host").at_solve is None  # a named backend is not reconsidered


def test_a_solver_context_is_optional_on_the_form():
    """A direct solver that may use a form's context still runs on a form without one (jno.fdm has none)."""
    from jax.experimental import sparse as jsparse

    from jno.utils.solver.solver_api import LinearSolver, compose_linear_solve_fn

    seen = []

    def fn(op, b, *, M, x0):
        seen.append(getattr(op, "solver_context", None))
        return b

    spec = LinearSolver(fn, name="probe", direct=True, traits={"solver_context": True})
    b = jnp.ones(3)
    A = jsparse.BCOO.fromdense(jnp.eye(3))
    np.testing.assert_allclose(compose_linear_solve_fn(spec, None, None, fem=object())(A, b), b)
    assert seen == [None]


def test_two_lu_asked_for_alike_share_a_factorisation():
    """Forms with the same operator group on the solver asked for, not on which object it is."""
    from jno.utils.solver import shared_solve as SS

    class _Fem:
        _operator_key = staticmethod(lambda: "op")

    class _Node:
        _fem_ref, _solve_fn = _Fem(), staticmethod(lambda A, b: b)

        def __init__(self, spec):
            self._linear_spec = spec

    key = lambda spec: SS._group_of(_Node(spec))[0]  # noqa: E731
    assert key(jno.solve.lu()) == key(jno.solve.lu())
    assert key(jno.solve.lu(backend="host")) != key(jno.solve.lu(backend="host", reuse=False))
    assert key(jno.solve.lu(backend="host")) != key(jno.solve.lu())


@gpu
def test_a_form_shown_symmetric_is_not_checked_on_the_host(monkeypatch):
    """the weak form's proof of symmetry stands in for the host checks of the pattern and the values (0.6 s of a 3.3 s solve
    at 11 M entries), the pattern is hashed once per form, and two eager solves use one plan"""
    import jno.utils.solver.linear as L

    fem, nc = _form()
    monkeypatch.setattr(L, "_symmetric_pattern", lambda *a: pytest.fail("the pattern was checked on the host"))
    monkeypatch.setattr(L, "_symmetric_values", lambda *a: pytest.fail("the values were checked on the host"))
    c = jnp.asarray(np.random.default_rng(0).uniform(0.7, 1.3, nc))
    x = fem.solve(k=c)
    y = fem.solve(k=c * 1.1)
    assert fem._solver_ctx["symmetric"]
    z = jax.jit(lambda c_: fem.solve(k=c_))(c)  # a traced solve: the pattern comes from the context, hashed once
    assert list(fem._solver_ctx["_ldl_digests"]) == [False]
    plans = [v for k, v in L._LDL_PLANS.items() if k[0] == id(fem)]
    assert len(plans) == 1 and len(plans[0]) == 1
    np.testing.assert_allclose(np.asarray(z), np.asarray(x), rtol=1e-12, atol=1e-14)
    ref = fem.solve(linear=jno.solve.lu(backend="host"), k=c)
    np.testing.assert_allclose(np.asarray(x), np.asarray(ref), rtol=1e-9, atol=1e-14)
    assert not np.allclose(np.asarray(x), np.asarray(y))


@gpu
def test_the_plans_go_with_their_form():
    """a slot holds the pattern and a values buffer on the device: a form that is gone leaves none behind"""
    import gc

    import jno.utils.solver.linear as L

    fem, nc = _form()
    fem.solve(k=jnp.ones(nc))
    key = id(fem)
    assert any(k[0] == key for k in L._LDL_PLANS)
    del fem
    gc.collect()
    assert not any(k[0] == key for k in L._LDL_PLANS)
