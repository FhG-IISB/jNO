"""``expr.integrate(d)``: the integral of KNOWN fields over a domain -- frozen data and frozen SOLUTIONS of any
number of forms (``u.bind(...).freeze(fem.solve())``) -- as a differentiable node for ``jno.core``.

It lets a reciprocity matrix be written as its formula, e.g. ``C_jk = ∫ ε ∇φ_k · ∇φ_j`` with φ_k, φ_j the
solutions of two different forms. Oracles: the forms' own matrices (``φ_k^T K φ_j``), and the per-form
``expr.integrate(fem)`` route on the eddy-current admittance.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for meshing")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver.fem_dofmap import build_dofmap  # noqa: E402

inner = jno.np.inner
HOST = jno.solve.lu(backend="host")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _empty():
    return jno.domain.from_array({"_": np.zeros((1, 1))})


def _eval(nodes):
    crux = jno.core([nodes[0]], domain=_empty())
    return [complex(np.asarray(v).reshape(-1)[0]) for v in crux.eval(list(nodes))]


def _two_conductors():
    """2-D Laplace, conductor k at 1 V, the other at 0 (left / right edges), a scalar parameter so each
    solve is a lazy node"""
    d = jno.shape.rect(0.0, 0.0, 2.0, 1.0, size=0.2).domain()
    phi, v = d.fem_symbols(names=("phi", "v"))
    xi, yi = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("left", split=True)[:2]
    xr, yr = d.variable("right", split=True)[:2]
    kap = jno.np.parameter((1,), name="kap")
    kap.dtype(jnp.float64)
    kap.initialize(jax.nn.initializers.constant(1.0))
    P, V = phi.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    grad = lambda f: jno.np.grad(f, [xi, yi])  # noqa: E731
    fems = [jno.fem([kap * inner(grad(P), grad(V)), phi(xl, yl) - gl, phi(xr, yr) - gr]) for gl, gr in ((1.0, 0.0), (0.0, 1.0))]
    return d, phi, (xi, yi), fems, grad


def test_cross_form_energy_is_phi_k_K_phi_j():
    d, phi, (xi, yi), fems, grad = _two_conductors()
    Phi = [phi.bind(x=xi, y=yi).freeze(f.solve(linear=HOST)) for f in fems]
    C = [[(inner(grad(Phi[k]), grad(Phi[j]))).integrate(d) for k in range(2)] for j in range(2)]
    got = np.array(_eval([C[0][0], C[0][1], C[1][0], C[1][1]])).real.reshape(2, 2)

    sol = [np.asarray(f.solve(linear=HOST, kap=jnp.ones(1))) for f in fems]
    u, w = d.fem_symbols(names=("u", "w"))
    K = np.asarray(jno.fem([inner(grad(u.bind(x=xi, y=yi)), grad(w.bind(x=xi, y=yi)))]).operator[0].todense())
    want = np.array([[sol[k] @ K @ sol[j] for k in range(2)] for j in range(2)])
    np.testing.assert_allclose(got, want, rtol=1e-9)
    assert got[0, 1] < 0 and abs(got[0, 1] - got[1, 0]) < 1e-10 * abs(got[0, 1])   # mutual: negative, symmetric


def test_gradient_with_respect_to_the_solution_values():
    """the node is differentiable in each frozen solution: d(phi_0^T K phi_1)/d(phi_0) = K phi_1"""
    d, phi, (xi, yi), fems, grad = _two_conductors()
    Phi = [phi.bind(x=xi, y=yi).freeze(f.solve(linear=HOST)) for f in fems]
    node = (inner(grad(Phi[0]), grad(Phi[1]))).integrate(d)
    sol = [jnp.asarray(f.solve(linear=HOST, kap=jnp.ones(1))) for f in fems]
    g = np.asarray(jax.grad(lambda a: node.fn(a, sol[1]))(sol[0]))
    u, w = d.fem_symbols(names=("u", "w"))
    K = np.asarray(jno.fem([inner(grad(u.bind(x=xi, y=yi)), grad(w.bind(x=xi, y=yi)))]).operator[0].todense())
    np.testing.assert_allclose(g, K @ np.asarray(sol[1]), rtol=1e-9, atol=1e-12)


def test_eddy_admittance_matches_the_per_form_integrals():
    """complex N1E solutions of two drives: Y_jk = ∫ σ (E_k - j w A_k) · E_j over the domain, with A_k the frozen
    solution of form k, equals the integral over form k"""
    d = jno.shape.box(0, 0, 0, 1, 1, 1).sized(0.5).domain()
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    e, _f = d.fem_symbols(value_shape=(3,), names=("e", "f"), space="N1E")
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    k = jno.np.parameter(s, name="k")
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(1.0))
    n = int(build_dofmap(np.asarray(d.mesh.cells_dict["tetra"]), "N1E", 1, n_verts=len(d.mesh.points)).n_dofs)
    H = np.random.default_rng(0).normal(size=(2, n))
    A, V = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    curl = lambda f_: f_.vector.curl(x, y, z)  # noqa: E731
    sigma, w = 3.0 * k, 2.0
    E = [e.bind(x=x, y=y, z=z).freeze(h) for h in H]
    fems = [jno.fem([inner(curl(u), curl(v)) + 1j * w * sigma * inner(A, V) + 0.1 * inner(A, V) - sigma * inner(Ek, V)]) for Ek in E]
    Ak = [u.bind(x=x, y=y, z=z).freeze(f.solve(linear=HOST)) for f in fems]
    Yd = [(sigma * inner(E[kk] - 1j * w * Ak[kk], E[j])).integrate(d) for j in range(2) for kk in range(2)]
    Yf = [(sigma * inner(E[kk] - 1j * w * A, E[j])).integrate(fems[kk], solver=HOST) for j in range(2) for kk in range(2)]
    got, want = np.array(_eval(Yd)), np.array(_eval(Yf))
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_refusals():
    d, phi, (xi, yi), fems, grad = _two_conductors()
    with pytest.raises(ValueError, match="unknown"):
        (inner(grad(phi.bind(x=xi, y=yi)), grad(phi.bind(x=xi, y=yi)))).integrate(d)
    frozen = phi.bind(x=xi, y=yi).freeze(fems[0].solve(linear=HOST))
    v = d.fem_symbols(names=("p", "q"))[1].bind(x=xi, y=yi)
    with pytest.raises(NotImplementedError, match="frozen to a solve"):
        jno.fem([inner(grad(phi.bind(x=xi, y=yi)), grad(v)) - frozen * v]).operator


def test_shared_solves_are_evaluated_once_per_evaluation(monkeypatch):
    """an expression that refers to the same integral (and so the same solves) several times assembles each form
    as often as one reference does: solve and integral nodes are combined per evaluation"""
    import jno.trace as T

    d, phi, (xi, yi), fems, grad = _two_conductors()
    Phi = [phi.bind(x=xi, y=yi).freeze(f.solve(linear=HOST)) for f in fems]
    c01 = (inner(grad(Phi[0]), grad(Phi[1]))).integrate(d)
    calls = []
    orig = T.FemLinearSystem.evaluate
    monkeypatch.setattr(T.FemLinearSystem, "evaluate", lambda self, args=None: calls.append(1) or orig(self, args))
    crux = jno.core([c01], domain=_empty())
    crux.eval([c01])
    once = len(calls)
    calls.clear()
    many = 2.0 * c01 - c01 * c01 + jno.np.stack([c01, c01])[1]
    crux.eval([many])
    assert len(calls) == once, f"{len(calls)} assemblies for one expression with 4 references, {once} for one"


def test_a_list_of_expressions_is_evaluated_as_one_program(monkeypatch):
    """crux.eval([a, b, c, d]) of four functionals of the same two solves compiles ONE program for the list (one
    evaluator scope, as the training step evaluates its entries), not one per expression; values equal the
    one-by-one evaluation"""
    from jno.trace_compiler import TraceCompiler

    d, phi, (xi, yi), fems, grad = _two_conductors()
    Phi = [phi.bind(x=xi, y=yi).freeze(f.solve(linear=HOST)) for f in fems]
    nodes = [(inner(grad(Phi[j]), grad(Phi[k]))).integrate(d) for j, k in ((0, 0), (0, 1), (1, 1))]
    nodes.append(2.0 * nodes[1] - nodes[0])
    alone = [complex(np.asarray(jno.core([n], domain=_empty()).eval(n)).reshape(-1)[0]) for n in nodes]
    built = []
    for name in ("compile_traced_expression", "compile_multi_expression"):
        orig = getattr(TraceCompiler, name)
        monkeypatch.setattr(TraceCompiler, name, staticmethod(lambda *a, _o=orig, _n=name, **k: built.append(_n) or _o(*a, **k)))
    crux = jno.core([nodes[0]], domain=_empty())
    built.clear()
    together = [complex(np.asarray(v).reshape(-1)[0]) for v in crux.eval(nodes)]
    assert built == ["compile_multi_expression"], f"programs built for the list: {built}"
    np.testing.assert_allclose(together, alone, rtol=1e-12)
    built.clear()
    crux.eval(nodes)
    assert built == [], "the list's program is cached like a single expression's"


def _localized_admittance():
    """the eddy admittance of two drives whose sources sit on a few edges at x < 0.3 (a terminal's bar)"""
    d = jno.shape.box(0, 0, 0, 1, 1, 1).sized(0.25).domain()
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    e, _f = d.fem_symbols(value_shape=(3,), names=("e", "f"), space="N1E")
    dm = build_dofmap(np.asarray(d.mesh.cells_dict["tetra"]), "N1E", 1, n_verts=len(d.mesh.points))
    P = np.asarray(d.mesh.points)
    ev = np.zeros((int(dm.n_dofs), 2), np.int64)
    ev[np.asarray(dm.cell_dofs).reshape(-1)] = np.asarray(d.mesh.cells_dict["tetra"])[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2)
    near = (P[ev[:, 0], 0] < 0.3) & (P[ev[:, 1], 0] < 0.3)
    H = np.random.default_rng(1).normal(size=(2, int(dm.n_dofs))) * near
    A, V = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    curl = lambda f_: f_.vector.curl(x, y, z)  # noqa: E731
    E = [e.bind(x=x, y=y, z=z).freeze(h) for h in H]
    kap = jno.np.parameter((1,), name="kap")  # a runtime parameter: the solves stay lazy, inputs of the integrals
    kap.dtype(jnp.float64)
    kap.initialize(jax.nn.initializers.constant(1.0))
    fems = [jno.fem([inner(curl(u), curl(v)) + 2j * 3.0 * inner(A, V) + 0.1 * kap * inner(A, V) - 3.0 * inner(Ek, V)])
            for Ek in E]
    Ak = [u.bind(x=x, y=y, z=z).freeze(f.solve(linear=HOST)) for f in fems]
    return d, lambda: [(3.0 * inner(E[k] - 2j * Ak[k], E[j])).integrate(d) for j in range(2) for k in range(2)], fems


def test_an_integral_runs_over_the_support_of_its_data_only(monkeypatch):
    """the integrand vanishes off the cells of the known source E_j: the loop runs over those, and value and
    gradient equal those of the loop over every cell"""
    import jno.utils.solver.fem_known as FK

    d, build, fems = _localized_admittance()
    seen = []
    orig = FK.cell_support
    monkeypatch.setattr(FK, "cell_support", lambda *a: seen.append(orig(*a)) or seen[-1])
    Y = build()
    n_cells = len(d.mesh.cells_dict["tetra"])
    assert all(s is not None and 0 < s.sum() < 0.5 * n_cells for s in seen), [None if s is None else int(s.sum()) for s in seen]
    monkeypatch.setattr(FK, "cell_support", lambda *a: None)
    Y_all = build()
    np.testing.assert_allclose(np.array(_eval(Y)), np.array(_eval(Y_all)), rtol=1e-12)
    sol = [jnp.asarray(f.solve(linear=HOST, kap=jnp.ones(1))) for f in fems]
    g = lambda node: np.asarray(jax.grad(lambda a: jnp.real(node.fn(a)) + jnp.imag(node.fn(a)))(sol[1]))  # noqa: E731
    np.testing.assert_allclose(g(Y[1]), g(Y_all[1]), rtol=1e-11, atol=1e-14)
    assert np.abs(g(Y_all[1])).max() > 0


def test_integrals_of_one_structure_are_one_kernel(monkeypatch):
    """Y_jk = ∫ 3 (E_k - 2j A_k)·E_j differ only in the fields they read: the diagonal pair and the off-diagonal
    pair are each ONE batched kernel (a diagonal entry reads one field twice, so it is a structure of its own),
    each member over its own support; value and gradient w.r.t. the parameter equal the one-by-one evaluation"""
    import equinox as eqx

    import jno.utils.solver.shared_integral as SI
    from jno.trace_compiler import TraceCompiler

    d, build, fems = _localized_admittance()
    plans = []
    orig = SI._plan
    monkeypatch.setattr(SI, "_plan", lambda m: plans.append(len(m)) or orig(m))

    def run(Y):
        F = jno.fn(lambda *ys: sum((i + 1) * jnp.real(y) + (i + 2) * jnp.imag(y) for i, y in enumerate(ys)), list(Y), name="F")
        crux = jno.core([F], domain=_empty())
        val = float(np.asarray(crux.eval([F])).reshape(-1)[0])
        fn = TraceCompiler.compile_traced_expression(F, crux.all_ops)
        params, static = eqx.partition(eqx.tree_inference(crux._unwrapped_models), eqx.is_inexact_array)
        g = jax.grad(lambda p: jnp.sum(fn(eqx.combine(p, static), crux.domain_data.context)))(params)
        return val, [np.asarray(x) for x in jax.tree_util.tree_leaves(g)]

    v_group, g_group = run(build())
    assert sorted(plans) == [2, 2], f"group sizes planned: {plans}"
    monkeypatch.setattr(SI, "integrate_shared", lambda *a, **k: False)
    v_alone, g_alone = run(build())
    np.testing.assert_allclose(v_group, v_alone, rtol=1e-12)
    for a, b in zip(g_group, g_alone):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-14 * (np.abs(b).max() + 1e-300))
    assert any(np.abs(b).max() > 0 for b in g_alone)


def test_only_integrands_that_differ_in_their_fields_alone_are_grouped():
    """the grouping key: the same integrand over other fields of the same element matches; a different parameter,
    literal, coordinate, component index or element does not"""
    from jno.utils.solver.shared_integral import _key

    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    x, y = d.variable("interior", split=True)[:2]
    phi, _v = d.fem_symbols(names=("phi", "v"))
    w, _q = d.fem_symbols(names=("w", "q"), value_shape=(2,))
    n = len(d.mesh.points)
    rng = np.random.default_rng(0)
    F = [phi.bind(x=x, y=y).freeze(rng.normal(size=n)) for _ in range(4)]
    W = [w.bind(x=x, y=y).freeze(rng.normal(size=2 * n)) for _ in range(2)]
    k1, k2 = (jno.np.parameter((1,), name=nm) for nm in ("k1", "k2"))
    key = lambda e: _key(getattr(e.integrate(d), "_expr", e.integrate(d)))  # noqa: E731
    base = key(k1 * F[0] * F[1])
    assert key(k1 * F[2] * F[3]) == base, "the same integrand over other fields of the same element must match"
    assert key(k2 * F[2] * F[3]) != base, "another parameter"
    assert key(2.0 * F[2] * F[3]) != key(3.0 * F[2] * F[3]), "another literal"
    assert key(k1 * x * F[2] * F[3]) != key(k1 * y * F[2] * F[3]), "another coordinate"
    assert key(k1 * F[2] * F[2]) != base, "one field read twice is another structure"
    assert key(W[0][0] * W[1][0]) != key(W[0][1] * W[1][1]), "another component"
    assert key(k1 * W[0][0] * F[1]) != key(k1 * F[0] * F[1]), "another element"


def test_support_of_region_masks_sums_and_zeros():
    """a region coefficient restricts to its region (a literal-0 default adds nothing), a sum with a term of
    unknown support is unrestricted, a literal 0 has no support"""
    from jno.utils.solver.fem_known import cell_support

    d = (jno.shape.rect(0, 0, 1, 1).name("a") + jno.shape.rect(1, 0, 2, 1).name("b")).sized(0.25).domain()
    d.attach("a", m=1.0).attach(m=0.0)
    from jno.utils.solver.fem_utils import _cell_region_mask

    n = len(d.mesh.cells_dict["triangle"])
    region = lambda r: np.asarray(_cell_region_mask(d, r)).reshape(-1) != 0  # noqa: E731
    phi, _v = d.fem_symbols(names=("phi", "v"))
    xi, yi = d.variable("interior", split=True)[:2]
    f = phi.bind(x=xi, y=yi).freeze(np.ones(len(d.mesh.points)))
    s = cell_support(d.m * f, n, lambda node: None, region)
    np.testing.assert_array_equal(s, region("a"))
    assert cell_support(d.m * f + xi, n, lambda node: None, region) is None
    assert not cell_support(0.0 * f, n, lambda node: None, region).any()
