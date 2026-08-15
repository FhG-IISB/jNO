"""Runtime Dirichlet parameters — a trainable ``jno.np.parameter`` in an ESSENTIAL value.

The documented limitation was "a trainable parameter may sit in the operator (stiffness) but not in
an essential/Dirichlet boundary *value*" — and the reality was worse: a scalar parameter without an
optimizer CRASHED (IndexError, gathered as a per-node data field), and with one it was silently
frozen at its stored value behind ``float(g)``.

Now a parameter in a Dirichlet value rides the exact plumbing the net-valued Dirichlet built:
collected into ``runtime_parameter_exprs`` (crux discovers it; it stays OUT of the per-cell
``runtime_parameter_tags``, like trainable mesh coordinates), and the held value is re-formed from
``args`` by ``_dirichlet_pairs_at`` — a traced JAX scalar, so ``∂b/∂g`` flows through the symmetric
elimination (steady linear), ``∂/∂g`` through the solve's ``custom_root`` (steady nonlinear), and
the adjoint through each step's ``custom_root`` (linear transient).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno


def _poisson_pieces(size=0.3, time=None):
    d = jno.Shape.rect(0, 0, 1, 1, size=size).domain(**({"time": time} if time else {}))
    u, v = d.fem_symbols()
    if time:
        xi, yi, ti = d.variable("interior", split=True)
        ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    else:
        xi, yi, _ = d.variable("interior", split=True)
        ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    xb, yb, _ = d.variable("boundary", split=True)
    return d, u, v, ui, vi, (xb, yb)


def _param(name, value, key=0):
    g = jno.np.parameter((1,), name=name, key=jax.random.PRNGKey(key))
    g.initialize(jax.nn.initializers.constant(value))
    return g


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _solve_at(fem, args):
    A, b = fem.operator.evaluate(args)
    return np.linalg.solve(_dense(A), np.asarray(b))


# --------------------------------------------------------------------------------------
# steady linear: b(args), oracle agreement, differentiability
# --------------------------------------------------------------------------------------
def test_a_dirichlet_parameter_makes_the_problem_parametric():
    d, u, v, ui, vi, (xb, yb) = _poisson_pieces()
    g = _param("g", 1.0)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - g])
    assert fem.operator.is_parametric
    b1 = np.asarray(fem.operator.evaluate({"g": np.asarray([1.0])})[1])
    b2 = np.asarray(fem.operator.evaluate({"g": np.asarray([5.0])})[1])
    assert not np.allclose(b1, b2), "b must ride the boundary value"


@pytest.mark.parametrize("gval", [0.0, -3.0, 1e3])
def test_solution_matches_the_constant_dirichlet_oracle(gval):
    """Zero, negative and large per the house rule — each must equal the hand-written constant BC."""
    d, u, v, ui, vi, (xb, yb) = _poisson_pieces()
    g = _param("g", 1.0)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - g])
    oracle = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - gval])
    got = _solve_at(fem, {"g": np.asarray([gval])})
    ref = np.asarray(oracle.solve())
    scale = max(1.0, abs(gval))
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5 * scale)


def test_gradient_through_the_linear_solve_fd_checks():
    import jax.numpy as jnp

    d, u, v, ui, vi, (xb, yb) = _poisson_pieces()
    g = _param("g", 1.0)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - g])
    op = fem.operator

    def loss(gv):
        A, b = op.evaluate({"g": gv})
        return jnp.sum(jnp.linalg.solve(A.todense() if hasattr(A, "todense") else A, b))

    gr = float(jax.grad(loss)(jnp.asarray([2.0]))[0])
    fd = float((loss(jnp.asarray([2.0 + 1e-3])) - loss(jnp.asarray([2.0 - 1e-3]))) / 2e-3)
    assert abs(gr - fd) < 1e-2 * abs(fd), f"grad {gr} vs FD {fd}"


def test_spatial_profile_times_parameter():
    """`u(top) - g*sin(pi x)`: the parameter scales a coordinate profile — evaluated per boundary
    node with the args-substituted parameter, not collapsed to a constant."""
    d, u, v, ui, vi, (xb, yb) = _poisson_pieces()
    g = _param("g", 1.0)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - g * jno.np.sin(jno.np.pi * xb)])
    oracle = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 3.0 * jno.np.sin(jno.np.pi * xb)])
    got = _solve_at(fem, {"g": np.asarray([3.0])})
    np.testing.assert_allclose(got, np.asarray(oracle.solve()), rtol=1e-4, atol=1e-5)


def test_one_parameter_in_operator_and_dirichlet_value():
    """The coupling case naive affine lowering cannot express: k scales the stiffness AND is the
    boundary value. The re-assembly route must handle both sites from one args entry."""
    d, u, v, ui, vi, (xb, yb) = _poisson_pieces()
    k = _param("k", 1.0, key=1)
    fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - k])
    oracle = jno.fem([2.0 * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 2.0])
    got = _solve_at(fem, {"k": np.asarray([2.0])})
    np.testing.assert_allclose(got, np.asarray(oracle.solve()), rtol=1e-4, atol=1e-5)


def test_fem_b_placeholder_reflects_the_stored_value():
    """`fem.b` (the static placeholder) evaluates the Dirichlet parameter at its STORED value at
    build time — like the net branch uses stored weights. NB `parameter.initialize` is lazy (applied
    at crux compile), so the build-time stored value is whatever the module holds then."""
    d, u, v, ui, vi, (xb, yb) = _poisson_pieces()
    g = _param("g", 2.0)
    stored = np.asarray(g.model.module.value)  # lazy initialize: this is what build sees
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - g])
    b_stored = np.asarray(fem.b)
    b_at_stored = np.asarray(fem.operator.evaluate({"g": stored})[1])
    np.testing.assert_allclose(b_stored, b_at_stored, rtol=1e-6, atol=1e-7)


def test_scalar_parameter_without_optimizer_no_longer_crashes():
    """The old path gathered a length-1 value by boundary-node id -> IndexError. A scalar
    optimizer-less parameter is now runtime-parametric like any other."""
    d, u, v, ui, vi, (xb, yb) = _poisson_pieces()
    g = _param("g", 1.5)  # no .optimizer(...) attached
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - g])
    got = _solve_at(fem, {"g": np.asarray([1.5])})
    assert np.all(np.isfinite(got))


# --------------------------------------------------------------------------------------
# the acceptance tests: recover a boundary value from data, all three paths
# --------------------------------------------------------------------------------------
def test_recover_boundary_value_steady_linear():
    import optax

    d, u, v, ui, vi, (xb, yb) = _poisson_pieces()
    truth = np.asarray(jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 2.5]).solve())
    g = _param("g", 1.0)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - g])
    crux = jno.core([(fem.solve() - truth).mse], domain=d)
    g.optimizer(optax.adam(1e-1))
    crux.solve(300)
    rec = float(np.asarray(crux.eval([g])).reshape(-1)[0])
    assert abs(rec - 2.5) / 2.5 < 0.01, f"recovered {rec}, want 2.5"


def test_recover_boundary_value_steady_nonlinear():
    import optax

    d, u, v, ui, vi, (xb, yb) = _poisson_pieces()
    truth = np.asarray(
        jno.fem([(1.0 + ui * ui) * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 1.5]).solve(
            nonlinear=jno.solve.newton(rtol=1e-6, atol=1e-6)
        )
    )
    g = _param("g", 0.5)
    fem = jno.fem([(1.0 + ui * ui) * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - g])
    crux = jno.core([(fem.solve() - truth).mse], domain=d)
    g.optimizer(optax.adam(1e-1))
    crux.solve(300)
    rec = float(np.asarray(crux.eval([g])).reshape(-1)[0])
    assert abs(rec - 1.5) / 1.5 < 0.01, f"recovered {rec}, want 1.5"


def test_recover_boundary_value_linear_transient():
    """A time-constant but TRAINABLE inlet value, recovered from the trajectory — the adjoint runs
    through every step's custom_root."""
    import optax

    d, u, v, ui, vi, (xb, yb) = _poisson_pieces(size=0.35, time=(0.0, 0.5, 26))
    ci = d.variable("initial", split=True)

    def build(gval):
        return jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - gval, u(ci[0], ci[1]) - 0.0])

    truth = np.asarray(build(2.5).solve().fn())
    g = _param("g", 1.0)
    fem = build(g)
    crux = jno.core([(fem.solve() - truth).mse], domain=d)
    g.optimizer(optax.adam(2e-1))
    crux.solve(150)
    rec = float(np.asarray(crux.eval([g])).reshape(-1)[0])
    assert abs(rec - 2.5) / 2.5 < 0.01, f"recovered {rec}, want 2.5"


# --------------------------------------------------------------------------------------
# breadth: vector per-component; the data-field branch must still work
# --------------------------------------------------------------------------------------
def test_vector_field_per_component_parametric_value():
    d = jno.Shape.rect(0, 0, 1, 1, size=0.3).domain()
    u, v = d.fem_symbols(value_shape=(2,))
    inner, grad = jno.np.inner, jno.np.grad
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    g = _param("g", 1.0)
    fem = jno.fem([inner(gu, gv, n_contract=2), u(xb, yb)[0] - g, u(xb, yb)[1] - 0.0])
    oracle = jno.fem([inner(gu, gv, n_contract=2), u(xb, yb)[0] - 4.0, u(xb, yb)[1] - 0.0])
    got = _solve_at(fem, {"g": np.asarray([4.0])})
    np.testing.assert_allclose(got, np.asarray(oracle.solve()), rtol=1e-4, atol=1e-4)


def test_parametric_times_temporal_value_refuses_loudly():
    """`u(top) - g*tau`: the parametric branch would un-ramp the load, the temporal branch would
    un-train the parameter — both silently wrong, so the combination must refuse at build."""
    d = jno.Shape.rect(0, 0, 0.5, 1, size=0.3).domain(tau=(0.0, 1.0, 4))
    u, v = d.fem_symbols(value_shape=(2,))
    inner, grad = jno.np.inner, jno.np.grad
    xi, yi, _ = d.variable("interior", split=True)
    ct = d.variable("top", where=lambda x, y: y > 1 - 1e-9, split=True)
    cb = d.variable("bottom", where=lambda x, y: y < 1e-9, split=True)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    g = _param("g", 0.01)
    with pytest.raises(NotImplementedError, match="BOTH runtime-parametric"):
        jno.fem([inner(gu, gv, n_contract=2), u(*ct[:2])[1] - g * ct[-1], u(*cb[:2])[0] - 0.0, u(*cb[:2])[1] - 0.0])


def test_field_sized_optimizerless_parameter_is_still_a_data_field():
    """The nodal data-field branch (a neighbour's field in a DD solve) must be untouched: a
    FIELD-sized optimizer-less parameter gathers per node, concretely, exactly as before."""
    import equinox as eqx
    import jax.numpy as jnp

    d, u, v, ui, vi, (xb, yb) = _poisson_pieces()
    n = int(np.asarray(d.mesh.points).shape[0])
    vals = np.linspace(0.0, 1.0, n)
    gfield = jno.np.parameter((n,), name="gfield")
    gfield.model.module = eqx.tree_at(lambda m: m.value, gfield.model.module, jnp.asarray(vals))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - gfield])
    assert np.all(np.isfinite(np.asarray(fem.b)))
