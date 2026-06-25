"""Frozen (known, non-trainable) coefficients in ``jno.fem`` — lowered to a coordinate coefficient.

A ``jno.np.parameter`` is a *trainable* inverse unknown by default (→ runtime-parametric, resolved
through ``crux``). Marking it ``.freeze()`` declares it a **known** coefficient: ``jno.fem`` evaluates
its ``.initialize`` value (a constant or a coordinate function ``(x, y[, z]) -> value``) directly at
the quadrature points — exactly like ``jno.fn(...)`` — so the system assembles non-parametrically and
the known coefficient works in **every** form (steady-linear, nonlinear, transient, coupled).

These tests lock: a frozen function/constant equals the ``jno.fn``/literal reference and is
non-parametric; it works (and is correct) in nonlinear / transient / multifield forms; an
**un**-frozen parameter stays trainable and still recovers through ``crux``; and a frozen parameter
with no value, or a raw per-node array, fails loud.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

from jno.trace import FemLinearSystem  # noqa: E402
from jno.utils.solver.newton_krylov import newton_krylov  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _setup(mesh_size=0.34):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    return d, u, phi, xi, yi, xb, yb, u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)


def _solve(fem):
    return np.linalg.solve(np.asarray(fem.A), np.asarray(fem.b).reshape(-1))


def _kfun(x, y):
    return 1.0 + 4.0 * x  # a smoothly varying known conductivity


# ==========================================================================
# frozen coordinate function / constant == jno.fn / literal, and non-parametric
# ==========================================================================
def test_frozen_function_equals_jno_fn_and_is_nonparametric():
    d, u, phi, xi, yi, xb, yb, ui, vi = _setup()
    ref = _solve(jno.fem([jno.fn(_kfun, [xi, yi]) * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0]))
    k = jno.np.parameter(phi, name="k").initialize(_kfun).freeze()  # coordinate function, frozen
    fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0])
    assert fem.is_linear and not isinstance(fem._op, FemLinearSystem)  # baked to a plain (A, b)
    assert np.allclose(_solve(fem), ref, atol=1e-9)


def test_frozen_constant_equals_literal():
    d, u, phi, xi, yi, xb, yb, ui, vi = _setup()
    ref = _solve(jno.fem([3.0 * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0]))
    k = jno.np.parameter((1,), name="k").initialize(3.0).freeze()
    fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0])
    assert not isinstance(fem._op, FemLinearSystem)
    assert np.allclose(_solve(fem), ref, atol=1e-9)


# ==========================================================================
# works (and is correct) in every form — the point of lowering to jno.fn
# ==========================================================================
def test_frozen_in_nonlinear_matches_reference():
    d, u, phi, xi, yi, xb, yb, ui, vi = _setup()

    def nonlinear(kc):
        fem = jno.fem([kc * (ui.x * vi.x + ui.y * vi.y) + (ui**3 - ui) * vi - 1.0 * vi, u(xb, yb) - 0.0])
        return np.asarray(newton_krylov(lambda w: fem.residual(w), np.full(fem.dofs, 0.1)))

    k = jno.np.parameter(phi, name="k").initialize(_kfun).freeze()
    assert np.allclose(nonlinear(k), nonlinear(jno.fn(_kfun, [xi, yi])), atol=1e-7)


def test_frozen_in_transient_builds():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.4, time=(0.0, 0.1, 5))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    k = jno.np.parameter(phi, name="k").initialize(_kfun).freeze()
    fem = jno.fem([ui.t * vi + k * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(xi0, yi0) - 1.0])
    assert fem.is_transient  # frozen coefficient on the transient stiffness — builds, non-parametric in k


def test_frozen_in_multifield_builds():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.4)
    a, qa = d.fem_symbols(names=("a", "qa"))
    b, qb = d.fem_symbols(names=("b", "qb"))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ai, qai = a.bind(x=xi, y=yi), qa.bind(x=xi, y=yi)
    bi, qbi = b.bind(x=xi, y=yi), qb.bind(x=xi, y=yi)
    k = jno.np.parameter(qa, name="k").initialize(_kfun).freeze()
    fem = jno.fem(
        [
            k * (ai.x * qai.x + ai.y * qai.y) + (bi.x * qbi.x + bi.y * qbi.y) + ai * qbi + bi * qai - 1.0 * qai - 1.0 * qbi,
            a(xb, yb) - 0.0,
            b(xb, yb) - 0.0,
        ]
    )
    assert fem.is_linear


def test_frozen_per_component_vector_coefficient():
    """A vector-valued coefficient is expressed per component with scalar frozen functions (a single
    callable returning a tuple hits an assembly-kernel limit — same for jno.fn)."""
    d, u, phi, xi, yi, xb, yb, ui, vi = _setup()
    kx = jno.np.parameter(phi, name="kx").initialize(lambda x, y: 1.0 + 4.0 * x).freeze()
    ky = jno.np.parameter(phi, name="ky").initialize(lambda x, y: 2.0 + 0.0 * y).freeze()
    fem = jno.fem([kx * ui.x * vi.x + ky * ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
    ref = jno.fem(
        [
            jno.fn(lambda x, y: 1.0 + 4.0 * x, [xi, yi]) * ui.x * vi.x
            + jno.fn(lambda x, y: 2.0 + 0.0 * y, [xi, yi]) * ui.y * vi.y
            - 1.0 * vi,
            u(xb, yb) - 0.0,
        ]
    )
    assert not isinstance(fem._op, FemLinearSystem)
    assert np.allclose(_solve(fem), _solve(ref), atol=1e-9)


# ==========================================================================
# the inverse path is untouched: an UN-frozen parameter still trains
# ==========================================================================
def test_unfrozen_parameter_stays_parametric():
    d, u, phi, xi, yi, xb, yb, ui, vi = _setup()
    k = jno.np.parameter(phi, name="k").initialize(lambda key, s, dtype=jnp.float64: jnp.full(s, 2.0, dtype))
    fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0])
    assert isinstance(fem._op, FemLinearSystem) and fem._op.is_parametric


def test_unfrozen_parameter_still_recovers_via_crux():
    d, u, phi, xi, yi, xb, yb, ui, vi = _setup(mesh_size=0.4)
    f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))
    u_obs = jno.fem([3.0 * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0]).solve()  # truth at k=3

    k = jno.np.parameter(phi, name="k", key=jax.random.PRNGKey(0))
    k.initialize(jax.nn.initializers.constant(1.0))
    fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
    assert isinstance(fem._op, FemLinearSystem)
    import optax  # noqa: E402

    crux = jno.core([(fem.solve() - u_obs).mse, 1e-4 * k.regularize("h1seminorm").mean], domain=d)
    k.optimizer(optax.adam(2e-1))
    crux.solve(120)
    rec = np.asarray(crux.eval([k])).reshape(-1)
    assert abs(float(rec.mean()) - 3.0) < 0.3, f"un-frozen param should still train toward k=3 (got {rec.mean():.3f})"


# ==========================================================================
# fail-loud
# ==========================================================================
def test_frozen_without_value_raises():
    d, u, phi, xi, yi, xb, yb, ui, vi = _setup()
    k = jno.np.parameter(phi, name="k").freeze()  # frozen but never given a value
    with pytest.raises((ValueError, NotImplementedError), match="frozen parameter|no value|\\.freeze"):
        jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0])


def test_frozen_raw_array_raises():
    d, u, phi, xi, yi, xb, yb, ui, vi = _setup()
    n = np.asarray(d.mesh.points).shape[0]
    k = jno.np.parameter(phi, name="k").initialize(jnp.ones((n,))).freeze()  # raw per-node array
    with pytest.raises((ValueError, NotImplementedError), match="raw per-node|constant or a|function"):
        jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0])


def test_frozen_jax_initializer_raises():
    """A JAX initializer (key, shape, dtype) is for *trainable* params; with .freeze() it is the wrong
    form for a known coefficient (a scalar or (x,y)->value function) and must fail loud, not silently
    misread the coordinates as (key, shape)."""
    d, u, phi, xi, yi, xb, yb, ui, vi = _setup()
    k = jno.np.parameter(phi, name="k").initialize(jax.nn.initializers.constant(0.8)).freeze()
    with pytest.raises(ValueError, match="JAX initializer|scalar|coordinate function"):
        jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0])
