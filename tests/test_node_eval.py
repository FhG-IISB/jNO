"""`Placeholder.eval()` — evaluate a trace node without hand-rolling a `jno.core`."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def _heat(coef, size=0.4, steps=3):
    """Transient heat problem; `coef` may be a float or a jno parameter."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(time=(0.0, 0.2, steps))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [ui.t * vi + coef * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(*ci) - 1.0],
        quad_degree=3,
    )
    return d, fem


def test_eval_matches_explicit_core():
    """The sugar returns exactly what `jno.core([node]).eval(node)` returns."""
    d, fem = _heat(1.0)
    sol = fem.solve()
    np.testing.assert_allclose(
        np.asarray(sol.eval()),
        np.asarray(jno.core([sol], domain=d).eval(sol)),
        rtol=0,
        atol=0,
    )


def test_eval_infers_domain_from_solve_node():
    """No `domain=` needed: the solve node records the domain it discretizes."""
    _, fem = _heat(1.0, steps=4)
    out = np.asarray(fem.solve().eval())
    assert out.ndim == 2 and out.shape[0] == 4  # one row per saved timestep


def test_eval_rejects_trainable_parameter():
    """A fresh core would re-run the initializer and silently return the initial guess."""
    a = jno.np.parameter((1,), name="alpha")
    a.initialize(jax.nn.initializers.constant(3.0))
    a.dtype(jnp.float64)
    a.optimizer(optax.adam(0.1))
    _, fem = _heat(a)
    with pytest.raises(ValueError, match="trainable parameter"):
        fem.solve().eval()


def test_eval_allows_frozen_parameter():
    """A frozen coefficient is baked into the assembly as a constant — safe to evaluate."""
    b = jno.np.parameter((1,), name="beta")
    b.initialize(1.0)
    b.dtype(jnp.float64)
    b.freeze()
    _, fem_p = _heat(b)
    _, fem_c = _heat(1.0)
    np.testing.assert_allclose(np.asarray(fem_p.solve().eval()), np.asarray(fem_c.solve().eval()), rtol=1e-10, atol=1e-12)


def test_eval_domain_override_resamples_expression():
    """`domain=` re-samples a Variable expression on another domain."""
    coarse = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain()
    x, y, _ = coarse.variable("interior", split=True)
    expr = x * y

    fine = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain()
    fine.variable("interior", split=True)  # register the tag on the new domain

    n_coarse = np.asarray(expr.eval()).size
    n_fine = np.asarray(expr.eval(domain=fine)).size
    assert n_fine > n_coarse


def test_eval_domain_override_rejected_on_solve_node():
    """A solve node owns its mesh; a domain= override would silently return the old solve."""
    _, fem = _heat(1.0, size=0.5)
    other = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain(time=(0.0, 0.2, 3))
    with pytest.raises(ValueError, match="owns the mesh"):
        fem.solve().eval(domain=other)


def test_parameter_preserves_declared_1d_shape():
    """A 1-D ``jno.np.parameter((N,))`` evaluates to ``(N,)``, not a spurious ``(N, 1)``.

    The ``(N, 1)`` channel axis is a *pointwise-network* output convention; it must not leak onto a bare
    parameter (it broke building a geometry functional from coordinate parameters for crux-driven
    r-adaptivity -- ``S @ cx`` then carried a phantom trailing dim)."""
    dom = jno.domain.from_array({"_": np.zeros((1, 1))})
    for shape in [(5,), (54,), (3, 2), (1,)]:
        p = jno.np.parameter(shape, name=f"p_{'x'.join(map(str, shape))}")
        got = np.asarray(jno.core([p], domain=dom).eval(p)).shape
        assert got == shape, f"parameter({shape}) evaluated to {got}, expected {shape}"


def test_field_parameter_keeps_channel_axis():
    """A NODAL FIELD parameter ``jno.np.parameter(<symbol>)`` evaluates to ``(N, 1)`` -- the SAME channel
    axis a pointwise network carries -- so ``network * field_parameter`` composes elementwise instead of
    broadcasting into a spurious ``(N, N)`` outer product. Collapsing it to ``(N,)`` (as a *bare* parameter
    must, see :func:`test_parameter_preserves_declared_1d_shape`) broke an eps coefficient
    ``network * f(field_param)`` in :mod:`jno.rcwa` -- an ``N**2`` boolean-index crash downstream. The two
    tests pin the two halves of the convention: bare parameters keep ``(N,)``, field parameters keep the
    channel axis."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain()
    _, phi = d.fem_symbols()
    n = len(d.mesh.points)
    rho = jno.np.parameter(phi, name="rho")  # one trainable value per mesh node
    got = np.asarray(jno.core([rho], domain=d).eval(rho)).shape
    assert got == (n, 1), f"field parameter evaluated to {got}, expected {(n, 1)} (channel axis kept)"
