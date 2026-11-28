"""Which direct backend a ``fem.solve()`` with no ``linear=`` factorizes with.

A steady linear form that needs a direct solve because a Krylov iteration does not converge on it (a periodic tie,
a line mesh, a complex form, the Hermite/Argyris/Morley families), and the transient step of the C1 families, take
whatever ``jno.solve.lu()`` resolves to where the solve runs: cuDSS on a float64 GPU that has it, else JAX's
``spsolve``. They used to call ``spsolve`` (a sparse QR on a GPU) whatever ``lu()`` would pick.

CPU CI never sees the cuDSS choice, so the ``forced_direct_backend`` fixture (tests/conftest.py) forces the choice and
counts the calls to ``linear.*_lu_solve``; ``"cudss"`` is a SuperLU-backed stand-in. A site that ignores the preferred
backend fails here as "spsolve was called".
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for meshing")
pytest.importorskip("shapely", reason="shapely required for the box domain")
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi
inner, grad, sin, hessian, laplacian = jno.np.inner, jno.np.grad, jno.np.sin, jno.np.hessian, jno.np.laplacian


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _line_poisson():
    d = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    return jno.fem([ui.x * vi.x - 1.0 * vi, u(xb) - 0.0])


def _periodic_in_x():
    dom = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.07)
    dom.tag("left", lambda x, y: (x < 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("right", lambda x, y: (x > 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("bottom", lambda x, y: y < 1e-6)
    dom.tag("top", lambda x, y: y > 1 - 1e-6)
    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = jno.np.cos(2 * PI * xi) * jno.np.sin(PI * yi)
    weak = ui.x * vi.x + ui.y * vi.y + ui * vi - f * vi
    return jno.fem([weak, u(xb, yb) - 0.0, u(xt, yt) - 0.0, u(xl, yl) - u(xr, yr)])


def _complex_helmholtz():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    c = 1.0 / (1.0 + 0.5j)
    f = (1.0 + 0.5j) * jno.np.cos(PI * xi) * jno.np.cos(PI * yi)
    return jno.fem([c * (ui.x * vi.x + ui.y * vi.y) - (1.0 + 0.2j) * (u * vi) - f * vi])


def _morley_plate():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain()
    u, phi = d.fem_symbols(space="Morley")
    X = d.variable("interior", split=True)[:2]
    Xb = d.variable("boundary", split=True)[:2]
    ui, vi = u.bind(x=X[0], y=X[1]), phi.bind(x=X[0], y=X[1])
    f = sin(PI * X[0]) * sin(PI * X[1])
    return jno.fem([inner(hessian(ui, X), hessian(vi, X), n_contract=2) - f * vi, u(*Xb) - 0.0])


SCENARIOS = {
    "periodic-reduced": _periodic_in_x,
    "one-dimensional": _line_poisson,
    "complex": _complex_helmholtz,
    "morley": _morley_plate,
}
BACKENDS = ["host", "cudss", "pardiso"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_a_steady_solve_with_no_linear_slot_uses_the_preferred_direct_backend(scenario, backend, forced_direct_backend):
    fem = SCENARIOS[scenario]()
    want = np.asarray(fem.solve())  # CPU: the choice is JAX's spsolve
    counts = forced_direct_backend(backend)
    got = np.asarray(fem.solve())
    assert counts[backend] >= 1 and counts["device"] == 0, (
        f"{scenario}: fem.solve() with the preferred backend {backend!r} called {dict(counts)}; "
        "a site ignores jno.solve.lu()'s choice and calls sparse_lu_solve"
    )
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)


def _argyris_plate_transient():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain(time=(0.0, 2.5e-3, 6))
    u, phi = d.fem_symbols(space="Argyris")
    co, cb = d.variable("interior", split=True), d.variable("boundary", split=True)
    X, t, Xb, ci = co[:-1], co[-1], cb[:-1], d.variable("initial", split=True)
    ui, vi = (s.bind(x=X[0], y=X[1], t=t) for s in (u, phi))
    u0 = sin(PI * ci[0]) * sin(PI * ci[1])
    return jno.fem([ui.t * vi + laplacian(ui, X) * laplacian(vi, X), u(*Xb) - 0.0, u(ci[0], ci[1]) - u0])


@pytest.mark.parametrize("scheme", [None, "bdf"], ids=["default-scheme", "bdf-slot"])
@pytest.mark.parametrize("backend", ["host", "cudss"])
def test_the_step_of_a_c1_transient_uses_the_preferred_direct_backend(backend, scheme, forced_direct_backend):
    def march():
        out = _argyris_plate_transient().solve(
            **({} if scheme is None else {"time": jno.solve.bdf()})
        )  # a fresh form: one that ran before keeps its compiled march  # a transient solve is deferred: evaluate it
        return np.asarray(out.fn() if hasattr(out, "fn") else out)

    want = march()
    counts = forced_direct_backend(backend)
    got = march()
    assert counts[backend] >= 1 and counts["device"] == 0, f"the step solve called {dict(counts)}"
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)


def test_a_continuation_march_of_a_line_form_uses_the_preferred_direct_backend(forced_direct_backend):
    def march():
        d = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        u, phi = d.fem_symbols()
        xi = d.variable("interior", split=True)[0]
        xb = d.variable("boundary", split=True)[0]
        ui, vi = u.bind(x=xi), phi.bind(x=xi)
        k = jno.np.parameter((1,), name="k")
        fem = jno.fem([k * ui.x * vi.x - 1.0 * vi, u(xb) - 0.0])
        return np.asarray(fem.solve(continuation=jno.solve.continuation(k=np.linspace(1.0, 2.0, 3))))

    want = march()
    counts = forced_direct_backend("host")
    got = march()
    assert counts["host"] >= 1 and counts["device"] == 0, f"the continuation march called {dict(counts)}"
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)


def test_newton_direct_uses_the_preferred_direct_backend(forced_direct_backend):
    """The same pattern for a site that was already routed (#391): the assembled tangent is factorized by the backend
    ``lu()`` resolves to."""

    def solve():
        d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=5).domain()
        u, v = d.fem_symbols(names=("u", "v"), order=1)
        x, y, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
        fem = jno.fem([(1.0 + ui * ui) * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0])
        return np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True)))

    want = solve()
    counts = forced_direct_backend("host")
    got = solve()
    # (the adjoint's rescue branch still traces a `sparse_lu_solve`, so only the factorization of the tangent is pinned)
    assert counts["host"] >= 1, f"newton(direct=True) called {dict(counts)}"
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)


# ---- the cuDSS-only half: the choice of LDL^T or LU, which a CPU run never reaches without the stand-in ----------------
def _parametric_poisson(symmetric=True):
    """A runtime-parametric form (the case `_default_direct` hands to cuDSS), symmetric or with a convection term."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    x, y = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    k = jno.np.parameter(s, name="k")
    k.dtype(jnp.float64)
    u, v = d.fem_symbols(names=("u", "v"))
    U, V = u.bind(x=x, y=y), v.bind(x=x, y=y)
    g = lambda f: jno.np.grad(f, [x, y])  # noqa: E731
    wind = 0.0 if symmetric else inner(jno.np.vector(0.7 + 0 * x, 0.3 + 0 * x), g(U)) * V
    fem = jno.fem([k * inner(g(U), g(V)) + wind - V, u(xb, yb) - 0.0])
    return fem, len(d.mesh.cells_dict["triangle"])


@pytest.mark.parametrize("symmetric", [True, False], ids=["symmetric", "non-symmetric"])
def test_a_parametric_form_takes_the_ldl_branch_only_where_cudss_would(symmetric, forced_direct_backend):
    """A symmetric operator is factored by the in-program LDL^T, a non-symmetric one by LU; on a CPU the stand-in
    makes that choice, the residual gate and the plan keys run. spsolve is called by neither."""
    fem, nc = _parametric_poisson(symmetric)
    c = jnp.asarray(np.random.default_rng(0).uniform(0.7, 1.3, nc))
    want = np.asarray(fem.solve(k=c))
    counts = forced_direct_backend("cudss")
    got = np.asarray(fem.solve(k=c))
    assert counts["device"] == 0, f"spsolve was called: {dict(counts)}"
    if symmetric:
        assert counts["cudss_ldl"] >= 1, f"the symmetric branch was not taken: {dict(counts)}"
    else:
        assert counts["cudss"] >= 1 and counts["cudss_ldl"] == 0, f"the LU branch was not taken: {dict(counts)}"
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)


def test_staggered_direct_uses_the_preferred_direct_backend(forced_direct_backend):
    """#220: each field's block is factorized by the backend lu() resolves to."""

    def solve():
        # two Poisson fields coupled both ways, with a nonlinearity so the problem takes the residual (Newton) route
        d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
        d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
        co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
        X = [co[0], co[1]]
        a, phi = d.fem_symbols()
        b, chi = d.fem_symbols()
        fem = jno.fem(
            [
                inner(grad(a, X), grad(phi, X), 1) + a * phi - 1.0 * phi - 0.3 * b * phi + 0.05 * a * a * phi,
                inner(grad(b, X), grad(chi, X), 1) + b * chi - 2.0 * a * chi,
                a(*cb) - 0.0,
                b(*cb) - 0.0,
            ]
        )
        return np.asarray(fem.solve(nonlinear=jno.solve.staggered([a, b], rtol=1e-11, atol=1e-13, direct=True)))

    want = solve()
    counts = forced_direct_backend("host")
    got = solve()
    assert counts["host"] >= 1 and counts["device"] == 0, f"staggered(direct=True) called {dict(counts)}"
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)
