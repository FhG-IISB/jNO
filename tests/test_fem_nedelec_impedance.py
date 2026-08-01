"""N1E tangential-trace surface term — the natural impedance / first-order absorbing Maxwell BC.

The weak boundary term ``c·∫_Γ (n×u)·(n×v)`` assembles into the stiffness ``A`` (a *surface mass* over
the boundary faces), distinct from the essential PEC ``n×u=0`` (which pins DOFs). This is the boundary
half of the Silver–Müller absorbing BC ``∫curl u·curl v − k₀²∫εu·v + i k₀∫(n×u)(n×v)`` used by RCWA.

Field discovery keys off the volume term, so the surface mass is isolated as ``A[vol+surf] − A[vol]``
(the volume block is identical in both, so the difference is exactly the surface contribution).
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402

inner = jno.np.inner


class _Const(eqx.Module):
    """A 'network' that outputs a constant per quad point — the degenerate extreme that reproduces a
    scalar-coefficient assembly exactly, and gives a single-leaf gradient check."""

    c: jnp.ndarray

    def __call__(self, *a):
        return jnp.broadcast_to(self.c.reshape(1, 1), (jnp.asarray(a[0]).shape[0], 1))


def _const_net(c):
    net = jno.nn.wrap(_Const(c=jnp.asarray(float(c), dtype=jnp.float64)))
    net.dtype(jnp.float64)
    return net


_dense = lambda A: np.asarray(jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A))  # noqa: E731


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _cube(mesh_size=0.5):
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=mesh_size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    ccu, ccv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    nvec = d.variable("boundary", normals=True)
    tu, tv = u.vector.cross(nvec), v.vector.cross(nvec)  # n×u, n×v (tangential trace)
    return d, (xi, yi, zi), (ui, vi), (ccu, ccv), (tu, tv)


def _surface_mass(vol, surf):
    """Isolate the surface-mass matrix S = A[vol + surf] − A[vol] (a real volume term registers the field)."""
    return _dense(jno.fem([vol, surf]).A) - _dense(jno.fem([vol]).A)


def test_surface_mass_is_symmetric_psd():
    """The tangential-trace surface term assembles to a symmetric, positive-SEMI-definite matrix (a mass on
    the boundary edges only — interior/normal DOFs give zero, hence semi-definite, not definite)."""
    d, _, (ui, vi), (ccu, ccv), (tu, tv) = _cube(0.5)
    S = _surface_mass(inner(ccu, ccv) + inner(ui, vi), inner(tu, tv))
    np.testing.assert_allclose(S, S.T, atol=1e-10)
    evals = np.linalg.eigvalsh(S)
    assert evals.min() > -1e-9, f"surface mass must be PSD; min eig = {evals.min():.2e}"
    assert evals.max() > 1e-6, "surface mass is entirely zero — the boundary faces were not integrated"


def test_surface_mass_matches_analytic_tangential_integral():
    """Decisive geometry check: the constant field E=(0,0,1) is exact in N1E0 (Whitney), and its
    tangential-trace surface integral over the unit cube is ∮|n×E|² = 4 (the four side faces contribute
    1 each; the top/bottom faces have n∥E so contribute 0). Projecting E to its edge DOFs and evaluating
    uᵀSu must reproduce 4."""
    vec = jno.np.vector
    d, (xi, yi, zi), (ui, vi), (ccu, ccv), (tu, tv) = _cube(0.4)
    E0 = vec(0.0 * xi, 0.0 * yi, 1.0 + 0.0 * zi)  # constant (0,0,1)

    M = _dense(jno.fem([inner(ui, vi)]).A)  # N1E mass
    load = np.asarray(jnp.asarray(jno.fem([inner(ui, vi) - inner(E0, vi)]).b)).reshape(-1)  # ∫ E0·φ
    u_dof = np.linalg.solve(M, load)  # exact L² projection of the constant field
    S = _surface_mass(inner(ui, vi), inner(tu, tv))

    np.testing.assert_allclose(float(u_dof @ S @ u_dof), 4.0, atol=1e-9)


def test_coefficient_scales_the_surface_mass():
    """A scalar coefficient multiplies the surface mass linearly: 2.5·inner(n×u,n×v) gives 2.5·S."""
    d, _, (ui, vi), (ccu, ccv), (tu, tv) = _cube(0.6)
    vol = inner(ccu, ccv) + inner(ui, vi)
    S = _surface_mass(vol, inner(tu, tv))
    S25 = _surface_mass(vol, 2.5 * inner(tu, tv))
    np.testing.assert_allclose(S25, 2.5 * S, atol=1e-9)


def test_imaginary_impedance_lands_in_the_imag_leg():
    """The physical use: an ``i·k₀`` impedance coefficient is purely imaginary, so the surface mass lands in
    the Im leg of the complex system (A_i) and the Re leg (A_r) keeps only the curl-curl − k₀² mass volume."""
    d, (xi, yi, zi), (ui, vi), (ccu, ccv), (tu, tv) = _cube(0.6)
    k0 = 2.0

    fem = jno.fem([inner(ccu, ccv) - k0**2 * inner(ui, vi), 1j * k0 * inner(tu, tv)])
    assert fem.is_complex
    op_r, op_i = fem._complex_legs  # unfused legs (``_op`` is the fused 2n block)

    S = _surface_mass(inner(ui, vi), inner(tu, tv))
    K = _dense(jno.fem([inner(ccu, ccv)]).A)
    Mm = _dense(jno.fem([inner(ui, vi)]).A)
    np.testing.assert_allclose(_dense(op_r[0]), K - k0**2 * Mm, atol=1e-8)  # Re leg: curl-curl − k0² mass
    np.testing.assert_allclose(_dense(op_i[0]), k0 * S, atol=1e-8)  # Im leg: k0 · surface mass


def test_parametric_impedance_surface_mass_matches_and_differentiates():
    """A runtime parameter in the N1E tangential-trace (impedance) surface coefficient re-assembles the
    surface mass per args, DIFFERENTIABLY (inverse design of a surface impedance). The parametric operator
    at ``k=k0`` equals the constant-coefficient assembly, and a scalar readout's gradient w.r.t. ``k``
    matches central finite differences."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    ccu, ccv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    nvec = d.variable("boundary", normals=True)
    tu, tv = u.vector.cross(nvec), v.vector.cross(nvec)  # n×u, n×v
    k = jno.np.parameter((), name="k").initialize(jax.nn.initializers.constant(2.0))
    fem = jno.fem([inner(ccu, ccv) + inner(ui, vi), k * inner(tu, tv)])  # volume + a parametric surface term
    assert fem.is_linear and list(fem.operator.runtime_parameter_exprs) == ["k"]

    # (1) the parametric operator at k=k0 equals assembling with the constant coefficient k0
    for k0 in (0.7, 2.5):
        A_param = _dense(fem.operator.evaluate({"k": k0})[0])
        A_const = _dense(jno.fem([inner(ccu, ccv) + inner(ui, vi), k0 * inner(tu, tv)]).A)
        assert np.max(np.abs(A_param - A_const)) < 1e-9, f"parametric surface mass mismatch at k={k0}"

    # (2) differentiable in k: the surface mass is linear in k, so the gradient matches central FD
    def _loss(kv):
        A, _b = fem.operator.evaluate({"k": kv})
        Ad = A.todense() if hasattr(A, "todense") else A
        return jnp.sum(Ad**2)

    g = float(jax.grad(_loss)(2.0))
    fd = (float(_loss(2.0 + 1e-4)) - float(_loss(2.0 - 1e-4))) / 2e-4
    assert abs(g - fd) / abs(fd) < 1e-5, f"autodiff {g} vs FD {fd}"


def test_transient_n1e_surface_impedance_reaches_steady_state():
    """An N1E impedance surface term now composes with a TRANSIENT problem: the tangential-trace surface
    mass is added to the spatial operator ``A`` of ``M u̇ + A u = c``. Marched to steady state, the transient
    solution converges to the STEADY solve of the same ``A u = c`` (u̇ → 0) — a real oracle that the surface
    mass is in the transient operator, not silently dropped."""
    T, NS, AL, CI = 1.4, 28, 8.0, 3.0  # end time, steps, reaction coeff, impedance coeff (fast decay → converges)

    def _build(transient):
        bx = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.6)
        d = bx.domain(time=(0.0, T, NS)) if transient else bx.domain()
        u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
        co = d.variable("interior", split=True)
        xi, yi, zi = co[0], co[1], co[2]
        nvec = d.variable("boundary", normals=True)
        tu, tv = u.vector.cross(nvec), v.vector.cross(nvec)
        ccu, ccv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
        if transient:
            ti = co[3]
            ui, vi = u.bind(x=xi, y=yi, z=zi, t=ti), v.bind(x=xi, y=yi, z=zi, t=ti)
        else:
            ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
        src = 1.0 * vi[0] + 0.5 * vi[1] + 0.2 * vi[2]  # a constant source
        vol = inner(ccu, ccv) + AL * inner(ui, vi) - src  # volume: curl-curl + reaction − load (interior region)
        surf = CI * inner(tu, tv)  # the N1E tangential-trace impedance surface term (its own boundary region)
        if not transient:
            return np.asarray(jno.fem([vol, surf]).solve()).reshape(-1)
        ci = d.variable("initial", split=True)
        u0 = u.bind(x=ci[0], y=ci[1], z=ci[2])
        traj = jno.fem([inner(ui.t, vi) + vol, surf, u0[0] - 0.0, u0[1] - 0.0, u0[2] - 0.0]).solve()
        return np.asarray(traj.fn() if hasattr(traj, "fn") else traj)[-1]  # final-time state

    steady = _build(False)
    final = _build(True)
    rel = np.linalg.norm(final - steady) / (np.linalg.norm(steady) + 1e-30)
    assert np.linalg.norm(steady) > 1e-4, "steady solve is trivial — the source/impedance did not drive it"
    assert rel < 5e-3, f"transient with an N1E surface impedance did not reach the steady solve (rel {rel:.2e})"


def test_neural_impedance_surface_coefficient_matches_and_differentiates():
    """A trainable NEURAL coefficient in the N1E tangential-trace (impedance) surface term now assembles
    (the non-nodal boundary net was previously rejected) and is DIFFERENTIABLE in the network weights — a
    *learned* surface impedance for inverse design. A constant-output net reproduces the scalar-coefficient
    operator exactly, and the operator's gradient w.r.t. the weights is finite and non-zero."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    ccu, ccv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    bc = d.variable("boundary", split=True)  # the net coefficient is evaluated on the boundary region
    xb, yb, zb = bc[0], bc[1], bc[2]
    nvec = d.variable("boundary", normals=True)
    tu, tv = u.vector.cross(nvec), v.vector.cross(nvec)
    net = _const_net(0.7)
    fem = jno.fem([inner(ccu, ccv) + inner(ui, vi), net(xb, yb, zb) * inner(tu, tv)])
    (name,) = fem.operator.runtime_parameter_exprs

    # a constant-output net reproduces the scalar-coefficient assembly exactly
    A_net = _dense(fem.operator.evaluate({name: net.module})[0])
    A_const = _dense(jno.fem([inner(ccu, ccv) + inner(ui, vi), 0.7 * inner(tu, tv)]).A)
    assert np.abs(A_net - A_const).max() < 1e-12, "neural surface impedance ≠ its constant-coefficient assembly"

    # differentiable in the network weights (a learned impedance): finite, non-zero ∂(operator)/∂weights
    def _loss(module):
        A, _b = fem.operator.evaluate({name: module})
        Ad = A.todense() if hasattr(A, "todense") else A
        return jnp.sum(Ad**2)

    g = eqx.filter_grad(_loss)(net.module)
    gc = float(np.asarray(g.c))
    assert np.isfinite(gc) and abs(gc) > 1e-8, f"gradient w.r.t. the net weight is degenerate: {gc}"
