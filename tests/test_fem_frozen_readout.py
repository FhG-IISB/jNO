"""Standalone `.eval()` of a FrozenField and its gradient — the boundary-functional readout.

`u.bind(x=xb, y=yb).freeze(sol)` pins a field to a solved nodal vector; this lets the SAME field's value
and gradient be read back as concrete arrays via `.eval()` — so a functional of the solution such as the
boundary normal-flux  ∇T·n  can be written as pure traced math:

    x, y, t, nx, ny = d.variable("boundary", normals=True, split=True)
    Tf   = u.bind(x=x, y=y).freeze(sol)
    flux = Tf.x * nx + Tf.y * ny          # ∇T·n  — e.g. a Stefan velocity  v_n = −k/L · ∇T·n
    flux.eval()

Oracles: the value/gradient of an affine field are P1-exact (machine precision, 2-D & 3-D); on a real
harmonic solve the normal-flux is first-order (O(h)); and the fail-loud guard when the field carries no
mesh domain to read from.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _solve_laplace_2d(d, gfun):
    """∇²u = 0 with Dirichlet u = gfun on the whole boundary; return (nodal sol, trial u)."""
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, u(xb, yb) - jno.fn(gfun, [xb, yb])])
    return np.asarray(fem.solve()), u


def test_frozen_value_readout_is_affine_exact():
    d = jno.Shape.rect(0, 0, 1, 1, size=0.06).domain()
    sol, u = _solve_laplace_2d(d, lambda x, y: 3.0 * x - 2.0 * y)
    xb, yb, _ = d.variable("boundary", split=True)
    Tf = u.bind(x=xb, y=yb).freeze(sol)
    val = np.asarray(Tf.eval()).reshape(-1)
    xv, yv = np.asarray(xb.eval()).reshape(-1), np.asarray(yb.eval()).reshape(-1)
    assert np.max(np.abs(val - (3.0 * xv - 2.0 * yv))) < 1e-8  # P1 reproduces the affine field exactly


def test_frozen_gradient_readout_is_affine_exact():
    d = jno.Shape.rect(0, 0, 1, 1, size=0.06).domain()
    sol, u = _solve_laplace_2d(d, lambda x, y: 3.0 * x - 2.0 * y)  # ∇T = (3, −2)
    xb, yb, _ = d.variable("boundary", split=True)
    Tf = u.bind(x=xb, y=yb).freeze(sol)
    gx = np.asarray(Tf.x.eval()).reshape(-1)
    gy = np.asarray(Tf.y.eval()).reshape(-1)
    assert np.max(np.abs(gx - 3.0)) < 1e-6 and np.max(np.abs(gy + 2.0)) < 1e-6


def test_boundary_normal_flux_readout_affine_exact():
    """The headline: ∇T·n written as Tf.x*nx + Tf.y*ny, evaluated, vs analytic (affine ⇒ exact)."""
    d = jno.Shape.rect(0, 0, 1, 1, size=0.05).domain()
    sol, u = _solve_laplace_2d(d, lambda x, y: 3.0 * x - 2.0 * y)
    x, y, t, nx, ny = d.variable("boundary", normals=True, split=True)
    Tf = u.bind(x=x, y=y).freeze(sol)
    vn = np.asarray((Tf.x * nx + Tf.y * ny).eval()).reshape(-1)
    nxb, nyb = np.asarray(nx.eval()).reshape(-1), np.asarray(ny.eval()).reshape(-1)
    vn_ana = 3.0 * nxb - 2.0 * nyb
    assert np.max(np.abs(vn - vn_ana)) < 1e-5  # FD-over-mesh gradient of an affine field ⇒ ~6-digit exact


def test_boundary_normal_flux_readout_harmonic_first_order():
    """On a real harmonic solve T=x²−y² (∇T=(2x,−2y)), ∇T·n is first-order accurate (O(h))."""
    d = jno.Shape.rect(0, 0, 1, 1, size=0.045).domain()
    sol, u = _solve_laplace_2d(d, lambda x, y: x**2 - y**2)
    x, y, t, nx, ny = d.variable("boundary", normals=True, split=True)
    Tf = u.bind(x=x, y=y).freeze(sol)
    vn = np.asarray((Tf.x * nx + Tf.y * ny).eval()).reshape(-1)
    xb, yb = np.asarray(x.eval()).reshape(-1), np.asarray(y.eval()).reshape(-1)
    nxb, nyb = np.asarray(nx.eval()).reshape(-1), np.asarray(ny.eval()).reshape(-1)
    vn_ana = 2.0 * xb * nxb - 2.0 * yb * nyb
    rel = np.linalg.norm(vn - vn_ana) / np.linalg.norm(vn_ana)
    assert rel < 6e-2, f"harmonic normal-flux readout too inaccurate: rel L2 = {rel:.3e}"


def test_frozen_gradient_readout_3d_affine_exact():
    """3-D: freeze an affine field on a box, read ∇T·n on the boundary (last `dim` split parts = normals)."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.34).domain()
    u, phi = d.fem_symbols()
    ci = d.variable("interior", split=True)
    cb = d.variable("boundary", split=True)
    ui, vi = u.bind(x=ci[0], y=ci[1], z=ci[2]), phi.bind(x=ci[0], y=ci[1], z=ci[2])
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y + ui.z * vi.z,
            u(cb[0], cb[1], cb[2]) - jno.fn(lambda x, y, z: 1.0 - x + 2.0 * y - 0.5 * z, [cb[0], cb[1], cb[2]]),
        ]
    )
    sol = np.asarray(fem.solve())  # ∇T = (−1, 2, −0.5)
    parts = d.variable("boundary", normals=True, split=True)
    x3, y3, z3 = parts[0], parts[1], parts[2]
    nx, ny, nz = parts[-3], parts[-2], parts[-1]  # (coords…, t, normals…) ⇒ last dim entries are normals
    Tf = u.bind(x=x3, y=y3, z=z3).freeze(sol)
    vn = np.asarray((Tf.x * nx + Tf.y * ny + Tf.z * nz).eval()).reshape(-1)
    nxb = np.asarray(nx.eval()).reshape(-1)
    nyb = np.asarray(ny.eval()).reshape(-1)
    nzb = np.asarray(nz.eval()).reshape(-1)
    vn_ana = -1.0 * nxb + 2.0 * nyb - 0.5 * nzb
    assert np.max(np.abs(vn - vn_ana)) < 1e-6


def test_frozen_readout_on_transient_domain_is_correct_and_differentiable():
    """The normals fix: on a TRANSIENT domain the normal tags are time-tiled (like the coords), so the
    boundary-flux readout evaluates correctly — the normals no longer collapse to one point — and it is
    differentiable in the field values (a Stefan velocity feeding back into the solve trains cleanly)."""
    d = jno.Shape.rect(0, 0, 1, 1, size=0.1).domain(time=(0.0, 0.3, 11))
    u, _ = d.fem_symbols()
    pts = np.asarray(d.mesh.points)[:, :2]
    sol = 3.0 * pts[:, 0] - 2.0 * pts[:, 1]  # ∇T = (3, -2)
    x, y, t, nx, ny = d.variable("boundary", normals=True, split=True)
    Tf = u.bind(x=x, y=y).freeze(sol)

    nxe, nye = np.asarray(nx.eval()).reshape(-1), np.asarray(ny.eval()).reshape(-1)
    assert nxe.shape[0] > 1 and nxe.min() < -0.9 and nxe.max() > 0.9  # many varying unit-normal comps, not 1
    flux = np.asarray((Tf.x * nx + Tf.y * ny).eval()).reshape(-1)
    assert np.max(np.abs(flux - (3.0 * nxe - 2.0 * nye))) < 1e-7  # Tf.x=3, Tf.y=-2 ⇒ flux = 3·nx - 2·ny exactly

    def loss(v):
        Tv = u.bind(x=x, y=y).freeze(v)
        return jnp.sum((Tv.x * nx + Tv.y * ny).eval() ** 2)

    g = np.asarray(jax.grad(loss)(jnp.asarray(sol)))
    assert np.all(np.isfinite(g)) and np.linalg.norm(g) > 0  # readout is differentiable in the field


def test_frozen_readout_without_domain_fails_loud():
    """A frozen field that carries no mesh domain (never coordinate-bound) cannot be read out — it must
    raise a clear error, not return garbage."""
    from jno.trace import FrozenField

    d = jno.Shape.rect(0, 0, 1, 1, size=0.2).domain()
    u, _ = d.fem_symbols()
    n = len(np.asarray(d.mesh.points))
    bare = FrozenField(u.scalar._expr, jnp.zeros(n))  # no domain / coord_tag
    with pytest.raises((ValueError, KeyError), match="domain|region|bind"):
        np.asarray(bare.eval())
