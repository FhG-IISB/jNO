"""Nothing added for the flow work may key on being flow.

Every piece here -- the vector Laplacian, `dom.cell_metric`, the preconditioner algebra, `lsc()`,
`bdf2()` -- was built while chasing Navier-Stokes, which is exactly when fluid-specific assumptions
leak in: a field called `u`, a block index of 0, two dimensions, exactly two fields. This module is
the guard against that, and it is deliberately hostile:

* the fields are named `sigma`, `chi` and `theta` -- no `u`, no `p`, in any spelling;
* the saddle field lands in the MIDDLE of the block stack, so any "constraint is first/last"
  assumption fails (block order follows TERM order, not declaration order);
* there is a third, coupled physics field, so "exactly two blocks" fails;
* it runs in 3-D, so any 2-D assumption fails.

If a future change starts matching on names or positions, these break.
"""

import jax
import numpy as np
import pytest

import jno

inner_, grad, trace, lap = jno.np.inner, jno.np.grad, jno.np.trace, jno.np.laplacian


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _odd_saddle_3d(size=0.5):
    """A 3-D saddle system with deliberately non-fluid names and a third coupled field.

    Physically this is a Stokes flow with a buoyancy coupling; nothing in the code is allowed to know
    that. The constraint equation is written first in the term list, which -- since block order
    follows TERM order -- puts the constraint field in the MIDDLE of the block stack."""
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    d.tag("drive", lambda x, y, z: z > 1 - 1e-9)
    d.tag("held", lambda x, y, z: (z < 1e-9) | (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    d.point_region("gauge", (0.0, 0.0, 0.0))

    chi, psi = d.fem_symbols(names=("chi", "psi"), order=1)  # the CONSTRAINT field, declared first
    sigma, tau = d.fem_symbols(value_shape=(3,), names=("sigma", "tau"), order=2)  # the momentum field
    theta, eta = d.fem_symbols(names=("theta", "eta"), order=1)  # a third, coupled physics

    xi, yi, zi = d.variable("interior", split=True)[:3]
    xd, yd, zd = d.variable("drive", split=True)[:3]
    xh, yh, zh = d.variable("held", split=True)[:3]
    xg, yg, zg = d.variable("gauge", split=True)[:3]
    ax = [xi, yi, zi]

    gs, gt = grad(sigma, ax), grad(tau, ax)
    ci, pi_ = chi.bind(x=xi, y=yi, z=zi), psi.bind(x=xi, y=yi, z=zi)
    th, et = theta.bind(x=xi, y=yi, z=zi), eta.bind(x=xi, y=yi, z=zi)
    taub = tau.bind(x=xi, y=yi, z=zi)
    drive = 16.0 * xd**2 * (1 - xd) ** 2 * 16.0 * yd**2 * (1 - yd) ** 2

    fem = jno.fem(
        [
            -pi_ * trace(gs),  # the CONSTRAINT equation, written first
            inner_(gs, gt, n_contract=2) - ci * trace(gt) + th * taub[2],  # momentum + coupling
            th.x * et.x + th.y * et.y + th.z * et.z,  # the third physics
            sigma(xd, yd, zd)[0] - drive,
            sigma(xd, yd, zd)[1] - 0.0,
            sigma(xd, yd, zd)[2] - 0.0,
            sigma(xh, yh, zh)[0] - 0.0,
            sigma(xh, yh, zh)[1] - 0.0,
            sigma(xh, yh, zh)[2] - 0.0,
            theta(xd, yd, zd) - 1.0,
            theta(xh, yh, zh) - 0.0,
            chi(xg, yg, zg) - 0.0,
        ]
    )
    return d, fem, sigma, chi, theta


def test_the_saddle_block_is_found_structurally_not_by_name_or_position():
    """`chi` is the constraint, `sigma` the momentum field, and neither sits where a fluid-shaped
    assumption would put it. The momentum block is found from the operator's own coupling."""
    _d, fem, sigma, chi, _theta = _odd_saddle_3d()
    assert len(fem.blocks) == 3, "the fixture must carry three fields"
    # Block order is TERM order, not declaration order, so the constraint lands in the MIDDLE here --
    # neither the first block nor the last. Anything keying on a conventional position gets it wrong.
    i_chi = fem.block_index(chi)
    assert 0 < i_chi < len(fem.blocks) - 1, f"the constraint should sit mid-stack, got block {i_chi}"

    spec = jno.precond.lsc()
    spec.prepare(fem)
    _s_m, _s_c, i_mom, i_con = spec._blocks
    assert i_con == fem.block_index(chi), f"constraint block misidentified: {i_con}"
    assert i_mom == fem.block_index(sigma), f"momentum block misidentified: {i_mom} (a third field is present)"


def test_the_geometry_symbols_are_dimension_generic():
    """`dom.cell_metric` is (dim, dim) per quadrature point -- 3x3 here -- and matches the mesh."""
    d, _fem, _s, _c, _t = _odd_saddle_3d()
    xi, yi, zi = d.variable("interior", split=True)[:3]
    a, b = d.fem_symbols(names=("a_g", "b_g"))
    ai, bi = a.bind(x=xi, y=yi, z=zi), b.bind(x=xi, y=yi, z=zi)
    got = float(np.asarray(jno.fem([ai * bi - trace(d.cell_metric) * bi]).b).reshape(-1).sum())

    pts = np.asarray(d.mesh.points)[:, :3]
    cells = np.asarray(d._cells_p1())
    v = pts[cells]
    J = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]], axis=-1)
    K = np.linalg.inv(J)
    G = np.einsum("cki,ckj->cij", K, K)
    vol = np.abs(np.linalg.det(J)) / 6.0
    want = float(np.sum(vol * np.trace(G, axis1=1, axis2=2)))
    assert got == pytest.approx(want, rel=1e-10), "cell_metric must be the mesh's own 3x3 metric"


def test_the_vector_laplacian_is_dimension_generic():
    """Three components in 3-D: the operator must be the scalar one per component, no cross-coupling."""
    d, _fem, _s, _c, _t = _odd_saddle_3d(size=0.6)
    xi, yi, zi = d.variable("interior", split=True)[:3]
    ax = [xi, yi, zi]
    uv, vv = d.fem_symbols(value_shape=(3,), names=("uv_g", "vv_g"), order=2)
    us, vs = d.fem_symbols(names=("us_g", "vs_g"), order=2)
    uvi, vvi = uv.bind(x=xi, y=yi, z=zi), vv.bind(x=xi, y=yi, z=zi)
    usi, vsi = us.bind(x=xi, y=yi, z=zi), vs.bind(x=xi, y=yi, z=zi)
    dense = lambda A: np.asarray(A.todense() if hasattr(A, "todense") else A)  # noqa: E731
    Kv = dense(jno.fem([inner_(lap(uvi, ax), lap(vvi, ax), n_contract=1)]).A)
    Ks = dense(jno.fem([lap(usi, ax) * lap(vsi, ax)]).A)
    assert Kv.shape[0] == 3 * Ks.shape[0]
    for c in range(3):
        np.testing.assert_allclose(Kv[c::3, c::3], Ks, atol=1e-10)
    cross = max(abs(Kv[i::3, j::3]).max() for i in range(3) for j in range(3) if i != j)
    assert cross < 1e-12, f"components must not couple, got {cross:.2e}"


@pytest.mark.slow
def test_the_preconditioner_stack_solves_this_system():
    """End to end: three fields, non-fluid names, 3-D, and a different preconditioner on each block.
    A preconditioner changes speed, never the answer -- the oracle is a direct factorisation."""
    pytest.importorskip("pyamg", reason="pyamg required for the momentum block")
    _d, fem, sigma, chi, theta = _odd_saddle_3d()
    host = jno.solve.lu(backend="host")
    ref = np.asarray(fem.solve(linear=host))
    got = np.asarray(
        fem.solve(
            linear=jno.solve.fgmres(tol=1e-10, restart=120, maxiter=800),
            precond=jno.precond.triangular(
                (sigma, jno.precond.inner(host)),
                (chi, jno.precond.lsc()),
                (theta, jno.precond.amg()),
            ),
        )
    )
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    assert rel < 5e-6, f"the preconditioned solve disagrees with the direct one by {rel:.2e}"
