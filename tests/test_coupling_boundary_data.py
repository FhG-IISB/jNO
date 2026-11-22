"""Runtime boundary data: a nodal (or per-face) data field inside a boundary condition, supplied at solve time.

A partitioned coupling hands each solver the neighbour's interface values or fluxes as such a field, so every
solver must take it in a Dirichlet, a Neumann and a Robin condition. The data are a linear function of the
boundary coordinates, which every discretisation here represents exactly, so a data field must give the
same solution as the same coefficient written as a coordinate expression (to the round-off of a direct
solve), and a constant flux must give the exact linear solution.
"""

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _square(h=0.1):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=h).domain()
    pts = np.asarray(d.mesh_connectivity["points"])
    return d, pts


# --------------------------------------------------------------------------------------------------- fem


def _fem_laplace(boundary_term):
    """-Δu = 0 on the unit square, u = 0 on the left, the given term on the right, insulated elsewhere."""
    d, pts = _square()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xl, yl, _ = d.variable("left", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    g = jno.np.parameter(d.fem_symbols()[0], name="g")
    term = boundary_term(u.bind(x=xr, y=yr), v.bind(x=xr, y=yr), xr, yr, g)
    return jno.fem([ui.x * vi.x + ui.y * vi.y, term, u(xl, yl) - 0.0]), pts


def test_fem_neumann_data_field_constant_flux_gives_the_exact_linear_solution():
    # flux 1 on the right, u = 0 on the left: u = x, which P1 represents exactly
    fem, pts = _fem_laplace(lambda ur, vr, xr, yr, g: -g * vr)
    sol = np.asarray(fem.solve(g=np.ones(len(pts))))
    np.testing.assert_allclose(sol, pts[:, 0], atol=1e-10)


@pytest.mark.parametrize(
    "kind, data_term, coord_term",
    [
        ("neumann", lambda ur, vr, xr, yr, g: -g * vr, lambda ur, vr, xr, yr: -(1.0 + yr) * vr),
        ("robin", lambda ur, vr, xr, yr, g: 2.0 * (ur - g) * vr, lambda ur, vr, xr, yr: 2.0 * (ur - (1.0 + yr)) * vr),
    ],
)
def test_fem_boundary_data_field_matches_the_coordinate_expression(kind, data_term, coord_term):
    fem_d, pts = _fem_laplace(data_term)
    fem_c, _ = _fem_laplace(lambda ur, vr, xr, yr, g: coord_term(ur, vr, xr, yr))
    got = np.asarray(fem_d.solve(g=1.0 + pts[:, 1]))  # linear data: P1 interpolation is exact on the facets
    ref = np.asarray(fem_c.solve())
    assert np.abs(ref).max() > 0.1
    np.testing.assert_allclose(got, ref, atol=1e-11)


# ------------------------------------------------------------------------------------------- fdm and fvm


def _strong_laplace(make, condition, data=None):
    """-Δu = 0, u = 0 on the left, ``condition(ur, nr, yr, g)`` on the right; ``g`` a nodal data field."""
    import equinox as eqx
    import jax.numpy as jnp

    d, pts = _square()
    x, y, _ = d.variable("interior", split=True)
    xr, yr, _ = d.variable("right", split=True)
    nr = d.variable("right", normals=True)
    xl, yl, _ = d.variable("left", split=True)
    u = d.unknown()
    ui, ur = u.bind(x=x, y=y), u.bind(x=xr, y=yr)
    g = None
    if data is not None:
        g = jno.np.parameter((len(pts),), name="g")  # data: no optimizer, so the solve stays eager
        g.model.module = eqx.tree_at(lambda m: m.value, g.model.module, jnp.asarray(data(pts)))
    return np.asarray(make([-(ui.xx + ui.yy), condition(ur, nr, yr, g), u(xl, yl) - 0.0]).solve()).reshape(-1)


_SOLVERS = {
    "fdm": jno.fdm,
    "fvm-cell": jno.fvm,
    "fvm-vertex": lambda cons: jno.fvm(cons, centering="vertex"),
}


@pytest.mark.parametrize("solver", list(_SOLVERS))
@pytest.mark.parametrize(
    "kind, data_condition, coord_condition",
    [
        ("neumann", lambda ur, nr, yr, g: ur.d(nr) - g, lambda ur, nr, yr, g: ur.d(nr) - (1.0 + 3.0 * yr)),
        (
            "robin",
            lambda ur, nr, yr, g: ur.d(nr) + 2.0 * (ur - g),
            lambda ur, nr, yr, g: ur.d(nr) + 2.0 * (ur - (1.0 + 3.0 * yr)),
        ),
    ],
)
def test_strong_form_boundary_data_field_matches_the_coordinate_expression(solver, kind, data_condition, coord_condition):
    make = _SOLVERS[solver]
    got = _strong_laplace(make, data_condition, data=lambda p: 1.0 + 3.0 * p[:, 1])
    ref = _strong_laplace(make, coord_condition)
    other = _strong_laplace(make, data_condition, data=lambda p: 1.0 + 0.0 * p[:, 1])
    assert np.abs(ref - other).max() > 0.1  # the data reach the solve: other data, another answer
    # fdm reads the data at its boundary nodes and fvm at face centroids (the mean of the face's nodes),
    # where linear data equal the coordinate expression, so the two solves agree to round-off
    np.testing.assert_allclose(got, ref, atol=1e-11)
