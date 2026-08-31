"""``ams().build(fem)`` on a MIXED system: build from the block AMS will precondition.

``build`` is the eager-setup escape hatch -- required to use AMS inside a traced solve, and the way
to stop it re-assembling its auxiliaries. It took the whole operator via ``_fem_concrete_operator``,
which on a mixed N1E x Lagrange system is the FULL (n_edges + n_verts) matrix while AMS's discrete
gradient ``G`` is (n_edges, n_verts). The result was a bare
``ValueError: matmul: dimension mismatch with signature (n,k=3319),(k=3939)`` from deep inside the
assembly, naming nothing the caller could act on.

That is also why ``_AMS.prepare``'s auto-freeze is a deliberate no-op inside a block composition:
the eager build raises there and is swallowed. So on the mixed path the documented efficient form of
AMS was simply unavailable.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")
import jax  # noqa: E402

import jno  # noqa: E402

inner, vec = jno.np.inner, jno.np.vector


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _av(size=0.5, w=1.0e4):
    """The A-V pair: curl-curl + j*w*mass on N1E, coupled to a Lagrange scalar through grad V."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    p, q = d.fem_symbols(names=("p", "q"), space="Lagrange")
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    b = d.variable("boundary", split=True)
    A_, V_ = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    Vs, Vt = p.bind(x=x, y=y, z=z), q.bind(x=x, y=y, z=z)
    cA, cV = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    g = lambda s: vec(s.x, s.y, s.z)  # noqa: E731
    tg = lambda s: jno.np.stack([s.x, s.y, s.z], axis=2)  # noqa: E731
    m = 1j * w
    return (
        d,
        u,
        p,
        jno.fem(
            [
                inner(cA, cV) + m * inner(A_, V_) + m * inner(g(Vs), V_) - inner(vec(1.0 + 0.0 * x, 0.0 * x, 0.0 * x), V_),
                m * inner(A_, tg(Vt)) + m * inner(g(Vs), tg(Vt)),
                u.vector.cross(d.variable("boundary", normals=True)),
                p.bind(x=b[0], y=b[1], z=b[2]) - 0.0,
            ]
        ),
    )


def test_build_without_a_field_says_which_field_to_name():
    """A raw dimension mismatch from inside the assembly tells the caller nothing. On a mixed system
    AMS cannot know which block is its own, so it has to be told -- and say so."""
    _d, _u, _p, fem = _av()
    with pytest.raises(ValueError, match="field="):
        jno.precond.ams().build(fem)


def test_build_on_the_named_block_succeeds():
    _d, u, _p, fem = _av()
    spec = jno.precond.ams().build(fem, field=u)
    assert spec is not None


def test_a_built_ams_composes_into_a_block_preconditioner():
    """The point of the fix: the efficient form of AMS is usable on the mixed path."""
    _d, u, p, fem = _av()
    tri = jno.precond.triangular((u, jno.precond.ams().build(fem, field=u)), (p, jno.precond.jacobi()))
    assert tri.complex_native, "a built AMS must still declare complex_native"


def test_single_field_build_is_unchanged():
    """The existing single-field call must keep working with no `field=`."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    fem = jno.fem(
        [
            inner(u.vector.curl(x, y, z), v.vector.curl(x, y, z))
            + 1.0 * inner(u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z))
            - inner(vec(1.0 + 0.0 * x, 0.0 * x, 0.0 * x), v.bind(x=x, y=y, z=z)),
            u.vector.cross(d.variable("boundary", normals=True)),
        ]
    )
    assert jno.precond.ams().build(fem) is not None


# --- a REAL auxiliary can stay on device ---------------------------------------------------------


def _real_curl_curl(size=0.4, beta=1.0):
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    return d, jno.fem(
        [
            inner(u.vector.curl(x, y, z), v.vector.curl(x, y, z))
            + beta * inner(u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z))
            - inner(vec(1.0 + 0.0 * x, 0.0 * x, 0.0 * x), v.bind(x=x, y=y, z=z)),
            u.vector.cross(d.variable("boundary", normals=True)),
        ]
    )


def test_a_real_amg_auxiliary_solves():
    """``aux=amg()`` on a REAL operator takes the JAX V-cycle rather than pyamg through a host
    callback. The Chebyshev objection that forces the host path is specific to a COMPLEX spectrum
    (measured at ~90 degrees of argument spread); a real auxiliary has a real spectrum and the
    existing device cycle carries it, which is what keeps the whole AMS apply off the host.

    It only has to be a good PRECONDITIONER, not an accurate solve: measured on a real curl-curl +
    mass problem, an exact SuperLU auxiliary takes 27 CG iterations, two V-cycles take 27, and a
    single V-cycle takes 27.
    """
    _d, fem = _real_curl_curl()
    ref = np.asarray(jno.np.asarray(fem.solve(linear=jno.solve.lu())))
    got = np.asarray(
        jno.np.asarray(
            fem.solve(linear=jno.solve.cg(tol=1e-10, maxiter=800), precond=jno.precond.ams(aux=jno.precond.amg()))
        )
    )
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-9 * max(np.abs(ref).max(), 1.0))


def test_a_real_auxiliary_matches_the_default_exact_one():
    """The device V-cycle auxiliary must not cost accuracy against the exact (SuperLU) default."""
    _d, fem = _real_curl_curl()
    a = np.asarray(jno.np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-10, maxiter=800), precond=jno.precond.ams())))
    b = np.asarray(
        jno.np.asarray(
            fem.solve(linear=jno.solve.cg(tol=1e-10, maxiter=800), precond=jno.precond.ams(aux=jno.precond.amg()))
        )
    )
    assert np.allclose(a, b, rtol=1e-5, atol=1e-9 * max(np.abs(a).max(), 1.0))
