"""A vector Lagrange field next to a non-nodal (N1E) field on the non-nodal assembler.

The Lagrange block used to be sized one DOF per mesh vertex and gathered by vertex id, i.e. scalar
only, so ``value_shape=(3,)`` beside an N1E field died in a reshape. It is now stored component-wise
(DOF ``node * n_comp + comp``). The oracle is the same problem with the vector split into scalar
fields: both must give the same solution.
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


def _problem(split, order=1, bc=False):
    """N1E field E (curl-curl + mass), a vector P_k field U (mass + gradient energy), coupled both ways
    through ``E . U``; U as one vector field, or as three scalar fields."""
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.6).domain()
    E, F = d.fem_symbols(value_shape=(3,), names=("E", "F"), space="N1E", order=1)
    c = d.variable("interior", split=True)
    X = dict(zip("xyz", c[:3]))
    x, y, z = c[0], c[1], c[2]
    Eb, Fb = E.bind(**X), F.bind(**X)
    load = [1.0 + x, y, z * x]
    terms = [inner(Eb, Fb) + inner(E.vector.curl(x, y, z), F.vector.curl(x, y, z)) - inner(vec(*load), Fb)]
    if split:
        uv = [d.fem_symbols(names=(f"U{k}", f"V{k}"), order=order) for k in range(3)]
        ub = [u.bind(**X) for u, _ in uv]
        vb = [v.bind(**X) for _, v in uv]
        terms[0] = terms[0] + 0.1 * inner(vec(*ub), Fb)
        for k in range(3):
            ek = vec(*[1.0 if j == k else 0.0 for j in range(3)])
            terms.append(
                ub[k] * vb[k]
                + ub[k].x * vb[k].x
                + ub[k].y * vb[k].y
                + ub[k].z * vb[k].z
                + 0.1 * inner(Eb, ek) * vb[k]
                - load[k] * vb[k]
            )
    else:
        U, V = d.fem_symbols(value_shape=(3,), names=("U", "V"), order=order)
        Ub, Vb = U.bind(**X), V.bind(**X)
        terms[0] = terms[0] + 0.1 * inner(Ub.vector, Fb)
        terms.append(
            inner(Ub.vector, Vb.vector)
            + inner(Ub.x, Vb.x)
            + inner(Ub.y, Vb.y)
            + inner(Ub.z, Vb.z)
            + 0.1 * inner(Eb, Vb.vector)
            - inner(vec(*load), Vb.vector)
        )
    if bc:  # essential condition U = (x, 0, y) on the whole boundary
        bd = d.variable("boundary", split=True)
        g = [bd[0], 0.0 * bd[0], bd[1]]
        B = dict(zip("xyz", bd[:3]))
        if split:
            terms += [uv[k][0].bind(**B) - g[k] for k in range(3)]
        else:
            terms.append(U.bind(**B) - vec(*g))
    return d, jno.fem(terms)


def _blocks(fem):
    offs = [int(o) for o in fem.offsets]
    sol = np.asarray(jno.np.asarray(fem.solve(linear=jno.solve.lu()))).reshape(-1)
    return [sol[offs[i] : offs[i + 1]] for i in range(len(offs) - 1)]


@pytest.mark.parametrize("order,bc", [(1, False), (2, False), (1, True)])
def test_vector_lagrange_beside_n1e_matches_split_scalar_fields(order, bc):
    """Oracle: one vector field == three scalar fields, DOF for DOF (node-major, component-minor)."""
    _, fv = _problem(split=False, order=order, bc=bc)
    _, fs = _problem(split=True, order=order, bc=bc)
    bv, bs = _blocks(fv), _blocks(fs)
    assert len(bv) == 2 and len(bs) == 4
    # fields are laid out in order of appearance: E first, then U (or U0, U1, U2)
    np.testing.assert_allclose(bv[0], bs[0], rtol=1e-8, atol=1e-10)
    assert bv[1].size == 3 * bs[1].size
    np.testing.assert_allclose(bv[1].reshape(-1, 3), np.stack(bs[1:], axis=1), rtol=1e-8, atol=1e-10)


def test_essential_condition_on_vector_lagrange_degree_two_is_refused_by_name():
    with pytest.raises(NotImplementedError, match="one scalar unknown per component"):
        _problem(split=False, order=2, bc=True)
