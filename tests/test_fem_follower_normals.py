"""``domain.variable(tag, normals=True, follow_normals=True)`` — the DEFORMED surface's normal.

A traction ``p n . phi`` is a **dead** load on the reference normal and a **follower** load on the
deformed one. Gravity is the first: it still points down after the body tips over. Pressure is the
second: it stays perpendicular to the skin however the skin moves. Contact belongs with pressure —
the contact force is normal to the surfaces that are actually touching — which is why a strip wrapped
round a die radius needs it and a lightly-loaded block does not.

The oracle is a RIGID ROTATION, where the answer is known exactly rather than plausible: rotate the
body by ``theta`` and the deformed normal is ``R(theta)`` times the reference normal, so the traction
it produces must rotate by exactly ``theta`` and keep its magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

E_F, NU_F = 100.0, 0.3
LAM_F = E_F * NU_F / ((1 + NU_F) * (1 - 2 * NU_F))
MU_F = E_F / (2 * (1 + NU_F))


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _form(follow, traction=True, size=0.34):
    """Unit square with a unit pressure on its top face. ``follow`` picks which normal that uses."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).sized(size).domain()
    _ = d.built_mesh
    d.tag("top", lambda x, y: y > 1.0 - 1e-9)
    u, phi = d.fem_symbols(value_shape=(2,))
    X = d.variable("interior", split=True)[:2]
    eu = jno.np.symgrad(u.bind(x=X[0], y=X[1]), list(X))
    ep = jno.np.symgrad(phi.bind(x=X[0], y=X[1]), list(X))
    terms = [LAM_F * jno.np.trace(eu) * jno.np.trace(ep) + 2 * MU_F * jno.np.inner(eu, ep, n_contract=2)]
    if traction:
        tp = d.variable("top", split=True)
        nrm = d.variable("top", normals=True, follow_normals=follow)
        terms.append(1.0 * jno.np.inner(nrm, phi.bind(x=tp[0], y=tp[1]), n_contract=1))
    # Clamp only the BOTTOM: a Dirichlet row replaces the equation at that node, so pinning the whole
    # boundary would overwrite the very traction rows this test is trying to read.
    d.tag("bot", lambda x, y: y < 1e-9)
    bb = d.variable("bot", split=True)
    terms.append(u(bb[0], bb[1]) - 0.0)
    return d, jno.fem(terms)


def _rigid(d, fem, deg):
    """The DOF vector for a rigid rotation of the whole body by ``deg`` about the origin."""
    t = np.radians(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    P = np.asarray(fem.field_points[0])[:, :2]
    return ((P @ R.T) - P).reshape(-1)


@pytest.mark.parametrize("deg", [10.0, 30.0, 55.0])
def test_a_following_normal_rotates_the_traction_by_exactly_the_body_rotation(deg):
    """The whole claim, against a closed form.

    Isolate the traction by differencing against the same form WITHOUT it, at the same state — the
    elastic part is then identical and cancels. The reference-normal traction must not move at all;
    the following one must rotate by exactly ``deg`` and keep its magnitude, because a rigid rotation
    changes the surface's orientation and nothing else.
    """
    d0, fem_none = _form(False, traction=False)
    d1, fem_ref = _form(False)
    d2, fem_fol = _form(True)
    u = _rigid(d0, fem_none, deg)

    def res(fem, uu):
        """Residual at `uu`, whichever way the form assembled. A reference-normal form is LINEAR, so
        its operator is the pair (A, b) and its residual is A u - b; a following one is nonlinear and
        carries a residual callable. That difference is itself the point -- see the mode note below."""
        import jax.numpy as jnp

        op = fem._op
        if isinstance(op, tuple):
            A, b = op
            return np.asarray(A @ jnp.asarray(uu) - b).reshape(-1)
        return np.asarray(op.residual(uu, None)).reshape(-1)

    r0 = res(fem_none, u)
    # Sum the FREE rows only. A clamped row holds no force -- it holds the constraint equation, and
    # the scale that equation is written at is a convention: the assembled path pins at the local
    # diagonal magnitude (so the operator stays uniformly scaled for the iterative solvers), the
    # residual path at one. The two do not cancel in a difference taken across both paths, and what
    # survives swamps the traction being measured -- it read 175.78deg for a 10deg rotation. The
    # traction lives on the free DOFs; read it there.
    free = ~(np.asarray(d0.mesh.points)[:, 1] < 1e-9)
    t_ref = (res(fem_ref, u) - r0).reshape(-1, 2)[free].sum(axis=0)
    t_fol = (res(fem_fol, u) - r0).reshape(-1, 2)[free].sum(axis=0)
    assert isinstance(fem_fol._op, tuple) is False, (
        "a following normal depends on the unknown, so the form must route to the residual path -- "
        "assembled-linear would build A and b at u=0, where the two normals coincide"
    )

    assert np.linalg.norm(t_ref) > 1e-8, "the traction must actually contribute something"
    got = np.degrees(np.arctan2(t_ref[0] * t_fol[1] - t_ref[1] * t_fol[0], t_ref @ t_fol))
    assert abs(abs(got) - deg) < 0.6, (
        f"a following normal must rotate the traction by the body rotation: expected {deg}deg, got {abs(got):.2f}deg"
    )
    rel = abs(np.linalg.norm(t_fol) - np.linalg.norm(t_ref)) / np.linalg.norm(t_ref)
    assert rel < 0.02, f"a rigid rotation must not change the traction's magnitude, moved {rel:.1%}"


def test_the_reference_normal_is_still_the_default():
    """None of the existing 140-odd `normals=True` uses may move: the default stays the reference
    normal, so a dead load stays a dead load."""
    _d, fem_a = _form(False)
    assert isinstance(fem_a._op, tuple), (
        "without follow_normals the traction is a DEAD load: linear in u, so the form must still take "
        "the assembled-linear path it always did"
    )
    assert "top" not in (getattr(_d, "_follow_normals", set()) or set())


def test_following_needs_an_unambiguous_displacement_field():
    """It has to know which field moves the surface; guessing wrong rotates every traction on it."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).sized(0.4).domain()
    _ = d.built_mesh
    d.tag("top", lambda x, y: y > 1.0 - 1e-9)
    u, phi = d.fem_symbols(value_shape=())  # a SCALAR field: nothing moves the surface
    X = d.variable("interior", split=True)[:2]
    gu = jno.np.grad(u, [X[0], X[1]])
    gp = jno.np.grad(phi, [X[0], X[1]])
    tp = d.variable("top", split=True)
    nrm = d.variable("top", normals=True, follow_normals=True, split=True)
    d.tag("bot", lambda x, y: y < 1e-9)
    bb = d.variable("bot", split=True)
    # the refusal may surface at build or at solve depending on the route; either is fine, both are loud
    with pytest.raises(ValueError, match="one component per dimension"):
        jno.fem([jno.np.inner(gu, gp, n_contract=1), nrm[0] * phi.bind(x=tp[0], y=tp[1]), u(bb[0], bb[1]) - 0.0]).solve()
