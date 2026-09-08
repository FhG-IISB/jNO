"""``u.slide(secondary, main)`` — the tangential sibling of ``u.gap``.

``u.gap`` is the normal component of the relative displacement across an interface; ``u.slide`` is what
is left in the tangent plane. Both are reads of ONE packed quantity ``D = u_secondary - u_main . Phi``
through one mortar projection, so they cannot disagree about the geometry — that sharing is the design.

    gap   = g0 - n . D      (scalar)
    slide = D - (n . D) n   (vector, in the global frame)

The oracles are mechanical, and the decisive one is an **ablation**: with only a normal penalty the
interface transmits no shear at all, so a sheared stack leaves the far body *exactly* at rest. Add a
tangential term built from ``u.slide`` and the shear comes through, converging to the answer the same
geometry gives when it is meshed as one bonded body. Nothing else in the term list changes.

Note on readout: interface symbols cannot be read back with ``fem.eval`` — that path resolves them
against the zero placeholder rather than running the assembly-time packer, so it reports "everywhere in
contact and never sliding" for any input. (Pre-existing; it affects ``u.gap`` identically.) These tests
therefore measure the *mechanics* the symbol produces, which is the thing worth pinning anyway.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

n = jno.np
SHEAR, CN = 0.02, 1.0e6


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _two_body(size=0.5, conforming=False):
    """Two blocks stacked in z. ``conforming=True`` fuses them into one body — the bonded reference."""
    return (
        jno.Shape.regions(
            lower=jno.Shape.box(0, 0, 0, 1, 1, 1),
            upper=jno.Shape.box(0, 0, 1, 1, 1, 2.0),
            conforming=conforming,
        )
        .sized(size)
        .domain()
    )


def _sides(d):
    return sorted(t for t in d.built_mesh.cell_sets if "|" in t)


def _stack(*, conforming, ct, drive, size=0.5, confine=False):
    """Clamp the bottom face, prescribe ``drive`` on the top face, and return the mean displacement of
    the LOWER block — i.e. exactly the motion that had to travel *through* the interface to get there.

    ``ct`` is the tangential penalty stiffness on ``u.slide``; ``ct = 0`` ablates it. The normal penalty
    on ``u.gap`` is always present, so the two configurations differ in one term and nothing else.

    ``confine=True`` adds roller conditions on the four side faces, making the deformation uniaxial. It
    matters for the pure-press test: without it the two blocks are meshed independently and bulge
    laterally by slightly different amounts, which is a REAL tangential mismatch of about 1% that the
    slide correctly reports — so an unconfined press is not a pure normal one.
    """
    d = _two_body(size, conforming)
    u, phi = d.fem_symbols(value_shape=(3,))
    co = d.variable("interior", split=True)
    X = [co[0], co[1], co[2]]
    ui, vi = u.bind(x=co[0], y=co[1], z=co[2]), phi.bind(x=co[0], y=co[1], z=co[2])
    cl = d.variable("clamp_bottom", where=lambda x, y, z: z < 1e-9, split=True)
    dr = d.variable("drive_top", where=lambda x, y, z: z > 2.0 - 1e-9, split=True)

    terms = [n.inner(n.symgrad(ui, X), n.symgrad(vi, X), 2)]
    if not conforming:
        sec, main = _sides(d)
        sv = d.variable(sec, split=True)
        vs = phi.bind(x=sv[0], y=sv[1], z=sv[2])
        nrm = d.variable(sec, normals=True)
        g = u.gap(sec, main, domain=d)
        s = u.slide(sec, main, domain=d)
        # Two-sided (bonded) penalties: the normal one always, the tangential one only when ct > 0.
        terms.append((-CN * g) * n.inner(nrm, vs, 1) + (-ct) * n.inner(s, vs, 1))
    terms += [u(cl[0], cl[1], cl[2])[k] - 0.0 for k in range(3)]
    terms += [u(dr[0], dr[1], dr[2])[k] - float(drive[k]) for k in range(3)]
    if confine:
        sx = d.variable("roll_x", where=lambda x, y, z: (x < 1e-9) | (x > 1 - 1e-9), split=True)
        sy = d.variable("roll_y", where=lambda x, y, z: (y < 1e-9) | (y > 1 - 1e-9), split=True)
        terms += [u(sx[0], sx[1], sx[2])[0] - 0.0, u(sy[0], sy[1], sy[2])[1] - 0.0]

    sol = np.asarray(jno.fem(terms).solve(linear=jno.solve.lu(backend="host"))).reshape(-1, 3)
    lower = np.asarray(d.built_mesh.points)[:, 2] < 1.0 - 1e-9
    return sol[lower].mean(axis=0)


# ----------------------------------------------------------------------------------------------
# The symbol layer
# ----------------------------------------------------------------------------------------------


def test_gap_and_slide_share_one_pairing_and_differ_only_in_rank():
    """One interface, one recorded pair. A second entry would be worse than redundant: the tangent's
    block geometry looks a pair up by secondary region and takes the FIRST match, so a second entry
    would silently shadow it."""
    d = _two_body()
    sec, main = _sides(d)
    u, _phi = d.fem_symbols(value_shape=(3,))
    g = u.gap(sec, main, domain=d)
    s = u.slide(sec, main, domain=d)

    assert list(d._contact_pairs) == [f"gap_{sec}"], "slide must NOT record a pairing of its own"
    assert g.dim == [0, 1], "the gap is the scalar normal component"
    assert s.dim == [0, 3], "the slide is a vector in the tangent plane"
    assert s.tag == f"slide_{sec}"


def test_slide_can_be_declared_before_the_gap_and_still_shares_the_pair():
    """Order must not matter — whichever symbol is written first records the pair, and the other joins
    it. If the second one overwrote the first, the interface would be rebuilt mid-form."""
    d = _two_body()
    sec, main = _sides(d)
    u, _phi = d.fem_symbols(value_shape=(3,))
    u.slide(sec, main, domain=d)
    u.gap(sec, main, domain=d)
    assert list(d._contact_pairs) == [f"gap_{sec}"]


def test_slide_refuses_the_same_ways_the_gap_does():
    d = _two_body()
    sec, main = _sides(d)
    u, _phi = d.fem_symbols(value_shape=(3,))
    scalar_u, _sp = d.fem_symbols()

    with pytest.raises(ValueError, match="not a boundary region"):
        u.slide("nope", main, domain=d)
    with pytest.raises(ValueError, match="must be different"):
        u.slide(sec, sec, domain=d)
    with pytest.raises(ValueError, match="vector field"):
        scalar_u.slide(sec, main, domain=d)
    with pytest.raises(TypeError, match="must be a jno domain"):
        u.slide(sec, main, domain=object())


def test_a_second_main_face_for_one_secondary_is_refused():
    """A face carries at most one contact pair, and the refusal must name the conflict rather than
    quietly rebinding — whichever symbol asks second."""
    d = _two_body()
    sec, main = _sides(d)
    u, _phi = d.fem_symbols(value_shape=(3,))
    u.gap(sec, main, domain=d)
    with pytest.raises(ValueError, match="already the secondary face"):
        u.slide(sec, "lower", domain=d)


# ----------------------------------------------------------------------------------------------
# The mechanics: shear crosses the interface only through the slide
# ----------------------------------------------------------------------------------------------


def test_without_the_slide_term_the_interface_transmits_no_shear_at_all():
    """The ablation, and the sharpest statement available: with only a normal penalty, a sheared stack
    leaves the far block at EXACTLY zero. Not 'small' — zero, because there is nothing in the form that
    could carry a tangential traction."""
    far = _stack(conforming=False, ct=0.0, drive=(SHEAR, 0.0, 0.0))
    assert np.abs(far).max() < 1e-12, f"shear crossed a frictionless interface: {far}"


def test_the_slide_term_transmits_shear_and_converges_to_the_bonded_body():
    """Add the tangential penalty and the shear comes through — to the answer the same geometry gives
    when meshed as ONE body. The residual difference is the two meshes, not the penalty, which the next
    test pins separately."""
    bonded = _stack(conforming=True, ct=0.0, drive=(SHEAR, 0.0, 0.0))[0]
    tied = _stack(conforming=False, ct=1.0e6, drive=(SHEAR, 0.0, 0.0))[0]
    assert bonded > 1e-4, "sanity: the bonded reference must actually move"
    assert tied == pytest.approx(bonded, rel=0.10)


#: The default 0.5 is too coarse for the penalty-insensitivity measurement below -- see that test.
_PENALTY_SIZE = 0.3


def test_the_transmitted_shear_is_insensitive_to_the_penalty_over_four_decades():
    """What separates 'the penalty is converged' from 'the number happens to look right'. If the answer
    still depended on `ct`, the agreement above would be a coincidence of the stiffness chosen.

    **Meshed finer than the rest of the file, on purpose.** At the default 0.5 the spread is not a
    property of the penalty at all -- the quantity is not mesh-converged there, so the number measures
    which tetrahedralisation gmsh happened to produce. Two trees that differ only in meshing, on
    identical physics:

        size    tets / spread (A)      tets / spread (B)
        0.50     194 / 0.00358          188 / 0.01927     <- disagree; neither is converged
        0.40     417 / 0.00098          384 / 0.00110
        0.30     811 / 0.00095          670 / 0.00094
        0.22    1460 / 0.00091         1146 / 0.00091     <- same answer either way

    Both converge to 0.00091, eleven times inside this gate; only the coarse mesh disagrees, and it
    passed on one tree by luck. The fix is therefore to measure where the quantity has converged, not
    to widen the tolerance until the unconverged number fits -- that would keep the test green while
    it went on measuring the mesh.
    """
    vals = [_stack(conforming=False, ct=ct, drive=(SHEAR, 0.0, 0.0), size=_PENALTY_SIZE)[0] for ct in (1e2, 1e4, 1e6)]
    spread = (max(vals) - min(vals)) / abs(np.mean(vals))
    assert spread < 0.01, f"transmitted shear still depends on the penalty: {vals}"


def test_the_tangential_term_acts_on_shear_and_barely_on_a_normal_press():
    """The discriminating contrast, stated with both numbers rather than an absolute tolerance.

    Under a shear the tangential term is everything: without it the far block does not move at all.
    Under a *confined* (uniaxial) press it is almost nothing — there is no tangential relative motion to
    act on. It is not exactly nothing, and pretending otherwise would be the wrong claim: the two blocks
    are meshed INDEPENDENTLY, so even under uniaxial strain their discrete lateral response differs by
    ~1e-6, and the slide correctly reports that. Measured here: the term moves a confined press by
    ~0.1%, an unconfined one (where the blocks genuinely bulge differently) by ~1%, and a shear by 100%.

    A projection that leaked the normal component into the tangent — the classic being a broadcast over
    a size-1 component axis — would stiffen the press by an O(1) amount and fail this, while looking
    perfectly plausible on the shear tests above.
    """
    press = 0.01
    free = _stack(conforming=False, ct=0.0, drive=(0.0, 0.0, -press), confine=True)
    stiff = _stack(conforming=False, ct=1.0e6, drive=(0.0, 0.0, -press), confine=True)
    assert abs(free[2]) > 1e-4, "sanity: the press must actually compress the stack"
    press_effect = np.abs(stiff - free).max() / abs(free[2])
    assert press_effect < 0.005, f"a confined press should barely feel the tangential term, got {press_effect:.2%}"

    sheared = _stack(conforming=False, ct=1.0e6, drive=(SHEAR, 0.0, 0.0))[0]
    unsheared = _stack(conforming=False, ct=0.0, drive=(SHEAR, 0.0, 0.0))[0]
    assert abs(unsheared) < 1e-12 and abs(sheared) > 1e-4
    # the contrast is the point: the same term is decisive for one load and inert for the other
    assert press_effect < 0.01 * (abs(sheared - unsheared) / abs(sheared))
