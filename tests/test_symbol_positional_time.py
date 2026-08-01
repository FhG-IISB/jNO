"""A temporal variable passed positionally to a field symbol is dropped, not bound to `z`.

``dom.variable(tag)`` hands back ``(x, y, t)``, so ``u(*dom.variable(tag))`` is the natural
gesture. Without the filter the ``t`` would land on the ``z`` axis — harmless-looking in 2D
(there is no z) but wrong, and it broke the periodic-tie parser outright.
"""

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def _periodic_transient(splat: bool):
    """Doubly-periodic transient solve; ties+IC written with either `u(*c)` or `u(c[0], c[1])`."""
    pitch, t_end, steps = 100.0, 45.0, 4
    d = jno.Shape.rect(0.0, 0.0, pitch, pitch, size=pitch / 6).domain(time=(0.0, t_end, steps))
    d.tag("left", lambda x, _: x < 1e-4)
    d.tag("right", lambda x, _: x > pitch - 1e-4)
    d.tag("bottom", lambda _, y: y < 1e-4)
    d.tag("top", lambda _, y: y > pitch - 1e-4)

    A, pA = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    ci, cl, cr = d.variable("initial"), d.variable("left"), d.variable("right")
    cb, ct = d.variable("bottom"), d.variable("top")
    Ai, qA = A.bind(x=xi, y=yi, t=ti), pA.bind(x=xi, y=yi, t=ti)

    trace = (lambda c: A(*c)) if splat else (lambda c: A(c[0], c[1]))
    fem = jno.fem(
        [
            Ai.t * qA + 4.0 * (Ai.x * qA.x + Ai.y * qA.y) + 5.0 * Ai * Ai * qA,
            trace(cl) - trace(cr),
            trace(cb) - trace(ct),
            trace(ci) - 1.0,
        ]
    )
    return np.asarray(fem.solve().eval())


def test_positional_time_is_dropped_not_bound_to_z():
    """`u(*c)` (x, y, t) must equal `u(x, y)` — the time coordinate carries no spatial axis.

    Compared to a tight tolerance rather than bitwise. The two spellings build slightly different
    graphs, and XLA is free to reassociate them differently — on GPU it does, giving ~1 ULP (rel
    6e-16) on the output of this NONLINEAR, time-stepped solve, where Newton iterations amplify a
    last-bit difference in the assembly. That is not the failure this test guards: binding ``t`` to
    the ``z`` axis changes which coordinate the trace reads, which is a gross difference (and broke
    the periodic-tie parser outright — see the test below), never a last-bit one."""
    np.testing.assert_allclose(_periodic_transient(splat=True), _periodic_transient(splat=False), rtol=1e-12, atol=1e-12)


def test_positional_time_does_not_break_periodic_ties():
    """The splat form used to raise 'could not read an essential condition'."""
    out = _periodic_transient(splat=True)
    assert out.ndim == 2 and out.shape[0] == 4


def test_third_positional_coord_still_binds_z_in_3d():
    """The filter drops *temporal* vars only — a real z axis must still bind.

    Note `dom.variable` appends a temporal var even on a steady domain, so 3D yields
    (x, y, z, t): the splat must bind x/y/z and drop t.
    """
    d = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=0.5).domain()
    u, phi = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    cb = d.variable("boundary", split=True)  # (x, y, z, t)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)

    weak = ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - 1.0 * vi
    explicit = np.asarray(jno.fem([weak, u(cb[0], cb[1], cb[2]) - 0.0]).solve())
    splatted = np.asarray(jno.fem([weak, u(*cb) - 0.0]).solve())

    assert splatted.size > 0 and np.isfinite(splatted).all()
    assert splatted.max() > 0.0  # non-trivial solution => the z trace really bound
    # tight, not bitwise -- see the note on the 2-D case above: the two spellings build slightly
    # different graphs and XLA reassociates them differently on GPU (~1 ULP). Dropping z instead of t
    # would change which face is constrained, which is a gross difference, not a last-bit one.
    np.testing.assert_allclose(explicit, splatted, rtol=1e-12, atol=1e-12)
