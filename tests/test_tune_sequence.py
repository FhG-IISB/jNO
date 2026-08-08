"""The ``sequence`` axis — parameter continuation through the ``tune`` vocabulary.

The agreed home for sweeps / load stepping / homotopy (2026-08-08): ``space.sequence(name, values)``
declares an **ordered, warm-started** axis, and ``crux.sweep`` (grid mode) marches it through the
internal continuation engine, while ``unique``/``float_range`` stay independent search trials. A
``fem.solve()`` kwarg for this was built, reviewed, and removed — one sweep vocabulary.
"""

import jax
import jax.numpy as jnp
import jno.numpy as jnn
import numpy as np
import pytest

import jno
from jno.utils.solver.solver_api import ContinuationSpec, run_continuation


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson():
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    u, v = d.fem_symbols()
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    kap = jno.np.parameter((1,), name="kap")
    q = jno.np.parameter((1,), name="q").initialize(jax.nn.initializers.constant(1.0))
    fem = jno.fem([kap * (ui.x * vi.x + ui.y * vi.y) - (q + 0.0) * vi, u(xb, yb) - 0.0])
    return d, fem


def test_a_sequence_axis_marches_through_crux_sweep():
    """``crux.sweep`` on a space with only a sequence axis returns the warm-started family, identical to
    the continuation engine called directly — and the parameter NOT being marched (``q``) is held at its
    current value, which is the lazy-solve contract."""
    d, fem = _poisson()
    crux = jno.core([(fem.solve() * 0.0).mae], domain=d)

    ks = jnp.linspace(0.5, 2.0, 4)
    space = jnn.tune.space()
    space.sequence("kap", ks)
    U = np.asarray(crux.sweep(space))

    ref = np.asarray(
        run_continuation(fem, ContinuationSpec(params={"kap": np.asarray(ks)}, keep="all"), kwargs={"q": jnp.array([1.0])})
    )
    assert U.shape == ref.shape and U.shape[0] == 4
    assert U == pytest.approx(ref, abs=1e-12), f"tune march != engine: {np.abs(U - ref).max():.2e}"
    # physics: u ~ q/kap, so the family ends sit at a 4x ratio
    assert float(U[0].max() / U[-1].max()) == pytest.approx(4.0, rel=1e-6)


def test_keep_last_is_the_homotopy_spelling():
    d, fem = _poisson()
    crux = jno.core([(fem.solve() * 0.0).mae], domain=d)
    ks = jnp.linspace(0.5, 2.0, 4)
    space = jnn.tune.space()
    space.sequence("kap", ks, keep="last")
    u_last = np.asarray(crux.sweep(space))
    space2 = jnn.tune.space()
    space2.sequence("kap", ks)
    assert u_last == pytest.approx(np.asarray(crux.sweep(space2))[-1], abs=0.0)


def test_the_walls_fail_loud():
    d, fem = _poisson()
    crux = jno.core([(fem.solve() * 0.0).mae], domain=d)
    space = jnn.tune.space()
    space.sequence("kap", jnp.array([1.0, 2.0]))

    with pytest.raises(NotImplementedError, match="reorder"):
        crux.sweep(space, optimizer="OnePlusOne", budget=3)

    space_mixed = jnn.tune.space()
    space_mixed.sequence("kap", jnp.array([1.0, 2.0]))
    space_mixed.unique("epochs", [10, 20])
    with pytest.raises(NotImplementedError, match="march per trial"):
        crux.sweep(space_mixed)

    # a core with no FEM behind its constraints has nothing to march
    d2 = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.4).domain()
    x = d2.variable("interior", split=True)[0]
    crux_nofem = jno.core([(x * 0.0).mae], domain=d2)
    space3 = jnn.tune.space()
    space3.sequence("kap", jnp.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="exactly one parametric fem.solve"):
        crux_nofem.sweep(space3)
