"""`domain.rebuild(when=...)`: a shape whose coordinates carry a design is built once and stretched; when
the condition holds during `jno.core.solve`, the domain, its FEM mesh and its PEEC grid are built again
at the current design and training continues from the same parameters.

Oracle: after a rebuild, the problem solved in place equals a FRESH build whose reference is the design
the rebuild happened at -- same mesh, same solution at any design -- and its cells are the fresh build's
cells (the stretch is gone). Also: the optimizer state is kept or restarted on request, and what cannot
rebuild is refused by name.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno
from jno._fem import _walk
from jno.trace_evaluator import TraceEvaluator

jax.config.update("jax_enable_x64", True)

mm, Q, H = 1e-3, 1e6, 0.7e-3


def _param(name, value):
    p = jno.np.parameter((1,), name=name)
    p.dtype(jnp.float64)
    p.initialize(jax.nn.initializers.constant(value))
    return p


def _blocks(y1):
    """A cooled block (y 0..4 mm) under a heated one (y 4 mm..y1): T_max grows with y1."""
    a = jno.shape.box(0, 0, 0, 10 * mm, 4 * mm, 1 * mm, size=H).name("a").attach(k=1.0, q=0.0)
    b = jno.shape.box(0, 4 * mm, 0, 10 * mm, y1, 1 * mm, size=H).name("b").attach(k=1.0, q=Q)
    d = (a + b).domain()
    d.tag("cold", lambda x, y, z: y < 1e-9)
    T, s = d.fem_symbols()
    x = d.variable("interior", split=True)[:3]
    Ti, si = T.bind(x=x[0], y=x[1], z=x[2]), s.bind(x=x[0], y=x[1], z=x[2])
    return d, jno.fem(
        [d.k * (Ti.x * si.x + Ti.y * si.y + Ti.z * si.z) - d.q * si, T(*d.variable("cold", split=True)[:3]) - 0.0]
    )


def _at(node, value):
    """``node`` evaluated with every parameter in it set to ``value``."""
    mcs = [m for m in _walk(node) if type(m).__name__ == "ModelCall"]
    table = {m.model.layer_id: eqx.tree_at(lambda q: q.value, m.model.module, jnp.asarray([value])) for m in mcs}
    return np.asarray(TraceEvaluator(table).evaluate(node, context={}))


def _reference(p):
    (mc,) = [m for m in _walk(p) if type(m).__name__ == "ModelCall"]
    return mc.model._rebuild_reference


def test_a_rebuild_is_a_fresh_build_at_the_design_it_happened_at():
    """sgd shortens the heated block (dT_max/dy1 > 0) until its cells pass aspect 4; the rebuilt problem
    then equals one built from scratch with that y1 as reference -- mesh, cells and solution."""
    y1 = _param("y1", 8 * mm)
    y1.optimizer(optax.sgd(6e-8))
    d, fem = _blocks(y1)
    worst = d.cell_aspect().max
    assert float(_at(worst, 8 * mm)) < 4.0 < float(_at(worst, 6 * mm))
    d.rebuild(when=worst > 4.0)
    crux = jno.core([jno.np.max(fem.solve())])
    crux.solve(7)
    # the first crossing is clear (aspect 4.27 after epoch 4); later ones sit within rounding of the threshold
    assert crux._rebuild_epochs and crux._rebuild_epochs[0] == 4, crux._rebuild_epochs
    assert crux._total_epochs == 7  # the stretches add up to the epochs asked for
    y_ref = float(np.asarray(_reference(y1)).reshape(-1)[0])
    assert 4.5 * mm < y_ref < 7.5 * mm
    y_now = float(np.asarray(crux.eval([y1])[0]).reshape(-1)[0])
    assert y_now < y_ref  # training went on after the rebuild

    fresh = _param("y1_fresh", y_ref)
    d2, fem2 = _blocks(fresh)
    np.testing.assert_allclose(np.asarray(d.mesh.points), np.asarray(d2.mesh.points), rtol=0, atol=1e-15)
    assert float(_at(worst, y_ref)) == pytest.approx(float(_at(d2.cell_aspect().max, y_ref)), rel=1e-12)
    assert float(_at(worst, y_ref)) < 4.0
    for yy in (y_ref, y_now):
        assert float(np.max(_at(fem.solve(), yy))) == pytest.approx(float(np.max(_at(fem2.solve(), yy))), rel=1e-10)


@pytest.mark.parametrize("keep", [True, False])
def test_the_optimizer_state_is_kept_unless_asked(keep):
    """Adam's step count runs on across a rebuild by default, and restarts with keep_opt_state=False."""
    y1 = _param("y1", 8 * mm)
    y1.optimizer(optax.adam(4e-4))
    d, fem = _blocks(y1)
    d.rebuild(when=d.cell_aspect().max > 4.0, keep_opt_state=keep)
    crux = jno.core([jno.np.max(fem.solve())])
    crux.solve(10)
    assert _reference(y1) is not None, "no rebuild happened"
    (count,) = {
        int(v) for v in jax.tree_util.tree_leaves(crux._opt_states) if jnp.issubdtype(jnp.asarray(v).dtype, jnp.integer)
    }
    assert (count == 10) == keep


def test_what_cannot_rebuild_is_refused_by_name():
    y1 = _param("y1", 8 * mm)
    d, _fem = _blocks(y1)
    with pytest.raises(TypeError, match="trace node over the design"):
        d.rebuild(when=True)
    with pytest.raises(TypeError):
        d.rebuild()  # `when` is required
    plain = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.5).name("a").domain()
    with pytest.raises(ValueError, match="no `jno.np.parameter` in its coordinates"):
        plain.rebuild(when=plain.cell_aspect().max > 2)
    assert not hasattr(d, "stretch"), "d.stretch() is replaced by rebuild conditions"


def test_a_peec_network_rebuilds_with_its_domain():
    """A face of the second trace moves; `em.cell_aspect()` reads its grid and a rebuild re-discretises
    the network at the design: its readouts made before the rebuild then solve the new network, which
    equals a network built from scratch at that reference."""
    P = 1.0 * mm

    def network(y0):
        a = jno.shape.box(0, 0, 0, 12 * mm, 2 * mm, 0.5 * mm, size=(P,) * 3).attach(sigma=5.8e7).name("t1")
        b = jno.shape.box(0, y0, 0, 12 * mm, 7 * mm, 0.5 * mm, size=(P,) * 3).attach(sigma=5.8e7).name("t2")
        d = (a + b).domain()
        d.tag("A", lambda x, y, z: x < 0.9 * mm)
        d.tag("B", lambda x, y, z: x > 11.1 * mm)
        i, v = d.peec_symbols()
        at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]  # noqa: E731
        return d, jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6, grid=jno.peec.faces(pitch=P), matrix_free=False)

    y0 = _param("y0", 5 * mm)
    y0.optimizer(optax.sgd(1.5e-6))  # a wider gap is a larger loop: descent on L closes the gap
    d, em = network(y0)
    L = em.solve().L
    worst = em.cell_aspect().max
    a_ref = float(_at(worst, 5 * mm))
    assert float(_at(worst, 3.6 * mm)) > a_ref  # the gap's cells shrink in y as it closes
    d.rebuild(when=worst > 1.3 * a_ref)  # 2.6: the narrowed trace's cells, which a fresh grid splits
    crux = jno.core([L * 1e9])
    crux.solve(6)
    assert crux._rebuild_epochs, "no rebuild happened"
    y_ref = float(np.asarray(_reference(y0)).reshape(-1)[0])
    assert y_ref < 5 * mm

    d2, em2 = network(_param("y0_fresh", y_ref))
    L2 = em2.solve().L
    for yy in (y_ref, y_ref - 0.1 * mm):
        assert float(np.real(_at(L, yy))) == pytest.approx(float(np.real(_at(L2, yy))), rel=1e-10)
    assert float(_at(worst, y_ref)) == pytest.approx(float(_at(em2.cell_aspect().max, y_ref)), rel=1e-12)
