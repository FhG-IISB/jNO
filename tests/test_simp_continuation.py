"""SIMP penalisation continuation — Jung, Yun & Kim, *Comput. Struct.* **331** (2026), Fig. 4a.

The schedule is *conditional*, not a fixed ramp: ``penal`` rises only when the objective has
settled AND the design is still grey. Both halves are tested, because either one alone would give
a plausible-looking but wrong schedule — a pure ramp binarises before the topology exists, and a
grey-only trigger fires while the objective is still moving.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
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


def _make(rho_values, *, physical=None, **kw):
    """A continuation hook plus the (grads, trainable) a step would hand it."""
    d = jno.domain.from_array({"_": np.zeros((1, 1))})
    rho = jno.np.parameter((len(rho_values),), name="rho_c")
    penal = jno.np.parameter((1,), name="penal_c")
    cont = jno.optimizers.simp_continuation(penal, rho, physical=physical, **kw)
    trainable = {
        rho.model.layer_id: jnp.asarray(rho_values, dtype=jnp.float64),
        penal.model.layer_id: jnp.asarray([cont.penal]),
    }
    grads = {k: jnp.zeros_like(v) for k, v in trainable.items()}
    return cont, grads, trainable, penal.model.layer_id


def _step(cont, grads, trainable, lid, loss, epoch):
    """One hook call; returns the value `sgd(1.0)` would land on penal."""
    out = cont.on_before_update(grads=grads, trainable=trainable, epoch=epoch, total_loss=loss)
    return float(trainable[lid][0] - out[lid][0])


class TestTrigger:
    def test_it_rises_only_once_the_objective_has_settled(self):
        grey = np.full(20, 0.5)  # M_nd = 1.0, as grey as a design gets
        cont, g, t, lid = _make(grey, tol=1e-4, window=3)

        # A moving objective must not trigger it, however grey the design is.
        for e, loss in enumerate([100.0, 50.0, 25.0, 12.0, 6.0]):
            assert _step(cont, g, t, lid, loss, e) == 3.0
        assert cont.penal == 3.0 and cont.history == []

        # Three consecutive relative changes under tol: now it fires, once.
        settled = [6.0, 6.0, 6.0, 6.0]
        landed = [_step(cont, g, t, lid, v, 5 + i) for i, v in enumerate(settled)]
        assert cont.penal == 4.0, "a converged, still-grey design must raise penal"
        # It fires on the step where the window first closes, not on the last one — after that
        # the returned delta is zero, which is what leaves penal alone.
        assert max(landed) == 4.0, "sgd(1.0) must land exactly on the new value"
        assert landed[-1] == 3.0, "and a non-firing step must return a zero delta"
        assert len(cont.history) == 1 and cont.history[0][1] == 4.0

    def test_a_binary_design_never_raises_it(self):
        """The second half of the condition. M_nd below tolerance means the job is done."""
        binary = np.array([1.0, 0.0] * 10)  # M_nd = 0
        cont, g, t, lid = _make(binary)
        for e in range(12):
            _step(cont, g, t, lid, 5.0, e)  # perfectly converged the whole time
        assert cont.m_nd == pytest.approx(0.0)
        assert cont.penal == 3.0 and cont.history == []

    def test_the_window_restarts_after_each_step(self):
        """Otherwise one convergence would ratchet penal up on every subsequent epoch."""
        cont, g, t, lid = _make(np.full(20, 0.5), window=3)
        for e in range(4):
            _step(cont, g, t, lid, 5.0, e)
        assert cont.penal == 4.0
        _step(cont, g, t, lid, 5.0, 4)  # immediately after: the window is empty again
        assert cont.penal == 4.0, "penal must not rise on consecutive epochs"
        for e in range(5, 9):
            _step(cont, g, t, lid, 5.0, e)
        assert cont.penal == 5.0, "and must rise again once a fresh window converges"

    def test_the_ceiling_holds(self):
        cont, g, t, lid = _make(np.full(20, 0.5), window=1, maximum=5.0)
        for e in range(60):
            _step(cont, g, t, lid, 5.0, e)
        assert cont.penal == 5.0, "a design that never binarises must not run the exponent away"


class TestPhysicalDensity:
    def test_m_nd_is_measured_after_the_reparameterisation(self):
        """`trainable` holds the RAW design density: a `constrain(...)` wrapper lives in the static
        half and cannot be applied here. Measuring the raw field would call this design binary
        while the field the stiffness sees is entirely grey."""
        raw = np.array([1.0, 0.0] * 10)                      # M_nd(raw) = 0
        half = lambda r: jnp.full_like(r, 0.5)  # noqa: E731  # M_nd(physical) = 1

        plain, g, t, lid = _make(raw)
        _step(plain, g, t, lid, 5.0, 0)
        assert plain.m_nd == pytest.approx(0.0)

        mapped, g, t, lid = _make(raw, physical=half)
        _step(mapped, g, t, lid, 5.0, 0)
        assert mapped.m_nd == pytest.approx(1.0), "M_nd must follow the physical density"


def test_penal_as_a_runtime_exponent_scales_the_stiffness():
    """The whole scheme rests on `penal` being a runtime parameter — changeable with no rebuild.

    Checked against the exact analytic scaling: a uniform density `r` with exponent `p` must give
    a stiffness `r**p` times the solid one.
    """
    inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
    lam, mu = 0.3 / (1 - 0.09), 1 / 2.6
    d = jno.Shape.rect(0, 0, 2, 1, size=0.5).domain()
    n_cells = int(d._cells_p1().shape[0])
    u, phi = d.fem_symbols(value_shape=(2,))
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    rho = jno.np.parameter(s, name="rho")
    penal = jno.np.parameter((1,), name="penal")
    eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
    fem = jno.fem(
        [
            rho**penal * (lam * trace(eu) * trace(ep) + 2 * mu * inner(eu, ep, n_contract=2)),
            u(xl, yl) - (0.0, 0.0),
        ],
        quad_degree=2,
    )

    def k(vals, p):
        a, _ = fem.operator.evaluate({"rho": jnp.asarray(vals), "penal": jnp.asarray([float(p)])})
        return np.asarray(jnp.asarray(a.todense()))

    k1 = k(np.ones(n_cells), 1.0)
    eye = np.eye(k1.shape[0])
    free = np.setdiff1d(np.arange(k1.shape[0]), np.where(np.all(np.isclose(k1, eye), axis=1))[0])
    solid = np.linalg.norm(k1[np.ix_(free, free)])
    for p in (1.0, 2.0, 3.0, 4.0, 5.0):
        got = np.linalg.norm(k(np.full(n_cells, 0.5), p)[np.ix_(free, free)]) / solid
        assert got == pytest.approx(0.5**p, rel=1e-10), f"penal={p} did not scale the stiffness"


class TestGeometricDecay:
    """eq. (41): ``beta_iter = gamma * beta_(iter-1)``, every iteration."""

    @staticmethod
    def _rig(gamma, **kw):
        b = jno.np.parameter((1,), name="beta_c")
        dec = jno.optimizers.geometric_decay(b, gamma, **kw)
        lid = b.model.layer_id
        trainable = {lid: jnp.asarray([dec.value])}
        return dec, {lid: jnp.zeros((1,))}, trainable, lid

    def test_it_decays_geometrically(self):
        dec, g, t, lid = self._rig(0.9, start=1.0)
        for e in range(10):
            dec.on_before_update(grads=g, trainable=t, epoch=e)
        assert dec.value == pytest.approx(0.9**10, rel=1e-12)
        assert len(dec.history) == 10

    def test_the_step_lands_the_new_value(self):
        dec, g, t, lid = self._rig(0.5, start=1.0)
        out = dec.on_before_update(grads=g, trainable=t, epoch=0)
        assert float(t[lid][0] - out[lid][0]) == pytest.approx(0.5), "sgd(1.0) must land beta*gamma"

    def test_the_floor_holds(self):
        """Without a floor the barrier eventually vanishes and the design can cross P*."""
        dec, g, t, lid = self._rig(0.5, start=1.0, minimum=0.01)
        for e in range(50):
            dec.on_before_update(grads=g, trainable=t, epoch=e)
        assert dec.value == pytest.approx(0.01)

    def test_gamma_one_is_no_decay(self):
        dec, g, t, lid = self._rig(1.0, start=0.2)
        for e in range(5):
            dec.on_before_update(grads=g, trainable=t, epoch=e)
        assert dec.value == pytest.approx(0.2)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_an_out_of_range_gamma_is_refused(self, bad):
        b = jno.np.parameter((1,), name="beta_bad")
        with pytest.raises(ValueError, match="gamma must be in"):
            jno.optimizers.geometric_decay(b, bad)


def test_the_continuation_can_watch_one_loss_term():
    """Under a decaying barrier the TOTAL objective keeps moving because beta does, so a window
    on the total never closes. `watch=` follows the compliance term alone — Fig. 4a's question."""
    d = jno.domain.from_array({"_": np.zeros((1, 1))})
    rho = jno.np.parameter((20,), name="rho_w")
    penal = jno.np.parameter((1,), name="penal_w")
    cont = jno.optimizers.simp_continuation(penal, rho, window=3, watch=0)
    lid = penal.model.layer_id
    trainable = {rho.model.layer_id: jnp.full(20, 0.5), lid: jnp.asarray([3.0])}
    grads = {k: jnp.zeros_like(v) for k, v in trainable.items()}

    # Term 0 (compliance) is settled; term 1 (the barrier) drifts, so the TOTAL never settles.
    for e in range(6):
        cont.on_before_update(
            grads=grads, trainable=trainable, epoch=e,
            total_loss=100.0 - e,                       # moving
            individual_losses=np.array([5.0, 95.0 - e]),  # term 0 settled
        )
    assert cont.penal > 3.0, "watching a settled term must let the continuation fire"


class TestIntervalStride:
    """``every=n`` samples the convergence window over an *interval*, as the paper's test does."""

    @staticmethod
    def _rig(**kw):
        rho = jno.np.parameter((20,), name="rho_e")
        penal = jno.np.parameter((1,), name="penal_e")
        cont = jno.optimizers.simp_continuation(penal, rho, **kw)
        lid = penal.model.layer_id
        t = {rho.model.layer_id: jnp.full(20, 0.5), lid: jnp.asarray([cont.penal])}
        return cont, {k: jnp.zeros_like(v) for k, v in t.items()}, t, lid

    def test_it_owns_the_penal_gradient_on_every_step(self):
        """The bug this stride first introduced, and the reason it is a regression test.

        The hook OWNS the ``penal`` entry. On any step where it does not overwrite it, the raw
        loss gradient w.r.t. ``penal`` survives and the injected ``sgd(1.0)`` applies it, so
        ``penal`` random-walks and the whole run diverges — measured as compliance 79 -> 3.7e10
        on the cantilever. Striding may skip the bookkeeping; it must never skip the write.
        """
        cont, _g, t, lid = self._rig(every=25, window=3)
        loss_grad = {k: jnp.full_like(v, 7.0) for k, v in t.items()}  # a large, nonzero gradient
        for e in range(1, 25):  # every step that is NOT a sampling step
            out = cont.on_before_update(grads=loss_grad, trainable=t, epoch=e, total_loss=5.0)
            assert float(out[lid][0]) == 0.0, f"epoch {e}: penal gradient was not overwritten"

    def test_the_window_spans_the_stride(self):
        """With every=10 and window=3 the test covers 30 iterations, not 3."""
        cont, g, t, lid = self._rig(every=10, window=3)
        for e in range(30):  # only epochs 0, 10, 20 are sampled -> 3 samples, one short
            cont.on_before_update(grads=g, trainable=t, epoch=e, total_loss=5.0)
        assert cont.penal == 3.0, "three samples is one short of closing a window of three"
        cont.on_before_update(grads=g, trainable=t, epoch=30, total_loss=5.0)
        assert cont.penal == 4.0, "the fourth sample closes it"

    def test_a_stride_makes_a_slow_drift_look_converged(self):
        """The point of the stride is a LOOSER test — but it must still be the real quantity.

        A drift too small to register per interval is what "converged" means at that resolution.
        """
        cont, g, t, lid = self._rig(every=5, window=3, tol=1e-4)
        loss = 100.0
        for e in range(21):
            cont.on_before_update(grads=g, trainable=t, epoch=e, total_loss=loss)
            loss -= 1e-4  # 5e-4 per interval on a value of 100 -> 5e-6 relative, under tol
        assert cont.penal > 3.0


class TestStallFallback:
    """`patience`: raise the exponent when the objective never settles.

    The paper's trigger assumes the objective converges. An unregularised topology-optimisation
    run fragments instead of converging, and an MMA iterate can oscillate above `tol` forever;
    in both cases `penal` sits at `start` for the whole run and the design has no reason to go
    binary. Measured on the half-MBB beam: 800 iterations, five sampled intervals, `penal` never
    moved off 3.0 and the design ended at M_nd = 0.37. Penalisation is needed MOST where
    convergence is worst, so the gate must not be the only route.
    """

    @staticmethod
    def _falling(n):
        """A loss that always decreases -- never converges under any finite tol."""
        return [100.0 * (0.97**i) for i in range(n)]

    def test_a_stalled_run_never_raises_without_patience(self):
        cont, g, t, lid = _make(np.full(20, 0.5), tol=1e-4, window=3)
        for e, loss in enumerate(self._falling(40)):
            _step(cont, g, t, lid, loss, e)
        assert cont.penal == 3.0 and cont.history == [], "the paper's rule, unchanged by default"

    def test_patience_raises_a_stalled_run(self):
        cont, g, t, lid = _make(np.full(20, 0.5), tol=1e-4, window=3, patience=3)
        landed = [_step(cont, g, t, lid, loss, e) for e, loss in enumerate(self._falling(9))]
        assert cont.penal == 6.0, "one raise per `patience` samples while the objective stalls"
        assert [h[1] for h in cont.history] == [4.0, 5.0, 6.0]
        assert all(h[2] == "stalled" for h in cont.history), "and each is recorded as a stall"
        # `_step` reads the harness's frozen penal entry (3.0), so a firing step returns
        # 3.0 + step and a quiet one returns 3.0 exactly -- one +1 delta per raise, no more.
        assert set(landed) == {3.0, 4.0}
        assert landed.count(4.0) == 3, "exactly one non-zero delta per raise"

    def test_convergence_still_takes_precedence_and_is_labelled(self):
        """A run that genuinely settles must fire on that, not be mislabelled a stall."""
        cont, g, t, lid = _make(np.full(20, 0.5), tol=1e-4, window=3, patience=999)
        for e in range(4):
            _step(cont, g, t, lid, 5.0, e)
        assert cont.penal == 4.0
        assert cont.history[0][2] == "converged"

    def test_patience_still_respects_the_grey_gate(self):
        """M_nd is the other half of the condition; a binary design needs no more penalisation."""
        cont, g, t, lid = _make(np.array([1.0, 0.0] * 10), patience=1)
        for e, loss in enumerate(self._falling(20)):
            _step(cont, g, t, lid, loss, e)
        assert cont.penal == 3.0 and cont.history == []

    def test_the_ceiling_holds_under_patience(self):
        cont, g, t, lid = _make(np.full(20, 0.5), patience=1, maximum=5.0)
        for e, loss in enumerate(self._falling(40)):
            _step(cont, g, t, lid, loss, e)
        assert cont.penal == 5.0
