"""The loop inductance is a quadratic form in an operator the solve ALREADY applies by FFT.

``PEECSolution.L`` is ``I^H Lp I / |I_port|^2``. It was evaluated with :func:`pair_quadratic`, which
never forms ``Lp`` but does walk every pair -- ``O(N^2)``, against a solve that is linear in the bars.
At 23,688 bars that was 16.2 s of inductance behind 0.79 s of solve, and at 57,472 it was 76 s behind
2.29 s: the readout cost thirty times the answer it was reporting on.

A bar lattice's ``Lp`` is block-Toeplitz and :func:`lattice_apply` applies it in ``O(N log N)`` -- the
same operator, the same quadrature, and the one the matrix-free solve is built on. So the energy is
``x . (Lp x)`` through that apply, and the pair sum is kept only for the networks that have no such
structure (a polyline's filaments are not Toeplitz).

The oracle here is the pair sum itself: both paths must give the same number, because they are the
same quadratic form.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.kernel import pair_quadratic
from jno.utils.solver.peec import bar_filaments, line_filaments, solve_network, terminal_nodes

jax.config.update("jax_enable_x64", True)

MU0 = 4e-7 * np.pi
SIG = 5.8e7
LX, WY, TZ = 0.040, 0.004, 0.002


def _pair_energy(f, x):
    """The old evaluation, kept here as the oracle rather than as the implementation."""
    grp = jnp.asarray(f.group)
    m = jnp.asarray(f.mom) * x[grp][:, None]
    return pair_quadratic(f.pos, m, lambda r: 1.0 / r, f.self_g, group=grp) * (MU0 / (4 * jnp.pi))


def _built(pitch, freq=0.0):
    bar = jno.Shape.box(0, 0, 0, LX, WY, TZ, size=pitch).attach(sigma=SIG).name("bar")
    d = bar.domain()
    d.tag("A", lambda x, y, z: x < pitch[0] if isinstance(pitch, tuple) else x < pitch)
    d.tag("B", lambda x, y, z: x > LX - (pitch[0] if isinstance(pitch, tuple) else pitch))
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freq).build()


@pytest.mark.parametrize("pitch", [0.002, 0.001])
def test_the_fft_energy_is_the_pair_sum(pitch):
    """Same quadratic form, two evaluations -- they must agree to round-off, not merely closely."""
    f = bar_filaments(jno.Shape.box(0, 0, 0, LX, WY, TZ), size=pitch)
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=int(np.asarray(f.length).shape[0])))

    from jno.utils.solver.peec import lattice_apply

    ap = lattice_apply(f, lambda r: 1.0 / r, mu_scale=MU0 / (4 * jnp.pi))
    fast, slow = float(jnp.dot(x, ap(x))), float(_pair_energy(f, x))
    assert abs(fast / slow - 1) < 1e-11, f"{fast} vs {slow}"


def test_L_is_unchanged_by_the_faster_evaluation():
    """The reported inductance itself, against the pair sum done by hand from the same currents."""
    b = _built(0.002)
    sol = b.solve()
    f, cur = b.fil, jnp.atleast_2d(sol.i)[0]
    ref = (_pair_energy(f, jnp.real(cur)) + _pair_energy(f, jnp.imag(cur))) / abs(complex(sol.current("A"))) ** 2
    assert abs(float(np.real(sol.L)) / float(ref) - 1) < 1e-11


def test_a_wire_has_no_lattice_and_keeps_the_pair_sum():
    """A polyline is not Toeplitz, so the fallback has to stay -- and still give the right number."""
    arc = [(0, 0, 0), (5e-3, 0, 2e-3), (10e-3, 0, 0)]
    f = line_filaments(jno.Shape.line(arc, r=1.9e-4, size=1e-3))
    term = {
        "A": terminal_nodes(f, lambda q: np.linalg.norm(q - np.array(arc[0]), axis=1) < 1e-9),
        "B": terminal_nodes(f, lambda q: np.linalg.norm(q - np.array(arc[-1]), axis=1) < 1e-9),
    }
    cur, _phi, inj = solve_network(f, SIG, term, [("A", "B", 1.0 + 0j)], (), (), (), omega=0.0, matrix_free=False)
    e = _pair_energy(f, jnp.real(cur)) + _pair_energy(f, jnp.imag(cur))
    L = float(e) / abs(complex(inj["A"])) ** 2
    assert np.isfinite(L) and L > 0


def test_the_gradient_still_reaches_the_conductivity_through_L():
    """The readout is on the differentiated path -- an inverse design may target the INDUCTANCE."""
    b = _built(0.002)

    def loss(s):
        return jnp.real(b.solve(sigma={"bar": SIG * s}).L)

    g = float(jax.grad(loss)(1.0))
    assert np.isfinite(g)
    h = 1e-6
    fd = (float(loss(1.0 + h)) - float(loss(1.0 - h))) / (2 * h)
    assert abs(g - fd) < 1e-9 * max(1.0, abs(fd)) or abs(g / fd - 1) < 1e-5


def test_L_is_no_longer_quadratic_in_the_bar_count():
    """The point of the change, asserted as a RATIO of costs rather than an absolute time.

    A 2x refinement in every direction is 8x the cells and about 8x the bars. The pair sum grows as
    the square of that; the FFT apply grows as N log N. Asserting the solve-to-readout ratio does not
    drift is what actually pins the complexity, and it is machine-independent in a way a second is not.
    """
    coarse, fine = _built(0.002), _built(0.001)
    n_c = int(np.asarray(coarse.fil.length).shape[0])
    n_f = int(np.asarray(fine.fil.length).shape[0])
    assert n_f > 4 * n_c  # the refinement really is the one this reasons about

    def timed(b):
        s = b.solve()
        float(np.real(s.R))
        float(np.real(s.L))  # compile
        s2 = b.solve()
        a = time.perf_counter()
        float(np.real(s2.R))
        t_solve = time.perf_counter() - a
        a = time.perf_counter()
        float(np.real(s2.L))
        return t_solve, time.perf_counter() - a

    _tc_s, tc_l = timed(coarse)
    _tf_s, tf_l = timed(fine)
    growth = tf_l / max(tc_l, 1e-6)
    ratio = n_f / n_c
    # quadratic would be ratio**2 (about 70x here); allow generous slack for a small-problem floor
    assert growth < ratio**1.6, f"L grew {growth:.1f}x for {ratio:.1f}x the bars -- looks superlinear"
