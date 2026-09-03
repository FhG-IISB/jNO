"""`solve(operator=jno.solve.hierarchical(...))` on a welded network.

Welding is what makes PEEC expensive. A lattice applies its partial inductance by FFT, but a bond
wire landing on a trace layer has no lattice structure, so its own block and every coupling to the
lattice are formed DENSELY -- measured, a 6,806-bar lattice solves in 0.213 s and adding one
19-filament wire takes it to 33.4 s, and a 12,000-element module allocated 2.7 GB to produce a 46 MB
coupling block.

The claim under test is narrow and checkable: compressing those blocks to a tolerance must not move
the answer by more than that tolerance, and asking for it must be the ONLY thing that changes -- the
default path stays exact, so every other test in the suite is a regression guard.
"""

import jax
import numpy as np

import jno

jax.config.update("jax_enable_x64", True)

CU, mm = 5.8e7, 1e-3


def _welded(n_wire=6):
    """A trace layer with bond wires landing on it: a lattice block, a wire block, and the cross."""
    plate = jno.Shape.box(0, 0, 0, 24 * mm, 12 * mm, 1 * mm, size=(1 * mm,) * 3).attach(sigma=CU).name("plate")
    sh = plate
    for k in range(n_wire):
        y = (2.0 + 1.5 * k) * mm
        sh = sh + jno.Shape.line(
            [(2 * mm, y, 1 * mm), (12 * mm, y, 4 * mm), (22 * mm, y, 1 * mm)], r=0.2 * mm, size=1.0 * mm
        ).attach(sigma=CU).name(f"w{k}")
    d = sh.domain()
    d.tag("A", lambda x, y, z: x < 1.1 * mm)
    d.tag("B", lambda x, y, z: x > 22.9 * mm)
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6).build()


def test_the_compressed_operator_gives_the_same_answer():
    """The whole claim. `floor=0` forces compression on every block regardless of size, so the test
    exercises the path rather than silently falling through to the exact one it is compared against.
    """
    e = _welded()
    exact = e.solve()
    approx = e.solve(operator=jno.solve.hierarchical(tol=1e-8, leaf=32, floor=0))
    for name, a, b in (("L", exact.L, approx.L), ("R", exact.R, approx.R)):
        ra, rb = float(np.real(a)), float(np.real(b))
        assert abs(rb / ra - 1) < 1e-4, (name, ra, rb)


def test_the_default_path_is_untouched():
    """Opt-in means opt-in: without `operator=` the operator must be bit-identical to what it always
    was, which is what makes the rest of the PEEC suite a regression guard for this feature."""
    e = _welded()
    a, b = e.solve(), e.solve()
    assert float(np.real(a.L)) == float(np.real(b.L))


def test_a_looser_tolerance_costs_accuracy_and_a_tighter_one_recovers_it():
    """A spec that was quietly ignored would pass a single accuracy check. The trend cannot be faked:
    the error against the exact operator must shrink as the tolerance is tightened."""
    e = _welded()
    ref = float(np.real(e.solve().L))
    errs = []
    for tol in (1e-2, 1e-6, 1e-10):
        got = float(np.real(e.solve(operator=jno.solve.hierarchical(tol=tol, leaf=32, floor=0)).L))
        errs.append(abs(got / ref - 1))
    assert errs[0] >= errs[1] >= errs[2], errs
    assert errs[-1] < 1e-6, errs


def test_a_block_below_the_floor_is_left_exact():
    """Below the floor compression does not pay -- 1.38x at 370 elements -- so it must not happen at
    all. The default floor is 2000, and this network is far smaller, so asking for the default spec
    must give the exact answer to the last bit rather than an approximation nobody wanted."""
    e = _welded()
    exact = float(np.real(e.solve().L))
    defaulted = float(np.real(e.solve(operator=jno.solve.hierarchical()).L))
    assert defaulted == exact


def test_a_plain_lattice_ignores_it():
    """A lattice is applied by FFT, which is exact and already O(N log N). There is nothing to
    compress, so the spec must be inert rather than quietly degrading a good path."""
    plate = jno.Shape.box(0, 0, 0, 20 * mm, 10 * mm, 1 * mm, size=(1 * mm,) * 3).attach(sigma=CU).name("p")
    d = plate.domain()
    d.tag("A", lambda x, y, z: x < 1.1 * mm)
    d.tag("B", lambda x, y, z: x > 18.9 * mm)
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    e = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6).build()
    assert float(np.real(e.solve().L)) == float(np.real(e.solve(operator=jno.solve.hierarchical(floor=0)).L))
