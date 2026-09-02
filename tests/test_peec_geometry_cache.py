"""What depends on GEOMETRY alone is built once per network, not once per solve.

The partial-inductance apply and the near-field triplets are functions of where the metal is: not of
the conductivity, not of the frequency. They were rebuilt on every `solve()` anyway -- about a
quarter of a welded module solve, paid again at every point of a frequency sweep and every iteration
of a design loop, to produce the same arrays each time. Measured on a 27,533-element module: first
solve 48.9 s, repeat 30.7 s.

The Krylov subspace size is chosen the same way -- by structure, not by a single number that suits
neither case:

    welded module, 27,533 elements    restart 16  49.5 s    restart 64  32.8 s   1.5x FASTER
    plain lattice, 10,048 bars        restart 16   0.39 s   restart 64   1.12 s  2.9x SLOWER
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
import jno.utils.solver.peec as P
from jno.utils.solver.peec import bar_filaments, line_filaments, solve_network, terminal_nodes

jax.config.update("jax_enable_x64", True)

SIG = 5.8e7
QUAD_W = 3 * 2**2  # the sub-point count a bar lattice carries; see jno.peec._QUAD / _QUAD_T


def _lattice():
    return bar_filaments(jno.Shape.box(0, 0, 0, 0.020, 0.004, 0.001), size=(0.002, 0.002, 0.001), sigma=SIG)


def _welded():
    """A bar lattice with a wire welded to it -- the case the near field is built for."""
    from jno.peec import _weld

    fb = _lattice()
    nb = int(np.asarray(fb.length).size)
    wires = [jno.Shape.line([(0.004, 0.002, 0.001), (0.004, 0.002, 0.003), (0.016, 0.002, 0.001)], r=2e-4, size=0.002)]
    fl = line_filaments(wires, quad=QUAD_W)
    nl = int(np.asarray(fl.length).size)
    box = jno.Shape.box(0, 0, 0, 0.020, 0.004, 0.001)
    fil, sg = _weld([(fb, jnp.full(nb, SIG)), (fl, jnp.full(nl, SIG))], [[box], wires])
    return fil, sg


def _port(fil, sigma, **kw):
    p = np.asarray(fil.nodes)
    a = terminal_nodes(fil, lambda q: q[:, 0] < p[:, 0].min() + 1e-9)
    b = terminal_nodes(fil, lambda q: q[:, 0] > p[:, 0].max() - 1e-9)
    return solve_network(fil, sigma, {"A": a, "B": b}, [("A", "B", 1.0 + 0j)], omega=2 * np.pi * 1e6, **kw)


def _counted(name):
    """Wrap a builder so the test can see how often it actually runs."""
    orig, calls = getattr(P, name), []

    def wrap(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    return orig, wrap, calls


def test_a_repeat_solve_does_not_rebuild_the_operator():
    """The whole point: geometry work happens once per network, not once per solve."""
    for name in ("welded_apply", "near_block"):
        fil, sg = _welded()  # a FRESH network each time, or the previous loop already cached it
        orig, wrap, calls = _counted(name)
        setattr(P, name, wrap)
        try:
            _port(fil, sg)
            _port(fil, sg)
            _port(fil, sg)
        finally:
            setattr(P, name, orig)
        assert len(calls) == 1, f"{name} ran {len(calls)} times over three solves"


def test_the_answer_does_not_change_across_repeat_solves():
    """A cache that changed the answer would be worse than no cache."""
    fil, sg = _welded()
    first = complex(1.0 / _port(fil, sg)[2]["A"])
    for _ in range(2):
        assert abs(complex(1.0 / _port(fil, sg)[2]["A"]) / first - 1) < 1e-12


def test_a_changed_conductivity_still_reuses_the_geometry():
    """Sigma is not geometry, so it must NOT invalidate the operator -- that is the design-loop case."""
    fil, sg = _welded()
    _port(fil, sg)  # warm the cache first, so the count below is about SIGMA and nothing else
    orig, wrap, calls = _counted("welded_apply")
    setattr(P, "welded_apply", wrap)
    try:
        a = complex(1.0 / _port(fil, sg)[2]["A"])
        b = complex(1.0 / _port(fil, sg * 0.5)[2]["A"])
    finally:
        setattr(P, "welded_apply", orig)
    assert len(calls) == 0  # geometry unchanged, so never rebuilt
    # and sigma really did reach the answer. Halving it multiplies R by sqrt(2), not 2: at 1 MHz the
    # elements take the SURFACE impedance, which goes as sqrt(rho), not the DC rho * l / A.
    assert 1.35 < b.real / a.real < 1.50, b.real / a.real


def test_nothing_is_kept_from_inside_a_trace():
    """Under jit even a concrete geometry yields tracers, and keeping a closure over them leaks.

    The guard is `jnp.zeros(())` in `_frozen_geometry`; without it this raises a leaked-tracer error
    on the second trace rather than failing an assertion here.
    """
    fil, sg = _welded()
    assert P._frozen_geometry(fil)  # outside a trace, cacheable

    seen = []

    def inside(s):
        seen.append(P._frozen_geometry(fil))
        return jnp.real(jnp.asarray(1.0 / _port(fil, s)[2]["A"]))

    jax.jit(inside)(sg)
    assert seen and not any(seen), "the cache must stand down inside a trace"


def test_the_krylov_subspace_is_chosen_by_structure():
    """A welded network needs the larger subspace; a lattice is 2.9x slower with it."""
    lat = _lattice()
    fil, sg = _welded()
    seen = {}
    orig = P.jax.scipy.sparse.linalg.gmres

    def spy(*a, **k):
        seen.setdefault("restart", k.get("restart"))
        return orig(*a, **k)

    P.jax.scipy.sparse.linalg.gmres = spy
    try:
        seen.clear()
        _port(lat, jnp.full(int(np.asarray(lat.length).size), SIG))
        lattice_rs = seen.get("restart")
        seen.clear()
        _port(fil, sg)
        welded_rs = seen.get("restart")
    finally:
        P.jax.scipy.sparse.linalg.gmres = orig
    # `_RS = min(restart, ne + nn)`, and these test networks are small enough to be clamped -- the
    # claim under test is that the two DIFFER by structure, not the exact value on a toy problem.
    assert lattice_rs == 16, lattice_rs
    assert welded_rs > lattice_rs, (lattice_rs, welded_rs)
