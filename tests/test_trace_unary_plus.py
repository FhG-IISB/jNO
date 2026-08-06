"""``+expr`` on a trace node — the unary-plus protocol.

``-expr`` has always worked; ``+expr`` raised ``TypeError: bad operand type for unary +``. That asymmetry
bites when a term list writes a signed source symmetrically, which is the natural spelling for a
stoichiometric reaction::

    src = [-rxn, -rxn, +rxn, +rxn]      # two phases consumed, two produced

The negative legs built fine and the positive legs raised, so the failure looked like a problem with the
*model* rather than a missing dunder. ``+expr`` is the identity and returns ``self`` — no graph node.
"""

import numpy as np
import pytest

import jno
import jno.jnp_ops as J


def _views():
    """One live instance of every trace class that defines ``__neg__``, so the two stay in step."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.4).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    vec, _ = d.fem_symbols(value_shape=(2,), names=("w", "q"))
    ui = u.bind(x=xi, y=yi)
    return {
        "Placeholder": xi,
        "ScalarView": ui,
        "ScalarView.partials": ui.x,
        "VectorView": vec.bind(x=xi, y=yi),
        "MatrixView": J.hessian(ui, [xi, yi]),
        "expression": 2.0 * ui + 1.0,
    }


@pytest.mark.parametrize("name", list(_views()))
def test_unary_plus_is_the_identity(name):
    """``+x is x`` — the identity, not a ``1 * x`` node, so it costs nothing in the graph."""
    node = _views()[name]
    assert (+node) is node, f"{name}: +x should return x itself"


def test_unary_plus_matches_no_operator_numerically():
    """The whole point: a term written with ``+`` must solve identically to the same term without it."""

    def build(sign):
        d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
        u, phi = d.fem_symbols()
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
        f = J.exp(-10.0 * ((xi - 0.5) ** 2 + (yi - 0.5) ** 2))
        src = +(f * vi) if sign else f * vi  # the ONLY difference between the two problems
        return jno.fem([ui.x * vi.x + ui.y * vi.y - src, u(xb, yb) - 0.0], quad_degree=3)

    a = np.asarray(build(False).solve()).reshape(-1)
    b = np.asarray(build(True).solve()).reshape(-1)
    # Not bitwise: two independent solves, and float32 reductions on GPU are not order-deterministic.
    # Graph identity is pinned by `test_unary_plus_is_the_identity`; this checks the solve agrees.
    assert np.allclose(a, b, rtol=1e-5, atol=1e-7), "unary + changed the solution"


def test_signed_source_list_builds():
    """The spelling that motivated this: a stoichiometric source list mixing both signs."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain(time=(0.0, 0.1, 3))
    a, pa = d.fem_symbols(names=("a", "pa"))
    b, pb = d.fem_symbols(names=("b", "pb"))
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    A, B = a.bind(x=xi, y=yi, t=ti), b.bind(x=xi, y=yi, t=ti)
    PA, PB = pa.bind(x=xi, y=yi), pb.bind(x=xi, y=yi)
    rxn = 2.0 * A * B
    src = [-rxn, +rxn]  # consumed / produced -- the +rxn leg used to raise
    fem = jno.fem(
        [
            A.t * PA + (A.x * PA.x + A.y * PA.y) - src[0] * PA,
            B.t * PB + (B.x * PB.x + B.y * PB.y) - src[1] * PB,
            a(ci[0], ci[1]) - 1.0,
            b(ci[0], ci[1]) - 0.0,
        ]
    )
    assert fem.is_transient
    sol = fem.solve()  # transient: a trace node, so evaluate it rather than np.asarray-ing the node
    traj = np.asarray(jno.core([sol.mean], domain=d).eval([sol]))
    assert np.isfinite(traj).all()
    assert traj.shape[0] > 1, "expected a trajectory over the time grid"
