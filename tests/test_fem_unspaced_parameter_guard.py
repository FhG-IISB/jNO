"""A coefficient array bound to no FE space is refused, not silently reduced to its first entry.

The assembler packs a runtime parameter per element. A parameter declared on an FE space knows
which value belongs to which element -- P0 gathers by cell index, P1 by the cell's nodes. One
declared with a bare SHAPE has no such map, and the packing took ``flat[:1]``: the first value,
broadcast over the whole mesh.

That is the worst kind of wrong. On a 2789-cell electro-thermal source spanning nine orders of
magnitude it returned 361.19 K where the same 2789 values, declared on P0, give 355.72 K -- a
plausible temperature, five kelvin out, with nothing said.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.linear import sparse_lu_solve

jax.config.update("jax_enable_x64", True)

L, H = 1.0, 0.25


def domain():
    d = jno.Shape.rect(0, 0, L, H, size=0.1).domain()
    _ = d.mesh
    return d


def poisson(d, coeff):
    """-div(grad u) = coeff, pinned at the left edge."""
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    return jno.fem([ui.x * vi.x + ui.y * vi.y - coeff * vi, u(xl, yl) - 0.0])


def peak(fem, vals):
    a, b = fem.operator.evaluate({"q": jnp.asarray(vals)})
    return float(jnp.max(sparse_lu_solve(a, jnp.asarray(b).reshape(-1))))


def test_an_array_on_no_space_is_refused_by_name_and_count():
    d = domain()
    n = int(d._cells_p1().shape[0])
    fem = poisson(d, jno.np.parameter((n,), name="q"))
    with pytest.raises(ValueError, match=rf"parameter 'q', which carries {n} values.*no FE space"):
        peak(fem, np.linspace(1.0, 2.0, n))


def test_the_message_names_the_spelling_that_works():
    d = domain()
    n = int(d._cells_p1().shape[0])
    fem = poisson(d, jno.np.parameter((n,), name="q"))
    with pytest.raises(ValueError, match=r"space='P0'"):
        peak(fem, np.ones(n))


def test_a_genuine_scalar_parameter_still_works():
    """Shape (1,) is a scalar coefficient and must stay unaffected -- the guard is about arrays."""
    d = domain()
    fem = poisson(d, jno.np.parameter((1,), name="q"))
    one, two = peak(fem, [1.0]), peak(fem, [2.0])
    assert one > 0
    assert two == pytest.approx(2 * one, rel=1e-9)  # linear in the source, so exactly double


def test_a_p0_parameter_carries_its_own_value_per_element():
    """The spelling the message points at: the values must actually differ across the mesh."""
    d = domain()
    n = int(d._cells_p1().shape[0])
    _r, s0 = d.fem_symbols(space="P0", names=("r", "s"))
    fem = poisson(d, jno.np.parameter(s0, name="q"))
    # the same TOTAL source, piled onto the half of the mesh AWAY from the pinned edge. A per-cell
    # field must notice; a first-entry broadcast could not tell the two apart.
    x = np.asarray(d._points)[np.asarray(d._cells_p1())].mean(axis=1)[:, 0]
    flat, piled = peak(fem, np.full(n, 1.0)), peak(fem, np.where(x > L / 2, 2.0, 0.0))
    assert abs(piled / flat - 1) > 0.05
