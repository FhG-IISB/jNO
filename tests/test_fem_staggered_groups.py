"""``jno.solve.staggered([[v, p], [T]])`` — sweeping a GROUP of fields together.

Alternate minimization solves one field at a time with the others frozen. That is only meaningful when
each field's own problem is well posed, and for a **velocity/pressure pair it is not**: the pressure
block is the constraint block, with no diagonal of its own (the block ``jno.precond.saddle`` locates
structurally), so "solve ``p`` with ``v`` frozen" is not a sub-problem at all. Any flow staggered
against a solid or a temperature therefore has to solve its Stokes pair *together*.

The oracle is a three-field problem — P2 velocity, P1 pressure, P1 temperature, coupled by buoyancy one
way and advection the other — solved three ways:

* monolithic Newton, the reference;
* ``staggered([[v, p], [T]], direct=True)``, which must land on the same root;
* ``staggered([v, p, T])`` flat, which must **fail** — that is what makes the grouped result mean
  something rather than being a slower path to an answer the flat form already reached.

The sweep machinery itself needed no change for this: ``staggered_newton`` consumes ``blocks`` as plain
index arrays, and its ``direct=True`` path zeroes the complement rather than slicing ``J[b][:, b]`` out,
so a group spanning two non-adjacent DOF ranges works unmodified.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest
from shapely.geometry import box

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


RA = 30.0


def _stokes_thermal(size=0.4):
    """Steady Stokes + an advected temperature: three blocks, of which ``(v, p)`` is a saddle pair.

    Buoyancy drives the flow from the temperature; advection feeds the velocity back into the energy
    equation, which is what makes the system genuinely coupled (and nonlinear, so the Newton/staggered
    slots apply at all)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=size)
    d.point_region("ppin", (0.0, 0.0))
    v, psi = d.fem_symbols(value_shape=(2,), names=("v", "psi"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    T, s = d.fem_symbols(names=("T", "s"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xpn, ypn, _ = d.variable("ppin", split=True)
    vb, wb = v.bind(x=xi, y=yi), psi.bind(x=xi, y=yi)
    pb, qb = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    Tb, sb = T.bind(x=xi, y=yi), s.bind(x=xi, y=yi)

    mom = (
        vb.x[0] * wb.x[0]
        + vb.y[0] * wb.y[0]
        + vb.x[1] * wb.x[1]
        + vb.y[1] * wb.y[1]
        - pb * (wb.x[0] + wb.y[1])
        - RA * Tb * wb[1]
    )
    cont = qb * (vb.x[0] + vb.y[1])
    ener = Tb.x * sb.x + Tb.y * sb.y + (vb[0] * Tb.x + vb[1] * Tb.y) * sb  # advection: the nonlinearity
    fem = jno.fem([mom, cont, ener, v(xb, yb) - 0.0, p(xpn, ypn) - 0.0, T(xb, yb) - (yb < 1e-9) * 1.0])
    return fem, v, p, T


def _lu():
    return jno.solve.lu(backend="host")


# --------------------------------------------------------------------------------------------------
# The oracle
# --------------------------------------------------------------------------------------------------
def test_a_grouped_sweep_finds_the_monolithic_root():
    """Correctness: grouping changes the ROUTE to the root, not the root."""
    fem, v, p, T = _stokes_thermal()
    mono = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-11, atol=1e-13), linear=_lu()))
    grouped = np.asarray(
        fem.solve(
            nonlinear=jno.solve.staggered([[v, p], [T]], direct=True, rtol=1e-11, atol=1e-13),
            linear=_lu(),
        )
    )
    assert np.abs(mono).max() > 1e-3, "trivial solution -- the comparison would be vacuous"
    for blk in fem.blocks:  # every block genuinely participates
        assert np.abs(mono[blk]).max() > 1e-6
    rel = np.abs(grouped - mono).max() / np.abs(mono).max()
    assert rel < 1e-7, f"grouped staggered and monolithic disagree: rel {rel:.3e}"


def test_the_flat_sweep_cannot_solve_what_the_grouped_one_can():
    """The discriminator. Swept apart, the pressure block has no diagonal and its sub-solve is not a
    well-posed problem -- so the flat form must fail rather than quietly return something."""
    fem, v, p, T = _stokes_thermal()
    with pytest.raises((RuntimeError, ValueError)) as e:
        np.asarray(
            fem.solve(
                nonlinear=jno.solve.staggered([v, p, T], direct=True, max_sweeps=40, rtol=1e-11, atol=1e-13),
                linear=_lu(),
            )
        )
    msg = str(e.value).lower()
    assert "converge" in msg or "singular" in msg, f"failed, but not for the expected reason: {e.value}"


# --------------------------------------------------------------------------------------------------
# Groups must not disturb the flat behaviour they generalise
# --------------------------------------------------------------------------------------------------
def test_a_group_of_one_is_the_flat_form(bitwise_backend):
    """``[[v, p], [T]]`` and ``[v, p, T]`` differ; ``[[a], [b]]`` and ``[a, b]`` must not."""
    fem, v, p, T = _stokes_thermal()
    a = fem.solve(nonlinear=jno.solve.staggered([[v, p], [T]], direct=True, rtol=1e-11, atol=1e-13), linear=_lu())
    b = fem.solve(nonlinear=jno.solve.staggered([[v, p], T], direct=True, rtol=1e-11, atol=1e-13), linear=_lu())
    assert np.array_equal(np.asarray(a), np.asarray(b)), "a bare symbol must be read as a group of one"


# --------------------------------------------------------------------------------------------------
# Fail loud
# --------------------------------------------------------------------------------------------------
def test_a_field_in_two_groups_fails_loud():
    fem, v, p, T = _stokes_thermal()
    with pytest.raises(ValueError, match="listed twice"):
        fem.solve(nonlinear=jno.solve.staggered([[v, p], [p, T]]))


def test_an_unlisted_field_fails_loud():
    fem, v, p, T = _stokes_thermal()
    with pytest.raises(ValueError, match="every field block must be swept"):
        fem.solve(nonlinear=jno.solve.staggered([[v, p]]))


def test_an_empty_group_fails_loud():
    fem, v, p, T = _stokes_thermal()
    with pytest.raises(ValueError, match="is empty"):
        fem.solve(nonlinear=jno.solve.staggered([[v, p], [], [T]]))
