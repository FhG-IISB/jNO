"""A traction term must assemble the traction it was given, on the facets it named.

Both assertions here are ANALYTIC, not a restatement of jNO's output: a constant traction ``t``
applied to one face of the unit cube has resultant ``t * area``, and that face has area 1. So a pull
of magnitude ``m`` must produce a resultant of magnitude ``m``, on any mesh, at any refinement.

Found while reviewing a topology-optimisation script that builds one ``jno.fem`` per body in a
single process (a reanalysis: the same problem re-solved on the extracted, then refined, geometry).

**The failure is reuse, not the first build.** Build a traction of 1.0 under some region name, then
build 10.0 under the SAME name: the second silently assembles the first's constant. Building 10.0
first and then 1.0 makes both assemble the 10.0 answer, exactly 10x the other order -- so assembly
itself is correct and it is the CONSTANT that is stale. Naming the two regions apart gives 1.0 and
10.0 exactly, and the first build under a reused name is correct too; only the second is wrong.

The other three tests here are the controls that establish that, and they should pass before and
after any fix. Keep them: without them a fix that broke ordinary traction assembly would look like
progress.

Why this matters more than a wrong number: nothing raises. A load sweep on a fixed mesh -- vary the
traction, re-solve, plot compliance against load -- returns a smooth, plausible, entirely wrong
curve. House rule 1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import jno

inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _assemble(magnitude, *, region, size=0.5):
    """|resultant| of the load vector for a traction of `magnitude` on the y = 1 face."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, phi = d.fem_symbols(value_shape=(3,))
    xi, yi, zi = d.variable("interior", split=True)[:3]
    eps = lambda w: symgrad(w, [xi, yi, zi])  # noqa: E731
    a = lambda p, q: 0.577 * trace(p) * trace(q) + 0.385 * inner(p, q, n_contract=2)  # noqa: E731
    held = d.variable(f"held_{region}", where=lambda x, y, z: y < 1e-9, split=True)[:3]
    pull = d.variable(region, where=lambda x, y, z: y > 1.0 - 1e-9, split=True)[:3]
    fem = jno.fem(
        [
            a(eps(u), eps(phi)),
            u(*held) - (0.0, 0.0, 0.0),
            -1.0 * inner(jnp.asarray([0.0, magnitude, 0.0]), phi.bind(**dict(zip("xyz", pull))), 1),
        ],
        quad_degree=2,
    )
    op = fem.operator
    _A, b = op if isinstance(op, tuple) else op.evaluate({})
    return float(jnp.linalg.norm(jnp.asarray(b).reshape(-1, 3).sum(axis=0)))


class TestTheTractionIsTheOneThatWasGiven:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "KNOWN BUG, not yet root-caused: a second jno.fem() built on the same mesh with the same "
            "region name silently reuses the first build's traction constant. Excluded so far -- "
            "_PLAN_CACHE/_ELEM_MAP_CACHE/_LEAF_DIGEST_CACHE (clearing them between builds changes "
            "nothing), the region Variable (re-sampled, distinct objects), id-reuse of the constant "
            "array (pinning both alive changes nothing), and domain/mesh caching (distinct objects). "
            "The trigger is (same mesh) AND (same region name): either alone assembles correctly. "
            "strict=True so this FAILS THE SUITE once fixed, forcing the xfail to be removed."
        ),
    )
    def test_a_reused_region_name_does_not_freeze_the_traction(self):
        """Build 1.0 then 10.0 under ONE region name; the second must not report the first."""
        first = _assemble(1.0, region="pull")
        second = _assemble(10.0, region="pull")
        assert second == pytest.approx(10.0 * first, rel=1e-9), (
            f"traction 1.0 assembled {first}, traction 10.0 assembled {second}; the second build "
            f"reused the first build's constant (expected {10.0 * first})"
        )

    def test_distinct_region_names_are_correct(self):
        """The control: the same two builds, named apart, hit the analytic answer exactly."""
        assert _assemble(1.0, region="pull_a") == pytest.approx(1.0, rel=1e-9)
        assert _assemble(10.0, region="pull_b") == pytest.approx(10.0, rel=1e-9)


class TestTheRightFacetsAreIntegrated:
    def test_a_reused_region_name_integrates_the_right_facets(self):
        """Analytic: constant traction m on a unit face has resultant m, first build included."""
        got = _assemble(1.0, region="pull")
        assert got == pytest.approx(1.0, rel=1e-9), (
            f"a unit traction on the unit y=1 face must give resultant 1.0, got {got}"
        )

    def test_it_is_mesh_independent(self):
        """The resultant is a property of the load, not of the discretisation."""
        coarse = _assemble(1.0, region="pull_coarse", size=0.5)
        fine = _assemble(1.0, region="pull_fine", size=0.25)
        assert coarse == pytest.approx(fine, rel=1e-9), (
            f"resultant moved with the mesh: {coarse} at h=0.5 against {fine} at h=0.25"
        )
