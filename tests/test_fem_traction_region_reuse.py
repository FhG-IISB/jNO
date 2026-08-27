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

WHERE THE FAULT IS NOT
======================
Recorded so the next person does not repeat it. Each line is an experiment, not a reading of the
source.

* Not the trace. The node holds the right constant BEFORE and AFTER assembly -- build both terms,
  print ``node.right``, assemble, print it again: ``inner([0. 10. 0.], TestFunction(phi))`` throughout
  while the assembled resultant reads 1.0. The correct value reaches ``jno.fem`` and the wrong one
  comes out.
* Not reuse of a built artifact. ``fem``, ``fem.operator``, the ``b`` array and the domain are all
  DISTINCT objects between the two builds, and the values are equal anyway -- the second build
  recomputes from scratch and independently arrives at the first build's number.
* Not any cache. Clearing all seven module-level caches in the package (``_FACET_CACHE``,
  ``_ELEM_MAP_CACHE``, ``_PLAN_CACHE``, ``_LEAF_DIGEST_CACHE``, ``_FACTOR_CACHE``, ``_CUDSS_CACHE``,
  ``_PARDISO_CACHE``) plus ``jax.clear_caches()`` between builds changes nothing.
* Not ``functools.lru_cache``. There are two in the package: ``_accepts_key`` (signature
  introspection) and one in ``fem_1d``; neither can carry a traction on a 3-D box.
* Not id-reuse of the constant array, which ``_bake_fingerprint`` would be vulnerable to since it
  keys closure leaves by ``id()``. Pinning both arrays alive for the whole run, with distinct ids,
  changes nothing.
* Not the region Variable -- re-sampled on every build ("Sampled 12 points for 'pull'" twice), and
  the objects are distinct.
* Not shared class-level domain state. ``_tag_facet_stores`` compares ``is``-identical across
  domains but is a tuple of STRING NAMES (interned), carrying no data; the stores it names
  (``_tag_edges``/``_tag_triangles``/``_tag_quads``) are per-domain and not shared.
* Not normalisation-to-unit: build 7.0 then 3.0 and both assemble 7.0, so it is the FIRST value that
  wins, not a fixed one.

What remains: something in the assembly path that resolves the boundary term's coefficient from
state keyed by (mesh content, region name) rather than from the node it was given.
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
            "KNOWN BUG, not yet root-caused. See the module docstring for the full search record. "
            "Trigger is the CONJUNCTION (same mesh) AND (same region name). The trace is exonerated: "
            "the node still holds the correct constant after assembly, while the assembled vector "
            "carries the first build's. It is NOT artifact reuse -- fem, operator, b array and domain "
            "are all distinct objects, so the second build recomputes and independently arrives at "
            "the first value. strict=True so this FAILS THE SUITE once fixed."
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
