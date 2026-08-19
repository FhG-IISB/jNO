"""Per-cell mesh geometry as trace nodes — ``cell_volume``, ``cell_angles``, ``measure``, ``pnorm``.

These exist so a mesh-quality or element-size constraint can be *written*, and they are only
useful if they are differentiable in the mesh coordinates: the whole point is to constrain a mesh
that an optimiser is moving. So each is checked twice — against a closed form on a mesh whose
answer is known, and against finite differences at a node where the derivative is genuinely
non-zero (a zero-vs-zero agreement would prove nothing).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _rect(size=0.4):
    return jno.Shape.rect(0, 0, 2, 1, size=size).domain()


class TestClosedForms:
    def test_cell_volumes_tile_the_domain_exactly(self):
        d = _rect()
        vol = np.asarray(d.cell_volume().eval()).reshape(-1)
        assert vol.shape == (d._cells_p1().shape[0],), "one entry per cell"
        assert vol.min() > 0.0, "a valid mesh has strictly positive areas"
        assert vol.sum() == pytest.approx(2.0, abs=1e-10), "the 2x1 rectangle has area 2"

    def test_measure_is_the_total(self):
        d = _rect()
        assert float(np.asarray(d.measure().eval())) == pytest.approx(2.0, abs=1e-10)

    def test_a_triangles_angles_sum_to_pi(self):
        d = _rect()
        ang = np.asarray(d.cell_angles().eval()).reshape(-1, 3)
        assert ang.shape[1] == 3
        assert np.allclose(ang.sum(axis=1), PI, atol=1e-9)
        assert ang.min() > 0.0 and ang.max() < PI

    def test_an_equilateral_triangle_is_sixty_degrees(self):
        """The sharpest available check on the angle formula: one cell, exact answer."""
        d = jno.Path(0, 0).line_to(1, 0).line_to(0.5, np.sqrt(3) / 2).line_to(0, 0).face().sized(9.0).domain()
        vol = np.asarray(d.cell_volume().eval()).reshape(-1)
        if vol.size != 1:  # gmsh is free to subdivide; the check only means anything on one cell
            pytest.skip(f"mesher produced {vol.size} cells, not 1")
        ang = np.asarray(d.cell_angles().eval()).reshape(-1, 3)
        assert np.allclose(np.degrees(ang[0]), 60.0, atol=1e-6)
        assert vol[0] == pytest.approx(np.sqrt(3) / 4, abs=1e-10)

    def test_cell_angles_gives_dihedrals_on_a_tetrahedral_mesh(self):
        """One name, both dimensions: 3 interior angles per triangle, 6 dihedrals per tet.

        The dispatch is on the dimension rather than on an argument, so a mesh-quality constraint is
        the same expression in 2-D and 3-D — which is the whole reason this is not a second method.
        """
        d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.6).domain()
        ang = np.asarray(d.cell_angles().eval())
        assert ang.shape == (d._cells_p1().shape[0], 6)
        assert ang.min() > 0.0 and ang.max() < PI


class TestPnorm:
    def test_pnorm_bounds_the_max_and_tightens_with_p(self):
        d = _rect()
        x = d.cell_volume()
        raw = np.asarray(x.eval()).reshape(-1)
        p50 = float(np.asarray(x.pnorm(50).eval()))
        p200 = float(np.asarray(x.pnorm(200).eval()))
        assert p50 >= raw.max() - 1e-12, "the p-norm is an upper bound on the max"
        assert p200 < p50, "a larger p is a tighter approximation"
        assert p200 >= raw.max() - 1e-12


class TestDifferentiableInTheMesh:
    """``∂g/∂X`` must flow, or a geometric constraint cannot steer a moving mesh."""

    @staticmethod
    def _moving_rect(size=0.5):
        d = jno.Shape.rect(0, 0, 2, 1, size=size).domain()
        xm, ym, _ = d.variable("design", where=lambda *c: np.ones_like(np.asarray(c[0]), dtype=bool), split=True)
        xm.trainable(name="mx")
        ym.trainable(name="my")
        return d

    def test_area_gradient_matches_finite_differences(self):
        d = self._moving_rect()
        _args, rebuild = d._moving_points()
        cells = jnp.asarray(d._cells_p1(), dtype=jnp.int32)
        pts = np.asarray(d.mesh.points)
        X0, Y0 = jnp.asarray(pts[:, 0]), jnp.asarray(pts[:, 1])

        def area(X):
            v = rebuild(X, Y0)[cells]
            jac = jnp.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]], axis=-1)
            return jnp.sum(jnp.abs(jnp.linalg.det(jac)) / 2.0)

        g = np.asarray(jax.grad(area)(X0))
        assert np.all(np.isfinite(g))
        # Pick a node the derivative actually depends on — an interior node moved in x changes the
        # split between its neighbouring triangles. A boundary-parallel node can legitimately give
        # zero, and checking one of those against a zero FD would assert nothing.
        i = int(np.argmax(np.abs(g)))
        assert abs(g[i]) > 1e-9, "no node has a non-zero area derivative; the test would be vacuous"
        h = 1e-6
        fd = float((area(X0.at[i].add(h)) - area(X0.at[i].add(-h))) / (2 * h))
        assert g[i] == pytest.approx(fd, abs=1e-7)

    def test_angle_gradient_matches_finite_differences(self):
        d = self._moving_rect()
        _args, rebuild = d._moving_points()
        cells = jnp.asarray(d._cells_p1(), dtype=jnp.int32)
        pts = np.asarray(d.mesh.points)
        X0, Y0 = jnp.asarray(pts[:, 0]), jnp.asarray(pts[:, 1])

        def worst_angle(X):
            v = rebuild(X, Y0)[cells]
            out = []
            for k in range(3):
                a = v[:, (k + 1) % 3] - v[:, k]
                b = v[:, (k + 2) % 3] - v[:, k]
                cos = jnp.sum(a * b, axis=-1) / (jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + 1e-12)
                out.append(jnp.arccos(jnp.clip(cos, -1.0, 1.0)))
            # a smooth proxy for "the smallest angle", which is what the constraint bounds
            return jnp.sum(jnp.stack(out, axis=-1) ** -8)

        g = np.asarray(jax.grad(worst_angle)(X0))
        assert np.all(np.isfinite(g)), "the angle formula must not produce NaN gradients"
        i = int(np.argmax(np.abs(g)))
        assert abs(g[i]) > 1e-9
        h = 1e-7
        fd = float((worst_angle(X0.at[i].add(h)) - worst_angle(X0.at[i].add(-h))) / (2 * h))
        assert g[i] == pytest.approx(fd, rel=1e-5)

    def test_the_node_tracks_moved_coordinates(self):
        """Not just differentiable — the VALUE must follow the parameters, not the initial mesh."""
        d = self._moving_rect()
        _args, rebuild = d._moving_points()
        cells = jnp.asarray(d._cells_p1(), dtype=jnp.int32)
        pts = np.asarray(d.mesh.points)
        X0, Y0 = jnp.asarray(pts[:, 0]), jnp.asarray(pts[:, 1])

        def area(X, Y):
            v = rebuild(X, Y)[cells]
            jac = jnp.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]], axis=-1)
            return float(jnp.sum(jnp.abs(jnp.linalg.det(jac)) / 2.0))

        assert area(X0, Y0) == pytest.approx(2.0, abs=1e-10)
        # Stretching every x by 1.5 must scale the total area by exactly 1.5.
        assert area(X0 * 1.5, Y0) == pytest.approx(3.0, abs=1e-10)


class TestConstraintComposition:
    """The operators are only useful if the paper's constraint forms can be written with them."""

    def test_the_element_angle_constraint_is_expressible_and_feasible(self):
        d = _rect(size=0.4)
        theta_min = np.radians(20.0)
        g = ((2 * PI - d.cell_angles()) / (2 * PI - theta_min)).pnorm(50)
        val = float(np.asarray(g.eval()))
        worst = float(np.asarray(d.cell_angles().eval()).min())
        # A well-shaped mesh satisfies g <= 1; the p-norm is slightly conservative, so allow the
        # small overshoot the paper also reports (its converged run sits at 19.943 deg vs a 20 bound).
        assert val < 1.02, f"g1 = {val} on a mesh whose worst angle is {np.degrees(worst):.2f} deg"
        assert np.degrees(worst) > 15.0

    def test_the_element_volume_constraint_is_expressible(self):
        d = _rect(size=0.4)
        v_max = float(np.asarray(d.cell_volume().eval()).max()) * 1.5
        g = (d.cell_volume() / v_max).pnorm(50)
        assert float(np.asarray(g.eval())) < 1.0, "every element is under the cap"

    def test_a_volume_fraction_constraint_composes_with_jno_le(self):
        d = _rect(size=0.4)
        node = jno.le(d.measure() / 4.0, 1.0)
        assert node.sense == "le"
        assert float(np.asarray(node.residual.eval())) == pytest.approx(2.0 / 4.0 - 1.0, abs=1e-9)


class TestNormalisedPnorm:
    """``pnorm(normalize=True)`` — the value is the true max, the gradient stays the p-norm's.

    Without it the aggregation overshoots with the entry count: ``N`` entries all equal to ``r``
    give ``N^(1/p) r``, which is enough to report a satisfied constraint as violated from the
    first iteration. Le et al., *Struct. Multidisc. Optim.* **41**(4), 2010, 605-620; used by
    Jung, Yun & Kim, *Computers & Structures* **331** (2026) 108403, eq. (29)-(30).
    """

    @staticmethod
    def _agg(r, p, normalize):
        v = jnp.abs(r)
        agg = jnp.sum(v**p) ** (1.0 / p)
        return jax.lax.stop_gradient(jnp.max(v) / (agg + 1e-30)) * agg if normalize else agg

    def test_the_plain_pnorm_overshoots_with_the_entry_count(self):
        """Quantifies the defect the flag exists for, on the paper's own numbers."""
        p, n = 50.0, 840  # 280 triangles x 3 angles
        r = jnp.full(n, 0.941)  # the angle ratio at a 40 deg minimum against a 20 deg bound
        plain = float(self._agg(r, p, False))
        assert plain > 1.0, "a satisfied constraint reported as violated is the whole problem"
        assert float(self._agg(r, p, True)) == pytest.approx(0.941, rel=1e-9)

    def test_the_normalised_value_is_the_maximum(self):
        p = 50.0
        r = jnp.asarray(np.random.default_rng(0).uniform(0.1, 0.9, 200))
        assert float(self._agg(r, p, True)) == pytest.approx(float(jnp.max(jnp.abs(r))), rel=1e-12)

    def test_the_gradient_keeps_the_pnorm_direction(self):
        """A true max has a one-hot gradient; the point of the p-norm is that this one does not."""
        p = 50.0
        r = jnp.asarray(np.random.default_rng(1).uniform(0.3, 0.9, 40))
        g_norm = jax.grad(lambda z: self._agg(z, p, True))(r)
        g_plain = jax.grad(lambda z: self._agg(z, p, False))(r)
        scale = float(jnp.max(jnp.abs(r)) / (jnp.sum(jnp.abs(r) ** p) ** (1.0 / p)))
        np.testing.assert_allclose(np.asarray(g_norm), scale * np.asarray(g_plain), rtol=1e-10)
        assert int(np.sum(np.abs(np.asarray(g_norm)) > 1e-12)) > 1, "must not collapse to one-hot"

    def test_it_reaches_the_trace_through_a_node(self):
        d = jno.Shape.rect(0.0, 0.0, 2.0, 1.0, size=0.4).domain()
        got = float(np.asarray(d.cell_volume().pnorm(50.0, normalize=True).eval()).reshape(-1)[0])
        assert got == pytest.approx(float(np.asarray(d.cell_volume().eval()).max()), rel=1e-9)


class TestLogBarrier:
    """``log_barrier(b)`` — keeps a quantity below ``b``, and keeps a GRADIENT above it."""

    @staticmethod
    def _f(x, b=350.0, tau=1e-3):
        t = tau * abs(b)
        sw = b - t
        gap = jnp.maximum(b - x, t)
        dx = x - sw
        return jnp.squeeze(jnp.where(x <= sw, -b * jnp.log(gap), -b * jnp.log(t) + (b / t) * dx + 0.5 * (b / t**2) * dx**2))

    def test_it_is_the_log_below_the_switch(self):
        b = 350.0
        for x in (0.0, 100.0, 349.0):
            assert float(self._f(x, b)) == pytest.approx(float(-b * np.log(b - x)), rel=1e-12)

    def test_it_is_continuous_and_smooth_at_the_switch(self):
        b, tau = 350.0, 1e-3
        sw = b - tau * b
        # Compare the two BRANCHES at the switch, not the function either side of it: the slope
        # there is b/t = 1000, so f(sw +- eps) differ by 2 eps b/t for any eps, which says nothing.
        assert float(self._f(sw, b)) == pytest.approx(float(-b * np.log(tau * b)), rel=1e-9)
        # The one-sided slopes both equal b/t.
        g = jax.grad(lambda z: self._f(z, b))
        for x in (sw - 1e-6, sw + 1e-6):
            assert float(g(x)) == pytest.approx(b / (tau * b), rel=1e-5)

    def test_the_gradient_survives_above_the_bound(self):
        """The whole reason this exists.

        ``log(maximum(b - x, eps))`` is finite above the bound but CONSTANT, so its gradient is
        exactly zero and the constraint silently stops constraining — measured on the cantilever
        as a perimeter of 992 against a target of 350, with nothing pushing back.
        """
        b = 350.0
        g = jax.grad(lambda z: self._f(z, b))
        for x in (350.0, 500.0, 992.0):
            grad = float(g(x))
            assert np.isfinite(grad) and grad > 1e3, f"no restoring gradient at x = {x}: {grad}"
        # and it strengthens the further out it goes, so an overshoot is pushed back harder
        assert float(g(992.0)) > float(g(500.0)) > float(g(350.0))

        naive = jax.grad(lambda z: -b * jnp.log(jnp.maximum(b - z, 1e-8)))
        assert float(naive(500.0)) == 0.0, "the naive clamp is exactly the failure this avoids"

    def test_everything_stays_finite(self):
        b = 350.0
        for x in (-1e4, 0.0, 349.9999, 350.0, 1e4):
            assert np.isfinite(float(self._f(x, b))) and np.isfinite(float(jax.grad(lambda z: self._f(z, b))(x)))

    def test_it_reaches_the_trace_through_a_node(self):
        d = jno.Shape.rect(0.0, 0.0, 2.0, 1.0, size=0.4).domain()
        total = d.cell_volume().sum  # the domain area, 2.0
        got = float(np.asarray(total.log_barrier(10.0).eval()).reshape(-1)[0])
        assert got == pytest.approx(float(-10.0 * np.log(10.0 - 2.0)), rel=1e-6)

    def test_a_non_positive_tau_is_refused(self):
        d = jno.Shape.rect(0.0, 0.0, 2.0, 1.0, size=0.4).domain()
        with pytest.raises(ValueError, match="tau must be positive"):
            d.cell_volume().sum.log_barrier(10.0, tau=0.0)


class TestDihedralAngles:
    """``cell_angles`` — the 3-D mesh-quality quantity, and the reason it is not ``cell_angles``.

    Jung, Yun & Kim, *Computers & Structures* **331** (2026) 108403, Sec. 2.3.3. Their eq. (21)/(24)
    bounds a triangle's minimum interior angle; on a tetrahedron the same job needs the dihedral,
    and needs BOTH bounds stated, for the reasons pinned below.
    """

    @staticmethod
    def _one_tet(tmp_path, verts):
        """A domain holding exactly one tetrahedron with the given vertices.

        Written as ``.inp`` rather than ``.vtk``: VTK stores points big-endian, and meshio hands
        them back as ``>f8``, which JAX refuses outright ("Error interpreting argument ... as a JAX
        value"). Abaqus ``.inp`` is ASCII, so it round-trips native dtype -- and is what
        ``test_domain_load_mesh.py`` already uses.
        """
        import meshio

        path = str(tmp_path / "tet.inp")
        meshio.write(path, meshio.Mesh(np.asarray(verts, dtype=float), [("tetra", np.array([[0, 1, 2, 3]]))]))
        return jno.domain(constructor=path, compute_mesh_connectivity=False)

    def test_a_regular_tetrahedron_is_arccos_one_third(self, tmp_path):
        """The exact oracle: every dihedral of a regular tet is ``arccos(1/3)`` = 70.5288 degrees."""
        r = self._one_tet(tmp_path, [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)])
        ang = np.asarray(r.cell_angles().eval())
        assert ang.shape == (1, 6), "a tet has six edges, so six dihedrals"
        np.testing.assert_allclose(ang[0], np.arccos(1.0 / 3.0), atol=1e-12)

    def test_the_six_dihedrals_do_not_sum_to_a_constant(self, tmp_path):
        """**The reason both bounds must be stated.** A triangle's three angles sum to pi, so in 2-D
        bounding the minimum bounds the maximum. Six dihedrals sum to a *variable* quantity strictly
        inside ``(2 pi, 3 pi)`` — 2.35 pi for the regular tet — so that argument does not carry over
        and a minimum-angle bound alone permits a cap.
        """
        d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
        s = np.asarray(d.cell_angles().eval()).sum(axis=1) / PI
        assert s.min() > 2.0 and s.max() < 3.0, f"sums must lie in (2pi, 3pi), got [{s.min()}, {s.max()}]"
        assert s.max() - s.min() > 0.02, (
            f"the sum must genuinely VARY or the 2-D argument would carry over after all; "
            f"spread is only {s.max() - s.min():.4f} pi"
        )
        reg = np.degrees(np.arccos(1.0 / 3.0)) * 6 / 180.0
        assert s.min() < reg < s.max(), "the regular tet's 2.35 pi should sit inside the observed range"

    def test_a_sliver_is_caught_by_the_dihedral_and_missed_by_the_aspect_free_face_angles(self, tmp_path):
        """A sliver — four nearly coplanar vertices — is the degeneracy that destroys interpolation
        accuracy (Babuska & Aziz 1976), and it is invisible to face angles: they stay near 45-90
        degrees while the dihedrals collapse towards 0 and pi.
        """
        eps = 1e-3
        sliver = self._one_tet(tmp_path, [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, eps)])
        ang = np.degrees(np.asarray(sliver.cell_angles().eval())[0])
        assert ang.min() < 1.0, f"a sliver's smallest dihedral must collapse, got {ang.min():.3f} deg"
        assert ang.max() > 179.0, f"and its largest must approach 180, got {ang.max():.3f} deg"

        # The face angles of the SAME cell stay moderate -- which is what makes the dihedral the
        # guard that catches this and the reason cell_angles' 2-D quantity is not enough.
        v = np.asarray(sliver.mesh.points)[np.asarray(sliver._cells_p1())[0]]
        face_angles = []
        for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            for k in range(3):
                a = v[f[(k + 1) % 3]] - v[f[k]]
                b = v[f[(k + 2) % 3]] - v[f[k]]
                face_angles.append(np.degrees(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))))
        assert min(face_angles) > 20.0, (
            f"this sliver's face angles should stay moderate (min {min(face_angles):.1f} deg); if they "
            "do not, the configuration is not isolating what the test claims"
        )

    def test_a_needle_keeps_moderate_dihedrals(self, tmp_path):
        """The converse, and why the dihedral is not the ONLY guard: a needle — a small base with a
        distant apex — is badly conditioned but satisfies the maximum-angle condition, so its
        dihedrals stay away from 0 and pi. ``cell_aspect`` is what sees it.
        """
        needle = self._one_tet(tmp_path, [(0, 0, 0), (0.02, 0, 0), (0.01, 0.017, 0), (0.01, 0.006, 3.0)])
        ang = np.degrees(np.asarray(needle.cell_angles().eval())[0])
        assert ang.min() > 5.0 and ang.max() < 175.0, f"a needle's dihedrals stay moderate, got {ang}"
        assert float(np.asarray(needle.cell_aspect().eval())[0]) > 20.0, (
            "but cell_aspect must flag it, or neither measure covers the needle"
        )

    def test_the_gradient_matches_finite_differences(self):
        """Differentiable in the vertex positions, or it cannot constrain a mesh an optimiser moves."""
        d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
        xm, ym, zm, _ = d.variable("mv3", where=lambda *c: np.ones_like(np.asarray(c[0]), dtype=bool), split=True)
        xm.trainable(name="mx3"), ym.trainable(name="my3"), zm.trainable(name="mz3")
        args, rebuild = d._moving_points()
        cells = jnp.asarray(d._cells_p1(), dtype=jnp.int32)
        pts = np.asarray(d.mesh.points)[:, :3]
        X0, Y0, Z0 = (jnp.asarray(pts[:, i]) for i in range(3))

        def worst(X):
            v = rebuild(X, Y0, Z0)[cells]
            n = []
            for k in range(4):
                keep = [i for i in range(4) if i != k]
                nk = jnp.cross(v[:, keep[1]] - v[:, keep[0]], v[:, keep[2]] - v[:, keep[0]])
                nk = nk / (jnp.linalg.norm(nk, axis=-1, keepdims=True) + 1e-12)
                s = jnp.sign(jnp.sum(nk * (v[:, k] - v[:, keep[0]]), axis=-1, keepdims=True))
                n.append(nk * jnp.where(s == 0.0, 1.0, s))
            out = []
            for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
                a, b = [k for k in range(4) if k not in (i, j)]
                out.append(np.pi - jnp.arccos(jnp.clip(jnp.sum(n[a] * n[b], axis=-1), -1.0, 1.0)))
            return jnp.sum(jnp.stack(out, axis=-1) ** -4)  # a smooth proxy for "the smallest"

        g = np.asarray(jax.grad(worst)(X0))
        assert np.all(np.isfinite(g)), "the dihedral formula must not produce NaN gradients"
        i = int(np.argmax(np.abs(g)))
        assert abs(g[i]) > 1e-9, "no node has a non-zero derivative; the test would be vacuous"
        h = 1e-7
        fd = float((worst(X0.at[i].add(h)) - worst(X0.at[i].add(-h))) / (2 * h))
        assert g[i] == pytest.approx(fd, rel=1e-5)

    def test_a_one_dimensional_domain_is_refused(self):
        """An interval has no angle to measure; 2-D and 3-D are the whole supported range."""
        d = jno.domain(constructor=jno.domain.line(mesh_size=0.2))
        assert d.dimension == 1
        with pytest.raises(NotImplementedError, match="simplices in 2-D or 3-D"):
            d.cell_angles()

    def test_the_constraint_form_is_expressible_and_feasible(self):
        """The paper's eq. (24) shape, written on dihedrals: a sound mesh must satisfy g <= 1."""
        d = jno.Shape.box(0, 0, 0, 2, 1, 1, size=0.4).domain()
        theta_min = np.radians(10.0)
        g = ((PI - d.cell_angles()) / (PI - theta_min)).pnorm(50, normalize=True)
        val = float(np.asarray(g.eval()))
        worst = float(np.asarray(d.cell_angles().eval()).min())
        assert val < 1.02, f"g = {val} on a mesh whose smallest dihedral is {np.degrees(worst):.2f} deg"
        assert np.degrees(worst) > 5.0
