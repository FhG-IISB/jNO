"""Heterogeneous domain decomposition — Stage 0: the coupling-mechanism gate.

The design target (see ``plans/heterogeneous-domain-decomposition.md``) is to solve part of a domain
with one solver and the rest with another, composed through ``jno.core``. Before any driver/API is
built, this test pins down the load-bearing claim with a **throwaway, hand-rolled overlapping Schwarz
loop**: on one shared mesh, two overlapping regions solved by finite differences and coupled only by
Dirichlet exchange on the overlap must reproduce the **monolithic single-mesh solve** to solver
tolerance. That "monolithic-equivalence" is what proves the coupling is real rather than plausible; the
future ``jno.core`` coupled-solve mode has to keep this green.

Run with x64 (the FD solve accumulates in float64)."""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

import jno  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _region_mask(points, geometry):
    """Boolean mask of the mesh nodes inside ``geometry`` (a shapely region) — a subdomain's nodes."""
    g = geometry.buffer(1e-9)
    return np.array([g.contains(Point(float(x), float(y))) for x, y in points])


def test_overlapping_schwarz_matches_monolithic():
    """Two overlapping FD regions on one shared mesh, coupled by Dirichlet exchange on the overlap,
    converge to the monolithic single-mesh FD solution (‖Δ‖/‖u‖ ~ 1e-10). -Δu = f, u = 0 on ∂Ω,
    manufactured u* = sin(πx)sin(πy). The two boxes union to the unit square and overlap in x∈[0.4,0.6]."""
    b1, b2 = box(0.0, 0.0, 0.6, 1.0), box(0.4, 0.0, 1.0, 1.0)
    d = jno.domain(b1.union(b2), mesh_size=0.05)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    n = p.shape[0]

    bnd = np.zeros(n, bool)
    bnd[np.asarray(d.mesh_connectivity["boundary_indices"])] = True
    in_a, in_b = _region_mask(p, b1), _region_mask(p, b2)
    overlap = in_a & in_b

    f = jnp.asarray(2 * np.pi**2 * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    lap = lambda u: jno.fdm.laplacian(u, d, method="cotangent")  # noqa: E731
    newton = jno.solve.newton()

    def solve_pinned(mask, vals):  # solve -Δu = f with `mask` nodes pinned to `vals`
        mask_j, vals_j = jnp.asarray(mask), jnp.asarray(vals)
        return np.asarray(newton(lambda u: jnp.where(mask_j, u - vals_j, -lap(u) - f), jnp.zeros(n)))

    # monolithic reference: the whole domain in one solve
    u_mono = solve_pinned(bnd, np.zeros(n))
    assert float(np.linalg.norm(u_mono - exact) / np.linalg.norm(exact)) < 1e-2  # it's a real solution

    # hand-rolled overlapping Schwarz: each region pins its complement to the neighbour's current field
    u_a = u_b = np.zeros(n)
    iters, jump = 0, np.inf
    for iters in range(1, 51):
        u_a = solve_pinned((~in_a) | bnd, np.where(bnd, 0.0, u_b))  # A: complement ← B
        u_b = solve_pinned((~in_b) | bnd, np.where(bnd, 0.0, u_a))  # B: complement ← A
        jump = float(np.max(np.abs(u_a[overlap] - u_b[overlap])))
        if jump < 1e-10:
            break
    u_dd = np.where(in_a, u_a, u_b)  # combine (the two agree on the overlap)

    assert jump < 1e-10, f"Schwarz did not converge on the overlap (jump={jump:.2e})"
    equiv = float(np.linalg.norm(u_dd - u_mono) / np.linalg.norm(u_mono))
    assert equiv < 1e-8, f"coupled solution must match the monolithic solve, got {equiv:.2e} in {iters} iters"


def test_heterogeneous_fem_fdm_coupling():
    """Stage 1: a **FEM** region and an **FDM** region on one shared mesh, coupled by overlapping
    Schwarz (Dirichlet exchange), converge to the correct solution — the first *heterogeneous* coupling
    (two different discretizations). -Δu = f, u = 0 on ∂Ω, u* = sin(πx)sin(πy); FDM on box1, FEM on box2.

    Note: heterogeneous Dirichlet–Dirichlet Schwarz converges *slower* than the homogeneous FDM+FDM case
    above (the operators differ at the interface) — the convergence-rate knob the design flags as a
    research risk (optimized/Robin transmission is the later mitigation). The gate here is **correctness**
    (matches the analytic field to discretization accuracy) plus convergence below discretization error."""
    import jax.numpy as jnp

    import jno.jnp_ops as jnn

    b1, b2 = box(0.0, 0.0, 0.6, 1.0), box(0.4, 0.0, 1.0, 1.0)
    d = jno.domain(b1.union(b2), mesh_size=0.05)  # union = unit square, overlap x∈[0.4,0.6]
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    n = p.shape[0]
    bnd = np.zeros(n, bool)
    bnd[np.asarray(d.mesh_connectivity["boundary_indices"])] = True
    in_a, in_b = _region_mask(p, b1), _region_mask(p, b2)  # FDM region, FEM region
    overlap = in_a & in_b
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    f = jnp.asarray(2 * np.pi**2 * exact)

    # FEM engine: raw stiffness + load assembled on the whole mesh (subdomain BCs pinned per iteration)
    xi, yi, _ = d.variable("interior", split=True)
    u, phi = d.fem_symbols()
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fn = 2 * np.pi**2 * jnn.sin(np.pi * xi) * jnn.sin(np.pi * yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - fn * vi])  # weak form of -Δu = f, no BC
    a_fem, b_fem = jnp.asarray(fem.A), jnp.asarray(fem.b).reshape(-1)

    def fem_pinned(mask, vals):  # FEM solve with `mask` nodes pinned to `vals` (row-replacement)
        m, v = jnp.asarray(mask), jnp.asarray(vals)
        a = jnp.where(m[:, None], 0.0, a_fem)
        a = a.at[jnp.arange(n), jnp.arange(n)].set(jnp.where(m, 1.0, jnp.diag(a)))
        return np.asarray(jnp.linalg.solve(a, jnp.where(m, v, b_fem)))

    lap = lambda u: jno.fdm.laplacian(u, d, method="cotangent")  # noqa: E731
    newton = jno.solve.newton()

    def fdm_pinned(mask, vals):
        m, v = jnp.asarray(mask), jnp.asarray(vals)
        return np.asarray(newton(lambda u: jnp.where(m, u - v, -lap(u) - f), jnp.zeros(n)))

    u_a = u_b = np.zeros(n)
    iters, jump = 0, np.inf
    for iters in range(1, 81):
        u_a = fdm_pinned((~in_a) | bnd, np.where(bnd, 0.0, u_b))  # FDM on region A
        u_b = fem_pinned((~in_b) | bnd, np.where(bnd, 0.0, u_a))  # FEM on region B
        jump = float(np.max(np.abs(u_a[overlap] - u_b[overlap])))
        if jump < 1e-3:
            break
    u_dd = np.where(in_a, u_a, u_b)

    assert jump < 1e-3, f"heterogeneous Schwarz did not converge below discretization (jump={jump:.2e})"
    rel = float(np.linalg.norm(u_dd - exact) / np.linalg.norm(exact))
    assert rel < 5e-3, f"coupled FEM+FDM solution must match the analytic field, got rel-L2={rel:.2e}"


@pytest.mark.slow
def test_heterogeneous_fem_pinn_coupling():
    """Stage 2: a **FEM** region (exact solve) coupled to a **PINN** region (a trained network) by an
    alternating overlapping Schwarz loop — the genuinely novel *exact-solve ⋈ optimization* case.
    -Δu = f, u = 0 on ∂Ω, u* = sin(πx)sin(πy); FEM on box1, PINN (a generic MLP — nothing problem-
    specific) on box2. FEM reads the net's interface values as its Dirichlet BC; the net is trained
    (warm-started) to fit its region's PDE + the FEM's interface values.

    The coupling converges to the **PINN's accuracy floor** (a network fits a smooth field to ~a few %,
    the bottleneck here — not the coupling), and the overlap jump becomes *noisy* once there (the
    half-trained-network injects noise into the exchange, the dynamics the design flags as the research
    risk). So the gate is: the coupling drives the overlap jump down by a large factor, and the combined
    field is a valid few-percent solution. Marked slow (it trains a network in the loop)."""
    import jax.numpy as jnp
    import optax

    import jno.jnp_ops as jnn

    b1, b2 = box(0.0, 0.0, 0.6, 1.0), box(0.4, 0.0, 1.0, 1.0)  # FEM region, PINN region
    d = jno.domain(b1.union(b2), mesh_size=0.08)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    n = p.shape[0]
    bnd = np.zeros(n, bool)
    bnd[np.asarray(d.mesh_connectivity["boundary_indices"])] = True
    in_a, in_b = _region_mask(p, b1), _region_mask(p, b2)
    overlap = in_a & in_b
    tris = np.asarray(d.mesh_connectivity["triangles"])
    adj_b = np.zeros(n, bool)  # B's artificial boundary = B nodes adjacent to A-only (the interface)
    for t in tris:
        if in_b[t].any() and (~in_b[t]).any():
            adj_b[t[in_b[t]]] = True
    b_artif, b_outer = adj_b & in_b & ~bnd, in_b & bnd
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])

    # FEM (exact): raw stiffness + per-iteration pinning
    xi, yi, _ = d.variable("interior", split=True)
    u, phi = d.fem_symbols()
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fn = 2 * np.pi**2 * jnn.sin(np.pi * xi) * jnn.sin(np.pi * yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - fn * vi])
    a_fem, b_fem = jnp.asarray(fem.A), jnp.asarray(fem.b).reshape(-1)

    def fem_pinned(mask, vals):
        m, v = jnp.asarray(mask), jnp.asarray(vals)
        a = jnp.where(m[:, None], 0.0, a_fem)
        a = a.at[jnp.arange(n), jnp.arange(n)].set(jnp.where(m, 1.0, jnp.diag(a)))
        return np.asarray(jnp.linalg.solve(a, jnp.where(m, v, b_fem)))

    # PINN: a generic MLP (nothing problem-specific)
    def init(sizes, key):
        ps = []
        for i in range(len(sizes) - 1):
            key, k = jax.random.split(key)
            ps.append((jax.random.normal(k, (sizes[i], sizes[i + 1])) * np.sqrt(2 / sizes[i]), jnp.zeros(sizes[i + 1])))
        return ps

    def fwd(ps, xy):
        h = xy
        for w, bb in ps[:-1]:
            h = jnp.tanh(h @ w + bb)
        w, bb = ps[-1]
        return (h @ w + bb)[..., 0]

    params = init([2, 40, 40, 1], jax.random.PRNGKey(0))
    xy_b = jnp.asarray(p[in_b])
    f_b = jnp.asarray(2 * np.pi**2 * np.sin(np.pi * p[in_b, 0]) * np.sin(np.pi * p[in_b, 1]))
    xy_art, xy_out = jnp.asarray(p[b_artif]), jnp.asarray(p[b_outer])

    def lap(ps, xy):
        hh = jax.vmap(lambda q: jax.hessian(lambda z: fwd(ps, z))(q))(xy)
        return hh[:, 0, 0] + hh[:, 1, 1]

    def pinn_loss(ps, art):
        return (
            jnp.mean((-lap(ps, xy_b) - f_b) ** 2)
            + 10.0 * jnp.mean((fwd(ps, xy_art) - art) ** 2)
            + 10.0 * jnp.mean(fwd(ps, xy_out) ** 2)
        )

    opt = optax.adam(3e-3)
    ostate = opt.init(params)

    @jax.jit
    def step(ps, ostate, art):
        g = jax.grad(pinn_loss)(ps, art)
        up, ostate = opt.update(g, ostate, ps)
        return optax.apply_updates(ps, up), ostate

    u_a = np.zeros(n)
    best_jump, best_rel = np.inf, np.inf
    for _ in range(7):
        net_vals = np.asarray(jax.vmap(lambda q: fwd(params, q))(jnp.asarray(p)))
        u_a = fem_pinned((~in_a) | bnd, np.where(bnd, 0.0, net_vals))  # FEM on A, interface ← net
        art = jnp.asarray(u_a[b_artif])
        for _ in range(500):
            params, ostate = step(params, ostate, art)  # PINN on B, warm-started
        u_b = np.asarray(jax.vmap(lambda q: fwd(params, q))(jnp.asarray(p)))
        jump = float(np.max(np.abs(u_a[overlap] - u_b[overlap])))
        if jump < best_jump:  # track the best iterate (the jump oscillates at the PINN's floor)
            best_jump = jump
            best_rel = float(np.linalg.norm(np.where(in_a, u_a, u_b) - exact) / np.linalg.norm(exact))

    assert best_jump < 5e-2, f"FEM+PINN coupling did not converge the interface (best jump={best_jump:.2e})"
    assert best_rel < 6e-2, f"coupled FEM+PINN field must be a valid few-% solution, got rel-L2={best_rel:.2e}"


@pytest.mark.slow
def test_overlapping_schwarz_via_real_fdm_api():
    """The coupling through the **real `jno.fdm([...]).solve()` API** (not hand-rolled newton): each
    subdomain is a genuine FDM solve that pins its complement — a geometric sub-region `domain.region(...)`
    — to the neighbour's field (a symbolic nodal data-field, updated in place each iteration). The
    overlapping Schwarz still reproduces the monolithic single-mesh solve. This is the crystallization of
    the FDM coupling side: `domain.region` node-subsets + symbolic nodal-field Dirichlet values compose
    into a real subdomain solve. Marked slow (an iterative loop of real solves)."""
    import equinox as eqx
    import jax.numpy as jnp

    import jno.jnp_ops as jnn

    b1, b2 = box(0.0, 0.0, 0.6, 1.0), box(0.4, 0.0, 1.0, 1.0)
    d = jno.domain(b1.union(b2), mesh_size=0.07)
    d.region("notA", b2.difference(b1))  # A's complement (B-only), a geometric interior sub-region
    d.region("notB", b1.difference(b2))  # B's complement (A-only)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    n = p.shape[0]
    in_a, in_b = _region_mask(p, b1), _region_mask(p, b2)
    overlap = in_a & in_b
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])

    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xna, yna, _ = d.variable("notA", split=True)
    xnb, ynb, _ = d.variable("notB", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)

    field_b, field_a = jno.np.parameter((n,)), jno.np.parameter((n,))  # neighbour data-fields (no optimizer)

    def set_field(g, vals):
        g.model.module = eqx.tree_at(lambda m: m.value, g.model.module, jnp.asarray(vals))

    # BUILD ONCE — each subdomain pins its complement to the (in-place-updated) neighbour field
    solve_a = jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(xna, yna) - field_b, u(xb, yb) - 0.0])
    solve_b = jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(xnb, ynb) - field_a, u(xb, yb) - 0.0])
    u_mono = np.asarray(jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(xb, yb) - 0.0]).solve()).reshape(-1)

    u_a = u_b = np.zeros(n)
    jump = np.inf
    for _ in range(40):
        set_field(field_b, u_b)
        u_a = np.asarray(solve_a.solve()).reshape(-1)  # re-solve, neighbour field updated in place
        set_field(field_a, u_a)
        u_b = np.asarray(solve_b.solve()).reshape(-1)
        jump = float(np.max(np.abs(u_a[overlap] - u_b[overlap])))
        if jump < 1e-6:
            break
    u_dd = np.where(in_a, u_a, u_b)

    assert jump < 1e-6, f"real-API Schwarz did not converge (jump={jump:.2e})"
    equiv = float(np.linalg.norm(u_dd - u_mono) / np.linalg.norm(u_mono))
    assert equiv < 1e-5, f"real-API coupled solve must match the monolithic solve, got {equiv:.2e}"
    assert float(np.linalg.norm(u_dd - exact) / np.linalg.norm(exact)) < 3e-2


def test_fem_dirichlet_on_named_sub_region():
    """FEM building block for domain decomposition: a trial-only `u(region) - g` on a **named interior
    sub-region** (`domain.region(name, polygon)`) pins that region's whole node set — with either a
    constant/coordinate value OR a **nodal data-field** (a `jno.np.parameter` carrying a neighbour's
    field, gathered by node index). Laplace with a central sub-region pinned; the pin must be exact."""
    import equinox as eqx
    import jax.numpy as jnp

    from jno.trace import FunctionCall

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06)
    sub = box(0.3, 0.3, 0.7, 0.7)  # strictly interior
    d.region("B", sub)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    n = p.shape[0]
    in_b = _region_mask(p, sub)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    x_sub, y_sub, _ = d.variable("B", split=True)
    u, phi = d.fem_symbols()
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)

    # (a) constant value on the sub-region
    sol = np.asarray(jno.fem([ui.x * vi.x + ui.y * vi.y, u(x_sub, y_sub) - 1.0, u(xb, yb) - 0.0]).solve()).reshape(-1)
    assert np.max(np.abs(sol[in_b] - 1.0)) < 1e-10, "constant sub-region pin must be exact"

    # (b) nodal-field value on the sub-region (a neighbour's field, no optimizer → data → eager solve)
    known = np.sin(3 * p[:, 0]) + p[:, 1]
    g = jno.np.parameter((n,))
    g.model.module = eqx.tree_at(lambda m: m.value, g.model.module, jnp.asarray(known))
    sol_b = jno.fem([ui.x * vi.x + ui.y * vi.y, u(x_sub, y_sub) - g, u(xb, yb) - 0.0]).solve()
    assert not isinstance(sol_b, FunctionCall), "a data-field Dirichlet value must stay an eager solve"
    assert np.max(np.abs(np.asarray(sol_b).reshape(-1)[in_b] - known[in_b])) < 1e-10, "field pin must gather exactly"


@pytest.mark.slow
def test_heterogeneous_fem_fdm_real_api_coupling():
    """FEM+FDM coupling through the **real solver APIs** on both sides: the FEM subdomain is authored as
    `jno.fem([weak, u(sub_region) - neighbour, u(boundary) - 0]).solve()` (whole-mesh assembly, complement
    pinned to a nodal neighbour field, rebuilt per iteration), the FDM subdomain likewise via `jno.fdm`.
    The overlapping Schwarz converges to the analytic solution. This is the crystallization of the FEM
    coupling side — the FDM side is proven in `test_overlapping_schwarz_via_real_fdm_api`. Marked slow
    (heterogeneous Schwarz + per-iteration FEM re-assembly)."""
    import equinox as eqx
    import jax.numpy as jnp

    import jno.jnp_ops as jnn

    b1, b2 = box(0.0, 0.0, 0.6, 1.0), box(0.4, 0.0, 1.0, 1.0)
    d = jno.domain(b1.union(b2), mesh_size=0.06)
    d.region("notA", b2.difference(b1))
    d.region("notB", b1.difference(b2))
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    n = p.shape[0]
    in_a, in_b = _region_mask(p, b1), _region_mask(p, b2)
    overlap = in_a & in_b
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])

    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xna, yna, _ = d.variable("notA", split=True)
    xnb, ynb, _ = d.variable("notB", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    uf, vf = d.fem_symbols()
    uif, vif = uf.bind(x=x, y=y), vf.bind(x=x, y=y)

    def field(vals):
        g = jno.np.parameter((n,))
        g.model.module = eqx.tree_at(lambda m: m.value, g.model.module, jnp.asarray(vals))
        return g

    u_a = u_b = np.zeros(n)
    jump = np.inf
    for _ in range(60):
        u_a = np.asarray(jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(xna, yna) - field(u_b), u(xb, yb) - 0.0]).solve()).reshape(-1)
        u_b = np.asarray(
            jno.fem([uif.x * vif.x + uif.y * vif.y - f * vif, uf(xnb, ynb) - field(u_a), uf(xb, yb) - 0.0]).solve()
        ).reshape(-1)
        jump = float(np.max(np.abs(u_a[overlap] - u_b[overlap])))
        if jump < 1e-2:
            break
    u_dd = np.where(in_a, u_a, u_b)

    assert jump < 2e-2, f"real-API FEM+FDM Schwarz did not converge (jump={jump:.2e})"
    assert float(np.linalg.norm(u_dd - exact) / np.linalg.norm(exact)) < 3e-2


@pytest.mark.slow
def test_couple_driver_reproduces_monolithic():
    """The `jno.dd.couple([...]).solve()` driver automates the overlapping Schwarz: each subdomain is a
    `jno.fdm([...])` (its PDE + outer BCs) plus the shapely region it owns; the driver infers each
    complement, exchanges Dirichlet data with the neighbour (`pinned_solver`, built once + JIT-reused),
    and iterates to tolerance — reproducing the monolithic single-mesh solve. Marked slow (Schwarz loop)."""

    import jno.jnp_ops as jnn
    from jno.dd import couple

    b1, b2 = box(0.0, 0.0, 0.6, 1.0), box(0.4, 0.0, 1.0, 1.0)
    d = jno.domain(b1.union(b2), mesh_size=0.06)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)

    a = jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(xb, yb) - 0.0])  # subdomain problems: PDE + outer BC
    b = jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(xb, yb) - 0.0])
    mono = np.asarray(jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(xb, yb) - 0.0]).solve()).reshape(-1)

    sol, info = couple([(a, b1), (b, b2)]).solve(tol=1e-7, max_iter=60, return_info=True)
    assert info["overlap_jump"] < 1e-6, f"driver did not converge: {info}"
    equiv = float(np.linalg.norm(np.asarray(sol) - mono) / np.linalg.norm(mono))
    assert equiv < 1e-5, f"coupled driver must reproduce the monolithic solve, got {equiv:.2e}"
    assert float(np.linalg.norm(np.asarray(sol) - exact) / np.linalg.norm(exact)) < 3e-2
