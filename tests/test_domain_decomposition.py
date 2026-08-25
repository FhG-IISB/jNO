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


@pytest.fixture(autouse=True)
def _x64():
    """These tests run in float64. The session default is x64-off (see tests/conftest.py), and this
    flag is process-wide -- save/restore keeps it from leaking to whatever module runs next."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


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
    sA, sB = jno.Shape.rect(0.0, 0.0, 0.6, 1.0), jno.Shape.rect(0.4, 0.0, 1.0, 1.0)
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

    sol, info = couple([(a, sA), (b, sB)]).solve(tol=1e-7, max_iter=60, return_info=True)
    assert info["overlap_jump"] < 1e-6, f"driver did not converge: {info}"
    equiv = float(np.linalg.norm(np.asarray(sol) - mono) / np.linalg.norm(mono))
    assert equiv < 1e-5, f"coupled driver must reproduce the monolithic solve, got {equiv:.2e}"
    assert float(np.linalg.norm(np.asarray(sol) - exact) / np.linalg.norm(exact)) < 3e-2


@pytest.mark.slow
def test_couple_via_jno_core():
    """The public entry: `jno.core([A, B]).solve()` where A, B are `jno.fdm([...])` subdomain problems
    whose PDE coordinates live on named regions (`domain.region(...)`). jno.core detects the subdomain
    solves, infers each region from the coords, and couples them by overlapping Schwarz — reproducing the
    monolithic single-mesh solve. No `jno.dd`, no `couple`, no explicit regions — just `jno.core`."""

    import jno.jnp_ops as jnn

    b1, b2 = box(0.0, 0.0, 0.6, 1.0), box(0.4, 0.0, 1.0, 1.0)
    d = jno.domain(b1.union(b2), mesh_size=0.06)
    d.region("A", b1)
    d.region("B", b2)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    xa, ya, _ = d.variable("A", split=True)
    xb2, yb2, _ = d.variable("B", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("interior", split=True)
    u = d.unknown()
    aa, ab, ui = u.bind(x=xa, y=ya), u.bind(x=xb2, y=yb2), u.bind(x=xi, y=yi)
    fa = 2 * np.pi**2 * jnn.sin(np.pi * xa) * jnn.sin(np.pi * ya)
    fb = 2 * np.pi**2 * jnn.sin(np.pi * xb2) * jnn.sin(np.pi * yb2)
    fi = 2 * np.pi**2 * jnn.sin(np.pi * xi) * jnn.sin(np.pi * yi)

    a = jno.fdm([-aa.d2(xa) - aa.d2(ya) - fa, u(xb, yb) - 0.0])  # PDE on region A
    b = jno.fdm([-ab.d2(xb2) - ab.d2(yb2) - fb, u(xb, yb) - 0.0])  # PDE on region B
    assert a.region == "A" and b.region == "B"  # regions inferred from the PDE coords
    mono = np.asarray(jno.fdm([-ui.d2(xi) - ui.d2(yi) - fi, u(xb, yb) - 0.0]).solve()).reshape(-1)

    sol = np.asarray(jno.core([a, b]).solve())  # the public entry couples the subdomains
    equiv = float(np.linalg.norm(sol - mono) / np.linalg.norm(mono))
    assert equiv < 1e-5, f"jno.core coupling must reproduce the monolithic solve, got {equiv:.2e}"
    assert float(np.linalg.norm(sol - exact) / np.linalg.norm(exact)) < 3e-2


def test_interface_tags_autocreated():
    """`domain.region(A) + region(B) + build_mesh` auto-creates first-class `interface_<A>_<B>` tags —
    the line where the two regions meet — alongside `boundary`/`interior`/`initial`. They resolve to the
    shared line's mesh nodes, so the user writes interface conditions on them like any other constraint
    (`uA(interface_A_B) - uB(interface_A_B)`, `k*uA.dn(...) - ...`), and jno.core spots interface
    conditions by the tag. Order-insensitive: `interface_B_A` is an alias for the same nodes."""
    regL, regR = box(0.0, 0.0, 0.5, 1.0), box(0.5, 0.0, 1.0, 1.0)  # partition, meet at x=0.5
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0))
    d.region("L", regL)
    d.region("R", regR)
    d.build_mesh(mesh_size=0.1)

    assert "interface_L_R" in d.avaiable_mesh_tags and "interface_R_L" in d.avaiable_mesh_tags
    assert d._interface_pairs["interface_L_R"] == ("L", "R")
    assert d._interface_pairs["interface_R_L"] == ("R", "L")  # order-insensitive alias

    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    on_line = set(np.where(np.abs(p[:, 0] - 0.5) < 1e-6)[0].tolist())
    ids = {int(i) for i in d._boundary_registry["interface_L_R"]["point_indices"]}
    assert ids == on_line and len(ids) > 2, "interface tag must resolve to EVERY mesh node on the shared line"
    alias_ids = {int(i) for i in d._boundary_registry["interface_R_L"]["point_indices"]}
    assert alias_ids == on_line, "the reversed-order alias must resolve to the same nodes"


@pytest.mark.slow
def test_couple_fem_fdm_line_via_jno_core():
    """Heterogeneous FDM+FEM on a NON-overlapping single interface line, through the public entry.

    The FEM region and FDM region *partition* the mesh (`domain.region(...)`), meeting only at the line
    x=0.5 — no overlap. `jno.core([femL, fdmR]).solve()` infers the line and couples by Dirichlet-Neumann
    (the FDM side supplies interface values, the FEM side consumes the interface flux) with no hand-rolled
    loop, reproducing the MMS solution u* = sin(pi x) sin(pi y). This is the coupling written in jno
    syntax — the delta over the overlapping FDM+FDM case above is: FEM subdomain + a single line + flux."""
    import jno.jnp_ops as jnn
    from jno.dd import couple

    regL, regR = box(0.0, 0.0, 0.5, 1.0), box(0.5, 0.0, 1.0, 1.0)  # partition, meet at x=0.5
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0))
    d.region("L", regL)
    d.region("R", regR)
    d.build_mesh(mesh_size=0.05)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])

    xL, yL, _ = d.variable("L", split=True)
    xR, yR, _ = d.variable("R", split=True)
    xb, yb, _ = d.variable("boundary", split=True)

    def f(x, y):
        return 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)

    uf, vf = d.fem_symbols()
    uif, vif = uf.bind(x=xL, y=yL), vf.bind(x=xL, y=yL)
    femL = jno.fem([uif.x * vif.x + uif.y * vif.y - f(xL, yL) * vif, uf(xb, yb) - 0.0])  # FEM on L (Neumann side)
    u = d.unknown()
    uiR = u.bind(x=xR, y=yR)
    fdmR = jno.fdm([-uiR.d2(xR) - uiR.d2(yR) - f(xR, yR), u(xb, yb) - 0.0])  # FDM on R (Dirichlet side)
    assert femL.region == "L" and fdmR.region == "R"  # regions inferred from the weak-form / PDE coords

    # the driver must pick the line (non-overlapping) mode and expose a real interface
    _, info = couple([(femL, regL), (fdmR, regR)]).solve(return_info=True)
    assert info["mode"] == "line-DN" and info["gamma_nodes"] > 0

    sol = np.asarray(jno.core([femL, fdmR]).solve()).reshape(-1)  # public entry, no hand-rolled coupling
    rel = float(np.linalg.norm(sol - exact) / np.linalg.norm(exact))
    assert rel < 5e-2, f"jno.core line coupling must match the MMS solution, got {rel:.2e}"


def test_normal_derivative_evaluates_as_flux_value():
    """`u.d(n)` (with `n = domain.variable(region, normals=True)`) now EVALUATES to the pointwise flux
    value `∇u·n` — the value form (distinct from the affine BC-assembly decomposition). This is what lets
    an interface residual be *evaluated* at interface nodes given a computed nodal field (the basis of a
    general interface solve). With u = x², at the vertical interface x=0.5 with outward normal (1,0),
    `∂u/∂n = 2x = 1.0`."""
    import equinox as eqx
    import jax.numpy as jnp

    from jno.dd import _element_partition
    from jno.trace_evaluator import TraceEvaluator

    d = jno.domain(box(0.0, 0.0, 1.0, 0.6))
    d.region("L", box(0.0, 0.0, 0.5, 0.6))
    d.region("R", box(0.5, 0.0, 1.0, 0.6))
    d.build_mesh(mesh_size=0.05)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    tris = np.asarray(d.mesh_connectivity["triangles"]).astype(int)
    _, gamma = _element_partition(p, tris, box(0.0, 0.0, 0.5, 0.6))

    xif, yif, _ = d.variable("interface_L_R", split=True)
    nrm = d.variable("interface_L_R", normals=True)
    u = d.unknown()
    expr = getattr(u.bind(x=xif, y=yif).d(nrm), "_expr")  # the raw Jacobian(u, [n]) trace node
    mod = eqx.tree_at(lambda m: m.value, u.model.module, jnp.asarray(p[:, 0] ** 2))  # inject u = x^2
    ctxt = {  # eval context: interface points + interface normals
        "interface_L_R": jnp.asarray(p[gamma]),
        "n_interface_L_R": jnp.tile(jnp.array([1.0, 0.0]), (len(gamma), 1)),
    }
    out = np.asarray(TraceEvaluator(params={u.model.layer_id: mod}).evaluate(expr, context=ctxt, var_bindings={})).reshape(
        -1
    )
    assert np.max(np.abs(out - 1.0)) < 0.02, f"u.d(n) must evaluate to the flux 2x·nx = 1.0 at x=0.5, got {out.mean():.3f}"


@pytest.mark.slow
def test_material_interface_via_overlap_and_kx():
    """A **material interface** (different conductivity each side) done the robust way: don't couple *on*
    the jump. Instead let the FEM subdomain carry the discontinuity via a spatially-varying ``k(x)`` (FEM
    handles a jump exactly when it lands on a mesh line), and overlap the two subdomains in the *uniform*
    region so plain **overlapping-Schwarz value exchange** couples them — no one-sided flux recovery.

    kL=1, kR=3 bar with the outer boundary held at the analytic profile; the interface value at x=0.5 is
    the material kink ``kL/(kL+kR) = 0.25`` (a uniform-k / naive coupling would give the symmetric 0.5)."""
    import jno.jnp_ops as jnn
    from jno.dd import couple

    kL, kR = 1.0, 3.0
    a, b = 2 * kR / (kL + kR), 2 * kL / (kL + kR)  # analytic slopes: u = 1-a·x (left), u = b·(1-x) (right)
    boxA, boxB = box(0.0, 0.0, 0.6, 1.0), box(0.5, 0.0, 1.0, 1.0)  # FEM covers the jump; overlap [0.5,0.6] uniform kR
    d = jno.domain(boxA.union(boxB), mesh_size=0.05)
    sA, sB = jno.Shape.rect(0.0, 0.0, 0.6, 1.0), jno.Shape.rect(0.5, 0.0, 1.0, 1.0)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    on = np.abs(p[:, 0] - 0.5) < 1e-6

    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)

    def g(x, y):  # outer boundary = analytic material profile (the kink is interior)
        return jnn.where(x <= 0.5, 1 - a * x, b * (1 - x))

    kx = jnn.where(xi < 0.5, kL, kR)  # the material jump lives in the COEFFICIENT field
    uf, vf = d.fem_symbols()
    uif, vif = uf.bind(x=xi, y=yi), vf.bind(x=xi, y=yi)
    femA = jno.fem([kx * (uif.x * vif.x + uif.y * vif.y), uf(xb, yb) - g(xb, yb)])  # ∫ k(x) ∇u·∇v
    u = d.unknown()
    uiB = u.bind(x=xi, y=yi)
    fdmB = jno.fdm([-kR * (uiB.d2(xi) + uiB.d2(yi)), u(xb, yb) - g(xb, yb)])  # uniform kR (smooth region)

    sol = np.asarray(couple([(femA, sA), (fdmB, sB)]).solve(max_iter=200)).reshape(-1)
    iface_val = float(np.mean(sol[on]))
    assert abs(iface_val - kL / (kL + kR)) < 0.03, f"material kink should be {kL / (kL + kR):.2f}, got {iface_val:.3f}"


@pytest.mark.slow
def test_overlap_coupled_solve_differentiable_in_fem_coefficient():
    """The differentiable-DD payoff, and the discriminator for a *correct* implicit-diff (not merely a
    forward that runs): with a **trainable conductivity ``kL``** in the FEM subdomain's weak form,
    ``couple([...]).solve()`` returns a differentiable trace node (like ``fem.solve()``), and
    ``∂(coupled field)/∂kL`` — which flows through the *converged* Schwarz fixed point via
    ``jax.lax.custom_root`` (no unrolled sweeps) — matches finite differences. A parameter in the FEM
    **coefficient** (re-assembled into ``A(θ)`` each solve) is the case a source-only gradient would
    silently miss, so it is the one asserted here. This is what makes an inverse domain-decomposition
    problem (recover a coefficient *through* the coupling) trainable via ``crux``.

    Homogeneous Dirichlet + a constant source, so the interface field scales with ``1/k`` (genuinely
    ``kL``-dependent); FEM (region A, ``k(x)`` carries ``kL`` on the left) overlaps FDM (region B)."""
    import jax

    import jno.jnp_ops as jnn
    from jno.dd import couple
    from jno.trace import FunctionCall

    kR, fsrc = 3.0, 10.0
    boxA, boxB = box(0.0, 0.0, 0.6, 1.0), box(0.5, 0.0, 1.0, 1.0)  # overlap x∈[0.5,0.6]
    d = jno.domain(boxA.union(boxB), mesh_size=0.08)
    sA, sB = jno.Shape.rect(0.0, 0.0, 0.6, 1.0), jno.Shape.rect(0.5, 0.0, 1.0, 1.0)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    in_a, in_b = _region_mask(p, boxA), _region_mask(p, boxB)
    overlap = jnp.asarray(in_a & in_b)
    ov_count = float((in_a & in_b).sum())

    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)

    kL = jno.np.parameter((1,), name="kL")  # trainable coefficient (the inverse parameter)
    kx = jnn.where(xi < 0.5, kL, kR)  # jump carried by k(x); kL lives in the bilinear form → A(θ)
    uf, vf = d.fem_symbols()
    uif, vif = uf.bind(x=xi, y=yi), vf.bind(x=xi, y=yi)
    femA = jno.fem([kx * (uif.x * vif.x + uif.y * vif.y) - fsrc * vif, uf(xb, yb) - 0.0])
    u = d.unknown()
    uiB = u.bind(x=xi, y=yi)
    fdmB = jno.fdm([-kR * (uiB.d2(xi) + uiB.d2(yi)) - fsrc, u(xb, yb) - 0.0])

    node = couple([(femA, sA), (fdmB, sB)]).solve(tol=1e-9, max_iter=300)
    assert isinstance(node, FunctionCall), "a trainable coefficient must make the coupled solve a differentiable node"
    assert getattr(node, "_domain", None) is d, "the coupled node must carry its domain for jno.core"

    def functional(kL_val):  # mean coupled field over the overlap band — a smooth functional of the field
        U = node.fn(kL_val)  # the differentiable coupled solve at this coefficient value
        return jnp.sum(jnp.where(overlap, U, 0.0)) / ov_count

    g = float(jax.grad(lambda t: functional(t))(jnp.array([1.5]))[0])
    eps = 1e-4
    fd = (float(functional(jnp.array([1.5 + eps]))) - float(functional(jnp.array([1.5 - eps])))) / (2 * eps)
    assert abs(g) > 1e-3, f"the FEM coefficient must actually influence the coupled field (got |g|={abs(g):.2e})"
    assert abs(g - fd) / (abs(fd) + 1e-12) < 1e-2, f"custom_root gradient must match FD: autodiff={g:.6e}, fd={fd:.6e}"


@pytest.mark.slow
def test_line_dn_coupled_solve_differentiable_in_fem_coefficient():
    """Differentiable **line** Dirichlet-Neumann coupling: with a trainable conductivity ``kc`` in the
    FEM (Neumann-side) weak form, ``couple([femL, fdmR]).solve()`` on a non-overlapping partition returns
    a differentiable node, and ``∂(coupled field)/∂kc`` — which flows through the converged DN fixed point
    via ``jax.lax.custom_root`` (no unrolled sweeps) — matches finite differences. The coefficient is
    re-assembled into the Neumann matrix ``A(θ)`` each solve; the FDM side supplies the interface flux.
    This makes an inverse problem trainable on the *sharp-interface* (non-overlapping) coupling too."""
    import jax

    from jno.dd import couple
    from jno.trace import FunctionCall

    regL, regR = box(0.0, 0.0, 0.5, 1.0), box(0.5, 0.0, 1.0, 1.0)  # partition, meet at x=0.5 (no overlap → line)
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0))
    d.region("L", regL)
    d.region("R", regR)
    d.build_mesh(mesh_size=0.09)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    on = jnp.asarray(np.abs(p[:, 0] - 0.5) < 1e-6)  # interface line nodes (matching mesh → x=0.5 exact)
    n_on = float(np.asarray(on).sum())
    assert n_on > 0

    xL, yL, _ = d.variable("L", split=True)
    xR, yR, _ = d.variable("R", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    fsrc = 20.0

    kc = jno.np.parameter((1,), name="kc")  # trainable coefficient in the FEM (Neumann) subdomain
    uf, vf = d.fem_symbols()
    uif, vif = uf.bind(x=xL, y=yL), vf.bind(x=xL, y=yL)
    femL = jno.fem([kc * (uif.x * vif.x + uif.y * vif.y) - fsrc * vif, uf(xb, yb) - 0.0])
    u = d.unknown()
    uiR = u.bind(x=xR, y=yR)
    fdmR = jno.fdm([-uiR.d2(xR) - uiR.d2(yR) - fsrc, u(xb, yb) - 0.0])

    node = couple([(femL, regL), (fdmR, regR)]).solve(tol=1e-9, max_iter=600)
    assert isinstance(node, FunctionCall), "a trainable coefficient must make the line coupled solve a node"
    assert getattr(node, "_domain", None) is d

    def functional(kc_val):  # mean coupled field on the interface line — a smooth functional of kc
        U = node.fn(kc_val)
        return jnp.sum(jnp.where(on, U, 0.0)) / n_on

    g = float(jax.grad(lambda t: functional(t))(jnp.array([1.2]))[0])
    eps = 1e-4
    fd = (float(functional(jnp.array([1.2 + eps]))) - float(functional(jnp.array([1.2 - eps])))) / (2 * eps)
    assert abs(g) > 1e-3, f"the FEM coefficient must influence the coupled field (got |g|={abs(g):.2e})"
    assert abs(g - fd) / (abs(fd) + 1e-12) < 1e-2, f"custom_root DN gradient must match FD: autodiff={g:.6e}, fd={fd:.6e}"


def test_line_dn_fem_fem_reaction_flux_forward_and_gradient():
    """FEM(Neumann) + FEM(Dirichlet) line coupling — exercises the **reaction-flux** Dirichlet branch
    (``(A u - b)|Γ``, distinct from the FDM ``∇u·n`` branch every other line test uses), for both the
    plain forward and the differentiable node. FEM+FEM Dirichlet-Neumann is well-conditioned (converges in
    a handful of sweeps), so this is the fast line test that also guards the gradient path.

    (a) forward at k=1 reproduces the MMS ``sin(πx)sin(πy)``; (b) a trainable coefficient makes the coupled
    solve a differentiable node whose ``∂/∂k`` (through the DN fixed point, ``custom_root``) matches FD."""
    import jax

    import jno.jnp_ops as jnn
    from jno.dd import couple
    from jno.trace import FunctionCall

    regL, regR = box(0.0, 0.0, 0.5, 1.0), box(0.5, 0.0, 1.0, 1.0)  # partition, meet at x=0.5 (line)
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0))
    d.region("L", regL)
    d.region("R", regR)
    d.build_mesh(mesh_size=0.08)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    on = jnp.asarray(np.abs(p[:, 0] - 0.5) < 1e-6)
    n_on = float(np.asarray(on).sum())

    xL, yL, _ = d.variable("L", split=True)
    xR, yR, _ = d.variable("R", split=True)
    xb, yb, _ = d.variable("boundary", split=True)

    def f(x, y):
        return 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)

    uf, vf = d.fem_symbols()
    uifR, vifR = uf.bind(x=xR, y=yR), vf.bind(x=xR, y=yR)
    femR = jno.fem([uifR.x * vifR.x + uifR.y * vifR.y - f(xR, yR) * vifR, uf(xb, yb) - 0.0])  # Dirichlet side

    # (a) forward: fixed k=1 (eager path, reaction-flux Dirichlet) reproduces the MMS field
    uifL, vifL = uf.bind(x=xL, y=yL), vf.bind(x=xL, y=yL)
    femL1 = jno.fem([uifL.x * vifL.x + uifL.y * vifL.y - f(xL, yL) * vifL, uf(xb, yb) - 0.0])  # Neumann side
    sol, info = couple([(femL1, regL), (femR, regR)]).solve(return_info=True)
    assert info["mode"] == "line-DN"
    rel = float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact))
    assert rel < 5e-2, f"FEM+FEM line (reaction flux) must match the MMS field, got rel-L2={rel:.2e}"

    # (b) gradient: a trainable coefficient in the Neumann FEM → differentiable node, FD-matched
    kc = jno.np.parameter((1,), name="kc")
    femLk = jno.fem([kc * (uifL.x * vifL.x + uifL.y * vifL.y) - f(xL, yL) * vifL, uf(xb, yb) - 0.0])
    node = couple([(femLk, regL), (femR, regR)]).solve(tol=1e-9, max_iter=200)
    assert isinstance(node, FunctionCall)

    def functional(kc_val):
        return jnp.sum(jnp.where(on, node.fn(kc_val), 0.0)) / n_on

    g = float(jax.grad(lambda t: functional(t))(jnp.array([1.1]))[0])
    eps = 1e-4
    fd = (float(functional(jnp.array([1.1 + eps]))) - float(functional(jnp.array([1.1 - eps])))) / (2 * eps)
    assert abs(g) > 1e-3, f"the FEM coefficient must influence the coupled field (got |g|={abs(g):.2e})"
    assert abs(g - fd) / (abs(fd) + 1e-12) < 1e-2, f"FEM+FEM DN gradient must match FD: autodiff={g:.6e}, fd={fd:.6e}"


@pytest.mark.slow
def test_overlap_through_jno_core_rebuilds_region_local_fem():
    """Overlap coupling of a REGION-TAGGED FEM straight through ``jno.core([...])``. A region-tagged FEM
    (needed so ``jno.core`` can detect it as a subdomain) assembles region-local (``RegionMask``), which
    can't reconcile an overlap band — its artificial boundary reaches no neighbour cells. The driver
    rebuilds it WHOLE-MESH (keeping the region label) so overlapping Schwarz closes. Verified on the
    material interface (``k(x)`` carries the jump): the interface value is the kink ``kL/(kL+kR) = 0.25``."""
    import jno.jnp_ops as jnn

    kL, kR = 1.0, 3.0
    a, b = 2 * kR / (kL + kR), 2 * kL / (kL + kR)
    boxA, boxB = box(0.0, 0.0, 0.6, 1.0), box(0.5, 0.0, 1.0, 1.0)
    d = jno.domain(boxA.union(boxB), mesh_size=0.05)
    d.region("A", boxA)
    d.region("B", boxB)  # node-subset labels over the single union mesh (overlap kept)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    on = np.abs(p[:, 0] - 0.5) < 1e-6

    xA, yA, _ = d.variable("A", split=True)
    xB, yB, _ = d.variable("B", split=True)
    xb, yb, _ = d.variable("boundary", split=True)

    def g(x, y):
        return jnn.where(x <= 0.5, 1 - a * x, b * (1 - x))

    kx = jnn.where(xA < 0.5, kL, kR)
    uf, vf = d.fem_symbols()
    uif, vif = uf.bind(x=xA, y=yA), vf.bind(x=xA, y=yA)  # ← region-tagged → RegionMask (region-local + detectable)
    femA = jno.fem([kx * (uif.x * vif.x + uif.y * vif.y), uf(xb, yb) - g(xb, yb)])
    u = d.unknown()
    uiB = u.bind(x=xB, y=yB)
    fdmB = jno.fdm([-kR * (uiB.d2(xB) + uiB.d2(yB)), u(xb, yb) - g(xb, yb)])

    assert femA.region == "A" and fdmB.region == "B"  # detectable by jno.core
    wm = femA._as_whole_mesh()
    assert wm.region == "A"  # the whole-mesh rebuild keeps the region label
    assert (np.abs(np.asarray(wm.A)).sum(1) > 1e-12).sum() > (np.abs(np.asarray(femA.A)).sum(1) > 1e-12).sum()  # more rows

    sol = np.asarray(jno.core([femA, fdmB]).solve(epochs=250)).reshape(-1)  # detect → overlap → whole-mesh rebuild
    iface = float(np.mean(sol[on]))
    assert abs(iface - kL / (kL + kR)) < 0.04, (
        f"jno.core overlap (rebuilt) should give ~{kL / (kL + kR):.2f}, got {iface:.3f}"
    )


@pytest.mark.slow
def test_couple_with_declared_interface_conditions():
    """The coupling written FULLY in jNO syntax — subdomain solves PLUS the interface conditions in the
    same ``jno.core([...])`` list, using the auto-created ``interface_L_R`` tag and its normal:

        uA(iface) - uB(iface)              # value continuity  (like a periodic tie)
        uA.d(n) - uB.d(n)                  # flux continuity   (n = domain.variable(iface, normals=True))

    jno.core recognises them (a residual referencing an ``interface_*`` tag) and routes the coupling from
    them instead of only inferring it — reproducing the MMS solution ``sin(pi x) sin(pi y)``."""
    import jno.jnp_ops as jnn
    from jno.dd import couple

    regL, regR = box(0.0, 0.0, 0.5, 1.0), box(0.5, 0.0, 1.0, 1.0)
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0))
    d.region("L", regL)
    d.region("R", regR)
    d.build_mesh(mesh_size=0.05)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])

    xL, yL, _ = d.variable("L", split=True)
    xR, yR, _ = d.variable("R", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xif, yif, _ = d.variable("interface_L_R", split=True)
    nrm = d.variable("interface_L_R", normals=True)

    def f(x, y):
        return 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)

    uf, vf = d.fem_symbols()
    uif, vif = uf.bind(x=xL, y=yL), vf.bind(x=xL, y=yL)
    femL = jno.fem([uif.x * vif.x + uif.y * vif.y - f(xL, yL) * vif, uf(xb, yb) - 0.0])
    u = d.unknown()
    uiR = u.bind(x=xR, y=yR)
    fdmR = jno.fdm([-uiR.d2(xR) - uiR.d2(yR) - f(xR, yR), u(xb, yb) - 0.0])

    # interface conditions, authored in jNO syntax on the auto-created tag + its normal
    uL_if, uR_if = uf.bind(x=xif, y=yif), u.bind(x=xif, y=yif)
    value_cond = uL_if - uR_if
    flux_cond = uL_if.d(nrm) - uR_if.d(nrm)

    # (a) both conditions are recognised (they reference the interface_L_R tag) and CLASSIFIED: the flux
    # condition carries a normal derivative (u.d(n)) even after the view subtraction; the value one does not.
    _, info = couple([(femL, regL), (fdmR, regR)], interface_conditions=[value_cond, flux_cond]).solve(return_info=True)
    assert info["interfaces"] == {"count": 2, "flux": 1, "value": 1} and info["mode"] == "line-DN"

    # (b) public entry: subproblems AND interface conditions in one list; coupling matches the MMS solution
    sol = np.asarray(jno.core([femL, fdmR, value_cond, flux_cond]).solve()).reshape(-1)
    rel = float(np.linalg.norm(sol - exact) / np.linalg.norm(exact))
    assert rel < 5e-2, f"declared-interface coupling must match the MMS solution, got {rel:.2e}"
