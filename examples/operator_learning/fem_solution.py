import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# ============================================
# Verification:  Constant κ = 1 case
# For -Δu = 2π² sin(πx) sin(πy), u = sin(πx) sin(πy)
# ============================================


def create_fem_solver(points, triangles, boundary_mask):
    """
    Create a JIT-compiled FEM solver for -div(κ ∇u) = f.

    Weak form: ∫ κ ∇u · ∇v dx = ∫ f v dx

    P1 elements on triangles.
    """

    points = jnp.asarray(points)
    triangles = jnp.asarray(triangles, dtype=jnp.int32)
    boundary_mask = jnp.asarray(boundary_mask)

    n_nodes = len(points)
    n_elements = len(triangles)

    # Interior node indices
    interior_mask = ~boundary_mask
    n_interior = int(interior_mask.sum())
    interior_idx = jnp.where(interior_mask, size=n_interior)[0]

    # Precompute element geometry
    # coords[e, i, : ] = coordinates of node i of element e
    coords = points[triangles]  # (M, 3, 2)

    x1, y1 = coords[:, 0, 0], coords[:, 0, 1]
    x2, y2 = coords[:, 1, 0], coords[:, 1, 1]
    x3, y3 = coords[:, 2, 0], coords[:, 2, 1]

    # Signed area (need to handle orientation)
    # Area = 0.5 * |det([x2-x1, x3-x1; y2-y1, y3-y1])|
    signed_area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    areas = jnp.abs(signed_area)  # (M,)

    # For P1 elements, shape functions are:
    #   N1 = (a1 + b1*x + c1*y) / (2*area)
    #   N2 = (a2 + b2*x + c2*y) / (2*area)
    #   N3 = (a3 + b3*x + c3*y) / (2*area)
    #
    # Gradients (constant on each element):
    #   ∇N1 = (b1, c1) / (2*area)
    #   ∇N2 = (b2, c2) / (2*area)
    #   ∇N3 = (b3, c3) / (2*area)
    #
    # where:
    #   b1 = y2 - y3,  c1 = x3 - x2
    #   b2 = y3 - y1,  c2 = x1 - x3
    #   b3 = y1 - y2,  c3 = x2 - x1

    # Shape function gradient coefficients (before dividing by 2*area)
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2  # (M,)
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1  # (M,)

    # Stack into (M, 3) arrays
    b = jnp.stack([b1, b2, b3], axis=1)  # (M, 3) - ∂N/∂x coefficients
    c = jnp.stack([c1, c2, c3], axis=1)  # (M, 3) - ∂N/∂y coefficients

    # Element stiffness contribution:
    # K_e[i,j] = ∫_e κ ∇Ni · ∇Nj dx
    #          = κ_e * Area * (bi*bj + ci*cj) / (4 * Area²)
    #          = κ_e * (bi*bj + ci*cj) / (4 * Area)

    # Precompute (bi*bj + ci*cj) for all i,j pairs
    # Shape:  (M, 3, 3)
    grad_dot = b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]

    # Centroids for evaluating κ and f
    centroids_x = (x1 + x2 + x3) / 3  # (M,)
    centroids_y = (y1 + y2 + y3) / 3  # (M,)

    # Prepare assembly indices
    # For scatter-add: need (row, col) pairs for all 9 entries of each element matrix
    rows = triangles[:, :, None].repeat(3, axis=2)  # (M, 3, 3)
    cols = triangles[:, None, :].repeat(3, axis=1)  # (M, 3, 3)
    rows_flat = rows.flatten()  # (M*9,)
    cols_flat = cols.flatten()  # (M*9,)

    # For load vector
    load_rows = triangles.flatten()  # (M*3,)

    @jit
    def solve(theta):
        """
        Solve -div(κ(x,y;θ)   u) = f with u = 0 on boundary.

        Args:
            theta:  (4,) parameters for κ = exp(θ0 + θ1 sin(2πx) + θ2 cos(2πy) + θ3 sin(2π(x+y)))

        Returns:
            u: (N,) solution values at mesh nodes
        """

        # Evaluate κ at element centroids
        log_kappa = theta[0] + theta[1] * jnp.sin(2 * jnp.pi * centroids_x) + theta[2] * jnp.cos(2 * jnp.pi * centroids_y) + theta[3] * jnp.sin(2 * jnp.pi * (centroids_x + centroids_y))
        kappa = jnp.exp(log_kappa)  # (M,)

        # Evaluate f at element centroids:  f = 2π² sin(πx) sin(πy)
        f_centroid = 2 * jnp.pi**2 * jnp.sin(jnp.pi * centroids_x) * jnp.sin(jnp.pi * centroids_y)  # (M,)

        # ========================================
        # Assemble stiffness matrix
        # K_e[i,j] = κ_e * (b_i*b_j + c_i*c_j) / (4 * Area_e)
        # ========================================
        K_local = kappa[:, None, None] * grad_dot / (4 * areas[:, None, None])  # (M, 3, 3)
        K_vals = K_local.flatten()  # (M*9,)

        # Scatter into global matrix
        K_global = jnp.zeros((n_nodes, n_nodes))
        K_global = K_global.at[rows_flat, cols_flat].add(K_vals)

        # ========================================
        # Assemble load vector
        # F_e[i] = ∫_e f * Ni dx ≈ f_centroid * Area / 3 (for each node)
        # ========================================
        F_local = (f_centroid[:, None] * areas[:, None] / 3) * jnp.ones((n_elements, 3))  # (M, 3)
        F_vals = F_local.flatten()  # (M*3,)

        F_global = jnp.zeros(n_nodes)
        F_global = F_global.at[load_rows].add(F_vals)

        # ========================================
        # Apply Dirichlet BC: u = 0 on boundary
        # Extract interior system and solve
        # ========================================
        K_int = K_global[interior_idx][:, interior_idx]  # (n_interior, n_interior)
        F_int = F_global[interior_idx]  # (n_interior,)

        # Solve the linear system
        u_int = jnp.linalg.solve(K_int, F_int)

        # Assemble full solution
        u = jnp.zeros(n_nodes)
        u = u.at[interior_idx].set(u_int)

        return u

    return solve


def create_batched_solver(points, triangles, boundary_mask):
    """Create both single and batched solvers."""
    solve = create_fem_solver(points, triangles, boundary_mask)
    solve_batch = jit(vmap(solve))
    return solve, solve_batch


def fem_solver(domain):
    # Extract arrays
    points = np.array(domain.mesh.points[:, :2], dtype=np.float64)
    for cell_block in domain.mesh.cells:
        if cell_block.type == "triangle":
            triangles = np.array(cell_block.data, dtype=np.int32)
            break

    # Boundary mask
    eps = 1e-8
    boundary_mask = (points[:, 0] < eps) | (points[:, 0] > 1 - eps) | (points[:, 1] < eps) | (points[:, 1] > 1 - eps)

    print(f"Mesh:  {len(points)} nodes, {len(triangles)} triangles")
    print(f"Interior:  {(~boundary_mask).sum()}, Boundary: {boundary_mask.sum()}")

    # Create solver
    solve, solve_batch = create_batched_solver(points, triangles, boundary_mask)
    return solve_batch


def test_solver():
    """Test solver against known analytical solution."""

    # Create a simple mesh on [0,1]²
    import pygmsh

    with pygmsh.geo.Geometry() as geom:
        mesh_size = 0.02  # Fine mesh for accuracy

        points_geo = [
            geom.add_point([0, 0], mesh_size=mesh_size),
            geom.add_point([1, 0], mesh_size=mesh_size),
            geom.add_point([1, 1], mesh_size=mesh_size),
            geom.add_point([0, 1], mesh_size=mesh_size),
        ]

        lines = [
            geom.add_line(points_geo[0], points_geo[1]),
            geom.add_line(points_geo[1], points_geo[2]),
            geom.add_line(points_geo[2], points_geo[3]),
            geom.add_line(points_geo[3], points_geo[0]),
        ]

        curve_loop = geom.add_curve_loop(lines)
        surface = geom.add_plane_surface(curve_loop)

        mesh = geom.generate_mesh()

    # Extract arrays
    points = np.array(mesh.points[:, :2], dtype=np.float64)
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            triangles = np.array(cell_block.data, dtype=np.int32)
            break

    # Boundary mask
    eps = 1e-8
    boundary_mask = (points[:, 0] < eps) | (points[:, 0] > 1 - eps) | (points[:, 1] < eps) | (points[:, 1] > 1 - eps)

    print(f"Mesh:  {len(points)} nodes, {len(triangles)} triangles")
    print(f"Interior:  {(~boundary_mask).sum()}, Boundary: {boundary_mask.sum()}")

    # Create solver
    solve, solve_batch = create_batched_solver(points, triangles, boundary_mask)

    # ========================================
    # Test 1: Constant κ = 1
    # -Δu = 2π² sin(πx) sin(πy)
    # Exact:  u = sin(πx) sin(πy)
    # ========================================
    print("\n" + "=" * 60)
    print("Test 1: Constant κ = 1 (Poisson equation)")
    print("=" * 60)

    # θ = [0, 0, 0, 0] gives κ = exp(0) = 1
    theta_const = jnp.array([0.0, 0.0, 0.0, 0.0])
    u_fem = solve(theta_const)

    # Exact solution
    u_exact = np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])

    # Errors (only at interior nodes, boundary is enforced exactly)
    interior_mask_np = ~boundary_mask
    error = np.abs(np.array(u_fem) - u_exact)
    l2_error = np.sqrt(np.mean(error[interior_mask_np] ** 2))
    linf_error = np.max(error[interior_mask_np])
    rel_l2 = l2_error / np.sqrt(np.mean(u_exact[interior_mask_np] ** 2))

    print(f"u_fem  ∈ [{float(u_fem.min()):.6f}, {float(u_fem.max()):.6f}]")
    print(f"u_exact ∈ [{u_exact.min():.6f}, {u_exact.max():.6f}]")
    print(f"L2 error:      {l2_error:.6e}")
    print(f"L∞ error:      {linf_error:.6e}")
    print(f"Relative L2:   {rel_l2:.4%}")

    # ========================================
    # Test 2: Variable κ
    # ========================================
    print("\n" + "=" * 60)
    print("Test 2: Variable κ(x,y)")
    print("=" * 60)

    theta_var = jnp.array([0.1, 0.3, 0.2, 0.15])
    u_var = solve(theta_var)

    print(f"θ = {theta_var}")
    print(f"u ∈ [{float(u_var.min()):.6f}, {float(u_var.max()):.6f}]")

    # Check that solution is smooth (no NaNs, reasonable range)
    assert not np.any(np.isnan(u_var)), "Solution contains NaNs!"
    assert float(u_var.max()) < 10, "Solution seems too large"

    # ========================================
    # Test 3: Batched solve
    # ========================================
    print("\n" + "=" * 60)
    print("Test 3: Batched solve")
    print("=" * 60)

    thetas = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.3, 0.2, 0.15],
            [0.2, -0.2, 0.4, 0.1],
            [-0.1, 0.5, -0.3, 0.25],
        ]
    )

    u_batch = solve_batch(thetas)
    print(f"Batch shape: {u_batch.shape}")

    for i, theta in enumerate(thetas):
        print(f"θ[{i}]:  u ∈ [{float(u_batch[i].min()):.6f}, {float(u_batch[i].max()):.6f}]")

    # ========================================
    # Timing
    # ========================================
    print("\n" + "=" * 60)
    print("Timing")
    print("=" * 60)

    import time

    # Warm up
    _ = solve(theta_var)

    n_runs = 100
    start = time.time()
    for _ in range(n_runs):
        u = solve(theta_var)
    u.block_until_ready()
    elapsed = time.time() - start
    print(f"Single solve:  {elapsed/n_runs*1000:.3f} ms")

    # Warm up batch
    _ = solve_batch(thetas)

    start = time.time()
    for _ in range(n_runs):
        u = solve_batch(thetas)
    u.block_until_ready()
    elapsed = time.time() - start
    print(f"Batched (4 θ): {elapsed/n_runs*1000:.3f} ms")

    # ========================================
    # Plotting
    # ========================================
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

    # Plot 1: Comparison with exact for κ=1
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    tcf = ax.tricontourf(triang, u_fem, levels=50, cmap="viridis")
    plt.colorbar(tcf, ax=ax, label="u")
    ax.set_title("FEM Solution (κ=1)")
    ax.set_aspect("equal")

    ax = axes[1]
    tcf = ax.tricontourf(triang, u_exact, levels=50, cmap="viridis")
    plt.colorbar(tcf, ax=ax, label="u")
    ax.set_title("Exact:  sin(πx)sin(πy)")
    ax.set_aspect("equal")

    ax = axes[2]
    tcf = ax.tricontourf(triang, error, levels=50, cmap="Reds")
    plt.colorbar(tcf, ax=ax, label="|error|")
    ax.set_title(f"Error (L2={l2_error:.2e})")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("fem_verification.png", dpi=300)
    print("Saved fem_verification.png")

    # Plot 2: Variable κ solutions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, theta, u in zip(axes.flat, thetas, u_batch):
        tcf = ax.tricontourf(triang, u, levels=50, cmap="viridis")
        ax.tricontour(triang, u, levels=10, colors="k", linewidths=0.3, alpha=0.5)
        plt.colorbar(tcf, ax=ax, label="u")
        ax.set_title(f"θ = [{theta[0]:.2f}, {theta[1]:.2f}, {theta[2]:.2f}, {theta[3]:.2f}]")
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("fem_variable_kappa.png", dpi=300)
    print("Saved fem_variable_kappa.png")

    return solve, solve_batch, points, triangles


if __name__ == "__main__":
    solve, solve_batch, points, triangles = test_solver()
    print(1)
