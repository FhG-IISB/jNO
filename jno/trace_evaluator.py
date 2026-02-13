"""CORE solver using new tracing system - NO INNER VMAPS version."""

from typing import Dict, List, Callable, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import inspect

from .trace import *
from .utils import get_logger, Logger
from flax import linen as nn
from flax.serialization import from_bytes

import jax.numpy as jnp
from jax import vmap


class EulerResidualsDiff:

    @staticmethod
    def compute_euler_residuals(rho: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, p: jnp.ndarray, rho_0: jnp.ndarray, u_0: jnp.ndarray, v_0: jnp.ndarray, p_0: jnp.ndarray, n_substeps: int = 4) -> jnp.ndarray:
        """
        Compute 2D compressible Euler equation residuals for each time interval.

        For each interval i (from t=i to t=i+1):
            - Initial condition: (rho_0[i], u_0[i], v_0[i], p_0[i])
            - Target (prediction): (rho[i], u[i], v[i], p[i])
            - Linearly interpolate n_substeps between them
            - Compute residuals at each sub-step

        Args:
            rho: Predicted density, shape (21, H, W)
            u: Predicted x-velocity, shape (21, H, W)
            v: Predicted y-velocity, shape (21, H, W)
            p: Predicted pressure, shape (21, H, W)
            rho_0: Initial density, shape (21, H, W)
            u_0: Initial x-velocity, shape (21, H, W)
            v_0: Initial y-velocity, shape (21, H, W)
            p_0: Initial pressure, shape (21, H, W)
            n_substeps: Number of sub-steps within each interval (>=2)

        Returns:
            Residuals, shape (21, n_substeps-2, H, W, 4)
            Or if flattened: (21 * (n_substeps-2), H, W, 4)
        """

        # Physical parameters
        dt_interval = 1.0 / 20.0  # Time between consecutive predictions
        dt = dt_interval / (n_substeps - 1)  # Sub-step size
        dx = 1.0 / 128.0
        dy = 1.0 / 128.0
        gamma = 1.4

        def compute_interval_residuals(rho_start, u_start, v_start, p_start, rho_end, u_end, v_end, p_end):
            """
            Compute residuals for a single interval [t, t+1].

            Args:
                *_start: Initial condition at t, shape (H, W)
                *_end: Prediction at t+1, shape (H, W)

            Returns:
                Residuals at interior sub-steps, shape (n_substeps-2, H, W, 4)
            """

            # Create sub-step interpolation weights: [0, 1/(n-1), 2/(n-1), ..., 1]
            alphas = jnp.linspace(0.0, 1.0, n_substeps)  # (n_substeps,)
            alphas = alphas[:, None, None]  # (n_substeps, 1, 1) for broadcasting

            # Linearly interpolate between start and end for each sub-step
            # Shape: (n_substeps, H, W)
            rho_sub = (1.0 - alphas) * rho_start + alphas * rho_end
            u_sub = (1.0 - alphas) * u_start + alphas * u_end
            v_sub = (1.0 - alphas) * v_start + alphas * v_end
            p_sub = (1.0 - alphas) * p_start + alphas * p_end

            # Compute derived quantities
            rho_u = rho_sub * u_sub
            rho_v = rho_sub * v_sub
            E = 0.5 * rho_sub * (u_sub**2 + v_sub**2) + p_sub / (gamma - 1.0)

            # Fluxes
            flux_x_mom_x = rho_u * u_sub + p_sub  # ρu² + p
            flux_x_mom_y = rho_u * v_sub  # ρuv
            flux_y_mom_x = rho_v * u_sub  # ρvu
            flux_y_mom_y = rho_v * v_sub + p_sub  # ρv² + p

            E_plus_p = E + p_sub
            flux_E_x = E_plus_p * u_sub
            flux_E_y = E_plus_p * v_sub

            # Finite difference coefficients
            inv_2dt = 1.0 / (2.0 * dt)
            inv_2dx = 1.0 / (2.0 * dx)
            inv_2dy = 1.0 / (2.0 * dy)

            def d_dx(f):
                return (jnp.roll(f, -1, axis=-1) - jnp.roll(f, 1, axis=-1)) * inv_2dx

            def d_dy(f):
                return (jnp.roll(f, -1, axis=-2) - jnp.roll(f, 1, axis=-2)) * inv_2dy

            # Time derivatives at interior points (indices 1 to n_substeps-2)
            # Central difference: (f[i+1] - f[i-1]) / (2*dt)
            drho_dt = (rho_sub[2:] - rho_sub[:-2]) * inv_2dt
            drhou_dt = (rho_u[2:] - rho_u[:-2]) * inv_2dt
            drhov_dt = (rho_v[2:] - rho_v[:-2]) * inv_2dt
            dE_dt = (E[2:] - E[:-2]) * inv_2dt

            # Spatial derivatives at interior points
            drhou_dx = d_dx(rho_u[1:-1])
            drhov_dy = d_dy(rho_v[1:-1])

            dflux_x_mom_x_dx = d_dx(flux_x_mom_x[1:-1])
            dflux_x_mom_y_dy = d_dy(flux_x_mom_y[1:-1])

            dflux_y_mom_x_dx = d_dx(flux_y_mom_x[1:-1])
            dflux_y_mom_y_dy = d_dy(flux_y_mom_y[1:-1])

            dflux_E_x_dx = d_dx(flux_E_x[1:-1])
            dflux_E_y_dy = d_dy(flux_E_y[1:-1])

            # Residuals
            con1 = drho_dt + drhou_dx + drhov_dy
            con2 = drhou_dt + dflux_x_mom_x_dx + dflux_x_mom_y_dy
            con3 = drhov_dt + dflux_y_mom_x_dx + dflux_y_mom_y_dy
            con4 = dE_dt + dflux_E_x_dx + dflux_E_y_dy

            # Shape: (n_substeps-2, H, W, 4)
            return jnp.stack([con1, con2, con3, con4], axis=-1)

        # Vectorize over all 21 intervals
        # Each interval i: from rho_0[i] to rho[i]
        compute_all_intervals = vmap(compute_interval_residuals)

        residuals = compute_all_intervals(rho_0, u_0, v_0, p_0, rho, u, v, p)  # Start of each interval: (21, H, W)  # End of each interval: (21, H, W)

        # Shape: (21, n_substeps-2, H, W, 4)
        return residuals

    @staticmethod
    def compute_euler_residuals_flat(rho: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, p: jnp.ndarray, rho_0: jnp.ndarray, u_0: jnp.ndarray, v_0: jnp.ndarray, p_0: jnp.ndarray, n_substeps: int = 4) -> jnp.ndarray:
        """
        Same as compute_euler_residuals but returns flattened output.

        Returns:
            Residuals, shape (21 * (n_substeps-2), H, W, 4)
        """
        residuals = EulerResidualsDiff.compute_euler_residuals(rho, u, v, p, rho_0, u_0, v_0, p_0, n_substeps)
        # Reshape from (21, n_substeps-2, H, W, 4) to (21*(n_substeps-2), H, W, 4)
        n_intervals, n_interior, H, W, n_eqs = residuals.shape
        return residuals.reshape(n_intervals * n_interior, H, W, n_eqs)


# ============================================================
# Finite Difference helpers
# ============================================================


class DifferentialOperators:

    @staticmethod
    def compute_fd_gradient_2d_simple(u_values: jnp.ndarray, points: jnp.ndarray, triangles: np.ndarray, dim: int) -> jnp.ndarray:
        n_points = len(u_values)
        triangles = jnp.array(triangles)
        i_idx, j_idx, k_idx = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        p0, p1, p2 = points[i_idx], points[j_idx], points[k_idx]
        u0, u1, u2 = u_values[i_idx], u_values[j_idx], u_values[k_idx]
        dx1, dy1 = p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1]
        dx2, dy2 = p2[:, 0] - p0[:, 0], p2[:, 1] - p0[:, 1]
        areas = 0.5 * jnp.abs(dx1 * dy2 - dx2 * dy1)
        if dim == 0:
            grads = ((u1 - u0) * dy2 - (u2 - u0) * dy1) / (2 * areas + 1e-12)
        else:
            grads = ((u2 - u0) * dx1 - (u1 - u0) * dx2) / (2 * areas + 1e-12)
        grads = jnp.where(areas > 1e-12, grads, 0.0)
        area_weights = jnp.where(areas > 1e-12, areas, 0.0)
        contributions = grads * area_weights
        gradients = jnp.zeros(n_points).at[i_idx].add(contributions).at[j_idx].add(contributions).at[k_idx].add(contributions)
        weights = jnp.zeros(n_points).at[i_idx].add(area_weights).at[j_idx].add(area_weights).at[k_idx].add(area_weights)
        return jnp.where(weights > 1e-12, gradients / weights, 0.0)

    @staticmethod
    def compute_fd_gradient_3d_simple(u_values: jnp.ndarray, points: jnp.ndarray, tetrahedra: np.ndarray, dim: int) -> jnp.ndarray:
        n_points = len(u_values)
        tetrahedra = jnp.array(tetrahedra)
        i_idx, j_idx, k_idx, l_idx = tetrahedra[:, 0], tetrahedra[:, 1], tetrahedra[:, 2], tetrahedra[:, 3]
        p0, p1, p2, p3 = points[i_idx], points[j_idx], points[k_idx], points[l_idx]
        u0, u1, u2, u3 = u_values[i_idx], u_values[j_idx], u_values[k_idx], u_values[l_idx]
        v1, v2, v3 = p1 - p0, p2 - p0, p3 - p0
        volumes = jnp.abs(v1[:, 0] * (v2[:, 1] * v3[:, 2] - v2[:, 2] * v3[:, 1]) - v1[:, 1] * (v2[:, 0] * v3[:, 2] - v2[:, 2] * v3[:, 0]) + v1[:, 2] * (v2[:, 0] * v3[:, 1] - v2[:, 1] * v3[:, 0])) / 6.0
        if dim == 0:
            grads = ((u1 - u0) * (v2[:, 1] * v3[:, 2] - v2[:, 2] * v3[:, 1]) + (u2 - u0) * (v3[:, 1] * v1[:, 2] - v3[:, 2] * v1[:, 1]) + (u3 - u0) * (v1[:, 1] * v2[:, 2] - v1[:, 2] * v2[:, 1])) / (6 * volumes + 1e-12)
        elif dim == 1:
            grads = ((u1 - u0) * (v2[:, 2] * v3[:, 0] - v2[:, 0] * v3[:, 2]) + (u2 - u0) * (v3[:, 2] * v1[:, 0] - v3[:, 0] * v1[:, 2]) + (u3 - u0) * (v1[:, 2] * v2[:, 0] - v1[:, 0] * v2[:, 2])) / (6 * volumes + 1e-12)
        else:
            grads = ((u1 - u0) * (v2[:, 0] * v3[:, 1] - v2[:, 1] * v3[:, 0]) + (u2 - u0) * (v3[:, 0] * v1[:, 1] - v3[:, 1] * v1[:, 0]) + (u3 - u0) * (v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])) / (6 * volumes + 1e-12)
        grads = jnp.where(volumes > 1e-12, grads, 0.0)
        volume_weights = jnp.where(volumes > 1e-12, volumes, 0.0)
        contributions = grads * volume_weights
        gradients = jnp.zeros(n_points).at[i_idx].add(contributions).at[j_idx].add(contributions).at[k_idx].add(contributions).at[l_idx].add(contributions)
        weights = jnp.zeros(n_points).at[i_idx].add(volume_weights).at[j_idx].add(volume_weights).at[k_idx].add(volume_weights).at[l_idx].add(volume_weights)
        return jnp.where(weights > 1e-12, gradients / weights, 0.0)

    @staticmethod
    def compute_fd_laplacian_1d_simple(u_values, points, lines):
        """
        Compute finite difference Laplacian on a 1D line mesh.

        Args:
            u_values: Function values at mesh points, shape (N,)
            points: Mesh point coordinates, shape (N, 1) or (N,)
            lines: Line element connectivity, shape (M, 2)

        Returns:
            Laplacian (d²u/dx²) at each point, shape (N,)
        """
        # In 1D, Laplacian is just the second derivative: d²u/dx²
        grad = DifferentialOperators.compute_fd_gradient_1d_simple(u_values, points, lines)
        return DifferentialOperators.compute_fd_gradient_1d_simple(grad, points, lines)

    @staticmethod
    def compute_fd_laplacian_2d_simple(u_values, points, triangles, dims):
        result = jnp.zeros_like(u_values)
        for dim in dims:
            grad = DifferentialOperators.compute_fd_gradient_2d_simple(u_values, points, triangles, dim)
            result = result + DifferentialOperators.compute_fd_gradient_2d_simple(grad, points, triangles, dim)
        return result

    @staticmethod
    def compute_fd_laplacian_3d_simple(u_values, points, tetrahedra, dims):
        result = jnp.zeros_like(u_values)
        for dim in dims:
            grad = DifferentialOperators.compute_fd_gradient_3d_simple(u_values, points, tetrahedra, dim)
            result = result + DifferentialOperators.compute_fd_gradient_3d_simple(grad, points, tetrahedra, dim)
        return result

    @staticmethod
    def compute_fd_hessian_2d_simple(u_values, points, triangles, var_dims):
        """
        Compute finite difference Hessian on a 2D triangular mesh.

        Args:
            u_values: Function values at mesh points, shape (N,)
            points: Mesh point coordinates, shape (N, 2)
            triangles: Triangle connectivity, shape (M, 3)
            var_dims: List of tuples (i, vi_dim, j, vj_dim) specifying which
                      Hessian components to compute

        Returns:
            Hessian values at each point, shape (N, n_vars, n_vars)
        """
        N = points.shape[0]
        n_vars = int(jnp.sqrt(len(var_dims)))

        # First compute gradients
        grad_x = DifferentialOperators.compute_fd_gradient_2d_simple(u_values, points, triangles, 0)
        grad_y = DifferentialOperators.compute_fd_gradient_2d_simple(u_values, points, triangles, 1)

        # Then compute gradients of gradients (second derivatives)
        # d²u/dx² = d/dx(du/dx)
        d2u_dx2 = DifferentialOperators.compute_fd_gradient_2d_simple(grad_x, points, triangles, 0)
        # d²u/dxdy = d/dy(du/dx)
        d2u_dxdy = DifferentialOperators.compute_fd_gradient_2d_simple(grad_x, points, triangles, 1)
        # d²u/dy² = d/dy(du/dy)
        d2u_dy2 = DifferentialOperators.compute_fd_gradient_2d_simple(grad_y, points, triangles, 1)

        # Build full Hessian matrix for each point
        # hess_full[i, j, k] = d²u/dx_j dx_k at point i
        hess_full = jnp.zeros((N, 2, 2))
        hess_full = hess_full.at[:, 0, 0].set(d2u_dx2)
        hess_full = hess_full.at[:, 0, 1].set(d2u_dxdy)
        hess_full = hess_full.at[:, 1, 0].set(d2u_dxdy)  # Symmetry
        hess_full = hess_full.at[:, 1, 1].set(d2u_dy2)

        # Extract only the requested components
        result = jnp.zeros((N, n_vars, n_vars))
        for i, vi_dim, j, vj_dim in var_dims:
            result = result.at[:, i, j].set(hess_full[:, vi_dim, vj_dim])

        return result

    @staticmethod
    def compute_fd_hessian_3d_simple(u_values, points, tetrahedra, var_dims):
        """
        Compute finite difference Hessian on a 3D tetrahedral mesh.

        Args:
            u_values: Function values at mesh points, shape (N,)
            points: Mesh point coordinates, shape (N, 3)
            tetrahedra: Tetrahedra connectivity, shape (M, 4)
            var_dims: List of tuples (i, vi_dim, j, vj_dim) specifying which
                      Hessian components to compute

        Returns:
            Hessian values at each point, shape (N, n_vars, n_vars)
        """
        N = points.shape[0]
        n_vars = int(jnp.sqrt(len(var_dims)))

        # First compute gradients
        grad_x = DifferentialOperators.compute_fd_gradient_3d_simple(u_values, points, tetrahedra, 0)
        grad_y = DifferentialOperators.compute_fd_gradient_3d_simple(u_values, points, tetrahedra, 1)
        grad_z = DifferentialOperators.compute_fd_gradient_3d_simple(u_values, points, tetrahedra, 2)

        # Compute all second derivatives
        d2u_dx2 = DifferentialOperators.compute_fd_gradient_3d_simple(grad_x, points, tetrahedra, 0)
        d2u_dxdy = DifferentialOperators.compute_fd_gradient_3d_simple(grad_x, points, tetrahedra, 1)
        d2u_dxdz = DifferentialOperators.compute_fd_gradient_3d_simple(grad_x, points, tetrahedra, 2)
        d2u_dy2 = DifferentialOperators.compute_fd_gradient_3d_simple(grad_y, points, tetrahedra, 1)
        d2u_dydz = DifferentialOperators.compute_fd_gradient_3d_simple(grad_y, points, tetrahedra, 2)
        d2u_dz2 = DifferentialOperators.compute_fd_gradient_3d_simple(grad_z, points, tetrahedra, 2)

        # Build full Hessian matrix
        hess_full = jnp.zeros((N, 3, 3))
        hess_full = hess_full.at[:, 0, 0].set(d2u_dx2)
        hess_full = hess_full.at[:, 0, 1].set(d2u_dxdy)
        hess_full = hess_full.at[:, 0, 2].set(d2u_dxdz)
        hess_full = hess_full.at[:, 1, 0].set(d2u_dxdy)  # Symmetry
        hess_full = hess_full.at[:, 1, 1].set(d2u_dy2)
        hess_full = hess_full.at[:, 1, 2].set(d2u_dydz)
        hess_full = hess_full.at[:, 2, 0].set(d2u_dxdz)  # Symmetry
        hess_full = hess_full.at[:, 2, 1].set(d2u_dydz)  # Symmetry
        hess_full = hess_full.at[:, 2, 2].set(d2u_dz2)

        # Extract only the requested components
        result = jnp.zeros((N, n_vars, n_vars))
        for i, vi_dim, j, vj_dim in var_dims:
            result = result.at[:, i, j].set(hess_full[:, vi_dim, vj_dim])

        return result

    @staticmethod
    def compute_fd_gradient_1d_simple(u_values: jnp.ndarray, points: jnp.ndarray, lines: np.ndarray) -> jnp.ndarray:
        """
        Compute finite difference gradient on a 1D line mesh.

        Args:
            u_values: Function values at mesh points, shape (N,)
            points: Mesh point coordinates, shape (N, 1) or (N,)
            lines: Line element connectivity, shape (M, 2)

        Returns:
            Gradient (du/dx) at each point, shape (N,)
        """
        n_points = len(u_values)
        lines = jnp.array(lines)

        # Get node indices for each line element
        i_idx, j_idx = lines[:, 0], lines[:, 1]

        # Get coordinates (handle both (N,1) and (N,) shapes)
        if points.ndim == 2:
            x0, x1 = points[i_idx, 0], points[j_idx, 0]
        else:
            x0, x1 = points[i_idx], points[j_idx]

        # Get function values at element nodes
        u0, u1 = u_values[i_idx], u_values[j_idx]

        # Compute element lengths
        dx = x1 - x0
        lengths = jnp.abs(dx)

        # Compute gradient on each element: du/dx = (u1 - u0) / (x1 - x0)
        grads = (u1 - u0) / (dx + 1e-12)

        # Zero out gradients for degenerate elements
        grads = jnp.where(lengths > 1e-12, grads, 0.0)
        length_weights = jnp.where(lengths > 1e-12, lengths, 0.0)

        # Weight contributions by element length
        contributions = grads * length_weights

        # Accumulate contributions to nodes (each element contributes to both its nodes)
        gradients = jnp.zeros(n_points).at[i_idx].add(contributions).at[j_idx].add(contributions)
        weights = jnp.zeros(n_points).at[i_idx].add(length_weights).at[j_idx].add(length_weights)

        # Return weighted average
        return jnp.where(weights > 1e-12, gradients / weights, 0.0)

    @staticmethod
    def compute_fd_hessian_1d_simple(u_values, points, lines, var_dims=None):
        """
        Compute finite difference Hessian on a 1D line mesh.

        In 1D, the Hessian is simply the second derivative d²u/dx².

        Args:
            u_values: Function values at mesh points, shape (N,)
            points: Mesh point coordinates, shape (N, 1) or (N,)
            lines: Line element connectivity, shape (M, 2)
            var_dims: Optional list of tuples (i, vi_dim, j, vj_dim) for compatibility.
                      In 1D, only (0, 0, 0, 0) is valid.

        Returns:
            Hessian values at each point, shape (N, 1, 1)
        """
        N = len(u_values)

        # Compute second derivative: d²u/dx²
        grad_x = DifferentialOperators.compute_fd_gradient_1d_simple(u_values, points, lines)
        d2u_dx2 = DifferentialOperators.compute_fd_gradient_1d_simple(grad_x, points, lines)

        # Build Hessian matrix (1x1 in 1D)
        hess_full = jnp.zeros((N, 1, 1))
        hess_full = hess_full.at[:, 0, 0].set(d2u_dx2)

        # If var_dims specified, extract requested components
        if var_dims is not None:
            n_vars = int(jnp.sqrt(len(var_dims)))
            result = jnp.zeros((N, n_vars, n_vars))
            for i, vi_dim, j, vj_dim in var_dims:
                # In 1D, vi_dim and vj_dim should both be 0
                result = result.at[:, i, j].set(hess_full[:, vi_dim, vj_dim])
            return result

        return hess_full


# ============================================================
# TraceEvaluator - NO INNER VMAPS
# ============================================================


class TraceEvaluator:
    """Evaluates traced expressions - designed for JIT compilation.

    This version has NO inner vmaps. All operations are batched over N points.
    The outer vmap in core.py handles the batch dimension B.

    Shapes inside evaluate():
        points_by_tag[tag]:  (N, D)  — N points, D spatial dimensions
        tensor_tags[tag]:    (F,) or (1, F) — feature vector (already sliced per batch)

    All intermediate results should be (N,) or (N, K).
    """

    def __init__(self, params: Dict, layer_info: Dict):
        self.params = params
        self.layer_info = layer_info
        self.log = get_logger()
        self._logged_schemes = {}

    def evaluate(self, expr: Placeholder, points_by_tag: Dict[str, jnp.ndarray], var_bindings: Dict = None, tensor_tags: Dict[str, jnp.ndarray] = None, key: jax.random.PRNGKey = None) -> jnp.ndarray:
        """Evaluate expression for a SINGLE batch (no batch dimension)."""
        var_bindings = var_bindings or {}
        tensor_tags = tensor_tags or {}

        # ----------------------------------------------------------
        # Constant - value is already stored in the Constant object
        # ----------------------------------------------------------
        if isinstance(expr, Constant):
            return expr.value

        # ----------------------------------------------------------
        # Constant - python literal
        # ----------------------------------------------------------
        if isinstance(expr, Literal):
            return expr.value

        # ----------------------------------------------------------
        # TensorTag:  return tensor, broadcast-ready
        # ----------------------------------------------------------
        elif isinstance(expr, TensorTag):
            if expr.tag not in tensor_tags:
                raise ValueError(f"TensorTag '{expr.tag}' not found. Available:  {list(tensor_tags.keys())}")
            tensor = jnp.asarray(tensor_tags[expr.tag])
            if expr.dim_index is not None and tensor.ndim >= 1:
                # Select one component; result is scalar or (1,)
                tensor = tensor[..., expr.dim_index]
            # Squeeze to scalar if shape is (1,) or ()
            tensor = jnp.squeeze(tensor)
            return tensor

        # ----------------------------------------------------------
        # Variable:  slice points -> (N,) or (N, K)
        # ----------------------------------------------------------
        elif isinstance(expr, Variable):
            bound_var = var_bindings.get(id(expr), expr)
            tag = bound_var.tag

            if tag in points_by_tag:
                # Spatial coordinates:  (N, D)
                tag_data = points_by_tag[tag]
                dim_start, dim_end = bound_var.dim
                result = tag_data[..., dim_start:dim_end]

                # Squeeze trailing dim if size 1
                if dim_end is None:
                    result = jnp.squeeze(result)
                elif result.ndim >= 1 and result.shape[-1] == 1:
                    result = result[..., 0]
                return result

            elif tag in tensor_tags:
                # Parameter tensor: arbitrary shape (F,) or (F1, F2, ...)
                # Return as-is; slicing handled by Slice node if needed
                return jnp.asarray(tensor_tags[tag])

            else:
                raise KeyError(f"Variable tag '{tag}' not found. " f"points_by_tag:  {list(points_by_tag.keys())}, " f"tensor_tags: {list(tensor_tags.keys())}")

        # ----------------------------------------------------------
        # Concat:  concatenate along last axis
        # ----------------------------------------------------------
        elif isinstance(expr, Concat):
            items = [self.evaluate(item, points_by_tag, var_bindings, tensor_tags, key) for item in expr.items]
            items = [i[..., jnp.newaxis] if i.ndim == 1 else i for i in items]
            return jnp.concatenate(items, axis=-1)

        elif isinstance(expr, Reshape):
            target = self.evaluate(expr.target, points_by_tag, var_bindings, tensor_tags, key)
            return target.reshape(expr.shape)

        # ----------------------------------------------------------
        # FunctionCall:  evaluate args, call fn
        # ----------------------------------------------------------
        elif isinstance(expr, FunctionCall):
            args = [self.evaluate(arg, points_by_tag, var_bindings, tensor_tags, key) if isinstance(arg, Placeholder) else arg for arg in expr.args]
            # Check if function accepts 'key' parameter
            sig = inspect.signature(expr.fn)
            if "key" in sig.parameters:
                return expr.fn(*args, key=key)
            else:
                return expr.fn(*args)

        # ----------------------------------------------------------
        # Slice
        # ----------------------------------------------------------
        elif isinstance(expr, Slice):
            target = self.evaluate(expr.target, points_by_tag, var_bindings, tensor_tags, key)

            # Convert symbolic slice to NumPy-compatible slice
            concrete_key = []
            for k in expr.key:
                if isinstance(k, NewAxis):
                    concrete_key.append(None)  # np.newaxis
                else:
                    concrete_key.append(k)

            result = target[tuple(concrete_key)]

            # Optional scalar squeezing (safe version)
            # while result.ndim > 0 and result.shape[-1] == 1:
            #    result = result[..., 0]

            if result.ndim == 0:
                return result

            return result

        # ----------------------------------------------------------
        # BinaryOp
        # ----------------------------------------------------------
        elif isinstance(expr, BinaryOp):
            left = self.evaluate(expr.left, points_by_tag, var_bindings, tensor_tags, key)
            right = self.evaluate(expr.right, points_by_tag, var_bindings, tensor_tags, key)
            ops = {"+": jnp.add, "-": jnp.subtract, "*": jnp.multiply, "/": jnp.divide, "**": jnp.power}

            res = ops[expr.op](left, right)
            expr.debug(res)

            return res

        # ----------------------------------------------------------
        # Tracker
        # ----------------------------------------------------------
        elif isinstance(expr, Tracker):
            return self.evaluate(expr.op, points_by_tag, var_bindings, tensor_tags, key)

        # ----------------------------------------------------------
        # OperationCall
        # ----------------------------------------------------------

        elif isinstance(expr, OperationCall):
            op = expr.operation
            new_bindings = dict(var_bindings)

            op_vars = op._collected_vars

            # We bind only Variable arguments; TensorTag arguments are left to be resolved
            # from tensor_tags during evaluation.
            for op_var, call_arg in zip(op_vars, expr.args):
                if isinstance(call_arg, Variable):
                    bound_arg = var_bindings.get(id(call_arg), call_arg)
                    new_bindings[id(op_var)] = bound_arg
                elif isinstance(call_arg, TensorTag):
                    # nothing to bind; TensorTags are resolved from tensor_tags by tag name
                    pass
                else:
                    raise ValueError(f"Unsupported OperationCall argument type: {type(call_arg)}")

            return self.evaluate(op.expr, points_by_tag, new_bindings, tensor_tags, key)

        # ----------------------------------------------------------
        # FlaxModuleCall:  net(x, y, theta, ...)
        # ----------------------------------------------------------

        elif isinstance(expr, FlaxModuleCall):
            # Evaluate all arguments
            arg_values = []
            arg_sources = []

            for arg in expr.args:

                if isinstance(arg, (Placeholder, TensorTag)):
                    val = self.evaluate(arg, points_by_tag, var_bindings, tensor_tags, key)
                    arg_values.append(val)

                    # Check if this argument varies per point (spatial)
                    # It's spatial if:
                    # 1. It's a Variable in points_by_tag, OR
                    # 2. It's a Variable/TensorTag in tensor_tags with matching point dimension
                    is_spatial = False
                    if isinstance(arg, Variable):
                        if arg.tag in points_by_tag:
                            is_spatial = True
                        elif arg.tag in tensor_tags:
                            # Check if tensor has same leading dimension as points
                            tensor = tensor_tags[arg.tag]
                            if hasattr(tensor, "shape") and len(tensor.shape) >= 1:
                                # It's spatial if it has a point dimension (not just features)
                                is_spatial = True
                    elif isinstance(arg, TensorTag):
                        if arg.tag in tensor_tags:
                            is_spatial = True

                    arg_sources.append(is_spatial)
                else:

                    arg_values.append(jnp.asarray(arg))
                    arg_sources.append(False)

            # Get params and apply_fn
            flax_mod = expr.model
            layer_params = self.params.get(flax_mod.layer_id)
            apply_fn = self.layer_info.get(flax_mod.layer_id)

            if layer_params is None:
                raise ValueError(f"No params for FlaxModule {flax_mod.layer_id}")

            if "poseidon" in flax_mod.name:
                if len(arg_values) < 2:
                    raise ValueError(f"Poseidon model expects at least 2 arguments (channels and time), got {len(arg_values)}")

                # Ensure proper shapes - Poseidon expects (B, H, W, 1) or (B, H, W)
                def ensure_poseidon_shape(arr):
                    if arr.ndim == 2:
                        return arr[None, :, :, None]
                    elif arr.ndim == 3:
                        return arr[None, ...]
                    elif arr.ndim == 4:
                        return arr
                    else:
                        raise ValueError(f"Unexpected shape for Poseidon input: {arr.shape}")

                # Time should be (B,)
                time_val = jnp.asarray(arg_values[1])
                if time_val.ndim == 0:
                    time_val = time_val[None]
                elif time_val.ndim == 2:
                    time_val = time_val[:, 0]  # (B, 1) -> (B,)

                # Call the JIT-compiled apply function
                result = apply_fn(layer_params, ensure_poseidon_shape(arg_values[0]), time_val, True)

                return jnp.squeeze(result.output)

            else:
                # Standard FlaxModule handling (existing code)
                N = 1
                for val, is_spatial in zip(arg_values, arg_sources):
                    if is_spatial:
                        val = jnp.asarray(val)
                        if val.ndim >= 1:
                            N = max(N, val.shape[0])

                def normalize_arg(val, is_spatial):
                    val = jnp.asarray(val)
                    if is_spatial:
                        if val.ndim == 0:
                            return jnp.full((N, 1), val)
                        elif val.ndim == 1:
                            return val[:, jnp.newaxis]
                        else:
                            return val
                    else:
                        if val.ndim == 0:
                            return val[jnp.newaxis]
                        else:
                            return val

                shaped_args = [normalize_arg(v, s) for v, s in zip(arg_values, arg_sources)]
                # try:
                if "poseidon" in flax_mod.name:
                    result = apply_fn(layer_params, shaped_args[:-1], shaped_args[-1])
                else:
                    result = apply_fn(layer_params, *shaped_args)

                while result.ndim >= 2 and result.shape[-1] == 1:
                    result = result[..., 0]

                return result

        # ----------------------------------------------------------
        # TunableModule:  delegate to current instance
        # ----------------------------------------------------------

        elif isinstance(expr, TunableModule):
            if expr._current_instance is None:
                raise ValueError(f"TunableModule {expr} has no current instance.  " "This should be set by core.solve() before evaluation.")
            # Evaluate as if it were the concrete FlaxModule
            return self.evaluate(expr._current_instance, points_by_tag, var_bindings, tensor_tags, key)

        # ----------------------------------------------------------
        # TunableModuleCall: delegate to current instance's call
        # ----------------------------------------------------------

        elif isinstance(expr, TunableModuleCall):
            tunable = expr.model

            if tunable._current_instance is None:
                raise ValueError(f"TunableModule has no current instance. " "This should be set by core.solve() before evaluation.")

            # Create equivalent FlaxModuleCall with the concrete instance
            concrete_call = FlaxModuleCall(tunable._current_instance, expr.args)
            concrete_call.op_id = expr.op_id

            return self.evaluate(concrete_call, points_by_tag, var_bindings, tensor_tags, key)

        # ----------------------------------------------------------
        # Gradient (AD or FD)
        # ----------------------------------------------------------
        elif isinstance(expr, Gradient):
            target = expr.target
            variable = expr.variable
            scheme = expr.scheme

            bound_var = var_bindings.get(id(variable), variable)
            points = points_by_tag[bound_var.tag]  # (N, D)
            tag = bound_var.tag
            dim = variable.dim[0]

            if scheme == "finite_difference":
                domain = bound_var._domain
                if domain is None or domain.mesh_connectivity is None:
                    raise ValueError("FD scheme requires domain with mesh connectivity")
                mesh_points = jnp.array(domain.mesh_connectivity["points"])
                mesh_dim = domain.mesh_connectivity["dimension"]

                # Evaluate u at all mesh points (still need to iterate, but outside vmap context)
                def u_at_pts(pts):
                    # Override only the tag we're differentiating; keep all others
                    pts_dict = {**points_by_tag, tag: pts}
                    return self.evaluate(target, pts_dict, var_bindings, tensor_tags, key)

                u_full = u_at_pts(mesh_points)

                if mesh_dim == 2:
                    grad_full = DifferentialOperators.compute_fd_gradient_2d_simple(u_full, mesh_points, domain.mesh_connectivity["triangles"], dim)
                else:
                    grad_full = DifferentialOperators.compute_fd_gradient_3d_simple(u_full, mesh_points, domain.mesh_connectivity["tetrahedra"], dim)

                # Map back to sampled points
                dists = jnp.sum((mesh_points[jnp.newaxis, :, :] - points[:, jnp.newaxis, :]) ** 2, axis=-1)
                vertex_indices = jnp.argmin(dists, axis=1)
                return grad_full[vertex_indices]

            elif scheme == "automatic_differentiation":

                def grad_single(idx):
                    """Gradient at point index idx."""
                    pt = jax.lax.dynamic_slice(points, (idx, 0), (1, points.shape[1]))[0]

                    # Build local points_by_tag
                    local_points = {}
                    for k, v in points_by_tag.items():
                        if k == tag:
                            local_points[k] = jax.lax.dynamic_slice(v, (idx, 0), (1, v.shape[1]))
                        elif v.ndim >= 2 and v.shape[0] == points.shape[0]:
                            local_points[k] = jax.lax.dynamic_slice(v, (idx, 0), (1, v.shape[1]))
                        else:
                            local_points[k] = v

                    # Same for tensor_tags
                    local_tensors = {}
                    for k, v in tensor_tags.items():
                        if v.ndim >= 2 and v.shape[0] == points.shape[0]:
                            local_tensors[k] = jax.lax.dynamic_slice(v, (idx, 0), (1, v.shape[1]))
                        elif v.ndim == 1 and v.shape[0] == points.shape[0]:
                            local_tensors[k] = jax.lax.dynamic_slice(v, (idx,), (1,))
                        else:
                            local_tensors[k] = v

                    def u_scalar(p):
                        pts_dict = {**local_points, tag: p[jnp.newaxis, ...]}
                        result = self.evaluate(target, pts_dict, var_bindings, local_tensors, key)
                        return jnp.squeeze(result)

                    return jax.grad(u_scalar)(pt)[dim]

                N = points.shape[0]
                return jax.vmap(grad_single)(jnp.arange(N))

        # ----------------------------------------------------------
        # Laplacian (AD or FD)
        # ----------------------------------------------------------

        elif isinstance(expr, Laplacian):
            target = expr.target
            variables = expr.variables
            scheme = expr.scheme

            first_var = variables[0]
            bound_var = var_bindings.get(id(first_var), first_var)

            points = None
            if bound_var.tag in points_by_tag:
                points = points_by_tag[bound_var.tag]
                dims = tuple(v.dim[0] for v in variables)
            else:
                dims = tuple(0 for v in variables)
            tag = bound_var.tag

            if scheme == "finite_difference":
                domain = bound_var._domain
                if domain is None or domain.mesh_connectivity is None:
                    raise ValueError("FD scheme requires domain with mesh connectivity")
                mesh_points = jnp.array(domain.mesh_connectivity["points"])
                mesh_dim = domain.mesh_connectivity["dimension"]

                def u_at_pts(pts):
                    # Override only the tag we're differentiating; keep all others
                    pts_dict = {**points_by_tag, tag: pts}
                    return self.evaluate(target, pts_dict, var_bindings, tensor_tags, key)

                u_full = u_at_pts(mesh_points)

                if mesh_dim == 1:
                    lap_full = DifferentialOperators.compute_fd_laplacian_1d_simple(u_full, mesh_points, domain.mesh_connectivity["lines"])
                elif mesh_dim == 2:
                    lap_full = DifferentialOperators.compute_fd_laplacian_2d_simple(u_full, mesh_points, domain.mesh_connectivity["triangles"], dims)
                elif mesh_dim == 3:
                    lap_full = DifferentialOperators.compute_fd_laplacian_3d_simple(u_full, mesh_points, domain.mesh_connectivity["tetrahedra"], dims)

                if points is None:
                    dists = jnp.sum((mesh_points[jnp.newaxis, :, :] - points[:, jnp.newaxis, :]) ** 2, axis=-1)
                    vertex_indices = jnp.argmin(dists, axis=1)
                    return lap_full[vertex_indices]
                else:
                    return lap_full

            elif scheme == "automatic_differentiation":

                def lap_single(idx):
                    """Compute Laplacian at point index idx."""
                    pt = jax.lax.dynamic_slice(points, (idx, 0), (1, points.shape[1]))[0]

                    # Build local points_by_tag with dynamically sliced values
                    local_points = {}
                    for k, v in points_by_tag.items():
                        if k == tag:
                            # Will be overridden in u_scalar
                            local_points[k] = jax.lax.dynamic_slice(v, (idx, 0), (1, v.shape[1]))
                        elif v.ndim >= 2 and v.shape[0] == points.shape[0]:
                            # This point data varies per point, slice it dynamically
                            local_points[k] = jax.lax.dynamic_slice(v, (idx, 0), (1, v.shape[1]))
                        else:
                            local_points[k] = v

                    # Same for tensor_tags
                    local_tensors = {}
                    for k, v in tensor_tags.items():
                        if v.ndim >= 2 and v.shape[0] == points.shape[0]:
                            local_tensors[k] = jax.lax.dynamic_slice(v, (idx, 0), (1, v.shape[1]))
                        elif v.ndim == 1 and v.shape[0] == points.shape[0]:
                            local_tensors[k] = jax.lax.dynamic_slice(v, (idx,), (1,))
                        else:
                            local_tensors[k] = v

                    def u_scalar(p):
                        # Override the differentiation variable with the input point
                        pts_dict = {**local_points, tag: p[jnp.newaxis, :]}
                        result = self.evaluate(target, pts_dict, var_bindings, local_tensors, key)
                        return jnp.squeeze(result)

                    hess = jax.hessian(u_scalar)(pt)
                    return sum(hess[d, d] for d in dims)

                N = points.shape[0]
                return jax.vmap(lap_single)(jnp.arange(N))

        # ----------------------------------------------------------
        # Jacobian (AD or FD)
        # ----------------------------------------------------------

        elif isinstance(expr, Jacobian):
            target = expr.target
            variables = expr.variables
            scheme = expr.scheme

            first_var = variables[0]
            bound_var = var_bindings.get(id(first_var), first_var)
            points = points_by_tag[bound_var.tag]
            tag = bound_var.tag
            n_vars = len(variables)

            # Map each variable to its dimension index in the input
            var_dims = [(i, vi.dim[0]) for i, vi in enumerate(variables)]

            if scheme == "finite_difference":
                domain = bound_var._domain
                if domain is None or domain.mesh_connectivity is None:
                    raise ValueError("FD scheme requires domain with mesh connectivity")
                mesh_points = jnp.array(domain.mesh_connectivity["points"])
                mesh_dim = domain.mesh_connectivity["dimension"]

                def u_at_pts(pts):
                    pts_dict = {**points_by_tag, tag: pts}
                    return self.evaluate(target, pts_dict, var_bindings, tensor_tags, key)

                u_full = u_at_pts(mesh_points)

                # Compute gradient for each variable dimension and stack them
                jac_components = []
                for i, vi_dim in var_dims:

                    if mesh_dim == 1:
                        grad_full = DifferentialOperators.compute_fd_gradient_1d_simple(u_full, mesh_points, domain.mesh_connectivity["lines"])
                    elif mesh_dim == 2:
                        grad_full = DifferentialOperators.compute_fd_gradient_2d_simple(u_full, mesh_points, domain.mesh_connectivity["triangles"], vi_dim)
                    elif mesh_dim == 3:
                        grad_full = DifferentialOperators.compute_fd_gradient_3d_simple(u_full, mesh_points, domain.mesh_connectivity["tetrahedra"], vi_dim)
                    jac_components.append(grad_full)

                # Stack to form Jacobian: shape (N_mesh, n_vars)
                jac_full = jnp.stack(jac_components, axis=-1)

                # Map back to sampled points
                dists = jnp.sum((mesh_points[jnp.newaxis, :, :] - points[:, jnp.newaxis, :]) ** 2, axis=-1)
                vertex_indices = jnp.argmin(dists, axis=1)
                return jac_full[vertex_indices]

            elif scheme == "automatic_differentiation":  # automatic_differentiation

                def jac_single(pt):
                    def u_fn(p):
                        result = self.evaluate(target, {tag: p[jnp.newaxis, :]}, var_bindings, tensor_tags, key)
                        return jnp.squeeze(result)

                    jac = jax.jacobian(u_fn)(pt)  # Shape: (input_dims,) for scalar output

                    # Extract only the columns corresponding to our variables
                    result = jnp.zeros((n_vars,))
                    for i, vi_dim in var_dims:
                        result = result.at[i].set(jac[vi_dim])
                    return result

                return jax.vmap(jac_single)(points)

        # ----------------------------------------------------------
        # Hessian (AD or FD)
        # ----------------------------------------------------------

        elif isinstance(expr, Hessian):
            target = expr.target
            variables = expr.variables
            scheme = expr.scheme

            first_var = variables[0]
            bound_var = var_bindings.get(id(first_var), first_var)
            points = points_by_tag[bound_var.tag]
            tag = bound_var.tag
            n = len(variables)
            var_dims = [(i, vi.dim[0], j, vj.dim[0]) for i, vi in enumerate(variables) for j, vj in enumerate(variables)]

            if scheme == "finite_difference":
                domain = bound_var._domain
                if domain is None or domain.mesh_connectivity is None:
                    raise ValueError("FD scheme requires domain with mesh connectivity")
                mesh_points = jnp.array(domain.mesh_connectivity["points"])
                mesh_dim = domain.mesh_connectivity["dimension"]

                def u_at_pts(pts):
                    pts_dict = {**points_by_tag, tag: pts}
                    return self.evaluate(target, pts_dict, var_bindings, tensor_tags, key)

                u_full = u_at_pts(mesh_points)

                if mesh_dim == 1:
                    hess_full = DifferentialOperators.compute_fd_hessian_1d_simple(u_full, mesh_points, domain.mesh_connectivity["lines"])
                elif mesh_dim == 2:
                    hess_full = DifferentialOperators.compute_fd_hessian_2d_simple(u_full, mesh_points, domain.mesh_connectivity["triangles"], var_dims)
                elif mesh_dim == 3:
                    hess_full = DifferentialOperators.compute_fd_hessian_3d_simple(u_full, mesh_points, domain.mesh_connectivity["tetrahedra"], var_dims)

                # Map back to sampled points
                dists = jnp.sum((mesh_points[jnp.newaxis, :, :] - points[:, jnp.newaxis, :]) ** 2, axis=-1)
                vertex_indices = jnp.argmin(dists, axis=1)
                return hess_full[vertex_indices]

            elif scheme == "automatic_differentiation":  # automatic_differentiation

                def hess_single(pt):
                    def u_scalar(p):
                        result = self.evaluate(target, {tag: p[jnp.newaxis, :]}, var_bindings, tensor_tags, key)
                        return jnp.squeeze(result)

                    hess = jax.hessian(u_scalar)(pt)
                    result = jnp.zeros((n, n))
                    for i, vi_dim, j, vj_dim in var_dims:
                        result = result.at[i, j].set(hess[vi_dim, vj_dim])
                    return result

                return jax.vmap(hess_single)(points)

        # ----------------------------------------------------------
        # EulerResiduals (Finite Difference for Compressible Euler)
        # ----------------------------------------------------------

        elif isinstance(expr, EulerResiduals):
            # Evaluate the four input fields
            rho_val = self.evaluate(expr.rho, points_by_tag, var_bindings, tensor_tags, key)
            u_val = self.evaluate(expr.u, points_by_tag, var_bindings, tensor_tags, key)
            v_val = self.evaluate(expr.v, points_by_tag, var_bindings, tensor_tags, key)
            p_val = self.evaluate(expr.p, points_by_tag, var_bindings, tensor_tags, key)

            rho_val0 = self.evaluate(expr.rho0, points_by_tag, var_bindings, tensor_tags, key)
            u_val0 = self.evaluate(expr.u0, points_by_tag, var_bindings, tensor_tags, key)
            v_val0 = self.evaluate(expr.v0, points_by_tag, var_bindings, tensor_tags, key)
            p_val0 = self.evaluate(expr.p0, points_by_tag, var_bindings, tensor_tags, key)

            # Compute residuals using finite differences
            return EulerResidualsDiff.compute_euler_residuals_flat(rho_val, u_val, v_val, p_val, rho_val0, u_val0, v_val0, p_val0, expr.n_substeps)

        # ----------------------------------------------------------
        # OperationDef
        # ----------------------------------------------------------
        elif isinstance(expr, OperationDef):
            return self.evaluate(expr.expr, points_by_tag, var_bindings, tensor_tags, key)

        else:
            raise ValueError(f"Cannot evaluate:  {type(expr)}")

    @staticmethod
    def collect_dense_layers(expr: Placeholder, traverse_calls: bool = False, tensor_dims: Dict[str, int] = None) -> List:
        """Collect all FlaxModule nodes from expression tree."""

        tensor_dims = tensor_dims or {}
        layers = []
        seen = set()

        def get_shape(arg):
            """Extract shape from any Placeholder type."""
            if isinstance(arg, Variable):
                shape = tensor_dims.get(arg.tag)
                if shape is None:
                    return (1,)
                base_shape = list(shape[:-1]) if len(shape) > 1 else []
                if arg.dim is not None and arg.dim[1] is not None:
                    last_dim = arg.dim[1] - arg.dim[0]
                else:
                    last_dim = shape[-1] if shape else 1
                return tuple(base_shape + [last_dim])

            elif isinstance(arg, TensorTag):

                if arg.dim_index is not None:
                    return tensor_dims[arg.tag][:-1]

                return tensor_dims.get(arg.tag)

            elif isinstance(arg, Slice):
                target_shape = list(get_shape(arg.target))
                result_shape = []
                target_idx = 0

                for k in arg.key:
                    if isinstance(k, NewAxis):
                        result_shape.append(1)
                    elif k is Ellipsis:
                        remaining = len(target_shape) - target_idx
                        result_shape.extend(target_shape[target_idx : target_idx + remaining])
                        target_idx += remaining
                    elif isinstance(k, int):
                        target_idx += 1
                    elif isinstance(k, slice):
                        result_shape.append(target_shape[target_idx] if target_idx < len(target_shape) else 1)
                        target_idx += 1
                    else:
                        if target_idx < len(target_shape):
                            result_shape.append(target_shape[target_idx])
                            target_idx += 1

                result_shape.extend(target_shape[target_idx:])
                return tuple(result_shape) if result_shape else (1,)

            elif isinstance(arg, Concat):
                if not arg.items:
                    return (1,)
                item_shapes = [get_shape(item) for item in arg.items]
                max_ndim = max(len(s) for s in item_shapes)
                axis = arg.axis if arg.axis >= 0 else max_ndim + arg.axis

                padded_shapes = []
                for s in item_shapes:
                    if len(s) < max_ndim:
                        s = (1,) * (max_ndim - len(s)) + s
                    padded_shapes.append(s)

                result = list(padded_shapes[0])
                result[axis] = sum(s[axis] for s in padded_shapes)
                return tuple(result)

            elif isinstance(arg, Literal):
                val = jnp.asarray(arg.value)
                return val.shape if val.ndim > 0 else (1,)

            elif isinstance(arg, BinaryOp):
                left_shape = get_shape(arg.left)
                right_shape = get_shape(arg.right)
                max_ndim = max(len(left_shape), len(right_shape))
                left_padded = (1,) * (max_ndim - len(left_shape)) + left_shape
                right_padded = (1,) * (max_ndim - len(right_shape)) + right_shape
                return tuple(max(l, r) for l, r in zip(left_padded, right_padded))

            else:
                return (1,)

        def visit(node):
            if isinstance(node, FlaxModule):
                if node.layer_id not in seen:
                    seen.add(node.layer_id)
                    layers.append(node)
            elif isinstance(node, TunableModule):
                if node._current_instance is not None:
                    if node._current_instance.layer_id not in seen:
                        seen.add(node._current_instance.layer_id)
                        layers.append(node._current_instance)
            elif isinstance(node, TunableModuleCall):
                tunable = node.model
                if tunable._current_instance is not None:
                    flax_mod = tunable._current_instance
                    flax_mod._n_args = len(node.args)

                    arg_dims = []
                    for arg in node.args:
                        arg_dims.append(get_shape(arg))  # Use the outer get_shape
                    flax_mod._arg_dims = arg_dims

                    if flax_mod.layer_id not in seen:
                        seen.add(flax_mod.layer_id)
                        layers.append(flax_mod)

                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)

            elif isinstance(node, FlaxModuleCall):
                flax_mod = node.model
                flax_mod._n_args = len(node.args)

                arg_dims = []
                for arg in node.args:
                    arg_dims.append(get_shape(arg))  # Now uses the outer get_shape
                flax_mod._arg_dims = arg_dims

                visit(flax_mod)
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)

            elif isinstance(node, Concat):
                for item in node.items:
                    visit(item)
            elif isinstance(node, BinaryOp):
                visit(node.left)
                visit(node.right)
            elif isinstance(node, FunctionCall):
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
            elif isinstance(node, OperationCall):
                if traverse_calls:
                    visit(node.operation.expr)
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
            elif isinstance(node, Laplacian):
                visit(node.target)
            elif isinstance(node, Gradient):
                visit(node.target)
            elif isinstance(node, Slice):
                visit(node.target)
            elif isinstance(node, EulerResiduals):
                visit(node.rho)
                visit(node.u)
                visit(node.v)
                visit(node.p)
            elif isinstance(node, Reshape):
                visit(node.target)

        visit(expr)
        return layers

    @staticmethod
    def infer_layer_input_dims(expr: Placeholder, domain_dim: int, tensor_dims: Dict[str, int]) -> Dict[int, int]:
        """Infer input dimension for each layer by tracing the expression tree.

        Args:
            expr: The expression tree to analyze
            domain_dim: Default dimension for spatial/temporal variables
            tensor_dims: Dict mapping tensor tag names to their dimensions

        Returns:
            Dict mapping layer_id -> input_dim
        """
        layer_input_dims = {}

        def get_output_dim(node) -> int:
            """Get the output dimension of a node."""
            if isinstance(node, Variable):
                return 1  # Each variable is 1D
            elif isinstance(node, TensorTag):
                return tensor_dims.get(node.tag, 1)
            elif isinstance(node, Literal):
                val = jnp.asarray(node.value)
                return val.shape[-1] if val.ndim > 0 else 1
            elif isinstance(node, Concat):
                return sum(get_output_dim(item) for item in node.items)
            elif isinstance(node, FlaxModule):
                # For Flax modules, we can't easily know output dim without running
                # Return None to indicate it needs to be inferred at init time
                return None
            elif isinstance(node, FlaxModuleCall):
                # FlaxModuleCall output dim is unknown (depends on the module)
                return None
            elif isinstance(node, BinaryOp):
                # For binary ops, output dim is max of left/right (broadcasting)
                return max(get_output_dim(node.left), get_output_dim(node.right))
            elif isinstance(node, FunctionCall):
                # Functions preserve shape, look at first Placeholder arg
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        return get_output_dim(arg)
                return 1
            elif isinstance(node, OperationCall):
                return get_output_dim(node.operation.expr)
            elif isinstance(node, OperationDef):
                return get_output_dim(node.expr)
            elif isinstance(node, Slice):
                # Slicing can change dimension, conservatively return 1
                return 1
            elif isinstance(node, (Gradient, Laplacian, Hessian)):
                return 1
            return domain_dim

        def visit(node, upstream_dim: int):
            """Visit nodes and record input dims for layers."""
            if isinstance(node, FlaxModule):
                # Store input dim for Flax module
                layer_input_dims[node.layer_id] = node.input_dim or upstream_dim
            elif isinstance(node, FlaxModuleCall):
                # For direct FlaxModuleCall, infer input dim from arguments
                n_args = len(node.args)
                input_dim = node.model.input_dim or n_args
                layer_input_dims[node.model.layer_id] = input_dim
                # Visit arguments
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg, upstream_dim)
            elif isinstance(node, BinaryOp):
                visit(node.left, upstream_dim)
                visit(node.right, upstream_dim)
            elif isinstance(node, Concat):
                for item in node.items:
                    visit(item, upstream_dim)
            elif isinstance(node, FunctionCall):
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg, upstream_dim)
            elif isinstance(node, OperationCall):
                visit(node.operation.expr, upstream_dim)
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg, upstream_dim)
            elif isinstance(node, OperationDef):
                visit(node.expr, upstream_dim)
            elif isinstance(node, (Gradient, Laplacian, Hessian)):
                visit(node.target, upstream_dim)
            elif isinstance(node, Slice):
                visit(node.target, upstream_dim)
            elif isinstance(node, Reshape):
                visit(node.target, upstream_dim)

        visit(expr, domain_dim)
        return layer_input_dims

    @staticmethod
    def merge_pretrained_params(pretrained_params: dict, new_params: dict, logger) -> dict:
        """
        Merge pretrained weights with new params, replacing embedding/recovery layers
        when shapes don't match (for different channel dimensions).
        """
        stats = {"matched": 0, "replaced": 0}

        def count_params(arr):
            return arr.size if hasattr(arr, "size") else 0

        def merge(pretrained, new, path=""):
            if isinstance(pretrained, dict) and isinstance(new, dict):
                result = {}
                all_keys = set(list(pretrained.keys()) + list(new.keys()))

                for key in all_keys:
                    current_path = f"{path}/{key}" if path else key

                    if key in pretrained and key in new:
                        if isinstance(pretrained[key], dict):
                            result[key] = merge(pretrained[key], new[key], current_path)
                        else:
                            if pretrained[key].shape == new[key].shape:
                                result[key] = pretrained[key]
                                stats["matched"] += count_params(pretrained[key])
                            else:
                                logger.info(f"Shape mismatch at {current_path}: " f"{pretrained[key].shape} -> {new[key].shape}, reinitializing")
                                result[key] = new[key]
                                stats["replaced"] += count_params(new[key])
                    elif key in pretrained:
                        result[key] = pretrained[key]
                        if not isinstance(pretrained[key], dict):
                            stats["matched"] += count_params(pretrained[key])
                    else:
                        result[key] = new[key]
                        if not isinstance(new[key], dict):
                            stats["replaced"] += count_params(new[key])

                return result
            else:
                if hasattr(pretrained, "shape") and hasattr(new, "shape"):
                    if pretrained.shape == new.shape:
                        stats["matched"] += count_params(pretrained)
                        return pretrained
                    else:
                        stats["replaced"] += count_params(new)
                        return new
                return new if new is not None else pretrained

        merged = merge(pretrained_params, new_params)

        total = stats["matched"] + stats["replaced"]
        pct = 100 * stats["matched"] / total
        logger.info(f"Pretrained weights: {stats['matched']:,}/{total:,} params matched ({pct:.8f}%), " f"{stats['replaced']:,} reinitialized")

        return merged

    @staticmethod
    def load_poseidon_params(module, rng: jax.random.PRNGKey, num_input_channels: int, logger: Logger, weight_path: str) -> Tuple[dict, Callable]:
        """
        Load Poseidon model parameters, handling channel dimension mismatches.

        Args:
            module: The Poseidon/ScOT Flax module
            rng: Random key for initialization
            num_input_channels: Number of input channels for the target problem
            image_size: Image resolution (default 128)

        Returns:
            Tuple of (parameters, apply_function)
        """
        logger.info("Poseidon model detected.")

        # Create dummy inputs matching target dimensions
        dummy_field = jnp.ones((1, 128, 128, num_input_channels))
        dummy_time = jnp.zeros((1,))
        # dummy_channels = [dummy_field for _ in range(num_input_channels)]

        # Initialize fresh parameters with target dimensions
        fresh_params = module.init(
            {"params": rng, "dropout": rng},
            pixel_values=dummy_field,
            time=dummy_time,
            deterministic=False,
        )

        # Load pretrained weights if path provided
        if weight_path is not None:
            with open(weight_path, "rb") as f:
                pretrained_bytes = f.read()

            # First, try to load with the fresh params structure as target
            # This handles the case where channel dims match
            try:
                layer_params = from_bytes(fresh_params, pretrained_bytes)
                logger.info(f"Poseidon weights loaded from: {weight_path}")
            except Exception as e:
                # If direct load fails, load raw and merge
                logger.info(f"Direct load failed ({e}), attempting merge with channel replacement")

            # Load pretrained with its original structure
            # Create dummy init for pretrained config (4 channels)
            # pretrained_dummy = [dummy_field for _ in range(4)]  # Original Poseidon has 4 channels
            pretrained_init = module.init(
                {"params": rng, "dropout": rng},
                pixel_values=jnp.ones((1, 128, 128, 4)),
                time=dummy_time,
                deterministic=False,
            )
            pretrained_params = from_bytes(pretrained_init, pretrained_bytes)

            # Merge: keep encoder/decoder, replace mismatched layers
            layer_params = TraceEvaluator.merge_pretrained_params(pretrained_params, fresh_params, logger)
        else:
            layer_params = fresh_params
            logger.info("No pretrained path provided, using fresh initialization")

        return layer_params, module.apply

    @staticmethod
    def build_single_layer_params(layer, rng, logger):
        """Build Flax parameters for a single layer."""

        if isinstance(layer, FlaxModule):
            n_args = getattr(layer, "_n_args", None)
            arg_dims = getattr(layer, "_arg_dims", None)

            if isinstance(layer.module, nn.Module):
                if n_args is not None:
                    dummies = [jnp.ones((1,) + dim) if len(dim) == 1 else jnp.ones(dim) for dim in arg_dims]

                    # Check if this is a Poseidon model
                    is_poseidon = False
                    if hasattr(layer.module, "config"):
                        if layer.module.config.name is not None and "poseidon" in layer.module.config.name:
                            is_poseidon = True

                    if is_poseidon:
                        # Use the new loading function with channel replacement support
                        layer_params, apply_fn = TraceEvaluator.load_poseidon_params(module=layer.module, rng=rng, num_input_channels=dummies[0].shape[-1], logger=logger, weight_path=layer.weight_path)  # Exclude time from channel count

                        if layer.show:
                            df = jnp.ones((1, 128, 128, dummies[0].shape[-1]))
                            dt = jnp.zeros((1,))
                            table_str = layer.module.tabulate(jax.random.key(0), df, dt, depth=2)
                            print(table_str)

                        return layer_params, apply_fn

                    else:
                        # Standard Flax linen module initialization
                        layer_params = layer.module.init(rng, *dummies)

                        if layer.weight_path is not None:

                            with open(layer.weight_path, "rb") as f:
                                pretrained_bytes = f.read()

                            pretrained_params = from_bytes(layer_params, pretrained_bytes)
                            layer_params = TraceEvaluator.merge_pretrained_params(pretrained_params, layer_params, logger)

                        if layer.show:
                            rng = jax.random.PRNGKey(0)
                            if len(dummies) > 1:
                                table_str = layer.module.tabulate(rng, *dummies, depth=2)
                            else:
                                table_str = layer.module.tabulate(rng, dummies[0], depth=2)
                            logger.info(table_str)

                        return layer_params, layer.module.apply

                else:
                    # For single parameters (pnp.parameter)
                    layer_params = layer.module.init(rng)

                    return layer_params, layer.module.apply

            else:
                from flax import nnx

                def make_dummy(dim):
                    """Create dummy array from shape tuple."""
                    if not isinstance(dim, tuple):
                        dim = (dim,)
                    if len(dim) == 1:
                        return jnp.ones((1, dim[0]))
                    else:
                        return jnp.ones(dim)

                dummies = [make_dummy(dim) for dim in arg_dims]

                def make_nnx_functional(module):
                    """Convert NNX module to functional style (params, apply_fn)."""
                    graphdef, params, other_state = nnx.split(module, nnx.Param, ...)

                    @jax.jit
                    def apply_fn(params, *args, **kwargs):
                        model = nnx.merge(graphdef, params, other_state)
                        return model(*args, **kwargs)

                    return params, apply_fn

                layer_params, apply_fn = make_nnx_functional(layer.module)

                if layer.weight_path is not None:

                    with open(layer.weight_path, "rb") as f:
                        pretrained_bytes = f.read()

                    pretrained_params = from_bytes(layer_params, pretrained_bytes)
                    layer_params = TraceEvaluator.merge_pretrained_params(pretrained_params, layer_params, logger)

                if layer.show:
                    if len(dummies) > 1:
                        table_str = nnx.tabulate(layer.module, *dummies, depth=2)
                    else:
                        table_str = nnx.tabulate(layer.module, dummies[0], depth=2)
                    logger.info(table_str)

                return layer_params, apply_fn

        else:
            raise ValueError(f"Unknown layer type: {type(layer)}")

    @staticmethod
    def compile_traced_expression(expr: Placeholder, all_ops: List[OperationDef], layer_info) -> Callable:
        """Compile traced expression into a JAX-compatible function."""

        def evaluate_single_batch(params, points_by_tag_single, tensor_tags_single, key):
            """Evaluate for a single batch - no batch dimension."""
            evaluator = TraceEvaluator(params, layer_info)
            return evaluator.evaluate(expr, points_by_tag_single, {}, tensor_tags_single, key)

        def compiled_fn(params, points_by_tag, tensor_tags=None, batchsize=None, key=None):
            """
            Evaluate the compiled expression.

            Args:
                params: Model parameters
                points_by_tag: Dictionary of points arrays by tag
                tensor_tags: Optional dictionary of tensor arrays by tag
                batchsize: If provided, randomly select this many samples from the batch dimension.
                        If None, use all samples.
                key: JAX random key for selecting random subset. Required if batchsize is provided.
            """
            tensor_tags = tensor_tags or {}

            # Collect all unique tags from all constraints to standardize ordering
            all_tags = set()
            for op in all_ops:
                if hasattr(op, "_collected_vars"):
                    for var in op._collected_vars:
                        all_tags.add(var.tag)

            tag_order = tuple(sorted(points_by_tag.keys(), key=lambda t: (t not in all_tags, t)))
            points_tuple = tuple(points_by_tag[tag] for tag in tag_order) if tag_order else ()
            tensor_order = tuple(sorted(tensor_tags.keys()))
            tensors_tuple = tuple(tensor_tags[tag] for tag in tensor_order) if tensor_order else ()

            # ============================================================
            # STEP 1: Determine the PRIMARY batch size B
            # Use the MAXIMUM batch size found among ndim=3 points
            # Arrays with smaller "batch" dimensions are treated as non-batched
            # ============================================================
            batched_sizes = []

            # Check points: Only ndim=3 arrays are potentially batched points: (B, N, D)
            for p in points_tuple:
                if hasattr(p, "ndim") and p.ndim == 3:
                    batched_sizes.append(p.shape[0])

            # Check tensors: assume first dimension is batch if ndim >= 1
            for t in tensors_tuple:
                if hasattr(t, "ndim") and t.ndim >= 1:
                    batched_sizes.append(t.shape[0])

            if not batched_sizes:
                # No batched points or tensors → no vmap needed
                return evaluate_single_batch(params, points_by_tag, tensor_tags)

            # Use the maximum as the primary batch size
            # Arrays with different sizes won't be vmapped over
            B = max(batched_sizes)

            # ============================================================
            # STEP 1.5: Handle random subset selection if batchsize is provided
            # ============================================================
            if batchsize is not None:
                if key is None:
                    raise ValueError("A JAX random key must be provided when batchsize is specified.")

                if batchsize > B:
                    raise ValueError(f"Requested batchsize ({batchsize}) exceeds available batch size ({B}).")

                if batchsize < B:
                    # Randomly select indices
                    indices = jax.random.choice(key, B, shape=(batchsize,), replace=False)
                    indices = jnp.sort(indices)

                    # Subset points that have THE PRIMARY batch dimension
                    def subset_point(p):
                        if hasattr(p, "ndim") and p.ndim == 3 and p.shape[0] == B:
                            return p[indices]
                        return p

                    # Subset tensors that have THE PRIMARY batch dimension
                    def subset_tensor(t):
                        if hasattr(t, "ndim") and t.ndim >= 1 and t.shape[0] == B:
                            return t[indices]
                        return t

                    points_tuple = tuple(subset_point(p) for p in points_tuple)
                    tensors_tuple = tuple(subset_tensor(t) for t in tensors_tuple)

                    # Update B to the new batch size
                    B = batchsize

            # ============================================================
            # STEP 2: Normalize points - only vmap over arrays with batch size == B
            # ============================================================
            def normalize_point(arg, idx: int):
                if not hasattr(arg, "ndim"):
                    return arg, None

                if arg.ndim == 3:
                    bs = arg.shape[0]
                    if bs == B:
                        # This array has the primary batch size - vmap over it
                        return arg, 0
                    elif bs == 1:
                        # Squeeze out the singleton batch dimension
                        return jnp.squeeze(arg, axis=0), None
                    else:
                        # Different batch size - DON'T vmap, treat as constant
                        # This array will be shared across all B evaluations
                        return arg, None
                else:
                    return arg, None

            new_points = []
            points_in_axes = []
            for i, p in enumerate(points_tuple):
                p2, ax = normalize_point(p, i)
                new_points.append(p2)
                points_in_axes.append(ax)
            points_tuple = tuple(new_points)
            points_in_axes = tuple(points_in_axes)

            # ============================================================
            # STEP 3: Normalize tensors - only vmap over arrays with batch size == B
            # ============================================================
            def normalize_tensor(arg, idx: int):
                if not hasattr(arg, "ndim") or arg.ndim == 0:
                    return arg, None

                bs = arg.shape[0]
                if bs == B:
                    # This array has the primary batch size - vmap over it
                    return arg, 0
                elif bs == 1:
                    # Squeeze out the singleton batch dimension
                    return jnp.squeeze(arg, axis=0), None
                else:
                    # Different batch size - DON'T vmap, treat as constant
                    return arg, None

            new_tensors = []
            tensors_in_axes = []
            for i, t in enumerate(tensors_tuple):
                t2, ax = normalize_tensor(t, i)
                new_tensors.append(t2)
                tensors_in_axes.append(ax)
            tensors_tuple = tuple(new_tensors)
            tensors_in_axes = tuple(tensors_in_axes)

            # ============================================================
            # STEP 4: vmap - only over axes marked with 0
            # ============================================================
            def eval_single_batch_tuple(points_vals, tensor_vals, rng_key):
                pts_dict = dict(zip(tag_order, points_vals))
                tens_dict = dict(zip(tensor_order, tensor_vals)) if tensor_order else {}
                # Pass rng_key to your evaluation function
                return evaluate_single_batch(params, pts_dict, tens_dict, key=rng_key)

            if key is not None:
                # Split the key into B subkeys, one for each batch element
                keys = jax.random.split(key, B)

                vmapped_fn = jax.vmap(eval_single_batch_tuple, in_axes=(points_in_axes, tensors_in_axes, 0))  # 0 for keys axis
                return vmapped_fn(points_tuple, tensors_tuple, keys)
            else:
                # No key provided - use original function without key
                def eval_single_batch_tuple_no_key(points_vals, tensor_vals):
                    pts_dict = dict(zip(tag_order, points_vals))
                    tens_dict = dict(zip(tensor_order, tensor_vals)) if tensor_order else {}
                    return evaluate_single_batch(params, pts_dict, tens_dict)

                vmapped_fn = jax.vmap(eval_single_batch_tuple_no_key, in_axes=(points_in_axes, tensors_in_axes))
                return vmapped_fn(points_tuple, tensors_tuple)

        return compiled_fn

    @staticmethod
    def init_layer_params(all_ops: List, domain_dim: int, tensor_dims: Dict[str, int], rng: jax.Array, logger) -> Tuple[Dict, Dict, jax.Array]:
        """Initialize parameters for all layers, sharing across operations.

        Returns:
            all_params: Dict mapping layer_id -> params (single copy per unique layer)
            all_layer_info: Dict mapping layer_id -> apply_fn (single copy per unique layer)
            rng: Updated RNG key
        """
        all_params = {}  # layer_id -> params
        all_layer_info = {}  # layer_id -> apply_fn

        # First pass: collect all layers and their input dims from all ops
        all_layers = {}  # layer_id -> (layer, layer_input_dims)

        for op in all_ops:
            layers = TraceEvaluator.collect_dense_layers(op.expr, tensor_dims=tensor_dims)
            if not layers:
                continue

            layer_input_dims = TraceEvaluator.infer_layer_input_dims(op.expr, domain_dim, tensor_dims)

            for layer in layers:
                if layer.layer_id not in all_layers:
                    all_layers[layer.layer_id] = (layer, layer_input_dims)

        # Second pass: initialize each unique layer once
        for layer_id, (layer, layer_input_dims) in all_layers.items():
            rng, init_rng = jax.random.split(rng)
            layer_params, apply_fn = TraceEvaluator.build_single_layer_params(layer, init_rng, logger)
            all_params[layer_id] = layer_params
            all_layer_info[layer_id] = apply_fn

        return all_params, all_layer_info, rng
