from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any

import meshio
import numpy as np

from ..utils.logger import Logger, PrintFallback
from .mesh_utils import MeshUtils


class MeshIOMixin(MeshUtils):
    """Mesh loading and export helpers shared by domain-like classes."""

    log: Logger | PrintFallback
    context: Dict[str, Any]
    _param_tags: set[str]
    spatial: List[str]
    mesh: meshio.Mesh | None
    dimension: int

    def _load_mesh(self, mesh_file: str):
        """Load mesh from file using Meshio."""
        import meshio

        from pathlib import Path

        if not Path(mesh_file).exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

        self.mesh = meshio.read(mesh_file)

        points = self.mesh.points  # type: ignore[attr-defined]
        if points.shape[1] == 3 and np.allclose(points[:, 2], 0):
            self.dimension = 2
        else:
            self.dimension = points.shape[1]

    def _write_meshio_safely(self, save_path: str, file_format: Optional[str] = None):
        """Write mesh with meshio, falling back to a cell_set-free copy if needed."""
        import os

        if self.mesh is None:
            raise ValueError("No mesh available to export")

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        try:
            meshio.write(save_path, self.mesh, file_format=file_format)
        except IndexError as e:
            # Some meshio VTK writers can fail when converting cell_sets to
            # cell_data if set indices are malformed or global-indexed.
            # Fallback: export geometry/cells without cell_sets.
            msg = str(e)
            if "cell_sets" not in msg and "out of bounds" not in msg:
                raise

            sanitized_mesh = meshio.Mesh(
                points=self.mesh.points,
                cells=self.mesh.cells,
                point_data=getattr(self.mesh, "point_data", None),
                cell_data=getattr(self.mesh, "cell_data", None),
                field_data=getattr(self.mesh, "field_data", None),
            )
            meshio.write(save_path, sanitized_mesh, file_format=file_format)
            self.log.warning("Mesh export dropped cell_sets due to meshio conversion issue")

    def export_vtk(self, save_path: str = "./runs/domain.vtk", file_format: Optional[str] = None):
        """Export the current mesh to a VTK file for external viewers.

        The output can be opened in ParaView, PyVista, or many browser-based
        viewers that support VTK-compatible formats.
        """
        self._write_meshio_safely(save_path, file_format=file_format)
        self.log.info(f"Saved mesh to {save_path}")

    def export_msh(self, save_path: str = "./runs/domain.msh", file_format: str = "gmsh22"):
        """Export the current mesh to Gmsh .msh format."""
        self._write_meshio_safely(save_path, file_format=file_format)
        self.log.info(f"Saved mesh to {save_path}")

    def export(self, save_path: str, fmt: Optional[str] = None, show_sampled: bool = True, figsize: Tuple[int, int] = (10, 8)):
        """Unified export helper.

        Supported formats:
        - png/jpg/jpeg/svg/pdf: static mesh image via ``plot_mesh``
        - vtk/vtu: VTK mesh export
        - msh: Gmsh mesh export
        - html/htm: interactive browser view via Plotly
        """
        import os

        _fmt = (fmt or os.path.splitext(save_path)[1].lower().lstrip(".")).lower()
        if _fmt in {"png", "jpg", "jpeg", "svg", "pdf"}:
            self.plot_mesh(save_path=save_path, figsize=figsize, show_sampled=show_sampled)
        elif _fmt in {"vtk", "vtu"}:
            self.export_vtk(save_path=save_path, file_format=_fmt)
        elif _fmt in {"msh", "gmsh"}:
            self.export_msh(save_path=save_path, file_format="gmsh22")
        elif _fmt in {"html", "htm"}:
            self.export_interactive_html(save_path=save_path, show_sampled=show_sampled)
        else:
            raise ValueError(f"Unsupported export format '{_fmt}'. Supported: png, jpg, svg, pdf, vtk, vtu, msh, html")

    @staticmethod
    def _set_3d_equal_axes(ax, points: np.ndarray):
        """Set equal scaling on all 3D axes for better shape perception."""
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = 0.5 * (mins + maxs)
        radius = 0.5 * np.max(maxs - mins)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    def plot_mesh(self, save_path: str = "./runs/domain_mesh.png", figsize: Tuple[int, int] = (10, 8), show_sampled: bool = True):
        """Plot the actual mesh (elements), optionally overlaying sampled points."""
        import os
        import matplotlib.pyplot as plt

        if self.mesh is None:
            raise ValueError("No mesh available to plot")

        points = np.asarray(self.mesh.points[:, : self.dimension])
        spatial_dim = self.dimension

        if spatial_dim == 3:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

            if "tetra" in self.mesh.cells_dict:
                boundary_faces = self._get_boundary_elements(self.mesh.cells_dict["tetra"], "tetra")
                face_xyz = points[boundary_faces]
                poly = Poly3DCollection(face_xyz, facecolor=(0.2, 0.55, 0.9, 0.12), edgecolor=(0.15, 0.15, 0.2, 0.35), linewidth=0.25)
                ax.add_collection3d(poly)
            elif "triangle" in self.mesh.cells_dict:
                tri = self.mesh.cells_dict["triangle"]
                face_xyz = points[tri]
                poly = Poly3DCollection(face_xyz, facecolor=(0.2, 0.55, 0.9, 0.12), edgecolor=(0.15, 0.15, 0.2, 0.35), linewidth=0.25)
                ax.add_collection3d(poly)

            if show_sampled and self.context:
                for tag, data in self.context.items():
                    if tag.startswith("n_") or tag in self._param_tags or tag == "__time__":
                        continue
                    arr = np.asarray(data)
                    if arr.ndim >= 4:
                        pts = arr[0, 0]
                        if pts.shape[-1] >= 3:
                            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, alpha=0.5)

            self._set_3d_equal_axes(ax, points)
            ax.set_xlabel(self.spatial[0])
            ax.set_ylabel(self.spatial[1])
            ax.set_zlabel(self.spatial[2])
            ax.set_title("Mesh")
        else:
            fig, ax = plt.subplots(figsize=figsize)
            if "triangle" in self.mesh.cells_dict and points.shape[1] >= 2:
                import matplotlib.tri as mtri

                tri = np.asarray(self.mesh.cells_dict["triangle"])
                triang = mtri.Triangulation(points[:, 0], points[:, 1], tri)
                ax.triplot(triang, color="0.3", linewidth=0.45, alpha=0.8)
            elif "line" in self.mesh.cells_dict and points.shape[1] >= 1:
                for e in np.asarray(self.mesh.cells_dict["line"]):
                    p0, p1 = points[e[0]], points[e[1]]
                    x0, x1 = p0[0], p1[0]
                    y0 = p0[1] if points.shape[1] > 1 else 0.0
                    y1 = p1[1] if points.shape[1] > 1 else 0.0
                    ax.plot([x0, x1], [y0, y1], color="0.3", linewidth=0.8)

            if show_sampled and self.context:
                for tag, data in self.context.items():
                    if tag.startswith("n_") or tag in self._param_tags or tag == "__time__":
                        continue
                    arr = np.asarray(data)
                    if arr.ndim >= 4:
                        pts = arr[0, 0]
                    elif arr.ndim >= 2:
                        pts = arr[0]
                    else:
                        continue
                    if pts.shape[-1] >= 2:
                        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.5, label=tag)

            if spatial_dim >= 2:
                ax.set_aspect("equal")
                ax.set_xlabel(self.spatial[0])
                ax.set_ylabel(self.spatial[1])
            else:
                ax.set_xlabel(self.spatial[0])
                ax.set_yticks([])
            ax.set_title("Mesh")
            if show_sampled and self.context:
                ax.legend(loc="best", fontsize=8)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close()
        self.log.info(f"Saved mesh plot to {save_path}")

    def export_interactive_html(self, save_path: str = "./runs/domain_mesh.html", show_sampled: bool = True):
        """Export an interactive mesh visualization as HTML (Plotly).

        Open the generated HTML in any browser to rotate/pan/zoom.
        """
        import os

        if self.mesh is None:
            raise ValueError("No mesh available to export")

        try:
            import plotly.graph_objects as go
        except Exception as e:
            raise ImportError("plotly is required for interactive HTML export") from e

        points = np.asarray(self.mesh.points[:, : self.dimension])
        traces = []

        if self.dimension == 3:
            if "tetra" in self.mesh.cells_dict:
                faces = self._get_boundary_elements(self.mesh.cells_dict["tetra"], "tetra")
            elif "triangle" in self.mesh.cells_dict:
                faces = np.asarray(self.mesh.cells_dict["triangle"])
            else:
                faces = None

            if faces is not None and len(faces) > 0:
                traces.append(
                    go.Mesh3d(
                        x=points[:, 0],
                        y=points[:, 1],
                        z=points[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        opacity=0.25,
                        color="royalblue",
                        name="mesh",
                    )
                )

            if show_sampled and self.context:
                for tag, data in self.context.items():
                    if tag.startswith("n_") or tag in self._param_tags or tag == "__time__":
                        continue
                    arr = np.asarray(data)
                    if arr.ndim >= 4:
                        pts = arr[0, 0]
                        if pts.shape[-1] >= 3:
                            traces.append(
                                go.Scatter3d(
                                    x=pts[:, 0],
                                    y=pts[:, 1],
                                    z=pts[:, 2],
                                    mode="markers",
                                    marker=dict(size=2),
                                    name=tag,
                                )
                            )
        else:
            if "triangle" in self.mesh.cells_dict and points.shape[1] >= 2:
                tri = np.asarray(self.mesh.cells_dict["triangle"])
                edge_segments = []
                for a, b, c in tri:
                    edge_segments.extend(
                        [
                            (points[a, 0], points[a, 1]),
                            (points[b, 0], points[b, 1]),
                            (None, None),
                            (points[b, 0], points[b, 1]),
                            (points[c, 0], points[c, 1]),
                            (None, None),
                            (points[c, 0], points[c, 1]),
                            (points[a, 0], points[a, 1]),
                            (None, None),
                        ]
                    )
                xe = [p[0] for p in edge_segments]
                ye = [p[1] for p in edge_segments]
                traces.append(go.Scatter(x=xe, y=ye, mode="lines", line=dict(width=1, color="rgba(70,70,70,0.5)"), name="mesh"))

            if show_sampled and self.context:
                for tag, data in self.context.items():
                    if tag.startswith("n_") or tag in self._param_tags or tag == "__time__":
                        continue
                    arr = np.asarray(data)
                    if arr.ndim >= 4:
                        pts = arr[0, 0]
                    elif arr.ndim >= 2:
                        pts = arr[0]
                    else:
                        continue
                    if pts.shape[-1] >= 2:
                        traces.append(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", marker=dict(size=4), name=tag))

        fig = go.Figure(data=traces)
        fig.update_layout(title="Interactive Mesh", template="plotly_white")
        if self.dimension == 2:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.write_html(save_path, include_plotlyjs="cdn")
        self.log.info(f"Saved interactive mesh HTML to {save_path}")
