"""
visualization.py - Visualization utilities for GCS-MMS demo.

Provides:
- Environment and region plotting
- Trajectory visualization
- Animation generation
- Result summary figures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba
from typing import List, Dict, Tuple, Optional
import os
import textwrap

from convex_regions import ConvexRegion
from graph_builder import RegionGraph, SOURCE, TARGET
from graph_types import is_region_node_id, region_index_from_node_id, region_node_label
from optimizer import OptimizationResult


def setup_plot_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'axes.facecolor': '#fbfbfb',
        'figure.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.grid': False,
    })


def _status_color(success: bool) -> str:
    """Consistent status color."""
    return '#2E7D32' if success else '#B71C1C'


def _compact_float(value: float) -> str:
    """Human-friendly numeric formatting for figure panels."""
    if not np.isfinite(value):
        return 'N/A'
    abs_value = abs(float(value))
    if abs_value != 0.0 and (abs_value < 1e-3 or abs_value >= 1e4):
        return f"{value:.2e}"
    return f"{value:.4g}"


def _result_total_duration(result: OptimizationResult) -> float:
    """Total physical trajectory duration."""
    return float(sum(result.time_durations.values()))


def _collect_profile_data(result: OptimizationResult) -> Tuple[np.ndarray, np.ndarray]:
    """Collect speed samples from trajectory chords for plotting."""
    times = []
    velocities = []
    cumulative_t = 0.0

    for traj, tau, delta in result.trajectories:
        for i in range(len(traj) - 1):
            dt = (tau[i + 1] - tau[i]) * delta
            if dt > 1e-10:
                pos_diff = traj[i + 1, :2] - traj[i, :2]
                vel = np.linalg.norm(pos_diff) / dt
                times.append(cumulative_t + tau[i] * delta)
                velocities.append(vel)
        cumulative_t += delta

    return np.asarray(times, dtype=np.float64), np.asarray(velocities, dtype=np.float64)


def _draw_text_panel(ax, title: str, lines: List[str],
                     accent_color: str = '#263238',
                     fontsize: float = 8.5) -> None:
    """Draw a clean text panel with a small accent bar."""
    ax.axis('off')
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.0, 0.0), 1.0, 1.0,
            boxstyle='round,pad=0.015,rounding_size=0.02',
            transform=ax.transAxes,
            facecolor='white',
            edgecolor='#dddddd',
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (0.0, 0.0), 0.018, 1.0,
            transform=ax.transAxes,
            facecolor=accent_color,
            edgecolor='none',
            zorder=1,
        )
    )
    ax.text(
        0.055, 0.94, title,
        transform=ax.transAxes,
        fontsize=10,
        fontweight='bold',
        color='#212121',
        va='top',
    )
    ax.text(
        0.055, 0.86, '\n'.join(lines),
        transform=ax.transAxes,
        fontfamily='monospace',
        fontsize=fontsize,
        color='#263238',
        va='top',
        linespacing=1.28,
    )


def _polygon_vertices(polygon_like) -> np.ndarray:
    """Convert a shapely polygon or raw vertex array into plot-ready vertices."""
    if hasattr(polygon_like, "exterior"):
        vertices = np.asarray(polygon_like.exterior.coords, dtype=np.float64)
        if len(vertices) > 1 and np.allclose(vertices[0], vertices[-1]):
            return vertices[:-1]
        return vertices
    return np.asarray(polygon_like, dtype=np.float64)


def _draw_geometry_lines(ax, geometry, *, color: str, linewidth: float,
                         linestyle: str, zorder: int) -> None:
    """Draw LineString/MultiLineString/Polygon-like shapely geometries."""
    if geometry is None or geometry.is_empty:
        return

    geom_type = geometry.geom_type
    if geom_type == "LineString":
        coords = np.asarray(geometry.coords, dtype=np.float64)
        if len(coords) >= 2:
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                solid_capstyle="round",
                dash_capstyle="round",
                zorder=zorder,
            )
        return

    if geom_type == "Polygon":
        coords = np.asarray(geometry.exterior.coords, dtype=np.float64)
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=zorder,
        )
        return

    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            _draw_geometry_lines(
                ax,
                part,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=zorder,
            )


def _draw_shared_region_edges(ax,
                              regions: List[ConvexRegion],
                              highlight_regions: List[int]) -> None:
    """Highlight shared boundaries/interfaces between consecutive path regions."""
    region_by_index = {region.index: region for region in regions}

    for left_idx, right_idx in zip(highlight_regions[:-1], highlight_regions[1:]):
        left_region = region_by_index.get(left_idx)
        right_region = region_by_index.get(right_idx)
        if left_region is None or right_region is None:
            continue

        left_poly = left_region.get_shapely_polygon()
        right_poly = right_region.get_shapely_polygon()
        boundary_intersection = left_poly.boundary.intersection(right_poly.boundary)
        if not boundary_intersection.is_empty and boundary_intersection.length > 1e-9:
            _draw_geometry_lines(
                ax,
                boundary_intersection,
                color="#ff8c00",
                linewidth=2.3,
                linestyle="--",
                zorder=9,
            )
            continue

        overlap = left_poly.intersection(right_poly)
        if not overlap.is_empty:
            _draw_geometry_lines(
                ax,
                overlap,
                color="#ff8c00",
                linewidth=2.0,
                linestyle="--",
                zorder=9,
            )


def plot_environment(ax, workspace_bounds: Tuple[float, float, float, float],
                     regions: List[ConvexRegion],
                     start_pos: np.ndarray, goal_pos: np.ndarray,
                     obstacles: Optional[List[object]] = None,
                     highlight_regions: Optional[List[int]] = None,
                     show_labels: bool = True,
                     alpha: float = 0.22,
                     max_labeled_regions: int = 45):
    """
    Plot the planning environment with convex regions.
    
    Args:
        ax: Matplotlib axes
        workspace_bounds: (x_min, x_max, y_min, y_max)
        regions: List of ConvexRegion objects
        start_pos: Start position [x, y]
        goal_pos: Goal position [x, y]
        obstacles: Obstacle polygons to overlay
        highlight_regions: Region indices to highlight (path)
        show_labels: Whether to show region labels
        alpha: Transparency for regions
    """
    x_min, x_max, y_min, y_max = workspace_bounds
    
    # Plot workspace boundary
    workspace_rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=1.8, edgecolor='#202020', facecolor='#f7f8fa', zorder=0
    )
    ax.add_patch(workspace_rect)
    
    # Colormap for regions
    cmap = plt.cm.get_cmap('tab10')
    use_path_style = highlight_regions is not None
    highlight_region_set = set(highlight_regions or [])
    highlight_set = highlight_region_set  # alias for post-loop label checks
    show_region_labels = bool(show_labels and len(regions) <= max_labeled_regions)
    selected_face = "#b8dcff"
    selected_edge = "#005ea8"

    # Plot regions
    for region in regions:
        color = cmap(region.index % 10)
        is_highlighted = region.index in highlight_region_set

        if use_path_style:
            face_color = to_rgba(selected_face, 0.36) if is_highlighted else "none"
            edge_width = 2.8 if is_highlighted else 0.9
            edge_color = selected_edge if is_highlighted else "#d0d0d0"
            zorder = 3 if is_highlighted else 1
        else:
            face_color = to_rgba(color, alpha)
            edge_width = 1.0
            edge_color = color
            zorder = 1
        
        poly = patches.Polygon(
            region.vertices,
            closed=True,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=edge_width,
            zorder=zorder,
        )
        ax.add_patch(poly)

        # Label — always drawn with a white backing so maze obstacles cannot hide it.
        if show_region_labels or region.index in highlight_set:
            centroid = region.get_centroid()
            label_color = "#003f73" if is_highlighted else ("#555555" if use_path_style else "#222222")
            ax.text(
                centroid[0], centroid[1], f'R{region.index}',
                ha='center', va='center',
                fontsize=8.5 if is_highlighted else 7.5 if use_path_style else 8.0,
                color=label_color,
                zorder=15,
                fontweight='bold' if is_highlighted else 'normal',
                bbox=dict(
                    facecolor='white',
                    alpha=0.72,
                    edgecolor='none',
                    pad=1.2,
                    boxstyle='round,pad=0.18',
                ),
            )

    # Overlay obstacles explicitly so blocked space is easy to see.
    for obstacle_idx, obstacle in enumerate(obstacles or []):
        obstacle_patch = patches.Polygon(
            _polygon_vertices(obstacle),
            closed=True,
            facecolor='#111111',
            edgecolor='#111111',
            linewidth=2.2,
            joinstyle='miter',
            alpha=1.0,
            zorder=6,
            label='Obstacle' if obstacle_idx == 0 else None,
        )
        ax.add_patch(obstacle_patch)

    if use_path_style:
        _draw_shared_region_edges(ax, regions, list(highlight_regions or []))

    # Plot start and goal
    ax.scatter(start_pos[0], start_pos[1], s=150, marker='o',
               facecolor='#2E7D32', edgecolor='white', linewidth=1.8,
               label='Start', zorder=10)
    ax.scatter(goal_pos[0], goal_pos[1], s=170, marker='*',
               facecolor='#C62828', edgecolor='white', linewidth=1.2,
               label='Goal', zorder=10)
    
    ax.set_xlim(x_min - 0.2, x_max + 0.2)
    ax.set_ylim(y_min - 0.2, y_max + 0.2)
    ax.set_aspect('equal')
    ax.grid(True, color='#d7dce0', linewidth=0.6, alpha=0.75)
    ax.tick_params(colors='#37474F')
    # Place legend outside the axes to avoid covering map content.
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(0.0, -0.04),
        borderaxespad=0,
        ncol=4,
        frameon=True,
        framealpha=0.94,
        facecolor='white',
        edgecolor='#dddddd',
        fontsize=8.5,
    )


def _collect_mesh_points(result: OptimizationResult,
                         fallback_points_per_segment: int = 5,
                         include_endpoints: bool = True,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect mesh sample points and their physical times.

    Uses exact solver mesh samples when available, with a trajectory-based
    fallback for older results.
    """
    mesh_points = []
    mesh_times = []
    cumulative_time = 0.0

    if result.mesh_samples:
        for mesh_positions, mesh_tau, delta in result.mesh_samples:
            if len(mesh_positions) > 0:
                if include_endpoints:
                    indices = range(len(mesh_positions))
                else:
                    indices = range(1, max(1, len(mesh_positions) - 1))
                for idx in indices:
                    mesh_points.append(mesh_positions[idx])
                    mesh_times.append(cumulative_time + mesh_tau[idx] * delta)
            cumulative_time += delta
    else:
        for traj, tau, delta in result.trajectories:
            sample_count = min(fallback_points_per_segment, len(traj))
            if sample_count > 0:
                indices = np.linspace(0, len(traj) - 1, sample_count, dtype=int)
                indices = np.unique(indices)
                if not include_endpoints:
                    indices = indices[1:-1] if len(indices) > 2 else []
                for idx in indices:
                    mesh_points.append(traj[idx, :2])
                    mesh_times.append(cumulative_time + tau[idx] * delta)
            cumulative_time += delta

    if not mesh_points:
        return np.empty((0, 2)), np.empty((0,))

    return np.array(mesh_points), np.array(mesh_times)


def _compute_velocity_profiles(
    result: OptimizationResult,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute time-series of linear speed and angular velocity from trajectory states.

    Returns (times, linear_velocities, angular_velocities) arrays aligned by sample
    midpoints.  Angular velocity is derived from finite-differences of the heading
    state (index 2 of unicycle state [px, py, theta]).
    """
    times: List[float] = []
    lin_vels: List[float] = []
    ang_vels: List[float] = []
    cumulative_t = 0.0

    for traj, tau, delta in result.trajectories:
        n = len(traj)
        for i in range(n - 1):
            dt = (tau[i + 1] - tau[i]) * delta
            if dt < 1e-10:
                continue
            t_mid = cumulative_t + 0.5 * (tau[i] + tau[i + 1]) * delta
            dp = traj[i + 1, :2] - traj[i, :2]
            v = float(np.linalg.norm(dp) / dt)
            if traj.shape[1] > 2:
                dtheta = float(traj[i + 1, 2] - traj[i, 2])
                # Wrap to [-pi, pi]
                dtheta = float(np.arctan2(np.sin(dtheta), np.cos(dtheta)))
                omega = dtheta / dt
            else:
                omega = 0.0
            times.append(t_mid)
            lin_vels.append(v)
            ang_vels.append(omega)
        cumulative_t += delta

    if not times:
        return np.empty(0), np.empty(0), np.empty(0)
    return np.asarray(times), np.asarray(lin_vels), np.asarray(ang_vels)


def plot_trajectory(ax, result: OptimizationResult, regions: List[ConvexRegion],
                    show_mesh_points: bool = True,
                    show_interface_points: bool = True,
                    show_heading: bool = True,
                    trajectory_color: str = '#2196F3',
                    linewidth: float = 2.5):
    """
    Plot the optimized trajectory.
    
    Args:
        ax: Matplotlib axes
        result: OptimizationResult from optimizer
        regions: List of ConvexRegion objects
        show_mesh_points: Whether to show mesh sampling points
        show_interface_points: Whether to show interface points
        show_heading: Whether to show heading arrows
        trajectory_color: Color for trajectory line
        linewidth: Line width
    """
    if not result.trajectories:
        return
    
    # Collect all trajectory segments
    all_positions = []
    all_times = []
    cumulative_time = 0.0
    
    for traj, tau, delta in result.trajectories:
        positions = traj[:, :2]
        times = tau * delta + cumulative_time
        
        all_positions.append(positions)
        all_times.append(times)
        cumulative_time += delta
    
    # Plot trajectory segments. A subtle path shadow helps on dense maze maps.
    for i, (positions, times) in enumerate(zip(all_positions, all_times)):
        ax.plot(positions[:, 0], positions[:, 1],
                color='white', linewidth=linewidth + 2.8,
                solid_capstyle='round', zorder=5, alpha=0.95)
        ax.plot(positions[:, 0], positions[:, 1],
                color=trajectory_color, linewidth=linewidth,
                solid_capstyle='round', zorder=6,
                label='Trajectory' if i == 0 else None)
        
        if show_heading:
            traj = result.trajectories[i][0]
            sample_count = min(3, len(traj))
            if sample_count > 0:
                indices = np.linspace(0, len(traj) - 1, sample_count, dtype=int)
                for idx in np.unique(indices):
                    pos = traj[idx, :2]
                    theta = traj[idx, 2]
                    arrow_len = 0.12
                    dx = arrow_len * np.cos(theta)
                    dy = arrow_len * np.sin(theta)
                    ax.arrow(pos[0], pos[1], dx, dy, head_width=0.055,
                            head_length=0.04, fc='#0D47A1', ec='#0D47A1',
                            alpha=0.72, zorder=7, length_includes_head=True)
    
    # Interface points
    if show_interface_points and result.interface_points:
        interface_pts = np.array(result.interface_points)
        ax.scatter(interface_pts[:, 0], interface_pts[:, 1],
                  c='#F57C00', s=76, marker='D', zorder=8,
                  edgecolors='white', linewidths=1.2,
                  label='Interface Points')

    # Mesh points — only for legacy solvers; log_barrier (CRD) uses RK4 nodes inline.
    if show_mesh_points and result.safety_mode != "log_barrier":
        mesh_pts, _ = _collect_mesh_points(result)
        if len(mesh_pts) > 0:
            ax.scatter(mesh_pts[:, 0], mesh_pts[:, 1],
                      c='#7B1FA2', s=24, marker='o', zorder=7,
                      alpha=0.72, edgecolors='white', linewidths=0.35,
                      label='Mesh Points')


def plot_graph_structure(ax, graph: RegionGraph, 
                         highlight_path: Optional[List[str]] = None):
    """
    Plot the graph structure overlaid on regions.
    
    Args:
        ax: Matplotlib axes
        graph: RegionGraph object
        highlight_path: Path to highlight
    """
    import networkx as nx
    
    # Compute node positions
    pos = {}
    pos[SOURCE] = graph.start_pos
    pos[TARGET] = graph.goal_pos
    
    for region in graph.regions:
        node_id = region_node_label(region.index)
        pos[node_id] = region.get_centroid()
    
    # Draw edges
    for u, v in graph.graph.edges():
        if highlight_path and u in highlight_path and v in highlight_path:
            # Check if consecutive in path
            try:
                idx_u = highlight_path.index(u)
                idx_v = highlight_path.index(v)
                if abs(idx_u - idx_v) == 1:
                    color = 'blue'
                    width = 2.5
                    alpha = 0.8
                else:
                    color = 'gray'
                    width = 0.5
                    alpha = 0.3
            except ValueError:
                color = 'gray'
                width = 0.5
                alpha = 0.3
        else:
            color = 'gray'
            width = 0.5
            alpha = 0.3
        
        p1 = pos[u]
        p2 = pos[v]
        
        ax.annotate("", xy=p2, xytext=p1,
                   arrowprops=dict(arrowstyle="->", color=color,
                                  lw=width, alpha=alpha,
                                  connectionstyle="arc3,rad=0.1"))


def create_result_figure(result: OptimizationResult, graph: RegionGraph,
                         workspace_bounds: Tuple[float, float, float, float],
                         start_pos: np.ndarray, goal_pos: np.ndarray,
                         obstacles: Optional[List[object]] = None,
                         title: str = "GCS-MMS Motion Planning Result",
                         figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Create comprehensive result figure.
    
    Args:
        result: OptimizationResult
        graph: RegionGraph
        workspace_bounds: Environment bounds
        start_pos, goal_pos: Start and goal positions
        obstacles: Obstacle polygons to overlay
        title: Figure title
        figsize: Figure size
        
    Returns:
        Matplotlib Figure
    """
    setup_plot_style()
    
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = gridspec.GridSpec(
        3, 3,
        figure=fig,
        width_ratios=[1.25, 1.25, 1.0],
        height_ratios=[1.0, 0.9, 0.95],
    )
    
    # Main plot: environment + trajectory
    ax_main = fig.add_subplot(grid[:, :2])
    
    # Plot environment
    plot_environment(ax_main, workspace_bounds, graph.regions,
                     start_pos, goal_pos,
                     obstacles=obstacles,
                     highlight_regions=result.path_regions if result.path_regions else None,
                     show_labels=len(graph.regions) <= 80)
    
    # Plot trajectory
    if result.trajectories:
        plot_trajectory(ax_main, result, graph.regions)
        violating_points = []
        violating_labels = []
        for region_idx, diagnostic in result.safety_diagnostics.items():
            if diagnostic.get('max_violation', -np.inf) > 0.0 and diagnostic.get('worst_point') is not None:
                violating_points.append(diagnostic['worst_point'])
                violating_labels.append(
                    f"R{region_idx}: {diagnostic.get('max_violation', 0.0):.2e}"
                )
        if violating_points:
            pts = np.asarray(violating_points, dtype=np.float64)
            ax_main.scatter(
                pts[:, 0], pts[:, 1],
                c='#D50000', s=90, marker='x', linewidths=2.2,
                zorder=12, label='Worst dense violation'
            )
            for point, label in zip(pts[:8], violating_labels[:8]):
                ax_main.text(
                    point[0], point[1], label,
                    fontsize=7.5, color='#B71C1C',
                    ha='left', va='bottom', zorder=13,
                    bbox=dict(facecolor='white', edgecolor='#ffcdd2', alpha=0.85, pad=1.5),
                )
    
    status = 'SUCCESS' if result.success else 'FAILED'
    ax_main.set_title(f"{title}\n{status}", color=_status_color(result.success),
                      fontweight='bold')
    ax_main.set_xlabel("x [m]")
    ax_main.set_ylabel("y [m]")
    # ax_main.legend(loc='upper right', frameon=True, framealpha=0.92,
    #                facecolor='white', edgecolor='#dddddd')
    
    # Info panel
    ax_info = fig.add_subplot(grid[0, 2])

    path_labels = []
    for node_id in result.path:
        if is_region_node_id(node_id):
            region_idx = region_index_from_node_id(node_id)
            delta = result.time_durations.get(region_idx)
            if delta is not None:
                path_labels.append(f"{node_id} [{delta:.2f}s]")
                continue
        path_labels.append(node_id)

    path_text = ' -> '.join(path_labels) if path_labels else 'N/A'
    path_lines = textwrap.wrap(path_text, width=34, break_long_words=False) or ['N/A']
    total_time = _result_total_duration(result)

    info_text = [
        f"Status  {'OK' if result.success else 'FAIL'}",
        f"Cost    {_compact_float(result.total_cost)}",
        f"Solve   {result.solve_time:.3f} s",
        f"Dur     {total_time:.3f} s",
        f"Paths   {result.n_paths_evaluated}",
        f"Regions {len(result.path_regions)}/{len(graph.regions)}",
        f"Defect  {result.defect_norm:.2e}",
        f"Gap     {result.max_connection_gap:.2e}",
        f"CtrlGap {result.max_control_jump:.2e}",
        f"Viol    {result.constraint_violation:.2e}",
    ]

    if result.max_integrality_gap > 0.0:
        info_text.append(f"IntGap  {result.max_integrality_gap:.2e}")

    _draw_text_panel(ax_info, "Optimization", info_text, _status_color(result.success))

    ax_safety = fig.add_subplot(grid[1, 2])
    if result.safety_mode == "log_barrier":
        safety_lines = [
            f"Mode    log_barrier",
            f"s_min   {_compact_float(result.min_safety_margin)}",
            f"s_cert  {_compact_float(result.certified_safety_margin)}",
            f"L_gap   {_compact_float(result.lipschitz_gap)}",
            f"Cert    {result.safety_certification}",
        ]
    else:
        safety_lines = [
            f"Mode    {result.safety_mode}",
            f"CTCS    {result.max_continuous_violation_integral:.2e}",
            f"Dense   {result.max_dense_region_violation:.2e}",
        ]
        if result.continuous_violation_integrals:
            top_ctcs = sorted(
                result.continuous_violation_integrals.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:4]
            safety_lines.append("")
            safety_lines.append("Top eta")
            safety_lines.extend([f"R{idx:<4} {value:.2e}" for idx, value in top_ctcs])
    _draw_text_panel(ax_safety, "Safety Diagnostics", safety_lines, '#6A1B9A')

    ax_path = fig.add_subplot(grid[2, 2])
    path_panel_lines = ["Path", *path_lines]
    if result.solver_status:
        path_panel_lines.extend([
            "",
            "Solver",
            *textwrap.wrap(result.solver_status, width=34, break_long_words=False),
        ])
    _draw_text_panel(ax_path, "Route", path_panel_lines, '#1565C0', fontsize=7.8)

    return fig


def create_velocity_profile_figure(
    result: OptimizationResult,
    figsize: Tuple[float, float] = (8, 3),
) -> plt.Figure:
    """Return a standalone figure showing the speed profile over time."""
    fig, ax = plt.subplots(figsize=figsize)

    if result.trajectories:
        times, velocities = _collect_profile_data(result)
        ax.plot(times, velocities, color='#1565C0', linewidth=1.5)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Speed [m/s]")
        ax.set_title("Speed Profile")
        ax.grid(True, color='#d7dce0', linewidth=0.5, alpha=0.7)
        ax.tick_params(labelsize=9)
    else:
        ax.text(0.5, 0.5, "No trajectory data", ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='#666666')
        ax.set_title("Speed Profile")

    fig.tight_layout()
    return fig


def create_animation(result: OptimizationResult, graph: RegionGraph,
                     workspace_bounds: Tuple[float, float, float, float],
                     start_pos: np.ndarray, goal_pos: np.ndarray,
                     filename: str,
                     obstacles: Optional[List[object]] = None,
                     fps: int = 30,
                     duration: float = 5.0,
                     show_mesh_points: bool = True) -> None:
    """
    Create trajectory animation as GIF.
    
    Args:
        result: OptimizationResult
        graph: RegionGraph
        workspace_bounds: Environment bounds
        start_pos, goal_pos: Start and goal positions
        obstacles: Obstacle polygons to overlay
        filename: Output filename
        fps: Frames per second
        duration: Animation duration in seconds
        show_mesh_points: Whether to display interior mesh points in the GIF
    """
    if not result.trajectories:
        print("Cannot create animation: no trajectory data available")
        return

    setup_plot_style()

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Left column  : trajectory map (spans all 3 rows)
    # Right column : [0] info panel  [1] linear-velocity chart  [2] angular-vel chart
    fig = plt.figure(figsize=(16.0, 9.5), constrained_layout=False)
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        width_ratios=[3.8, 1.35],
        height_ratios=[1.55, 1.0, 1.0],
        left=0.055, right=0.97,
        top=0.93, bottom=0.07,
        hspace=0.42, wspace=0.30,
    )
    ax = fig.add_subplot(gs[:, 0])       # trajectory map
    ax_info = fig.add_subplot(gs[0, 1])  # info/status panel
    ax_vlin = fig.add_subplot(gs[1, 1])  # linear velocity trace
    ax_vang = fig.add_subplot(gs[2, 1])  # angular velocity trace
    ax_info.axis('off')

    # ── Static map background ─────────────────────────────────────────────────
    plot_environment(ax, workspace_bounds, graph.regions,
                     start_pos, goal_pos,
                     obstacles=obstacles,
                     highlight_regions=result.path_regions,
                     show_labels=len(graph.regions) <= 60)

    # ── Pre-compute trajectory arrays ─────────────────────────────────────────
    all_points: List[np.ndarray] = []
    all_times_list: List[float] = []
    all_headings_list: List[float] = []
    cumulative_time = 0.0

    for traj, tau, delta in result.trajectories:
        for i in range(len(traj)):
            all_points.append(traj[i, :2])
            all_times_list.append(cumulative_time + tau[i] * delta)
            all_headings_list.append(float(traj[i, 2]) if traj.shape[1] > 2 else 0.0)
        cumulative_time += delta

    all_points_arr = np.array(all_points)
    all_times = np.array(all_times_list)
    all_headings = np.array(all_headings_list)
    mesh_points, mesh_times = _collect_mesh_points(result)

    # Velocity profiles for live charts
    vel_times, lin_vels, ang_vels = _compute_velocity_profiles(result)
    v_max = float(np.max(np.abs(lin_vels))) if len(lin_vels) > 0 else 1.0
    w_max = float(np.max(np.abs(ang_vels))) if len(ang_vels) > 0 else 1.0

    total_time = float(all_times[-1])
    n_frames = max(2, int(fps * duration))
    status_color = _status_color(result.success)
    status_text = 'OK' if result.success else 'FAIL'

    # ── Velocity subplot setup ─────────────────────────────────────────────────
    for _ax, _label, _color, _ymax, _unit in [
        (ax_vlin, 'Linear velocity  v(t)', '#1565C0', v_max, 'm/s'),
        (ax_vang, 'Angular velocity  ω(t)', '#C62828', w_max, 'rad/s'),
    ]:
        _ax.set_xlim(0, total_time)
        _ax.set_ylim(-_ymax * 1.15 - 0.05, _ymax * 1.15 + 0.05)
        _ax.set_xlabel('t [s]', fontsize=8)
        _ax.set_ylabel(_unit, fontsize=8)
        _ax.set_title(_label, fontsize=9, fontweight='bold', color=_color, pad=3)
        _ax.tick_params(labelsize=7.5)
        _ax.grid(True, color='#e0e0e0', linewidth=0.6)
        _ax.axhline(0, color='#aaaaaa', linewidth=0.7, linestyle='--')
        _ax.set_facecolor('#fafbfd')

    # Full (ghost) velocity traces
    if len(vel_times) > 0:
        ax_vlin.plot(vel_times, lin_vels, color='#90CAF9', linewidth=1.0, alpha=0.45)
        ax_vang.plot(vel_times, ang_vels, color='#EF9A9A', linewidth=1.0, alpha=0.45)

    # Live-trace lines (will be extended each frame)
    vlin_trace, = ax_vlin.plot([], [], color='#1565C0', linewidth=1.8, solid_capstyle='round')
    vang_trace, = ax_vang.plot([], [], color='#C62828', linewidth=1.8, solid_capstyle='round')
    # Current-time vertical cursors
    vlin_cursor = ax_vlin.axvline(0, color='#FFA000', linewidth=1.2, linestyle='-', alpha=0.85)
    vang_cursor = ax_vang.axvline(0, color='#FFA000', linewidth=1.2, linestyle='-', alpha=0.85)

    # ── Map animation elements ─────────────────────────────────────────────────
    full_line, = ax.plot(all_points_arr[:, 0], all_points_arr[:, 1],
                         color='#90CAF9', linewidth=2.0, alpha=0.8,
                         zorder=5, label='Planned path')
    robot_marker, = ax.plot([], [], 'o', markersize=11, zorder=12,
                            color='#0D47A1', markeredgecolor='white',
                            markeredgewidth=1.5)
    trail_line, = ax.plot([], [], color='#0D47A1', linewidth=3.0,
                          alpha=0.95, zorder=8, solid_capstyle='round',
                          label='Executed trail')
    mesh_scatter = ax.scatter([], [], c='#9C27B0', s=35, marker='o',
                              zorder=7, alpha=0.75, edgecolors='white',
                              linewidths=0.35, label='Reached mesh')
    heading_arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle='->', color='#0D47A1', lw=2.2))

    # ── Info panel ────────────────────────────────────────────────────────────
    ax_info.add_patch(patches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle='round,pad=0.02,rounding_size=0.02',
        transform=ax_info.transAxes,
        facecolor='white', edgecolor='#d7dce0', linewidth=1.0,
    ))
    status_badge = ax_info.text(
        0.50, 0.96, status_text,
        transform=ax_info.transAxes,
        fontsize=12, ha='center', va='top',
        color='white', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.35', facecolor=status_color,
                  edgecolor='none', alpha=0.96),
    )
    time_text = ax_info.text(
        0.08, 0.85, '', transform=ax_info.transAxes,
        fontsize=9.5, verticalalignment='top',
        fontfamily='monospace', color='#263238',
    )
    legend_text = ax_info.text(
        0.08, 0.44,
        "Legend\n"
        "blue   trail\n"
        "pale   planned\n"
        "dot    robot\n"
        "purple mesh\n"
        "star   goal",
        transform=ax_info.transAxes,
        fontsize=8.5, fontfamily='monospace',
        color='#37474F', va='top',
    )
    # NLP iteration count
    # n_nlp = getattr(result, 'n_nlp_iterations', 0)
    # nlp_label = f"NLP iters : {n_nlp}" if n_nlp > 0 else f"paths eval: {result.n_paths_evaluated}"
    # ax_info.text(
    #     0.08, 0.215, nlp_label,
    #     transform=ax_info.transAxes,
    #     fontsize=8.0, fontfamily='monospace',
    #     color='#5D4037',
    # )

    ax.set_title(
        f"GCS-MMS trajectory replay | total {total_time:.2f} s",
        color='#263238', fontweight='bold',
    )

    # Pre-compute static safety snippet (varies by safety_mode, fixed across frames).
    if result.safety_mode == "log_barrier":
        _safety_snippet = (
            f"s_min  {_compact_float(result.min_safety_margin)}\n"
            f"s_cert {_compact_float(result.certified_safety_margin)}\n"
            f"Defect {result.defect_norm:.2e}\n"
            f"Gap    {result.max_connection_gap:.2e}"
        )
    else:
        _safety_snippet = (
            f"CTCS   {result.max_continuous_violation_integral:.2e}\n"
            f"Dense  {result.max_dense_region_violation:.2e}\n"
            f"Defect {result.defect_norm:.2e}\n"
            f"Gap    {result.max_connection_gap:.2e}"
        )

    # ── Animation callbacks ────────────────────────────────────────────────────
    def init():
        robot_marker.set_data([], [])
        trail_line.set_data([], [])
        mesh_scatter.set_offsets(np.empty((0, 2)))
        heading_arrow.set_position((0, 0))
        heading_arrow.xy = (0, 0)
        time_text.set_text('')
        vlin_trace.set_data([], [])
        vang_trace.set_data([], [])
        vlin_cursor.set_xdata(np.full(2, 0.0))
        vang_cursor.set_xdata(np.full(2, 0.0))
        return (
            full_line, robot_marker, trail_line, mesh_scatter,
            heading_arrow, time_text, status_badge, legend_text,
            vlin_trace, vang_trace, vlin_cursor, vang_cursor,
        )

    def animate(frame):
        t_anim = frame / max(n_frames - 1, 1) * duration
        t_traj = t_anim / duration * total_time

        # ── Robot position ──
        idx = int(np.searchsorted(all_times, t_traj, side='right')) - 1
        idx = max(0, min(idx, len(all_times) - 2))
        t0, t1 = all_times[idx], all_times[idx + 1]
        alpha = np.clip((t_traj - t0) / (t1 - t0 + 1e-10), 0.0, 1.0)
        pos = (1 - alpha) * all_points_arr[idx] + alpha * all_points_arr[idx + 1]
        heading = (1 - alpha) * all_headings[idx] + alpha * all_headings[idx + 1]

        robot_marker.set_data([pos[0]], [pos[1]])
        trail_idx = idx + 1
        trail_positions = np.vstack([all_points_arr[:trail_idx], pos])
        trail_line.set_data(trail_positions[:, 0], trail_positions[:, 1])

        if show_mesh_points and len(mesh_points) > 0:
            visible_mesh = mesh_points[mesh_times <= t_traj + 1e-10]
            mesh_scatter.set_offsets(visible_mesh if len(visible_mesh) > 0 else np.empty((0, 2)))
        else:
            mesh_scatter.set_offsets(np.empty((0, 2)))

        arrow_len = 0.2
        heading_arrow.set_position((pos[0], pos[1]))
        heading_arrow.xy = (pos[0] + arrow_len * np.cos(heading),
                            pos[1] + arrow_len * np.sin(heading))

        # ── Velocity live traces ──
        if len(vel_times) > 0:
            mask = vel_times <= t_traj + 1e-10
            vlin_trace.set_data(vel_times[mask], lin_vels[mask])
            vang_trace.set_data(vel_times[mask], ang_vels[mask])
        vlin_cursor.set_xdata(np.full(2, t_traj))
        vang_cursor.set_xdata(np.full(2, t_traj))

        # ── Info panel ──
        time_text.set_text(
            f"mode   {result.safety_mode}\n"
            f"time   {t_traj:6.2f}/{total_time:.2f}s\n"
            f"cost   {_compact_float(result.total_cost)}\n"
            + _safety_snippet
        )

        return (
            full_line, robot_marker, trail_line, mesh_scatter,
            heading_arrow, time_text, status_badge, legend_text,
            vlin_trace, vang_trace, vlin_cursor, vang_cursor,
        )

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=n_frames, interval=1000 / fps, blit=False)

    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer)
    plt.close(fig)
    print(f"Animation saved to {filename}")


def save_result_summary(result: OptimizationResult, filename: str):
    """
    Save result summary to JSON file.
    
    Args:
        result: OptimizationResult
        filename: Output filename
    """
    import json
    
    summary = {
        'success': result.success,
        'solver_status': result.solver_status,
        'total_cost': float(result.total_cost),
        'solve_time': float(result.solve_time),
        'n_paths_evaluated': result.n_paths_evaluated,
        'pipeline_mode': result.pipeline_mode,
        'candidate_paths_ranked': result.candidate_paths_ranked,
        'n_paths_screened': result.n_paths_screened,
        'n_paths_polished': result.n_paths_polished,
        'early_stopped': result.early_stopped,
        'repair_attempted': result.repair_attempted,
        'repaired_regions': result.repaired_regions,
        'per_candidate_diagnostics': result.per_candidate_diagnostics,
        'stage_timings': result.stage_timings,
        'path': result.path,
        'path_regions': result.path_regions,
        'defect_norm': float(result.defect_norm),
        'max_connection_gap': float(result.max_connection_gap),
        'max_integrality_gap': float(result.max_integrality_gap),
        'constraint_violation': float(result.constraint_violation),
        'safety_mode': result.safety_mode,
        'continuous_violation_integrals': {
            str(k): float(v) for k, v in result.continuous_violation_integrals.items()
        },
        'max_ctcs_integral': float(result.max_continuous_violation_integral),
        'max_dense_region_violation': float(result.max_dense_region_violation),
        'safety_diagnostics': result.safety_diagnostics,
        'transition_diagnostics': result.transition_diagnostics,
        'failure_reasons': result.failure_reasons,
        'time_durations': {str(k): float(v) for k, v in result.time_durations.items()},
        'interface_points': [pt.tolist() for pt in result.interface_points],
    }
    
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to {filename}")
