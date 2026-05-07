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
    
    # Colormap for regions. Keep non-path regions quiet so the path reads first.
    cmap = plt.cm.get_cmap('tab20')
    highlight_set = set(highlight_regions or [])
    show_region_labels = bool(show_labels and len(regions) <= max_labeled_regions)
    
    # Plot regions
    for region in regions:
        color = cmap(region.index % 20)
        
        # Determine if highlighted
        if region.index in highlight_set:
            face_alpha = 0.45
            edge_width = 2.0
            edge_color = '#1565C0'
            zorder = 2
        else:
            face_alpha = alpha
            edge_width = 0.8
            edge_color = '#b8c0c8'
            zorder = 1
        
        poly = patches.Polygon(
            region.vertices,
            closed=True,
            facecolor=(*color[:3], face_alpha),
            edgecolor=edge_color,
            linewidth=edge_width,
            zorder=zorder,
        )
        ax.add_patch(poly)

        # Label
        if show_region_labels or region.index in highlight_set:
            centroid = region.get_centroid()
            ax.text(centroid[0], centroid[1], f'R{region.index}',
                   ha='center', va='center',
                   fontsize=8 if len(regions) <= max_labeled_regions else 7,
                   color='#0D47A1' if region.index in highlight_set else '#455A64',
                   fontweight='bold' if region.index in highlight_set else 'normal',
                   zorder=5)

    # Overlay obstacles explicitly so blocked space is easy to see.
    for obstacle_idx, obstacle in enumerate(obstacles or []):
        obstacle_patch = patches.Polygon(
            _polygon_vertices(obstacle),
            closed=True,
            facecolor='#404040',
            edgecolor='#ffffff',
            linewidth=1.0,
            alpha=0.95,
            zorder=4,
            label='Obstacle' if obstacle_idx == 0 else None,
        )
        ax.add_patch(obstacle_patch)

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
    ax.legend(loc='upper right', frameon=True, framealpha=0.92,
              facecolor='white', edgecolor='#dddddd')


def _collect_mesh_points(result: OptimizationResult,
                         fallback_points_per_segment: int = 5
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect interior mesh points and their physical times.

    Uses exact solver mesh samples when available, with a trajectory-based
    fallback for older results.
    """
    mesh_points = []
    mesh_times = []
    cumulative_time = 0.0

    if result.mesh_samples:
        for mesh_positions, mesh_tau, delta in result.mesh_samples:
            if len(mesh_positions) > 2:
                for idx in range(1, len(mesh_positions) - 1):
                    mesh_points.append(mesh_positions[idx])
                    mesh_times.append(cumulative_time + mesh_tau[idx] * delta)
            cumulative_time += delta
    else:
        for traj, tau, delta in result.trajectories:
            sample_count = min(fallback_points_per_segment, len(traj))
            if sample_count > 2:
                indices = np.linspace(0, len(traj) - 1, sample_count, dtype=int)
                indices = np.unique(indices)
                for idx in indices[1:-1]:
                    mesh_points.append(traj[idx, :2])
                    mesh_times.append(cumulative_time + tau[idx] * delta)
            cumulative_time += delta

    if not mesh_points:
        return np.empty((0, 2)), np.empty((0,))

    return np.array(mesh_points), np.array(mesh_times)


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
    
    # Mesh points (sample from trajectories)
    if show_mesh_points:
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
        node_id = f"R{region.index}"
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
    ax_main.legend(loc='upper right', frameon=True, framealpha=0.92,
                   facecolor='white', edgecolor='#dddddd')
    
    # Info panel
    ax_info = fig.add_subplot(grid[0, 2])

    path_labels = []
    for node_id in result.path:
        if node_id.startswith("R"):
            region_idx = int(node_id[1:])
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

    # Inset trajectory profile on top of the main map area.
    ax_profile = ax_main.inset_axes([0.04, 0.04, 0.38, 0.20])
    
    if result.trajectories:
        times, velocities = _collect_profile_data(result)
        ax_profile.plot(times, velocities, color='#1565C0', linewidth=1.5, label='speed')
        ax_profile.set_xlabel("Time [s]")
        ax_profile.set_ylabel("Speed")
        ax_profile.set_title("Speed profile", fontsize=9)
        ax_profile.grid(True, color='#d7dce0', linewidth=0.5, alpha=0.7)
        ax_profile.tick_params(labelsize=8)
        ax_profile.set_facecolor((1, 1, 1, 0.92))
    else:
        ax_profile.text(0.5, 0.5, "No trajectory data",
                       ha='center', va='center', transform=ax_profile.transAxes)
        ax_profile.set_title("Speed profile")
    
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
    
    fig, (ax, ax_info) = plt.subplots(
        1, 2,
        figsize=(12.5, 9.5),
        gridspec_kw={'width_ratios': [4.6, 1.25]},
        constrained_layout=True,
    )
    ax_info.axis('off')
    
    # Static background
    plot_environment(ax, workspace_bounds, graph.regions,
                     start_pos, goal_pos,
                     obstacles=obstacles,
                     highlight_regions=result.path_regions,
                     show_labels=len(graph.regions) <= 60)
    
    # Collect trajectory points with time
    all_points = []
    all_times = []
    all_headings = []
    cumulative_time = 0.0
    
    for traj, tau, delta in result.trajectories:
        for i in range(len(traj)):
            all_points.append(traj[i, :2])
            all_times.append(cumulative_time + tau[i] * delta)
            all_headings.append(traj[i, 2] if len(traj[i]) > 2 else 0)
        cumulative_time += delta
    
    all_points = np.array(all_points)
    all_times = np.array(all_times)
    all_headings = np.array(all_headings)
    mesh_points, mesh_times = _collect_mesh_points(result)
    
    # Total trajectory time
    total_time = all_times[-1]
    n_frames = max(2, int(fps * duration))
    status_color = _status_color(result.success)
    status_text = 'OK' if result.success else 'FAIL'
    
    # Animation elements
    full_line, = ax.plot(all_points[:, 0], all_points[:, 1],
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
    panel_bg = patches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle='round,pad=0.02,rounding_size=0.02',
        transform=ax_info.transAxes,
        facecolor='white',
        edgecolor='#d7dce0',
        linewidth=1.0,
    )
    ax_info.add_patch(panel_bg)
    status_badge = ax_info.text(
        0.50, 0.94,
        f"{status_text}",
        transform=ax_info.transAxes,
        fontsize=12,
        ha='center',
        va='top',
        color='white',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.35', facecolor=status_color,
                  edgecolor='none', alpha=0.96),
    )
    time_text = ax_info.text(
        0.08, 0.84, '', transform=ax_info.transAxes,
        fontsize=10.0, verticalalignment='top',
        fontfamily='monospace',
        color='#263238',
    )
    legend_text = ax_info.text(
        0.08, 0.43,
        "Legend\n"
        "blue  executed trail\n"
        "pale  planned path\n"
        "dot   robot\n"
        "purple reached mesh\n"
        "star  goal",
        transform=ax_info.transAxes,
        fontsize=9.0,
        fontfamily='monospace',
        color='#37474F',
        va='top',
    )
    progress_bg = patches.Rectangle(
        (0.08, 0.26), 0.84, 0.024,
        transform=ax_info.transAxes,
        facecolor='#E0E0E0',
        edgecolor='none',
    )
    progress_fg = patches.Rectangle(
        (0.08, 0.26), 0.0, 0.024,
        transform=ax_info.transAxes,
        facecolor='#1565C0',
        edgecolor='none',
    )
    ax_info.add_patch(progress_bg)
    ax_info.add_patch(progress_fg)
    ax_info.text(
        0.08, 0.30, "Progress",
        transform=ax_info.transAxes,
        fontsize=9,
        fontweight='bold',
        color='#263238',
    )
    ax.set_title(
        f"GCS-MMS trajectory replay | total {total_time:.2f} s",
        color='#263238',
        fontweight='bold',
    )
    
    def init():
        robot_marker.set_data([], [])
        trail_line.set_data([], [])
        mesh_scatter.set_offsets(np.empty((0, 2)))
        heading_arrow.set_position((0, 0))
        heading_arrow.xy = (0, 0)
        time_text.set_text('')
        progress_fg.set_width(0.0)
        return (
            full_line, robot_marker, trail_line, mesh_scatter,
            heading_arrow, time_text, status_badge, legend_text, progress_fg
        )
    
    def animate(frame):
        # Current animation time (mapped to trajectory time)
        t_anim = frame / (n_frames - 1) * duration
        t_traj = t_anim / duration * total_time
        
        # Find position at current time
        idx = np.searchsorted(all_times, t_traj, side='right') - 1
        idx = max(0, min(idx, len(all_times) - 2))
        
        # Interpolate
        t0, t1 = all_times[idx], all_times[idx + 1]
        alpha = (t_traj - t0) / (t1 - t0 + 1e-10)
        alpha = np.clip(alpha, 0, 1)
        
        pos = (1 - alpha) * all_points[idx] + alpha * all_points[idx + 1]
        heading = (1 - alpha) * all_headings[idx] + alpha * all_headings[idx + 1]
        
        # Update robot marker
        robot_marker.set_data([pos[0]], [pos[1]])
        
        # Update trail (all points up to current)
        trail_idx = idx + 1
        trail_positions = np.vstack([all_points[:trail_idx], pos])
        trail_line.set_data(trail_positions[:, 0], trail_positions[:, 1])

        # Show mesh points that have been reached so far
        if show_mesh_points and len(mesh_points) > 0:
            visible_mesh = mesh_points[mesh_times <= t_traj + 1e-10]
            mesh_scatter.set_offsets(visible_mesh if len(visible_mesh) > 0 else np.empty((0, 2)))
        else:
            mesh_scatter.set_offsets(np.empty((0, 2)))
        
        # Update heading arrow
        arrow_len = 0.2
        dx = arrow_len * np.cos(heading)
        dy = arrow_len * np.sin(heading)
        heading_arrow.set_position((pos[0], pos[1]))
        heading_arrow.xy = (pos[0] + dx, pos[1] + dy)
        
        # Update time text
        progress = t_traj / max(total_time, 1e-12)
        progress_fg.set_width(0.84 * np.clip(progress, 0.0, 1.0))
        time_text.set_text(
            f"mode   {result.safety_mode}\n"
            f"time   {t_traj:6.2f} / {total_time:.2f}s\n"
            f"cost   {_compact_float(result.total_cost)}\n"
            f"CTCS   {result.max_continuous_violation_integral:.2e}\n"
            f"Dense  {result.max_dense_region_violation:.2e}\n"
            f"Defect {result.defect_norm:.2e}\n"
            f"Gap    {result.max_connection_gap:.2e}"
        )
        
        return (
            full_line, robot_marker, trail_line, mesh_scatter,
            heading_arrow, time_text, status_badge, legend_text, progress_fg
        )
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=n_frames, interval=1000/fps, blit=False)
    
    # Save as GIF
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
