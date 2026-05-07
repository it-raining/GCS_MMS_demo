#!/usr/bin/env python3
"""
maze_benchmark.py - Benchmark the integrated solver on random maze instances.

The geometry pipeline intentionally mirrors scenario_builder.py:
1. Generate a random maze.
2. Convert remaining walls into obstacle polygons.
3. Build an Environment from fixed workspace_vertices + obstacle_vertices.
4. Call ACD on (workspace, holes=obstacles).
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from acd2d.acd2d import ACD2D
from app_config import DemoConfig
from convex_regions import create_buffered_regions_from_vertices_list
from dynamics import UnicycleModel
from environment import create_environment_from_vertices
from graph_builder import build_region_graph
from models.maze import Maze
from optimizer import create_integrated_optimizer_from_config
from problem_data import PROBLEM_PRESETS
from visualization import create_animation, create_result_figure, save_result_summary


# Keep the random-maze pipeline aligned with scenario_builder.py.
SCENARIO_BUILDER_ACD_TAU = 0.0
# The regular scenario path uses 1e-3; using 1e-4 here creates extremely thin
# overlaps in random mazes and makes unicycle transitions through ACD slivers
# unnecessarily tight.
SCENARIO_BUILDER_REGION_BUFFER = 1e-3


@dataclass
class MazeBenchmarkCaseResult:
    """Compact benchmark summary for one generated maze instance."""

    seed: int
    maze_size: int
    knock_downs: int
    wall_thickness: float
    n_wall_boxes: int
    n_regions: int
    n_edges: int
    success: bool
    setup_time: float
    solve_time: float
    total_cost: float | None
    path_length: int
    defect_norm: float
    constraint_violation: float
    max_connection_gap: float
    max_integrality_gap: float
    solver_status: str
    pipeline_mode: str = "integrated"
    n_paths_screened: int = 0
    n_paths_polished: int = 0
    early_stopped: bool = False
    repair_attempted: bool = False
    max_ctcs_integral: float = 0.0
    max_dense_region_violation: float = 0.0
    failure_reasons: List[str] | None = None
    geometry_debug_path: str | None = None
    geometry_classification_debug_path: str | None = None
    free_space_coordinates_path: str | None = None
    obstacle_coordinates_path: str | None = None
    region_debug_path: str | None = None
    result_summary_path: str | None = None
    result_figure_path: str | None = None
    result_animation_path: str | None = None


def _generate_maze(seed_value: int, maze_size: int, knock_downs: int) -> Maze:
    """Generate one maze instance matching the user's reference snippet."""
    if maze_size < 3:
        raise ValueError("maze_size must be at least 3.")
    if knock_downs < 0:
        raise ValueError("knock_downs must be non-negative.")

    random.seed(seed_value)
    maze = Maze(maze_size, maze_size)
    maze.make_maze()

    remaining = knock_downs
    while remaining > 0:
        cell = maze.cell_at(
            random.randint(1, maze_size - 2),
            random.randint(1, maze_size - 2),
        )
        walls = [wall for wall, up in cell.walls.items() if up]
        if not walls:
            continue
        maze.knock_down_wall(cell, random.choice(walls))
        remaining -= 1

    return maze


def _maze_wall_boxes(maze: Maze, wall_thickness: float) -> List[Polygon]:
    """Convert maze walls into rectangular obstacle strips."""
    if not 0.0 < wall_thickness < 1.0:
        raise ValueError("wall_thickness must lie in (0, 1).")

    half_thickness = 0.5 * wall_thickness
    workspace_box = box(0.0, 0.0, float(maze.nx), float(maze.ny))
    wall_boxes: List[Polygon] = [
        # Treat the maze frame as true obstacle geometry instead of folding it
        # into the workspace boundary. These strips stay fully inside the
        # workspace so Environment.free_space captures the interior maze domain.
        box(0.0, 0.0, wall_thickness, float(maze.ny)),
        box(float(maze.nx) - wall_thickness, 0.0, float(maze.nx), float(maze.ny)),
        box(0.0, 0.0, float(maze.nx), wall_thickness),
        box(0.0, float(maze.ny) - wall_thickness, float(maze.nx), float(maze.ny)),
    ]

    for x in range(maze.nx):
        for y in range(maze.ny):
            cell = maze.cell_at(x, y)

            if x < maze.nx - 1 and cell.walls["E"]:
                # Extend by half-thickness along the wall direction so right-angle
                # joints are watertight instead of using butt-ended strips.
                vertical_wall = box(
                    x + 1 - half_thickness,
                    y - half_thickness,
                    x + 1 + half_thickness,
                    y + 1 + half_thickness,
                ).intersection(workspace_box)
                if not vertical_wall.is_empty:
                    wall_boxes.append(vertical_wall)
            if y < maze.ny - 1 and cell.walls["N"]:
                horizontal_wall = box(
                    x - half_thickness,
                    y + 1 - half_thickness,
                    x + 1 + half_thickness,
                    y + 1 + half_thickness,
                ).intersection(workspace_box)
                if not horizontal_wall.is_empty:
                    wall_boxes.append(horizontal_wall)

    return wall_boxes


def _iter_polygon_components(geometry) -> Iterable[Polygon]:
    """Yield polygon components from a shapely geometry."""
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [geom for geom in geometry.geoms if geom.geom_type == "Polygon"]
    return []


def _simplify_ring(vertices: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    """Remove duplicate and collinear vertices from a polygon ring."""
    if len(vertices) < 3:
        return vertices

    cleaned: List[np.ndarray] = []
    for vertex in vertices:
        if not cleaned or np.linalg.norm(vertex - cleaned[-1]) > tolerance:
            cleaned.append(vertex.astype(np.float64, copy=False))

    if len(cleaned) >= 2 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= tolerance:
        cleaned.pop()

    if len(cleaned) < 3:
        return np.asarray(cleaned, dtype=np.float64)

    simplified: List[np.ndarray] = []
    total = len(cleaned)
    for idx in range(total):
        prev_vertex = cleaned[idx - 1]
        curr_vertex = cleaned[idx]
        next_vertex = cleaned[(idx + 1) % total]

        edge_prev = curr_vertex - prev_vertex
        edge_next = next_vertex - curr_vertex
        cross = edge_prev[0] * edge_next[1] - edge_prev[1] * edge_next[0]

        if np.linalg.norm(edge_prev) <= tolerance or np.linalg.norm(edge_next) <= tolerance:
            continue
        if abs(cross) <= tolerance:
            continue
        simplified.append(curr_vertex)

    return np.asarray(simplified, dtype=np.float64)


def _round_coordinate(value: float, ndigits: int = 6) -> float:
    """Round debug coordinates and normalize negative zero."""
    rounded = round(float(value), ndigits)
    return 0.0 if abs(rounded) < 10 ** (-ndigits) else rounded


def _vertices_to_list(vertices: np.ndarray, ndigits: int = 6) -> List[List[float]]:
    """Convert vertices to a stable list-of-lists representation."""
    return [
        [_round_coordinate(vertex[0], ndigits), _round_coordinate(vertex[1], ndigits)]
        for vertex in np.asarray(vertices, dtype=np.float64)
    ]


def _raw_wall_box_vertices(wall_boxes: List[Polygon]) -> List[List[List[float]]]:
    """Serialize individual wall boxes as obstacle vertex lists."""
    obstacle_vertices: List[List[List[float]]] = []
    for wall_box in wall_boxes:
        vertices = _simplify_ring(
            np.asarray(wall_box.exterior.coords[:-1], dtype=np.float64)
        )
        if len(vertices) < 3:
            continue
        obstacle_vertices.append(_vertices_to_list(vertices))
    return obstacle_vertices


def _merged_wall_vertices(wall_boxes: List[Polygon]) -> tuple[List[List[List[float]]], bool]:
    """
    Merge touching wall strips into larger obstacle polygons.

    Returns:
        obstacle_vertices: merged polygon vertex lists
        has_interior_holes: whether any merged polygon contains interior rings
    """
    merged = unary_union(wall_boxes) if wall_boxes else None
    components = sorted(
        _iter_polygon_components(merged),
        key=lambda poly: (-poly.area, poly.bounds),
    )

    obstacle_vertices: List[List[List[float]]] = []
    has_interior_holes = False
    for component in components:
        if component.area <= 1e-9:
            continue
        if len(component.interiors) > 0:
            has_interior_holes = True
        vertices = _simplify_ring(
            np.asarray(component.exterior.coords[:-1], dtype=np.float64)
        )
        if len(vertices) < 3:
            continue
        obstacle_vertices.append(_vertices_to_list(vertices))

    return obstacle_vertices, has_interior_holes


def _save_obstacle_coordinates(
    workspace_vertices: List[List[float]],
    obstacle_vertices: List[List[List[float]]],
    output_path: Path,
) -> None:
    """Persist a pasteable geometry snippet for debugging."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '"""Debug obstacle coordinates for a generated maze benchmark case."""',
        "",
        "# This file can be reused with create_environment_from_vertices(...).",
        "",
        f"workspace_vertices = {json.dumps(workspace_vertices, indent=4)}",
        "",
        f"obstacle_vertices = {json.dumps(obstacle_vertices, indent=4)}",
        "",
    ]
    with output_path.open("w") as handle:
        handle.write("\n".join(lines))


def _save_environment_debug_plot(
    workspace_vertices: List[List[float]],
    obstacle_vertices: List[List[List[float]]],
    output_path: Path,
    start_pos: Optional[np.ndarray] = None,
    goal_pos: Optional[np.ndarray] = None,
) -> None:
    """Plot workspace and obstacle polygons exactly as passed to ACD."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workspace = np.asarray(workspace_vertices, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 8))

    workspace_patch = patches.Polygon(
        workspace,
        closed=True,
        facecolor="#f5f5f5",
        edgecolor="black",
        linewidth=2.0,
    )
    ax.add_patch(workspace_patch)

    for vertices in obstacle_vertices:
        polygon = np.asarray(vertices, dtype=np.float64)
        if len(polygon) < 3:
            continue
        obstacle_patch = patches.Polygon(
            polygon,
            closed=True,
            facecolor="#404040",
            edgecolor="#111111",
            linewidth=1.0,
            alpha=0.9,
            hatch="///",
        )
        ax.add_patch(obstacle_patch)

    if start_pos is not None:
        ax.plot(start_pos[0], start_pos[1], "go", markersize=10, markeredgecolor="darkgreen")
    if goal_pos is not None:
        ax.plot(goal_pos[0], goal_pos[1], "r^", markersize=10, markeredgecolor="darkred")

    x_min = min(vertex[0] for vertex in workspace_vertices)
    x_max = max(vertex[0] for vertex in workspace_vertices)
    y_min = min(vertex[1] for vertex in workspace_vertices)
    y_max = max(vertex[1] for vertex in workspace_vertices)
    pad = 0.25

    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Workspace + Obstacle Coordinates Passed to ACD\n{output_path.stem}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_region_debug_plot(
    workspace_vertices: List[List[float]],
    obstacle_vertices: List[List[List[float]]],
    regions,
    output_path: Path,
    start_pos: Optional[np.ndarray] = None,
    goal_pos: Optional[np.ndarray] = None,
) -> None:
    """Plot decomposed convex regions before graph construction and solving."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workspace = np.asarray(workspace_vertices, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9, 9))

    workspace_patch = patches.Polygon(
        workspace,
        closed=True,
        facecolor="#f7f7f7",
        edgecolor="black",
        linewidth=2.0,
    )
    ax.add_patch(workspace_patch)

    cmap = plt.cm.get_cmap("tab20")
    for region in regions:
        color = cmap(region.index % 20)
        region_patch = patches.Polygon(
            np.asarray(region.vertices, dtype=np.float64),
            closed=True,
            facecolor=(color[0], color[1], color[2], 0.28),
            edgecolor=(color[0], color[1], color[2], 0.95),
            linewidth=1.0,
        )
        ax.add_patch(region_patch)

        centroid = region.get_centroid()
        ax.text(
            centroid[0],
            centroid[1],
            f"R{region.index}",
            ha="center",
            va="center",
            fontsize=6,
            color="#222222",
        )

    for vertices in obstacle_vertices:
        polygon = np.asarray(vertices, dtype=np.float64)
        if len(polygon) < 3:
            continue
        obstacle_patch = patches.Polygon(
            polygon,
            closed=True,
            facecolor="#2f2f2f",
            edgecolor="#0f0f0f",
            linewidth=1.0,
            alpha=0.9,
            hatch="///",
        )
        ax.add_patch(obstacle_patch)

    if start_pos is not None:
        ax.plot(start_pos[0], start_pos[1], "go", markersize=10, markeredgecolor="darkgreen")
    if goal_pos is not None:
        ax.plot(goal_pos[0], goal_pos[1], "r^", markersize=10, markeredgecolor="darkred")

    x_min = min(vertex[0] for vertex in workspace_vertices)
    x_max = max(vertex[0] for vertex in workspace_vertices)
    y_min = min(vertex[1] for vertex in workspace_vertices)
    y_max = max(vertex[1] for vertex in workspace_vertices)
    pad = 0.25

    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Convex Regions Before Solving\n{output_path.stem}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_environment_and_regions(
    acd: ACD2D,
    workspace_vertices: List[List[float]],
    obstacle_vertices: List[List[List[float]]],
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    buffer_size: float = 1e-3,
):
    """Build an Environment and decompose it exactly like scenario_builder.py."""
    environment = create_environment_from_vertices(
        workspace_vertices,
        obstacle_vertices,
    )

    free_space_components = list(_iter_polygon_components(environment.free_space))
    if not free_space_components:
        raise RuntimeError("Maze free space is empty after applying obstacle geometry.")

    start_point = Point(float(start_pos[0]), float(start_pos[1]))
    goal_point = Point(float(goal_pos[0]), float(goal_pos[1]))

    selected_components = [
        component for component in free_space_components
        if component.covers(start_point) or component.covers(goal_point)
    ]
    if not selected_components:
        raise RuntimeError(
            "Neither start nor goal lies inside any free-space component. "
            "Boundary-wall obstacle generation likely blocks the configured states."
        )

    shared_components = [
        component for component in selected_components
        if component.covers(start_point) and component.covers(goal_point)
    ]
    if not shared_components:
        raise RuntimeError(
            "Start and goal lie in different free-space components after adding "
            "independent boundary-wall obstacles."
        )
    selected_components = shared_components

    region_vertices: List[np.ndarray] = []
    decomposition_errors: List[str] = []
    for component in selected_components:
        component_workspace = np.asarray(component.exterior.coords[:-1], dtype=np.float64)
        component_holes = [
            np.asarray(interior.coords[:-1], dtype=np.float64)
            for interior in component.interiors
        ]

        acd_output = io.StringIO()
        try:
            with contextlib.redirect_stdout(acd_output):
                component_regions, _result = acd.decompose_to_polygons(
                    component_workspace,
                    holes=component_holes,
                )
            region_vertices.extend(component_regions)
        except Exception as exc:
            acd_logs = acd_output.getvalue().strip()
            detail = str(exc) if not acd_logs else f"{exc}\n{acd_logs}"
            decomposition_errors.append(detail)

    if not region_vertices:
        if decomposition_errors:
            raise RuntimeError(
                "ACD2D failed on all selected free-space components:\n" +
                "\n\n---\n\n".join(decomposition_errors)
            )
        raise RuntimeError("ACD decomposition did not produce usable polygons.")

    regions = create_buffered_regions_from_vertices_list(
        region_vertices,
        workspace_vertices,
        buffer_size=buffer_size,
    )
    return environment, regions


class MazeBenchmarkRunner:
    """Generate random mazes and benchmark the integrated solver on them."""

    def __init__(self, config_path: str = "config.yaml", template_scenario: str | None = None):
        self.config = DemoConfig(config_path)
        if template_scenario is None and "maze" in self.config.list_scenarios():
            template_scenario = "maze"
        self.template_scenario = template_scenario or self.config.default_scenario_name()
        self.resolved = self.config.resolve_scenario(self.template_scenario)
        self.template_preset = PROBLEM_PRESETS[self.resolved.problem_preset]
        self.template_workspace_bounds = self._workspace_bounds(
            self.template_preset.workspace_vertices
        )
        self.output_dir = Path(self.config.output_dir) / "maze_benchmarks"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_geometry_dir = self.output_dir / "geometry_debug"
        self.result_artifact_dir = self.output_dir / "case_results"
        self.result_artifact_dir.mkdir(parents=True, exist_ok=True)

        self.acd = ACD2D()
        self.acd.set_parameters(tau=SCENARIO_BUILDER_ACD_TAU)

    @staticmethod
    def _workspace_bounds(vertices: List[List[float]]) -> tuple[float, float, float, float]:
        """Axis-aligned bounds for a workspace polygon."""
        xs = [float(vertex[0]) for vertex in vertices]
        ys = [float(vertex[1]) for vertex in vertices]
        return min(xs), min(ys), max(xs), max(ys)

    def _scale_position_from_template(
        self,
        position: np.ndarray,
        maze_size: int,
    ) -> np.ndarray:
        """
        Map a template-scenario position into the random maze workspace.

        The benchmark workspace is always [0, maze_size]^2, while the template
        scenario may come from a different workspace scale such as the 5x5
        preset used by `python main_demo.py --scenario maze`.
        """
        x_min, y_min, x_max, y_max = self.template_workspace_bounds
        width = max(x_max - x_min, 1e-9)
        height = max(y_max - y_min, 1e-9)

        frac_x = np.clip((float(position[0]) - x_min) / width, 0.0, 1.0)
        frac_y = np.clip((float(position[1]) - y_min) / height, 0.0, 1.0)
        return np.array(
            [frac_x * float(maze_size), frac_y * float(maze_size)],
            dtype=np.float64,
        )

    def _build_states(self, maze_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Scale the template scenario's boundary conditions to the maze workspace."""
        start_pos = self._scale_position_from_template(
            self.resolved.start_state[:2],
            maze_size,
        )
        goal_pos = self._scale_position_from_template(
            self.resolved.goal_state[:2],
            maze_size,
        )
        start_state = np.array(
            [start_pos[0], start_pos[1], float(self.resolved.start_state[2])],
            dtype=np.float64,
        )
        goal_state = np.array(
            [goal_pos[0], goal_pos[1], float(self.resolved.goal_state[2])],
            dtype=np.float64,
        )
        return start_state, goal_state

    def _prepare_obstacle_vertices(
        self,
        wall_boxes: List[Polygon],
        verbose: bool,
    ) -> tuple[List[List[List[float]]], str]:
        """
        Prepare obstacle coordinates from maze walls.

        Prefer merged wall components because they are closer to the hand-written
        obstacle format in problem_data.py. Fall back to raw wall boxes only if
        merged components contain interior holes.
        """
        merged_vertices, has_interior_holes = _merged_wall_vertices(wall_boxes)
        if merged_vertices and not has_interior_holes:
            return merged_vertices, "merged-wall-components"

        if verbose and has_interior_holes:
            print("[maze] Merged wall polygons contain holes; falling back to raw wall boxes.")
        return _raw_wall_box_vertices(wall_boxes), "raw-wall-boxes"

    def _case_artifact_prefix(
        self,
        seed_value: int,
        maze_size: int,
        knock_downs: int,
    ) -> Path:
        """Stable filename prefix for one benchmark case."""
        return (
            self.result_artifact_dir /
            f"maze_case_size{maze_size}_kd{knock_downs}_seed{seed_value}"
        )

    def _save_result_artifacts(
        self,
        result,
        graph,
        environment,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        seed_value: int,
        maze_size: int,
        knock_downs: int,
        verbose: bool,
    ) -> tuple[str | None, str | None, str | None]:
        """
        Persist benchmark result artifacts without affecting solver success.

        Returns:
            (summary_json_path, figure_png_path, animation_gif_path)
        """
        prefix = self._case_artifact_prefix(seed_value, maze_size, knock_downs)
        summary_path = f"{prefix}_summary.json" if self.config.save_json else None
        figure_path = f"{prefix}_result.png" if self.config.save_png else None
        animation_path = f"{prefix}_animation.gif" if self.config.save_gif else None

        workspace_bounds = environment.get_bounds()
        if summary_path is not None:
            save_result_summary(result, summary_path)

        if figure_path is not None:
            fig = create_result_figure(
                result,
                graph,
                workspace_bounds,
                start_state[:2],
                goal_state[:2],
                obstacles=environment.obstacles,
                title=(
                    f"Maze Benchmark: size={maze_size}, kd={knock_downs}, "
                    f"seed={seed_value}"
                ),
            )
            fig.savefig(figure_path)
            plt.close(fig)

        if animation_path is not None and result.trajectories:
            create_animation(
                result,
                graph,
                workspace_bounds,
                start_state[:2],
                goal_state[:2],
                animation_path,
                obstacles=environment.obstacles,
                fps=self.resolved.runtime_config.get("visualization", {})
                .get("animation", {})
                .get("fps", 30),
                duration=self.resolved.runtime_config.get("visualization", {})
                .get("animation", {})
                .get("duration", 5.0),
            )
        elif animation_path is not None:
            animation_path = None

        if summary_path is not None:
            print(f"[maze seed={seed_value}] Saved result summary: {summary_path}")
        if figure_path is not None:
            print(f"[maze seed={seed_value}] Saved result figure: {figure_path}")
        if animation_path is not None:
            print(f"[maze seed={seed_value}] Saved result animation: {animation_path}")

        return summary_path, figure_path, animation_path

    def run_case(
        self,
        seed_value: int,
        maze_size: int,
        knock_downs: int,
        wall_thickness: float,
        save_debug_geometry: bool = False,
        verbose: bool = True,
    ) -> MazeBenchmarkCaseResult:
        """Run one generated maze benchmark case."""
        setup_start = time.time()

        workspace_vertices = [
            [0.0, 0.0],
            [float(maze_size), 0.0],
            [float(maze_size), float(maze_size)],
            [0.0, float(maze_size)],
        ]
        maze = _generate_maze(seed_value, maze_size, knock_downs)
        wall_boxes = _maze_wall_boxes(maze, wall_thickness)
        obstacle_vertices, obstacle_mode = self._prepare_obstacle_vertices(wall_boxes, verbose)
        start_state, goal_state = self._build_states(maze_size)

        geometry_debug_path = (
            self.debug_geometry_dir /
            f"maze_geometry_size{maze_size}_kd{knock_downs}_seed{seed_value}.png"
            if save_debug_geometry else None
        )
        obstacle_coords_path = (
            self.debug_geometry_dir /
            f"maze_obstacle_coords_size{maze_size}_kd{knock_downs}_seed{seed_value}.py"
            if save_debug_geometry else None
        )
        region_debug_path = (
            self.debug_geometry_dir /
            f"maze_regions_size{maze_size}_kd{knock_downs}_seed{seed_value}.png"
            if save_debug_geometry else None
        )

        try:
            if verbose:
                print(
                    f"[maze seed={seed_value}] Generated maze with {len(wall_boxes)} wall boxes, "
                    f"{len(obstacle_vertices)} obstacle polygons ({obstacle_mode})."
                )

            if obstacle_coords_path is not None:
                _save_obstacle_coordinates(
                    workspace_vertices,
                    obstacle_vertices,
                    obstacle_coords_path,
                )
            if geometry_debug_path is not None:
                _save_environment_debug_plot(
                    workspace_vertices,
                    obstacle_vertices,
                    geometry_debug_path,
                    start_pos=start_state[:2],
                    goal_pos=goal_state[:2],
                )

            environment, regions = _build_environment_and_regions(
                self.acd,
                workspace_vertices,
                obstacle_vertices,
                start_state[:2],
                goal_state[:2],
                buffer_size=SCENARIO_BUILDER_REGION_BUFFER,
            )
            if region_debug_path is not None:
                _save_region_debug_plot(
                    workspace_vertices,
                    obstacle_vertices,
                    regions,
                    region_debug_path,
                    start_pos=start_state[:2],
                    goal_pos=goal_state[:2],
                )
            print(
                f"[maze seed={seed_value}] Built environment with "
                f"{len(environment.obstacles)} obstacles and {len(regions)} regions"
            )

            graph = build_region_graph(regions, start_state[:2], goal_state[:2])

            dyn_cfg = self.resolved.runtime_config.get("dynamics", {})
            dynamics = UnicycleModel(
                v_min=dyn_cfg.get("v_min", -2.0),
                v_max=dyn_cfg.get("v_max", 2.0),
                omega_min=dyn_cfg.get("omega_min", -np.pi),
                omega_max=dyn_cfg.get("omega_max", np.pi),
            )
            optimizer = create_integrated_optimizer_from_config(
                graph,
                dynamics,
                self.resolved.runtime_config,
            )

            setup_time = time.time() - setup_start
            result = optimizer.solve(start_state, goal_state, verbose=verbose)
        except Exception as exc:
            return MazeBenchmarkCaseResult(
                seed=seed_value,
                maze_size=maze_size,
                knock_downs=knock_downs,
                wall_thickness=wall_thickness,
                n_wall_boxes=len(wall_boxes),
                n_regions=0,
                n_edges=0,
                success=False,
                setup_time=time.time() - setup_start,
                solve_time=0.0,
                total_cost=None,
                path_length=0,
                defect_norm=0.0,
                constraint_violation=0.0,
                max_connection_gap=0.0,
                max_integrality_gap=0.0,
                max_ctcs_integral=0.0,
                max_dense_region_violation=0.0,
                failure_reasons=[f"Maze benchmark setup failed: {exc}"],
                solver_status=f"Maze benchmark setup failed: {exc}",
                geometry_debug_path=str(geometry_debug_path) if geometry_debug_path is not None else None,
                geometry_classification_debug_path=None,
                free_space_coordinates_path=(
                    str(obstacle_coords_path) if obstacle_coords_path is not None else None
                ),
                obstacle_coordinates_path=(
                    str(obstacle_coords_path) if obstacle_coords_path is not None else None
                ),
                region_debug_path=str(region_debug_path) if region_debug_path is not None else None,
                result_summary_path=None,
                result_figure_path=None,
                result_animation_path=None,
            )

        result_summary_path = None
        result_figure_path = None
        result_animation_path = None
        try:
            (
                result_summary_path,
                result_figure_path,
                result_animation_path,
            ) = self._save_result_artifacts(
                result,
                graph,
                environment,
                start_state,
                goal_state,
                seed_value,
                maze_size,
                knock_downs,
                verbose,
            )
        except Exception as exc:
            if verbose:
                print(f"[maze seed={seed_value}] Warning: could not save result artifacts: {exc}")

        if verbose:
            status = "OK" if result.success else "FAIL"
            print(
                f"[maze seed={seed_value}] {status} | "
                f"regions={len(regions)} edges={graph.graph.number_of_edges()} "
                f"solve={result.solve_time:.3f}s"
            )
            if geometry_debug_path is not None:
                print(f"[maze seed={seed_value}] Saved geometry debug: {geometry_debug_path}")
            if obstacle_coords_path is not None:
                print(f"[maze seed={seed_value}] Saved obstacle coordinates: {obstacle_coords_path}")
            if region_debug_path is not None:
                print(f"[maze seed={seed_value}] Saved region debug: {region_debug_path}")

        total_cost = float(result.total_cost) if math.isfinite(result.total_cost) else None
        return MazeBenchmarkCaseResult(
            seed=seed_value,
            maze_size=maze_size,
            knock_downs=knock_downs,
            wall_thickness=wall_thickness,
            n_wall_boxes=len(wall_boxes),
            n_regions=len(regions),
            n_edges=graph.graph.number_of_edges(),
            success=result.success,
            setup_time=setup_time,
            solve_time=result.solve_time,
            total_cost=total_cost,
            path_length=len(result.path_regions),
            defect_norm=result.defect_norm,
            constraint_violation=result.constraint_violation,
            max_connection_gap=result.max_connection_gap,
            max_integrality_gap=result.max_integrality_gap,
            pipeline_mode=result.pipeline_mode,
            n_paths_screened=result.n_paths_screened,
            n_paths_polished=result.n_paths_polished,
            early_stopped=result.early_stopped,
            repair_attempted=result.repair_attempted,
            max_ctcs_integral=result.max_continuous_violation_integral,
            max_dense_region_violation=result.max_dense_region_violation,
            failure_reasons=result.failure_reasons,
            solver_status=result.solver_status,
            geometry_debug_path=str(geometry_debug_path) if geometry_debug_path is not None else None,
            geometry_classification_debug_path=None,
            free_space_coordinates_path=(
                str(obstacle_coords_path) if obstacle_coords_path is not None else None
            ),
            obstacle_coordinates_path=(
                str(obstacle_coords_path) if obstacle_coords_path is not None else None
            ),
            region_debug_path=str(region_debug_path) if region_debug_path is not None else None,
            result_summary_path=result_summary_path,
            result_figure_path=result_figure_path,
            result_animation_path=result_animation_path,
        )

    @staticmethod
    def _aggregate(cases: List[MazeBenchmarkCaseResult]) -> Dict[str, float | int]:
        """Aggregate benchmark metrics across all maze instances."""
        successful = [case for case in cases if case.success]
        successful_costs = [case.total_cost for case in successful if case.total_cost is not None]
        return {
            "count": len(cases),
            "successes": len(successful),
            "success_rate": (len(successful) / len(cases)) if cases else 0.0,
            "avg_regions": float(np.mean([case.n_regions for case in cases])) if cases else 0.0,
            "avg_edges": float(np.mean([case.n_edges for case in cases])) if cases else 0.0,
            "avg_setup_time": float(np.mean([case.setup_time for case in cases])) if cases else 0.0,
            "avg_solve_time": float(np.mean([case.solve_time for case in cases])) if cases else 0.0,
            "avg_success_cost": float(np.mean(successful_costs)) if successful_costs else 0.0,
        }

    def _save_summary(
        self,
        cases: List[MazeBenchmarkCaseResult],
        maze_size: int,
        knock_downs: int,
        seed_start: int,
        wall_thickness: float,
    ) -> Path:
        """Persist benchmark summary as JSON."""
        filename = (
            f"maze_benchmark_size{maze_size}_kd{knock_downs}"
            f"_count{len(cases)}_seed{seed_start}.json"
        )
        output_path = self.output_dir / filename
        payload = {
            "template_scenario": self.template_scenario,
            "maze_size": maze_size,
            "knock_downs": knock_downs,
            "seed_start": seed_start,
            "wall_thickness": wall_thickness,
            "aggregate": self._aggregate(cases),
            "cases": [asdict(case) for case in cases],
        }
        with output_path.open("w") as handle:
            json.dump(payload, handle, indent=2)
        return output_path

    def run(
        self,
        count: int,
        maze_size: int,
        knock_downs: int,
        seed_start: int,
        wall_thickness: float,
        save_debug_geometry: bool = False,
        verbose: bool = True,
    ) -> Path:
        """Run the maze benchmark over a consecutive block of seeds."""
        if count <= 0:
            raise ValueError("count must be positive.")

        cases = [
            self.run_case(
                seed_value=seed_start + offset,
                maze_size=maze_size,
                knock_downs=knock_downs,
                wall_thickness=wall_thickness,
                save_debug_geometry=save_debug_geometry,
                verbose=verbose,
            )
            for offset in range(count)
        ]

        summary_path = self._save_summary(
            cases=cases,
            maze_size=maze_size,
            knock_downs=knock_downs,
            seed_start=seed_start,
            wall_thickness=wall_thickness,
        )

        if verbose:
            aggregate = self._aggregate(cases)
            print(
                f"[maze benchmark] Saved summary: {summary_path} | "
                f"success_rate={aggregate['success_rate']:.2%} "
                f"avg_regions={aggregate['avg_regions']:.1f}"
            )

        return summary_path
