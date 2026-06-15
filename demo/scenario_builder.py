"""
scenario_builder.py - Build a runnable planning scenario from config + preset.
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app_config import DemoConfig, ResolvedScenario
from convex_regions import ConvexRegion, create_buffered_regions_from_vertices_list, create_regions_from_vertices_list
from dynamics import UnicycleModel
from environment import Environment, create_environment_from_vertices
from graph_builder import RegionGraph, build_region_graph
from graph_types import region_node_label
from optimizer import IntegratedMIOCPSolver, create_integrated_optimizer_from_config
from problem_data import PROBLEM_PRESETS, ProblemPresetSpec

# Decompose
from acd2d.acd2d import ACD2D

# Setup
acd = ACD2D()
acd.set_parameters(tau=0.0)

@dataclass
class PreparedScenario:
    """Fully built runtime objects for a scenario."""
    resolved: ResolvedScenario
    preset: ProblemPresetSpec
    environment: Environment
    regions: List[ConvexRegion]
    graph: RegionGraph
    dynamics: UnicycleModel
    optimizer: IntegratedMIOCPSolver
    workspace_bounds: tuple[float, float, float, float]


def list_problem_presets() -> List[str]:
    """Available geometry presets."""
    return list(PROBLEM_PRESETS.keys())


def get_problem_preset(name: str) -> ProblemPresetSpec:
    """Look up a geometry preset by key."""
    try:
        return PROBLEM_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(list_problem_presets())
        raise ValueError(f"Unknown problem preset '{name}'. Available: {available}") from exc


def build_environment_and_regions(
    preset: ProblemPresetSpec,
    overlap_width: float = 0.5,
) -> tuple[Environment, List[ConvexRegion], List[ConvexRegion]]:
    """Construct environment and convex regions from a preset.

    overlap_width: buffer size for creating region overlap (spec Part 0.2:
    overlap_width >= delta_safe + delta_extra).  Obstacles are subtracted
    from the buffered polygons so the H-rep stays in free space.
    """
    environment = create_environment_from_vertices(
        preset.workspace_vertices,
        preset.obstacle_vertices
    )

    decomposition_error = None
    try:
        region_vertices, _result = acd.decompose_to_polygons(
            environment.workspace,
            holes=environment.obstacles
        )
    except Exception as exc:
        print(f"Warning: ACD decomposition failed with error: {exc}")
        decomposition_error = exc
        region_vertices = []

    if not region_vertices:
        if not preset.region_vertices:
            raise RuntimeError(
                "ACD decomposition did not produce usable polygons and the preset "
                "does not define fallback region vertices."
            ) from decomposition_error
        region_vertices = [
            np.asarray(vertices, dtype=np.float64)
            for vertices in preset.region_vertices
        ]

    print(f"Built environment with {len(environment.obstacles)} obstacles and {len(region_vertices)} regions")

    # Buffered-but-not-tightened regions: used as adjacency_regions so that
    # corridor corners (point-contact in original ACD2D, area-overlap after
    # buffering) are still detected as adjacent. The Shapely polygon of a
    # buffered region is exact (no _vrep_from_hrep approximation), giving
    # accurate adjacency detection via regions_intersect.
    adjacency_regions = create_buffered_regions_from_vertices_list(
        region_vertices,
        preset.workspace_vertices,
        buffer_size=overlap_width,
        obstacle_polygons=None,
    )
    # Optimizer regions: fully tightened so that each region's H-rep stays
    # in free space. compute_intersection gates out edges that cross obstacle
    # walls (tightening separates wall-facing pairs, returning None).
    regions = create_buffered_regions_from_vertices_list(
        region_vertices,
        preset.workspace_vertices,
        buffer_size=overlap_width,
        obstacle_polygons=environment.obstacles,
    )
    return environment, regions, adjacency_regions


def _select_region_subset(regions: List[ConvexRegion],
                          active_region_indices: List[int] | None) -> List[ConvexRegion]:
    """Copy and reindex a region subset for a specific scenario."""
    if active_region_indices is None:
        selected = copy.deepcopy(regions)
    else:
        selected = [copy.deepcopy(regions[idx]) for idx in active_region_indices]

    for index, region in enumerate(selected):
        region.index = index
        region.label = region_node_label(index)

    return selected


def prepare_scenario(config: DemoConfig, scenario_name: str) -> PreparedScenario:
    """Build all runtime objects needed to solve one scenario."""
    resolved = config.resolve_scenario(scenario_name)
    preset = get_problem_preset(resolved.problem_preset)
    rd_cfg = resolved.runtime_config.get("region_decomposition", {})
    overlap_width = float(rd_cfg.get("overlap_width", 0.5))
    environment, all_regions, all_adjacency_regions = build_environment_and_regions(
        preset, overlap_width=overlap_width
    )
    regions = _select_region_subset(all_regions, resolved.active_region_indices)
    adjacency_regions = _select_region_subset(
        all_adjacency_regions,
        resolved.active_region_indices,
    )

    dyn_cfg = resolved.runtime_config.get("dynamics", {})
    dynamics = UnicycleModel(
        v_min=dyn_cfg.get("v_min", -2.0),
        v_max=dyn_cfg.get("v_max", 2.0),
        omega_min=dyn_cfg.get("omega_min", -np.pi),
        omega_max=dyn_cfg.get("omega_max", np.pi),
    )

    graph = build_region_graph(
        regions,
        resolved.start_state[:2],
        resolved.goal_state[:2],
        adjacency_regions=adjacency_regions,
    )
    optimizer = create_integrated_optimizer_from_config(graph, dynamics, resolved.runtime_config)

    return PreparedScenario(
        resolved=resolved,
        preset=preset,
        environment=environment,
        regions=regions,
        graph=graph,
        dynamics=dynamics,
        optimizer=optimizer,
        workspace_bounds=environment.get_bounds(),
    )
