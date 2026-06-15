"""warmstart.py - Centroid warm-start builder for Centroid-Refine-DMS."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from dynamics import DynamicsModel
from graph_builder import RegionGraph


@dataclass
class WarmStartConfig:
    v_nom_fraction: float = 0.5
    n_int: int = 10


def build_centroid_warmstart(
    graph: RegionGraph,
    path_regions: List[int],
    anchor_points: np.ndarray,
    dynamics: DynamicsModel,
    config: WarmStartConfig,
) -> Dict[int, Dict]:
    """
    Build a simple piecewise-linear warm start from anchor points.

    anchor_points: shape (m+1, n_pos) - start, interfaces, goal in position space.
    Returns dict keyed by region_idx with 's_minus', 'w', 'delta'.
    """
    u_lb, u_ub = dynamics.control_bounds()
    v_max = float(u_ub[0]) if u_ub.size > 0 else 1.0
    v_nom = config.v_nom_fraction * v_max
    v_nom = max(v_nom, 1e-3)

    result = {}
    for i, region_idx in enumerate(path_regions):
        q_entry = anchor_points[i]
        q_exit = anchor_points[i + 1]
        direction = q_exit - q_entry
        dist = float(np.linalg.norm(direction))

        delta = max(dist / v_nom, 0.1)

        # Build entry state
        s_minus = np.zeros(dynamics.n_x, dtype=np.float64)
        s_minus[:2] = q_entry
        if dynamics.angle_indices:
            if dist > 1e-9:
                heading = float(np.arctan2(direction[1], direction[0]))
            else:
                heading = 0.0
            s_minus[dynamics.angle_indices[0]] = heading

        # Build nominal control (forward velocity, zero turning)
        w = np.zeros(dynamics.control_bounds()[0].shape[0], dtype=np.float64)
        w[0] = v_nom  # forward velocity

        result[region_idx] = {
            's_minus': s_minus,
            'w': w,
            'delta': delta,
        }

    return result
