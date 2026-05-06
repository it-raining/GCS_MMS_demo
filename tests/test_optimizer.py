from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from convex_regions import create_regions_from_vertices_list
from dynamics import UnicycleModel
from graph_builder import SOURCE, TARGET, build_region_graph
from optimizer import OptimizationConfig, PathNLPSolver


class OptimizerTests(unittest.TestCase):
    def test_fixed_path_solver_on_single_square(self) -> None:
        regions = create_regions_from_vertices_list(
            [
                np.array(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 1.0],
                    ]
                )
            ]
        )
        start_state = np.array([0.2, 0.2, 0.0], dtype=np.float64)
        goal_state = np.array([0.8, 0.2, 0.0], dtype=np.float64)
        graph = build_region_graph(regions, start_state[:2], goal_state[:2])
        dynamics = UnicycleModel()
        config = OptimizationConfig(
            n_integration_steps=8,
            n_mesh_points=3,
            n_control_segments=1,
            safety_margin=0.01,
            delta_min=0.01,
            delta_max=3.0,
            max_iter=500,
            tol=1e-7,
        )

        solver = PathNLPSolver(graph, dynamics, config)
        result = solver.solve_path([SOURCE, "R0", TARGET], start_state, goal_state)

        self.assertTrue(result.success, result.solver_status)
        self.assertLess(result.defect_norm, 1e-6)
        self.assertLess(result.max_connection_gap, 1e-6)
        self.assertEqual(result.path_regions, [0])
        self.assertGreater(result.total_cost, 0.0)


if __name__ == "__main__":
    unittest.main()
