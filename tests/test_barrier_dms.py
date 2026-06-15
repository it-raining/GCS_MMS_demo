from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from convex_regions import create_regions_from_vertices_list
from graph_builder import build_region_graph
from dynamics import UnicycleModel
from barrier_dms import BarrierDMSConfig, BarrierDMSSolver, BarrierPathResult


def _two_region_setup():
    regions = create_regions_from_vertices_list([
        np.array([[0, 0], [1.2, 0], [1.2, 1], [0, 1]], dtype=float),
        np.array([[0.8, 0], [2.0, 0], [2.0, 1], [0.8, 1]], dtype=float),
    ])
    start = np.array([0.1, 0.5])
    goal  = np.array([1.9, 0.5])
    graph = build_region_graph(regions, start, goal)
    dynamics = UnicycleModel(v_max=2.0)
    return graph, dynamics


class BarrierDMSSolverTests(unittest.TestCase):
    def test_solver_returns_result_object(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(n_int=20, barrier_levels=[1.0], time_limit_s=30.0)
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        result = solver.solve([0, 1], anchor_points)
        self.assertIsInstance(result, BarrierPathResult)

    def test_result_has_correct_regions(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(n_int=20, barrier_levels=[1.0], time_limit_s=30.0)
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        result = solver.solve([0, 1], anchor_points)
        self.assertEqual(result.path_regions, [0, 1])

    def test_lipschitz_gap_populated(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(n_int=20, barrier_levels=[1.0], time_limit_s=30.0)
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        result = solver.solve([0, 1], anchor_points)
        self.assertFalse(np.isnan(result.lipschitz_gap))
        self.assertGreaterEqual(result.lipschitz_gap, 0.0)

    def test_barrier_level_results_logged(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(n_int=20, barrier_levels=[1.0, 0.1], time_limit_s=30.0)
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        result = solver.solve([0, 1], anchor_points)
        self.assertGreater(len(result.barrier_level_results), 0)

    def test_solved_segments_populate_exit_states(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(n_int=20, barrier_levels=[1.0], time_limit_s=30.0)
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        result = solver.solve([0, 1], anchor_points)

        self.assertEqual(set(result.exit_states), set(result.entry_states))
        for state in result.exit_states.values():
            self.assertEqual(state.shape, (dynamics.n_x,))


if __name__ == "__main__":
    unittest.main()
