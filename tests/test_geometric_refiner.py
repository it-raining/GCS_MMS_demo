from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from convex_regions import create_regions_from_vertices_list
from graph_builder import build_region_graph
from geometric_refiner import (
    InterfaceQPConfig, NarrowInterfaceError, solve_interface_refinement,
)


def _two_region_graph():
    regions = create_regions_from_vertices_list([
        np.array([[0, 0], [1.2, 0], [1.2, 1], [0, 1]], dtype=float),
        np.array([[0.8, 0], [2.0, 0], [2.0, 1], [0.8, 1]], dtype=float),
    ])
    start = np.array([0.1, 0.5])
    goal  = np.array([1.9, 0.5])
    return build_region_graph(regions, start, goal), [0, 1]


class InterfaceQPTests(unittest.TestCase):
    def test_qp_returns_correct_shape(self) -> None:
        graph, path_regions = _two_region_graph()
        cfg = InterfaceQPConfig(delta_safe=0.02, delta_extra=0.01)
        z = solve_interface_refinement(graph, path_regions, np.array([0.1, 0.5]), np.array([1.9, 0.5]), cfg)
        self.assertEqual(z.shape, (3, 2))

    def test_qp_fixed_endpoints(self) -> None:
        graph, path_regions = _two_region_graph()
        cfg = InterfaceQPConfig(delta_safe=0.02, delta_extra=0.01)
        q_start = np.array([0.1, 0.5])
        q_goal  = np.array([1.9, 0.5])
        z = solve_interface_refinement(graph, path_regions, q_start, q_goal, cfg)
        np.testing.assert_allclose(z[0], q_start, atol=1e-9)
        np.testing.assert_allclose(z[-1], q_goal, atol=1e-9)

    def test_strict_interior_guarantee(self) -> None:
        graph, path_regions = _two_region_graph()
        cfg = InterfaceQPConfig(delta_safe=0.02, delta_extra=0.01)
        z = solve_interface_refinement(graph, path_regions, np.array([0.1, 0.5]), np.array([1.9, 0.5]), cfg)
        margin = cfg.delta_safe + cfg.delta_extra
        m = len(path_regions)
        for i in range(1, m):
            for ri in [path_regions[i-1], path_regions[i]]:
                region = graph._regions_by_index[ri]
                pt = z[i]
                slacks = region.b - region.A @ pt
                self.assertTrue(np.all(slacks >= margin - 1e-6))

    def test_narrow_interface_raises(self) -> None:
        regions = create_regions_from_vertices_list([
            np.array([[0, 0], [1.001, 0], [1.001, 1], [0, 1]], dtype=float),
            np.array([[0.999, 0], [2.0, 0], [2.0, 1], [0.999, 1]], dtype=float),
        ])
        start = np.array([0.1, 0.5])
        goal  = np.array([1.9, 0.5])
        graph = build_region_graph(regions, start, goal)
        cfg = InterfaceQPConfig(delta_safe=0.02, delta_extra=0.05)
        with self.assertRaises(NarrowInterfaceError):
            solve_interface_refinement(graph, [0, 1], start, goal, cfg)


if __name__ == "__main__":
    unittest.main()
