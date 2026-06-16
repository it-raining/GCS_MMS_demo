from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from convex_regions import create_buffered_regions_from_vertices_list, create_regions_from_vertices_list
from graph_builder import (
    SOURCE, TARGET, build_region_graph,
    add_composite_costs_to_graph, dijkstra_geometric_length,
)
from graph_types import (
    is_region_node_id,
    region_index_from_node_id,
    region_node_label,
)


def two_region_graph():
    regions = create_regions_from_vertices_list(
        [
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [0.75, 0.0],
                    [1.75, 0.0],
                    [1.75, 1.0],
                    [0.75, 1.0],
                ]
            ),
        ]
    )
    return build_region_graph(
        regions,
        start_pos=np.array([0.25, 0.5]),
        goal_pos=np.array([1.25, 0.5]),
    )


class GraphBuilderTests(unittest.TestCase):
    def test_region_node_helpers_validate_labels(self) -> None:
        self.assertEqual(region_node_label(12), "R12")
        self.assertTrue(is_region_node_id("R12"))
        self.assertEqual(region_index_from_node_id("R12"), 12)
        self.assertFalse(is_region_node_id("source"))
        with self.assertRaises(ValueError):
            region_index_from_node_id("bad")

    def test_graph_contains_expected_flow_structure(self) -> None:
        graph = two_region_graph()

        self.assertIn((SOURCE, "R0"), graph.source_edges)
        self.assertIn(("R1", TARGET), graph.target_edges)
        self.assertIn(("R0", "R1"), graph.region_edges)
        self.assertIn(("R1", "R0"), graph.region_edges)
        self.assertTrue(graph.is_valid_path([SOURCE, "R0", "R1", TARGET]))
        self.assertEqual(graph.get_region_by_id("R1").index, 1)

    def test_unbuffered_adjacency_rejects_point_touch_edge(self) -> None:
        raw_vertices = [
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0, 1.0],
                    [2.0, 1.0],
                    [2.0, 2.0],
                    [1.0, 2.0],
                ]
            ),
        ]
        workspace = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ],
            dtype=np.float64,
        )
        adjacency_regions = create_regions_from_vertices_list(raw_vertices)
        buffered_regions = create_buffered_regions_from_vertices_list(
            raw_vertices,
            workspace,
            buffer_size=0.01,
        )

        buffered_graph = build_region_graph(
            buffered_regions,
            start_pos=np.array([0.25, 0.25]),
            goal_pos=np.array([1.75, 1.75]),
        )
        adjacency_graph = build_region_graph(
            buffered_regions,
            start_pos=np.array([0.25, 0.25]),
            goal_pos=np.array([1.75, 1.75]),
            adjacency_regions=adjacency_regions,
        )

        self.assertIn(("R0", "R1"), buffered_graph.region_edges)
        self.assertNotIn(("R0", "R1"), adjacency_graph.region_edges)
        self.assertNotIn(("R1", "R0"), adjacency_graph.region_edges)

    def test_buffered_region_does_not_refill_obstacle_space(self) -> None:
        raw_vertices = [
            np.array(
                [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [2.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float64,
            )
        ]
        workspace = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ],
            dtype=np.float64,
        )
        obstacle = Polygon(
            [
                [0.8, 1.0],
                [1.2, 1.0],
                [1.2, 2.0],
                [0.8, 2.0],
            ]
        )

        regions = create_buffered_regions_from_vertices_list(
            raw_vertices,
            workspace,
            buffer_size=0.5,
            obstacle_polygons=[obstacle],
        )

        self.assertEqual(len(regions), len(raw_vertices))
        self.assertLessEqual(
            regions[0].get_shapely_polygon().intersection(obstacle).area,
            1e-10,
        )


class ChebyshevCenterTests(unittest.TestCase):
    def test_unit_square_center(self) -> None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
        from graph_builder import chebyshev_center
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        b = np.array([1.0, 0.0, 1.0, 0.0])
        center, radius = chebyshev_center(A, b)
        np.testing.assert_allclose(center, [0.5, 0.5], atol=1e-6)
        self.assertAlmostEqual(radius, 0.5, places=6)

    def test_rectangle_center(self) -> None:
        from graph_builder import chebyshev_center
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        b = np.array([2.0, 0.0, 1.0, 0.0])
        center, radius = chebyshev_center(A, b)
        self.assertAlmostEqual(radius, 0.5, places=6)

    def test_infeasible_raises(self) -> None:
        from graph_builder import chebyshev_center
        A = np.array([[1.0], [-1.0]])
        b = np.array([-1.0, -1.0])
        with self.assertRaises(ValueError):
            chebyshev_center(A, b)


class CompositeCostTests(unittest.TestCase):
    def _make_two_region_graph(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
        from convex_regions import create_regions_from_vertices_list
        from graph_builder import build_region_graph
        regions = create_regions_from_vertices_list([
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
            np.array([[0.8, 0], [2, 0], [2, 1], [0.8, 1]], dtype=float),
        ])
        start = np.array([0.1, 0.5])
        goal  = np.array([1.9, 0.5])
        return build_region_graph(regions, start, goal)

    def test_add_composite_costs_returns_centers(self) -> None:
        from graph_builder import add_composite_costs_to_graph
        graph = self._make_two_region_graph()
        centers = add_composite_costs_to_graph(
            graph,
            start_pos=np.array([0.1, 0.5]),
            goal_pos=np.array([1.9, 0.5]),
        )
        self.assertEqual(len(centers), 2)
        for idx, (c, r) in centers.items():
            self.assertEqual(len(c), 2)
            self.assertGreater(r, 0.0)

    def test_composite_cost_on_edges(self) -> None:
        from graph_builder import add_composite_costs_to_graph, SOURCE, TARGET
        graph = self._make_two_region_graph()
        add_composite_costs_to_graph(
            graph,
            start_pos=np.array([0.1, 0.5]),
            goal_pos=np.array([1.9, 0.5]),
        )
        for u, v, data in graph.graph.edges(data=True):
            self.assertIn('composite_cost', data)
            self.assertGreater(data['composite_cost'], 0.0)


class KShortestPathsTests(unittest.TestCase):
    def _make_graph(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
        from convex_regions import create_regions_from_vertices_list
        from graph_builder import build_region_graph, add_composite_costs_to_graph
        regions = create_regions_from_vertices_list([
            np.array([[0, 0], [1.2, 0], [1.2, 1], [0, 1]], dtype=float),
            np.array([[0.8, 0], [2.2, 0], [2.2, 1], [0.8, 1]], dtype=float),
            np.array([[1.8, 0], [3.0, 0], [3.0, 1], [1.8, 1]], dtype=float),
        ])
        start = np.array([0.1, 0.5])
        goal  = np.array([2.9, 0.5])
        graph = build_region_graph(regions, start, goal)
        add_composite_costs_to_graph(graph, start, goal)
        return graph

    def test_generator_yields_paths(self) -> None:
        from graph_builder import k_shortest_paths_generator, SOURCE, TARGET
        graph = self._make_graph()
        gen = k_shortest_paths_generator(graph, SOURCE, TARGET)
        path = next(gen)
        self.assertEqual(path[0], SOURCE)
        self.assertEqual(path[-1], TARGET)

    def test_paths_are_simple(self) -> None:
        from graph_builder import k_shortest_paths_generator, SOURCE, TARGET
        graph = self._make_graph()
        gen = k_shortest_paths_generator(graph, SOURCE, TARGET)
        for _ in range(3):
            try:
                path = next(gen)
                self.assertEqual(len(path), len(set(path)), "path has repeated nodes")
            except StopIteration:
                break

    def test_all_yielded_paths_are_distinct(self) -> None:
        from graph_builder import k_shortest_paths_generator, SOURCE, TARGET
        graph = self._make_graph()
        all_paths = []
        gen = k_shortest_paths_generator(graph, SOURCE, TARGET)
        for _ in range(5):
            try:
                all_paths.append(tuple(next(gen)))
            except StopIteration:
                break
        self.assertEqual(len(all_paths), len(set(all_paths)))


class DijkstraGeometricLengthTests(unittest.TestCase):
    def test_returns_pure_euclidean_path_length(self) -> None:
        graph = two_region_graph()
        add_composite_costs_to_graph(
            graph, start_pos=graph.start_pos, goal_pos=graph.goal_pos,
        )
        length = dijkstra_geometric_length(graph, SOURCE, TARGET)
        self.assertGreater(length, 0.0)
        direct = float(np.linalg.norm(graph.goal_pos - graph.start_pos))
        # Going through region centroids cannot be shorter than the
        # straight-line start-to-goal distance.
        self.assertGreaterEqual(length, direct - 1e-9)


if __name__ == "__main__":
    unittest.main()
