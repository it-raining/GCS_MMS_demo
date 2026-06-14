from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from convex_regions import create_buffered_regions_from_vertices_list, create_regions_from_vertices_list
from graph_builder import SOURCE, TARGET, build_region_graph
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
            adjacency_tolerance=0.0,
        )

        self.assertIn(("R0", "R1"), buffered_graph.region_edges)
        self.assertNotIn(("R0", "R1"), adjacency_graph.region_edges)
        self.assertNotIn(("R1", "R0"), adjacency_graph.region_edges)


if __name__ == "__main__":
    unittest.main()
