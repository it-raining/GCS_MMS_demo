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
from warmstart import WarmStartConfig, build_centroid_warmstart


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


class WarmStartTests(unittest.TestCase):
    def test_warmstart_returns_correct_n_regions(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = WarmStartConfig(v_nom_fraction=0.5, n_int=5)
        ws = build_centroid_warmstart(
            graph=graph,
            path_regions=[0, 1],
            anchor_points=np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]]),
            dynamics=dynamics,
            config=cfg,
        )
        self.assertEqual(len(ws), 2)

    def test_warmstart_keys_present(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = WarmStartConfig(v_nom_fraction=0.5, n_int=5)
        ws = build_centroid_warmstart(
            graph=graph,
            path_regions=[0, 1],
            anchor_points=np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]]),
            dynamics=dynamics,
            config=cfg,
        )
        for region_idx, data in ws.items():
            self.assertIn('s_minus', data)
            self.assertIn('w', data)
            self.assertIn('delta', data)

    def test_warmstart_delta_positive(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = WarmStartConfig(v_nom_fraction=0.5, n_int=5)
        ws = build_centroid_warmstart(
            graph=graph,
            path_regions=[0, 1],
            anchor_points=np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]]),
            dynamics=dynamics,
            config=cfg,
        )
        for region_idx, data in ws.items():
            self.assertGreater(float(data['delta']), 0.0)

    def test_warmstart_state_dim(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = WarmStartConfig(v_nom_fraction=0.5, n_int=5)
        ws = build_centroid_warmstart(
            graph=graph,
            path_regions=[0, 1],
            anchor_points=np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]]),
            dynamics=dynamics,
            config=cfg,
        )
        for region_idx, data in ws.items():
            self.assertEqual(len(data['s_minus']), dynamics.n_x)


if __name__ == "__main__":
    unittest.main()
