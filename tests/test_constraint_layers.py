from __future__ import annotations

import sys
import unittest
from pathlib import Path

import casadi as ca
import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from constraint_layers import (
    build_fixed_path_boundary_constraints,
    build_network_flow_constraints,
    merge_constraint_layers,
    stack_constraints,
)
from dynamics import UnicycleModel
from graph_builder import build_region_graph
from convex_regions import create_regions_from_vertices_list


def make_graph():
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


class ConstraintLayerTests(unittest.TestCase):
    def test_network_flow_layer_has_expected_scalar_constraints(self) -> None:
        graph = make_graph()
        edges = list(graph.graph.edges())
        region_nodes = graph.get_region_nodes()

        y_vars = {edge: ca.MX.sym(f"y_{idx}") for idx, edge in enumerate(edges)}
        p_vars = {node: ca.MX.sym(f"p_{node}") for node in region_nodes}

        g, lbg, ubg = build_network_flow_constraints(
            y_vars,
            p_vars,
            graph.source_edges,
            graph.target_edges,
            region_nodes,
            edges,
        )

        self.assertEqual(len(g), 2 + 2 * len(region_nodes))
        self.assertEqual(len(lbg), len(g))
        self.assertEqual(len(ubg), len(g))
        self.assertEqual(stack_constraints(g).numel(), len(g))

    def test_boundary_layer_projects_positions_only(self) -> None:
        dynamics = UnicycleModel()
        s0 = ca.MX.sym("s0", dynamics.n_x)
        s1 = ca.MX.sym("s1", dynamics.n_x)
        start = np.array([0.0, 0.0, 1.2])
        goal = np.array([1.0, 0.0, -0.7])

        layer = build_fixed_path_boundary_constraints(dynamics, s0, s1, start, goal)
        g, lbg, ubg = merge_constraint_layers([layer])

        self.assertEqual(stack_constraints(g).numel(), 4)
        self.assertEqual(lbg, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(ubg, [0.0, 0.0, 0.0, 0.0])


class BarrierLogTermsTests(unittest.TestCase):
    def _make_setup(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
        from convex_regions import create_regions_from_vertices_list
        from graph_builder import build_region_graph
        from dynamics import UnicycleModel
        import casadi as ca
        regions = create_regions_from_vertices_list([
            np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float),
        ])
        graph = build_region_graph(regions, np.array([0.1, 0.1]), np.array([1.9, 1.9]))
        dynamics = UnicycleModel()
        return graph, dynamics, ca

    def test_barrier_log_terms_scalar(self) -> None:
        from constraint_layers import build_barrier_log_terms
        graph, dynamics, ca = self._make_setup()
        n_int = 2
        x_vars = [[ca.MX.sym(f'x_{k}', dynamics.n_x) for k in range(n_int + 1)]]
        delta_list = [ca.MX.sym('delta_0', 1)]
        B = build_barrier_log_terms(
            path_regions=[0],
            graph=graph,
            dynamics=dynamics,
            x_node_vars=x_vars,
            delta_list=delta_list,
            n_int=n_int,
            delta_safe=0.02,
        )
        self.assertEqual(B.numel(), 1)

    def test_lipschitz_gap_positive(self) -> None:
        from constraint_layers import compute_lipschitz_safety_gap
        from dynamics import UnicycleModel
        from convex_regions import create_regions_from_vertices_list
        from graph_builder import build_region_graph
        regions = create_regions_from_vertices_list([
            np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float),
        ])
        graph = build_region_graph(regions, np.array([0.1, 0.1]), np.array([1.9, 1.9]))
        dynamics = UnicycleModel(v_max=2.0)
        gap = compute_lipschitz_safety_gap(
            dynamics=dynamics,
            path_regions=[0],
            graph=graph,
            delta_arr=np.array([1.0]),
            n_int=10,
        )
        self.assertGreater(gap, 0.0)


if __name__ == "__main__":
    unittest.main()
