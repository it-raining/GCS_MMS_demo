from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from convex_regions import (  # noqa: E402
    create_buffered_regions_from_vertices_list,
    create_regions_from_vertices_list,
)
from dynamics import (  # noqa: E402
    ControlParameterization,
    RK4Integrator,
    UnicycleModel,
    create_integration_bundle,
)
from graph_builder import SOURCE, TARGET, build_region_graph  # noqa: E402
from optimizer import IntegratedMIOCPSolver, OptimizationConfig, PathNLPSolver  # noqa: E402


def _square(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
        ],
        dtype=np.float64,
    )


def _small_solve_config(**overrides) -> OptimizationConfig:
    values = dict(
        n_integration_steps=8,
        n_mesh_points=3,
        n_control_segments=1,
        safety_margin=0.0,
        delta_min=0.01,
        delta_max=3.0,
        max_iter=700,
        tol=1e-7,
        print_level=0,
    )
    values.update(overrides)
    return OptimizationConfig(**values)


class EdgeCaseTests(unittest.TestCase):
    def test_single_region_near_identical_start_goal_with_cbf(self) -> None:
        regions = create_regions_from_vertices_list([_square(0.0, 0.0, 1.0, 1.0)])
        start_state = np.array([0.5, 0.5, 0.0], dtype=np.float64)
        goal_state = np.array([0.5001, 0.5, 0.0], dtype=np.float64)
        graph = build_region_graph(regions, start_state[:2], goal_state[:2])
        dynamics = UnicycleModel()
        config = _small_solve_config(cbf_alpha=5.0)

        solver = PathNLPSolver(graph, dynamics, config)
        result = solver.solve_path([SOURCE, "R0", TARGET], start_state, goal_state)

        self.assertTrue(result.success, result.solver_status)
        self.assertGreaterEqual(result.time_durations[0], config.delta_min - 1e-9)
        self.assertLess(result.defect_norm, 1e-8)
        self.assertLess(result.max_connection_gap, 1e-6)

        positions, velocities = solver.cbf_sampler(
            result.entry_states[0],
            result.control_params[0],
            result.time_durations[0],
        )
        positions = np.array(positions, dtype=float)
        velocities = np.array(velocities, dtype=float)
        region = graph.regions[0]
        for k in range(1, config.n_mesh_points - 1):
            h = region.b - region.A @ positions[k]
            lhs = result.time_durations[0] * (region.A @ velocities[k])
            rhs = config.cbf_alpha * h
            self.assertLessEqual(float(np.max(lhs - rhs)), 1e-6)

    def test_two_region_path_with_very_thin_overlap_solves(self) -> None:
        regions = create_regions_from_vertices_list(
            [
                _square(0.0, 0.0, 1.0, 1.0),
                _square(0.999, 0.0, 2.0, 1.0),
            ]
        )
        start_state = np.array([0.2, 0.5, 0.0], dtype=np.float64)
        goal_state = np.array([1.8, 0.5, 0.0], dtype=np.float64)
        graph = build_region_graph(regions, start_state[:2], goal_state[:2])
        dynamics = UnicycleModel()
        config = _small_solve_config(delta_max=4.0)

        self.assertTrue(graph.is_valid_path([SOURCE, "R0", "R1", TARGET]))
        intersection = graph.intersections[("R0", "R1")]
        self.assertLess(intersection.get_shapely_polygon().area, 0.01)

        solver = PathNLPSolver(graph, dynamics, config)
        result = solver.solve_path([SOURCE, "R0", "R1", TARGET], start_state, goal_state)

        self.assertTrue(result.success, result.solver_status)
        self.assertLess(result.defect_norm, 1e-8)
        self.assertLess(result.max_connection_gap, 1e-6)

    def test_extract_active_path_returns_empty_for_near_zero_edges(self) -> None:
        regions = create_regions_from_vertices_list(
            [
                _square(0.0, 0.0, 1.0, 1.0),
                _square(0.75, 0.0, 1.75, 1.0),
            ]
        )
        start_state = np.array([0.25, 0.5, 0.0], dtype=np.float64)
        goal_state = np.array([1.25, 0.5, 0.0], dtype=np.float64)
        graph = build_region_graph(regions, start_state[:2], goal_state[:2])
        solver = IntegratedMIOCPSolver(graph, UnicycleModel(), _small_solve_config())

        edge_values = {edge: 1e-9 for edge in graph.graph.edges()}
        self.assertEqual(solver._extract_active_path(edge_values), [])

    def test_casadi_and_numpy_endpoint_defects_match(self) -> None:
        dynamics = UnicycleModel()
        control_param = ControlParameterization(
            n_u=dynamics.n_u,
            parameterization="piecewise_constant",
            n_segments=2,
        )
        bundle = create_integration_bundle(dynamics, control_param, n_steps=12, n_mesh=4)
        integrator = RK4Integrator(dynamics, control_param, n_steps=12)

        x0 = np.array([0.3, 0.25, 0.4], dtype=np.float64)
        w = np.array([0.6, 0.2, 0.4, -0.1], dtype=np.float64)
        delta = 0.75

        casadi_endpoint = np.array(bundle.F_endpoint(x0, w, delta), dtype=float).reshape(-1)
        numpy_endpoint = integrator.integrate(x0, w, delta)
        np.testing.assert_allclose(casadi_endpoint, numpy_endpoint, atol=1e-10)

    def test_buffered_regions_create_expected_diagnostic_overlap(self) -> None:
        workspace = _square(0.0, 0.0, 2.0, 1.0)
        regions = create_buffered_regions_from_vertices_list(
            [
                _square(0.0, 0.0, 1.0, 1.0),
                _square(1.0, 0.0, 2.0, 1.0),
            ],
            workspace,
            buffer_size=0.05,
        )

        overlap = (
            regions[0]
            .get_shapely_polygon()
            .intersection(regions[1].get_shapely_polygon())
        )
        self.assertGreater(overlap.area, 0.0)


if __name__ == "__main__":
    unittest.main()
