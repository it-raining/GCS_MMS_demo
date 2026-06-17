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
from optimizer import CentroidRefineDMSConfig, CentroidRefineDMSSolver, OptimizationResult


def _make_two_region_scenario():
    regions = create_regions_from_vertices_list([
        np.array([[0, 0], [1.2, 0], [1.2, 1], [0, 1]], dtype=float),
        np.array([[0.8, 0], [2.0, 0], [2.0, 1], [0.8, 1]], dtype=float),
    ])
    x_start = np.array([0.1, 0.5, 0.0])
    x_goal  = np.array([1.9, 0.5, 0.0])
    graph = build_region_graph(regions, x_start[:2], x_goal[:2])
    dynamics = UnicycleModel(v_max=2.0)
    return graph, dynamics, x_start, x_goal


class CentroidRefineDMSConfigTests(unittest.TestCase):
    def test_default_config_valid(self):
        cfg = CentroidRefineDMSConfig()
        self.assertEqual(cfg.mode, "first_feasible")
        self.assertGreater(cfg.epsilon_final, 0.0)
        self.assertTrue(cfg.use_centroid_cost)
        self.assertTrue(cfg.use_interface_qp)
        self.assertTrue(cfg.use_log_barrier)

    def test_epsilon_certificate_buffer_defaults_to_zero(self):
        cfg = CentroidRefineDMSConfig()
        self.assertEqual(cfg.epsilon_certificate_buffer, 0.0)

    def test_epsilon_certificate_buffer_parsed_from_config_dict(self):
        from optimizer import create_integrated_optimizer_from_config
        graph, dynamics, _, _ = _make_two_region_scenario()
        cfg_dict = {
            'optimizer': {'solver_mode': 'centroid_refine_dms'},
            'dynamics': {}, 'shooting': {}, 'cost': {}, 'control': {},
            'centroid_refine_dms': {'epsilon_certificate_buffer': 0.003},
        }
        solver = create_integrated_optimizer_from_config(graph, dynamics, cfg_dict)
        self.assertEqual(solver.config.epsilon_certificate_buffer, 0.003)

    def test_ablation_flags_exist(self):
        cfg = CentroidRefineDMSConfig(
            use_centroid_cost=False,
            use_interface_qp=False,
            use_log_barrier=False,
            use_barrier_continuation=False,
            use_inexact_tolerance=False,
        )
        self.assertFalse(cfg.use_centroid_cost)


class OptimizationResultNewFieldsTests(unittest.TestCase):
    def test_new_fields_exist_with_defaults(self):
        result = OptimizationResult(
            success=False, path=[], path_regions=[],
            total_cost=0.0, solve_time=0.0, n_paths_evaluated=0
        )
        self.assertTrue(np.isnan(result.certified_safety_margin))
        self.assertTrue(np.isnan(result.lipschitz_gap))
        self.assertEqual(result.safety_certification, "NOT_SET")
        self.assertEqual(result.lb_geometric, 0.0)
        self.assertEqual(result.optimality_gap, float("inf"))
        self.assertEqual(result.n_barrier_levels, 0)


class CentroidRefineDMSSolverTests(unittest.TestCase):
    def test_solver_returns_result(self):
        graph, dynamics, x_start, x_goal = _make_two_region_scenario()
        cfg = CentroidRefineDMSConfig(
            n_int=8, epsilon_final=1e-3, time_limit_s=60.0
        )
        solver = CentroidRefineDMSSolver(graph, dynamics, cfg)
        result = solver.solve(x_start, x_goal)
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.safety_mode, "log_barrier")

    def test_mandatory_disclaimer_set(self):
        graph, dynamics, x_start, x_goal = _make_two_region_scenario()
        cfg = CentroidRefineDMSConfig(n_int=8, epsilon_final=1e-3)
        solver = CentroidRefineDMSSolver(graph, dynamics, cfg)
        result = solver.solve(x_start, x_goal)
        self.assertIn("KKT", result.global_optimality_claim)
        self.assertNotIn("globally optimal", result.global_optimality_claim.lower())

    def test_successful_result_has_both_safety_margins(self):
        graph, dynamics, x_start, x_goal = _make_two_region_scenario()
        cfg = CentroidRefineDMSConfig(n_int=8, epsilon_final=1e-3)
        solver = CentroidRefineDMSSolver(graph, dynamics, cfg)
        result = solver.solve(x_start, x_goal)
        if result.success:
            self.assertTrue(np.isfinite(result.min_safety_margin))
            self.assertTrue(np.isfinite(result.certified_safety_margin))
            self.assertLessEqual(result.certified_safety_margin, result.min_safety_margin)
            self.assertNotEqual(result.safety_certification, "NOT_SET")
            self.assertEqual(
                set(result.exit_states),
                set(result.path_regions),
            )
            self.assertEqual(
                len(result.interface_points),
                len(result.path_regions) - 1,
            )
            self.assertEqual(len(result.trajectories), len(result.path_regions))
            for region_idx, (trajectory, _, _) in zip(
                result.path_regions, result.trajectories
            ):
                np.testing.assert_allclose(
                    trajectory[-1],
                    result.exit_states[region_idx],
                    atol=1e-8,
                )

    def test_failure_log_is_list(self):
        graph, dynamics, x_start, x_goal = _make_two_region_scenario()
        cfg = CentroidRefineDMSConfig(n_int=5, epsilon_final=1e-3)
        solver = CentroidRefineDMSSolver(graph, dynamics, cfg)
        result = solver.solve(x_start, x_goal)
        self.assertIsInstance(result.failure_log, list)

    def test_defect_and_connection_gap_are_measured(self):
        graph, dynamics, x_start, x_goal = _make_two_region_scenario()
        cfg = CentroidRefineDMSConfig(
            n_int=40, barrier_levels=[1.0, 0.5, 0.1, 0.01], time_limit_s=30.0,
        )
        solver = CentroidRefineDMSSolver(graph, dynamics, cfg)
        result = solver.solve(x_start, x_goal)
        self.assertTrue(result.success)
        self.assertGreater(result.defect_norm, 0.0)
        self.assertLess(result.max_connection_gap, 1e-3)

    def test_global_optimality_claim_matches_mandatory_disclaimer(self):
        graph, dynamics, x_start, x_goal = _make_two_region_scenario()
        cfg = CentroidRefineDMSConfig(n_int=8, epsilon_final=1e-3)
        solver = CentroidRefineDMSSolver(graph, dynamics, cfg)
        result = solver.solve(x_start, x_goal)
        expected = (
            "LB is a geometric lower bound on the time component only. "
            "gap_k is NOT a certificate for the full DMS objective. "
            "No global optimality claim is made. The reported solution is a "
            "KKT point of the fixed-path barrier-augmented NLP under LICQ + SOSC."
        )
        self.assertEqual(result.global_optimality_claim, expected)

    def test_lower_bound_and_optimality_gap_are_populated(self):
        graph, dynamics, x_start, x_goal = _make_two_region_scenario()
        cfg = CentroidRefineDMSConfig(
            n_int=40, barrier_levels=[1.0, 0.5, 0.1, 0.01], time_limit_s=30.0,
        )
        solver = CentroidRefineDMSSolver(graph, dynamics, cfg)
        result = solver.solve(x_start, x_goal)
        self.assertTrue(result.success)
        self.assertGreater(result.lb_geometric, 0.0)
        self.assertTrue(np.isfinite(result.optimality_gap))
        self.assertGreaterEqual(result.optimality_gap, 0.0)

    def test_solver_mode_dispatch(self):
        from optimizer import create_integrated_optimizer_from_config, CentroidRefineDMSSolver
        graph, dynamics, x_start, x_goal = _make_two_region_scenario()
        cfg_dict = {
            'optimizer': {'solver_mode': 'centroid_refine_dms'},
            'dynamics': {},
            'shooting': {},
            'cost': {},
            'control': {},
            'centroid_refine_dms': {'n_int': 5, 'epsilon_final': 1e-3},
        }
        solver = create_integrated_optimizer_from_config(graph, dynamics, cfg_dict)
        self.assertIsInstance(solver, CentroidRefineDMSSolver)


if __name__ == "__main__":
    unittest.main()
