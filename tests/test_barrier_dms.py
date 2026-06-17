from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np
import casadi as ca

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from convex_regions import create_regions_from_vertices_list
from graph_builder import build_region_graph
from dynamics import UnicycleModel
from barrier_dms import BarrierDMSConfig, BarrierDMSSolver, BarrierPathResult
from constraint_layers import compute_lipschitz_safety_gap


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

    def test_compute_min_slack_scales_with_configured_delta_safe(self) -> None:
        graph, dynamics = _two_region_setup()
        result = BarrierPathResult(
            success=True, path_regions=[0], total_cost=0.0, solve_time=0.0,
            solver_status="ok",
            entry_states={0: np.array([0.5, 0.5, 0.0])},
            control_params={0: np.zeros(dynamics.n_u * 2)},
            time_durations={0: 1.0},
        )
        cfg_low = BarrierDMSConfig(n_int=5, barrier_levels=[1.0], delta_safe=0.0)
        cfg_high = BarrierDMSConfig(n_int=5, barrier_levels=[1.0], delta_safe=0.05)
        slack_low = BarrierDMSSolver(graph, dynamics, cfg_low)._compute_min_slack(result)
        slack_high = BarrierDMSSolver(graph, dynamics, cfg_high)._compute_min_slack(result)
        # Only cfg.delta_safe differs between the two configs; L_s and eps_int
        # depend on dynamics/path/Delta_i, which are identical, so the gap
        # between the two slacks must equal exactly the delta_safe gap.
        self.assertAlmostEqual(slack_low - slack_high, 0.05, places=9)

    def test_safety_slack_lower_bound_is_delta_extra_plus_buffer(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(
            n_int=5, barrier_levels=[1.0], delta_extra=0.02,
            epsilon_certificate_buffer=0.01,
        )
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        node_delta_safe = np.full((2, cfg.n_int + 1), cfg.delta_safe)
        out = solver._build_nlp_symbols(
            [0, 1], anchor_points, mu=1.0, node_delta_safe=node_delta_safe,
        )
        lbx = out[4]
        expected_floor = cfg.delta_extra + cfg.epsilon_certificate_buffer
        # lbx layout per segment: [state_lb..., w_lb..., delta_min, then
        # one entry per safety-slack component per node]. Every safety-slack
        # lower bound must equal delta_extra + epsilon_certificate_buffer,
        # never the old hardcoded 1e-10.
        self.assertNotIn(1e-10, lbx)
        self.assertTrue(any(abs(v - expected_floor) < 1e-12 for v in lbx))

    def test_live_margin_keeps_min_slack_above_delta_extra(self) -> None:
        # The defined_slack expression must depend on delta_vars[i]: once
        # the per-segment Lipschitz/RK4 terms are embedded live, a converged
        # solve's sampled min_safety_margin (which nets out that live
        # margin) must still clear the hard delta_extra floor enforced on
        # the slack variable.
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(
            n_int=40, barrier_levels=[1.0, 0.5, 0.1, 0.01], time_limit_s=30.0,
        )
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        result = solver.solve(
            [0, 1], anchor_points,
            start_state=np.array([0.1, 0.5, 0.0]),
            goal_state=np.array([1.9, 0.5, 0.0]),
        )
        self.assertTrue(result.success)
        self.assertGreaterEqual(
            result.min_safety_margin, cfg.delta_extra - 1e-6,
        )

    def test_solved_segments_populate_exit_states(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(n_int=20, barrier_levels=[1.0], time_limit_s=30.0)
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        result = solver.solve([0, 1], anchor_points)

        self.assertEqual(set(result.exit_states), set(result.entry_states))
        for state in result.exit_states.values():
            self.assertEqual(state.shape, (dynamics.n_x,))

    def test_defect_norm_is_small_and_positive(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(
            n_int=40, barrier_levels=[1.0, 0.5, 0.1, 0.01], time_limit_s=30.0,
        )
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        result = solver.solve(
            [0, 1], anchor_points,
            start_state=np.array([0.1, 0.5, 0.0]),
            goal_state=np.array([1.9, 0.5, 0.0]),
        )
        self.assertTrue(result.success)
        self.assertGreater(result.defect_norm, 0.0)
        self.assertLess(result.defect_norm, cfg.epsilon_final * 10)

    def test_build_nlp_symbols_splits_barrier_from_base_cost(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(n_int=20, barrier_levels=[1.0])
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        delta_arr = np.array([1.0, 1.0])
        lip_gap = compute_lipschitz_safety_gap(dynamics, [0, 1], graph, delta_arr, cfg.n_int)
        delta_safe = max(cfg.delta_safe, lip_gap + cfg.epsilon_final)
        node_delta_safe = np.full((2, cfg.n_int + 1), delta_safe)

        mu = 1.0
        out1 = solver._build_nlp_symbols(
            [0, 1], anchor_points, mu, node_delta_safe=node_delta_safe,
        )
        x_sym, obj_base, barrier_term = out1[0], out1[1], out1[2]
        x0 = out1[8]

        out2 = solver._build_nlp_symbols(
            [0, 1], anchor_points, 2 * mu, node_delta_safe=node_delta_safe,
        )
        x_sym2, obj_base2, barrier_term2 = out2[0], out2[1], out2[2]
        x0_2 = out2[8]

        base_fn = ca.Function('base', [x_sym], [obj_base])
        barrier_fn = ca.Function('barrier', [x_sym], [barrier_term])
        base_fn2 = ca.Function('base2', [x_sym2], [obj_base2])
        barrier_fn2 = ca.Function('barrier2', [x_sym2], [barrier_term2])

        base_val = float(base_fn(x0))
        barrier_val = float(barrier_fn(x0))

        self.assertNotEqual(barrier_val, 0.0)
        # obj_base must not depend on mu; barrier_term must scale linearly with mu.
        # (Both calls use the same anchor_points/delta_safe/node_delta_safe, so
        # x0 == x0_2 and the fresh CasADi symbols line up positionally.)
        self.assertAlmostEqual(float(base_fn2(x0_2)), base_val, places=8)
        self.assertAlmostEqual(float(barrier_fn2(x0_2)), 2 * barrier_val, places=8)

    def test_n_nlp_iterations_recorded(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(
            n_int=40, barrier_levels=[1.0, 0.5, 0.1, 0.01], time_limit_s=30.0,
        )
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        result = solver.solve(
            [0, 1], anchor_points,
            start_state=np.array([0.1, 0.5, 0.0]),
            goal_state=np.array([1.9, 0.5, 0.0]),
        )
        self.assertTrue(result.success)
        # 4 barrier levels run; n_nlp_iterations must be the sum across all
        # of them, not just the final (winning) level's iteration count.
        self.assertEqual(len(result.barrier_level_results), len(cfg.barrier_levels))
        self.assertEqual(
            result.n_nlp_iterations,
            sum(lr['n_iter'] for lr in result.barrier_level_results),
        )

    def test_certified_margin_accounts_for_defect_and_integration_error(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(
            n_int=40, barrier_levels=[1.0, 0.5, 0.1, 0.01], time_limit_s=30.0,
        )
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        result = solver.solve(
            [0, 1], anchor_points,
            start_state=np.array([0.1, 0.5, 0.0]),
            goal_state=np.array([1.9, 0.5, 0.0]),
        )
        self.assertTrue(result.success)
        self.assertLessEqual(
            result.certified_safety_margin,
            result.min_safety_margin - result.lipschitz_gap,
        )


if __name__ == "__main__":
    unittest.main()
