from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))


@unittest.skipUnless(
    os.environ.get("RUN_DEMO_REGRESSION") == "1",
    "Set RUN_DEMO_REGRESSION=1 to run the full default scenario regression.",
)
class DefaultScenarioRegressionTests(unittest.TestCase):
    def test_default_scenario_matches_baseline(self) -> None:
        from experiments import ExperimentRunner

        runner = ExperimentRunner(config_path="config.yaml")
        exp_result, _prepared = runner.run_scenario("default", verbose=False)
        result = exp_result.optimization_result

        self.assertTrue(result.success, result.solver_status)
        self.assertEqual(
            result.path,
            ["source", "R9", "R13", "R4", "R7", "R6", "R10", "R11", "R0", "target"],
        )
        self.assertAlmostEqual(result.total_cost, 51.601708606135446, delta=1e-4)
        self.assertLess(result.defect_norm, 1e-6)
        self.assertLess(result.max_connection_gap, 1e-8)


class ScenarioResultTests(unittest.TestCase):
    def test_scenario_result_construction(self) -> None:
        from experiments import ScenarioResult
        result = ScenarioResult(
            scenario_name="test",
            solver_type="centroid_refine_dms",
            success=True,
            total_cost=42.0,
            solve_time=1.23,
            n_paths_evaluated=3,
            path_regions=[0, 1, 2],
        )
        self.assertEqual(result.scenario_name, "test")
        self.assertEqual(result.solver_type, "centroid_refine_dms")
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.total_cost, 42.0)
        self.assertEqual(result.path_regions, [0, 1, 2])
        self.assertEqual(result.formulation_mode, "INTEGRATED_MIOCP")
        self.assertEqual(result.safety_certification, "NOT_SET")

    def test_scenario_result_defaults(self) -> None:
        from experiments import ScenarioResult
        result = ScenarioResult(
            scenario_name="x",
            solver_type="integrated",
            success=False,
            total_cost=float('inf'),
            solve_time=0.0,
            n_paths_evaluated=0,
            path_regions=[],
        )
        self.assertTrue(np.isnan(result.min_safety_margin))
        self.assertTrue(np.isnan(result.certified_safety_margin))


if __name__ == "__main__":
    unittest.main()
