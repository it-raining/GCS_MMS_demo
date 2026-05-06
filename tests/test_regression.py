from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
