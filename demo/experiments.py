"""
experiments.py - Scenario runner for the integrated one-phase MIOCP demo.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from app_config import DemoConfig
from optimizer import OptimizationResult
from scenario_builder import PreparedScenario, prepare_scenario

from matplotlib import pyplot as plt


@dataclass
class ExperimentResult:
    """Result of running one configured scenario."""
    scenario_name: str
    optimization_result: OptimizationResult
    n_regions: int
    n_edges: int
    n_paths: int
    setup_time: float
    
    def summary(self) -> str:
        """Return human-readable summary."""
        result = self.optimization_result
        lines = [
            "===========================================",
            f"  Scenario: {self.scenario_name}",
            "===========================================",
            f"  Regions: {self.n_regions}",
            f"  Edges: {self.n_edges}",
            f"  Candidate Paths: {self.n_paths}",
            f"  Setup Time: {self.setup_time:.3f} s",
            "",
            f"  Optimization Status: {'SUCCESS' if result.success else 'FAILED'}",
            f"  Solver: {result.solver_status}",
            f"  Total Cost: {result.total_cost:.4f}",
            f"  Solve Time: {result.solve_time:.3f} s",
            f"  Paths Evaluated: {result.n_paths_evaluated}",
            f"  Safety Mode: {result.safety_mode}",
        ]
        if result.safety_mode == "both":
            lines.extend([
                f"  Max CTCS Integral: {result.max_continuous_violation_integral:.2e}",
                f"  Max Dense Region Violation: {result.max_dense_region_violation:.2e}",
            ])
        else:
            # Spec Sec 8.3: CTCS/dense fields are not populated outside
            # safety_mode="both" -- "n/a" avoids implying a measured zero.
            lines.append(f"  Max CTCS Integral: n/a ({result.safety_mode} mode)")
            lines.append(f"  Max Dense Region Violation: n/a ({result.safety_mode} mode)")

        if result.success:
            lines.extend([
                "",
                f"  Path: {' -> '.join(result.path)}",
                f"  Total Duration: {sum(result.time_durations.values()):.3f} s",
                f"  Defect Norm: {result.defect_norm:.2e}",
                f"  Max Connection Gap: {result.max_connection_gap:.2e}",
                f"  Max Control Jump: {result.max_control_jump:.2e}",
            ])
            if result.max_integrality_gap > 0.0:
                lines.append(
                    f"  Max Integrality Gap: {result.max_integrality_gap:.2e}"
                )

        return "\n".join(lines)


@dataclass
class ScenarioResult:
    """Lightweight result record for centroid_refine_dms scenario comparison."""
    scenario_name: str
    solver_type: str
    success: bool
    total_cost: float
    solve_time: float
    n_paths_evaluated: int
    path_regions: List[int]
    formulation_mode: str = "INTEGRATED_MIOCP"
    min_safety_margin: float = float("nan")
    certified_safety_margin: float = float("nan")
    lipschitz_gap: float = float("nan")
    safety_certification: str = "NOT_SET"
    solver_status: str = ""


class ExperimentRunner:
    """Run configured scenarios with the integrated IPOPT-based solver."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = DemoConfig(config_path)
        self.output_dir = self.config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run_scenario(self, scenario_name: str, verbose: bool = True) -> Tuple[ExperimentResult, PreparedScenario]:
        """Build, solve, and summarize one scenario."""
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Running Scenario: {scenario_name}")
            print(f"{'=' * 60}")

        setup_start = time.time()
        prepared = prepare_scenario(self.config, scenario_name)

        if verbose:
            print(f"Preset: {prepared.preset.name}")
            print(f"Description: {prepared.resolved.description}")
            print(f"Active regions: {len(prepared.regions)}")
            prepared.graph.print_summary()

        max_paths = prepared.resolved.runtime_config.get("optimizer", {}).get("max_paths", 1000)
        n_paths = len(prepared.graph.enumerate_simple_paths(max_paths=max_paths))
        setup_time = time.time() - setup_start

        if verbose:
            print(f"Enumerated {n_paths} candidate paths")
            print("\nSolving with CentroidRefineDMS...")

        result = prepared.optimizer.solve(
            prepared.resolved.start_state,
            prepared.resolved.goal_state,
            verbose=verbose
        )

        exp_result = ExperimentResult(
            scenario_name=scenario_name,
            optimization_result=result,
            n_regions=len(prepared.regions),
            n_edges=prepared.graph.graph.number_of_edges(),
            n_paths=n_paths,
            setup_time=setup_time,
        )

        if verbose:
            print(exp_result.summary())

        return exp_result, prepared

    def run_all_scenarios(self, verbose: bool = True) -> Dict[str, Tuple[ExperimentResult, PreparedScenario]]:
        """Run all scenarios declared in the config."""
        results: Dict[str, Tuple[ExperimentResult, PreparedScenario]] = {}

        for scenario_name in self.config.list_scenarios():
            try:
                results[scenario_name] = self.run_scenario(scenario_name, verbose=verbose)
            except Exception as exc:
                print(f"Error running scenario {scenario_name}: {exc}")
                import traceback
                traceback.print_exc()

        return results

    def save_results(self,
                     results: Dict[str, Tuple[ExperimentResult, PreparedScenario]],
                     save_png: bool | None = None,
                     save_gif: bool | None = None,
                     save_json: bool | None = None) -> None:
        """Persist figures, animations, summaries, and a benchmark table."""
        save_png = self.config.save_png if save_png is None else save_png
        save_gif = self.config.save_gif if save_gif is None else save_gif
        save_json = self.config.save_json if save_json is None else save_json

        for scenario_name, (exp_result, prepared) in results.items():
            from visualization import create_animation, create_result_figure, save_result_summary

            opt_result = exp_result.optimization_result
            prefix = os.path.join(self.output_dir, scenario_name)

            if save_png:
                from visualization import create_velocity_profile_figure
                fig = create_result_figure(
                    opt_result,
                    prepared.graph,
                    prepared.workspace_bounds,
                    prepared.resolved.start_state[:2],
                    prepared.resolved.goal_state[:2],
                    obstacles=prepared.environment.obstacles,
                    title=f"GCS-MMS: {prepared.resolved.name}",
                )
                fig.savefig(f"{prefix}_result.png")
                plt.close(fig)
                print(f"Saved: {prefix}_result.png")

                fig_profile = create_velocity_profile_figure(opt_result)
                fig_profile.savefig(f"{prefix}_velocity_profile.png", dpi=150)
                plt.close(fig_profile)
                print(f"Saved: {prefix}_velocity_profile.png")

            if save_gif and opt_result.success:
                try:
                    create_animation(
                        opt_result,
                        prepared.graph,
                        prepared.workspace_bounds,
                        prepared.resolved.start_state[:2],
                        prepared.resolved.goal_state[:2],
                        filename=f"{prefix}_animation.gif",
                        obstacles=prepared.environment.obstacles,
                        fps=prepared.resolved.runtime_config.get("visualization", {}).get("animation", {}).get("fps", 30),
                        duration=prepared.resolved.runtime_config.get("visualization", {}).get("animation", {}).get("duration", 5.0),
                    )
                except Exception as exc:
                    print(f"Warning: Could not create animation: {exc}")

            if save_json:
                save_result_summary(opt_result, f"{prefix}_summary.json")

        self._save_benchmark_summary(results)

    def _save_benchmark_summary(self, results: Dict[str, Tuple[ExperimentResult, PreparedScenario]]) -> None:
        """Save a compact benchmark JSON and print a console table."""
        summary = {
            "scenarios": {},
            "metadata": {
                "config_file": str(self.config.path),
            },
        }

        for name, (exp_result, prepared) in results.items():
            opt_result = exp_result.optimization_result
            summary["scenarios"][name] = {
                "success": opt_result.success,
                "preset": prepared.preset.key,
                "n_regions": exp_result.n_regions,
                "n_edges": exp_result.n_edges,
                "n_paths": exp_result.n_paths,
                "setup_time": exp_result.setup_time,
                "solve_time": opt_result.solve_time,
                "total_cost": opt_result.total_cost,
                "paths_evaluated": opt_result.n_paths_evaluated,
                "pipeline_mode": opt_result.pipeline_mode,
                "n_paths_screened": opt_result.n_paths_screened,
                "n_paths_polished": opt_result.n_paths_polished,
                "early_stopped": opt_result.early_stopped,
                "repair_attempted": opt_result.repair_attempted,
                "path_length": len(opt_result.path_regions),
                "safety_mode": opt_result.safety_mode,
                "max_ctcs_integral": opt_result.max_continuous_violation_integral,
                "max_dense_region_violation": opt_result.max_dense_region_violation,
            }

        filename = os.path.join(self.output_dir, "benchmark_summary.json")
        with open(filename, "w") as handle:
            json.dump(summary, handle, indent=2)

        print(f"\nBenchmark summary saved to {filename}")
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"{'Scenario':<20} {'Status':<10} {'Regions':<10} {'Paths':<10} {'Cost':<12} {'Time':<10}")
        print("-" * 80)

        for name, (exp_result, _) in results.items():
            opt_result = exp_result.optimization_result
            status = "OK" if opt_result.success else "FAIL"
            cost_str = f"{opt_result.total_cost:.4f}" if opt_result.success else "N/A"
            print(
                f"{name:<20} {status:<10} {exp_result.n_regions:<10} "
                f"{exp_result.n_paths:<10} {cost_str:<12} {opt_result.solve_time:.3f}s"
            )

        print("=" * 80)


def run_quick_test(config_path: str = "config.yaml", scenario_name: str | None = None) -> OptimizationResult:
    """Run the integrated solver on one small scenario to verify the pipeline."""
    runner = ExperimentRunner(config_path=config_path)
    scenario_name = scenario_name or runner.config.default_scenario_name()
    exp_result, _ = runner.run_scenario(scenario_name, verbose=True)
    return exp_result.optimization_result


if __name__ == "__main__":
    run_quick_test()
