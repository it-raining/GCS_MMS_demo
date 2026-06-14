#!/usr/bin/env python3
"""
main_demo.py - CLI for the integrated one-phase GCS-MMS demo.
"""

from __future__ import annotations

import argparse
import os
import sys

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DEMO_DIR)

sys.path.insert(0, DEMO_DIR)
sys.path.insert(0, REPO_ROOT)

from app_config import DemoConfig


def visualize_environment_only(config_path: str, scenario_name: str | None = None) -> None:
    """Visualize the configured preset and graph structure for one scenario."""
    import matplotlib.pyplot as plt

    from scenario_builder import prepare_scenario
    from visualization import plot_environment, plot_graph_structure, setup_plot_style

    config = DemoConfig(config_path)
    scenario_name = scenario_name or config.default_scenario_name()
    prepared = prepare_scenario(config, scenario_name)

    print(f"Visualizing scenario '{scenario_name}' with preset '{prepared.preset.key}'...")
    prepared.graph.print_summary()

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    plot_environment(
        axes[0],
        prepared.workspace_bounds,
        prepared.regions,
        prepared.resolved.start_state[:2],
        prepared.resolved.goal_state[:2],
        obstacles=prepared.environment.obstacles,
        show_labels=True,
        alpha=0.4,
    )
    axes[0].set_title(f"Safe Convex Regions: {prepared.resolved.name}")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")

    plot_environment(
        axes[1],
        prepared.workspace_bounds,
        prepared.regions,
        prepared.resolved.start_state[:2],
        prepared.resolved.goal_state[:2],
        obstacles=prepared.environment.obstacles,
        show_labels=True,
        alpha=0.2,
    )
    plot_graph_structure(axes[1], prepared.graph)
    axes[1].set_title("Region Graph G = (V, E)")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")

    plt.tight_layout()
    os.makedirs(config.output_dir, exist_ok=True)
    output_path = os.path.join(config.output_dir, f"{scenario_name}_environment.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.show()


def run_single_scenario(config_path: str, scenario_name: str,
                        verbose: bool = True, save_md: bool | None = None):
    """Run one configured scenario and persist its outputs."""
    from experiments import ExperimentRunner

    runner = ExperimentRunner(config_path=config_path)
    exp_result, prepared = runner.run_scenario(scenario_name, verbose=verbose)
    runner.save_results({scenario_name: (exp_result, prepared)}, save_md=save_md)
    return exp_result


def run_all_experiments(config_path: str, verbose: bool = True,
                        save_md: bool | None = None):
    """Run all configured scenarios."""
    from experiments import ExperimentRunner

    print("\n" + "=" * 70)
    print("  GCS-MMS Motion Planning Demo")
    print("  Integrated One-Phase MIOCP Relaxation")
    print("=" * 70)

    runner = ExperimentRunner(config_path=config_path)
    results = runner.run_all_scenarios(verbose=verbose)
    runner.save_results(results, save_md=save_md)

    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print(f"  Results saved to: {runner.output_dir}/")
    print("=" * 70)
    return results


def run_maze_benchmark(config_path: str,
                       scenario_name: str | None,
                       count: int,
                       maze_size: int,
                       knock_downs: int,
                       seed_start: int,
                       wall_thickness: float,
                       save_debug_geometry: bool,
                       verbose: bool = True):
    """Benchmark the solver on randomly generated maze instances."""
    from maze_benchmark import MazeBenchmarkRunner

    runner = MazeBenchmarkRunner(config_path=config_path, template_scenario=scenario_name)
    return runner.run(
        count=count,
        maze_size=maze_size,
        knock_downs=knock_downs,
        seed_start=seed_start,
        wall_thickness=wall_thickness,
        save_debug_geometry=save_debug_geometry,
        verbose=verbose,
    )


def run_shooting_animation(config_path: str,
                           scenario_name: str | None,
                           verbose: bool = True) -> str:
    """Run a fixed-path NLP with iteration recording and create a GIF."""
    import numpy as np

    from graph_types import is_terminal_node_id, region_index_from_node_id
    from scenario_builder import prepare_scenario
    from shooting_animation import (
        ShootingIterationRecorder,
        create_shooting_convergence_animation,
        parse_recorder_snapshots,
        parse_shooting_snapshot,
    )

    config = DemoConfig(config_path)
    scenario_name = scenario_name or config.default_scenario_name()
    prepared = prepare_scenario(config, scenario_name)

    if verbose:
        print(f"Running integrated solve to recover path for scenario '{scenario_name}'...")

    relaxation_result = prepared.optimizer.solve(
        prepared.resolved.start_state,
        prepared.resolved.goal_state,
        verbose=verbose,
    )
    if not relaxation_result.success:
        raise RuntimeError(
            f"Could not recover a path for shooting animation: "
            f"{relaxation_result.solver_status}"
        )

    path = relaxation_result.path
    path_regions = [
        region_index_from_node_id(node_id)
        for node_id in path
        if not is_terminal_node_id(node_id)
    ]

    if verbose:
        print(f"Recording fixed-path NLP iterations on path: {' -> '.join(path)}")

    path_solver = prepared.optimizer.path_solver
    recorder = ShootingIterationRecorder(record_every=1, max_snapshots=500)
    fixed_result = path_solver.solve_path(
        path,
        prepared.resolved.start_state,
        prepared.resolved.goal_state,
        iteration_recorder=recorder,
    )
    if not fixed_result.success:
        raise RuntimeError(
            f"Fixed-path solve failed during shooting animation: "
            f"{fixed_result.solver_status}"
        )

    shooting_cfg = prepared.resolved.runtime_config.get("shooting", {})
    n_integration_steps = shooting_cfg.get("n_integration_steps", 20)
    n_mesh_points = shooting_cfg.get("n_mesh_points", 5)

    snapshots = parse_recorder_snapshots(
        recorder,
        path_regions,
        prepared.dynamics.n_x,
        path_solver.control_param.n_w,
        prepared.dynamics,
        path_solver.control_param,
        path_solver.F_endpoint,
        n_integration_steps,
        n_mesh_points=n_mesh_points,
    )

    final_x = []
    for region_idx in path_regions:
        final_x.extend(fixed_result.entry_states[region_idx].tolist())
        final_x.extend(fixed_result.control_params[region_idx].tolist())
        final_x.append(float(fixed_result.time_durations[region_idx]))

    final_snapshot = parse_shooting_snapshot(
        np.asarray(final_x, dtype=np.float64),
        path_regions,
        prepared.dynamics.n_x,
        path_solver.control_param.n_w,
        prepared.dynamics,
        path_solver.control_param,
        path_solver.F_endpoint,
        n_integration_steps,
        n_mesh_points=n_mesh_points,
        cost=fixed_result.total_cost,
        iteration=(snapshots[-1]["iteration"] + 1) if snapshots else 0,
    )
    if not snapshots or snapshots[-1]["max_gap"] > final_snapshot["max_gap"] + 1e-10:
        snapshots.append(final_snapshot)

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scenario_name}_shooting_convergence.gif")
    animation_cfg = prepared.resolved.runtime_config.get("visualization", {}).get("animation", {})
    create_shooting_convergence_animation(
        snapshots,
        prepared.graph,
        prepared.workspace_bounds,
        prepared.resolved.start_state,
        prepared.resolved.goal_state,
        filename=output_path,
        obstacles=prepared.environment.obstacles,
        fps=animation_cfg.get("shooting_fps", animation_cfg.get("fps", 12)),
        max_animation_frames=animation_cfg.get("max_shooting_frames", 200),
    )

    return output_path


def print_formulation_summary() -> None:
    """Print a concise summary of the integrated formulation."""
    summary = """
==============================================================================
                    GCS-MMS INTEGRATED MIOCP FORMULATION
==============================================================================

  PROBLEM CLASS:
    - Continuous-time: Mixed-Integer Nonlinear Optimal Control (MIOCP)
    - Transcribed solve: Continuous relaxation of the integrated MINLP

  DECISION VARIABLES:
    - Relaxed edge flow y_uv in [0,1]
    - Relaxed region activation p_v in [0,1]
    - Entry/exit states s_v^-, s_v^+
    - Control parameters w_v and dwell time Delta_v
    - Interface states z_uv and epigraph costs rho_v

  CONSTRAINT LAYERS:
    1. Network flow over the region graph
    2. On/off convex geometry via Big-M
    3. Interface and boundary coupling
    4. Multiple-shooting dynamics, safety sampling, local cost epigraph

  SOLUTION STRATEGY:
    - IPOPT solves the integrated continuous relaxation directly
    - If extracted relaxed paths are fractional/discontinuous,
      a fixed-path NLP polish enforces a continuous final trajectory

  MODULAR SETUP:
    - Geometry presets live in problem_data.py
    - Scenario overrides live in config.yaml
    - Runtime assembly lives in scenario_builder.py

==============================================================================
"""
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GCS-MMS Integrated One-Phase MIOCP Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_demo.py
  python main_demo.py --scenario default
  python main_demo.py --scenario default --quiet
  python main_demo.py --scenario maze
  python main_demo.py --list-presets
  python main_demo.py --visualize-only --scenario default
  python main_demo.py --shooting-animation --scenario default
  python main_demo.py --config config.yaml --quick-test
  python main_demo.py --maze-benchmark --maze-count 5 --maze-size 25 --maze-knock-downs 25 --maze-seed 4
  python main_demo.py --maze-benchmark --maze-debug-geometry --maze-count 1 --maze-size 25 --maze-knock-downs 25 --maze-seed 4
        """,
    )

    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--scenario", type=str, help="Run one named scenario from the config")
    parser.add_argument("--quick-test", action="store_true", help="Run one integrated-solver quick test")
    parser.add_argument("--visualize-only", action="store_true", help="Only visualize one scenario layout")
    parser.add_argument("--shooting-animation", action="store_true", help="Create a fixed-path shooting convergence GIF")
    parser.add_argument("--formulation", action="store_true", help="Print formulation summary")
    parser.add_argument("--list-presets", action="store_true", help="List available geometry presets")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios")
    parser.add_argument("--maze-benchmark", action="store_true", help="Benchmark randomly generated maze instances")
    parser.add_argument("--maze-count", type=int, default=5, help="Number of random mazes to benchmark")
    parser.add_argument("--maze-size", type=int, default=25, help="Maze width/height in cells")
    parser.add_argument("--maze-knock-downs", type=int, default=25, help="Extra random wall removals after maze generation")
    parser.add_argument("--maze-seed", type=int, default=4, help="Starting random seed for maze generation")
    parser.add_argument("--maze-wall-thickness", type=float, default=0.08, help="Thickness of generated interior maze walls")
    parser.add_argument("--maze-debug-geometry", action="store_true", help="Save workspace + hole geometry plots before ACD")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--save-md", action="store_true", default=None,
                        help="Save Markdown reports to docs/ (overrides config if set)")

    args = parser.parse_args()
    config = DemoConfig(args.config)

    if args.list_scenarios:
        print("Available scenarios:")
        for scenario_name in config.list_scenarios():
            print(f"  - {scenario_name}")
        return

    if args.list_presets:
        from problem_data import PROBLEM_PRESETS

        print("Available problem presets:")
        for preset_name in PROBLEM_PRESETS:
            print(f"  - {preset_name}")
        return

    if args.formulation:
        print_formulation_summary()
        return

    if args.visualize_only:
        visualize_environment_only(args.config, args.scenario)
        return

    if args.shooting_animation:
        output_path = run_shooting_animation(
            args.config,
            args.scenario,
            verbose=not args.quiet,
        )
        print(f"Saved: {output_path}")
        return

    if args.quick_test:
        from experiments import run_quick_test

        run_quick_test(args.config, args.scenario)
        return

    if args.maze_benchmark:
        run_maze_benchmark(
            config_path=args.config,
            scenario_name=args.scenario,
            count=args.maze_count,
            maze_size=args.maze_size,
            knock_downs=args.maze_knock_downs,
            seed_start=args.maze_seed,
            wall_thickness=args.maze_wall_thickness,
            save_debug_geometry=args.maze_debug_geometry,
            verbose=not args.quiet,
        )
        return

    save_md = args.save_md  # None means "use config default"

    if args.scenario:
        run_single_scenario(args.config, args.scenario,
                            verbose=not args.quiet, save_md=save_md)
        return

    print_formulation_summary()
    run_all_experiments(args.config, verbose=not args.quiet, save_md=save_md)


if __name__ == "__main__":
    main()
