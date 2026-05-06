from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import casadi as ca
import matplotlib
import numpy as np

matplotlib.use("Agg")

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from convex_regions import create_regions_from_vertices_list  # noqa: E402
from dynamics import ControlParameterization, UnicycleModel, create_integration_bundle  # noqa: E402
from graph_builder import SOURCE, TARGET, build_region_graph  # noqa: E402
from optimizer import OptimizationConfig, PathNLPSolver  # noqa: E402
from shooting_animation import (  # noqa: E402
    ShootingIterationRecorder,
    create_shooting_convergence_animation,
    parse_recorder_snapshots,
    parse_shooting_snapshot,
)


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


class ShootingAnimationTests(unittest.TestCase):
    def test_recorder_captures_snapshots(self) -> None:
        x = ca.MX.sym("x")
        nlp = {"x": x, "f": (x - 1.0) ** 2, "g": x}
        recorder = ShootingIterationRecorder(
            name="test_recorder_captures_snapshots",
            nx=1,
            ng=1,
            record_every=1,
            max_snapshots=20,
        )
        solver = ca.nlpsol(
            "test_recorder_solver",
            "ipopt",
            nlp,
            {
                "ipopt.print_level": 0,
                "print_time": 0,
                "iteration_callback": recorder,
            },
        )
        solver(x0=[0.0], lbg=[-ca.inf], ubg=[ca.inf])

        self.assertGreaterEqual(len(recorder.snapshots), 1)
        self.assertEqual(np.asarray(recorder.snapshots[0]["x"]).shape, (1,))

    def test_parse_snapshot_matches_solution(self) -> None:
        regions = create_regions_from_vertices_list([_square(0.0, 0.0, 1.0, 1.0)])
        start_state = np.array([0.2, 0.2, 0.0], dtype=np.float64)
        goal_state = np.array([0.8, 0.2, 0.0], dtype=np.float64)
        graph = build_region_graph(regions, start_state[:2], goal_state[:2])
        dynamics = UnicycleModel()
        config = OptimizationConfig(
            n_integration_steps=8,
            n_mesh_points=3,
            n_control_segments=1,
            safety_margin=0.0,
            delta_min=0.01,
            delta_max=3.0,
            max_iter=500,
            tol=1e-7,
        )
        solver = PathNLPSolver(graph, dynamics, config)
        recorder = ShootingIterationRecorder(
            name="test_path_solver_recorder",
            record_every=1,
            max_snapshots=20,
        )
        result = solver.solve_path(
            [SOURCE, "R0", TARGET],
            start_state,
            goal_state,
            iteration_recorder=recorder,
        )
        self.assertTrue(result.success, result.solver_status)
        self.assertGreaterEqual(len(recorder.snapshots), 1)
        self.assertGreater(np.asarray(recorder.snapshots[0]["x"]).size, 0)
        parsed_recorder_snapshots = parse_recorder_snapshots(
            recorder,
            [0],
            dynamics.n_x,
            solver.control_param.n_w,
            dynamics,
            solver.control_param,
            solver.F_endpoint,
            config.n_integration_steps,
            n_mesh_points=config.n_mesh_points,
        )
        self.assertGreaterEqual(len(parsed_recorder_snapshots), 1)

        x_vec = np.concatenate(
            [
                result.entry_states[0],
                result.control_params[0],
                np.array([result.time_durations[0]], dtype=np.float64),
            ]
        )
        snapshot = parse_shooting_snapshot(
            x_vec,
            [0],
            dynamics.n_x,
            solver.control_param.n_w,
            dynamics,
            solver.control_param,
            solver.F_endpoint,
            config.n_integration_steps,
            n_mesh_points=config.n_mesh_points,
            cost=result.total_cost,
            iteration=99,
        )

        parsed_region = snapshot["regions"][0]
        np.testing.assert_allclose(parsed_region["s_minus"], result.entry_states[0])
        np.testing.assert_allclose(parsed_region["s_plus"], result.exit_states[0], atol=1e-10)
        self.assertEqual(parsed_region["trajectory"].shape, result.trajectories[0][0].shape)
        self.assertEqual(parsed_region["mesh_positions"].shape, (config.n_mesh_points, dynamics.n_pos))
        self.assertEqual(snapshot["iteration"], 99)
        self.assertAlmostEqual(snapshot["cost"], result.total_cost)

    def test_animation_creates_gif(self) -> None:
        regions = create_regions_from_vertices_list(
            [
                _square(0.0, 0.0, 1.0, 1.0),
                _square(0.75, 0.0, 1.75, 1.0),
            ]
        )
        start_state = np.array([0.2, 0.5, 0.0], dtype=np.float64)
        goal_state = np.array([1.6, 0.5, 0.0], dtype=np.float64)
        graph = build_region_graph(regions, start_state[:2], goal_state[:2])
        dynamics = UnicycleModel()
        control_param = ControlParameterization(
            n_u=dynamics.n_u,
            parameterization="constant",
            n_segments=1,
        )
        bundle = create_integration_bundle(dynamics, control_param, n_steps=6, n_mesh=3)
        path_regions = [0, 1]

        disconnected = np.array(
            [
                0.2, 0.5, 0.0, 0.2, 0.0, 1.0,
                1.2, 0.5, 0.0, 0.2, 0.0, 1.0,
            ],
            dtype=np.float64,
        )
        connected = np.array(
            [
                0.2, 0.5, 0.0, 0.8, 0.0, 1.0,
                1.0, 0.5, 0.0, 0.6, 0.0, 1.0,
            ],
            dtype=np.float64,
        )
        snapshots = [
            parse_shooting_snapshot(
                disconnected,
                path_regions,
                dynamics.n_x,
                control_param.n_w,
                dynamics,
                control_param,
                bundle.F_endpoint,
                n_integration_steps=6,
                n_mesh_points=3,
                cost=10.0,
                iteration=0,
            ),
            parse_shooting_snapshot(
                connected,
                path_regions,
                dynamics.n_x,
                control_param.n_w,
                dynamics,
                control_param,
                bundle.F_endpoint,
                n_integration_steps=6,
                n_mesh_points=3,
                cost=1.0,
                iteration=1,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "shooting.gif"
            create_shooting_convergence_animation(
                snapshots,
                graph,
                (0.0, 1.75, 0.0, 1.0),
                start_state,
                goal_state,
                str(filename),
                fps=2,
                hold_first_frames=0,
                hold_last_frames=0,
                max_animation_frames=2,
            )

            self.assertTrue(filename.exists())
            self.assertGreater(filename.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
