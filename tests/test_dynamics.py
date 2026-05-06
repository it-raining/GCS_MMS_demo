from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from dynamics import (
    ControlParameterization,
    RK4Integrator,
    UnicycleModel,
    create_integration_bundle,
)


class DynamicsTests(unittest.TestCase):
    def test_rk4_matches_solve_ivp_for_constant_control(self) -> None:
        dynamics = UnicycleModel()
        control_param = ControlParameterization(
            n_u=dynamics.n_u,
            parameterization="constant",
            n_segments=1,
        )
        integrator = RK4Integrator(dynamics, control_param, n_steps=200)

        x0 = np.array([0.1, -0.2, 0.3], dtype=np.float64)
        w = np.array([0.7, 0.2], dtype=np.float64)
        delta = 1.2

        rk4_result = integrator.integrate(x0, w, delta)

        def rhs(_time: float, state: np.ndarray) -> np.ndarray:
            return dynamics.f(state, w)

        ivp_result = solve_ivp(
            rhs,
            (0.0, delta),
            x0,
            rtol=1e-10,
            atol=1e-12,
        )

        np.testing.assert_allclose(rk4_result, ivp_result.y[:, -1], atol=1e-8)

    def test_casadi_bundle_matches_numpy_integrator(self) -> None:
        dynamics = UnicycleModel()
        control_param = ControlParameterization(
            n_u=dynamics.n_u,
            parameterization="piecewise_constant",
            n_segments=2,
        )
        n_steps = 20
        bundle = create_integration_bundle(
            dynamics,
            control_param,
            n_steps=n_steps,
            n_mesh=5,
        )
        integrator = RK4Integrator(dynamics, control_param, n_steps=n_steps)

        x0 = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        w = np.array([0.5, 0.1, 0.4, -0.2], dtype=np.float64)
        delta = 0.9

        casadi_result = np.array(bundle.F_endpoint(x0, w, delta), dtype=float).reshape(-1)
        numpy_result = integrator.integrate(x0, w, delta)
        np.testing.assert_allclose(casadi_result, numpy_result, atol=1e-10)

        mesh = np.array(bundle.mesh_sampler(x0, w, delta), dtype=float)
        self.assertEqual(mesh.shape, (5, dynamics.n_pos))

        cost = float(bundle.local_cost_fn(x0, w, delta, 1.0, 1.0, 1.0, 0.1))
        self.assertGreater(cost, 0.0)

        cbf_positions, cbf_velocities = bundle.cbf_sampler(x0, w, delta)
        self.assertEqual(np.array(cbf_positions, dtype=float).shape, (5, dynamics.n_pos))
        self.assertEqual(np.array(cbf_velocities, dtype=float).shape, (5, dynamics.n_pos))

    def test_piecewise_control_uses_exact_endpoint_segments(self) -> None:
        dynamics = UnicycleModel()
        control_param = ControlParameterization(
            n_u=dynamics.n_u,
            parameterization="piecewise_constant",
            n_segments=2,
        )
        w = np.array([0.2, -0.3, 1.1, 0.7], dtype=np.float64)

        np.testing.assert_allclose(control_param.evaluate(0.0, w), w[:2])
        np.testing.assert_allclose(control_param.evaluate(1.0, w), w[2:])

        u0 = np.array(control_param.evaluate_casadi(0.0, w), dtype=float).reshape(-1)
        u1 = np.array(control_param.evaluate_casadi(1.0, w), dtype=float).reshape(-1)
        np.testing.assert_allclose(u0, w[:2])
        np.testing.assert_allclose(u1, w[2:])


if __name__ == "__main__":
    unittest.main()
