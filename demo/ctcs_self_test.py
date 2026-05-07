"""
Small CTCS safety-certificate self-tests.

These tests exercise the RK4 accumulated path-constraint violation diagnostic
without claiming an exact continuous-time guarantee.
"""

import numpy as np

from convex_regions import ConvexRegion
from dynamics import UnicycleModel
from shooting import ShootingBlock, ShootingBlockConfig


def _unit_square() -> ConvexRegion:
    return ConvexRegion(
        vertices=np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]),
        index=0,
        label="unit_square",
    )


def _make_block() -> ShootingBlock:
    return ShootingBlock(
        region=_unit_square(),
        dynamics=UnicycleModel(),
        config=ShootingBlockConfig(
            region_index=0,
            n_integration_steps=8,
            n_mesh_points=3,
            n_control_segments=1,
            safety_margin=0.0,
            safety_mode="both",
            dense_check_points=200,
        ),
    )


def test_inside_square() -> None:
    block = _make_block()
    s_minus = np.array([0.5, 0.5, 0.0])
    w = np.array([0.0, 0.0])
    diag = block.compute_ctcs_violation_integral(s_minus, w, delta=1.0)
    assert diag["eta_end"] <= 1e-10, diag


def test_obvious_violation() -> None:
    block = _make_block()
    s_minus = np.array([0.5, 1.2, 0.0])
    w = np.array([0.0, 0.0])
    diag = block.compute_ctcs_violation_integral(s_minus, w, delta=1.0)
    assert diag["eta_end"] > 1e-3, diag
    assert diag["max_stage_violation"] > 0.0, diag


def test_mesh_vs_ctcs_distinction() -> None:
    region = _unit_square()

    def point(tau: float) -> np.ndarray:
        # Endpoints and tau=0.5 are feasible, but the curve rises above y=1
        # between those sampled points.
        return np.array([0.5, 0.5 + 0.9 * np.sin(2.0 * np.pi * tau)])

    mesh_tau = np.array([0.0, 0.5, 1.0])
    mesh_max = max(
        float(np.max(region.A @ point(tau) - region.b))
        for tau in mesh_tau
    )
    stage_tau = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    stage_lambda = sum(
        float(np.sum(np.maximum(0.0, region.A @ point(tau) - region.b) ** 2))
        for tau in stage_tau
    )

    assert mesh_max <= 1e-10
    assert stage_lambda > 0.0


def run_self_tests() -> None:
    test_inside_square()
    test_obvious_violation()
    test_mesh_vs_ctcs_distinction()
    print("CTCS self-tests passed")


if __name__ == "__main__":
    run_self_tests()
