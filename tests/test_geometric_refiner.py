"""Tests for geometric_refiner.py interface QP."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))

import numpy as np
import pytest
from unittest.mock import MagicMock

from geometric_refiner import (
    InterfaceQPConfig, NarrowInterfaceError, InterfaceQPInfeasible,
    solve_interface_refinement,
)


def _make_graph(regions):
    graph = MagicMock()
    graph.regions = regions
    return graph


def _box_region(cx, cy, r):
    region = MagicMock()
    region.A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    region.b = np.array([cx + r, -cx + r, cy + r, -cy + r], dtype=float)
    return region


class TestInterfaceQPShape:
    def test_output_shape_single_region(self):
        r0 = _box_region(0.0, 0.0, 1.0)
        graph = _make_graph([r0])
        q_start = np.array([0.0, 0.0])
        q_goal  = np.array([0.0, 0.0])
        cfg = InterfaceQPConfig(delta_safe=0.01, delta_extra=0.005)
        z = solve_interface_refinement(graph, [0], q_start, q_goal, cfg)
        assert z.shape == (3, 2), f"Expected (3,2) got {z.shape}"

    def test_output_shape_two_regions(self):
        r0 = _box_region(-0.3, 0.0, 0.8)
        r1 = _box_region(0.3, 0.0, 0.8)
        graph = _make_graph([r0, r1])
        q_start = np.array([-0.5, 0.0])
        q_goal  = np.array([0.5, 0.0])
        cfg = InterfaceQPConfig(delta_safe=0.02, delta_extra=0.01)
        z = solve_interface_refinement(graph, [0, 1], q_start, q_goal, cfg)
        assert z.shape == (4, 2)

    def test_fixed_endpoints(self):
        r0 = _box_region(0.0, 0.0, 1.0)
        graph = _make_graph([r0])
        q_start = np.array([-0.4, 0.0])
        q_goal  = np.array([0.4, 0.0])
        cfg = InterfaceQPConfig(delta_safe=0.01, delta_extra=0.005)
        z = solve_interface_refinement(graph, [0], q_start, q_goal, cfg)
        np.testing.assert_allclose(z[0], q_start, atol=1e-8)
        np.testing.assert_allclose(z[-1], q_goal, atol=1e-8)

    def test_strict_interior(self):
        r0 = _box_region(0.0, 0.0, 1.0)
        graph = _make_graph([r0])
        q_start = np.array([-0.3, 0.0])
        q_goal  = np.array([0.3, 0.0])
        cfg = InterfaceQPConfig(delta_safe=0.02, delta_extra=0.01)
        margin = cfg.delta_safe + cfg.delta_extra
        z = solve_interface_refinement(graph, [0], q_start, q_goal, cfg)
        for i in range(1, len(z) - 1):
            slack = r0.b - r0.A @ z[i]
            assert np.all(slack >= margin - 1e-4), f"Interior point {i} not strictly inside"

    def test_narrow_interface_raises(self):
        # Two regions that barely overlap — radius < margin
        r0 = _box_region(-0.5, 0.0, 0.51)
        r1 = _box_region(0.5, 0.0, 0.51)
        graph = _make_graph([r0, r1])
        q_start = np.array([-0.5, 0.0])
        q_goal  = np.array([0.5, 0.0])
        cfg = InterfaceQPConfig(delta_safe=0.05, delta_extra=0.05)
        with pytest.raises(NarrowInterfaceError):
            solve_interface_refinement(graph, [0, 1], q_start, q_goal, cfg)
