"""Tests for warmstart.py IVP warm-start generation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))

import numpy as np
import pytest
from unittest.mock import MagicMock

from warmstart import WarmStartConfig, WarmStart, generate_warm_start_ivp
from dynamics import DoubleIntegratorDynamics, UnicycleModel


def _make_z(starts, goals, n_regions):
    m = n_regions
    z = np.zeros((m + 2, 2))
    z[0] = starts
    z[m + 1] = goals
    for i in range(m):
        tau = (i + 1) / (m + 1)
        z[i + 1] = starts * (1 - tau) + goals * tau
    return z


class TestWarmStartStructure:
    def test_di_returns_warmstart(self):
        dyn = DoubleIntegratorDynamics(v_max=2.0)
        cfg = WarmStartConfig(n_int=5)
        z = _make_z(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 2)
        ws = generate_warm_start_ivp(z, [0, 1], dyn, cfg)
        assert isinstance(ws, WarmStart)
        assert len(ws.x_nodes) == 2
        assert len(ws.u_list) == 2
        assert len(ws.delta_list) == 2

    def test_di_node_shape(self):
        dyn = DoubleIntegratorDynamics(v_max=2.0)
        cfg = WarmStartConfig(n_int=10)
        z = _make_z(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 1)
        ws = generate_warm_start_ivp(z, [0], dyn, cfg)
        assert ws.x_nodes[0].shape == (11, 4)

    def test_di_strict_interior(self):
        dyn = DoubleIntegratorDynamics(v_max=2.0)
        cfg = WarmStartConfig(n_int=5)
        z = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        ws = generate_warm_start_ivp(z, [0], dyn, cfg)
        seg = ws.x_nodes[0]
        xs = seg[:, 0]
        assert float(xs.min()) >= -0.01
        assert float(xs.max()) <= 1.01

    def test_duration_bounds(self):
        dyn = DoubleIntegratorDynamics(v_max=2.0)
        cfg = WarmStartConfig(n_int=5, delta_min=0.1, delta_max=5.0)
        z = _make_z(np.array([0.0, 0.0]), np.array([3.0, 0.0]), 1)
        ws = generate_warm_start_ivp(z, [0], dyn, cfg)
        assert cfg.delta_min <= ws.delta_list[0] <= cfg.delta_max

    def test_di_zero_acceleration(self):
        dyn = DoubleIntegratorDynamics(v_max=2.0)
        cfg = WarmStartConfig(n_int=5)
        z = _make_z(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 1)
        ws = generate_warm_start_ivp(z, [0], dyn, cfg)
        np.testing.assert_allclose(ws.u_list[0], np.zeros(2), atol=1e-9)

    def test_di_constant_velocity(self):
        dyn = DoubleIntegratorDynamics(v_max=3.0)
        cfg = WarmStartConfig(n_int=4, v_nom_fraction=0.5)
        z = _make_z(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 1)
        ws = generate_warm_start_ivp(z, [0], dyn, cfg)
        vel_x = ws.x_nodes[0][:, 2]
        assert np.all(vel_x >= -3.0 - 1e-6) and np.all(vel_x <= 3.0 + 1e-6)

    def test_unknown_dynamics_raises(self):
        dyn = MagicMock(spec=[])
        cfg = WarmStartConfig(n_int=5)
        z = _make_z(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 1)
        with pytest.raises(NotImplementedError):
            generate_warm_start_ivp(z, [0], dyn, cfg)
