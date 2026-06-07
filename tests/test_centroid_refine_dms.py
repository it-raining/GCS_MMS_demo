"""Tests for CentroidRefineDMSConfig and CentroidRefineDMSSolver in optimizer.py."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from optimizer import (
    CentroidRefineDMSConfig, CentroidRefineDMSSolver, OptimizationResult,
    create_integrated_optimizer_from_config,
)
from dynamics import DoubleIntegratorDynamics


class TestCentroidRefineDMSConfig:
    def test_defaults(self):
        cfg = CentroidRefineDMSConfig()
        assert cfg.gamma_w == 1.0
        assert cfg.gamma_h == 0.64
        assert cfg.delta_safe == 0.02
        assert cfg.delta_extra == 0.01
        assert cfg.mode == "first_feasible"
        assert cfg.mu_min == 1e-5
        assert cfg.tau == 0.1

    def test_ablation_flags_all_true_by_default(self):
        cfg = CentroidRefineDMSConfig()
        assert cfg.use_centroid_cost is True
        assert cfg.use_interface_qp is True
        assert cfg.use_log_barrier is True
        assert cfg.use_barrier_continuation is True
        assert cfg.use_inexact_tolerance is True

    def test_ablation_flags_can_disable(self):
        cfg = CentroidRefineDMSConfig(
            use_centroid_cost=False, use_interface_qp=False,
            use_log_barrier=False, use_barrier_continuation=False,
            use_inexact_tolerance=False,
        )
        assert cfg.use_centroid_cost is False
        assert cfg.use_interface_qp is False

    def test_custom_values(self):
        cfg = CentroidRefineDMSConfig(gamma_w=2.0, n_int=30, time_limit_s=120.0)
        assert cfg.gamma_w == 2.0
        assert cfg.n_int == 30
        assert cfg.time_limit_s == 120.0


def _make_result(**kwargs):
    defaults = dict(success=False, path=[], path_regions=[],
                    total_cost=float('inf'), solve_time=0.0, n_paths_evaluated=0)
    defaults.update(kwargs)
    return OptimizationResult(**defaults)


class TestOptimizationResultNewFields:
    def test_new_fields_exist(self):
        r = _make_result()
        assert hasattr(r, 'certified_safety_margin')
        assert hasattr(r, 'lipschitz_gap')
        assert hasattr(r, 'safety_certification')
        assert hasattr(r, 'lb_geometric')
        assert hasattr(r, 'optimality_gap')
        assert hasattr(r, 'n_barrier_levels')
        assert hasattr(r, 'failure_log')

    def test_new_fields_defaults(self):
        r = _make_result()
        assert r.safety_certification == "NOT_SET"
        assert r.lb_geometric == 0.0
        assert r.optimality_gap == float("inf")
        assert r.n_barrier_levels == 0
        assert isinstance(r.failure_log, list)


class TestCentroidRefineDMSSolverDisclaimer:
    def test_disclaimer_text(self):
        disc = CentroidRefineDMSSolver.MANDATORY_DISCLAIMER
        assert "LB" in disc
        assert "geometric lower bound" in disc
        assert "No global optimality claim" in disc

    def test_disclaimer_in_empty_result(self):
        dyn = DoubleIntegratorDynamics(v_max=2.0)
        graph = MagicMock()
        cfg = CentroidRefineDMSConfig()
        solver = CentroidRefineDMSSolver(graph, dyn, cfg)
        result = solver._empty_result(0.0, 0, 0.0, [])
        assert result.global_optimality_claim == CentroidRefineDMSSolver.MANDATORY_DISCLAIMER


class TestFactoryDispatch:
    def test_factory_returns_centroid_solver(self):
        graph = MagicMock()
        graph.graph = MagicMock()
        dyn = DoubleIntegratorDynamics(v_max=2.0)
        config_dict = {
            'optimizer': {'solver_mode': 'centroid_refine_dms'},
            'centroid_refine_dms': {'gamma_w': 2.0, 'n_int': 15},
        }
        solver = create_integrated_optimizer_from_config(graph, dyn, config_dict)
        assert isinstance(solver, CentroidRefineDMSSolver)
        assert solver.config.gamma_w == 2.0
        assert solver.config.n_int == 15
