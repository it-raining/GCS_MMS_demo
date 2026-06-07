"""Tests for barrier_dms.py — schedule, failure classification, solver."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demo'))

import numpy as np
import pytest

from barrier_dms import (
    BarrierSchedule, FailureCode, FailureRecord,
    classify_dms_failure, BarrierDMSSolverConfig,
)


class TestBarrierSchedule:
    def test_schedule_decreasing(self):
        sch = BarrierSchedule.from_warm_start(f0=10.0, B0=5.0, alpha=0.1, tau=0.1, mu_min=1e-5)
        assert len(sch.mu_schedule) >= 1
        for i in range(len(sch.mu_schedule) - 1):
            assert sch.mu_schedule[i] > sch.mu_schedule[i + 1]

    def test_schedule_ends_at_or_below_mu_min(self):
        sch = BarrierSchedule.from_warm_start(f0=10.0, B0=5.0, alpha=0.1, tau=0.1, mu_min=1e-5)
        assert sch.mu_schedule[-1] <= 1e-5 + 1e-12

    def test_mu0_formula(self):
        f0, B0, alpha = 10.0, 5.0, 0.2
        sch = BarrierSchedule.from_warm_start(f0=f0, B0=B0, alpha=alpha, tau=0.1, mu_min=1e-5)
        expected_mu0 = float(np.clip(alpha * f0 / B0, 1e-5, 1.0))
        assert abs(sch.mu_schedule[0] - expected_mu0) < 1e-9

    def test_schedule_length_formula(self):
        import math
        alpha, tau, mu_min = 0.1, 0.1, 1e-5
        f0, B0 = 10.0, 5.0
        mu0 = float(np.clip(alpha * f0 / B0, mu_min, 1.0))
        N_expected = max(1, math.ceil(math.log(mu_min / mu0) / math.log(tau)))
        sch = BarrierSchedule.from_warm_start(f0=f0, B0=B0, alpha=alpha, tau=tau, mu_min=mu_min)
        assert len(sch.mu_schedule) == N_expected

    def test_large_B0_gives_small_mu0(self):
        sch_large = BarrierSchedule.from_warm_start(f0=1.0, B0=100.0, alpha=0.1, tau=0.1, mu_min=1e-5)
        sch_small = BarrierSchedule.from_warm_start(f0=1.0, B0=1.0, alpha=0.1, tau=0.1, mu_min=1e-5)
        assert sch_large.mu_schedule[0] < sch_small.mu_schedule[0]

    def test_tau_affects_length(self):
        sch_fast = BarrierSchedule.from_warm_start(f0=10.0, B0=1.0, alpha=0.1, tau=0.5, mu_min=1e-5)
        sch_slow = BarrierSchedule.from_warm_start(f0=10.0, B0=1.0, alpha=0.1, tau=0.1, mu_min=1e-5)
        assert len(sch_fast.mu_schedule) > len(sch_slow.mu_schedule)


class TestFailureClassification:
    def test_no_failure_returns_none(self):
        result = classify_dms_failure("Solve_Succeeded", 1e-8, 1e-8)
        assert result is None

    def test_restoration_detected(self):
        result = classify_dms_failure("Restoration_Failed", 1e-8, 1e-8)
        assert result == FailureCode.IPOPT_RESTORATION

    def test_max_iter_detected(self):
        result = classify_dms_failure("Maximum_Iterations_Exceeded", 1e-8, 1e-8)
        assert result == FailureCode.IPOPT_MAX_ITER

    def test_large_coupling_gap_detected(self):
        result = classify_dms_failure("Solve_Succeeded", 1e-8, 1.0)
        assert result == FailureCode.CONNECTION_GAP_LARGE


class TestBarrierDMSConfig:
    def test_defaults(self):
        cfg = BarrierDMSSolverConfig()
        assert cfg.n_int > 0
        assert cfg.delta_safe > 0
        assert cfg.mu_min < 1e-3
        assert cfg.tau < 1.0

    def test_custom_values(self):
        cfg = BarrierDMSSolverConfig(n_int=30, delta_safe=0.05, mu_min=1e-6)
        assert cfg.n_int == 30
        assert cfg.delta_safe == 0.05
        assert cfg.mu_min == 1e-6

    def test_failure_record_dataclass(self):
        fr = FailureRecord(path=('s', 'r0', 't'), code=FailureCode.IPOPT_MAX_ITER,
                           barrier_level=2, ipopt_status="Maximum_Iterations_Exceeded",
                           defect_norm=0.5, coupling_gap=0.1)
        assert fr.code == FailureCode.IPOPT_MAX_ITER
        assert fr.barrier_level == 2
