"""
test_crd_default.py — Quick end-to-end test of CentroidRefineDMS on the default map.

Logs everything to /tmp/crd_debug.log for post-mortem inspection.

Run from the demo/ directory:
    python test_crd_default.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ── logging setup ──────────────────────────────────────────────────────────────
LOG_FILE = Path("/tmp/crd_debug.log")
LOG_FILE.unlink(missing_ok=True)

root_log = logging.getLogger()
root_log.setLevel(logging.DEBUG)
fh = logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"))
root_log.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("[%(levelname)-7s] %(message)s"))
root_log.addHandler(ch)

log = logging.getLogger("crd_test")

# ── path setup ─────────────────────────────────────────────────────────────────
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)
CONFIG_PATH = os.path.join(DEMO_DIR, "config.yaml")

# ── import after path is set ───────────────────────────────────────────────────
from app_config import DemoConfig
from scenario_builder import prepare_scenario


# ── patch CentroidRefineDMSSolver to inject per-stage logging ─────────────────
def _patch_crd_solver():
    import optimizer as _opt
    import geometric_refiner as _qp
    import barrier_dms as _bdms

    OrigSolve = _opt.CentroidRefineDMSSolver.solve

    def patched_solve(self, start_state, goal_state, verbose=False):
        log.info("=== CentroidRefineDMSSolver.solve() START ===")
        log.info(f"  start={start_state}, goal={goal_state}")
        log.info(f"  config={self.config}")
        log.info(f"  graph regions: {len(self.graph.regions)}, edges: {len(self.graph.region_edges)}")
        result = OrigSolve(self, start_state, goal_state, verbose=verbose)
        log.info(f"=== CentroidRefineDMSSolver.solve() END: success={result.success}, "
                 f"paths_evaluated={result.n_paths_evaluated}, cost={result.total_cost:.4f}")
        if result.failure_log:
            log.info(f"  Failure log ({len(result.failure_log)} entries):")
            for i, fl in enumerate(result.failure_log[:20]):
                log.info(f"    [{i}] stage={fl.get('stage','?')} reason={fl.get('reason','?')}")
            if len(result.failure_log) > 20:
                log.info(f"    ... and {len(result.failure_log)-20} more")
        return result

    _opt.CentroidRefineDMSSolver.solve = patched_solve

    # Patch interface QP to log each attempt
    OrigQP = _qp.solve_interface_refinement

    def patched_qp(graph, path_regions, q_start, q_goal, config):
        log.debug(f"  Interface QP: path_regions={path_regions}, "
                  f"delta_safe={config.delta_safe:.4f}, delta_extra={config.delta_extra:.4f}")
        from graph_builder import chebyshev_center
        for i in range(len(path_regions) - 1):
            ri = path_regions[i]
            ri1 = path_regions[i + 1]
            A_int = np.vstack([graph._regions_by_index[ri].A, graph._regions_by_index[ri1].A])
            b_int = np.concatenate([graph._regions_by_index[ri].b, graph._regions_by_index[ri1].b])
            try:
                _, rho = chebyshev_center(A_int, b_int)
                margin = config.delta_safe + config.delta_extra
                log.debug(f"    Interface {i}/{i+1} (regions {ri}/{ri1}): rho={rho:.4f}, "
                          f"required={margin:.4f}, OK={rho >= margin}")
            except Exception as ex:
                log.debug(f"    Interface {i}/{i+1}: chebyshev LP failed: {ex}")
        try:
            result = OrigQP(graph, path_regions, q_start, q_goal, config)
            log.debug(f"  Interface QP: SUCCESS, waypoints shape={result.shape}")
            return result
        except Exception as ex:
            log.debug(f"  Interface QP: FAILED ({type(ex).__name__}: {ex})")
            raise

    _qp.solve_interface_refinement = patched_qp

    # Patch barrier solver to log each level
    OrigBarrierSolve = _bdms.BarrierDMSSolver.solve

    def patched_barrier_solve(
        self, path_regions, anchor_points, warm_start=None, **kwargs
    ):
        log.debug(f"  BarrierDMS: path_regions={path_regions}, "
                  f"anchors shape={anchor_points.shape}")
        t0 = time.time()
        result = OrigBarrierSolve(
            self, path_regions, anchor_points, warm_start, **kwargs
        )
        elapsed = time.time() - t0
        log.debug(f"  BarrierDMS: success={result.success}, cost={result.total_cost:.4f}, "
                  f"time={elapsed:.3f}s, lipschitz_gap={result.lipschitz_gap:.4f}")
        log.debug(f"  BarrierDMS: safety_margin={result.min_safety_margin:.4f}, "
                  f"certified={result.certified_safety_margin:.4f}")
        for lr in result.barrier_level_results:
            log.debug(f"    level mu={lr.get('mu')}: success={lr.get('success')}, "
                      f"cost={lr.get('cost', 'N/A')}, error={lr.get('error', None)}")
        return result

    _bdms.BarrierDMSSolver.solve = patched_barrier_solve

    OrigBuildLevel = _bdms.BarrierDMSSolver._solve_barrier_level

    def patched_build_level(self, path_regions, anchor_points, warm_start,
                             delta_safe, mu, x_init, tol, **kwargs):
        log.debug(f"    _solve_barrier_level mu={mu:.4f}, tol={tol:.2e}, "
                  f"delta_safe={delta_safe:.4f}, "
                  f"warm={'from_prev' if x_init is not None else 'default'}")
        t0 = time.time()
        result, x_opt = OrigBuildLevel(self, path_regions, anchor_points, warm_start,
                                        delta_safe, mu, x_init, tol, **kwargs)
        elapsed = time.time() - t0
        log.debug(f"    Level done: success={result.success}, "
                  f"status={result.solver_status!r}, time={elapsed:.3f}s, "
                  f"cost={result.total_cost:.4f}")
        return result, x_opt

    _bdms.BarrierDMSSolver._solve_barrier_level = patched_build_level

    log.info("Patches applied to CentroidRefineDMSSolver, InterfaceQP, BarrierDMSSolver")


def run_test():
    log.info("Loading config from %s", CONFIG_PATH)
    config = DemoConfig(config_path=CONFIG_PATH)

    scenario_name = "default"
    log.info("Building scenario: %s", scenario_name)

    _patch_crd_solver()

    t_setup = time.time()
    try:
        import copy
        orig_raw = config.raw
        patched_raw = copy.deepcopy(orig_raw)
        patched_raw.setdefault("optimizer", {})["solver_mode"] = "centroid_refine_dms"
        config.raw = patched_raw

        log.info("  problem_preset for 'default' scenario: %s",
                 patched_raw["scenarios"]["default"].get("problem_preset", "default"))

        prepared = prepare_scenario(config, scenario_name)
        setup_time = time.time() - t_setup

        log.info("Environment setup complete in %.3fs", setup_time)
        log.info("  Regions: %d", len(prepared.graph.regions))
        log.info("  Edges: %d", len(prepared.graph.region_edges))

        for ri, region in prepared.graph._regions_by_index.items():
            log.debug(f"  Region {ri}: {region.A.shape[0]} halfplanes, "
                      f"vertices={len(region.vertices)}")

        start_state = prepared.resolved.start_state
        goal_state = prepared.resolved.goal_state
        log.info("  start_state=%s", start_state)
        log.info("  goal_state=%s", goal_state)

        log.info("=== Solving with CentroidRefineDMS ===")
        t_solve = time.time()
        result = prepared.optimizer.solve(start_state, goal_state)
        solve_time = time.time() - t_solve

        log.info("")
        log.info("=== RESULT SUMMARY ===")
        log.info("  Success: %s", result.success)
        log.info("  Solver status: %s", result.solver_status)
        log.info("  Paths evaluated: %d", result.n_paths_evaluated)
        log.info("  Solve time: %.3fs", solve_time)
        log.info("  Total cost: %.4f", result.total_cost)
        log.info("  Formulation: %s", result.formulation_mode)
        log.info("  Safety: sampled=%.4f, certified=%.4f, lipschitz_gap=%.4f",
                 result.min_safety_margin, result.certified_safety_margin, result.lipschitz_gap)
        log.info("  Safety certification: %s", result.safety_certification)
        if result.success:
            log.info("  Path: %s", " -> ".join(result.path))
            log.info("  Durations: %s", {k: f"{v:.3f}" for k, v in result.time_durations.items()})
            log.info("  Barrier levels used: %d", result.n_barrier_levels)
        log.info("  Global optimality claim: %s", result.global_optimality_claim)
        log.info("  Failure log entries: %d", len(result.failure_log))

        config.raw = orig_raw
        return result

    except Exception:
        log.exception("Test failed with exception")
        config.raw = orig_raw
        raise
    finally:
        log.info("Log written to %s", LOG_FILE)


if __name__ == "__main__":
    run_test()
