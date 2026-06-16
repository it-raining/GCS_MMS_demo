"""Diagnostic script: instrument CentroidRefineDMS internals to find why
the 'maze' scenario fails to find a feasible path.

Not part of the production demo -- ad hoc debugging tool.
"""
from __future__ import annotations

import sys
import time
import numpy as np

sys.path.insert(0, ".")

from app_config import DemoConfig
from scenario_builder import prepare_scenario
import optimizer as _opt
import geometric_refiner as _qp
import barrier_dms as _bdms
from graph_builder import chebyshev_center, k_shortest_paths_generator

N_PATHS_TO_PROBE = 25

narrow_interface_details = []
qp_success = 0
qp_fail = 0

_orig_qp = _qp.solve_interface_refinement


def patched_solve_interface_refinement(graph, path_regions, q_start, q_goal, config):
    global qp_success, qp_fail
    for i in range(len(path_regions) - 1):
        ri, ri1 = path_regions[i], path_regions[i + 1]
        A_int = np.vstack([graph._regions_by_index[ri].A, graph._regions_by_index[ri1].A])
        b_int = np.concatenate([graph._regions_by_index[ri].b, graph._regions_by_index[ri1].b])
        try:
            _, rho = chebyshev_center(A_int, b_int)
        except Exception:
            rho = float("nan")
        required = config.delta_safe + config.delta_extra
        if rho < required or np.isnan(rho):
            narrow_interface_details.append((ri, ri1, rho, required))
    try:
        result = _orig_qp(graph, path_regions, q_start, q_goal, config)
        qp_success += 1
        return result
    except Exception:
        qp_fail += 1
        raise


def main():
    config = DemoConfig(config_path="config.yaml")
    prepared = prepare_scenario(config, "maze")
    graph = prepared.graph
    print(f"Regions: {len(graph.regions)}  Edges: {len(graph.region_edges)}")

    cfg = prepared.optimizer.config
    print(f"CRD config: delta_safe={cfg.delta_safe} delta_extra={cfg.delta_extra} "
          f"n_int={cfg.n_int} delta_max={cfg.delta_max} mode={cfg.mode} "
          f"barrier_levels={cfg.barrier_levels} time_limit_s={cfg.time_limit_s}")

    dyn = prepared.optimizer.dynamics
    h_max = cfg.delta_max / cfg.n_int
    all_A = np.vstack([r.A for r in graph.regions])
    L_s = dyn.compute_lipschitz_bound(all_A)
    lip_gap = L_s * h_max / 2
    print(f"h_max={h_max:.4f}  L_s={L_s:.4f}  lipschitz_gap={lip_gap:.4f}  "
          f"=> required_delta_safe ~= {max(cfg.delta_safe, lip_gap + cfg.epsilon_final):.4f}")

    from geometric_refiner import InterfaceQPConfig

    SOURCE, TARGET = "source", "target"
    gen = k_shortest_paths_generator(graph, SOURCE, TARGET)

    required_delta_safe = max(cfg.delta_safe, lip_gap + cfg.epsilon_final)

    t0 = time.time()
    n = 0
    for path in gen:
        if n >= N_PATHS_TO_PROBE:
            break
        path_regions = prepared.optimizer._extract_region_indices(path)
        if not path_regions:
            continue
        n += 1
        print(f"\n--- path {n}: {len(path_regions)} regions ---")
        try:
            qp_cfg = InterfaceQPConfig(delta_safe=required_delta_safe, delta_extra=cfg.delta_extra, lambda_s=cfg.alpha_s)
            z = patched_solve_interface_refinement(
                graph, path_regions, prepared.resolved.start_state[:2],
                prepared.resolved.goal_state[:2], qp_cfg,
            )
            print(f"  QP ok, z shape={z.shape}")
        except Exception as e:
            print(f"  QP FAILED: {type(e).__name__}: {e}")
            continue

    elapsed = time.time() - t0
    print(f"\n=== SUMMARY over {n} probed paths in {elapsed:.2f}s ===")
    print(f"QP success={qp_success}  QP fail={qp_fail}")
    print(f"Narrow-interface violations recorded: {len(narrow_interface_details)}")
    for ri, ri1, rho, req in narrow_interface_details[:15]:
        print(f"   regions {ri}<->{ri1}: rho={rho:.4f} required={req:.4f}")


def main_single_path_timing():
    """Time each stage (QP, warm-start+Lipschitz loop, barrier DMS) for the
    first candidate path only, to localize where the solver stalls."""
    config = DemoConfig(config_path="config.yaml")
    prepared = prepare_scenario(config, "maze")
    graph = prepared.graph
    cfg = prepared.optimizer.config
    print(f"CRD config: delta_safe={cfg.delta_safe} n_int={cfg.n_int} "
          f"delta_max={cfg.delta_max} mode={cfg.mode} barrier_levels={cfg.barrier_levels}")

    SOURCE, TARGET = "source", "target"
    gen = k_shortest_paths_generator(graph, SOURCE, TARGET)
    path = next(gen)
    path_regions = prepared.optimizer._extract_region_indices(path)
    print(f"First path: {len(path_regions)} regions")

    from geometric_refiner import InterfaceQPConfig
    from warmstart import build_centroid_warmstart, WarmStartConfig
    from constraint_layers import compute_lipschitz_safety_gap
    import barrier_dms as bdms_mod

    x_start = prepared.resolved.start_state[:2]
    x_goal = prepared.resolved.goal_state[:2]

    t0 = time.time()
    qp_cfg = InterfaceQPConfig(delta_safe=cfg.delta_safe, delta_extra=cfg.delta_extra, lambda_s=cfg.alpha_s)
    z = _orig_qp(graph, path_regions, x_start, x_goal, qp_cfg)
    print(f"[t={time.time()-t0:.2f}s] Stage B (QP) done, z shape={z.shape}")

    ws_cfg = WarmStartConfig(v_nom_fraction=cfg.v_nom_fraction, n_int=cfg.n_int)
    warm_start = build_centroid_warmstart(
        graph=graph, path_regions=path_regions, anchor_points=z,
        dynamics=prepared.optimizer.dynamics, config=ws_cfg,
    )
    print(f"[t={time.time()-t0:.2f}s] Stage C (warm-start) done")

    delta_safe_barrier = cfg.delta_safe
    for it in range(4):
        delta_arr = np.array([warm_start[ri]['delta'] for ri in path_regions], dtype=float)
        lip_gap = compute_lipschitz_safety_gap(
            prepared.optimizer.dynamics, path_regions, graph, delta_arr, cfg.n_int
        )
        required_delta_safe = max(cfg.delta_safe, lip_gap + cfg.epsilon_final)
        print(f"  [iter {it}] lip_gap={lip_gap:.4f} required_delta_safe={required_delta_safe:.4f}")
        if required_delta_safe <= delta_safe_barrier + 1e-9:
            delta_safe_barrier = required_delta_safe
            break
        delta_safe_barrier = required_delta_safe
        barrier_qp_cfg = InterfaceQPConfig(delta_safe=delta_safe_barrier, delta_extra=cfg.delta_extra, lambda_s=cfg.alpha_s)
        z = _orig_qp(graph, path_regions, x_start, x_goal, barrier_qp_cfg)
        warm_start = build_centroid_warmstart(
            graph=graph, path_regions=path_regions, anchor_points=z,
            dynamics=prepared.optimizer.dynamics, config=ws_cfg,
        )
    print(f"[t={time.time()-t0:.2f}s] Lipschitz correction loop done, delta_safe_barrier={delta_safe_barrier:.4f}")

    from optimizer import BarrierDMSConfig
    barrier_cfg = BarrierDMSConfig(
        n_int=cfg.n_int, delta_safe=delta_safe_barrier, delta_extra=cfg.delta_extra,
        barrier_levels=cfg.barrier_levels, epsilon_final=cfg.epsilon_final,
        time_limit_s=60.0, delta_min=cfg.delta_min, delta_max=cfg.delta_max,
        n_control_segments=cfg.n_control_segments,
        w_T=cfg.w_T, w_L=cfg.w_L, w_U=cfg.w_U, w_S=cfg.w_S,
    )
    print(f"[t={time.time()-t0:.2f}s] Building BarrierDMSSolver (NLP construction)...")
    barrier_solver = bdms_mod.BarrierDMSSolver(graph, prepared.optimizer.dynamics, barrier_cfg)
    print(f"[t={time.time()-t0:.2f}s] Solver built, calling .solve()...")
    result = barrier_solver.solve(
        path_regions, z, warm_start,
        start_state=prepared.resolved.start_state, goal_state=prepared.resolved.goal_state,
    )
    print(f"[t={time.time()-t0:.2f}s] DONE: success={result.success} status={result.solver_status}")


def main_full_run():
    """Run the real solver loop on 'maze' with per-candidate / per-barrier-level
    progress logged as it happens (flushed), so a killed/timed-out run still
    leaves a readable trail."""
    config = DemoConfig(config_path="config.yaml")
    prepared = prepare_scenario(config, "maze")
    cfg = prepared.optimizer.config
    print(f"CRD config: delta_safe={cfg.delta_safe} n_int={cfg.n_int} "
          f"delta_max={cfg.delta_max} mode={cfg.mode} time_limit_s={cfg.time_limit_s} "
          f"barrier_levels={cfg.barrier_levels}", flush=True)

    OrigBuildLevel = _bdms.BarrierDMSSolver._solve_barrier_level

    def patched_build_level(self, path_regions, anchor_points, warm_start,
                             delta_safe, mu, x_init, tol, **kwargs):
        t0 = time.time()
        print(f"    [level mu={mu:.4f}] solving... (path len={len(path_regions)})", flush=True)
        result, x_opt = OrigBuildLevel(self, path_regions, anchor_points, warm_start,
                                        delta_safe, mu, x_init, tol, **kwargs)
        print(f"    [level mu={mu:.4f}] done in {time.time()-t0:.2f}s "
              f"success={result.success} status={result.solver_status!r}", flush=True)
        return result, x_opt

    _bdms.BarrierDMSSolver._solve_barrier_level = patched_build_level

    n_candidates = [0]
    orig_extract = _opt.CentroidRefineDMSSolver._extract_region_indices

    def patched_extract(self, path):
        n_candidates[0] += 1
        print(f"--- candidate #{n_candidates[0]}: {path[:3]}...{path[-2:] if len(path) > 3 else ''} "
              f"(len={len(path)}) ---", flush=True)
        return orig_extract(self, path)

    _opt.CentroidRefineDMSSolver._extract_region_indices = patched_extract

    t0 = time.time()
    result = prepared.optimizer.solve(prepared.resolved.start_state, prepared.resolved.goal_state)
    print(f"\n=== FINAL: success={result.success} status={result.solver_status} "
          f"paths_evaluated={result.n_paths_evaluated} elapsed={time.time()-t0:.2f}s ===", flush=True)


def main_explicit_path():
    """Run the full Stage B-E pipeline on a known margin-clear path (found via
    networkx on the rho-filtered graph) to validate the delta_safe/n_int tuning
    independent of how long Yen's k-shortest search takes to find it."""
    config = DemoConfig(config_path="config.yaml")
    prepared = prepare_scenario(config, "maze")
    graph = prepared.graph
    cfg = prepared.optimizer.config

    explicit_path = ['source', 'R56', 'R99', 'R119', 'R120', 'R139', 'R138', 'R121',
                      'R100', 'R97', 'R98', 'R96', 'R118', 'R136', 'R137', 'R156',
                      'R155', 'R113', 'R93', 'R76', 'R117', 'R116', 'R95', 'R159',
                      'R160', 'R143', 'R141', 'R142', 'R122', 'R140', 'R157', 'R124',
                      'R123', 'R78', 'target']
    path_regions = prepared.optimizer._extract_region_indices(explicit_path)
    print(f"Explicit path: {len(path_regions)} regions")

    from geometric_refiner import InterfaceQPConfig
    from warmstart import build_centroid_warmstart, WarmStartConfig
    from constraint_layers import compute_lipschitz_safety_gap
    from barrier_dms import BarrierDMSConfig
    import barrier_dms as bdms_mod

    x_start = prepared.resolved.start_state[:2]
    x_goal = prepared.resolved.goal_state[:2]
    t0 = time.time()

    qp_cfg = InterfaceQPConfig(delta_safe=cfg.delta_safe, delta_extra=cfg.delta_extra, lambda_s=cfg.alpha_s)
    z = _orig_qp(graph, path_regions, x_start, x_goal, qp_cfg)
    print(f"[t={time.time()-t0:.2f}s] Stage B (QP) OK")

    ws_cfg = WarmStartConfig(v_nom_fraction=cfg.v_nom_fraction, n_int=cfg.n_int)
    warm_start = build_centroid_warmstart(graph=graph, path_regions=path_regions, anchor_points=z,
                                           dynamics=prepared.optimizer.dynamics, config=ws_cfg)
    delta_safe_barrier = cfg.delta_safe
    for it in range(4):
        delta_arr = np.array([warm_start[ri]['delta'] for ri in path_regions], dtype=float)
        delta_arr = np.minimum(delta_arr, cfg.delta_max)
        lip_gap = compute_lipschitz_safety_gap(prepared.optimizer.dynamics, path_regions, graph, delta_arr, cfg.n_int)
        required_delta_safe = max(cfg.delta_safe, lip_gap + cfg.epsilon_final)
        print(f"  [iter {it}] lip_gap={lip_gap:.4f} required_delta_safe={required_delta_safe:.4f}")
        if required_delta_safe <= delta_safe_barrier + 1e-9:
            delta_safe_barrier = required_delta_safe
            break
        delta_safe_barrier = required_delta_safe
        barrier_qp_cfg = InterfaceQPConfig(delta_safe=delta_safe_barrier, delta_extra=cfg.delta_extra, lambda_s=cfg.alpha_s)
        z = _orig_qp(graph, path_regions, x_start, x_goal, barrier_qp_cfg)
        warm_start = build_centroid_warmstart(graph=graph, path_regions=path_regions, anchor_points=z,
                                               dynamics=prepared.optimizer.dynamics, config=ws_cfg)
    print(f"[t={time.time()-t0:.2f}s] Stage C done, delta_safe_barrier={delta_safe_barrier:.4f}")

    barrier_cfg = BarrierDMSConfig(
        n_int=cfg.n_int, delta_safe=delta_safe_barrier, delta_extra=cfg.delta_extra,
        barrier_levels=cfg.barrier_levels, epsilon_final=cfg.epsilon_final,
        time_limit_s=180.0, delta_min=cfg.delta_min, delta_max=cfg.delta_max,
        n_control_segments=cfg.n_control_segments,
        w_T=cfg.w_T, w_L=cfg.w_L, w_U=cfg.w_U, w_S=cfg.w_S,
    )
    print(f"[t={time.time()-t0:.2f}s] Building+solving BarrierDMSSolver...", flush=True)
    barrier_solver = bdms_mod.BarrierDMSSolver(graph, prepared.optimizer.dynamics, barrier_cfg)
    result = barrier_solver.solve(path_regions, z, warm_start,
                                   start_state=prepared.resolved.start_state,
                                   goal_state=prepared.resolved.goal_state)
    print(f"[t={time.time()-t0:.2f}s] DONE: success={result.success} status={result.solver_status} "
          f"s_min_sampled={result.min_safety_margin:.4f} s_min_certified={result.certified_safety_margin:.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "explicit":
        main_explicit_path()
    elif len(sys.argv) > 1 and sys.argv[1] == "single":
        main_single_path_timing()
    elif len(sys.argv) > 1 and sys.argv[1] == "full":
        main_full_run()
    else:
        main()
