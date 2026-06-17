"""barrier_dms.py - Barrier-DMS solver for Centroid-Refine-DMS."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import casadi as ca
import numpy as np
import time

from dynamics import DynamicsModel, ControlParameterization, RK4Integrator
from graph_builder import RegionGraph
from constraint_layers import compute_lipschitz_safety_gap


@dataclass
class BarrierDMSConfig:
    n_int: int = 10
    delta_safe: float = 0.02
    delta_extra: float = 0.01
    barrier_levels: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.1, 0.01])
    epsilon_final: float = 1e-6
    time_limit_s: float = 60.0
    delta_min: float = 0.1
    delta_max: float = 10.0
    n_control_segments: int = 2
    epsilon_certificate_buffer: float = 0.0
    mu_weight: float = 1.0
    w_T: float = 1.0
    w_L: float = 1.0
    w_U: float = 1.0
    w_S: float = 0.2


@dataclass
class BarrierPathResult:
    success: bool
    path_regions: List[int]
    total_cost: float
    solve_time: float
    solver_status: str
    min_safety_margin: float = float('nan')
    certified_safety_margin: float = float('nan')
    lipschitz_gap: float = float('nan')
    safety_certification: str = "NOT_SET"
    defect_norm: float = float('nan')
    n_nlp_iterations: int = 0
    entry_states: Dict[int, np.ndarray] = field(default_factory=dict)
    exit_states: Dict[int, np.ndarray] = field(default_factory=dict)
    control_params: Dict[int, np.ndarray] = field(default_factory=dict)
    time_durations: Dict[int, float] = field(default_factory=dict)
    barrier_level_results: List[Dict] = field(default_factory=list)


class BarrierDMSSolver:
    """
    Direct multiple shooting solver with log-barrier interior safety.
    Implements the D-E stages of Centroid-Refine-DMS.
    """

    def __init__(self, graph: RegionGraph, dynamics: DynamicsModel, config: BarrierDMSConfig):
        self.graph = graph
        self.dynamics = dynamics
        self.config = config
        self.control_param = ControlParameterization(
            n_u=dynamics.n_u,
            parameterization="piecewise_constant",
            n_segments=config.n_control_segments,
        )
        self.integrator = RK4Integrator(
            dynamics, self.control_param, config.n_int
        )

    def solve(self, path_regions: List[int], anchor_points: np.ndarray,
              warm_start: Optional[Dict[int, Dict]] = None,
              start_state: Optional[np.ndarray] = None,
              goal_state: Optional[np.ndarray] = None) -> BarrierPathResult:
        start_time = time.time()
        cfg = self.config

        delta_arr = np.array([
            warm_start[ri]['delta'] if (warm_start and ri in warm_start) else 1.0
            for ri in path_regions
        ], dtype=float)
        # Clamp to delta_max (Part 2.4 variable bound): Delta_i can never
        # exceed delta_max once solved, so h_max (R13/Part 8.2) must reflect
        # that bound, not an unclamped warm-start estimate (Part 4.1 defines
        # Delta_i^0 with no upper clamp), which can otherwise re-inflate
        # delta_safe here past what the caller already converged on.
        delta_arr = np.minimum(delta_arr, cfg.delta_max)
        lip_gap = compute_lipschitz_safety_gap(self.dynamics, path_regions, self.graph, delta_arr, cfg.n_int)
        delta_safe = max(cfg.delta_safe, lip_gap + cfg.epsilon_final)
        node_delta_safe = np.full(
            (len(path_regions), cfg.n_int + 1), delta_safe, dtype=float
        )

        best_result = None
        level_results = []
        x_init = None
        total_n_iter = 0

        self._log_warm_start_slacks(
            path_regions, anchor_points, node_delta_safe, delta_arr
        )

        for mu in cfg.barrier_levels:
            if time.time() - start_time > cfg.time_limit_s:
                break
            # Inexact tolerance schedule (spec §5.2): tol = max(epsilon_final, mu)
            tol = max(cfg.epsilon_final, mu)
            try:
                result, x_init = self._solve_barrier_level(
                    path_regions=path_regions,
                    anchor_points=anchor_points,
                    warm_start=warm_start,
                    mu=mu,
                    x_init=x_init,
                    tol=tol,
                    start_state=start_state,
                    goal_state=goal_state,
                    node_delta_safe=node_delta_safe,
                )
                total_n_iter += result.n_nlp_iterations
                level_results.append({
                    'mu': mu, 'success': result.success, 'cost': result.total_cost,
                    'n_iter': result.n_nlp_iterations,
                })
                if result.success:
                    best_result = result
            except Exception as e:
                level_results.append({'mu': mu, 'success': False, 'error': str(e), 'n_iter': 0})

        solve_time = time.time() - start_time

        if best_result is None:
            return BarrierPathResult(
                success=False, path_regions=path_regions, total_cost=float('inf'),
                solve_time=solve_time, solver_status="All barrier levels failed",
                lipschitz_gap=lip_gap, barrier_level_results=level_results,
            )

        best_result.solve_time = solve_time
        best_result.barrier_level_results = level_results

        # min_slack already embeds the Lipschitz-gap/RK4-truncation terms
        # for the OPTIMIZED Delta_i (Task 2/3) -- do not subtract lip_gap or
        # eps_int again here, that would double-count them.
        min_slack = self._compute_min_slack(best_result)
        optimized_delta_arr = np.array(
            [best_result.time_durations[ri] for ri in path_regions], dtype=float
        )
        # lip_gap/eps_int below are PATH-WIDE WORST-CASE diagnostics only
        # (reported on BarrierPathResult for backward-compatible
        # logging/reporting) -- the per-segment values actually enforced
        # live inside the NLP are already folded into min_slack.
        lip_gap = compute_lipschitz_safety_gap(
            self.dynamics, path_regions, self.graph,
            optimized_delta_arr, cfg.n_int,
        )
        h_max = float(np.max(optimized_delta_arr)) / cfg.n_int
        defect_norm = self._compute_defect_norm(best_result, start_state, goal_state)
        eps_int = self.dynamics.f_lipschitz_bound() * (h_max ** 4) / 30.0
        best_result.defect_norm = defect_norm
        best_result.lipschitz_gap = lip_gap
        best_result.min_safety_margin = float(min_slack)
        # min_slack is ALREADY hard-floored at delta_extra +
        # epsilon_certificate_buffer by the NLP's slack lbx (Task 3) -- do
        # NOT subtract that floor again here. Doing so previously made
        # certified_safety_margin collapse to ~ -defect_norm whenever the
        # floor was the active/binding constraint (the common case for any
        # genuinely narrow corridor), making epsilon_certificate_buffer
        # incapable of ever raising the certified margin. The only
        # remaining correction is the hard-equality residual (defect_norm)
        # at solver tolerance.
        best_result.certified_safety_margin = float(min_slack - defect_norm)
        best_result.safety_certification = (
            "CERTIFIED" if best_result.certified_safety_margin > 0 else "NOT_CERTIFIED"
        )
        best_result.n_nlp_iterations = total_n_iter
        final_level = cfg.barrier_levels[-1]
        final_level_succeeded = bool(
            level_results
            and level_results[-1].get('mu') == final_level
            and level_results[-1].get('success')
        )
        if not final_level_succeeded or best_result.certified_safety_margin <= 0:
            best_result.success = False
            best_result.solver_status = (
                "CERTIFICATE_FAIL"
                if final_level_succeeded else "FINAL_BARRIER_LEVEL_NOT_SOLVED"
            )

        return best_result

    def _log_warm_start_slacks(
        self, path_regions, anchor_points, node_delta_safe, delta_arr,
    ):
        """Log s_ijk values for the linear-interpolation warm start (spec R3 check)."""
        import logging
        log = logging.getLogger("barrier_dms.slack_check")
        min_slack = float('inf')
        worst = None
        for i, region_idx in enumerate(path_regions):
            region = self.graph._regions_by_index[region_idx]
            q_entry = anchor_points[i]
            q_exit = anchor_points[i + 1]
            delta_i = float(delta_arr[i])
            for k in range(self.config.n_int + 1):
                tau = k / self.config.n_int
                pos = (1 - tau) * q_entry + tau * q_exit
                slacks = region.b - node_delta_safe[i, k] - region.A @ pos
                s_min_k = float(np.min(slacks))
                if s_min_k < min_slack:
                    min_slack = s_min_k
                    worst = (i, region_idx, k, tau, pos.tolist(), float(np.argmin(slacks)), s_min_k)
        log.debug("R3 warm-start slack check: s_min=%.6f (seg=%s, region=%s, k=%s, "
                  "tau=%.2f, pos=%s, j=%s)",
                  min_slack, worst[0], worst[1], worst[2], worst[3], worst[4], worst[5])
        if min_slack < self.config.delta_extra:
            raise ValueError(
                "R3 strict-interior invariant failed: "
                f"s_ijk={min_slack:.6f} < delta_extra="
                f"{self.config.delta_extra:.6f} at segment={worst[0]}, "
                f"region={worst[1]}, node={worst[2]}"
            )
        log.debug("R3 OK: all s_ijk >= %.6f (delta_extra=%.6f)",
                  min_slack, self.config.delta_extra)

    def _build_nlp_symbols(
        self, path_regions, anchor_points, mu,
        start_state=None, goal_state=None, node_delta_safe=None,
    ):
        m = len(path_regions)
        n_x = self.dynamics.n_x
        n_w = self.control_param.n_w
        cfg = self.config
        dt = 1.0 / cfg.n_int
        A_list = [self.graph._regions_by_index[ri].A for ri in path_regions]
        L_s = self.dynamics.compute_lipschitz_bound(A_list)
        f_lip = self.dynamics.f_lipschitz_bound()

        all_verts = np.vstack([r.vertices for r in self.graph.regions])
        pos_lb = np.array([np.min(all_verts[:, 0]) - 1e-6, np.min(all_verts[:, 1]) - 1e-6])
        pos_ub = np.array([np.max(all_verts[:, 0]) + 1e-6, np.max(all_verts[:, 1]) + 1e-6])
        state_lb, state_ub = self.dynamics.state_bounds(pos_lb, pos_ub)
        u_lb, u_ub = self.dynamics.control_bounds()
        w_lb = np.tile(u_lb, self.control_param.n_segments)
        w_ub = np.tile(u_ub, self.control_param.n_segments)

        x_sym_list = []
        lbx = []
        ubx = []
        x0_list = []
        s_minus_vars = []
        w_vars = []
        delta_vars = []
        x_node_vars_list = []

        for i, region_idx in enumerate(path_regions):
            s_m = ca.MX.sym(f's_m_{i}', n_x)
            w_v = ca.MX.sym(f'w_{i}', n_w)
            dv = ca.MX.sym(f'd_{i}', 1)

            s_minus_vars.append(s_m)
            w_vars.append(w_v)
            delta_vars.append(dv)

            x_sym_list.extend([s_m, w_v, dv])
            lbx.extend(state_lb.tolist())
            ubx.extend(state_ub.tolist())
            lbx.extend(w_lb.tolist())
            ubx.extend(w_ub.tolist())
            lbx.append(cfg.delta_min)
            ubx.append(cfg.delta_max)

            q_entry = anchor_points[i]
            q_exit = anchor_points[i + 1]
            s0 = np.zeros(n_x)
            s0[:2] = q_entry
            if self.dynamics.angle_indices:
                d = q_exit - q_entry
                if np.linalg.norm(d) > 1e-9:
                    s0[self.dynamics.angle_indices[0]] = float(np.arctan2(d[1], d[0]))
            dist = float(np.linalg.norm(q_exit - q_entry))
            v_nom = float(u_ub[0]) * 0.5 if u_ub.size > 0 else 1.0
            d0 = max(dist / max(v_nom, 1e-3), cfg.delta_min)

            x0_list.extend(s0.tolist())
            w0 = np.zeros(n_w)
            if u_ub.size > 0:
                for control_idx in range(self.control_param.n_segments):
                    w0[control_idx * self.dynamics.n_u] = v_nom
            x0_list.extend(w0.tolist())
            x0_list.append(d0)

        g_list = []
        lbg = []
        ubg = []
        obj_base = ca.MX(0.0)
        barrier_term = ca.MX(0.0)

        for i in range(m):
            s_m = s_minus_vars[i]
            w_v = w_vars[i]
            dv = delta_vars[i]

            traj = [s_m]
            x_k = s_m
            for k in range(cfg.n_int):
                tau = k * dt
                u_k = self.control_param.evaluate_casadi(tau, w_v)
                h_k = dv / (2.0 * cfg.n_int) if k == 0 else dv / cfg.n_int
                vel_k = self.dynamics.position_velocity_casadi(x_k, u_k)
                obj_base = (
                    obj_base
                    + h_k * cfg.w_L * ca.dot(vel_k, vel_k)
                    + h_k * cfg.w_U * ca.dot(u_k, u_k)
                )
                k1 = dv * self.dynamics.f_casadi(x_k, u_k)
                u_mid = self.control_param.evaluate_casadi(tau + 0.5 * dt, w_v)
                k2 = dv * self.dynamics.f_casadi(x_k + 0.5 * dt * k1, u_mid)
                k3 = dv * self.dynamics.f_casadi(x_k + 0.5 * dt * k2, u_mid)
                u_end = self.control_param.evaluate_casadi(tau + dt, w_v)
                k4 = dv * self.dynamics.f_casadi(x_k + dt * k3, u_end)
                x_k = x_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                traj.append(x_k)

            u_end = self.control_param.evaluate_casadi(1.0, w_v)
            vel_end = self.dynamics.position_velocity_casadi(x_k, u_end)
            h_end = dv / (2.0 * cfg.n_int)
            obj_base = (
                obj_base
                + h_end * cfg.w_L * ca.dot(vel_end, vel_end)
                + h_end * cfg.w_U * ca.dot(u_end, u_end)
            )
            x_node_vars_list.append(traj)
            s_plus = traj[-1]
            obj_base = obj_base + cfg.w_T * dv

            for control_idx in range(self.control_param.n_segments - 1):
                start = control_idx * self.dynamics.n_u
                u_left = w_v[start:start + self.dynamics.n_u]
                u_right = w_v[
                    start + self.dynamics.n_u:start + 2 * self.dynamics.n_u
                ]
                obj_base = obj_base + cfg.w_S * ca.dot(u_right - u_left, u_right - u_left)

            if i < m - 1:
                s_minus_next = s_minus_vars[i + 1]
                gap = s_plus - s_minus_next
                g_list.append(gap)
                lbg.extend([0.0] * n_x)
                ubg.extend([0.0] * n_x)
                u_exit = self.control_param.evaluate_casadi(1.0, w_v)
                u_entry_next = self.control_param.evaluate_casadi(
                    0.0, w_vars[i + 1]
                )
                control_jump = u_entry_next - u_exit
                obj_base = obj_base + cfg.w_S * ca.dot(control_jump, control_jump)

        boundary_start = np.zeros(n_x, dtype=float)
        boundary_goal = np.zeros(n_x, dtype=float)
        boundary_start[:2] = anchor_points[0]
        boundary_goal[:2] = anchor_points[m]
        if self.dynamics.angle_indices:
            first_direction = anchor_points[1] - anchor_points[0]
            last_direction = anchor_points[m] - anchor_points[m - 1]
            angle_idx = self.dynamics.angle_indices[0]
            if np.linalg.norm(first_direction) > 1e-9:
                boundary_start[angle_idx] = np.arctan2(
                    first_direction[1], first_direction[0]
                )
            if np.linalg.norm(last_direction) > 1e-9:
                boundary_goal[angle_idx] = np.arctan2(
                    last_direction[1], last_direction[0]
                )
        if start_state is None:
            start_state = boundary_start
        if goal_state is None:
            goal_state = boundary_goal
        boundary_start = np.asarray(start_state, dtype=float)
        boundary_goal = np.asarray(goal_state, dtype=float)
        g_list.append(s_minus_vars[0] - ca.DM(boundary_start))
        lbg.extend([0.0] * n_x)
        ubg.extend([0.0] * n_x)
        g_list.append(x_node_vars_list[-1][-1] - ca.DM(boundary_goal))
        lbg.extend([0.0] * n_x)
        ubg.extend([0.0] * n_x)

        for i, region_idx in enumerate(path_regions):
            region = self.graph._regions_by_index[region_idx]
            delta_i = delta_vars[i]
            # Live margin (spec Sec 1): grows with this segment's OWN
            # optimized Delta_i, so the constraint the solver satisfies
            # always matches the true post-hoc Lipschitz/RK4 requirement
            # for whatever Delta_i it converges to -- no pre-solve estimate.
            h_i = delta_i / cfg.n_int
            margin_i = cfg.delta_safe + L_s * h_i / 2.0 + f_lip * (h_i ** 4) / 30.0
            slack_floor = cfg.delta_extra + cfg.epsilon_certificate_buffer

            for k, x_k in enumerate(x_node_vars_list[i]):
                slack = ca.MX.sym(
                    f'safety_slack_{i}_{k}', region.A.shape[0]
                )
                x_sym_list.append(slack)
                # Hard floor (spec Sec 2): guarantees certified_safety_margin
                # >= delta_extra + epsilon_certificate_buffer - defect_norm
                # for any successful solve, by construction.
                lbx.extend([slack_floor] * region.A.shape[0])
                ubx.extend([np.inf] * region.A.shape[0])

                tau = k / cfg.n_int
                warm_pos = (
                    (1.0 - tau) * anchor_points[i]
                    + tau * anchor_points[i + 1]
                )
                warm_slack = (
                    region.b - node_delta_safe[i, k]
                    - region.A @ warm_pos
                )
                x0_list.extend(np.maximum(warm_slack, slack_floor).tolist())

                pos_k = self.dynamics.project_to_position_casadi(x_k)
                defined_slack = (
                    ca.DM(region.b)
                    - margin_i
                    - ca.mtimes(ca.DM(region.A), pos_k)
                )
                g_list.append(slack - defined_slack)
                lbg.extend([0.0] * region.A.shape[0])
                ubg.extend([0.0] * region.A.shape[0])

                h_k = (
                    delta_i / (2.0 * cfg.n_int)
                    if k in (0, cfg.n_int)
                    else delta_i / cfg.n_int
                )
                barrier_term = (
                    barrier_term
                    - mu * cfg.mu_weight * h_k * ca.sum1(ca.log(slack))
                )

        x_sym = ca.vertcat(*x_sym_list)
        g_sym = ca.vertcat(*g_list) if g_list else ca.MX(0, 1)

        return (x_sym, obj_base, barrier_term, g_sym, lbx, ubx, lbg, ubg, x0_list,
                s_minus_vars, w_vars, delta_vars, x_node_vars_list)

    def _solve_barrier_level(self, path_regions, anchor_points, warm_start,
                              mu, x_init, tol,
                              start_state=None, goal_state=None,
                              node_delta_safe=None):
        (x_sym, obj_base, barrier_term, g_sym, lbx, ubx, lbg, ubg,
         x0_default, s_minus_vars, w_vars, delta_vars, x_node_vars_list) = self._build_nlp_symbols(
            path_regions, anchor_points, mu,
            start_state=start_state, goal_state=goal_state,
            node_delta_safe=node_delta_safe,
        )

        x0 = x_init.tolist() if x_init is not None else x0_default

        # Per-level max_iter schedule (spec Table §5.2)
        if mu >= 1.0 or mu <= 0.01:
            max_iter = 300
        elif mu >= 0.5:
            max_iter = 200
        else:
            max_iter = 150

        nlp = {'x': x_sym, 'f': obj_base + barrier_term, 'g': g_sym}
        opts = {
            'ipopt.max_iter': max_iter, 'ipopt.tol': tol,
            'ipopt.print_level': 0, 'print_time': 0,
        }
        solver = ca.nlpsol('barrier_dms', 'ipopt', nlp, opts)
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        stats = solver.stats()
        success = bool(stats.get('success', False))
        x_opt = np.array(sol['x']).flatten()

        # Spec Sec 7.2 / Part 9 Stage F: UB must be J|mu=0 (no barrier term).
        base_cost_fn = ca.Function('barrier_dms_base_cost', [x_sym], [obj_base])
        total_cost = float(base_cost_fn(x_opt))

        m = len(path_regions)
        n_x = self.dynamics.n_x
        n_w = self.control_param.n_w
        vars_per_seg = n_x + n_w + 1

        result = BarrierPathResult(
            success=success, path_regions=path_regions, total_cost=total_cost,
            solve_time=0.0, solver_status=stats.get('return_status', 'unknown'),
            n_nlp_iterations=int(stats.get('iter_count', 0)),
        )

        for i, region_idx in enumerate(path_regions):
            off = i * vars_per_seg
            s_m = x_opt[off:off + n_x]
            w = x_opt[off + n_x:off + n_x + n_w]
            delta = float(x_opt[off + n_x + n_w])
            result.entry_states[region_idx] = s_m
            result.exit_states[region_idx] = self.integrator.integrate(
                s_m, w, delta
            )
            result.control_params[region_idx] = w
            result.time_durations[region_idx] = delta

        return result, x_opt

    def _compute_min_slack(self, result: BarrierPathResult) -> float:
        """
        Sampled safety margin at the optimized solution, using the SAME
        per-segment margin formula enforced live inside the NLP (spec
        2026-06-17-crd-certificate-margin-relaxation-design.md Sec 1/4):
        margin_i = delta_safe + L_s * h_i/2 + f_lip * h_i**4/30, with
        h_i = Delta_i / n_int computed from the OPTIMIZED Delta_i, not a
        pre-solve estimate.
        """
        cfg = self.config
        if not result.path_regions:
            return float('nan')
        A_list = [self.graph._regions_by_index[ri].A for ri in result.path_regions]
        L_s = self.dynamics.compute_lipschitz_bound(A_list)
        f_lip = self.dynamics.f_lipschitz_bound()

        min_slack = float('inf')
        for seg_idx, region_idx in enumerate(result.path_regions):
            if (
                region_idx not in result.entry_states
                or region_idx not in result.control_params
                or region_idx not in result.time_durations
            ):
                continue
            region = self.graph._regions_by_index[region_idx]
            x_k = np.asarray(result.entry_states[region_idx], dtype=float).copy()
            w = np.asarray(result.control_params[region_idx], dtype=float)
            delta = float(result.time_durations[region_idx])
            dt = 1.0 / cfg.n_int
            h_i = delta / cfg.n_int
            margin = cfg.delta_safe + L_s * h_i / 2.0 + f_lip * (h_i ** 4) / 30.0

            for k in range(cfg.n_int + 1):
                pos = self.dynamics.project_to_position(x_k)
                slacks = region.b - margin - region.A @ pos
                min_slack = min(min_slack, float(np.min(slacks)))
                if k == cfg.n_int:
                    break

                tau = k * dt
                u_k = self.control_param.evaluate(tau, w)
                k1 = delta * self.dynamics.f(x_k, u_k)
                u_mid = self.control_param.evaluate(tau + 0.5 * dt, w)
                k2 = delta * self.dynamics.f(x_k + 0.5 * dt * k1, u_mid)
                k3 = delta * self.dynamics.f(x_k + 0.5 * dt * k2, u_mid)
                u_end = self.control_param.evaluate(tau + dt, w)
                k4 = delta * self.dynamics.f(x_k + dt * k3, u_end)
                x_k = x_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return min_slack if np.isfinite(min_slack) else float('nan')

    def _compute_defect_norm(
        self,
        result: BarrierPathResult,
        start_state: Optional[np.ndarray],
        goal_state: Optional[np.ndarray],
    ) -> float:
        """
        Inf-norm of the coupling + boundary residuals at the NLP solution
        (spec Sec 6 Tier 1 / Part 9 Stage F: defect_norm, coupling_gap).
        These are hard equalities inside the NLP (R1) -- this only measures
        the residual at the reported solution, it does not change them.
        """
        path_regions = result.path_regions
        if not path_regions:
            return 0.0

        gaps = []
        if start_state is not None and path_regions[0] in result.entry_states:
            gaps.append(
                float(np.max(np.abs(
                    result.entry_states[path_regions[0]] - np.asarray(start_state, dtype=float)
                )))
            )
        if goal_state is not None and path_regions[-1] in result.exit_states:
            gaps.append(
                float(np.max(np.abs(
                    result.exit_states[path_regions[-1]] - np.asarray(goal_state, dtype=float)
                )))
            )
        for left_region, right_region in zip(path_regions[:-1], path_regions[1:]):
            if left_region in result.exit_states and right_region in result.entry_states:
                gaps.append(
                    float(np.max(np.abs(
                        result.exit_states[left_region] - result.entry_states[right_region]
                    )))
                )

        return float(max(gaps)) if gaps else 0.0
