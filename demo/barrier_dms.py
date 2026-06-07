"""
barrier_dms.py - Barrier continuation core for Centroid-Refine-DMS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class FailureCode(str, Enum):
    NARROW_INTERFACE    = "NARROW_INTERFACE"
    QP_INFEASIBLE       = "QP_INFEASIBLE"
    WARM_START_VIOLATION = "WARM_START_VIOLATION"
    IPOPT_RESTORATION   = "IPOPT_RESTORATION"
    IPOPT_MAX_ITER      = "IPOPT_MAX_ITER"
    CONTROL_SATURATION  = "CONTROL_SATURATION"
    DURATION_SATURATION = "DURATION_SATURATION"
    CONNECTION_GAP_LARGE = "CONNECTION_GAP_LARGE"
    KKT_DEGENERATE      = "KKT_DEGENERATE"
    NO_PATH_FOUND       = "NO_PATH_FOUND"


@dataclass
class FailureRecord:
    path: Tuple[str, ...]
    code: FailureCode
    barrier_level: int = -1
    ipopt_status: str = ""
    defect_norm: float = float("nan")
    coupling_gap: float = float("nan")


@dataclass
class BarrierSchedule:
    mu_schedule: List[float]
    n_mu: int

    @classmethod
    def from_warm_start(
        cls,
        f0: float,
        B0: float,
        alpha: float = 0.1,
        tau: float = 0.1,
        mu_min: float = 1e-5,
        delta_safe: float = 0.02,
        w_T: float = 1.0,
    ) -> "BarrierSchedule":
        if f0 <= 0.0 or B0 <= 0.0:
            mu_0 = delta_safe * w_T
        else:
            mu_0 = float(np.clip(alpha * f0 / B0, mu_min, 1.0))

        if mu_0 <= mu_min:
            n_mu = 1
        else:
            n_mu = max(1, math.ceil(math.log(mu_min / mu_0) / math.log(tau)))

        schedule = [mu_0 * (tau ** r) for r in range(n_mu)]
        if schedule and schedule[-1] > mu_min:
            schedule[-1] = mu_min

        return cls(mu_schedule=schedule, n_mu=n_mu)


_IPOPT_MAX_ITER_STATUSES = {"Maximum_Iterations_Exceeded", "max_iter_exceeded"}
_IPOPT_RESTORATION_STATUSES = {"Restoration_Failed", "restoration_failed", "Error_In_Step_Computation"}
_IPOPT_DEGENERATE_STATUSES = {
    "Converged_To_A_Locally_Infeasible_Point", "Infeasible_Problem_Detected", "locally_infeasible"
}
_COUPLING_GAP_THRESHOLD = 1e-4


def classify_dms_failure(
    ipopt_status: str,
    defect_norm: float,
    coupling_gap: float,
) -> Optional[FailureCode]:
    if ipopt_status in _IPOPT_RESTORATION_STATUSES:
        return FailureCode.IPOPT_RESTORATION
    if ipopt_status in _IPOPT_MAX_ITER_STATUSES:
        return FailureCode.IPOPT_MAX_ITER
    if ipopt_status in _IPOPT_DEGENERATE_STATUSES:
        return FailureCode.KKT_DEGENERATE
    if coupling_gap > _COUPLING_GAP_THRESHOLD:
        return FailureCode.CONNECTION_GAP_LARGE
    if ipopt_status in {"Solve_Succeeded", "Solved_To_Acceptable_Level", "acceptable"}:
        return None
    return FailureCode.KKT_DEGENERATE


@dataclass
class BarrierDMSSolverConfig:
    n_int: int = 20
    delta_safe: float = 0.02
    delta_min: float = 0.01
    delta_max: float = 10.0
    w_T: float = 1.0
    w_L: float = 1.0
    w_U: float = 1.0
    w_S: float = 0.2
    alpha_mu: float = 0.1
    tau: float = 0.1
    mu_min: float = 1e-5
    epsilon_final: float = 1e-6
    ipopt_max_iter_schedule: Tuple[int, ...] = (300, 200, 150, 100, 100)


@dataclass
class BarrierPathResult:
    success: bool
    failure_code: Optional[FailureCode] = None
    cost_unbarred: float = float("nan")
    x_nodes_opt: Optional[List[np.ndarray]] = None
    u_list_opt: Optional[List[np.ndarray]] = None
    delta_opt: Optional[List[float]] = None
    defect_norm: float = float("nan")
    coupling_gap: float = float("nan")
    s_min_sampled: float = float("nan")
    s_min_certified: float = float("nan")
    lipschitz_gap: float = float("nan")
    n_barrier_levels_run: int = 0
    n_ipopt_iters_total: int = 0
    optimality_note: str = "KKT-feasible under LICQ+SOSC; no global optimality certificate."
    failure_log: List[FailureRecord] = field(default_factory=list)


class BarrierDMSSolver:
    """
    Multiple-shooting DMS NLP with log-barrier safety constraints.
    Runs N_mu barrier levels from a warm-start; returns BarrierPathResult.
    """

    def __init__(self, dynamics, config: BarrierDMSSolverConfig):
        import casadi as ca
        self.dynamics = dynamics
        self.config = config
        n_x, n_u = dynamics.n_x, dynamics.n_u
        x_sym = ca.MX.sym('x_rk4', n_x)
        u_sym = ca.MX.sym('u_rk4', n_u)
        h_sym = ca.MX.sym('h_rk4', 1)
        f = dynamics.f_casadi
        k1 = h_sym * f(x_sym, u_sym)
        k2 = h_sym * f(x_sym + k1 / 2, u_sym)
        k3 = h_sym * f(x_sym + k2 / 2, u_sym)
        k4 = h_sym * f(x_sym + k3, u_sym)
        self._rk4_step = ca.Function('rk4', [x_sym, u_sym, h_sym],
                                     [x_sym + (k1 + 2*k2 + 2*k3 + k4) / 6])

    def solve_path(self, graph, path_regions, x_start, x_goal, warm_start) -> BarrierPathResult:
        import casadi as ca
        from constraint_layers import build_barrier_log_terms, compute_lipschitz_safety_gap

        cfg = self.config
        dynamics = self.dynamics
        n_x, n_u, n_int = dynamics.n_x, dynamics.n_u, cfg.n_int
        m = len(path_regions)
        failure_log: List[FailureRecord] = []

        f0, B0 = self._eval_f0_B0(graph, path_regions, warm_start)
        schedule = BarrierSchedule.from_warm_start(
            f0=f0, B0=B0, alpha=cfg.alpha_mu, tau=cfg.tau,
            mu_min=cfg.mu_min, delta_safe=cfg.delta_safe, w_T=cfg.w_T
        )

        (x_all_sym, lbx, ubx, g_sym, lbg, ubg,
         cost_no_barrier, x_node_syms, u_syms, delta_syms) = self._build_nlp(
            graph, path_regions, x_start, x_goal)

        x_w = self._flatten_ws(warm_start, m, n_x, n_u, n_int)
        n_iters_total = 0

        for r, mu_r in enumerate(schedule.mu_schedule):
            eps_r = max(cfg.epsilon_final, mu_r)
            max_iter_r = (cfg.ipopt_max_iter_schedule[r]
                          if r < len(cfg.ipopt_max_iter_schedule)
                          else cfg.ipopt_max_iter_schedule[-1])

            B_sym = build_barrier_log_terms(
                path_regions, graph, dynamics,
                x_node_syms, delta_syms, n_int, cfg.delta_safe)
            obj_full = cost_no_barrier - mu_r * B_sym

            nlp = {'x': x_all_sym, 'f': obj_full, 'g': g_sym}
            opts = {'ipopt.max_iter': max_iter_r, 'ipopt.tol': eps_r,
                    'ipopt.print_level': 0, 'print_time': False}
            solver = ca.nlpsol('bdms', 'ipopt', nlp, opts)
            sol = solver(x0=x_w, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            stats = solver.stats()
            ipopt_status = stats.get('return_status', 'unknown')
            n_iters_total += int(stats.get('iter_count', 0))
            x_w = np.array(sol['x']).flatten()

            g_val = np.array(sol['g']).flatten()
            n_defect = m * n_x * n_int
            defect_r = float(np.max(np.abs(g_val[:n_defect]))) if n_defect > 0 and len(g_val) >= n_defect else 0.0
            coupling_r = float(np.max(np.abs(g_val[n_defect:]))) if len(g_val) > n_defect else 0.0

            fc = classify_dms_failure(ipopt_status, defect_r, coupling_r)
            if fc is not None:
                failure_log.append(FailureRecord(
                    path=(), code=fc, barrier_level=r,
                    ipopt_status=ipopt_status, defect_norm=defect_r, coupling_gap=coupling_r))
                return BarrierPathResult(
                    success=False, failure_code=fc,
                    n_barrier_levels_run=r + 1, n_ipopt_iters_total=n_iters_total,
                    defect_norm=defect_r, coupling_gap=coupling_r, failure_log=failure_log)

        return self._parse_solution(x_w, graph, path_regions, cfg, schedule.n_mu,
                                    n_iters_total, compute_lipschitz_safety_gap)

    def _eval_f0_B0(self, graph, path_regions, ws) -> Tuple[float, float]:
        cfg = self.config
        f0, B0 = 0.0, 0.0
        n_int = cfg.n_int
        for seg_i, region_idx in enumerate(path_regions):
            region = graph.regions[region_idx]
            delta_i, u_i = ws.delta_list[seg_i], ws.u_list[seg_i]
            f0 += cfg.w_T * delta_i
            for k in range(n_int + 1):
                h_k = delta_i / (2 * n_int) if (k == 0 or k == n_int) else delta_i / n_int
                x_k = ws.x_nodes[seg_i][k]
                pos_k = self.dynamics.project_to_position(x_k)
                vel_k = self.dynamics.position_velocity(x_k, u_i)
                f0 += h_k * (cfg.w_L * float(np.dot(vel_k, vel_k)) +
                              cfg.w_U * float(np.dot(u_i, u_i)))
                for j in range(region.A.shape[0]):
                    s = float(region.b[j]) - cfg.delta_safe - float(region.A[j] @ pos_k)
                    B0 -= h_k * np.log(max(s, 1e-9))
        return f0, B0

    def _flatten_ws(self, ws, m, n_x, n_u, n_int) -> np.ndarray:
        flat = []
        for i in range(m):
            for k in range(n_int + 1):
                flat.extend(ws.x_nodes[i][k].tolist())
            flat.extend(ws.u_list[i].tolist())
            flat.append(ws.delta_list[i])
        return np.array(flat, dtype=float)

    def _build_nlp(self, graph, path_regions, x_start, x_goal):
        import casadi as ca
        cfg = self.config
        dynamics = self.dynamics
        n_x, n_u, n_int = dynamics.n_x, dynamics.n_u, cfg.n_int
        m = len(path_regions)

        u_lb, u_ub = dynamics.control_bounds()
        all_verts = np.vstack([r.vertices for r in graph.regions])
        pos_lb = np.array([np.min(all_verts[:, 0]) - 1e-6, np.min(all_verts[:, 1]) - 1e-6])
        pos_ub = np.array([np.max(all_verts[:, 0]) + 1e-6, np.max(all_verts[:, 1]) + 1e-6])
        state_lb, state_ub = dynamics.state_bounds(pos_lb, pos_ub)

        x_node_syms, u_syms, delta_syms = [], [], []
        all_vars, lbx, ubx = [], [], []

        for i in range(m):
            nodes_i = []
            for k in range(n_int + 1):
                xk = ca.MX.sym(f'x_{i}_{k}', n_x)
                nodes_i.append(xk)
                all_vars.append(xk)
                lbx.extend(state_lb.tolist())
                ubx.extend(state_ub.tolist())
            x_node_syms.append(nodes_i)
            u_i = ca.MX.sym(f'u_{i}', n_u)
            u_syms.append(u_i)
            all_vars.append(u_i)
            lbx.extend(u_lb.tolist())
            ubx.extend(u_ub.tolist())
            d_i = ca.MX.sym(f'd_{i}')
            delta_syms.append(d_i)
            all_vars.append(d_i)
            lbx.append(cfg.delta_min)
            ubx.append(cfg.delta_max)

        x_all_sym = ca.vertcat(*all_vars)
        g_list, lbg, ubg = [], [], []

        for i in range(m):
            h_i = delta_syms[i] / n_int
            for k in range(n_int):
                defect = x_node_syms[i][k + 1] - self._rk4_step(x_node_syms[i][k], u_syms[i], h_i)
                g_list.append(defect)
                lbg.extend([0.0] * n_x); ubg.extend([0.0] * n_x)

        for i in range(m - 1):
            coupling = x_node_syms[i][n_int] - x_node_syms[i + 1][0]
            g_list.append(coupling)
            lbg.extend([0.0] * n_x); ubg.extend([0.0] * n_x)

        g_list.append(x_node_syms[0][0] - ca.DM(x_start))
        lbg.extend([0.0] * n_x); ubg.extend([0.0] * n_x)
        g_list.append(x_node_syms[-1][n_int] - ca.DM(x_goal))
        lbg.extend([0.0] * n_x); ubg.extend([0.0] * n_x)

        g_sym = ca.vertcat(*g_list)

        cost = ca.MX(0.0)
        for i in range(m):
            cost = cost + cfg.w_T * delta_syms[i]
            for k in range(n_int + 1):
                h_k = delta_syms[i] / (2 * n_int) if (k == 0 or k == n_int) else delta_syms[i] / n_int
                vel_k = dynamics.position_velocity_casadi(x_node_syms[i][k], u_syms[i])
                cost = cost + h_k * (cfg.w_L * ca.dot(vel_k, vel_k) +
                                     cfg.w_U * ca.dot(u_syms[i], u_syms[i]))
        if m > 1:
            for i in range(m - 1):
                jump = u_syms[i] - u_syms[i + 1]
                cost = cost + cfg.w_S * ca.dot(jump, jump)

        return (x_all_sym, lbx, ubx, g_sym, lbg, ubg, cost, x_node_syms, u_syms, delta_syms)

    def _parse_solution(self, x_opt, graph, path_regions, cfg, n_mu, n_iters_total, lipschitz_fn):
        dynamics = self.dynamics
        n_x, n_u, n_int = dynamics.n_x, dynamics.n_u, cfg.n_int
        m = len(path_regions)

        x_nodes_opt, u_list_opt, delta_opt = [], [], []
        offset = 0
        for i in range(m):
            seg = np.zeros((n_int + 1, n_x))
            for k in range(n_int + 1):
                seg[k] = x_opt[offset:offset + n_x]; offset += n_x
            x_nodes_opt.append(seg)
            u_list_opt.append(x_opt[offset:offset + n_u].copy()); offset += n_u
            delta_opt.append(float(x_opt[offset])); offset += 1

        defect_norms = []
        for i in range(m):
            h_i = delta_opt[i] / n_int
            for k in range(n_int):
                rk4_val = np.array(self._rk4_step(x_nodes_opt[i][k], u_list_opt[i], h_i)).flatten()
                defect_norms.append(float(np.linalg.norm(x_nodes_opt[i][k + 1] - rk4_val)))
        defect_norm = max(defect_norms) if defect_norms else 0.0

        coupling_gaps = [float(np.linalg.norm(x_nodes_opt[i][n_int] - x_nodes_opt[i + 1][0]))
                         for i in range(m - 1)]
        coupling_gap = max(coupling_gaps) if coupling_gaps else 0.0

        cost = 0.0
        for i in range(m):
            d, u = delta_opt[i], u_list_opt[i]
            cost += cfg.w_T * d
            for k in range(n_int + 1):
                h_k = d / (2 * n_int) if (k == 0 or k == n_int) else d / n_int
                vel = dynamics.position_velocity(x_nodes_opt[i][k], u)
                cost += h_k * (cfg.w_L * float(np.dot(vel, vel)) + cfg.w_U * float(np.dot(u, u)))
        if m > 1:
            for i in range(m - 1):
                jump = u_list_opt[i] - u_list_opt[i + 1]
                cost += cfg.w_S * float(np.dot(jump, jump))

        s_min = float("inf")
        for i, ridx in enumerate(path_regions):
            region = graph.regions[ridx]
            for k in range(n_int + 1):
                pos_k = dynamics.project_to_position(x_nodes_opt[i][k])
                slacks = region.b - cfg.delta_safe - region.A @ pos_k
                s_min = min(s_min, float(np.min(slacks)))

        gap = lipschitz_fn(dynamics, path_regions, graph, np.array(delta_opt), n_int)

        return BarrierPathResult(
            success=True, cost_unbarred=cost,
            x_nodes_opt=x_nodes_opt, u_list_opt=u_list_opt, delta_opt=delta_opt,
            defect_norm=defect_norm, coupling_gap=coupling_gap,
            s_min_sampled=s_min, s_min_certified=s_min - gap, lipschitz_gap=gap,
            n_barrier_levels_run=n_mu, n_ipopt_iters_total=n_iters_total,
        )
