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


def _debug_ws_theta(warm_start, x_start, x_goal, n_x: int) -> None:
    """Print warm-start start/end theta vs DMS boundary constraints."""
    if n_x < 3:
        return
    ws_theta0 = float(warm_start.x_nodes[0][0, 2])
    ws_theta_last = float(warm_start.x_nodes[-1][-1, 2])
    print(f"  WS  x_{{0,0}}.theta={ws_theta0:.4f}  x_start.theta={float(x_start[2]):.4f}"
          f"  err={abs(ws_theta0 - float(x_start[2])):.4f}")
    print(f"  WS  x_last.theta  ={ws_theta_last:.4f}  x_goal.theta  ={float(x_goal[2]):.4f}"
          f"  err={abs(ws_theta_last - float(x_goal[2])):.4f}")
    max_coup = 0.0
    for i in range(len(warm_start.x_nodes) - 1):
        err = abs(float(warm_start.x_nodes[i][-1, 2]) - float(warm_start.x_nodes[i + 1][0, 2]))
        max_coup = max(max_coup, err)
    print(f"  WS  max_internal_coupling_theta_err={max_coup:.4f}")


def _debug_g_breakdown(g_val, n_defect: int, n_equality: int,
                       m: int, n_x: int, path_regions) -> None:
    """Print per-segment defect and per-coupling residuals."""
    n_int_per_seg = n_defect // (m * n_x) if m * n_x > 0 else 0
    seg_defects = []
    for i in range(m):
        start = i * n_x * n_int_per_seg
        end = start + n_x * n_int_per_seg
        seg_defects.append(float(np.max(np.abs(g_val[start:end]))) if end <= len(g_val) else float('nan'))
    print(f"  g_breakdown: n_defect={n_defect}  n_equality={n_equality}  n_g={len(g_val)}")
    print(f"  per-seg defect (regions {path_regions}): {[f'{v:.3e}' for v in seg_defects]}")
    if n_equality > n_defect:
        coup_block = g_val[n_defect:n_equality]
        n_coup_vars = len(coup_block) // n_x if n_x > 0 else 0
        for k in range(n_coup_vars):
            block = coup_block[k * n_x:(k + 1) * n_x]
            label = ("start" if k == 0 else "goal" if k == n_coup_vars - 1
                     else f"coup[{k}]")
            print(f"    {label}: {np.round(block, 4)}")


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
    # n_ctrl_per_seg: piecewise-constant controls per segment.
    # n_int must be divisible by n_ctrl_per_seg.
    n_ctrl_per_seg: int = 1
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
    ipopt_max_iter_schedule: Tuple[int, ...] = (1000, 500, 300, 200, 200)
    # When False, skip the log-barrier objective; safety is enforced by the
    # explicit halfplane inequality constraints added to the NLP.
    use_log_barrier: bool = False


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

        if not cfg.use_log_barrier:
            max_iter = cfg.ipopt_max_iter_schedule[0] if cfg.ipopt_max_iter_schedule else 1000
            nlp = {'x': x_all_sym, 'f': cost_no_barrier, 'g': g_sym}
            opts = {'ipopt.max_iter': max_iter, 'ipopt.tol': cfg.epsilon_final,
                    'ipopt.print_level': 0, 'print_time': False,
                    'ipopt.nlp_scaling_method': 'gradient-based'}
            solver = ca.nlpsol('bdms', 'ipopt', nlp, opts)
            sol = solver(x0=x_w, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            stats = solver.stats()
            ipopt_status = stats.get('return_status', 'unknown')
            n_iters_total = int(stats.get('iter_count', 0))
            x_w = np.array(sol['x']).flatten()
            g_val = np.array(sol['g']).flatten()
            n_defect_c = m * n_x * n_int
            n_equality = n_defect_c + max(0, m - 1) * n_x + 2 * n_x
            defect_r = float(np.max(np.abs(g_val[:n_defect_c]))) if n_defect_c > 0 else 0.0
            coupling_r = float(np.max(np.abs(g_val[n_defect_c:n_equality]))) if len(g_val) > n_defect_c else 0.0
            print(f"[DMS no-barrier] ipopt={ipopt_status!r}  iters={n_iters_total}"
                  f"  defect={defect_r:.3e}  coupling={coupling_r:.3e}")
            _debug_ws_theta(warm_start, x_start, x_goal, n_x)
            _debug_g_breakdown(g_val, n_defect_c, n_equality, m, n_x, path_regions)
            fc = classify_dms_failure(ipopt_status, defect_r, coupling_r)
            if fc is not None:
                failure_log.append(FailureRecord(
                    path=(), code=fc, barrier_level=0,
                    ipopt_status=ipopt_status, defect_norm=defect_r, coupling_gap=coupling_r))
                return BarrierPathResult(
                    success=False, failure_code=fc,
                    n_barrier_levels_run=1, n_ipopt_iters_total=n_iters_total,
                    defect_norm=defect_r, coupling_gap=coupling_r, failure_log=failure_log)
            return self._parse_solution(x_w, graph, path_regions, cfg, 1,
                                        n_iters_total, compute_lipschitz_safety_gap)

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
            # coupling + boundary equalities immediately after defect constraints
            n_equality = n_defect + max(0, m - 1) * n_x + 2 * n_x
            defect_r = float(np.max(np.abs(g_val[:n_defect]))) if n_defect > 0 and len(g_val) >= n_defect else 0.0
            coupling_r = float(np.max(np.abs(g_val[n_defect:n_equality]))) if len(g_val) > n_defect else 0.0
            print(f"[DMS barrier r={r} mu={mu_r:.2e}] ipopt={ipopt_status!r}  iters={n_iters_total}"
                  f"  defect={defect_r:.3e}  coupling={coupling_r:.3e}")
            if r == 0:
                _debug_ws_theta(warm_start, x_start, x_goal, n_x)
            _debug_g_breakdown(g_val, n_defect, n_equality, m, n_x, path_regions)

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
        n_ctrl = cfg.n_ctrl_per_seg
        n_steps = n_int // n_ctrl
        for seg_i, region_idx in enumerate(path_regions):
            region = graph.regions[region_idx]
            delta_i = ws.delta_list[seg_i]
            h = delta_i / n_int
            f0 += cfg.w_T * delta_i
            for j in range(n_ctrl):
                # Warm start provides one constant control per segment; replicate
                # across sub-intervals to mirror _build_nlp quadrature structure.
                u_ij = ws.u_list[seg_i]
                for k in range(j * n_steps, (j + 1) * n_steps):
                    h_k = delta_i / (2 * n_int) if k == 0 else h
                    x_k = ws.x_nodes[seg_i][k]
                    pos_k = self.dynamics.project_to_position(x_k)
                    vel_k = self.dynamics.position_velocity(x_k, u_ij)
                    f0 += h_k * (cfg.w_L * float(np.dot(vel_k, vel_k)) +
                                  cfg.w_U * float(np.dot(u_ij, u_ij)))
                    for jj in range(region.A.shape[0]):
                        s = float(region.b[jj]) - cfg.delta_safe - float(region.A[jj] @ pos_k)
                        B0 -= h_k * np.log(max(s, 1e-9))
        return f0, B0

    def _flatten_ws(self, ws, m, n_x, n_u, n_int) -> np.ndarray:
        n_ctrl = self.config.n_ctrl_per_seg
        flat = []
        for i in range(m):
            for k in range(n_int + 1):
                flat.extend(ws.x_nodes[i][k].tolist())
            for _ in range(n_ctrl):
                flat.extend(ws.u_list[i].tolist())
            flat.append(ws.delta_list[i])
        return np.array(flat, dtype=float)

    def _build_nlp(self, graph, path_regions, x_start, x_goal):
        import casadi as ca
        cfg = self.config
        dynamics = self.dynamics
        n_x, n_u, n_int = dynamics.n_x, dynamics.n_u, cfg.n_int
        n_ctrl = cfg.n_ctrl_per_seg
        m = len(path_regions)

        if n_int % n_ctrl != 0:
            raise ValueError(f"n_int={n_int} must be divisible by n_ctrl_per_seg={n_ctrl}")
        n_steps = n_int // n_ctrl  # RK4 steps per control interval

        u_lb, u_ub = dynamics.control_bounds()
        all_verts = np.vstack([r.vertices for r in graph.regions])
        pos_lb = np.array([np.min(all_verts[:, 0]) - 1e-6, np.min(all_verts[:, 1]) - 1e-6])
        pos_ub = np.array([np.max(all_verts[:, 0]) + 1e-6, np.max(all_verts[:, 1]) + 1e-6])
        state_lb, state_ub = dynamics.state_bounds(pos_lb, pos_ub)

        # u_syms[i] is a list of n_ctrl CasADi symbols, one per control interval.
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
            u_segs_i = []
            for j in range(n_ctrl):
                u_ij = ca.MX.sym(f'u_{i}_{j}', n_u)
                u_segs_i.append(u_ij)
                all_vars.append(u_ij)
                lbx.extend(u_lb.tolist())
                ubx.extend(u_ub.tolist())
            u_syms.append(u_segs_i)
            d_i = ca.MX.sym(f'd_{i}')
            delta_syms.append(d_i)
            all_vars.append(d_i)
            lbx.append(cfg.delta_min)
            ubx.append(cfg.delta_max)

        x_all_sym = ca.vertcat(*all_vars)
        g_list, lbg, ubg = [], [], []

        for i in range(m):
            h_i = delta_syms[i] / n_int
            for j in range(n_ctrl):
                u_ij = u_syms[i][j]
                for k in range(j * n_steps, (j + 1) * n_steps):
                    defect = x_node_syms[i][k + 1] - self._rk4_step(x_node_syms[i][k], u_ij, h_i)
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

        from graph_types import region_node_label as _rnl
        for i, region_idx in enumerate(path_regions):
            region = graph.regions[region_idx]
            A_dm = ca.DM(region.A)
            b_safe = ca.DM((region.b - cfg.delta_safe).reshape(-1, 1))
            n_half = region.A.shape[0]
            for k in range(n_int + 1):
                pos_k = dynamics.project_to_position_casadi(x_node_syms[i][k])

                # Exit node of non-last segment: use intersection polygon H-rep to avoid
                # infeasibility from obstacle-tightening constraints conflicting across
                # the shared boundary of two adjacent regions.
                if k == n_int and i < m - 1:
                    isect = graph.intersections.get((_rnl(region_idx), _rnl(path_regions[i + 1])))
                    if isect is not None:
                        A_i = ca.DM(isect.A)
                        b_i = ca.DM((isect.b - cfg.delta_safe).reshape(-1, 1))
                        g_list.append(ca.mtimes(A_i, pos_k) - b_i)
                        lbg.extend([-np.inf] * isect.A.shape[0])
                        ubg.extend([0.0] * isect.A.shape[0])
                        continue

                # Entry node of non-first segment: covered by previous segment's exit
                # intersection constraint (coupling equality links them).
                if k == 0 and i > 0:
                    continue

                s_expr = ca.mtimes(A_dm, pos_k) - b_safe
                g_list.append(s_expr)
                lbg.extend([-np.inf] * n_half)
                ubg.extend([0.0] * n_half)

        g_sym = ca.vertcat(*g_list)

        cost = ca.MX(0.0)
        for i in range(m):
            cost = cost + cfg.w_T * delta_syms[i]
            h_i = delta_syms[i] / n_int
            for j in range(n_ctrl):
                u_ij = u_syms[i][j]
                for k in range(j * n_steps, (j + 1) * n_steps):
                    h_k = delta_syms[i] / (2 * n_int) if (k == 0 or k == n_int) else h_i
                    vel_k = dynamics.position_velocity_casadi(x_node_syms[i][k], u_ij)
                    cost = cost + h_k * (cfg.w_L * ca.dot(vel_k, vel_k) +
                                         cfg.w_U * ca.dot(u_ij, u_ij))
        # Smoothness: last control of segment i vs first control of segment i+1.
        if m > 1:
            for i in range(m - 1):
                jump = u_syms[i][-1] - u_syms[i + 1][0]
                cost = cost + cfg.w_S * ca.dot(jump, jump)

        return (x_all_sym, lbx, ubx, g_sym, lbg, ubg, cost, x_node_syms, u_syms, delta_syms)

    def _parse_solution(self, x_opt, graph, path_regions, cfg, n_mu, n_iters_total, lipschitz_fn):
        dynamics = self.dynamics
        n_x, n_u, n_int = dynamics.n_x, dynamics.n_u, cfg.n_int
        n_ctrl = cfg.n_ctrl_per_seg
        n_steps = n_int // n_ctrl
        m = len(path_regions)

        x_nodes_opt, u_all, delta_opt = [], [], []
        offset = 0
        for i in range(m):
            seg = np.zeros((n_int + 1, n_x))
            for k in range(n_int + 1):
                seg[k] = x_opt[offset:offset + n_x]; offset += n_x
            x_nodes_opt.append(seg)
            u_segs = []
            for _ in range(n_ctrl):
                u_segs.append(x_opt[offset:offset + n_u].copy()); offset += n_u
            u_all.append(u_segs)
            delta_opt.append(float(x_opt[offset])); offset += 1

        # Backward-compat: expose first control of each segment.
        u_list_opt = [u_all[i][0] for i in range(m)]

        defect_norms = []
        for i in range(m):
            h_i = delta_opt[i] / n_int
            for j in range(n_ctrl):
                u_ij = u_all[i][j]
                for k in range(j * n_steps, (j + 1) * n_steps):
                    rk4_val = np.array(self._rk4_step(x_nodes_opt[i][k], u_ij, h_i)).flatten()
                    defect_norms.append(float(np.linalg.norm(x_nodes_opt[i][k + 1] - rk4_val)))
        defect_norm = max(defect_norms) if defect_norms else 0.0

        coupling_gaps = [float(np.linalg.norm(x_nodes_opt[i][n_int] - x_nodes_opt[i + 1][0]))
                         for i in range(m - 1)]
        coupling_gap = max(coupling_gaps) if coupling_gaps else 0.0

        cost = 0.0
        h_i_arr = [delta_opt[i] / n_int for i in range(m)]
        for i in range(m):
            d = delta_opt[i]
            h_i = h_i_arr[i]
            cost += cfg.w_T * d
            for j in range(n_ctrl):
                u_ij = u_all[i][j]
                for k in range(j * n_steps, (j + 1) * n_steps):
                    h_k = d / (2 * n_int) if (k == 0 or k == n_int) else h_i
                    vel = dynamics.position_velocity(x_nodes_opt[i][k], u_ij)
                    cost += h_k * (cfg.w_L * float(np.dot(vel, vel)) +
                                   cfg.w_U * float(np.dot(u_ij, u_ij)))
        if m > 1:
            for i in range(m - 1):
                jump = u_all[i][-1] - u_all[i + 1][0]
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
