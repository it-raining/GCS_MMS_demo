"""geometric_refiner.py - Interface QP for Centroid-Refine-DMS."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import casadi as ca
import numpy as np
from graph_builder import RegionGraph, chebyshev_center


@dataclass
class InterfaceQPConfig:
    delta_safe: float = 0.02
    delta_extra: float = 0.01
    lambda_s: float = 0.0


class NarrowInterfaceError(Exception):
    pass


class InterfaceQPInfeasible(Exception):
    pass


def solve_interface_refinement(
    graph: RegionGraph,
    path_regions: List[int],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    config: InterfaceQPConfig,
) -> np.ndarray:
    m = len(path_regions)
    n_pos = graph._regions_by_index[path_regions[0]].A.shape[1]
    margin = config.delta_safe + config.delta_extra + 5e-4

    for i in range(m - 1):
        ri  = path_regions[i]
        ri1 = path_regions[i + 1]
        A_int = np.vstack([graph._regions_by_index[ri].A, graph._regions_by_index[ri1].A])
        b_int = np.concatenate([graph._regions_by_index[ri].b, graph._regions_by_index[ri1].b])
        try:
            _, rho_ij = chebyshev_center(A_int, b_int)
        except ValueError:
            raise NarrowInterfaceError(f"Interface {i}-{i+1} (regions {ri}/{ri1}) is empty.")
        if rho_ij < margin:
            raise NarrowInterfaceError(f"Interface {i}-{i+1} radius {rho_ij:.4f} < required {margin:.4f}.")

    n_free = m - 1
    if n_free == 0:
        z_out = np.zeros((m + 1, n_pos))
        z_out[0] = q_start
        z_out[m] = q_goal
        return z_out

    z_vars = [ca.MX.sym(f'z_{i}', n_pos) for i in range(n_free)]
    z_chain = [ca.DM(q_start)] + z_vars + [ca.DM(q_goal)]

    obj = ca.MX(0.0)
    for i in range(m):
        diff = z_chain[i + 1] - z_chain[i]
        obj = obj + ca.dot(diff, diff)
    if config.lambda_s > 0.0:
        for i in range(1, n_free):
            curv = z_chain[i + 1] - 2.0 * z_chain[i] + z_chain[i - 1]
            obj = obj + config.lambda_s * ca.dot(curv, curv)

    g_list: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    for i in range(n_free):
        z_i = z_vars[i]
        for ri in [path_regions[i], path_regions[i + 1]]:
            region = graph._regions_by_index[ri]
            for j in range(region.A.shape[0]):
                a_j = ca.DM(region.A[j, :])
                g_list.append(ca.dot(a_j, z_i) - float(region.b[j] - margin))
                lbg.append(-np.inf)
                ubg.append(0.0)

    x_sym = ca.vertcat(*z_vars)
    g_sym = ca.vertcat(*g_list) if g_list else ca.MX(0, 1)

    qp = {'x': x_sym, 'f': obj, 'g': g_sym}
    opts = {'osqp': {'verbose': False}, 'print_time': False, 'error_on_fail': False}
    solver = ca.qpsol('interface_qp', 'osqp', qp, opts)

    x0_vals = []
    for i in range(n_free):
        tau = (i + 1) / m
        x0_vals.extend((q_start * (1 - tau) + q_goal * tau).tolist())

    sol = solver(x0=x0_vals, lbg=lbg, ubg=ubg)
    stats = solver.stats()

    if not stats.get('success', False):
        raise InterfaceQPInfeasible(f"Interface QP failed: {stats.get('return_status', 'unknown')}")

    x_opt = np.array(sol['x']).flatten()
    z_out = np.zeros((m + 1, n_pos))
    z_out[0] = q_start
    z_out[m] = q_goal
    for i in range(n_free):
        z_out[i + 1] = x_opt[i * n_pos:(i + 1) * n_pos]

    return z_out
