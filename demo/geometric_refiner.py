"""
geometric_refiner.py - Interface QP for Centroid-Refine-DMS.

Guarantees s_{i,j,k}(warm-start) >= delta_extra (strict-interior invariant).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import casadi as ca
import numpy as np

from graph_builder import RegionGraph, chebyshev_center
from graph_types import region_node_label


@dataclass
class InterfaceQPConfig:
    delta_safe: float = 0.02
    delta_extra: float = 0.01
    lambda_s: float = 0.0


class NarrowInterfaceError(Exception):
    """Interface Chebyshev radius < delta_safe + delta_extra."""
    pass


class InterfaceQPInfeasible(Exception):
    """QP solver found no feasible point."""
    pass


def solve_interface_refinement(
    graph: RegionGraph,
    path_regions: List[int],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    config: InterfaceQPConfig,
) -> np.ndarray:
    """
    Solve interface refinement QP for a region path.

    Returns z of shape (m+2, n_pos):
      z[0]=q_start (fixed), z[1..m]=free interface points, z[m+1]=q_goal (fixed).

    Each z[i] lies strictly inside shrunken region path_regions[i-1] with
    margin >= delta_safe + delta_extra, plus region path_regions[i] for i < m.

    Raises NarrowInterfaceError if pre-solve check fails.
    Raises InterfaceQPInfeasible if QP solver fails.
    """
    m = len(path_regions)
    n_pos = graph.regions[path_regions[0]].A.shape[1]
    margin = config.delta_safe + config.delta_extra

    def _interface_hrep(ri: int, ri1: int):
        u_id = region_node_label(ri)
        v_id = region_node_label(ri1)
        isect = graph.intersections.get((u_id, v_id))
        if isect is not None:
            return isect.A, isect.b
        A = np.vstack([graph.regions[ri].A, graph.regions[ri1].A])
        b = np.concatenate([graph.regions[ri].b, graph.regions[ri1].b])
        return A, b

    # Pre-solve narrow-interface check using intersection polygon H-rep
    for i in range(m - 1):
        ri, ri1 = path_regions[i], path_regions[i + 1]
        A_int, b_int = _interface_hrep(ri, ri1)
        try:
            _, rho_ij = chebyshev_center(A_int, b_int)
        except ValueError:
            raise NarrowInterfaceError(f"Interface {i}-{i+1} (regions {ri}/{ri1}) is empty.")
        if rho_ij < margin:
            raise NarrowInterfaceError(
                f"Interface {i}-{i+1} radius {rho_ij:.4f} < required {margin:.4f}."
            )

    z_vars = [ca.MX.sym(f'z_{i}', n_pos) for i in range(m)]
    z_chain = [ca.DM(q_start)] + z_vars + [ca.DM(q_goal)]

    obj = ca.MX(0.0)
    for i in range(m + 1):
        diff = z_chain[i + 1] - z_chain[i]
        obj = obj + ca.dot(diff, diff)
    if config.lambda_s > 0.0:
        for i in range(1, m):
            curv = z_chain[i + 1] - 2.0 * z_chain[i] + z_chain[i - 1]
            obj = obj + config.lambda_s * ca.dot(curv, curv)

    g_list: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    for i in range(m):
        z_i = z_vars[i]
        if i < m - 1:
            # Interface point: constrain to intersection polygon H-rep
            A_c, b_c = _interface_hrep(path_regions[i], path_regions[i + 1])
            constr = ca.mtimes(ca.DM(A_c), z_i) - (ca.DM(b_c.reshape(-1, 1)) - margin)
            for j in range(A_c.shape[0]):
                g_list.append(constr[j])
                lbg.append(-np.inf)
                ubg.append(0.0)
        else:
            # Last waypoint: constrain to last region only
            region_last = graph.regions[path_regions[i]]
            A_l = ca.DM(region_last.A)
            b_l = ca.DM(region_last.b.reshape(-1, 1))
            constr = ca.mtimes(A_l, z_i) - (b_l - margin)
            for j in range(region_last.A.shape[0]):
                g_list.append(constr[j])
                lbg.append(-np.inf)
                ubg.append(0.0)

    x_sym = ca.vertcat(*z_vars)
    g_sym = ca.vertcat(*g_list) if g_list else ca.MX(0, 1)

    qp = {'x': x_sym, 'f': obj, 'g': g_sym}
    opts = {'osqp': {'verbose': False}, 'print_time': False, 'error_on_fail': False}
    solver = ca.qpsol('interface_qp', 'osqp', qp, opts)

    x0_vals = []
    for i in range(m):
        tau = (i + 1) / (m + 1)
        x0_vals.extend((q_start * (1 - tau) + q_goal * tau).tolist())

    sol = solver(x0=x0_vals, lbg=lbg, ubg=ubg)
    stats = solver.stats()

    if not stats.get('success', False):
        raise InterfaceQPInfeasible(f"Interface QP failed: {stats.get('return_status', 'unknown')}")

    x_opt = np.array(sol['x']).flatten()

    z_out = np.zeros((m + 2, n_pos))
    z_out[0] = q_start
    z_out[m + 1] = q_goal
    for i in range(m):
        z_out[i + 1] = x_opt[i * n_pos:(i + 1) * n_pos]

    return z_out
