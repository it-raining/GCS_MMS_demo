"""
constraint_layers.py - Reusable CasADi constraint builders for the demo solver.

Each builder returns `(g_exprs, lbg, ubg)` and has no side effects on decision
variables or solver state. The optimizer owns variable creation and parsing;
this module owns the repeated mathematical constraint patterns.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import casadi as ca
import numpy as np

from dynamics import ControlParameterization, DynamicsModel
from graph_builder import RegionGraph


ConstraintTriplet = Tuple[List[ca.MX], List[float], List[float]]


def _as_column(expr) -> ca.MX:
    """Return a CasADi expression as a column vector."""
    if not hasattr(expr, "numel"):
        expr = ca.MX(expr)
    return ca.reshape(expr, int(expr.numel()), 1)


def _append_eq(g: List[ca.MX], lbg: List[float], ubg: List[float], expr) -> None:
    expr = _as_column(expr)
    g.append(expr)
    lbg.extend([0.0] * int(expr.numel()))
    ubg.extend([0.0] * int(expr.numel()))


def _append_leq(g: List[ca.MX], lbg: List[float], ubg: List[float], expr) -> None:
    expr = _as_column(expr)
    g.append(expr)
    lbg.extend([-np.inf] * int(expr.numel()))
    ubg.extend([0.0] * int(expr.numel()))


def interior_mesh_indices(n_mesh: int) -> range:
    """Return mesh indices that exclude segment endpoints."""
    if n_mesh <= 2:
        return range(0)
    return range(1, n_mesh - 1)


def merge_constraint_layers(layers: Iterable[ConstraintTriplet]) -> ConstraintTriplet:
    """Concatenate multiple constraint-layer outputs."""
    g: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    for layer_g, layer_lbg, layer_ubg in layers:
        g.extend(layer_g)
        lbg.extend(layer_lbg)
        ubg.extend(layer_ubg)

    return g, lbg, ubg


def stack_constraints(g: Sequence[ca.MX]) -> ca.MX:
    """Stack constraints into a single CasADi vector."""
    return ca.vertcat(*g) if g else ca.MX.zeros(0, 1)


def build_network_flow_constraints(y_vars: Dict[Tuple[str, str], ca.MX],
                                   p_vars: Dict[str, ca.MX],
                                   source_edges: Sequence[Tuple[str, str]],
                                   target_edges: Sequence[Tuple[str, str]],
                                   region_nodes: Sequence[str],
                                   edges: Sequence[Tuple[str, str]]) -> ConstraintTriplet:
    """Build the source-target unit-flow and region activation constraints."""
    g: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    source_flow = ca.sum1(ca.vertcat(*[y_vars[edge] for edge in source_edges]))
    target_flow = ca.sum1(ca.vertcat(*[y_vars[edge] for edge in target_edges]))
    _append_eq(g, lbg, ubg, source_flow - 1.0)
    _append_eq(g, lbg, ubg, target_flow - 1.0)

    for node_id in region_nodes:
        in_edges = [edge for edge in edges if edge[1] == node_id]
        out_edges = [edge for edge in edges if edge[0] == node_id]

        in_flow = ca.sum1(ca.vertcat(*[y_vars[edge] for edge in in_edges])) if in_edges else 0.0
        out_flow = ca.sum1(ca.vertcat(*[y_vars[edge] for edge in out_edges])) if out_edges else 0.0

        _append_eq(g, lbg, ubg, in_flow - p_vars[node_id])
        _append_eq(g, lbg, ubg, out_flow - p_vars[node_id])

    return g, lbg, ubg


def build_fixed_path_dynamics_constraints(path_regions: Sequence[int],
                                          s_minus_vars: Sequence[ca.MX],
                                          s_plus_vars: Sequence[ca.MX],
                                          w_vars: Sequence[ca.MX],
                                          delta_vars: Sequence[ca.MX],
                                          F_endpoint: Callable) -> ConstraintTriplet:
    """Build exact multiple-shooting defect constraints for a fixed path."""
    g: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    for i, _region_idx in enumerate(path_regions):
        defect = s_plus_vars[i] - F_endpoint(
            s_minus_vars[i],
            w_vars[i],
            delta_vars[i],
        )
        _append_eq(g, lbg, ubg, defect)

    return g, lbg, ubg


def build_fixed_path_geometry_constraints(graph: RegionGraph,
                                          dynamics: DynamicsModel,
                                          path_regions: Sequence[int],
                                          s_minus_vars: Sequence[ca.MX],
                                          s_plus_vars: Sequence[ca.MX],
                                          w_vars: Sequence[ca.MX],
                                          delta_vars: Sequence[ca.MX],
                                          mesh_sampler: Callable,
                                          n_mesh: int,
                                          safety_margin: float,
                                          boundary_tolerance: float) -> ConstraintTriplet:
    """Build closed-endpoint and interior safety constraints for a fixed path."""
    g: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    for i, region_idx in enumerate(path_regions):
        region = graph.regions[region_idx]
        A_dm = ca.DM(region.A)
        b_dm = ca.DM(region.b)

        for endpoint in (s_minus_vars[i], s_plus_vars[i]):
            endpoint_pos = dynamics.project_to_position_casadi(endpoint)
            closure_violation = ca.mtimes(A_dm, endpoint_pos) - b_dm - boundary_tolerance
            _append_leq(g, lbg, ubg, closure_violation)

        mesh_positions = mesh_sampler(s_minus_vars[i], w_vars[i], delta_vars[i])
        for k in interior_mesh_indices(n_mesh):
            pos = mesh_positions[k, :]
            violation = ca.mtimes(A_dm, pos.T) - b_dm + safety_margin
            _append_leq(g, lbg, ubg, violation)

    return g, lbg, ubg


def build_fixed_path_cbf_safety_constraints(graph: RegionGraph,
                                            path_regions: Sequence[int],
                                            s_minus_vars: Sequence[ca.MX],
                                            w_vars: Sequence[ca.MX],
                                            delta_vars: Sequence[ca.MX],
                                            cbf_sampler: Callable,
                                            n_mesh: int,
                                            cbf_alpha: float) -> ConstraintTriplet:
    """
    Build discrete CBF safety constraints for a fixed path.

    For each halfspace h_j(q) = b_j - a_j^T q, enforce
    a_j^T Delta f_pos(x, u) <= alpha h_j(q) at interior mesh points.
    """
    g: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    if cbf_alpha <= 0.0:
        return g, lbg, ubg

    for i, region_idx in enumerate(path_regions):
        region = graph.regions[region_idx]
        A_dm = ca.DM(region.A)
        b_dm = ca.DM(region.b)

        mesh_positions, mesh_velocities = cbf_sampler(
            s_minus_vars[i],
            w_vars[i],
            delta_vars[i],
        )
        for k in interior_mesh_indices(n_mesh):
            pos = mesh_positions[k, :].T
            vel = mesh_velocities[k, :].T
            h = b_dm - ca.mtimes(A_dm, pos)
            normalized_boundary_rate = delta_vars[i] * ca.mtimes(A_dm, vel)
            _append_leq(g, lbg, ubg, normalized_boundary_rate - cbf_alpha * h)

    return g, lbg, ubg


def build_fixed_path_coupling_constraints(s_minus_vars: Sequence[ca.MX],
                                          s_plus_vars: Sequence[ca.MX],
                                          w_vars: Sequence[ca.MX],
                                          control_param: ControlParameterization,
                                          enforce_control_continuity: bool) -> ConstraintTriplet:
    """Build equality coupling between consecutive regions on a fixed path."""
    g: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    for i in range(len(s_minus_vars) - 1):
        _append_eq(g, lbg, ubg, s_plus_vars[i] - s_minus_vars[i + 1])

        if enforce_control_continuity:
            u_exit = control_param.evaluate_casadi(1.0, w_vars[i])
            u_entry = control_param.evaluate_casadi(0.0, w_vars[i + 1])
            _append_eq(g, lbg, ubg, u_exit - u_entry)

    return g, lbg, ubg


def build_fixed_path_boundary_constraints(dynamics: DynamicsModel,
                                          s_minus_first: ca.MX,
                                          s_plus_last: ca.MX,
                                          start_state: np.ndarray,
                                          goal_state: np.ndarray) -> ConstraintTriplet:
    """Build fixed start and goal position constraints for a fixed path."""
    g: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    start_pos = dynamics.project_to_position(start_state)
    goal_pos = dynamics.project_to_position(goal_state)
    _append_eq(g, lbg, ubg, dynamics.project_to_position_casadi(s_minus_first) - start_pos)
    _append_eq(g, lbg, ubg, dynamics.project_to_position_casadi(s_plus_last) - goal_pos)

    return g, lbg, ubg


def build_activation_constraints(region_nodes: Sequence[str],
                                 p_vars: Dict[str, ca.MX],
                                 s_minus_vars: Dict[str, ca.MX],
                                 s_plus_vars: Dict[str, ca.MX],
                                 w_vars: Dict[str, ca.MX],
                                 delta_vars: Dict[str, ca.MX],
                                 rho_vars: Dict[str, ca.MX],
                                 state_big_m: np.ndarray,
                                 control_big_m: np.ndarray,
                                 delta_min: float,
                                 delta_max: float,
                                 rho_big_m: float) -> ConstraintTriplet:
    """Force region-local variables to zero when a region is inactive."""
    g: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    for node_id in region_nodes:
        p = p_vars[node_id]

        for state_var in (s_minus_vars[node_id], s_plus_vars[node_id]):
            _append_leq(g, lbg, ubg, state_var - state_big_m * p)
            _append_leq(g, lbg, ubg, -state_var - state_big_m * p)

        _append_leq(g, lbg, ubg, w_vars[node_id] - control_big_m * p)
        _append_leq(g, lbg, ubg, -w_vars[node_id] - control_big_m * p)
        _append_leq(g, lbg, ubg, delta_vars[node_id] - delta_max * p)
        _append_leq(g, lbg, ubg, delta_min * p - delta_vars[node_id])
        _append_leq(g, lbg, ubg, rho_vars[node_id] - rho_big_m * p)

    return g, lbg, ubg


def build_integrated_dynamics_constraints(region_nodes: Sequence[str],
                                          s_minus_vars: Dict[str, ca.MX],
                                          s_plus_vars: Dict[str, ca.MX],
                                          w_vars: Dict[str, ca.MX],
                                          delta_vars: Dict[str, ca.MX],
                                          rho_vars: Dict[str, ca.MX],
                                          F_endpoint: Callable,
                                          local_cost_fn: Callable,
                                          cost_weights: Tuple[float, float, float, float]) -> ConstraintTriplet:
    """Build exact defect constraints and local cost epigraphs."""
    g: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []
    a, w_L, w_E, w_u_smooth = cost_weights

    for node_id in region_nodes:
        s_minus = s_minus_vars[node_id]
        w = w_vars[node_id]
        delta = delta_vars[node_id]

        _append_eq(g, lbg, ubg, s_plus_vars[node_id] - F_endpoint(s_minus, w, delta))
        local_cost = local_cost_fn(s_minus, w, delta, a, w_L, w_E, w_u_smooth)
        _append_leq(g, lbg, ubg, local_cost - rho_vars[node_id])

    return g, lbg, ubg


def build_integrated_geometry_constraints(graph: RegionGraph,
                                          dynamics: DynamicsModel,
                                          region_nodes: Sequence[str],
                                          p_vars: Dict[str, ca.MX],
                                          s_minus_vars: Dict[str, ca.MX],
                                          s_plus_vars: Dict[str, ca.MX],
                                          w_vars: Dict[str, ca.MX],
                                          delta_vars: Dict[str, ca.MX],
                                          mesh_sampler: Callable,
                                          n_mesh: int,
                                          safety_margin: float,
                                          boundary_tolerance: float,
                                          position_big_m: float) -> ConstraintTriplet:
    """Build Big-M guarded region membership and safety constraints."""
    g: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    for node_id in region_nodes:
        region = graph.get_region_by_id(node_id)
        p = p_vars[node_id]
        A_dm = ca.DM(region.A)
        b_dm = ca.DM(region.b)

        for endpoint in (s_minus_vars[node_id], s_plus_vars[node_id]):
            endpoint_pos = dynamics.project_to_position_casadi(endpoint)
            closure_violation = ca.mtimes(A_dm, endpoint_pos) - b_dm - boundary_tolerance
            _append_leq(g, lbg, ubg, closure_violation - position_big_m * (1 - p))

        mesh_positions = mesh_sampler(
            s_minus_vars[node_id],
            w_vars[node_id],
            delta_vars[node_id],
        )
        for k in interior_mesh_indices(n_mesh):
            pos = mesh_positions[k, :]
            violation = ca.mtimes(A_dm, pos.T) - b_dm + safety_margin
            _append_leq(g, lbg, ubg, violation - position_big_m * (1 - p))

    return g, lbg, ubg


def build_integrated_coupling_constraints(graph: RegionGraph,
                                          dynamics: DynamicsModel,
                                          y_vars: Dict[Tuple[str, str], ca.MX],
                                          s_minus_vars: Dict[str, ca.MX],
                                          s_plus_vars: Dict[str, ca.MX],
                                          w_vars: Dict[str, ca.MX],
                                          z_vars: Dict[Tuple[str, str], ca.MX],
                                          control_param: ControlParameterization,
                                          state_big_m: np.ndarray,
                                          position_big_m: float,
                                          interface_big_m: float,
                                          control_value_big_m: np.ndarray,
                                          boundary_tolerance: float,
                                          start_state: np.ndarray,
                                          goal_state: np.ndarray,
                                          enforce_control_continuity: bool) -> ConstraintTriplet:
    """Build Big-M guarded interface, source, and target coupling constraints."""
    g: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    for edge in graph.region_edges:
        u, v = edge
        y = y_vars[edge]
        z = z_vars[edge]

        _append_leq(g, lbg, ubg, z - state_big_m * y)
        _append_leq(g, lbg, ubg, -z - state_big_m * y)

        z_pos = dynamics.project_to_position_casadi(z)
        for region in (graph.get_region_by_id(u), graph.get_region_by_id(v)):
            A_dm = ca.DM(region.A)
            b_dm = ca.DM(region.b)
            closure_violation = ca.mtimes(A_dm, z_pos) - b_dm - boundary_tolerance
            _append_leq(g, lbg, ubg, closure_violation - position_big_m * (1 - y))

        diff_upstream = s_plus_vars[u] - z
        diff_downstream = s_minus_vars[v] - z
        _append_leq(g, lbg, ubg, diff_upstream - interface_big_m * (1 - y))
        _append_leq(g, lbg, ubg, -diff_upstream - interface_big_m * (1 - y))
        _append_leq(g, lbg, ubg, diff_downstream - interface_big_m * (1 - y))
        _append_leq(g, lbg, ubg, -diff_downstream - interface_big_m * (1 - y))

        if enforce_control_continuity:
            u_exit = control_param.evaluate_casadi(1.0, w_vars[u])
            u_entry = control_param.evaluate_casadi(0.0, w_vars[v])
            control_diff = u_exit - u_entry
            _append_leq(g, lbg, ubg, control_diff - control_value_big_m * (1 - y))
            _append_leq(g, lbg, ubg, -control_diff - control_value_big_m * (1 - y))

    start_pos = dynamics.project_to_position(start_state)
    goal_pos = dynamics.project_to_position(goal_state)

    for edge in graph.source_edges:
        _, v = edge
        y = y_vars[edge]
        diff = dynamics.project_to_position_casadi(s_minus_vars[v]) - start_pos
        _append_leq(g, lbg, ubg, diff - interface_big_m * (1 - y))
        _append_leq(g, lbg, ubg, -diff - interface_big_m * (1 - y))

    for edge in graph.target_edges:
        u, _ = edge
        y = y_vars[edge]
        diff = dynamics.project_to_position_casadi(s_plus_vars[u]) - goal_pos
        _append_leq(g, lbg, ubg, diff - interface_big_m * (1 - y))
        _append_leq(g, lbg, ubg, -diff - interface_big_m * (1 - y))

    return g, lbg, ubg


def build_barrier_log_terms(
    path_regions,
    graph,
    dynamics,
    x_node_vars,
    delta_list,
    n_int: int,
    delta_safe: float,
    endpoint_delta_safe=None,
    node_delta_safe=None,
):
    """Compute B = -Sigma_{i,j,k} h_k * log(s_{i,j,k}) as a CasADi expression."""
    B = ca.MX(0.0)
    for seg_idx, region_idx in enumerate(path_regions):
        region = graph._regions_by_index[region_idx]
        delta_i = delta_list[seg_idx]
        states_i = x_node_vars[seg_idx]
        for k in range(n_int + 1):
            if k == 0 or k == n_int:
                h_k = delta_i / (2.0 * n_int)
            else:
                h_k = delta_i / n_int
            x_k = states_i[k]
            pos_k = dynamics.project_to_position_casadi(x_k)
            margin_k = delta_safe
            if node_delta_safe is not None:
                margin_k = float(node_delta_safe[seg_idx, k])
            elif endpoint_delta_safe is not None and (
                (seg_idx == 0 and k == 0)
                or (seg_idx == len(path_regions) - 1 and k == n_int)
            ):
                margin_k = endpoint_delta_safe
            for j in range(region.A.shape[0]):
                a_j = region.A[j, :]
                b_j = float(region.b[j])
                s_ijk = b_j - margin_k - float(a_j[0]) * pos_k[0] - float(a_j[1]) * pos_k[1]
                B = B - h_k * ca.log(s_ijk)
    return B


def compute_lipschitz_safety_gap(
    dynamics,
    path_regions,
    graph,
    delta_arr,
    n_int: int,
) -> float:
    """Return L_s * h_rk4_max / 2."""
    A_list = [graph._regions_by_index[ri].A for ri in path_regions]
    L_s = dynamics.compute_lipschitz_bound(A_list)
    h_rk4_max = float(np.max(delta_arr)) / n_int
    return L_s * h_rk4_max / 2.0
