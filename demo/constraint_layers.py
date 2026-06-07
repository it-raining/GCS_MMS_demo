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
    """
    TYPE-A Big-M: force region-local variables to zero when a region is inactive.

        |s_minus|, |s_plus| <= state_big_m * p
        |w|                 <= control_big_m * p
        delta_min*p <= delta <= delta_max * p
        rho                 <= rho_big_m * p

    Used only in the integrated one-phase relaxation.  The fixed-path NLP
    (PathNLPSolver) creates no variables for inactive regions and needs no
    activation Big-M.
    """
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
    """
    Build exact DMS defect constraints and local cost epigraphs.

    Constraints added for each region v:
        s_plus_v = F_endpoint(s_minus_v, w_v, delta_v)   [nonlinear equality]
        local_cost(s_minus_v, w_v, delta_v) <= rho_v      [epigraph]

    NOTE: We intentionally do NOT perspective-transform the nonlinear DMS defect
    equality ``s_plus = F_endpoint(s_minus, w, delta)``.

    F_endpoint integrates unicycle dynamics dx/dt = [v cos θ, v sin θ, ω] over
    [0,1] via RK4 with time-scaling by delta.  This map is nonlinear in
    (s_minus, w, delta) due to cos(θ), sin(θ), and the delta * f(x,u) product.

    Perspective transformation applies to constraints of the form A x <= b
    (convex polytope containment).  The DMS defect is a nonlinear equality,
    not a convex set containment condition.  Scaling by an activation variable
    p would produce a nonlinear expression that is neither convex nor equivalent
    to the original.

    For inactive regions the Type-A activation constraints (build_activation_constraints)
    force s_minus → 0, w → 0, delta → 0, so F_endpoint(0, 0, 0) = 0 = s_plus,
    making the defect trivially satisfied without any Big-M on the defect itself.
    """
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
    """
    TYPE-B Big-M: guard region membership and interior safety for active regions.

    For each active region v (p_v = 1):
        A_v q_endpoint - b_v <= position_big_m * (1 - p_v)  [endpoint containment]
        A_v q_mesh     - b_v <= position_big_m * (1 - p_v)  [interior safety]

    This is a standard Big-M relaxation of convex polytope containment.
    It can in principle be replaced by a perspective/homogenized form:
        A_v q_tilde_v <= b_v * p_v
    where q_tilde_v = p_v * q_v is a new scaled variable.  However, that
    substitution requires creating separate scaled position variables and
    only applies to the 2D position projection, NOT the full state.  See
    build_perspective_region_position_constraints for the GCS-style form.
    """
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
    """
    Big-M guarded interface, source, and target coupling constraints.

    For each region-region edge e = (u, v):
      TYPE-C (edge activation):
        |z_e| <= state_big_m * y_e              — force interface state to 0
      TYPE-B/C (interface containment):
        A_u z_pos - b_u <= position_big_m*(1-y_e)
        A_v z_pos - b_v <= position_big_m*(1-y_e)  — z in intersection when active
      TYPE-D (interface equality):
        |s_plus_u - z_e| <= interface_big_m*(1-y_e)
        |s_minus_v - z_e| <= interface_big_m*(1-y_e)  — continuity across edge
      TYPE-E (control continuity, optional):
        |u_exit_u - u_entry_v| <= control_big_m*(1-y_e)

    For source/target edges:
      TYPE-D (boundary conditions):
        |pos(s_minus_v) - start_pos| <= interface_big_m*(1-y_e)
        |pos(s_plus_u)  - goal_pos|  <= interface_big_m*(1-y_e)

    NOTES:
    - TYPE-C/B interface containment (z in convex set) can be replaced by a
      perspective form: A_u z_tilde_e <= b_u*y_e, A_v z_tilde_e <= b_v*y_e.
      See build_perspective_edge_position_constraints.
    - TYPE-D interface equality Big-M (s_plus = z when y=1) is harder to
      replace because s_plus = F_endpoint(s_minus, w, delta) is nonlinear.
      The cleanest removal is path-first decomposition (TwoStageGCSDMSSolver).
    - TYPE-E control continuity Big-M can be replaced by a direct equality in
      the fixed-path NLP (build_fixed_path_coupling_constraints).
    """
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


# ---------------------------------------------------------------------------
# GCS-style perspective / homogenized constraints
# ---------------------------------------------------------------------------

def build_perspective_region_position_constraints(
    region_nodes: Sequence[str],
    p_vars: Dict[str, ca.MX],
    q_tilde_vars: Dict[str, ca.MX],
    graph: RegionGraph,
    boundary_tolerance: float = 0.0,
) -> ConstraintTriplet:
    """
    GCS-style homogenized (perspective) region position containment.

    Replaces the TYPE-B Big-M constraint:
        A_v q - b_v <= M(1 - p_v)
    with the scaled (perspective) form:
        A_v q_tilde_v <= b_v * p_v
    where q_tilde_v is a SEPARATE scaled position decision variable satisfying
    q_tilde_v = p_v * q_v.

    This formulation is convex in (q_tilde_v, p_v) when C_v = {q | A_v q <= b_v}
    is a convex polytope, because it is a linear constraint on the scaled variable.

    Requirements and limitations:
    - q_tilde_vars must be separate 2D position variables (not full state vectors).
      For full state s = [px, py, theta], only the position projection
      q = [px, py] should appear here.
    - For angular components (theta), use separate physical box bounds.
    - C_v must contain the origin, or a shifted version of this formulation must
      be used; verify before applying.
    - The caller is responsible for linking q_tilde to p and q via
      additional constraints (e.g. q_tilde = p * q in a McCormick relaxation
      or by treating q_tilde as the primary variable).

    NOTE: This function implements the geometry containment layer only.
    It does NOT replace the nonlinear DMS defect s_plus = F_endpoint(s_minus, w, delta),
    which is nonconvex for unicycle dynamics regardless of this substitution.
    Full integration into the one-phase MIOCP solver would require refactoring
    to introduce q_tilde as a separate variable; the current solver uses Big-M
    directly on the projected endpoint of s_minus / s_plus.
    """
    g: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    for node_id in region_nodes:
        region = graph.get_region_by_id(node_id)
        p = p_vars[node_id]
        q_tilde = q_tilde_vars[node_id]
        A_dm = ca.DM(region.A)
        b_dm = ca.DM(region.b)
        # A_v q_tilde_v <= b_v * p_v  (perspective of A_v q <= b_v)
        _append_leq(g, lbg, ubg, ca.mtimes(A_dm, q_tilde) - b_dm * p + boundary_tolerance * p)

    return g, lbg, ubg


def build_perspective_edge_position_constraints(
    edges: Sequence[Tuple[str, str]],
    y_vars: Dict[Tuple[str, str], ca.MX],
    z_tilde_vars: Dict[Tuple[str, str], ca.MX],
    graph: RegionGraph,
    boundary_tolerance: float = 0.0,
) -> ConstraintTriplet:
    """
    GCS-style homogenized (perspective) edge interface position containment.

    For edge e = (u, v), replaces the TYPE-C/B Big-M constraints:
        A_u z_pos - b_u <= M(1 - y_e)
        A_v z_pos - b_v <= M(1 - y_e)
    with the scaled (perspective) form:
        A_u z_tilde_e <= b_u * y_e
        A_v z_tilde_e <= b_v * y_e
    where z_tilde_e is a SEPARATE scaled interface position variable satisfying
    z_tilde_e = y_e * z_pos_e, giving z_tilde_e in y_e (C_u ∩ C_v).

    Requirements and limitations:
    - z_tilde_vars must be 2D position variables (not full state vectors).
    - Both C_u and C_v must contain the origin, or shifted formulations are
      needed; verify before applying.
    - This removes Big-M for TYPE-C/B interface containment only.
      TYPE-D interface equality (s_plus[u] = z, s_minus[v] = z when y_e = 1)
      still requires Big-M or path-first decomposition, because
      s_plus = F_endpoint(s_minus, w, delta) is nonlinear.
    - For the orientation component (theta), use physical bounds or the
      fixed-path NLP approach.

    NOTE: For nonlinear DMS with unicycle dynamics, the cleanest way to remove
    ALL interface Big-M (including TYPE-D equality) is path-first decomposition
    (TwoStageGCSDMSSolver), not perspective transformation of the defect.
    """
    g: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    for edge in edges:
        u, v = edge
        y = y_vars[edge]
        z_tilde = z_tilde_vars[edge]
        for region_id in (u, v):
            region = graph.get_region_by_id(region_id)
            A_dm = ca.DM(region.A)
            b_dm = ca.DM(region.b)
            _append_leq(g, lbg, ubg, ca.mtimes(A_dm, z_tilde) - b_dm * y + boundary_tolerance * y)

    return g, lbg, ubg


def build_barrier_log_terms(
    path_regions,
    graph,
    dynamics,
    x_node_vars,
    delta_list,
    n_int: int,
    delta_safe: float,
):
    """
    Compute B = -sum_{i,j,k} h_k * log(s_{i,j,k}) as a CasADi expression.
    Trapezoidal weights: h_k = delta_i/(2*n_int) at endpoints, delta_i/n_int interior.
    """
    B = ca.MX(0.0)
    for seg_idx, region_idx in enumerate(path_regions):
        region = graph.regions[region_idx]
        delta_i = delta_list[seg_idx]
        states_i = x_node_vars[seg_idx]
        for k in range(n_int + 1):
            h_k = delta_i / (2.0 * n_int) if (k == 0 or k == n_int) else delta_i / n_int
            pos_k = dynamics.project_to_position_casadi(states_i[k])
            for j in range(region.A.shape[0]):
                s_ijk = float(region.b[j]) - delta_safe - float(region.A[j, 0]) * pos_k[0] - float(region.A[j, 1]) * pos_k[1]
                B = B - h_k * ca.log(s_ijk)
    return B


def compute_lipschitz_safety_gap(
    dynamics,
    path_regions,
    graph,
    delta_arr,
    n_int: int,
) -> float:
    """Return L_s * h_rk4_max / 2 — worst-case safety slack deviation between RK4 nodes."""
    import numpy as _np
    A_list = [graph.regions[ri].A for ri in path_regions]
    L_s = dynamics.compute_lipschitz_bound(A_list)
    h_rk4_max = float(_np.max(delta_arr)) / n_int
    return L_s * h_rk4_max / 2.0
