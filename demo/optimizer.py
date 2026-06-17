"""
optimizer.py - Integrated MIOCP optimizer for GCS-MMS motion planning.

Implements the four-layer constraint structure:
1. Discrete path layer (network flow)
2. Convex on/off geometry layer (Big-M approximation of perspective)
3. On/off coupling layer (interface matching)
4. Nonlinear multiple-shooting dynamics layer

Solver strategy:
- One-phase integrated MIOCP continuous relaxation with IPOPT
- Optional fixed-path NLP polish after relaxed path extraction

Classification: MIOCP -> MINLP after transcription
Solvers: CasADi + IPOPT
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field, replace
import casadi as ca
import time

from graph_builder import RegionGraph, SOURCE, TARGET
from graph_types import region_index_from_node_id, is_terminal_node_id, region_node_label
from constraint_layers import (
    merge_constraint_layers,
    build_fixed_path_geometry_constraints,
    build_fixed_path_cbf_safety_constraints,
    build_fixed_path_coupling_constraints,
    build_fixed_path_boundary_constraints,
)
from dynamics import (DynamicsModel, ControlParameterization,
                      create_casadi_integrator, create_casadi_trajectory_sampler,
                      create_casadi_ctcs_integrator, RK4Integrator,
                      create_integration_bundle)
from shooting import create_casadi_local_cost


def _tile_control_vector(u: np.ndarray, control_param: "ControlParameterization") -> np.ndarray:
    """Tile a single per-segment control vector into the full w vector (n_segments copies)."""
    return np.tile(u, control_param.n_segments)


def _state_guess_from_position(
    dynamics: "DynamicsModel",
    position: np.ndarray,
    heading: float,
    warm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a full state guess: position + heading (unicycle) or zeros for other dims."""
    if warm is not None:
        return np.asarray(warm, dtype=float)
    state = np.zeros(dynamics.n_x, dtype=float)
    pos_idx = list(dynamics.position_indices)
    state[pos_idx] = position[: len(pos_idx)]
    for ang_idx in dynamics.angle_indices:
        state[ang_idx] = heading
    return state


@dataclass
class OptimizationConfig:
    """Configuration for the optimizer."""
    # Cost weights
    a: float = 1.0       # Time penalty
    w_L: float = 1.0     # Velocity/path length penalty
    w_E: float = 1.0     # Control effort penalty
    w_u_smooth: float = 0.2  # Penalty on control changes across segments
    w_theta_smooth: float = 0.0  # Penalty on interface heading mismatch
    
    # Shooting parameters
    n_integration_steps: int = 20
    n_mesh_points: int = 5
    n_control_segments: int = 2
    safety_margin: float = 0.02
    cbf_alpha: float = 0.0
    safety_mode: str = "both"
    ctcs_tolerance: float = 1.0e-6
    ctcs_penalty: str = "squared_hinge"
    ctcs_integral_mode: str = "normalized"
    ctcs_use_rk4_stages: bool = True
    ctcs_eta_big_m: float = 100.0
    dense_check_points: int = 1000
    dense_check_tolerance: float = 1.0e-4
    fail_on_dense_violation: bool = True
    boundary_tolerance: float = 1e-8
    
    # Time bounds
    delta_min: float = 0.1
    active_delta_min: float = 0.05
    delta_max: float = 10.0
    
    # Big-M values (used only in the integrated legacy relaxation)
    M_position: float = 20.0
    M_interface: float = 20.0
    # M_time is declared for config backward compatibility but is NOT
    # referenced in any constraint builder.  It has no effect on the NLP.
    M_time: float = 100.0
    enforce_control_continuity: bool = True

    # IPOPT options
    max_iter: int = 3000
    tol: float = 1e-6
    print_level: int = 0
    max_polish_path_candidates: int = 20
    final_connection_tolerance: float = 1e-6
    final_defect_tolerance: float = 1e-6
    final_integrality_tolerance: float = 1e-6

    # Pipeline controls
    pipeline_mode: str = "integrated"
    path_screen_top_k: int = 5
    path_screen_max_paths: int = 1500
    early_stop_on_feasible: bool = True
    acceptable_dense_tolerance_for_repair: float = 5e-3
    screening_ipopt_tol: float = 1e-4
    screening_max_iter: int = 500
    final_ipopt_tol: float = 1e-6
    final_max_iter: int = 3000
    repair_enabled: bool = True
    repair_max_rounds: int = 2
    local_shrink_margin: float = 0.003
    adaptive_refine_enabled: bool = True
    screening_safety_mode: Optional[str] = None
    local_region_safety_margins: Dict[int, float] = field(default_factory=dict)


@dataclass
class CentroidRefineDMSConfig:
    """Configuration for the centroid_refine_dms solver."""
    gamma_w: float = 1.0
    gamma_h: float = 0.64
    delta_safe: float = 0.02
    delta_extra: float = 0.01
    epsilon_certificate_buffer: float = 0.0
    alpha_s: float = 0.0
    v_nom_fraction: float = 0.5
    barrier_levels: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.1, 0.01])
    epsilon_final: float = 1e-6
    epsilon_gap: float = 0.05
    time_limit_s: float = 60.0
    mode: str = "first_feasible"
    n_int: int = 10
    n_control_segments: int = 2
    delta_min: float = 0.1
    delta_max: float = 10.0
    w_T: float = 1.0
    w_L: float = 1.0
    w_U: float = 1.0
    w_S: float = 0.2
    use_centroid_cost: bool = True
    use_interface_qp: bool = True
    use_log_barrier: bool = True
    use_barrier_continuation: bool = True
    use_inexact_tolerance: bool = True


@dataclass
class OptimizationResult:
    """Result of optimization."""
    success: bool
    path: List[str]                          # Node IDs in path
    path_regions: List[int]                  # Region indices
    total_cost: float
    solve_time: float
    n_paths_evaluated: int
    
    # Per-region results
    entry_states: Dict[int, np.ndarray] = field(default_factory=dict)
    exit_states: Dict[int, np.ndarray] = field(default_factory=dict)
    control_params: Dict[int, np.ndarray] = field(default_factory=dict)
    time_durations: Dict[int, float] = field(default_factory=dict)
    
    # Interface points
    interface_points: List[np.ndarray] = field(default_factory=list)
    
    # Trajectory data
    trajectories: List[Tuple[np.ndarray, np.ndarray, float]] = field(default_factory=list)
    mesh_samples: List[Tuple[np.ndarray, np.ndarray, float]] = field(default_factory=list)
    
    # Diagnostics
    solver_status: str = ""
    defect_norm: float = 0.0
    constraint_violation: float = 0.0
    continuous_violation_integrals: Dict[int, float] = field(default_factory=dict)
    max_continuous_violation_integral: float = 0.0
    max_dense_region_violation: float = 0.0
    safety_mode: str = "both"
    max_connection_gap: float = 0.0
    max_control_jump: float = 0.0
    max_integrality_gap: float = 0.0
    safety_diagnostics: Dict[int, Dict] = field(default_factory=dict)
    transition_diagnostics: List[Dict] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    pipeline_mode: str = "integrated"
    candidate_paths_ranked: List[Dict] = field(default_factory=list)
    n_paths_screened: int = 0
    n_paths_polished: int = 0
    early_stopped: bool = False
    repair_attempted: bool = False
    repaired_regions: List[int] = field(default_factory=list)
    per_candidate_diagnostics: List[Dict] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)

    # Fields for centroid_refine_dms
    formulation_mode: str = "INTEGRATED_MIOCP"
    global_optimality_claim: str = ""
    min_safety_margin: float = float("nan")
    certified_safety_margin: float = float("nan")
    lipschitz_gap: float = float("nan")
    safety_certification: str = "NOT_SET"
    lb_geometric: float = 0.0
    optimality_gap: float = float("inf")
    n_barrier_levels: int = 0
    failure_log: list = field(default_factory=list)
    n_nlp_iterations: int = 0


def _compute_connection_gap(path_regions: List[int],
                            entry_states: Dict[int, np.ndarray],
                            exit_states: Dict[int, np.ndarray],
                            start_state: np.ndarray,
                            goal_state: np.ndarray,
                            position_indices: Tuple[int, ...] = (0, 1)) -> float:
    """
    Maximum 2D position jump across start, interfaces, and goal.

    This measures the continuity of the trajectory shown in the plots.
    """
    if not path_regions:
        return 0.0

    gaps = [
        np.linalg.norm(
            entry_states[path_regions[0]][list(position_indices)] -
            start_state[list(position_indices)]
        ),
        np.linalg.norm(
            exit_states[path_regions[-1]][list(position_indices)] -
            goal_state[list(position_indices)]
        ),
    ]

    for left_region, right_region in zip(path_regions[:-1], path_regions[1:]):
        gaps.append(
            np.linalg.norm(
                exit_states[left_region][list(position_indices)] -
                entry_states[right_region][list(position_indices)]
            )
        )

    return float(max(gaps)) if gaps else 0.0


def _compute_control_jump(path_regions: List[int],
                          control_params: Dict[int, np.ndarray],
                          control_param: ControlParameterization) -> float:
    """Maximum control mismatch across consecutive active regions."""
    if len(path_regions) <= 1:
        return 0.0

    jumps = []
    for left_region, right_region in zip(path_regions[:-1], path_regions[1:]):
        if left_region not in control_params or right_region not in control_params:
            continue

        u_exit = control_param.evaluate(1.0, control_params[left_region])
        u_entry = control_param.evaluate(0.0, control_params[right_region])
        jumps.append(np.linalg.norm(u_exit - u_entry))

    return float(max(jumps)) if jumps else 0.0


def _compute_bound_violation(g_val: np.ndarray,
                             lbg: List[float],
                             ubg: List[float]) -> float:
    """Maximum violation of lower/upper constraint bounds."""
    if g_val.size == 0:
        return 0.0

    lbg_arr = np.asarray(lbg, dtype=np.float64)
    ubg_arr = np.asarray(ubg, dtype=np.float64)

    low_viol = np.where(np.isfinite(lbg_arr), np.maximum(lbg_arr - g_val, 0.0), 0.0)
    high_viol = np.where(np.isfinite(ubg_arr), np.maximum(g_val - ubg_arr, 0.0), 0.0)
    return float(max(np.max(low_viol), np.max(high_viol)))


def _interior_mesh_indices(n_mesh: int) -> range:
    """Return mesh indices that exclude the segment endpoints."""
    if n_mesh <= 2:
        return range(0)
    return range(1, n_mesh - 1)


def _uses_mesh_safety(safety_mode: str) -> bool:
    return safety_mode in ("mesh", "both")


def _uses_ctcs_safety(safety_mode: str) -> bool:
    return safety_mode in ("ctcs", "both")


def _validate_safety_mode(safety_mode: str) -> str:
    if safety_mode not in ("mesh", "ctcs", "both"):
        raise ValueError(f"Unsupported shooting.safety_mode: {safety_mode}")
    return safety_mode


def _validate_pipeline_mode(pipeline_mode: str) -> str:
    if pipeline_mode not in ("integrated", "two_stage"):
        raise ValueError(f"Unsupported optimizer.pipeline_mode: {pipeline_mode}")
    return pipeline_mode


def _region_safety_margin(config: OptimizationConfig, region_idx: int) -> float:
    return float(
        config.local_region_safety_margins.get(region_idx, config.safety_margin)
    )


def _warm_start_from_result(
    result: OptimizationResult,
) -> Dict[int, Dict[str, np.ndarray | float]]:
    warm_start = {}
    for region_idx in result.path_regions:
        if (
            region_idx in result.entry_states and
            region_idx in result.exit_states and
            region_idx in result.control_params and
            region_idx in result.time_durations
        ):
            warm_start[region_idx] = {
                's_minus': result.entry_states[region_idx],
                's_plus': result.exit_states[region_idx],
                'w': result.control_params[region_idx],
                'delta': result.time_durations[region_idx],
            }
    return warm_start


def _get_graph_edge_anchor(graph: RegionGraph,
                           edge: Tuple[str, str],
                           start_state: np.ndarray,
                           goal_state: np.ndarray) -> np.ndarray:
    """Representative 2D anchor point for a graph edge."""
    u, v = edge

    if u == SOURCE:
        return np.asarray(start_state[:2], dtype=np.float64)
    if v == TARGET:
        return np.asarray(goal_state[:2], dtype=np.float64)

    intersection = graph.intersections.get(edge)
    if intersection is not None:
        return np.asarray(intersection.get_centroid(), dtype=np.float64)

    u_centroid = graph.get_region_by_id(u).get_centroid()
    v_centroid = graph.get_region_by_id(v).get_centroid()
    return 0.5 * (u_centroid + v_centroid)


def _region_dense_diagnostic(graph: RegionGraph,
                             dynamics: DynamicsModel,
                             control_param: ControlParameterization,
                             config: OptimizationConfig,
                             region_idx: int,
                             s_minus: np.ndarray,
                             w: np.ndarray,
                             delta: float) -> Dict:
    """Detailed dense post-check diagnostic for one local region segment."""
    region = graph.regions[region_idx]
    n_steps = max(int(config.dense_check_points) - 1, 1)
    dense_integrator = RK4Integrator(dynamics, control_param, n_steps=n_steps)
    traj, tau_values = dense_integrator.integrate_with_trajectory(s_minus, w, delta)

    max_violation = -np.inf
    worst_point = None
    worst_tau = 0.0
    worst_halfspace = -1
    n_violating_samples = 0

    for state, tau in zip(traj, tau_values):
        pos = dynamics.project_to_position(state)
        residual = region.A @ pos - region.b
        local_idx = int(np.argmax(residual))
        local_max = float(residual[local_idx])
        if local_max > config.dense_check_tolerance:
            n_violating_samples += 1
        if local_max > max_violation:
            max_violation = local_max
            worst_point = np.asarray(pos, dtype=np.float64)
            worst_tau = float(tau)
            worst_halfspace = local_idx

    return {
        'region': int(region_idx),
        'delta': float(delta),
        'max_violation': float(max_violation),
        'n_violating_samples': int(n_violating_samples),
        'worst_point': worst_point.tolist() if worst_point is not None else None,
        'worst_tau': float(worst_tau),
        'worst_halfspace': int(worst_halfspace),
    }


def _transition_diagnostics(graph: RegionGraph,
                            path_regions: List[int],
                            entry_states: Dict[int, np.ndarray],
                            exit_states: Dict[int, np.ndarray]) -> List[Dict]:
    """Connection and intersection diagnostics for a fixed region path."""
    diagnostics = []
    for left_idx, right_idx in zip(path_regions[:-1], path_regions[1:]):
        left_node = f"R{left_idx}"
        right_node = f"R{right_idx}"
        z_left = exit_states[left_idx][:2]
        z_right = entry_states[right_idx][:2]
        z_mid = 0.5 * (z_left + z_right)
        gap = float(np.linalg.norm(z_left - z_right))
        region_left = graph.regions[left_idx]
        region_right = graph.regions[right_idx]
        left_violation = float(np.max(region_left.A @ z_mid - region_left.b))
        right_violation = float(np.max(region_right.A @ z_mid - region_right.b))
        diagnostics.append({
            'edge': [left_node, right_node],
            'connection_gap': gap,
            'intersection_violation': max(left_violation, right_violation),
            'z_mid': z_mid.tolist(),
        })
    return diagnostics


def _apply_final_success_criteria(result: OptimizationResult,
                                  config: OptimizationConfig,
                                  require_integral_final: bool = False,
                                  require_integrality: bool = False,
                                  solver_ok: bool = True) -> None:
    """Centralized final feasibility gate for returned trajectories."""
    reasons = []
    if not solver_ok:
        reasons.append("solver did not report acceptable success")
    if result.max_connection_gap > config.final_connection_tolerance:
        reasons.append(
            f"connection gap {result.max_connection_gap:.2e} > "
            f"{config.final_connection_tolerance:.2e}"
        )
    if result.defect_norm > config.final_defect_tolerance:
        reasons.append(
            f"defect norm {result.defect_norm:.2e} > "
            f"{config.final_defect_tolerance:.2e}"
        )
    if require_integrality and result.max_integrality_gap > config.final_integrality_tolerance:
        reasons.append(
            f"integrality gap {result.max_integrality_gap:.2e} > "
            f"{config.final_integrality_tolerance:.2e}"
        )
    if result.max_dense_region_violation > config.dense_check_tolerance:
        reasons.append(
            f"dense violation {result.max_dense_region_violation:.2e} > "
            f"{config.dense_check_tolerance:.2e}"
        )
    if require_integral_final and result.max_continuous_violation_integral > config.ctcs_tolerance:
        reasons.append(
            f"CTCS integral {result.max_continuous_violation_integral:.2e} > "
            f"{config.ctcs_tolerance:.2e}"
        )
    active_delta_min = max(config.delta_min, config.active_delta_min)
    collapsed = [
        (idx, delta)
        for idx, delta in result.time_durations.items()
        if delta < active_delta_min - 1e-8
    ]
    if collapsed:
        worst_idx, worst_delta = min(collapsed, key=lambda item: item[1])
        reasons.append(
            f"active duration collapse R{worst_idx}={worst_delta:.2e} < "
            f"{active_delta_min:.2e}"
        )

    result.failure_reasons = reasons
    if reasons:
        result.success = False
        detail = "; ".join(reasons)
        result.solver_status = (
            f"{result.solver_status} | FINAL CHECK FAILED: {detail}"
            if result.solver_status else
            f"FINAL CHECK FAILED: {detail}"
        )


class PathNLPSolver:
    """
    NLP solver for a fixed discrete path.
    
    Given a path through regions, solves the continuous trajectory
    optimization problem with:
    - Defect constraints (multiple shooting)
    - Safety constraints (mesh sampling)
    - Interface coupling constraints
    - Boundary conditions
    """
    
    def __init__(self, graph: RegionGraph, dynamics: DynamicsModel,
                 config: OptimizationConfig,
                 control_param: Optional[ControlParameterization] = None,
                 integration_bundle: Optional[CasADiIntegrationBundle] = None):
        """
        Args:
            graph: RegionGraph object
            dynamics: Dynamics model
            config: Optimization configuration
        """
        self.graph = graph
        self.dynamics = dynamics
        self.config = config
        self.config.safety_mode = _validate_safety_mode(self.config.safety_mode)
        self.config.pipeline_mode = _validate_pipeline_mode(self.config.pipeline_mode)
        
        # Control parameterization
        self.control_param = control_param or ControlParameterization(
            n_u=dynamics.n_u,
            parameterization="piecewise_constant",
            n_segments=config.n_control_segments
        )

        self.integration_bundle = integration_bundle or create_integration_bundle(
            dynamics,
            self.control_param,
            config.n_integration_steps,
            config.n_mesh_points,
        )
        self.F_endpoint = self.integration_bundle.F_endpoint
        self.mesh_sampler = self.integration_bundle.mesh_sampler
        self.cbf_sampler = self.integration_bundle.cbf_sampler
        self.local_cost_fn = self.integration_bundle.local_cost_fn
        self.ctcs_integrators = {
            region.index: create_casadi_ctcs_integrator(
                dynamics,
                self.control_param,
                region,
                config.n_integration_steps,
                _region_safety_margin(config, region.index),
                config.ctcs_penalty,
                config.ctcs_integral_mode
            )
            for region in graph.regions
        }

        if graph.regions:
            all_vertices = np.vstack([region.vertices for region in graph.regions])
            self.position_lb = np.array(
                [
                    float(np.min(all_vertices[:, 0])) - 1e-6,
                    float(np.min(all_vertices[:, 1])) - 1e-6,
                ],
                dtype=np.float64,
            )
            self.position_ub = np.array(
                [
                    float(np.max(all_vertices[:, 0])) + 1e-6,
                    float(np.max(all_vertices[:, 1])) + 1e-6,
                ],
                dtype=np.float64,
            )
        else:
            self.position_lb = np.array([-1e-6, -1e-6], dtype=np.float64)
            self.position_ub = np.array([1e-6, 1e-6], dtype=np.float64)

        self.state_lb, self.state_ub = dynamics.state_bounds(
            self.position_lb,
            self.position_ub,
        )
        self.control_lb, self.control_ub = dynamics.control_bounds()
        self.w_lb = _tile_control_vector(self.control_lb, self.control_param)
        self.w_ub = _tile_control_vector(self.control_ub, self.control_param)
        self.nominal_w = _tile_control_vector(
            dynamics.nominal_control(),
            self.control_param,
        )
        self.speed_guess_scale = (
            max(abs(float(self.control_lb[0])), abs(float(self.control_ub[0])), 1e-6)
            if self.control_lb.size
            else 1e-6
        )
    
    def solve_path(self, path: List[str], start_state: np.ndarray,
                   goal_state: np.ndarray,
                   warm_start: Optional[Dict[int, Dict[str, np.ndarray | float]]] = None,
                   ipopt_tol: Optional[float] = None,
                   max_iter: Optional[int] = None,
                   safety_mode: Optional[str] = None,
                   iteration_recorder=None,
                   ) -> OptimizationResult:
        """
        Solve NLP for a fixed path.
        
        Args:
            path: List of node IDs [source, R0, R1, ..., target]
            start_state: Initial state [px, py, theta]
            goal_state: Goal state [px, py, theta]
            
        Returns:
            OptimizationResult
        """
        original_tol = self.config.tol
        original_max_iter = self.config.max_iter
        original_safety_mode = self.config.safety_mode
        if ipopt_tol is not None:
            self.config.tol = float(ipopt_tol)
        if max_iter is not None:
            self.config.max_iter = int(max_iter)
        if safety_mode is not None:
            self.config.safety_mode = _validate_safety_mode(safety_mode)

        # Extract region sequence (excluding source/target)
        path_regions = []
        for node_id in path:
            if not is_terminal_node_id(node_id):
                path_regions.append(region_index_from_node_id(node_id))
        
        n_regions = len(path_regions)
        
        if n_regions == 0:
            self.config.tol = original_tol
            self.config.max_iter = original_max_iter
            self.config.safety_mode = original_safety_mode
            return OptimizationResult(
                success=False, path=path, path_regions=[],
                total_cost=np.inf, solve_time=0.0, n_paths_evaluated=1,
                solver_status="Empty path"
            )
        
        # Build NLP
        start_time = time.time()
        
        try:
            result = self._build_and_solve_nlp(
                path_regions,
                start_state,
                goal_state,
                warm_start=warm_start,
                iteration_recorder=iteration_recorder,
            )
            result.path = path
            result.solve_time = time.time() - start_time
            result.n_paths_evaluated = 1
            return result
            
        except Exception as e:
            return OptimizationResult(
                success=False, path=path, path_regions=path_regions,
                total_cost=np.inf, solve_time=time.time() - start_time,
                n_paths_evaluated=1, solver_status=f"Error: {str(e)}"
            )
        finally:
            self.config.tol = original_tol
            self.config.max_iter = original_max_iter
            self.config.safety_mode = original_safety_mode
    
    def _build_and_solve_nlp(self, path_regions: List[int],
                             start_state: np.ndarray,
                             goal_state: np.ndarray,
                             warm_start: Optional[Dict[int, Dict[str, np.ndarray | float]]] = None,
                             iteration_recorder=None,
                             ) -> OptimizationResult:
        """Build and solve the NLP for a path."""
        n_regions = len(path_regions)
        n_x = self.dynamics.n_x
        n_w = self.control_param.n_w
        n_mesh = self.config.n_mesh_points
        
        # =====================================================================
        # Decision variables
        # For each region v: s_v^-, w_v, Delta_v. The exit state is the
        # endpoint-map expression F_v(s_v^-, w_v, Delta_v).
        # =====================================================================
        
        # Variable vectors
        s_minus_list = []  # Entry states
        w_list = []        # Control parameters
        delta_list = []    # Time durations
        
        # Variable bounds
        lbx = []
        ubx = []
        
        # Initial guess
        x0 = []

        default_heading = np.arctan2(
            goal_state[1] - start_state[1],
            goal_state[0] - start_state[0]
        )
        
        for i, region_idx in enumerate(path_regions):
            warm = warm_start.get(region_idx) if warm_start is not None else None
            node_id = region_node_label(region_idx)
            incoming_edge = (
                (SOURCE, node_id)
                if i == 0 else
                (region_node_label(path_regions[i - 1]), node_id)
            )
            outgoing_edge = (
                (node_id, TARGET)
                if i == n_regions - 1 else
                (node_id, region_node_label(path_regions[i + 1]))
            )
            entry_pos = _get_graph_edge_anchor(
                self.graph, incoming_edge, start_state, goal_state
            )
            exit_pos = _get_graph_edge_anchor(
                self.graph, outgoing_edge, start_state, goal_state
            )
            direction = exit_pos - entry_pos
            init_theta = default_heading
            if np.linalg.norm(direction) > 1e-9:
                init_theta = float(np.arctan2(direction[1], direction[0]))
            delta_min = max(self.config.delta_min, self.config.active_delta_min)
            delta_init = max(
                delta_min,
                min(
                    self.config.delta_max,
                    float(np.linalg.norm(direction)) / self.speed_guess_scale
                )
            )
            
            # Entry state s_v^-
            s_minus = ca.MX.sym(f's_minus_{i}', n_x)
            s_minus_list.append(s_minus)
            
            lbx.extend(self.state_lb.tolist())
            ubx.extend(self.state_ub.tolist())

            s_minus_init = _state_guess_from_position(
                self.dynamics,
                entry_pos,
                init_theta,
                warm.get('s_minus') if warm is not None and 's_minus' in warm else None,
            )
            x0.extend(s_minus_init.tolist())
            
            # Control parameters w_v
            w = ca.MX.sym(f'w_{i}', n_w)
            w_list.append(w)
            
            lbx.extend(self.w_lb.tolist())
            ubx.extend(self.w_ub.tolist())
            if warm is not None and 'w' in warm:
                x0.extend(np.asarray(warm['w'], dtype=float).tolist())
            else:
                x0.extend(self.nominal_w.tolist())
            
            # Time duration Delta_v
            delta = ca.MX.sym(f'delta_{i}', 1)
            delta_list.append(delta)
            
            lbx.append(delta_min)
            ubx.append(self.config.delta_max)
            if warm is not None and 'delta' in warm:
                x0.append(max(delta_min, min(self.config.delta_max, float(warm['delta']))))
            else:
                x0.append(delta_init)
        
        # Stack all variables
        x_vars = []
        for i in range(n_regions):
            x_vars.extend([s_minus_list[i], w_list[i], delta_list[i]])
        
        x = ca.vertcat(*x_vars)
        s_plus_expr_list = [
            self.F_endpoint(s_minus_list[i], w_list[i], delta_list[i])
            for i in range(n_regions)
        ]
        
        # =====================================================================
        # Constraints
        # =====================================================================
        
        g = []      # Constraint expressions
        lbg = []    # Lower bounds
        ubg = []    # Upper bounds
        
        # -----------------------------------------------------------------
        # Layer 4: Nonlinear multiple-shooting dynamics
        # -----------------------------------------------------------------
        
        ctcs_outputs = {}
        for i, region_idx in enumerate(path_regions):
            s_minus = s_minus_list[i]
            s_plus = s_plus_expr_list[i]
            w = w_list[i]
            delta = delta_list[i]
            
            # Defect constraint: s_plus - F(s_minus, w, delta) = 0.
            # In CTCS modes, use the augmented RK4 endpoint so the defect and
            # accumulated violation integral follow the same stages.
            if _uses_ctcs_safety(self.config.safety_mode):
                F_result, eta_end = self.ctcs_integrators[region_idx](s_minus, w, delta)
                ctcs_outputs[i] = (F_result, eta_end)
            else:
                F_result = self.F_endpoint(s_minus, w, delta)
            defect = s_plus - F_result
            
            g.append(defect)
            lbg.extend([0.0] * n_x)
            ubg.extend([0.0] * n_x)
        
        # -----------------------------------------------------------------
        # Layer 2: Region geometry constraints.
        # Interior mesh points must stay strictly inside the region, while
        # entry/exit states are only required to remain in the closed set so
        # transitions across shared boundaries stay feasible.
        # -----------------------------------------------------------------
        
        g, lbg, ubg = merge_constraint_layers([
            build_fixed_path_geometry_constraints(
                self.graph,
                self.dynamics,
                path_regions,
                s_minus_list,
                s_plus_expr_list,
                w_list,
                delta_list,
                self.mesh_sampler,
                n_mesh,
                self.config.safety_margin,
                self.config.boundary_tolerance,
            ),
            build_fixed_path_cbf_safety_constraints(
                self.graph,
                path_regions,
                s_minus_list,
                w_list,
                delta_list,
                self.cbf_sampler,
                n_mesh,
                self.config.cbf_alpha,
            ),
            build_fixed_path_coupling_constraints(
                s_minus_list,
                s_plus_expr_list,
                w_list,
                self.control_param,
                self.config.enforce_control_continuity,
            ),
            build_fixed_path_boundary_constraints(
                self.dynamics,
                s_minus_list[0],
                s_plus_expr_list[-1],
                start_state,
                goal_state,
            ),
        ])

        # -----------------------------------------------------------------
        # Layer 3: On/off coupling (interface matching)
        # Since path is fixed, enforce: s_u^+ = s_v^- at interfaces
        # -----------------------------------------------------------------
        
        # Stack constraints
        g = ca.vertcat(*g)
        
        # =====================================================================
        # Objective: sum of local costs
        # =====================================================================
        
        cost = 0
        a = self.config.a
        w_L = self.config.w_L
        w_E = self.config.w_E
        w_u_smooth = self.config.w_u_smooth
        w_theta_smooth = self.config.w_theta_smooth
        
        for i in range(n_regions):
            local_cost = self.local_cost_fn(
                s_minus_list[i], w_list[i], delta_list[i],
                a, w_L, w_E, w_u_smooth
            )
            cost = cost + local_cost

        if w_theta_smooth > 0.0 and self.dynamics.angle_indices:
            for i in range(n_regions - 1):
                for angle_idx in self.dynamics.angle_indices:
                    theta_jump = s_plus_expr_list[i][angle_idx] - s_minus_list[i + 1][angle_idx]
                    cost = cost + w_theta_smooth * theta_jump ** 2
        
        # =====================================================================
        # Solve NLP
        # =====================================================================
        
        nlp = {
            'x': x,
            'f': cost,
            'g': g
        }
        
        opts = {
            'ipopt.max_iter': self.config.max_iter,
            'ipopt.tol': self.config.tol,
            'ipopt.print_level': self.config.print_level,
            'print_time': 0
        }
        if iteration_recorder is not None:
            if hasattr(iteration_recorder, "configure"):
                iteration_recorder.configure(int(x.numel()), int(g.numel()))
            opts['iteration_callback'] = iteration_recorder
        
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # Solve
        sol = solver(
            x0=x0,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg
        )
        
        # Extract solution
        x_opt = np.array(sol['x']).flatten()
        
        # Check solver status
        stats = solver.stats()
        success = bool(stats.get('success', False))
        return_status = stats.get('return_status', 'unknown')
        n_nlp_iters = int(stats.get('iter_count', 0))

        # Parse solution
        result = self._parse_solution(
            x_opt, path_regions, n_x, n_w,
            start_state, goal_state
        )

        result.success = success
        result.total_cost = float(sol['f'])
        result.solver_status = return_status
        result.n_nlp_iterations = n_nlp_iters
        result.constraint_violation = _compute_bound_violation(
            np.array(sol['g']).flatten(),
            lbg,
            ubg
        )
        self._populate_safety_diagnostics(result)
        _apply_final_success_criteria(
            result,
            self.config,
            require_integral_final=_uses_ctcs_safety(self.config.safety_mode),
            solver_ok=success,
        )
        
        return result
    
    def _compute_ctcs_eta_numpy(self, region_idx: int, s_minus: np.ndarray,
                                w: np.ndarray, delta: float) -> float:
        """Evaluate the CTCS RK4 accumulated violation integral numerically."""
        _, eta_end = self.ctcs_integrators[region_idx](s_minus, w, delta)
        return float(np.asarray(eta_end).reshape(-1)[0])

    def _compute_dense_region_violation(self, region_idx: int, s_minus: np.ndarray,
                                        w: np.ndarray, delta: float) -> float:
        """Sample a local trajectory densely and return max signed H-rep violation."""
        diagnostic = _region_dense_diagnostic(
            self.graph,
            self.dynamics,
            self.control_param,
            self.config,
            region_idx,
            s_minus,
            w,
            delta,
        )
        return float(diagnostic['max_violation'])

    def _populate_safety_diagnostics(self, result: OptimizationResult) -> None:
        """Attach CTCS and dense post-check diagnostics to a parsed result."""
        result.safety_mode = self.config.safety_mode
        ctcs_values = {}
        dense_values = []
        dense_diagnostics = {}

        for region_idx in result.path_regions:
            if region_idx not in result.entry_states:
                continue
            s_minus = result.entry_states[region_idx]
            w = result.control_params[region_idx]
            delta = result.time_durations[region_idx]
            eta = self._compute_ctcs_eta_numpy(region_idx, s_minus, w, delta)
            ctcs_values[region_idx] = eta
            dense_diag = _region_dense_diagnostic(
                self.graph,
                self.dynamics,
                self.control_param,
                self.config,
                region_idx,
                s_minus,
                w,
                delta,
            )
            dense_diag['ctcs_integral'] = eta
            dense_diagnostics[region_idx] = dense_diag
            dense_values.append(float(dense_diag['max_violation']))

        result.continuous_violation_integrals = ctcs_values
        result.safety_diagnostics = dense_diagnostics
        result.max_continuous_violation_integral = (
            float(max(ctcs_values.values())) if ctcs_values else 0.0
        )
        result.max_dense_region_violation = (
            float(max(dense_values)) if dense_values else 0.0
        )

        if result.max_dense_region_violation > self.config.dense_check_tolerance:
            warning = (
                "dense post-check violation "
                f"{result.max_dense_region_violation:.2e} > "
                f"{self.config.dense_check_tolerance:.2e}"
            )
            result.solver_status = (
                f"{result.solver_status} | WARNING: {warning}"
                if result.solver_status else
                f"WARNING: {warning}"
            )
            if self.config.fail_on_dense_violation:
                result.success = False
        result.transition_diagnostics = _transition_diagnostics(
            self.graph,
            result.path_regions,
            result.entry_states,
            result.exit_states,
        )

    def _parse_solution(self, x_opt: np.ndarray, path_regions: List[int],
                        n_x: int, n_w: int,
                        start_state: np.ndarray,
                        goal_state: np.ndarray) -> OptimizationResult:
        """Parse optimization solution into structured result."""
        result = OptimizationResult(
            success=False,
            path=[],
            path_regions=path_regions,
            total_cost=0.0,
            solve_time=0.0,
            n_paths_evaluated=1
        )
        
        # Variables per region: s_minus (n_x) + w (n_w) + delta (1).
        vars_per_region = n_x + n_w + 1
        
        for i, region_idx in enumerate(path_regions):
            offset = i * vars_per_region
            
            s_minus = x_opt[offset:offset + n_x]
            offset += n_x
            
            w = x_opt[offset:offset + n_w]
            offset += n_w
            
            delta = float(x_opt[offset])
            s_plus = np.array(
                self.F_endpoint(s_minus, w, delta),
                dtype=float,
            ).reshape(-1)
            
            result.entry_states[region_idx] = s_minus
            result.exit_states[region_idx] = s_plus
            result.control_params[region_idx] = w
            result.time_durations[region_idx] = delta
        
        # Compute trajectories for visualization
        integrator = RK4Integrator(
            self.dynamics, self.control_param,
            self.config.n_integration_steps
        )
        mesh_tau = np.linspace(0.0, 1.0, self.config.n_mesh_points)
        
        for region_idx in path_regions:
            s_minus = result.entry_states[region_idx]
            w = result.control_params[region_idx]
            delta = result.time_durations[region_idx]
            
            traj, tau = integrator.integrate_with_trajectory(s_minus, w, delta)
            result.trajectories.append((traj, tau, delta))
            mesh_positions = np.array(
                self.mesh_sampler(s_minus, w, delta), dtype=float
            )
            result.mesh_samples.append((mesh_positions, mesh_tau.copy(), delta))
        
        # Interface points
        for i in range(len(path_regions) - 1):
            region_idx = path_regions[i]
            interface_pt = self.dynamics.project_to_position(
                result.exit_states[region_idx]
            )
            result.interface_points.append(interface_pt)
        
        # Compute defect norm
        defect_norms = []
        for i, region_idx in enumerate(path_regions):
            s_minus = result.entry_states[region_idx]
            s_plus = result.exit_states[region_idx]
            w = result.control_params[region_idx]
            delta = result.time_durations[region_idx]
            
            F_result = np.array(
                self.F_endpoint(s_minus, w, delta),
                dtype=float,
            ).reshape(-1)
            defect = np.linalg.norm(s_plus - F_result)
            defect_norms.append(defect)
        
        result.defect_norm = max(defect_norms) if defect_norms else 0.0
        result.max_connection_gap = _compute_connection_gap(
            path_regions,
            result.entry_states,
            result.exit_states,
            start_state,
            goal_state,
            self.dynamics.position_indices,
        )
        result.max_control_jump = _compute_control_jump(
            path_regions,
            result.control_params,
            self.control_param
        )
        if result.time_durations:
            result.total_duration = float(sum(result.time_durations.values()))

        return result

class IntegratedMIOCPSolver:
    """
    One-phase integrated MIOCP continuous-relaxation solver.
    
    Solves the discrete graph flow and continuous multiple-shooting trajectory
    variables concurrently in a single MINLP:
    - Edge activation y_uv
    - Region activation p_v
    - Entry/exit states s_v^-, s_v^+
    - Control parameters w_v
    - Region durations Delta_v
    - Edge interface states z_uv
    - Local cost epigraphs rho_v
    
    The graph-flow variables y_uv and p_v remain in the formulation, but they
    are always relaxed to [0, 1] and solved together with the continuous
    multiple-shooting variables by IPOPT. The formulation uses Big-M on/off
    geometry and coupling constraints, while the defect dynamics and local
    cost epigraph stay exact for every region because inactive regions are
    forced to the zero state/control/time configuration.

    If the relaxed solution is noticeably fractional or discontinuous after
    path extraction, a fixed-path NLP polish is run on the extracted path so
    the returned trajectory is continuous and physically meaningful.
    """
    
    def __init__(self, graph: RegionGraph, dynamics: DynamicsModel,
                 config: OptimizationConfig):
        self.graph = graph
        self.dynamics = dynamics
        self.config = config
        self.config.safety_mode = _validate_safety_mode(self.config.safety_mode)
        self.config.pipeline_mode = _validate_pipeline_mode(self.config.pipeline_mode)
        self.path_solver = PathNLPSolver(graph, dynamics, config)
        
        self.control_param = ControlParameterization(
            n_u=dynamics.n_u,
            parameterization="piecewise_constant",
            n_segments=config.n_control_segments
        )

        self.integration_bundle = create_integration_bundle(
            dynamics,
            self.control_param,
            config.n_integration_steps,
            config.n_mesh_points,
        )
        self.F_endpoint = self.integration_bundle.F_endpoint
        self.mesh_sampler = self.integration_bundle.mesh_sampler
        self.cbf_sampler = self.integration_bundle.cbf_sampler
        self.local_cost_fn = self.integration_bundle.local_cost_fn

        self.path_solver = PathNLPSolver(
            graph,
            dynamics,
            config,
            control_param=self.control_param,
            integration_bundle=self.integration_bundle,
        )
        self.local_cost_fn = create_casadi_local_cost(
            dynamics, self.control_param, config.n_integration_steps
        )
        self.ctcs_integrators = {
            region.index: create_casadi_ctcs_integrator(
                dynamics,
                self.control_param,
                region,
                config.n_integration_steps,
                _region_safety_margin(config, region.index),
                config.ctcs_penalty,
                config.ctcs_integral_mode
            )
            for region in graph.regions
        }
        
        all_vertices = np.vstack([region.vertices for region in graph.regions])
        max_abs_pos = float(np.max(np.abs(all_vertices))) if all_vertices.size else 1.0
        self.position_big_m = max(config.M_position, max_abs_pos + 1.0)
        self.state_big_m = np.full(dynamics.n_x, self.position_big_m, dtype=np.float64)
        for angle_index in dynamics.angle_indices:
            self.state_big_m[angle_index] = dynamics.angle_big_m()

        self.control_lb, self.control_ub = dynamics.control_bounds()
        control_abs_single = np.maximum(np.abs(self.control_lb), np.abs(self.control_ub))
        control_abs_single = np.maximum(control_abs_single, 1e-6)
        self.nominal_w = _tile_control_vector(
            dynamics.nominal_control(),
            self.control_param,
        )
        self.speed_guess_scale = (
            max(abs(float(self.control_lb[0])), abs(float(self.control_ub[0])), 1e-6)
            if self.control_lb.size
            else 1e-6
        )
        self.control_big_m = _tile_control_vector(
            control_abs_single,
            self.control_param,
        )
        self.control_value_big_m = 2.0 * control_abs_single

        max_v = float(control_abs_single[0]) if control_abs_single.size else 0.0
        max_control_energy = float(np.dot(control_abs_single, control_abs_single))
        running_cost_upper = (
            config.w_L * (max_v ** 2) +
            config.w_E * max_control_energy
        )
        smoothness_upper = 0.0
        if config.n_control_segments > 1:
            max_control_jump_sq = float(np.dot(2.0 * control_abs_single, 2.0 * control_abs_single))
            smoothness_upper = (
                config.w_u_smooth *
                (config.n_control_segments - 1) *
                max_control_jump_sq
            )
        self.rho_big_m = (
            2.0 * config.delta_max * (config.a + running_cost_upper) +
            smoothness_upper +
            1.0
        )
    
    @staticmethod
    def _symbol_node_id(node_id: str) -> str:
        """Make node IDs safe for CasADi symbol names."""
        if node_id == SOURCE:
            return "source"
        if node_id == TARGET:
            return "target"
        return node_id.lower()
    
    def _edge_symbol_id(self, edge: Tuple[str, str]) -> str:
        """Make edge IDs safe for CasADi symbol names."""
        return f"{self._symbol_node_id(edge[0])}_{self._symbol_node_id(edge[1])}"
    
    def _get_edge_anchor(self, edge: Tuple[str, str],
                         start_state: np.ndarray,
                         goal_state: np.ndarray) -> np.ndarray:
        """Representative 2D anchor point for a graph edge."""
        return _get_graph_edge_anchor(self.graph, edge, start_state, goal_state)
    
    def _compute_warm_start_path(self, start_state: np.ndarray,
                                 goal_state: np.ndarray) -> List[str]:
        """Use a centroid-distance shortest path as a warm start only."""
        import networkx as nx
        
        def edge_weight(u: str, v: str, _attrs: Dict) -> float:
            if u == SOURCE:
                p1 = start_state[:2]
            else:
                p1 = self.graph.get_region_by_id(u).get_centroid()
            
            if v == TARGET:
                p2 = goal_state[:2]
            else:
                p2 = self.graph.get_region_by_id(v).get_centroid()
            
            return float(np.linalg.norm(p2 - p1))
        
        try:
            # centroid-distance shortest path as warm start
            return nx.shortest_path(self.graph.graph, SOURCE, TARGET, weight=edge_weight)
        except nx.NetworkXNoPath:
            return []

    def _iter_candidate_paths(self, start_state: np.ndarray,
                              goal_state: np.ndarray,
                              max_candidates: int,
                              edge_values: Optional[Dict[Tuple[str, str], float]] = None
                              ) -> List[List[str]]:
        """Generate source-target candidate paths, weighted by relaxed flow if available."""
        import networkx as nx

        def edge_weight(u: str, v: str, _attrs: Dict) -> float:
            flow_term = 0.0
            if edge_values is not None:
                flow = max(float(edge_values.get((u, v), 0.0)), 1e-9)
                flow_term = -np.log(flow)

            if u == SOURCE:
                p1 = start_state[:2]
            else:
                p1 = self.graph.get_region_by_id(u).get_centroid()

            if v == TARGET:
                p2 = goal_state[:2]
            else:
                p2 = self.graph.get_region_by_id(v).get_centroid()

            return float(flow_term + 1e-3 * np.linalg.norm(p2 - p1))

        candidates = []
        seen = set()
        try:
            for path in nx.shortest_simple_paths(
                self.graph.graph,
                SOURCE,
                TARGET,
                weight=edge_weight
            ):
                path_key = tuple(path)
                if path_key in seen:
                    continue
                seen.add(path_key)
                candidates.append(path)
                if len(candidates) >= max_candidates:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        return candidates

    def _score_candidate_path(self,
                              path: List[str],
                              start_state: np.ndarray,
                              goal_state: np.ndarray) -> Dict:
        """Cheap geometric score for screening fixed discrete paths."""
        edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        anchors = [
            self._get_edge_anchor(edge, start_state, goal_state)
            for edge in edges
        ]
        points = [np.asarray(start_state[:2], dtype=np.float64)] + anchors + [
            np.asarray(goal_state[:2], dtype=np.float64)
        ]

        distance = 0.0
        for p0, p1 in zip(points[:-1], points[1:]):
            distance += float(np.linalg.norm(p1 - p0))

        turn_penalty = 0.0
        for p0, p1, p2 in zip(points[:-2], points[1:-1], points[2:]):
            v0 = p1 - p0
            v1 = p2 - p1
            n0 = np.linalg.norm(v0)
            n1 = np.linalg.norm(v1)
            if n0 <= 1e-9 or n1 <= 1e-9:
                continue
            cos_angle = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
            turn_penalty += float(np.arccos(cos_angle) ** 2)

        narrow_penalty = 0.0
        for edge in edges:
            intersection = self.graph.intersections.get(edge)
            if intersection is None:
                continue
            poly = intersection.get_shapely_polygon()
            width_proxy = max(float(poly.area), float(poly.length) * 1e-3, 1e-12)
            narrow_penalty += 1.0 / np.sqrt(width_proxy)

        sliver_penalty = 0.0
        for node_id in path:
            if node_id in (SOURCE, TARGET):
                continue
            region = self.graph.get_region_by_id(node_id)
            poly = region.get_shapely_polygon()
            area = max(float(poly.area), 1e-12)
            perimeter = max(float(poly.length), 1e-12)
            compactness = 4.0 * np.pi * area / (perimeter ** 2)
            sliver_penalty += max(0.0, 0.25 - compactness) / area

        path_length = max(len(path) - 2, 0)
        score = (
            distance +
            0.05 * path_length +
            0.02 * turn_penalty +
            1e-4 * narrow_penalty +
            1e-4 * sliver_penalty
        )
        return {
            'path': path,
            'path_regions': [
                int(node_id[1:]) for node_id in path if node_id not in (SOURCE, TARGET)
            ],
            'score': float(score),
            'distance': float(distance),
            'path_length': int(path_length),
            'turn_penalty': float(turn_penalty),
            'narrow_penalty': float(narrow_penalty),
            'sliver_penalty': float(sliver_penalty),
        }

    def enumerate_rank_candidate_paths(self,
                                       start_state: np.ndarray,
                                       goal_state: np.ndarray,
                                       top_k: Optional[int] = None,
                                       max_paths: Optional[int] = None
                                       ) -> List[Dict]:
        """Enumerate simple source-target paths and rank them geometrically."""
        max_paths = int(max_paths or self.config.path_screen_max_paths)
        top_k = int(top_k or self.config.path_screen_top_k)
        paths = self._iter_candidate_paths(
            start_state,
            goal_state,
            max(max_paths, top_k),
            edge_values=None,
        )
        ranked = [
            self._score_candidate_path(path, start_state, goal_state)
            for path in paths[:max_paths]
        ]
        ranked.sort(
            key=lambda item: (
                item['score'],
                item['path_length'],
                item['distance'],
            )
        )
        for rank, item in enumerate(ranked, start=1):
            item['rank'] = rank
        return ranked[:top_k]

    def _fallback_fixed_path_search(self,
                                    start_state: np.ndarray,
                                    goal_state: np.ndarray,
                                    relaxation_result: OptimizationResult,
                                    excluded_paths: List[List[str]],
                                    edge_values: Optional[Dict[Tuple[str, str], float]] = None
                                    ) -> Optional[OptimizationResult]:
        """Try a few nearby discrete paths when the relaxed path cannot be polished."""
        excluded = {tuple(path) for path in excluded_paths}
        best_failed: Optional[OptimizationResult] = None
        evaluated = relaxation_result.n_paths_evaluated

        for candidate_path in self._iter_candidate_paths(
            start_state,
            goal_state,
            self.config.max_polish_path_candidates,
            edge_values=edge_values,
        ):
            if tuple(candidate_path) in excluded:
                continue

            candidate_result = self.path_solver.solve_path(
                candidate_path,
                start_state,
                goal_state
            )
            evaluated += 1
            candidate_result.max_integrality_gap = 0.0
            candidate_result.n_paths_evaluated = evaluated

            if candidate_result.success:
                candidate_result.solver_status = (
                    f"{relaxation_result.solver_status} + fallback fixed-path NLP: "
                    f"{candidate_result.solver_status}"
                )
                return candidate_result
            if (
                best_failed is None or
                candidate_result.max_dense_region_violation < best_failed.max_dense_region_violation
            ):
                best_failed = candidate_result

        if best_failed is not None:
            best_failed.n_paths_evaluated = evaluated
        return best_failed
    
    def _compute_constraint_violation(self, g_val: np.ndarray,
                                      lbg: List[float],
                                      ubg: List[float]) -> float:
        """Maximum bound violation across all constraints."""
        return _compute_bound_violation(g_val, lbg, ubg)

    def _compute_ctcs_eta_numpy(self, region_idx: int, s_minus: np.ndarray,
                                w: np.ndarray, delta: float) -> float:
        """Evaluate the CTCS RK4 accumulated violation integral numerically."""
        _, eta_end = self.ctcs_integrators[region_idx](s_minus, w, delta)
        return float(np.asarray(eta_end).reshape(-1)[0])

    def _compute_dense_region_violation(self, region_idx: int, s_minus: np.ndarray,
                                        w: np.ndarray, delta: float) -> float:
        """Sample a local trajectory densely and return max signed H-rep violation."""
        diagnostic = _region_dense_diagnostic(
            self.graph,
            self.dynamics,
            self.control_param,
            self.config,
            region_idx,
            s_minus,
            w,
            delta,
        )
        return float(diagnostic['max_violation'])

    def _populate_safety_diagnostics(self, result: OptimizationResult) -> None:
        """Attach CTCS and dense post-check diagnostics to a parsed result."""
        result.safety_mode = self.config.safety_mode
        ctcs_values = {}
        dense_values = []
        dense_diagnostics = []

        for region_idx in result.path_regions:
            if region_idx not in result.entry_states:
                continue
            s_minus = result.entry_states[region_idx]
            w = result.control_params[region_idx]
            delta = result.time_durations[region_idx]
            eta = self._compute_ctcs_eta_numpy(region_idx, s_minus, w, delta)
            ctcs_values[region_idx] = eta
            dense_diag = _region_dense_diagnostic(
                self.graph,
                self.dynamics,
                self.control_param,
                self.config,
                region_idx,
                s_minus,
                w,
                delta,
            )
            dense_diag['ctcs_integral'] = eta
            dense_diagnostics.append((region_idx, dense_diag))
            dense_values.append(float(dense_diag['max_violation']))

        result.continuous_violation_integrals = ctcs_values
        result.safety_diagnostics = {
            region_idx: diagnostic for region_idx, diagnostic in dense_diagnostics
        }
        result.max_continuous_violation_integral = (
            float(max(ctcs_values.values())) if ctcs_values else 0.0
        )
        result.max_dense_region_violation = (
            float(max(dense_values)) if dense_values else 0.0
        )

        if result.max_dense_region_violation > self.config.dense_check_tolerance:
            warning = (
                "dense post-check violation "
                f"{result.max_dense_region_violation:.2e} > "
                f"{self.config.dense_check_tolerance:.2e}"
            )
            result.solver_status = (
                f"{result.solver_status} | WARNING: {warning}"
                if result.solver_status else
                f"WARNING: {warning}"
            )
            if self.config.fail_on_dense_violation:
                result.success = False
        result.transition_diagnostics = _transition_diagnostics(
            self.graph,
            result.path_regions,
            result.entry_states,
            result.exit_states,
        )
    
    @staticmethod
    def _compute_integrality_gap(values: List[float]) -> float:
        """Maximum distance from the binary set {0, 1}."""
        if not values:
            return 0.0
        return float(max(min(val, 1.0 - val) for val in values))
    
    def _extract_active_path(self, edge_values: Dict[Tuple[str, str], float],
                             prefer_weighted: bool = False) -> List[str]:
        """Recover a source-target path from exact or relaxed edge decisions."""
        import networkx as nx
        
        if not prefer_weighted:
            path = [SOURCE]
            current = SOURCE
            visited = {SOURCE}
            
            while current != TARGET:
                candidates = [
                    (edge, val) for edge, val in edge_values.items()
                    if edge[0] == current and val > 0.5
                ]
                if not candidates:
                    candidates = [
                        (edge, val) for edge, val in edge_values.items()
                        if edge[0] == current and val > 1e-3
                    ]
                
                if not candidates:
                    break
                
                next_edge, _ = max(candidates, key=lambda item: item[1])
                next_node = next_edge[1]
                
                if next_node in visited and next_node != TARGET:
                    break
                
                path.append(next_node)
                current = next_node
                visited.add(next_node)
            
            if path and path[-1] == TARGET:
                return path
        
        weighted_graph = nx.DiGraph()
        weighted_graph.add_nodes_from(self.graph.graph.nodes())
        
        for edge, val in edge_values.items():
            if val <= 1e-6:
                continue
            weighted_graph.add_edge(
                edge[0],
                edge[1],
                weight=float(-np.log(max(val, 1e-9)))
            )
        
        try:
            return nx.shortest_path(weighted_graph, SOURCE, TARGET, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def _polish_relaxed_path(self, path: List[str],
                             start_state: np.ndarray,
                             goal_state: np.ndarray,
                             relaxation_result: OptimizationResult,
                             edge_values: Optional[Dict[Tuple[str, str], float]] = None
                             ) -> OptimizationResult:
        """
        Convert a relaxed integrated solution into a continuous fixed-path NLP
        solution for the extracted discrete path.
        """
        warm_start = {}
        for region_idx in relaxation_result.path_regions:
            if (
                region_idx in relaxation_result.entry_states and
                region_idx in relaxation_result.exit_states and
                region_idx in relaxation_result.control_params and
                region_idx in relaxation_result.time_durations
            ):
                warm_start[region_idx] = {
                    's_minus': relaxation_result.entry_states[region_idx],
                    's_plus': relaxation_result.exit_states[region_idx],
                    'w': relaxation_result.control_params[region_idx],
                    'delta': relaxation_result.time_durations[region_idx],
                }

        polished = self.path_solver.solve_path(
            path,
            start_state,
            goal_state,
            warm_start=warm_start or None
        )
        polished.max_integrality_gap = 0.0
        polished.n_paths_evaluated = max(relaxation_result.n_paths_evaluated, 1)
        
        if polished.success:
            polished.solver_status = (
                f"{relaxation_result.solver_status} + fixed-path NLP polish: "
                f"{polished.solver_status}"
            )
            return polished
        
        relaxation_result.solver_status = (
            f"{relaxation_result.solver_status} | fixed-path NLP polish failed: "
            f"{polished.solver_status}"
        )
        fallback_result = self._fallback_fixed_path_search(
            start_state,
            goal_state,
            relaxation_result,
            excluded_paths=[path],
            edge_values=edge_values,
        )
        if fallback_result is not None:
            if not fallback_result.success:
                fallback_result.solver_status = (
                    f"{relaxation_result.solver_status} | no feasible fixed-path "
                    f"NLP among {fallback_result.n_paths_evaluated} candidates; "
                    f"best failed candidate: {fallback_result.solver_status}"
                )
            return fallback_result

        relaxation_result.success = False
        self._populate_safety_diagnostics(relaxation_result)
        _apply_final_success_criteria(
            relaxation_result,
            self.config,
            require_integral_final=_uses_ctcs_safety(self.config.safety_mode),
            require_integrality=True,
            solver_ok=False,
        )
        return relaxation_result

    @staticmethod
    def _candidate_record(stage: str,
                          rank: int,
                          path: List[str],
                          result: OptimizationResult) -> Dict:
        return {
            'stage': stage,
            'rank': int(rank),
            'path': path,
            'path_regions': result.path_regions,
            'success': bool(result.success),
            'solve_time': float(result.solve_time),
            'status': result.solver_status,
            'cost': float(result.total_cost) if np.isfinite(result.total_cost) else None,
            'max_dense_region_violation': float(result.max_dense_region_violation),
            'max_ctcs_integral': float(result.max_continuous_violation_integral),
            'defect_norm': float(result.defect_norm),
            'constraint_violation': float(result.constraint_violation),
            'failure_reasons': result.failure_reasons,
        }

    @staticmethod
    def _candidate_sort_key(result: OptimizationResult) -> Tuple:
        return (
            0 if result.success else 1,
            max(float(result.max_dense_region_violation), 0.0),
            max(float(result.max_continuous_violation_integral), 0.0),
            float(result.total_cost) if np.isfinite(result.total_cost) else np.inf,
        )

    def _repair_fixed_path(self,
                           result: OptimizationResult,
                           start_state: np.ndarray,
                           goal_state: np.ndarray,
                           rank: int,
                           diagnostics: List[Dict],
                           verbose: bool = False) -> OptimizationResult:
        """Try local tightened fixed-path resolves around the worst dense region."""
        if not self.config.repair_enabled:
            return result
        dense_bad_enough = (
            self.config.dense_check_tolerance <
            result.max_dense_region_violation <=
            self.config.acceptable_dense_tolerance_for_repair
        )
        ctcs_near_miss = (
            result.max_dense_region_violation <= self.config.dense_check_tolerance and
            result.max_continuous_violation_integral > self.config.ctcs_tolerance and
            result.max_continuous_violation_integral <= 1.05 * self.config.ctcs_tolerance
        )
        if not dense_bad_enough and not ctcs_near_miss:
            return result

        repaired_regions: List[int] = []
        current = result
        local_margins = dict(self.config.local_region_safety_margins)

        for repair_round in range(1, int(self.config.repair_max_rounds) + 1):
            if not current.safety_diagnostics:
                break
            worst_region = max(
                current.safety_diagnostics.items(),
                key=lambda item: float(item[1].get('max_violation', -np.inf))
            )[0]
            repaired_regions.append(int(worst_region))
            local_margins[int(worst_region)] = (
                float(local_margins.get(int(worst_region), self.config.safety_margin)) +
                float(self.config.local_shrink_margin)
            )

            repair_config = replace(
                self.config,
                local_region_safety_margins=dict(local_margins),
                tol=float(self.config.final_ipopt_tol),
                max_iter=int(self.config.final_max_iter),
                safety_mode=self.config.safety_mode,
            )
            repair_solver = PathNLPSolver(self.graph, self.dynamics, repair_config)
            repaired = repair_solver.solve_path(
                current.path,
                start_state,
                goal_state,
                warm_start=_warm_start_from_result(current),
                ipopt_tol=self.config.final_ipopt_tol,
                max_iter=self.config.final_max_iter,
                safety_mode=self.config.safety_mode,
            )
            repaired.pipeline_mode = self.config.pipeline_mode
            repaired.repair_attempted = True
            repaired.repaired_regions = list(repaired_regions)
            repaired.solver_status = (
                f"{repaired.solver_status} | repair round {repair_round}, "
                f"tightened R{worst_region}"
            )
            diagnostics.append(
                self._candidate_record(
                    f"repair_round_{repair_round}",
                    rank,
                    current.path,
                    repaired,
                )
            )
            if verbose:
                print(
                    f"Repair round {repair_round}: R{worst_region}, "
                    f"dense={repaired.max_dense_region_violation:.2e}, "
                    f"ctcs={repaired.max_continuous_violation_integral:.2e}, "
                    f"success={repaired.success}"
                )
            current = repaired
            if current.success:
                break

        return current

    def _solve_two_stage(self,
                         start_state: np.ndarray,
                         goal_state: np.ndarray,
                         verbose: bool = False) -> OptimizationResult:
        total_start = time.time()
        stage_timings: Dict[str, float] = {}

        rank_start = time.time()
        ranked = self.enumerate_rank_candidate_paths(
            start_state,
            goal_state,
            top_k=self.config.path_screen_top_k,
            max_paths=self.config.path_screen_max_paths,
        )
        stage_timings['graph_screening'] = time.time() - rank_start
        if verbose:
            print(
                f"Two-stage graph screening ranked {len(ranked)} paths "
                f"in {stage_timings['graph_screening']:.3f}s"
            )

        if not ranked:
            return OptimizationResult(
                success=False,
                path=[],
                path_regions=[],
                total_cost=np.inf,
                solve_time=time.time() - total_start,
                n_paths_evaluated=0,
                solver_status="No candidate path exists in region graph",
                pipeline_mode="two_stage",
                stage_timings=stage_timings,
            )

        diagnostics: List[Dict] = []
        screening_results: List[Tuple[int, Dict, OptimizationResult]] = []
        screening_mode = self.config.screening_safety_mode or self.config.safety_mode

        for item in ranked:
            rank = int(item['rank'])
            path = item['path']
            solve_start = time.time()
            screened = self.path_solver.solve_path(
                path,
                start_state,
                goal_state,
                ipopt_tol=self.config.screening_ipopt_tol,
                max_iter=self.config.screening_max_iter,
                safety_mode=screening_mode,
            )
            stage_timings[f"screen_path_{rank}"] = time.time() - solve_start
            screened.pipeline_mode = "two_stage"
            screening_results.append((rank, item, screened))
            diagnostics.append(
                self._candidate_record("screening", rank, path, screened)
            )
            if verbose:
                print(
                    f"Screen path {rank}/{len(ranked)}: "
                    f"time={screened.solve_time:.3f}s dense="
                    f"{screened.max_dense_region_violation:.2e} "
                    f"ctcs={screened.max_continuous_violation_integral:.2e} "
                    f"success={screened.success}"
                )

        ordered = sorted(screening_results, key=lambda item: self._candidate_sort_key(item[2]))
        best: Optional[OptimizationResult] = None
        n_polished = 0
        early_stopped = False

        for rank, item, screened in ordered[:self.config.max_polish_path_candidates]:
            solve_start = time.time()
            final = self.path_solver.solve_path(
                item['path'],
                start_state,
                goal_state,
                warm_start=_warm_start_from_result(screened),
                ipopt_tol=self.config.final_ipopt_tol,
                max_iter=self.config.final_max_iter,
                safety_mode=self.config.safety_mode,
            )
            stage_timings[f"final_path_{rank}"] = time.time() - solve_start
            final.pipeline_mode = "two_stage"
            n_polished += 1
            diagnostics.append(
                self._candidate_record("final", rank, item['path'], final)
            )
            if verbose:
                print(
                    f"Final polish path {rank}: time={final.solve_time:.3f}s "
                    f"dense={final.max_dense_region_violation:.2e} "
                    f"ctcs={final.max_continuous_violation_integral:.2e} "
                    f"success={final.success}"
                )

            repaired = self._repair_fixed_path(
                final,
                start_state,
                goal_state,
                rank,
                diagnostics,
                verbose=verbose,
            )
            if repaired is not final:
                stage_timings[f"repair_path_{rank}"] = max(
                    0.0,
                    sum(
                        record['solve_time']
                        for record in diagnostics
                        if record['rank'] == rank and record['stage'].startswith('repair_round_')
                    )
                )
            candidate = repaired
            if best is None or self._candidate_sort_key(candidate) < self._candidate_sort_key(best):
                best = candidate
            if candidate.success and self.config.early_stop_on_feasible:
                early_stopped = True
                break

        if best is None:
            _, item, best_screen = ordered[0]
            best = best_screen
            best.path = item['path']

        best.pipeline_mode = "two_stage"
        best.candidate_paths_ranked = ranked
        best.n_paths_screened = len(screening_results)
        best.n_paths_polished = n_polished
        best.n_paths_evaluated = len(screening_results) + n_polished
        best.early_stopped = early_stopped
        best.per_candidate_diagnostics = diagnostics
        best.stage_timings = stage_timings
        best.solve_time = time.time() - total_start
        if best.repair_attempted:
            best.solver_status = f"{best.solver_status} | two-stage repaired"
        else:
            best.repair_attempted = any(
                record['stage'].startswith('repair_round_')
                for record in diagnostics
            )
        return best
    
    def solve(self, start_state: np.ndarray, goal_state: np.ndarray,
              verbose: bool = False) -> OptimizationResult:
        """
        Solve the motion planning problem as a single integrated continuous
        relaxation of the MIOCP.
        """
        start_time = time.time()

        if self.config.pipeline_mode == "two_stage":
            return self._solve_two_stage(start_state, goal_state, verbose=verbose)

        try:
            result = self._build_and_solve_integrated_miocp(
                start_state, goal_state, verbose
            )
        except Exception as error:
            return OptimizationResult(
                success=False,
                path=[],
                path_regions=[],
                total_cost=np.inf,
                solve_time=time.time() - start_time,
                n_paths_evaluated=1,
                solver_status=f"IntegratedMIOCP relaxation error: {error}"
            )

        result.solve_time = time.time() - start_time
        result.pipeline_mode = "integrated"
        return result

    def _build_and_solve_integrated_miocp(self, start_state: np.ndarray,
                                          goal_state: np.ndarray,
                                          verbose: bool) -> OptimizationResult:
        """Build and solve the integrated one-phase continuous relaxation."""
        from dynamics import RK4Integrator
        
        n_x = self.dynamics.n_x
        n_w = self.control_param.n_w
        n_mesh = self.config.n_mesh_points
        
        edges = list(self.graph.graph.edges())
        region_nodes = self.graph.get_region_nodes()
        warm_path = self._compute_warm_start_path(start_state, goal_state)
        
        if not warm_path:
            return OptimizationResult(
                success=False,
                path=[],
                path_regions=[],
                total_cost=np.inf,
                solve_time=0.0,
                n_paths_evaluated=1,
                solver_status="No path exists in region graph"
            )
        
        warm_edges = [(warm_path[i], warm_path[i + 1]) for i in range(len(warm_path) - 1)]
        warm_edge_set = set(warm_edges)
        warm_region_set = {node for node in warm_path if not is_terminal_node_id(node)}
        global_heading = float(np.arctan2(
            goal_state[1] - start_state[1],
            goal_state[0] - start_state[0]
        ))
        
        region_init: Dict[str, Dict[str, np.ndarray | float]] = {}
        for i, node_id in enumerate(warm_path[1:-1], start=1):
            incoming_edge = warm_edges[i - 1]
            outgoing_edge = warm_edges[i]
            
            entry_pos = self._get_edge_anchor(incoming_edge, start_state, goal_state)
            exit_pos = self._get_edge_anchor(outgoing_edge, start_state, goal_state)
            direction = exit_pos - entry_pos
            heading = global_heading
            if np.linalg.norm(direction) > 1e-9:
                heading = float(np.arctan2(direction[1], direction[0]))
            
            distance = float(np.linalg.norm(direction))
            delta_min = max(self.config.delta_min, self.config.active_delta_min)
            delta_init = max(
                delta_min,
                min(self.config.delta_max,
                    distance / self.speed_guess_scale)
            )
            
            region_init[node_id] = {
                's_minus': _state_guess_from_position(self.dynamics, entry_pos, heading),
                's_plus': _state_guess_from_position(self.dynamics, exit_pos, heading),
                'w': self.nominal_w.copy(),
                'delta': delta_init,
                'rho': max(self.config.a * delta_init, 1e-3),
            }
        
        interface_init: Dict[Tuple[str, str], np.ndarray] = {}
        for edge in self.graph.region_edges:
            z0 = np.zeros(n_x, dtype=np.float64)
            if edge in warm_edge_set:
                pos = self._get_edge_anchor(edge, start_state, goal_state)
                heading = global_heading
                if edge[1] in region_init:
                    angle_indices = self.dynamics.angle_indices
                    if angle_indices:
                        heading = float(region_init[edge[1]]['s_minus'][angle_indices[0]])
                z0 = _state_guess_from_position(self.dynamics, pos, heading)
            interface_init[edge] = z0
        
        y_vars = {}
        p_vars = {}
        s_minus_vars = {}
        s_plus_vars = {}
        w_vars = {}
        delta_vars = {}
        rho_vars = {}
        z_vars = {}
        
        var_slices = {
            'y': {},
            'p': {},
            's_minus': {},
            's_plus': {},
            'w': {},
            'delta': {},
            'rho': {},
            'z': {},
        }
        
        x_vars = []
        lbx = []
        ubx = []
        x0 = []
        discrete = []
        offset = 0
        
        def _expand(values, size: int) -> np.ndarray:
            arr = np.asarray(values, dtype=np.float64).reshape(-1)
            if arr.size == 1:
                return np.full(size, float(arr[0]), dtype=np.float64)
            if arr.size != size:
                raise ValueError(f"Expected size {size}, got {arr.size}")
            return arr
        
        def _register(symbol: ca.MX, storage_key: str, key,
                      lb, ub, init, is_discrete: bool = False):
            nonlocal offset
            size = int(symbol.numel())
            var_slices[storage_key][key] = slice(offset, offset + size)
            offset += size
            
            x_vars.append(ca.reshape(symbol, size, 1))
            lbx.extend(_expand(lb, size).tolist())
            ubx.extend(_expand(ub, size).tolist())
            x0.extend(_expand(init, size).tolist())
            discrete.extend([is_discrete] * size)
        
        for edge in edges:
            edge_id = self._edge_symbol_id(edge)
            y = ca.MX.sym(f'y_{edge_id}', 1)
            y_vars[edge] = y
            _register(y, 'y', edge, 0.0, 1.0, 1.0 if edge in warm_edge_set else 0.0, True)
        
        for node_id in region_nodes:
            node_label = self._symbol_node_id(node_id)
            active_init = node_id in warm_region_set
            init = region_init.get(node_id, None)
            
            p = ca.MX.sym(f'p_{node_label}', 1)
            p_vars[node_id] = p
            _register(p, 'p', node_id, 0.0, 1.0, 1.0 if active_init else 0.0, True)
            
            s_minus = ca.MX.sym(f's_minus_{node_label}', n_x)
            s_minus_vars[node_id] = s_minus
            _register(
                s_minus, 's_minus', node_id,
                -self.state_big_m, self.state_big_m,
                init['s_minus'] if init is not None else np.zeros(n_x)
            )
            
            s_plus = ca.MX.sym(f's_plus_{node_label}', n_x)
            s_plus_vars[node_id] = s_plus
            _register(
                s_plus, 's_plus', node_id,
                -self.state_big_m, self.state_big_m,
                init['s_plus'] if init is not None else np.zeros(n_x)
            )
            
            w = ca.MX.sym(f'w_{node_label}', n_w)
            w_vars[node_id] = w
            _register(
                w, 'w', node_id,
                -self.control_big_m, self.control_big_m,
                init['w'] if init is not None else np.zeros(n_w)
            )
            
            delta = ca.MX.sym(f'delta_{node_label}', 1)
            delta_vars[node_id] = delta
            _register(
                delta, 'delta', node_id,
                0.0, self.config.delta_max,
                init['delta'] if init is not None else 0.0
            )
            
            rho = ca.MX.sym(f'rho_{node_label}', 1)
            rho_vars[node_id] = rho
            _register(
                rho, 'rho', node_id,
                0.0, self.rho_big_m,
                init['rho'] if init is not None else 0.0
            )
        
        for edge in self.graph.region_edges:
            edge_id = self._edge_symbol_id(edge)
            z = ca.MX.sym(f'z_{edge_id}', n_x)
            z_vars[edge] = z
            _register(
                z, 'z', edge,
                -self.state_big_m, self.state_big_m,
                interface_init[edge]
            )
        
        x = ca.vertcat(*x_vars)
        
        g = []
        lbg = []
        ubg = []
        
        def add_eq(expr):
            expr = ca.reshape(expr, expr.numel(), 1)
            g.append(expr)
            lbg.extend([0.0] * expr.numel())
            ubg.extend([0.0] * expr.numel())
        
        def add_leq(expr):
            expr = ca.reshape(expr, expr.numel(), 1)
            g.append(expr)
            lbg.extend([-np.inf] * expr.numel())
            ubg.extend([0.0] * expr.numel())
        
        # Network flow constraints.
        add_eq(ca.sum1(ca.vertcat(*[y_vars[edge] for edge in self.graph.source_edges])) - 1.0)
        add_eq(ca.sum1(ca.vertcat(*[y_vars[edge] for edge in self.graph.target_edges])) - 1.0)
        
        for node_id in region_nodes:
            in_edges = [edge for edge in edges if edge[1] == node_id]
            out_edges = [edge for edge in edges if edge[0] == node_id]
            
            in_flow = ca.sum1(ca.vertcat(*[y_vars[edge] for edge in in_edges])) if in_edges else 0.0
            out_flow = ca.sum1(ca.vertcat(*[y_vars[edge] for edge in out_edges])) if out_edges else 0.0
            
            add_eq(in_flow - p_vars[node_id])
            add_eq(out_flow - p_vars[node_id])
        
        # Region-level variables are zero when the region is inactive.
        for node_id in region_nodes:
            p = p_vars[node_id]
            
            for state_var in [s_minus_vars[node_id], s_plus_vars[node_id]]:
                add_leq(state_var - self.state_big_m * p)
                add_leq(-state_var - self.state_big_m * p)
            
            add_leq(w_vars[node_id] - self.control_big_m * p)
            add_leq(-w_vars[node_id] - self.control_big_m * p)
            add_leq(delta_vars[node_id] - self.config.delta_max * p)
            add_leq(max(self.config.delta_min, self.config.active_delta_min) * p - delta_vars[node_id])
            add_leq(rho_vars[node_id] - self.rho_big_m * p)
        
        # Exact defect dynamics and local cost epigraph.
        ctcs_outputs = {}
        for node_id in region_nodes:
            region = self.graph.get_region_by_id(node_id)
            s_minus = s_minus_vars[node_id]
            s_plus = s_plus_vars[node_id]
            w = w_vars[node_id]
            delta = delta_vars[node_id]
            rho = rho_vars[node_id]
            
            if _uses_ctcs_safety(self.config.safety_mode):
                F_result, eta_end = self.ctcs_integrators[region.index](s_minus, w, delta)
                ctcs_outputs[node_id] = (F_result, eta_end)
            else:
                F_result = self.F_endpoint(s_minus, w, delta)
            add_eq(s_plus - F_result)
            
            local_cost = self.local_cost_fn(
                s_minus, w, delta,
                self.config.a, self.config.w_L, self.config.w_E, self.config.w_u_smooth
            )
            add_leq(local_cost - rho)
        
        # On/off region geometry via Big-M. Interior mesh points keep a strict
        # safety margin, while entry/exit/interface points are allowed on the
        # boundary so touching regions remain feasible.
        for node_id in region_nodes:
            region = self.graph.get_region_by_id(node_id)
            p = p_vars[node_id]
            s_minus = s_minus_vars[node_id]
            s_plus = s_plus_vars[node_id]
            w = w_vars[node_id]
            delta = delta_vars[node_id]
            
            A_dm = ca.DM(region.A)
            b_dm = ca.DM(region.b)
            boundary_tol = self.config.boundary_tolerance

            for endpoint in (s_minus[:2], s_plus[:2]):
                closure_violation = ca.mtimes(A_dm, endpoint) - b_dm - boundary_tol
                add_leq(closure_violation - self.position_big_m * (1 - p))

            if _uses_mesh_safety(self.config.safety_mode):
                mesh_positions = self.mesh_sampler(s_minus, w, delta)
                
                for k in _interior_mesh_indices(n_mesh):
                    pos = mesh_positions[k, :]
                    violation = (
                        ca.mtimes(A_dm, pos.T) - b_dm +
                        _region_safety_margin(self.config, region.index)
                    )
                    add_leq(violation - self.position_big_m * (1 - p))

            if _uses_ctcs_safety(self.config.safety_mode):
                _, eta_end = ctcs_outputs[node_id]
                add_leq(
                    eta_end - self.config.ctcs_tolerance -
                    self.config.ctcs_eta_big_m * (1 - p)
                )
        
        # Interface membership and on/off coupling for region-region edges.
        for edge in self.graph.region_edges:
            u, v = edge
            y = y_vars[edge]
            z = z_vars[edge]
            
            add_leq(z - self.state_big_m * y)
            add_leq(-z - self.state_big_m * y)

            region_u = self.graph.get_region_by_id(u)
            region_v = self.graph.get_region_by_id(v)
            z_pos = z[:2]
            boundary_tol = self.config.boundary_tolerance

            for region in (region_u, region_v):
                A_dm = ca.DM(region.A)
                b_dm = ca.DM(region.b)
                closure_violation = ca.mtimes(A_dm, z_pos) - b_dm - boundary_tol
                add_leq(closure_violation - self.position_big_m * (1 - y))
            
            diff_upstream = s_plus_vars[u] - z
            diff_downstream = s_minus_vars[v] - z
            add_leq(diff_upstream - self.config.M_interface * (1 - y))
            add_leq(-diff_upstream - self.config.M_interface * (1 - y))
            add_leq(diff_downstream - self.config.M_interface * (1 - y))
            add_leq(-diff_downstream - self.config.M_interface * (1 - y))

            if self.config.enforce_control_continuity:
                u_exit = self.control_param.evaluate_casadi(1.0, w_vars[u])
                u_entry = self.control_param.evaluate_casadi(0.0, w_vars[v])
                control_diff = u_exit - u_entry
                add_leq(control_diff - self.control_value_big_m * (1 - y))
                add_leq(-control_diff - self.control_value_big_m * (1 - y))
        
        # Source and target boundary coupling.
        for edge in self.graph.source_edges:
            _, v = edge
            y = y_vars[edge]
            diff = s_minus_vars[v][:2] - start_state[:2]
            add_leq(diff - self.config.M_interface * (1 - y))
            add_leq(-diff - self.config.M_interface * (1 - y))
        
        for edge in self.graph.target_edges:
            u, _ = edge
            y = y_vars[edge]
            diff = s_plus_vars[u][:2] - goal_state[:2]
            add_leq(diff - self.config.M_interface * (1 - y))
            add_leq(-diff - self.config.M_interface * (1 - y))
        
        cost = ca.sum1(ca.vertcat(*[rho_vars[node_id] for node_id in region_nodes]))
        g_expr = stack_constraints(g)
        
        nlp = {'x': x, 'f': cost, 'g': g_expr}
        solver = ca.nlpsol(
            'integrated_miocp_relaxed',
            'ipopt',
            nlp,
            {
                'ipopt.max_iter': self.config.max_iter,
                'ipopt.tol': self.config.tol,
                'ipopt.print_level': self.config.print_level,
                'print_time': 0,
            }
        )
        
        if verbose:
            print(f"Integrated relaxation variables: {x.numel()} ({sum(discrete)} relaxed binary-like)")
            print(f"Integrated relaxation constraints: {g_expr.numel()}")
            print(f"Warm-start path: {' -> '.join(warm_path)}")
            print("Solver backend: IPOPT continuous relaxation")
        
        sol = solver(
            x0=x0,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg
        )
        
        x_opt = np.array(sol['x']).flatten()
        g_val = np.array(sol['g']).flatten() if g_expr.numel() else np.array([])
        stats = solver.stats()
        success = bool(stats.get('success', False))
        
        edge_values = {
            edge: float(x_opt[var_slices['y'][edge]].item())
            for edge in edges
        }
        region_values = {
            node_id: float(x_opt[var_slices['p'][node_id]].item())
            for node_id in region_nodes
        }
        max_integrality_gap = self._compute_integrality_gap(
            list(edge_values.values()) + list(region_values.values())
        )
        path = self._extract_active_path(
            edge_values,
            prefer_weighted=max_integrality_gap > 1e-3
        )
        
        if not path:
            return OptimizationResult(
                success=False,
                path=[],
                path_regions=[],
                total_cost=float(sol['f']),
                solve_time=0.0,
                n_paths_evaluated=1,
                solver_status=f"IntegratedMIOCP path extraction failed: {stats.get('return_status', 'unknown')}",
                constraint_violation=self._compute_constraint_violation(g_val, lbg, ubg)
            )
        
        path_regions = [
            region_index_from_node_id(node_id)
            for node_id in path
            if not is_terminal_node_id(node_id)
        ]
        result = OptimizationResult(
            success=success,
            path=path,
            path_regions=path_regions,
            total_cost=float(sol['f']),
            solve_time=0.0,
            n_paths_evaluated=1,
            solver_status=f"IntegratedMIOCP relaxation: {stats.get('return_status', 'unknown')}",
            constraint_violation=self._compute_constraint_violation(g_val, lbg, ubg)
        )
        result.max_integrality_gap = max_integrality_gap
        
        integrator = RK4Integrator(
            self.dynamics, self.control_param,
            self.config.n_integration_steps
        )
        mesh_tau = np.linspace(0.0, 1.0, self.config.n_mesh_points)
        
        for node_id in path[1:-1]:
            region_idx = region_index_from_node_id(node_id)
            s_minus = x_opt[var_slices['s_minus'][node_id]]
            s_plus = x_opt[var_slices['s_plus'][node_id]]
            w = x_opt[var_slices['w'][node_id]]
            delta = float(x_opt[var_slices['delta'][node_id]].item())
            
            result.entry_states[region_idx] = s_minus
            result.exit_states[region_idx] = s_plus
            result.control_params[region_idx] = w
            result.time_durations[region_idx] = delta
            
            traj, tau = integrator.integrate_with_trajectory(s_minus, w, delta)
            result.trajectories.append((traj, tau, delta))
            mesh_positions = np.array(
                self.mesh_sampler(s_minus, w, delta), dtype=float
            )
            result.mesh_samples.append((mesh_positions, mesh_tau.copy(), delta))
        
        defect_norms = []
        for i in range(len(path) - 2):
            region_id = path[i + 1]
            region_idx = region_index_from_node_id(region_id)
            
            s_minus = result.entry_states[region_idx]
            s_plus = result.exit_states[region_idx]
            w = result.control_params[region_idx]
            delta = result.time_durations[region_idx]
            
            F_result = np.array(
                self.F_endpoint(s_minus, w, delta),
                dtype=float,
            ).reshape(-1)
            defect = np.linalg.norm(s_plus - F_result)
            defect_norms.append(defect)
            
            next_edge = (path[i + 1], path[i + 2])
            if next_edge in var_slices['z']:
                result.interface_points.append(
                    self.dynamics.project_to_position(x_opt[var_slices['z'][next_edge]])
                )
            elif i < len(path) - 3:
                result.interface_points.append(
                    self.dynamics.project_to_position(s_plus)
                )
        
        result.defect_norm = max(defect_norms) if defect_norms else 0.0
        result.max_connection_gap = _compute_connection_gap(
            path_regions,
            result.entry_states,
            result.exit_states,
            start_state,
            goal_state,
            self.dynamics.position_indices,
        )
        result.max_control_jump = _compute_control_jump(
            path_regions,
            result.control_params,
            self.control_param
        )
        
        if (
            (not success) or
            result.max_integrality_gap > 1e-3 or
            result.max_connection_gap > 1e-4
        ):
            return self._polish_relaxed_path(
                path, start_state, goal_state, result, edge_values=edge_values
            )

        self._populate_safety_diagnostics(result)
        _apply_final_success_criteria(
            result,
            self.config,
            require_integral_final=_uses_ctcs_safety(self.config.safety_mode),
            require_integrality=True,
            solver_ok=success,
        )
        
        return result


class CentroidRefineDMSSolver:
    """
    Centroid-Refine-DMS solver.

    Stage A: Composite-cost k-shortest-paths graph search.
    Stage B: Interface QP geometric refinement.
    Stage C: Centroid warm-start construction.
    Stage D-E: Barrier-DMS NLP solve with continuation.
    """

    def __init__(self, graph: RegionGraph, dynamics: DynamicsModel, config: "CentroidRefineDMSConfig"):
        self.graph = graph
        self.dynamics = dynamics
        self.config = config

    def _extract_region_indices(self, path: List[str]) -> List[int]:
        from graph_types import region_index_from_node_id, is_terminal_node_id
        return [
            region_index_from_node_id(n)
            for n in path
            if not is_terminal_node_id(n)
        ]

    def solve(self, start_state: np.ndarray, goal_state: np.ndarray,
              verbose: bool = False) -> OptimizationResult:
        import time as _time
        from graph_builder import (
            add_composite_costs_to_graph, k_shortest_paths_generator,
            dijkstra_geometric_length, SOURCE, TARGET,
        )
        from geometric_refiner import InterfaceQPConfig, NarrowInterfaceError, solve_interface_refinement
        from warmstart import WarmStartConfig, build_centroid_warmstart
        from barrier_dms import BarrierDMSConfig, BarrierDMSSolver
        from constraint_layers import compute_lipschitz_safety_gap

        cfg = self.config
        start_clock = _time.time()
        x_start = start_state[:2]
        x_goal = goal_state[:2]

        try:
            centers = add_composite_costs_to_graph(
                self.graph, start_pos=x_start, goal_pos=x_goal,
                gamma_w=cfg.gamma_w, gamma_h=cfg.gamma_h,
            )
        except Exception as e:
            return OptimizationResult(
                success=False, path=[], path_regions=[], total_cost=float('inf'),
                solve_time=_time.time() - start_clock, n_paths_evaluated=0,
                solver_status=f"CentroidRefineDMS Stage A failed: {e}",
                formulation_mode="centroid_refine_dms",
                safety_mode="log_barrier",
            )

        best_result: Optional[OptimizationResult] = None
        n_evaluated = 0
        failure_log = []
        _KKT_DISCLAIMER = (
            "LB is a geometric lower bound on the time component only. "
            "gap_k is NOT a certificate for the full DMS objective. "
            "No global optimality claim is made. The reported solution is a "
            "KKT point of the fixed-path barrier-augmented NLP under LICQ + SOSC."
        )

        import networkx as _nx
        if not _nx.has_path(self.graph.graph, SOURCE, TARGET):
            return OptimizationResult(
                success=False, path=[], path_regions=[], total_cost=float('inf'),
                solve_time=_time.time() - start_clock, n_paths_evaluated=0,
                solver_status="CentroidRefineDMS: graph has no path from source to target",
                formulation_mode="centroid_refine_dms",
                safety_mode="log_barrier",
            )

        # Spec Sec 7.1: LB_geom = D*_path / v_max, pure-geometry Dijkstra on
        # the centroid graph -- computed once, independent of path candidate.
        d_star = dijkstra_geometric_length(self.graph, SOURCE, TARGET)
        lb_geometric = d_star / self.dynamics.v_max

        gen = k_shortest_paths_generator(self.graph, SOURCE, TARGET)

        for path in gen:
            if _time.time() - start_clock > cfg.time_limit_s:
                break

            path_regions = self._extract_region_indices(path)
            if not path_regions:
                continue

            n_evaluated += 1

            qp_cfg = InterfaceQPConfig(
                delta_safe=cfg.delta_safe, delta_extra=cfg.delta_extra, lambda_s=cfg.alpha_s,
            )
            ws_cfg = WarmStartConfig(v_nom_fraction=cfg.v_nom_fraction, n_int=cfg.n_int)
            try:
                if cfg.use_interface_qp:
                    z = solve_interface_refinement(self.graph, path_regions, x_start, x_goal, qp_cfg)
                else:
                    m = len(path_regions)
                    z = np.zeros((m + 1, 2))
                    z[0] = x_start
                    z[m] = x_goal
                    for ii in range(1, m):
                        tau = ii / m
                        z[ii] = x_start * (1 - tau) + x_goal * tau
            except NarrowInterfaceError as e:
                failure_log.append({'path': path, 'stage': 'B', 'reason': str(e)})
                continue
            except Exception as e:
                failure_log.append({'path': path, 'stage': 'B', 'reason': str(e)})
                continue

            try:
                warm_start = build_centroid_warmstart(
                    graph=self.graph, path_regions=path_regions, anchor_points=z,
                    dynamics=self.dynamics, config=ws_cfg,
                )
                # Single-pass warm-start clearance estimate (spec Sec 5):
                # this only sizes the Interface QP anchor-point placement so
                # the warm start passes the R3 strict-interior check and
                # gives IPOPT a reasonable first iterate. It is NOT
                # load-bearing for safety correctness anymore -- the live
                # per-segment margin inside BarrierDMSSolver (Task 3) is
                # what actually guarantees the certified safety margin,
                # regardless of how good this estimate is.
                delta_arr = np.array(
                    [warm_start[ri]['delta'] for ri in path_regions], dtype=float
                )
                delta_arr = np.minimum(delta_arr, cfg.delta_max)
                lip_gap = compute_lipschitz_safety_gap(
                    self.dynamics, path_regions, self.graph, delta_arr, cfg.n_int
                )
                h_max = float(np.max(delta_arr)) / cfg.n_int
                eps_int = self.dynamics.f_lipschitz_bound() * (h_max ** 4) / 30.0
                required_delta_safe = max(
                    cfg.delta_safe, lip_gap + eps_int + cfg.epsilon_final
                )
                if cfg.use_interface_qp and required_delta_safe > cfg.delta_safe + 1e-9:
                    barrier_qp_cfg = InterfaceQPConfig(
                        delta_safe=required_delta_safe,
                        delta_extra=cfg.delta_extra,
                        lambda_s=cfg.alpha_s,
                    )
                    z = solve_interface_refinement(
                        self.graph, path_regions, x_start, x_goal, barrier_qp_cfg
                    )
                    warm_start = build_centroid_warmstart(
                        graph=self.graph, path_regions=path_regions, anchor_points=z,
                        dynamics=self.dynamics, config=ws_cfg,
                    )
            except NarrowInterfaceError as e:
                failure_log.append({'path': path, 'stage': 'B2', 'reason': str(e)})
                continue
            except Exception as e:
                failure_log.append({'path': path, 'stage': 'C', 'reason': str(e)})
                continue

            # IMPORTANT: delta_safe here is the RAW config value, not the
            # warm-start-inflated required_delta_safe estimate above. The
            # live per-segment margin inside BarrierDMSSolver already adds
            # the Lipschitz-gap/RK4-truncation terms on top of this base
            # using the NLP's own Delta_i -- reusing the inflated estimate
            # here would double-count those terms.
            barrier_cfg = BarrierDMSConfig(
                n_int=cfg.n_int, delta_safe=cfg.delta_safe, delta_extra=cfg.delta_extra,
                epsilon_certificate_buffer=cfg.epsilon_certificate_buffer,
                barrier_levels=cfg.barrier_levels if cfg.use_barrier_continuation else [cfg.barrier_levels[-1]],
                epsilon_final=cfg.epsilon_final,
                time_limit_s=max(1.0, cfg.time_limit_s - (_time.time() - start_clock)),
                delta_min=cfg.delta_min, delta_max=cfg.delta_max,
                n_control_segments=cfg.n_control_segments,
                w_T=cfg.w_T, w_L=cfg.w_L, w_U=cfg.w_U, w_S=cfg.w_S,
            )

            try:
                barrier_solver = BarrierDMSSolver(self.graph, self.dynamics, barrier_cfg)
                barrier_result = barrier_solver.solve(
                    path_regions, z, warm_start,
                    start_state=start_state, goal_state=goal_state,
                )
            except Exception as e:
                failure_log.append({'path': path, 'stage': 'DE', 'reason': str(e)})
                continue

            opt_result = OptimizationResult(
                success=barrier_result.success, path=path, path_regions=path_regions,
                total_cost=barrier_result.total_cost, solve_time=_time.time() - start_clock,
                n_paths_evaluated=n_evaluated, solver_status=barrier_result.solver_status,
                formulation_mode="centroid_refine_dms",
                safety_mode="log_barrier",
                global_optimality_claim=_KKT_DISCLAIMER,
                min_safety_margin=barrier_result.min_safety_margin,
                certified_safety_margin=barrier_result.certified_safety_margin,
                lipschitz_gap=barrier_result.lipschitz_gap,
                safety_certification=barrier_result.safety_certification,
                n_barrier_levels=len(barrier_result.barrier_level_results),
                failure_log=failure_log,
                entry_states=barrier_result.entry_states,
                exit_states=barrier_result.exit_states,
                control_params=barrier_result.control_params,
                time_durations=barrier_result.time_durations,
                pipeline_mode="centroid_refine_dms",
                defect_norm=barrier_result.defect_norm,
                max_connection_gap=_compute_connection_gap(
                    path_regions, barrier_result.entry_states, barrier_result.exit_states,
                    start_state, goal_state,
                ),
                n_nlp_iterations=barrier_result.n_nlp_iterations,
            )

            if barrier_result.success:
                best_result = opt_result
                if cfg.mode == "first_feasible":
                    break
            elif best_result is None:
                best_result = opt_result

        if best_result is None:
            return OptimizationResult(
                success=False, path=[], path_regions=[], total_cost=float('inf'),
                solve_time=_time.time() - start_clock, n_paths_evaluated=n_evaluated,
                solver_status="CentroidRefineDMS: no feasible path found",
                formulation_mode="centroid_refine_dms", failure_log=failure_log,
                safety_mode="log_barrier",
                global_optimality_claim=_KKT_DISCLAIMER,
                lb_geometric=lb_geometric,
            )

        best_result.solve_time = _time.time() - start_clock

        if best_result.success and best_result.entry_states:
            control_param = ControlParameterization(
                n_segments=cfg.n_control_segments,
                n_u=self.dynamics.n_u,
                parameterization="piecewise_constant",
            )
            integrator = RK4Integrator(self.dynamics, control_param, cfg.n_int)
            for ri in best_result.path_regions:
                s0 = best_result.entry_states[ri]
                w = best_result.control_params[ri]
                delta = best_result.time_durations[ri]
                traj, tau = integrator.integrate_with_trajectory(s0, w, delta)
                best_result.trajectories.append((traj, tau, delta))
            for ri in best_result.path_regions[:-1]:
                s_exit = best_result.exit_states.get(ri)
                if s_exit is not None:
                    best_result.interface_points.append(
                        self.dynamics.project_to_position(s_exit)
                        if hasattr(self.dynamics, "project_to_position") else s_exit[:2]
                    )

        # Spec Sec 7.1/7.2 + Part 9 post-processing.
        best_result.lb_geometric = lb_geometric
        denom = max(1.0, abs(best_result.total_cost))
        best_result.optimality_gap = (best_result.total_cost - cfg.w_T * lb_geometric) / denom

        return best_result


def create_integrated_optimizer_from_config(graph: RegionGraph,
                                            dynamics: DynamicsModel,
                                            config_dict: Dict) -> IntegratedMIOCPSolver:
    """
    Create the one-phase integrated MIOCP optimizer from configuration.
    """
    optimizer_cfg = config_dict.get('optimizer', {})
    solver_type = (optimizer_cfg.get('solver_mode')
                   or config_dict.get('solver_type', 'integrated'))
    if solver_type == 'centroid_refine_dms':
        cr_cfg_dict = config_dict.get('centroid_refine_dms', {})
        shooting_cfg = config_dict.get('shooting', {})
        control_cfg = config_dict.get('control', {})
        dynamics_cfg = config_dict.get('dynamics', {})
        cost_cfg = config_dict.get('cost', {})
        cr_config = CentroidRefineDMSConfig(
            gamma_w=cr_cfg_dict.get('gamma_w', 1.0),
            gamma_h=cr_cfg_dict.get('gamma_h', 0.64),
            delta_safe=cr_cfg_dict.get('delta_safe', 0.02),
            delta_extra=cr_cfg_dict.get('delta_extra', 0.01),
            epsilon_certificate_buffer=cr_cfg_dict.get('epsilon_certificate_buffer', 0.0),
            alpha_s=cr_cfg_dict.get('alpha_s', 0.0),
            v_nom_fraction=cr_cfg_dict.get('v_nom_fraction', 0.5),
            barrier_levels=cr_cfg_dict.get('barrier_levels', [1.0, 0.5, 0.1, 0.01]),
            epsilon_final=cr_cfg_dict.get('epsilon_final', 1e-6),
            epsilon_gap=cr_cfg_dict.get('epsilon_gap', 0.05),
            time_limit_s=cr_cfg_dict.get('time_limit_s', 60.0),
            mode=cr_cfg_dict.get('mode', 'first_feasible'),
            n_int=shooting_cfg.get('n_integration_steps', 10),
            n_control_segments=control_cfg.get('n_segments', 2),
            delta_min=dynamics_cfg.get('delta_min', 0.1),
            delta_max=dynamics_cfg.get('delta_max', 10.0),
            w_T=cost_cfg.get('a', 1.0),
            w_L=cost_cfg.get('w_L', 1.0),
            w_U=cost_cfg.get('w_E', 1.0),
            w_S=cost_cfg.get('w_u_smooth', 0.2),
            use_centroid_cost=cr_cfg_dict.get('use_centroid_cost', True),
            use_interface_qp=cr_cfg_dict.get('use_interface_qp', True),
            use_log_barrier=cr_cfg_dict.get('use_log_barrier', True),
            use_barrier_continuation=cr_cfg_dict.get('use_barrier_continuation', True),
            use_inexact_tolerance=cr_cfg_dict.get('use_inexact_tolerance', True),
        )
        return CentroidRefineDMSSolver(graph, dynamics, cr_config)

    cost_config = config_dict.get('cost', {})
    shooting_config = config_dict.get('shooting', {})
    optimizer_config = config_dict.get('optimizer', {})
    control_config = config_dict.get('control', {})
    dynamics_config = config_dict.get('dynamics', {})
    pipeline_mode = optimizer_config.get('pipeline_mode', 'integrated')
    default_polish_candidates = 5 if pipeline_mode == 'two_stage' else 20
    
    opt_config = OptimizationConfig(
        a=cost_config.get('a', 1.0),
        w_L=cost_config.get('w_L', 1.0),
        w_E=cost_config.get('w_E', 1.0),
        w_u_smooth=cost_config.get('w_u_smooth', 0.2),
        w_theta_smooth=cost_config.get('w_theta_smooth', 0.0),
        n_integration_steps=shooting_config.get('n_integration_steps', 20),
        n_mesh_points=shooting_config.get('n_mesh_points', 3),
        n_control_segments=control_config.get('n_segments', 2),
        safety_margin=shooting_config.get('safety_margin', 0.02),
        safety_mode=shooting_config.get('safety_mode', 'both'),
        ctcs_tolerance=shooting_config.get('ctcs_tolerance', 1.0e-6),
        ctcs_penalty=shooting_config.get('ctcs_penalty', 'squared_hinge'),
        ctcs_integral_mode=shooting_config.get('ctcs_integral_mode', 'normalized'),
        ctcs_use_rk4_stages=shooting_config.get('ctcs_use_rk4_stages', True),
        ctcs_eta_big_m=shooting_config.get('ctcs_eta_big_m', 100.0),
        dense_check_points=shooting_config.get('dense_check_points', 1000),
        dense_check_tolerance=shooting_config.get('dense_check_tolerance', 1.0e-4),
        fail_on_dense_violation=shooting_config.get('fail_on_dense_violation', True),
        boundary_tolerance=shooting_config.get('boundary_tolerance', 1e-8),
        delta_min=dynamics_config.get('delta_min', 0.1),
        delta_max=dynamics_config.get('delta_max', 10.0),
        active_delta_min=shooting_config.get(
            'active_delta_min',
            max(dynamics_config.get('delta_min', 0.1), 0.05)
        ),
        M_position=optimizer_config.get('big_M', {}).get('position', 20.0),
        M_interface=optimizer_config.get('big_M', {}).get('interface', 20.0),
        M_time=optimizer_config.get('big_M', {}).get('time', 100.0),
        enforce_control_continuity=optimizer_config.get('enforce_control_continuity', True),
        max_iter=optimizer_config.get('ipopt', {}).get('max_iter', 3500),
        tol=optimizer_config.get('ipopt', {}).get('tol', 1e-6),
        print_level=optimizer_config.get('ipopt', {}).get('print_level', 0),
        max_polish_path_candidates=optimizer_config.get(
            'max_polish_path_candidates',
            optimizer_config.get('path_polish_candidates', default_polish_candidates)
        ),
        final_connection_tolerance=optimizer_config.get('final_connection_tolerance', 1e-6),
        final_defect_tolerance=optimizer_config.get('final_defect_tolerance', 1e-6),
        final_integrality_tolerance=optimizer_config.get('final_integrality_tolerance', 1e-6),
        pipeline_mode=pipeline_mode,
        path_screen_top_k=optimizer_config.get('path_screen_top_k', 5),
        path_screen_max_paths=optimizer_config.get('path_screen_max_paths', 1500),
        early_stop_on_feasible=optimizer_config.get('early_stop_on_feasible', True),
        acceptable_dense_tolerance_for_repair=optimizer_config.get(
            'acceptable_dense_tolerance_for_repair', 5e-3
        ),
        screening_ipopt_tol=optimizer_config.get('screening_ipopt_tol', 1e-4),
        screening_max_iter=optimizer_config.get('screening_max_iter', 500),
        final_ipopt_tol=optimizer_config.get('final_ipopt_tol', 1e-6),
        final_max_iter=optimizer_config.get('final_max_iter', 3000),
        repair_enabled=optimizer_config.get('repair_enabled', True),
        repair_max_rounds=optimizer_config.get('repair_max_rounds', 2),
        local_shrink_margin=optimizer_config.get('local_shrink_margin', 0.003),
        adaptive_refine_enabled=optimizer_config.get('adaptive_refine_enabled', True),
        screening_safety_mode=optimizer_config.get('screening_safety_mode', None),
    )

    return IntegratedMIOCPSolver(graph, dynamics, opt_config)
