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

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import casadi as ca
import time

from graph_builder import RegionGraph, SOURCE, TARGET
from graph_types import (
    is_terminal_node_id,
    region_index_from_node_id,
    region_node_label,
)
from dynamics import (
    CasADiIntegrationBundle,
    DynamicsModel,
    ControlParameterization,
    create_integration_bundle,
)
from constraint_layers import (
    build_activation_constraints,
    build_fixed_path_boundary_constraints,
    build_fixed_path_cbf_safety_constraints,
    build_fixed_path_coupling_constraints,
    build_fixed_path_geometry_constraints,
    build_integrated_coupling_constraints,
    build_integrated_dynamics_constraints,
    build_integrated_geometry_constraints,
    build_network_flow_constraints,
    merge_constraint_layers,
    stack_constraints,
)


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
    boundary_tolerance: float = 1e-8
    
    # Time bounds
    delta_min: float = 0.1
    delta_max: float = 10.0
    
    # Big-M values
    M_position: float = 20.0
    M_interface: float = 20.0
    M_time: float = 100.0
    enforce_control_continuity: bool = True
    
    # IPOPT options
    max_iter: int = 3000
    tol: float = 1e-6
    print_level: int = 0
    max_polish_path_candidates: int = 3
    
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
    max_connection_gap: float = 0.0
    max_control_jump: float = 0.0
    max_integrality_gap: float = 0.0


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


def _state_guess_from_position(dynamics: DynamicsModel,
                               position: np.ndarray,
                               heading: float,
                               warm_state: Optional[np.ndarray] = None) -> np.ndarray:
    """Build a state initial guess from a 2D graph anchor."""
    if warm_state is not None:
        state = np.asarray(warm_state, dtype=np.float64).copy()
        if state.size != dynamics.n_x:
            state = np.zeros(dynamics.n_x, dtype=np.float64)
    else:
        state = np.zeros(dynamics.n_x, dtype=np.float64)

    for local_idx, state_idx in enumerate(dynamics.position_indices):
        state[state_idx] = position[local_idx]
    for state_idx in dynamics.angle_indices:
        state[state_idx] = heading

    return state


def _tile_control_vector(control: np.ndarray,
                         control_param: ControlParameterization) -> np.ndarray:
    """Repeat one control vector across all parameterization segments."""
    return np.tile(np.asarray(control, dtype=np.float64), control_param.n_segments)


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
        # Extract region sequence (excluding source/target)
        path_regions = []
        for node_id in path:
            if not is_terminal_node_id(node_id):
                path_regions.append(region_index_from_node_id(node_id))
        
        n_regions = len(path_regions)
        
        if n_regions == 0:
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
            delta_init = max(
                self.config.delta_min,
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
            
            lbx.append(self.config.delta_min)
            ubx.append(self.config.delta_max)
            if warm is not None and 'delta' in warm:
                x0.append(max(self.config.delta_min, min(self.config.delta_max, float(warm['delta']))))
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

        g = stack_constraints(g)
        
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
        
        # Parse solution
        result = self._parse_solution(
            x_opt, path_regions, n_x, n_w,
            start_state, goal_state
        )
        
        result.success = success
        result.total_cost = float(sol['f'])
        result.solver_status = return_status
        result.constraint_violation = _compute_bound_violation(
            np.array(sol['g']).flatten(),
            lbg,
            ubg
        )
        
        return result
    
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
        from dynamics import RK4Integrator
        
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
                              max_candidates: int) -> List[List[str]]:
        """Generate a small set of centroid-shortest source-target paths."""
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

        candidates = []
        try:
            for path in nx.shortest_simple_paths(
                self.graph.graph,
                SOURCE,
                TARGET,
                weight=edge_weight
            ):
                candidates.append(path)
                if len(candidates) >= max_candidates:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        return candidates

    def _fallback_fixed_path_search(self,
                                    start_state: np.ndarray,
                                    goal_state: np.ndarray,
                                    relaxation_result: OptimizationResult,
                                    excluded_paths: List[List[str]]) -> Optional[OptimizationResult]:
        """Try a few nearby discrete paths when the relaxed path cannot be polished."""
        excluded = {tuple(path) for path in excluded_paths}

        for candidate_path in self._iter_candidate_paths(
            start_state,
            goal_state,
            self.config.max_polish_path_candidates
        ):
            if tuple(candidate_path) in excluded:
                continue

            candidate_result = self.path_solver.solve_path(
                candidate_path,
                start_state,
                goal_state
            )
            candidate_result.max_integrality_gap = relaxation_result.max_integrality_gap
            candidate_result.n_paths_evaluated = relaxation_result.n_paths_evaluated + 1

            if candidate_result.success:
                candidate_result.solver_status = (
                    f"{relaxation_result.solver_status} + fallback fixed-path NLP: "
                    f"{candidate_result.solver_status}"
                )
                return candidate_result

        return None
    
    def _compute_constraint_violation(self, g_val: np.ndarray,
                                      lbg: List[float],
                                      ubg: List[float]) -> float:
        """Maximum bound violation across all constraints."""
        return _compute_bound_violation(g_val, lbg, ubg)
    
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
                             relaxation_result: OptimizationResult) -> OptimizationResult:
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
        polished.max_integrality_gap = relaxation_result.max_integrality_gap
        
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
            excluded_paths=[path]
        )
        if fallback_result is not None:
            return fallback_result

        relaxation_result.success = False
        return relaxation_result
    
    def solve(self, start_state: np.ndarray, goal_state: np.ndarray,
              verbose: bool = False) -> OptimizationResult:
        """
        Solve the motion planning problem as a single integrated continuous
        relaxation of the MIOCP.
        """
        start_time = time.time()

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
            delta_init = max(
                self.config.delta_min,
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
        
        g, lbg, ubg = merge_constraint_layers([
            build_network_flow_constraints(
                y_vars,
                p_vars,
                self.graph.source_edges,
                self.graph.target_edges,
                region_nodes,
                edges,
            ),
            build_activation_constraints(
                region_nodes,
                p_vars,
                s_minus_vars,
                s_plus_vars,
                w_vars,
                delta_vars,
                rho_vars,
                self.state_big_m,
                self.control_big_m,
                self.config.delta_min,
                self.config.delta_max,
                self.rho_big_m,
            ),
            build_integrated_dynamics_constraints(
                region_nodes,
                s_minus_vars,
                s_plus_vars,
                w_vars,
                delta_vars,
                rho_vars,
                self.F_endpoint,
                self.local_cost_fn,
                (
                    self.config.a,
                    self.config.w_L,
                    self.config.w_E,
                    self.config.w_u_smooth,
                ),
            ),
            build_integrated_geometry_constraints(
                self.graph,
                self.dynamics,
                region_nodes,
                p_vars,
                s_minus_vars,
                s_plus_vars,
                w_vars,
                delta_vars,
                self.mesh_sampler,
                n_mesh,
                self.config.safety_margin,
                self.config.boundary_tolerance,
                self.position_big_m,
            ),
            build_integrated_coupling_constraints(
                self.graph,
                self.dynamics,
                y_vars,
                s_minus_vars,
                s_plus_vars,
                w_vars,
                z_vars,
                self.control_param,
                self.state_big_m,
                self.position_big_m,
                self.config.M_interface,
                self.control_value_big_m,
                self.config.boundary_tolerance,
                start_state,
                goal_state,
                self.config.enforce_control_continuity,
            ),
        ])
        
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
        
        if result.success and (
            result.max_integrality_gap > 1e-3 or result.max_connection_gap > 1e-4
        ):
            return self._polish_relaxed_path(path, start_state, goal_state, result)
        
        return result

def create_integrated_optimizer_from_config(graph: RegionGraph,
                                            dynamics: DynamicsModel,
                                            config_dict: Dict) -> IntegratedMIOCPSolver:
    """
    Create the one-phase integrated MIOCP optimizer from configuration.
    """
    cost_config = config_dict.get('cost', {})
    shooting_config = config_dict.get('shooting', {})
    optimizer_config = config_dict.get('optimizer', {})
    control_config = config_dict.get('control', {})
    
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
        cbf_alpha=shooting_config.get('cbf_alpha', 0.0),
        boundary_tolerance=shooting_config.get('boundary_tolerance', 1e-8),
        delta_min=config_dict.get('dynamics', {}).get('delta_min', 0.1),
        delta_max=config_dict.get('dynamics', {}).get('delta_max', 10.0),
        M_position=optimizer_config.get('big_M', {}).get('position', 20.0),
        M_interface=optimizer_config.get('big_M', {}).get('interface', 20.0),
        M_time=optimizer_config.get('big_M', {}).get('time', 100.0),
        enforce_control_continuity=optimizer_config.get('enforce_control_continuity', True),
        max_iter=optimizer_config.get('ipopt', {}).get('max_iter', 3500),
        tol=optimizer_config.get('ipopt', {}).get('tol', 1e-6),
        print_level=optimizer_config.get('ipopt', {}).get('print_level', 0),
        max_polish_path_candidates=optimizer_config.get('path_polish_candidates', 3),
    )
    
    return IntegratedMIOCPSolver(graph, dynamics, opt_config)
