"""
shooting.py - Multiple shooting blocks for trajectory optimization.

Implements:
- Local IVP on each region with normalized time
- Endpoint map F_v(s_v^-, w_v, Delta_v) = s_v^+
- Defect constraints: s_v^+ - F_v(...) = 0
- Safety constraints via mesh point sampling
- Local cost computation with quadrature
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
import casadi as ca

from dynamics import (DynamicsModel, UnicycleModel, ControlParameterization,
                      RK4Integrator, create_casadi_integrator, 
                      create_casadi_trajectory_sampler,
                      create_casadi_ctcs_integrator)
from convex_regions import ConvexRegion


def _get_config_value(config: Dict, key_path: Tuple[str, ...], default):
    """Read a nested config value while staying backward compatible."""
    current = config
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


@dataclass
class ShootingBlockConfig:
    """Configuration for a multiple shooting block."""
    region_index: int
    n_integration_steps: int = 20
    n_mesh_points: int = 10
    n_control_segments: int = 2
    safety_margin: float = 0.02
    safety_mode: str = "both"
    ctcs_tolerance: float = 1.0e-6
    ctcs_penalty: str = "squared_hinge"
    ctcs_integral_mode: str = "normalized"
    ctcs_eta_big_m: float = 100.0
    dense_check_points: int = 1000
    dense_check_tolerance: float = 1.0e-4
    fail_on_dense_violation: bool = True
    delta_min: float = 0.1
    delta_max: float = 10.0


@dataclass
class ShootingBlock:
    """
    Multiple shooting block for a single convex region.
    
    Represents the local trajectory segment with:
    - Entry state s_v^-
    - Exit state s_v^+
    - Control parameters w_v
    - Time duration Delta_v
    - Defect constraint: s_v^+ = F_v(s_v^-, w_v, Delta_v)
    - Safety constraints: A_v @ pi(x_v(tau_k)) <= b_v - epsilon_v
    
    Attributes:
        region: ConvexRegion object
        dynamics: Dynamics model
        config: Block configuration
        control_param: Control parameterization
        integrator: RK4 integrator
        F_casadi: CasADi endpoint map function
        mesh_sampler: CasADi trajectory sampling function
    """
    region: ConvexRegion
    dynamics: DynamicsModel
    config: ShootingBlockConfig
    control_param: ControlParameterization = field(init=False)
    integrator: RK4Integrator = field(init=False)
    F_casadi: Callable = field(init=False)
    F_ctcs_casadi: Callable = field(init=False)
    mesh_sampler: Callable = field(init=False)
    
    def __post_init__(self):
        """Initialize integrators and CasADi functions."""
        # Control parameterization
        self.control_param = ControlParameterization(
            n_u=self.dynamics.n_u,
            parameterization="piecewise_constant",
            n_segments=self.config.n_control_segments
        )
        
        # NumPy integrator for evaluation
        self.integrator = RK4Integrator(
            dynamics=self.dynamics,
            control_param=self.control_param,
            n_steps=self.config.n_integration_steps
        )
        
        # CasADi endpoint map
        self.F_casadi = create_casadi_integrator(
            dynamics=self.dynamics,
            control_param=self.control_param,
            n_steps=self.config.n_integration_steps
        )

        self.F_ctcs_casadi = create_casadi_ctcs_integrator(
            dynamics=self.dynamics,
            control_param=self.control_param,
            region=self.region,
            n_steps=self.config.n_integration_steps,
            safety_margin=self.config.safety_margin,
            penalty=self.config.ctcs_penalty,
            integral_mode=self.config.ctcs_integral_mode
        )
        
        # CasADi mesh sampler
        self.mesh_sampler = create_casadi_trajectory_sampler(
            dynamics=self.dynamics,
            control_param=self.control_param,
            n_steps=self.config.n_integration_steps,
            n_mesh=self.config.n_mesh_points
        )
    
    @property
    def n_x(self) -> int:
        """State dimension."""
        return self.dynamics.n_x
    
    @property
    def n_u(self) -> int:
        """Control dimension."""
        return self.dynamics.n_u
    
    @property
    def n_w(self) -> int:
        """Control parameter dimension."""
        return self.control_param.n_w
    
    def endpoint_map(self, s_minus: np.ndarray, w: np.ndarray, 
                     delta: float) -> np.ndarray:
        """
        Compute endpoint map F_v(s_v^-, w_v, Delta_v).
        
        Args:
            s_minus: Entry state
            w: Control parameters
            delta: Time duration
            
        Returns:
            Exit state s_plus
        """
        return self.integrator.integrate(s_minus, w, delta)
    
    def compute_defect(self, s_minus: np.ndarray, s_plus: np.ndarray,
                       w: np.ndarray, delta: float) -> np.ndarray:
        """
        Compute defect: s_plus - F_v(s_minus, w, delta).
        
        Should be zero for valid trajectory.
        """
        F_result = self.endpoint_map(s_minus, w, delta)
        return s_plus - F_result
    
    def check_safety(self, s_minus: np.ndarray, w: np.ndarray,
                     delta: float) -> Tuple[bool, np.ndarray]:
        """
        Check safety constraints at mesh points.
        
        Args:
            s_minus: Entry state
            w: Control parameters
            delta: Time duration
            
        Returns:
            (is_safe, violations) where violations is (n_mesh, n_constraints)
        """
        # Sample trajectory at mesh points
        mesh_states = self.integrator.sample_trajectory_at_mesh(
            s_minus, w, delta, self.config.n_mesh_points
        )
        
        # Check containment at each mesh point
        margin = self.config.safety_margin
        violations = []
        
        for state in mesh_states:
            pos = self.dynamics.project_to_position(state)
            # A @ pos - (b - margin) should be <= 0
            violation = self.region.A @ pos - (self.region.b - margin)
            violations.append(violation)
        
        violations = np.array(violations)
        is_safe = np.all(violations <= 0)
        
        return is_safe, violations

    def continuous_safety_certificate_casadi(self, s_minus, w, delta):
        """Return symbolic eta_end for the CTCS RK4 violation integral."""
        _, eta_end = self.F_ctcs_casadi(s_minus, w, delta)
        return eta_end

    def compute_ctcs_violation_integral(self, s_minus: np.ndarray, w: np.ndarray,
                                        delta: float) -> Dict:
        """NumPy RK4 equivalent of the CTCS accumulated violation diagnostic."""
        x = np.asarray(s_minus, dtype=np.float64).copy()
        w = np.asarray(w, dtype=np.float64)
        delta = float(delta)
        dt = 1.0 / self.config.n_integration_steps
        eta = 0.0
        stage_violations = []

        def lambda_and_max_violation(x_stage: np.ndarray) -> Tuple[float, float]:
            pos = self.dynamics.project_to_position(x_stage)
            residual = self.region.A @ pos - self.region.b
            hinge = np.maximum(0.0, residual)
            return float(np.sum(hinge * hinge)), float(np.max(residual))

        for i in range(self.config.n_integration_steps):
            tau = i * dt
            tau_mid = tau + 0.5 * dt
            tau_end = tau + dt

            u0 = self.control_param.evaluate(tau, w)
            um = self.control_param.evaluate(tau_mid, w)
            u1 = self.control_param.evaluate(tau_end, w)

            lam1, max1 = lambda_and_max_violation(x)
            k1_x = delta * self.dynamics.f(x, u0)
            eta_scale = delta if self.config.ctcs_integral_mode == "physical_time" else 1.0
            k1_eta = eta_scale * lam1

            x2 = x + 0.5 * dt * k1_x
            lam2, max2 = lambda_and_max_violation(x2)
            k2_x = delta * self.dynamics.f(x2, um)
            k2_eta = eta_scale * lam2

            x3 = x + 0.5 * dt * k2_x
            lam3, max3 = lambda_and_max_violation(x3)
            k3_x = delta * self.dynamics.f(x3, um)
            k3_eta = eta_scale * lam3

            x4 = x + dt * k3_x
            lam4, max4 = lambda_and_max_violation(x4)
            k4_x = delta * self.dynamics.f(x4, u1)
            k4_eta = eta_scale * lam4

            x = x + (dt / 6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
            eta += (dt / 6) * (k1_eta + 2*k2_eta + 2*k3_eta + k4_eta)
            stage_violations.extend([max1, max2, max3, max4])

        return {
            'eta_end': float(eta),
            'max_stage_violation': float(max(stage_violations)) if stage_violations else 0.0,
            'stage_violations': np.asarray(stage_violations, dtype=np.float64),
        }

    def check_continuous_safety(self, s_minus: np.ndarray, w: np.ndarray,
                                delta: float) -> Dict:
        """Check eta_end against the configured CTCS tolerance."""
        diagnostic = self.compute_ctcs_violation_integral(s_minus, w, delta)
        diagnostic['is_safe'] = diagnostic['eta_end'] <= self.config.ctcs_tolerance
        return diagnostic

    def dense_check_safety(self, s_minus: np.ndarray, w: np.ndarray,
                           delta: float, n_points: Optional[int] = None) -> Dict:
        """Dense signed region violation check for post-solve verification."""
        n_points = int(n_points or self.config.dense_check_points)
        n_points = max(n_points, 2)
        dense_integrator = RK4Integrator(
            dynamics=self.dynamics,
            control_param=self.control_param,
            n_steps=n_points - 1
        )
        traj, tau_values = dense_integrator.integrate_with_trajectory(s_minus, w, delta)
        max_violation = -np.inf
        violations = []

        for state in traj:
            pos = self.dynamics.project_to_position(state)
            residual = self.region.A @ pos - self.region.b
            violations.append(residual)
            max_violation = max(max_violation, float(np.max(residual)))

        return {
            'max_violation': float(max_violation),
            'is_safe': bool(max_violation <= self.config.dense_check_tolerance),
            'tau': tau_values,
            'violations': np.asarray(violations, dtype=np.float64),
        }
    
    def compute_local_cost(self, s_minus: np.ndarray, w: np.ndarray,
                           delta: float, cost_weights: Dict[str, float]) -> float:
        """
        Compute local cost J_v using trapezoidal quadrature.
        
        J_v = a * Delta_v
            + integral[w_L * ||q_dot||^2 + w_E * ||u||^2] dtau
            + w_u_smooth * sum_k ||u_{k+1} - u_k||^2
        
        Args:
            s_minus: Entry state
            w: Control parameters
            delta: Time duration
            cost_weights: Dict with 'a', 'w_L', 'w_E', 'w_u_smooth'
            
        Returns:
            Local cost value
        """
        a = cost_weights.get('a', 1.0)
        w_L = cost_weights.get('w_L', 1.0)
        w_E = cost_weights.get('w_E', 1.0)
        w_u_smooth = cost_weights.get('w_u_smooth', 0.0)
        
        # Time cost
        time_cost = a * delta
        
        # Get trajectory
        traj, tau_vals = self.integrator.integrate_with_trajectory(s_minus, w, delta)
        
        # Compute running cost via trapezoidal rule on the physical running
        # integrand. Using position chords would systematically underestimate
        # curved motion and make loops artificially cheap.
        running_cost = 0.0
        n_pts = len(traj)

        for i in range(n_pts - 1):
            tau0 = tau_vals[i]
            tau1 = tau_vals[i + 1]
            dt = tau1 - tau0

            u0 = self.control_param.evaluate(tau0, w)
            u1 = self.control_param.evaluate(tau1, w)
            q_dot0 = self.dynamics.position_velocity(traj[i], u0)
            q_dot1 = self.dynamics.position_velocity(traj[i + 1], u1)

            running0 = w_L * np.dot(q_dot0, q_dot0) + w_E * np.dot(u0, u0)
            running1 = w_L * np.dot(q_dot1, q_dot1) + w_E * np.dot(u1, u1)
            running_cost += delta * dt * 0.5 * (running0 + running1)

        smoothness_cost = 0.0
        if self.control_param.n_segments > 1 and w_u_smooth > 0.0:
            for seg in range(self.control_param.n_segments - 1):
                u_left = self.control_param.get_segment_control(w, seg)
                u_right = self.control_param.get_segment_control(w, seg + 1)
                diff = u_right - u_left
                smoothness_cost += w_u_smooth * np.dot(diff, diff)
        
        return time_cost + running_cost + smoothness_cost


class ShootingBlockManager:
    """
    Manages multiple shooting blocks across all regions.
    
    Coordinates:
    - Block creation
    - Variable indexing
    - Constraint generation
    - Cost aggregation
    """
    
    def __init__(self, regions: List[ConvexRegion], dynamics: DynamicsModel,
                 config: Dict):
        """
        Args:
            regions: List of ConvexRegion objects
            dynamics: Dynamics model
            config: Configuration dictionary
        """
        self.regions = regions
        self.dynamics = dynamics
        self.raw_config = config
        self.config = {
            'n_integration_steps': _get_config_value(
                config, ('shooting', 'n_integration_steps'),
                config.get('n_integration_steps', 20)
            ),
            'n_mesh_points': _get_config_value(
                config, ('shooting', 'n_mesh_points'),
                config.get('n_mesh_points', 10)
            ),
            'n_control_segments': _get_config_value(
                config, ('control', 'n_segments'),
                config.get('n_segments', 2)
            ),
            'safety_margin': _get_config_value(
                config, ('shooting', 'safety_margin'),
                config.get('safety_margin', 0.02)
            ),
            'safety_mode': _get_config_value(
                config, ('shooting', 'safety_mode'),
                config.get('safety_mode', 'both')
            ),
            'ctcs_tolerance': _get_config_value(
                config, ('shooting', 'ctcs_tolerance'),
                config.get('ctcs_tolerance', 1.0e-6)
            ),
            'ctcs_penalty': _get_config_value(
                config, ('shooting', 'ctcs_penalty'),
                config.get('ctcs_penalty', 'squared_hinge')
            ),
            'ctcs_integral_mode': _get_config_value(
                config, ('shooting', 'ctcs_integral_mode'),
                config.get('ctcs_integral_mode', 'normalized')
            ),
            'ctcs_eta_big_m': _get_config_value(
                config, ('shooting', 'ctcs_eta_big_m'),
                config.get('ctcs_eta_big_m', 100.0)
            ),
            'dense_check_points': _get_config_value(
                config, ('shooting', 'dense_check_points'),
                config.get('dense_check_points', 1000)
            ),
            'dense_check_tolerance': _get_config_value(
                config, ('shooting', 'dense_check_tolerance'),
                config.get('dense_check_tolerance', 1.0e-4)
            ),
            'fail_on_dense_violation': _get_config_value(
                config, ('shooting', 'fail_on_dense_violation'),
                config.get('fail_on_dense_violation', True)
            ),
            'delta_min': _get_config_value(
                config, ('dynamics', 'delta_min'),
                config.get('delta_min', 0.1)
            ),
            'delta_max': _get_config_value(
                config, ('dynamics', 'delta_max'),
                config.get('delta_max', 10.0)
            ),
            'enforce_control_continuity': _get_config_value(
                config, ('optimizer', 'enforce_control_continuity'),
                config.get('enforce_control_continuity', True)
            ),
        }
        self.enforce_control_continuity = bool(self.config['enforce_control_continuity'])
        
        # Create shooting blocks
        self.blocks: Dict[int, ShootingBlock] = {}
        
        for region in regions:
            block_config = ShootingBlockConfig(
                region_index=region.index,
                n_integration_steps=self.config['n_integration_steps'],
                n_mesh_points=self.config['n_mesh_points'],
                n_control_segments=self.config['n_control_segments'],
                safety_margin=self.config['safety_margin'],
                safety_mode=self.config['safety_mode'],
                ctcs_tolerance=self.config['ctcs_tolerance'],
                ctcs_penalty=self.config['ctcs_penalty'],
                ctcs_integral_mode=self.config['ctcs_integral_mode'],
                ctcs_eta_big_m=self.config['ctcs_eta_big_m'],
                dense_check_points=self.config['dense_check_points'],
                dense_check_tolerance=self.config['dense_check_tolerance'],
                fail_on_dense_violation=self.config['fail_on_dense_violation'],
                delta_min=self.config['delta_min'],
                delta_max=self.config['delta_max']
            )
            
            self.blocks[region.index] = ShootingBlock(
                region=region,
                dynamics=dynamics,
                config=block_config
            )
    
    def get_block(self, region_index: int) -> ShootingBlock:
        """Get shooting block for a region."""
        return self.blocks[region_index]
    
    def get_variable_dimensions(self) -> Dict:
        """
        Get dimensions of decision variables per block.
        
        Returns:
            Dict with dimensions for each variable type
        """
        block = list(self.blocks.values())[0]  # All blocks have same dimensions
        
        return {
            'n_x': block.n_x,           # Entry/exit state dimension
            'n_w': block.n_w,           # Control parameter dimension
            'n_delta': 1,               # Time duration
            'n_rho': 1,                 # Cost epigraph variable
            'n_per_block': 2 * block.n_x + block.n_w + 2  # s-, s+, w, delta, rho
        }
    
    def evaluate_path(self,
                      path_regions: List[int],
                      variables: Dict[int, Dict[str, np.ndarray]],
                      start_state: np.ndarray,
                      goal_state: np.ndarray,
                      cost_weights: Dict[str, float]) -> Dict:
        """
        Evaluate a complete path through regions.
        
        Args:
            path_regions: List of region indices in path order.
            variables: Region-indexed decision variable values.
            start_state: Initial state.
            goal_state: Target state.
            cost_weights: Cost function weights.
            
        Returns:
            Dict with trajectory data, costs, and feasibility checks.
        """
        result = {
            'trajectories': [],
            'mesh_states': [],
            'defects': [],
            'safety_violations': [],
            'interface_errors': [],
            'control_jumps': [],
            'local_costs': [],
            'total_cost': 0.0,
            'max_control_jump': 0.0,
            'is_feasible': True
        }
        
        for i, region_idx in enumerate(path_regions):
            block = self.blocks[region_idx]
            var = variables[region_idx]
            
            s_minus = var['s_minus']
            s_plus = var['s_plus']
            w = var['w']
            delta = var['delta']
            
            # Compute trajectory
            traj, tau = block.integrator.integrate_with_trajectory(s_minus, w, delta)
            result['trajectories'].append((traj, tau, delta))
            
            # Compute defect
            defect = block.compute_defect(s_minus, s_plus, w, delta)
            result['defects'].append(defect)
            
            if np.linalg.norm(defect) > 1e-4:
                result['is_feasible'] = False
            
            # Check safety
            is_safe, violations = block.check_safety(s_minus, w, delta)
            result['safety_violations'].append(violations)
            
            if not is_safe:
                result['is_feasible'] = False
            
            # Compute local cost
            local_cost = block.compute_local_cost(s_minus, w, delta, cost_weights)
            result['local_costs'].append(local_cost)
            result['total_cost'] += local_cost
        
        # Check interface continuity
        for i in range(len(path_regions) - 1):
            curr_idx = path_regions[i]
            next_idx = path_regions[i + 1]
            
            s_plus_curr = variables[curr_idx]['s_plus']
            s_minus_next = variables[next_idx]['s_minus']
            
            interface_error = np.linalg.norm(s_plus_curr - s_minus_next)
            result['interface_errors'].append(interface_error)
            if interface_error > 1e-4:
                result['is_feasible'] = False

            control_param = self.blocks[curr_idx].control_param
            u_exit = control_param.evaluate(1.0, variables[curr_idx]['w'])
            u_entry = control_param.evaluate(0.0, variables[next_idx]['w'])
            control_jump = float(np.linalg.norm(u_exit - u_entry))
            result['control_jumps'].append(control_jump)

            if self.enforce_control_continuity and control_jump > 1e-4:
                result['is_feasible'] = False

        if result['control_jumps']:
            result['max_control_jump'] = max(result['control_jumps'])
        
        # Check boundary conditions
        first_idx = path_regions[0]
        last_idx = path_regions[-1]
        
        start_error = np.linalg.norm(variables[first_idx]['s_minus'][:2] - start_state[:2])
        goal_error = np.linalg.norm(variables[last_idx]['s_plus'][:2] - goal_state[:2])
        
        if start_error > 1e-4 or goal_error > 1e-4:
            result['is_feasible'] = False
        
        return result


def create_casadi_local_cost(dynamics: DynamicsModel,
                             control_param: ControlParameterization,
                             n_steps: int = 20) -> Callable:
    """
    Create CasADi function for local cost computation.
    
    J_v = a * Delta
        + integral[Delta * (w_L / Delta^2 * ||dq/dtau||^2 + w_E * ||u||^2)] dtau
        + w_u_smooth * sum_k ||u_{k+1} - u_k||^2
        
    Returns:
        Function(s_minus, w, delta, a, w_L, w_E, w_u_smooth) -> cost
    """
    # Symbolic variables
    s_minus = ca.MX.sym('s_minus', dynamics.n_x)
    w = ca.MX.sym('w', control_param.n_w)
    delta = ca.MX.sym('delta', 1)
    a = ca.MX.sym('a', 1)
    w_L = ca.MX.sym('w_L', 1)
    w_E = ca.MX.sym('w_E', 1)
    w_u_smooth = ca.MX.sym('w_u_smooth', 1)
    
    dt = 1.0 / n_steps
    
    # Time cost
    cost = a * delta
    
    # Integrate and accumulate running cost
    x = s_minus
    
    for i in range(n_steps):
        tau = i * dt
        tau_end = tau + dt
        
        # Get control
        u_start = control_param.evaluate_casadi(tau, w)
        u_end = control_param.evaluate_casadi(tau_end, w)
        u_mid = control_param.evaluate_casadi(tau + 0.5 * dt, w)
        
        # RK4 step
        k1 = delta * dynamics.f_casadi(x, u_start)
        k2 = delta * dynamics.f_casadi(x + 0.5 * dt * k1, u_mid)
        k3 = delta * dynamics.f_casadi(x + 0.5 * dt * k2, u_mid)
        k4 = delta * dynamics.f_casadi(x + dt * k3, u_end)
        
        x_next = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        q_dot_start = dynamics.position_velocity_casadi(x, u_start)
        q_dot_end = dynamics.position_velocity_casadi(x_next, u_end)
        running_start = w_L * ca.dot(q_dot_start, q_dot_start) + w_E * ca.dot(u_start, u_start)
        running_end = w_L * ca.dot(q_dot_end, q_dot_end) + w_E * ca.dot(u_end, u_end)

        cost = cost + delta * dt * 0.5 * (running_start + running_end)
        
        x = x_next

    if control_param.n_segments > 1:
        for seg in range(control_param.n_segments - 1):
            start_left = seg * control_param.n_u
            start_right = (seg + 1) * control_param.n_u
            u_left = w[start_left:start_left + control_param.n_u]
            u_right = w[start_right:start_right + control_param.n_u]
            diff = u_right - u_left
            cost = cost + w_u_smooth * ca.dot(diff, diff)
    
    F = ca.Function('local_cost', [s_minus, w, delta, a, w_L, w_E, w_u_smooth], [cost],
                   ['s_minus', 'w', 'delta', 'a', 'w_L', 'w_E', 'w_u_smooth'], ['cost'])
    
    return F
