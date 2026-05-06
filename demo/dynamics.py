"""
dynamics.py - Dynamics models for motion planning.

Implements:
- Unicycle model (simplified): state = [px, py, theta], control = [v, omega]
- RK4 integration on normalized time tau in [0, 1]
- Control parameterization (constant, piecewise constant)
"""

import numpy as np
from typing import Tuple, Callable, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import casadi as ca


class DynamicsModel(ABC):
    """Abstract base class for dynamics models."""
    
    @property
    @abstractmethod
    def n_x(self) -> int:
        """State dimension."""
        pass
    
    @property
    @abstractmethod
    def n_u(self) -> int:
        """Control dimension."""
        pass
    
    @property
    @abstractmethod
    def n_pos(self) -> int:
        """Position dimension (for safety constraints)."""
        pass

    @property
    @abstractmethod
    def position_indices(self) -> Tuple[int, ...]:
        """State indices that represent workspace position."""
        pass

    @property
    def angle_indices(self) -> Tuple[int, ...]:
        """State indices with angular wrapping semantics."""
        return ()
    
    @abstractmethod
    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Continuous-time dynamics: dx/dt = f(x, u).
        
        Args:
            x: State vector
            u: Control vector
            
        Returns:
            State derivative dx/dt
        """
        pass
    
    @abstractmethod
    def f_casadi(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """CasADi symbolic version of dynamics."""
        pass
    
    @abstractmethod
    def project_to_position(self, x: np.ndarray) -> np.ndarray:
        """
        Project state to position space for safety constraints.
        
        Args:
            x: Full state vector
            
        Returns:
            Position vector (2D)
        """
        pass
    
    @abstractmethod
    def project_to_position_casadi(self, x: ca.MX) -> ca.MX:
        """CasADi symbolic version of projection."""
        pass

    @abstractmethod
    def position_velocity(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Physical position derivative dq/dt for the current state/control.

        This is used by objective terms that penalize translational motion.
        """
        pass

    @abstractmethod
    def position_velocity_casadi(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """CasADi symbolic version of the physical position derivative."""
        pass

    @abstractmethod
    def state_bounds(self,
                     position_lb: Optional[np.ndarray] = None,
                     position_ub: Optional[np.ndarray] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return lower/upper bounds for state variables.

        Position bounds are supplied by the environment because they are graph
        dependent; non-position bounds belong to the dynamics model.
        """
        pass

    @abstractmethod
    def control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower/upper bounds for one instantaneous control vector."""
        pass

    def nominal_control(self) -> np.ndarray:
        """Return a feasible initial control guess."""
        u_min, u_max = self.control_bounds()
        return 0.5 * (u_min + u_max)

    def angle_big_m(self) -> float:
        """Return a Big-M value for angular state components."""
        return 2.0 * np.pi


@dataclass
class UnicycleModel(DynamicsModel):
    """
    Simplified unicycle (kinematic) model.
    
    State: x = [px, py, theta]
        - px, py: position in 2D plane
        - theta: heading angle (radians)
    
    Control: u = [v, omega]
        - v: linear velocity
        - omega: angular velocity
    
    Dynamics:
        dx/dt = v * cos(theta)
        dy/dt = v * sin(theta)
        dtheta/dt = omega
    
    This is a nonholonomic system - cannot move sideways directly.
    """
    
    v_min: float = -2.0
    v_max: float = 2.0
    omega_min: float = -np.pi
    omega_max: float = np.pi
    
    @property
    def n_x(self) -> int:
        return 3
    
    @property
    def n_u(self) -> int:
        return 2
    
    @property
    def n_pos(self) -> int:
        return 2

    @property
    def position_indices(self) -> Tuple[int, ...]:
        return (0, 1)

    @property
    def angle_indices(self) -> Tuple[int, ...]:
        return (2,)
    
    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Unicycle dynamics.
        
        Args:
            x: [px, py, theta]
            u: [v, omega]
            
        Returns:
            [v*cos(theta), v*sin(theta), omega]
        """
        px, py, theta = x[0], x[1], x[2]
        v, omega = u[0], u[1]
        
        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            omega
        ])
    
    def f_casadi(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """CasADi symbolic dynamics."""
        theta = x[2]
        v = u[0]
        omega = u[1]
        
        return ca.vertcat(
            v * ca.cos(theta),
            v * ca.sin(theta),
            omega
        )
    
    def project_to_position(self, x: np.ndarray) -> np.ndarray:
        """Extract position [px, py] from state."""
        return x[:2]
    
    def project_to_position_casadi(self, x: ca.MX) -> ca.MX:
        """CasADi symbolic position extraction."""
        return x[:2]

    def position_velocity(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Physical translational velocity [dx/dt, dy/dt]."""
        theta = x[2]
        v = u[0]
        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
        ])

    def position_velocity_casadi(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """CasADi symbolic translational velocity."""
        theta = x[2]
        v = u[0]
        return ca.vertcat(
            v * ca.cos(theta),
            v * ca.sin(theta),
        )

    def state_bounds(self,
                     position_lb: Optional[np.ndarray] = None,
                     position_ub: Optional[np.ndarray] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Return bounds for [px, py, theta]."""
        if position_lb is None:
            position_lb = np.array([-np.inf, -np.inf], dtype=np.float64)
        if position_ub is None:
            position_ub = np.array([np.inf, np.inf], dtype=np.float64)

        lb = np.array([position_lb[0], position_lb[1], -2.0 * np.pi], dtype=np.float64)
        ub = np.array([position_ub[0], position_ub[1], 2.0 * np.pi], dtype=np.float64)
        return lb, ub

    def control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return bounds for [v, omega]."""
        return (
            np.array([self.v_min, self.omega_min], dtype=np.float64),
            np.array([self.v_max, self.omega_max], dtype=np.float64),
        )

    def nominal_control(self) -> np.ndarray:
        """Prefer a small positive forward speed as the NLP initial guess."""
        u_min, u_max = self.control_bounds()
        return np.clip(np.array([0.5, 0.0], dtype=np.float64), u_min, u_max)
    
    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get control bounds.
        
        Returns:
            (u_min, u_max) arrays
        """
        return self.control_bounds()


class ControlParameterization:
    """
    Control parameterization for a region.
    
    Maps parameter vector w_v to control signal u_v(tau) on [0, 1].
    """
    
    def __init__(self, n_u: int, parameterization: str = "constant", 
                 n_segments: int = 1):
        """
        Args:
            n_u: Control dimension
            parameterization: "constant" or "piecewise_constant"
            n_segments: Number of segments (for piecewise_constant)
        """
        self.n_u = n_u
        self.parameterization = parameterization
        self.n_segments = n_segments if parameterization == "piecewise_constant" else 1
        
        # Parameter dimension
        self.n_w = n_u * self.n_segments
        self.soft_switch_sharpness = 50.0

    def _smooth_segment_weights_numpy(self, tau: float) -> np.ndarray:
        """Mirror the CasADi soft segment activation in NumPy."""
        weights = np.zeros(self.n_segments, dtype=np.float64)
        sharpness = self.soft_switch_sharpness

        for seg in range(self.n_segments):
            t_start = seg / self.n_segments
            t_end = (seg + 1) / self.n_segments

            weight_start = 1.0 / (1.0 + np.exp(-sharpness * (tau - t_start)))
            weight_end = 1.0 / (1.0 + np.exp(-sharpness * (t_end - tau)))
            weights[seg] = weight_start * weight_end

        return weights

    @staticmethod
    def _normalize_weights_numpy(weights: np.ndarray) -> np.ndarray:
        """Normalize soft segment weights so control amplitude stays consistent."""
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            return weights
        return weights / weight_sum

    def _boundary_segment_control_numpy(self, tau: float, w: np.ndarray) -> Optional[np.ndarray]:
        """Return exact endpoint controls for Python numeric boundary times."""
        if self.parameterization != "piecewise_constant":
            return None
        if tau <= 0.0:
            return self.get_segment_control(w, 0)
        if tau >= 1.0:
            return self.get_segment_control(w, self.n_segments - 1)
        return None

    def _boundary_segment_control_casadi(self, tau, w: ca.MX) -> Optional[ca.MX]:
        """Return exact endpoint controls when CasADi evaluation receives a Python scalar."""
        if self.parameterization != "piecewise_constant":
            return None
        if not isinstance(tau, (int, float, np.floating)):
            return None

        tau_float = float(tau)
        if tau_float <= 0.0:
            return w[:self.n_u]
        if tau_float >= 1.0:
            start_idx = (self.n_segments - 1) * self.n_u
            return w[start_idx:start_idx + self.n_u]
        return None
    
    def evaluate(self, tau: float, w: np.ndarray) -> np.ndarray:
        """
        Evaluate control at normalized time tau.
        
        Args:
            tau: Normalized time in [0, 1]
            w: Parameter vector of length n_w
            
        Returns:
            Control vector of length n_u
        """
        if self.parameterization == "constant":
            return w[:self.n_u]
        
        elif self.parameterization == "piecewise_constant":
            tau_float = float(tau)
            endpoint_control = self._boundary_segment_control_numpy(tau_float, w)
            if endpoint_control is not None:
                return endpoint_control

            result = np.zeros(self.n_u, dtype=np.float64)
            weights = self._normalize_weights_numpy(
                self._smooth_segment_weights_numpy(tau_float)
            )

            for seg, weight in enumerate(weights):
                start_idx = seg * self.n_u
                u_seg = w[start_idx:start_idx + self.n_u]
                result = result + weight * u_seg

            return result
        
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
    
    def evaluate_casadi(self, tau: ca.MX, w: ca.MX) -> ca.MX:
        """
        CasADi symbolic evaluation.
        
        For piecewise constant, we use smooth approximation to avoid
        discontinuities in the optimization.
        """
        if self.parameterization == "constant":
            return w[:self.n_u]
        
        elif self.parameterization == "piecewise_constant":
            endpoint_control = self._boundary_segment_control_casadi(tau, w)
            if endpoint_control is not None:
                return endpoint_control

            # Use weighted combination with soft switching
            # This is an approximation for optimization purposes
            result = ca.MX.zeros(self.n_u)
            sharpness = self.soft_switch_sharpness
            weight_sum = 0
            
            for seg in range(self.n_segments):
                # Segment boundaries
                t_start = seg / self.n_segments
                t_end = (seg + 1) / self.n_segments
                
                # Soft indicator function (sigmoid-like)
                weight_start = 1 / (1 + ca.exp(-sharpness * (tau - t_start)))
                weight_end = 1 / (1 + ca.exp(-sharpness * (t_end - tau)))
                weight = weight_start * weight_end
                
                # Control for this segment
                start_idx = seg * self.n_u
                u_seg = w[start_idx:start_idx + self.n_u]
                
                result = result + weight * u_seg
                weight_sum = weight_sum + weight
            
            return result / (weight_sum + 1e-12)
        
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
    
    def get_segment_control(self, w: np.ndarray, segment: int) -> np.ndarray:
        """Get control vector for a specific segment."""
        start_idx = segment * self.n_u
        return w[start_idx:start_idx + self.n_u]


class RK4Integrator:
    """
    Runge-Kutta 4th order integrator for normalized time.
    
    Integrates: dx/dtau = Delta * f(x, u) on tau in [0, 1]
    """
    
    def __init__(self, dynamics: DynamicsModel, control_param: ControlParameterization,
                 n_steps: int = 20):
        """
        Args:
            dynamics: DynamicsModel object
            control_param: ControlParameterization object
            n_steps: Number of integration steps
        """
        self.dynamics = dynamics
        self.control_param = control_param
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps  # Step size in normalized time
    
    def integrate(self, s_minus: np.ndarray, w: np.ndarray, delta: float) -> np.ndarray:
        """
        Integrate from s_minus to s_plus.
        
        Args:
            s_minus: Initial state (entry state)
            w: Control parameters
            delta: Time duration (scaling factor)
            
        Returns:
            Final state (exit state)
        """
        x = s_minus.copy()

        for i in range(self.n_steps):
            x = _rk4_step_numpy(
                self.dynamics,
                self.control_param,
                x,
                w,
                delta,
                i * self.dt,
                self.dt,
            )
        
        return x
    
    def integrate_with_trajectory(self, s_minus: np.ndarray, w: np.ndarray, 
                                   delta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate and return full trajectory.
        
        Returns:
            (trajectory, tau_values)
            trajectory: Array of shape (n_steps+1, n_x)
            tau_values: Array of shape (n_steps+1,)
        """
        trajectory = [s_minus.copy()]
        tau_values = [0.0]
        
        x = s_minus.copy()

        for i in range(self.n_steps):
            tau = i * self.dt
            x = _rk4_step_numpy(
                self.dynamics,
                self.control_param,
                x,
                w,
                delta,
                tau,
                self.dt,
            )
            
            trajectory.append(x.copy())
            tau_values.append(tau + self.dt)
        
        return np.array(trajectory), np.array(tau_values)
    
    def sample_trajectory_at_mesh(self, s_minus: np.ndarray, w: np.ndarray,
                                   delta: float, n_mesh: int) -> np.ndarray:
        """
        Sample trajectory at mesh points for safety constraints.
        
        Args:
            s_minus: Initial state
            w: Control parameters  
            delta: Time duration
            n_mesh: Number of mesh points
            
        Returns:
            Array of shape (n_mesh, n_x)
        """
        # Get full trajectory
        traj, tau_vals = self.integrate_with_trajectory(s_minus, w, delta)
        
        # Sample at mesh points
        mesh_tau = np.linspace(0, 1, n_mesh)
        mesh_states = np.zeros((n_mesh, self.dynamics.n_x))
        
        for i, tau in enumerate(mesh_tau):
            # Find bracketing indices
            idx = np.searchsorted(tau_vals, tau, side='right') - 1
            idx = max(0, min(idx, len(tau_vals) - 2))
            
            # Linear interpolation
            t0, t1 = tau_vals[idx], tau_vals[idx + 1]
            alpha = (tau - t0) / (t1 - t0 + 1e-12)
            mesh_states[i] = (1 - alpha) * traj[idx] + alpha * traj[idx + 1]
        
        return mesh_states


@dataclass(frozen=True)
class CasADiIntegrationBundle:
    """Shared CasADi functions for one dynamics/control transcription."""

    F_endpoint: Callable
    mesh_sampler: Callable
    cbf_sampler: Callable
    local_cost_fn: Callable


def _rk4_step_numpy(dynamics: DynamicsModel,
                    control_param: ControlParameterization,
                    x: np.ndarray,
                    w: np.ndarray,
                    delta: float,
                    tau: float,
                    dt: float) -> np.ndarray:
    """One normalized-time RK4 step for NumPy evaluation."""
    tau_mid = tau + 0.5 * dt
    tau_end = tau + dt

    u_start = control_param.evaluate(tau, w)
    u_mid = control_param.evaluate(tau_mid, w)
    u_end = control_param.evaluate(tau_end, w)

    k1 = delta * dynamics.f(x, u_start)
    k2 = delta * dynamics.f(x + 0.5 * dt * k1, u_mid)
    k3 = delta * dynamics.f(x + 0.5 * dt * k2, u_mid)
    k4 = delta * dynamics.f(x + dt * k3, u_end)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rk4_step_casadi(dynamics: DynamicsModel,
                     control_param: ControlParameterization,
                     x: ca.MX,
                     w: ca.MX,
                     delta: ca.MX,
                     tau: float,
                     dt: float) -> ca.MX:
    """One normalized-time RK4 step for CasADi graph construction."""
    tau_mid = tau + 0.5 * dt
    tau_end = tau + dt

    u_start = control_param.evaluate_casadi(tau, w)
    u_mid = control_param.evaluate_casadi(tau_mid, w)
    u_end = control_param.evaluate_casadi(tau_end, w)

    k1 = delta * dynamics.f_casadi(x, u_start)
    k2 = delta * dynamics.f_casadi(x + 0.5 * dt * k1, u_mid)
    k3 = delta * dynamics.f_casadi(x + 0.5 * dt * k2, u_mid)
    k4 = delta * dynamics.f_casadi(x + dt * k3, u_end)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _casadi_trajectory_states(dynamics: DynamicsModel,
                              control_param: ControlParameterization,
                              s_minus: ca.MX,
                              w: ca.MX,
                              delta: ca.MX,
                              n_steps: int) -> List[ca.MX]:
    """Build CasADi state expressions at every RK4 grid point."""
    dt = 1.0 / n_steps
    x = s_minus
    trajectory = [x]

    for i in range(n_steps):
        x = _rk4_step_casadi(
            dynamics,
            control_param,
            x,
            w,
            delta,
            i * dt,
            dt,
        )
        trajectory.append(x)

    return trajectory


def create_casadi_integrator(dynamics: DynamicsModel,
                             control_param: ControlParameterization,
                             n_steps: int = 20) -> Callable:
    """
    Create CasADi symbolic integrator function.
    
    Returns:
        Function(s_minus, w, delta) -> s_plus
    """
    # Symbolic variables
    s_minus = ca.MX.sym('s_minus', dynamics.n_x)
    w = ca.MX.sym('w', control_param.n_w)
    delta = ca.MX.sym('delta', 1)
    
    s_plus = _casadi_trajectory_states(
        dynamics, control_param, s_minus, w, delta, n_steps
    )[-1]
    
    # Create function
    F = ca.Function('endpoint_map', [s_minus, w, delta], [s_plus],
                   ['s_minus', 'w', 'delta'], ['s_plus'])
    
    return F


def create_casadi_ctcs_integrator(dynamics: DynamicsModel,
                                  control_param: ControlParameterization,
                                  region,
                                  n_steps: int = 20,
                                  safety_margin: float = 0.02,
                                  penalty: str = "squared_hinge",
                                  integral_mode: str = "normalized") -> Callable:
    """
    Create a CasADi augmented RK4 integrator for the CTCS safety certificate.

    Returns:
        Function(s_minus, w, delta) -> (s_plus, eta_end)

    eta_end is the RK4 approximation of one of:
        normalized:    integral_0^1 Lambda(x(tau)) dtau
        physical_time: integral_0^1 delta * Lambda(x(tau)) dtau

    Lambda uses the actual region membership residual A*pi(x)-b. The mesh
    safety transcription may still use safety_margin as a stricter finite-node
    interior guard, but CTCS itself certificates membership in Q_v.

    The normalized mode is the safer default for geometric containment because
    the optimizer cannot hide large violations by shrinking the segment time.
    """
    if penalty != "squared_hinge":
        raise ValueError(f"Unsupported CTCS penalty: {penalty}")
    if integral_mode not in ("normalized", "physical_time"):
        raise ValueError(f"Unsupported CTCS integral mode: {integral_mode}")

    s_minus = ca.MX.sym('s_minus', dynamics.n_x)
    w = ca.MX.sym('w', control_param.n_w)
    delta = ca.MX.sym('delta', 1)

    A_dm = ca.DM(region.A)
    b_dm = ca.DM(region.b)
    dt = 1.0 / n_steps
    x = s_minus
    eta = ca.MX.zeros(1)

    def lambda_v(x_stage: ca.MX) -> ca.MX:
        pos = dynamics.project_to_position_casadi(x_stage)
        residual = ca.mtimes(A_dm, pos) - b_dm
        hinge = ca.fmax(0, residual)
        return ca.sum1(hinge * hinge)

    for i in range(n_steps):
        tau = i * dt
        tau_mid = tau + 0.5 * dt
        tau_end = tau + dt

        u_start = control_param.evaluate_casadi(tau, w)
        u_mid = control_param.evaluate_casadi(tau_mid, w)
        u_end = control_param.evaluate_casadi(tau_end, w)

        k1_x = delta * dynamics.f_casadi(x, u_start)
        eta_scale = delta if integral_mode == "physical_time" else 1.0
        k1_eta = eta_scale * lambda_v(x)

        x2 = x + 0.5 * dt * k1_x
        k2_x = delta * dynamics.f_casadi(x2, u_mid)
        k2_eta = eta_scale * lambda_v(x2)

        x3 = x + 0.5 * dt * k2_x
        k3_x = delta * dynamics.f_casadi(x3, u_mid)
        k3_eta = eta_scale * lambda_v(x3)

        x4 = x + dt * k3_x
        k4_x = delta * dynamics.f_casadi(x4, u_end)
        k4_eta = eta_scale * lambda_v(x4)

        x = x + (dt / 6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        eta = eta + (dt / 6) * (k1_eta + 2*k2_eta + 2*k3_eta + k4_eta)

    F = ca.Function(
        'ctcs_endpoint_map',
        [s_minus, w, delta],
        [x, eta],
        ['s_minus', 'w', 'delta'],
        ['s_plus', 'eta_end']
    )

    return F


def create_casadi_trajectory_sampler(dynamics: DynamicsModel,
                                      control_param: ControlParameterization,
                                      n_steps: int = 20,
                                      n_mesh: int = 10) -> Callable:
    """
    Create CasADi function that returns mesh point positions.
    
    Returns:
        Function(s_minus, w, delta) -> [pos_0, ..., pos_{n_mesh-1}]
    """
    # Symbolic variables
    s_minus = ca.MX.sym('s_minus', dynamics.n_x)
    w = ca.MX.sym('w', control_param.n_w)
    delta = ca.MX.sym('delta', 1)
    
    trajectory = _casadi_trajectory_states(
        dynamics, control_param, s_minus, w, delta, n_steps
    )
    
    # Sample at mesh points using linear interpolation
    mesh_tau = np.linspace(0, 1, n_mesh)
    tau_vals = np.linspace(0, 1, n_steps + 1)
    
    mesh_positions = []
    for tau in mesh_tau:
        # Find bracketing indices
        idx = min(int(tau * n_steps), n_steps - 1)
        
        t0, t1 = tau_vals[idx], tau_vals[idx + 1]
        alpha = (tau - t0) / (t1 - t0 + 1e-12)
        
        # Interpolate state
        state = (1 - alpha) * trajectory[idx] + alpha * trajectory[idx + 1]
        
        # Extract position
        pos = dynamics.project_to_position_casadi(state)
        mesh_positions.append(pos)
    
    # Stack into matrix
    mesh_matrix = ca.hcat(mesh_positions).T  # Shape: (n_mesh, n_pos)
    
    F = ca.Function('trajectory_mesh', [s_minus, w, delta], [mesh_matrix],
                   ['s_minus', 'w', 'delta'], ['mesh_positions'])
    
    return F


def create_casadi_cbf_sampler(dynamics: DynamicsModel,
                              control_param: ControlParameterization,
                              n_steps: int = 20,
                              n_mesh: int = 10) -> Callable:
    """
    Create CasADi function returning mesh positions and physical velocities.

    Returns:
        Function(s_minus, w, delta) -> (positions, velocities), each with shape
        (n_mesh, n_pos). Velocities are dq/dt, not normalized by Delta.
    """
    s_minus = ca.MX.sym('s_minus', dynamics.n_x)
    w = ca.MX.sym('w', control_param.n_w)
    delta = ca.MX.sym('delta', 1)

    trajectory = _casadi_trajectory_states(
        dynamics, control_param, s_minus, w, delta, n_steps
    )

    mesh_tau = np.linspace(0, 1, n_mesh)
    tau_vals = np.linspace(0, 1, n_steps + 1)

    mesh_positions = []
    mesh_velocities = []
    for tau in mesh_tau:
        idx = min(int(tau * n_steps), n_steps - 1)

        t0, t1 = tau_vals[idx], tau_vals[idx + 1]
        alpha = (tau - t0) / (t1 - t0 + 1e-12)
        state = (1 - alpha) * trajectory[idx] + alpha * trajectory[idx + 1]

        control = control_param.evaluate_casadi(float(tau), w)
        pos = dynamics.project_to_position_casadi(state)
        vel = dynamics.position_velocity_casadi(state, control)
        mesh_positions.append(pos)
        mesh_velocities.append(vel)

    position_matrix = ca.hcat(mesh_positions).T
    velocity_matrix = ca.hcat(mesh_velocities).T

    return ca.Function(
        'trajectory_cbf_mesh',
        [s_minus, w, delta],
        [position_matrix, velocity_matrix],
        ['s_minus', 'w', 'delta'],
        ['mesh_positions', 'mesh_velocities'],
    )


def create_casadi_local_cost(dynamics: DynamicsModel,
                             control_param: ControlParameterization,
                             n_steps: int = 20) -> Callable:
    """
    Create CasADi function for local cost computation.

    J_v = a * Delta
        + integral[w_L * ||q_dot||^2 + w_E * ||u||^2] dtau
        + w_u_smooth * sum_k ||u_{k+1} - u_k||^2
    """
    s_minus = ca.MX.sym('s_minus', dynamics.n_x)
    w = ca.MX.sym('w', control_param.n_w)
    delta = ca.MX.sym('delta', 1)
    a = ca.MX.sym('a', 1)
    w_L = ca.MX.sym('w_L', 1)
    w_E = ca.MX.sym('w_E', 1)
    w_u_smooth = ca.MX.sym('w_u_smooth', 1)

    dt = 1.0 / n_steps
    cost = a * delta
    x = s_minus

    for i in range(n_steps):
        tau = i * dt
        tau_end = tau + dt

        u_start = control_param.evaluate_casadi(tau, w)
        u_end = control_param.evaluate_casadi(tau_end, w)

        x_next = _rk4_step_casadi(
            dynamics,
            control_param,
            x,
            w,
            delta,
            tau,
            dt,
        )

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

    return ca.Function(
        'local_cost',
        [s_minus, w, delta, a, w_L, w_E, w_u_smooth],
        [cost],
        ['s_minus', 'w', 'delta', 'a', 'w_L', 'w_E', 'w_u_smooth'],
        ['cost'],
    )


def create_integration_bundle(dynamics: DynamicsModel,
                              control_param: ControlParameterization,
                              n_steps: int = 20,
                              n_mesh: int = 10) -> CasADiIntegrationBundle:
    """Build all CasADi functions used by one optimizer transcription."""
    return CasADiIntegrationBundle(
        F_endpoint=create_casadi_integrator(dynamics, control_param, n_steps),
        mesh_sampler=create_casadi_trajectory_sampler(
            dynamics, control_param, n_steps, n_mesh
        ),
        cbf_sampler=create_casadi_cbf_sampler(
            dynamics, control_param, n_steps, n_mesh
        ),
        local_cost_fn=create_casadi_local_cost(dynamics, control_param, n_steps),
    )
