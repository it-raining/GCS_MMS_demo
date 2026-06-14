"""
warmstart.py - IVP warm-start generation for Centroid-Refine-DMS.

Strict-interior invariant: since z[i] and z[i+1] both satisfy
A q <= b - (delta_safe + delta_extra), every convex combination does too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from dynamics import DynamicsModel, DoubleIntegratorDynamics, UnicycleModel


@dataclass
class WarmStartConfig:
    delta_min: float = 0.01
    delta_max: float = 10.0
    v_nom_fraction: float = 0.5
    n_int: int = 20


@dataclass
class WarmStart:
    x_nodes: List[np.ndarray]  # [segment][node, n_x]
    u_list: List[np.ndarray]   # [segment] constant control
    delta_list: List[float]    # [segment] duration


def generate_warm_start_ivp(
    z: np.ndarray,
    path_regions: List[int],
    dynamics: DynamicsModel,
    config: WarmStartConfig,
    theta_start: float | None = None,
    theta_end: float | None = None,
) -> WarmStart:
    """Generate warm-start from interface waypoints z (shape m+2, n_pos).

    For unicycle: theta_start/theta_end pin the heading at the very first and
    last warm-start node to match the DMS start/goal equality constraints,
    eliminating start/goal theta violations that cause IPOPT infeasibility.
    """
    m = len(path_regions)
    n_int = config.n_int
    if isinstance(dynamics, UnicycleModel):
        return _warmstart_unicycle(z, m, dynamics, config, n_int, theta_start, theta_end)
    elif isinstance(dynamics, DoubleIntegratorDynamics):
        return _warmstart_double_integrator(z, m, dynamics, config, n_int)
    else:
        raise NotImplementedError(f"warmstart not implemented for {type(dynamics).__name__}")


def _v_nom(dynamics: DynamicsModel, config: WarmStartConfig) -> float:
    u_lb, u_ub = dynamics.control_bounds()
    if dynamics.n_u > 0:
        return config.v_nom_fraction * (float(u_ub[0]) + abs(float(u_lb[0]))) / 2.0
    return 1.0


def _seg_heading(z_from: np.ndarray, z_to: np.ndarray, fallback: float) -> float:
    d = z_to - z_from
    seg_len = float(np.linalg.norm(d))
    return float(np.arctan2(d[1], d[0])) if seg_len > 1e-9 else fallback


def _angle_interp(theta_a: float, theta_b: float, t: float) -> float:
    """Linearly interpolate angle from theta_a to theta_b via shortest arc."""
    diff = theta_b - theta_a
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return theta_a + t * diff


def _warmstart_unicycle(
    z, m, dynamics: UnicycleModel, config, n_int,
    theta_start: float | None = None,
    theta_end: float | None = None,
) -> WarmStart:
    """Unicycle warm start via forward RK4 integration (zero defect).

    Each segment is integrated forward with constant control (v_i, omega_i)
    so x_{i,k+1} = RK4(x_{i,k}, u_i, h_i) exactly, giving zero DMS defect
    at warm-start.  Coupling is also zero because x_{i+1,0} is set to
    x_{i,n_int} (the propagated endpoint).

    theta_start pins x_{0,0}.theta to match the DMS start boundary.
    theta_end steers the last segment toward the DMS goal heading.
    """
    x_nodes_list, u_list, delta_list = [], [], []
    v_nom = max(_v_nom(dynamics, config), 1e-6)

    # Planned heading at each waypoint z[0..m].
    seg_theta: list[float] = []
    for i in range(m + 1):
        fallback = seg_theta[-1] if seg_theta else 0.0
        seg_theta.append(_seg_heading(z[i], z[i + 1], fallback))

    if theta_start is not None:
        seg_theta[0] = float(theta_start)
    if theta_end is not None:
        seg_theta[m] = float(theta_end)

    # x_prev_end tracks the propagated state; initialise at the start waypoint.
    x_prev_end = np.array([z[0][0], z[0][1], seg_theta[0]], dtype=float)

    for i in range(m):
        z_next = z[i + 1]
        seg_len = float(np.linalg.norm(z_next - z[i]))

        th_s = float(x_prev_end[2])
        th_e = seg_theta[i + 1]
        diff = th_e - th_s
        diff = float((diff + np.pi) % (2 * np.pi) - np.pi)  # shortest arc
        dtheta = abs(diff)

        delta_i = float(np.clip(seg_len / v_nom, config.delta_min, config.delta_max))
        if dtheta > 0 and dtheta / delta_i > dynamics.omega_max:
            delta_i = float(np.clip(dtheta / dynamics.omega_max, config.delta_min, config.delta_max))

        h_i = delta_i / n_int
        v_i = float(np.clip(seg_len / delta_i if delta_i > 0 else 0.0, dynamics.v_min, dynamics.v_max))
        omega_i = float(np.clip(diff / delta_i if delta_i > 0 else 0.0,
                                dynamics.omega_min, dynamics.omega_max))
        u_i = np.array([v_i, omega_i], dtype=float)

        # Forward RK4: zero defect by construction.
        x_seg = np.zeros((n_int + 1, dynamics.n_x))
        x_seg[0] = x_prev_end
        for k in range(n_int):
            xk = x_seg[k]
            k1 = h_i * dynamics.f(xk, u_i)
            k2 = h_i * dynamics.f(xk + k1 / 2, u_i)
            k3 = h_i * dynamics.f(xk + k2 / 2, u_i)
            k4 = h_i * dynamics.f(xk + k3, u_i)
            x_seg[k + 1] = xk + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        x_nodes_list.append(x_seg)
        u_list.append(u_i)
        delta_list.append(delta_i)
        x_prev_end = x_seg[-1].copy()

    return WarmStart(x_nodes=x_nodes_list, u_list=u_list, delta_list=delta_list)


def _warmstart_double_integrator(z, m, dynamics: DoubleIntegratorDynamics, config, n_int) -> WarmStart:
    x_nodes_list, u_list, delta_list = [], [], []
    v_nom = max(_v_nom(dynamics, config), 1e-6)

    for i in range(m):
        z_prev, z_next = z[i], z[i + 1]
        seg_len = float(np.linalg.norm(z_next - z_prev))
        delta_i = float(np.clip(seg_len / v_nom, config.delta_min, config.delta_max))

        vel = np.clip((z_next - z_prev) / delta_i, -dynamics.v_max, dynamics.v_max)

        taus = np.linspace(0.0, 1.0, n_int + 1)
        x_seg = np.zeros((n_int + 1, dynamics.n_x))
        for k, tau in enumerate(taus):
            x_seg[k, 0] = z_prev[0] * (1 - tau) + z_next[0] * tau
            x_seg[k, 1] = z_prev[1] * (1 - tau) + z_next[1] * tau
            x_seg[k, 2] = vel[0]
            x_seg[k, 3] = vel[1]

        x_nodes_list.append(x_seg)
        u_list.append(np.zeros(dynamics.n_u, dtype=float))
        delta_list.append(delta_i)

    return WarmStart(x_nodes=x_nodes_list, u_list=u_list, delta_list=delta_list)
