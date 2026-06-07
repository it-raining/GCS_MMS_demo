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
) -> WarmStart:
    """Generate warm-start from interface waypoints z (shape m+2, n_pos)."""
    m = len(path_regions)
    n_int = config.n_int
    if isinstance(dynamics, UnicycleModel):
        return _warmstart_unicycle(z, m, dynamics, config, n_int)
    elif isinstance(dynamics, DoubleIntegratorDynamics):
        return _warmstart_double_integrator(z, m, dynamics, config, n_int)
    else:
        raise NotImplementedError(f"warmstart not implemented for {type(dynamics).__name__}")


def _v_nom(dynamics: DynamicsModel, config: WarmStartConfig) -> float:
    u_lb, u_ub = dynamics.control_bounds()
    if dynamics.n_u > 0:
        return config.v_nom_fraction * (float(u_ub[0]) + abs(float(u_lb[0]))) / 2.0
    return 1.0


def _warmstart_unicycle(z, m, dynamics: UnicycleModel, config, n_int) -> WarmStart:
    x_nodes_list, u_list, delta_list = [], [], []
    v_nom = max(_v_nom(dynamics, config), 1e-6)

    for i in range(m):
        z_prev, z_next = z[i], z[i + 1]
        seg_len = float(np.linalg.norm(z_next - z_prev))
        delta_i = float(np.clip(seg_len / v_nom, config.delta_min, config.delta_max))

        direction = z_next - z_prev
        theta_i = float(np.arctan2(direction[1], direction[0])) if seg_len > 1e-9 else 0.0

        if i > 0:
            prev_dir = z[i] - z[i - 1]
            prev_len = float(np.linalg.norm(prev_dir))
            theta_prev = float(np.arctan2(prev_dir[1], prev_dir[0])) if prev_len > 1e-9 else theta_i
            dtheta = min(abs(theta_i - theta_prev), 2 * np.pi - abs(theta_i - theta_prev))
            if dtheta > 0 and dtheta / delta_i > dynamics.omega_max:
                delta_i = float(np.clip(dtheta / dynamics.omega_max, config.delta_min, config.delta_max))

        taus = np.linspace(0.0, 1.0, n_int + 1)
        x_seg = np.zeros((n_int + 1, dynamics.n_x))
        for k, tau in enumerate(taus):
            x_seg[k, 0] = z_prev[0] * (1 - tau) + z_next[0] * tau
            x_seg[k, 1] = z_prev[1] * (1 - tau) + z_next[1] * tau
            x_seg[k, 2] = theta_i

        v_i = float(np.clip(seg_len / delta_i if delta_i > 0 else 0.0, dynamics.v_min, dynamics.v_max))
        omega_i = 0.0
        if i > 0:
            prev_dir = z[i] - z[i - 1]
            prev_len = float(np.linalg.norm(prev_dir))
            theta_prev = float(np.arctan2(prev_dir[1], prev_dir[0])) if prev_len > 1e-9 else theta_i
            omega_i = float(np.clip((theta_i - theta_prev) / delta_i if delta_i > 0 else 0.0,
                                    dynamics.omega_min, dynamics.omega_max))

        x_nodes_list.append(x_seg)
        u_list.append(np.array([v_i, omega_i], dtype=float))
        delta_list.append(delta_i)

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
