"""
shooting_animation.py - Direct multiple-shooting convergence animation tools.

The module records fixed-path IPOPT iterates, parses raw NLP vectors into
per-region shooting segments, and renders a GIF showing the defect gaps closing.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from dynamics import ControlParameterization, DynamicsModel, RK4Integrator
from graph_builder import RegionGraph
from visualization import plot_environment, setup_plot_style


class ShootingIterationRecorder(ca.Callback):
    """CasADi iteration callback that records IPOPT decision-vector snapshots."""

    def __init__(
        self,
        name: str = "shooting_iteration_recorder",
        nx: int = 0,
        ng: int = 0,
        record_every: int = 1,
        max_snapshots: Optional[int] = 500,
        opts: Optional[Dict] = None,
    ):
        ca.Callback.__init__(self)
        self._name = name
        self._opts = opts or {}
        self.nx = int(nx)
        self.ng = int(ng)
        self.record_every = max(1, int(record_every))
        self.max_snapshots = max_snapshots
        self.snapshots: List[Dict[str, np.ndarray | float | int]] = []
        self._callback_count = 0
        self._constructed = False
        if self.nx > 0:
            self.construct(self._name, self._opts)
            self._constructed = True

    def configure(self, nx: int, ng: int) -> None:
        """Set dimensions before the callback is attached to a concrete NLP."""
        nx = int(nx)
        ng = int(ng)
        if self._constructed:
            if nx != self.nx or ng != self.ng:
                raise ValueError(
                    "ShootingIterationRecorder was already constructed with "
                    f"dimensions nx={self.nx}, ng={self.ng}; got nx={nx}, ng={ng}"
                )
            return

        self.nx = nx
        self.ng = ng
        self.construct(self._name, self._opts)
        self._constructed = True

    def get_n_in(self) -> int:
        return ca.nlpsol_n_out()

    def get_n_out(self) -> int:
        return 1

    def get_name_in(self, i: int) -> str:
        return ca.nlpsol_out(i)

    def get_name_out(self, i: int) -> str:
        return "ret"

    def get_sparsity_in(self, i: int) -> ca.Sparsity:
        name = ca.nlpsol_out(i)
        if name == "f":
            return ca.Sparsity.scalar()
        if name in ("x", "lam_x"):
            return ca.Sparsity.dense(self.nx, 1)
        if name in ("g", "lam_g"):
            return ca.Sparsity.dense(self.ng, 1)
        return ca.Sparsity(0, 0)

    def get_sparsity_out(self, i: int) -> ca.Sparsity:
        return ca.Sparsity.scalar()

    def eval(self, arg):
        should_record = self._callback_count % self.record_every == 0
        has_capacity = (
            self.max_snapshots is None or
            len(self.snapshots) < int(self.max_snapshots)
        )

        if should_record and has_capacity:
            self.snapshots.append(
                {
                    "iteration": self._callback_count,
                    "x": np.array(arg[0], dtype=float).reshape(-1).copy(),
                    "f": float(arg[1]),
                }
            )

        self._callback_count += 1
        return [0]


def _sample_mesh_positions(
    positions: np.ndarray,
    tau: np.ndarray,
    n_mesh_points: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample trajectory positions at the configured containment mesh points."""
    if n_mesh_points is None or int(n_mesh_points) <= 0:
        return np.empty((0, positions.shape[1]), dtype=float), np.empty((0,), dtype=float)

    mesh_tau = np.linspace(0.0, 1.0, int(n_mesh_points))
    positions = np.asarray(positions, dtype=float)
    tau = np.asarray(tau, dtype=float).reshape(-1)

    if positions.shape[0] == 1 or tau.size == 1:
        return np.repeat(positions[:1], mesh_tau.size, axis=0), mesh_tau

    mesh_positions = np.empty((mesh_tau.size, positions.shape[1]), dtype=float)
    for i, mesh_t in enumerate(mesh_tau):
        if mesh_t <= tau[0]:
            mesh_positions[i] = positions[0]
            continue
        if mesh_t >= tau[-1]:
            mesh_positions[i] = positions[-1]
            continue

        left = int(np.searchsorted(tau, mesh_t, side="right") - 1)
        left = min(max(left, 0), tau.size - 2)
        right = left + 1
        span = max(float(tau[right] - tau[left]), 1e-12)
        alpha = float((mesh_t - tau[left]) / span)
        mesh_positions[i] = (1.0 - alpha) * positions[left] + alpha * positions[right]

    return mesh_positions, mesh_tau


def parse_shooting_snapshot(
    x_vec: np.ndarray,
    path_regions: List[int],
    n_x: int,
    n_w: int,
    dynamics: DynamicsModel,
    control_param: ControlParameterization,
    F_endpoint: Callable,
    n_integration_steps: int,
    n_mesh_points: Optional[int] = None,
    cost: Optional[float] = None,
    iteration: Optional[int] = None,
) -> Dict:
    """Parse one fixed-path NLP vector into per-region shooting trajectories."""
    x_vec = np.asarray(x_vec, dtype=np.float64).reshape(-1)
    vars_per_region = n_x + n_w + 1
    expected = len(path_regions) * vars_per_region
    if x_vec.size < expected:
        raise ValueError(f"Snapshot has {x_vec.size} values, expected at least {expected}")

    integrator = RK4Integrator(dynamics, control_param, n_steps=n_integration_steps)
    regions = []

    for i, region_idx in enumerate(path_regions):
        offset = i * vars_per_region
        s_minus = x_vec[offset:offset + n_x].copy()
        offset += n_x
        w = x_vec[offset:offset + n_w].copy()
        offset += n_w
        delta = float(x_vec[offset])
        s_plus = np.array(F_endpoint(s_minus, w, delta), dtype=float).reshape(-1)
        trajectory, tau = integrator.integrate_with_trajectory(s_minus, w, delta)
        positions = np.vstack([
            dynamics.project_to_position(state)
            for state in trajectory
        ])
        mesh_positions, mesh_tau = _sample_mesh_positions(
            positions,
            tau,
            n_mesh_points,
        )

        regions.append(
            {
                "region_idx": region_idx,
                "s_minus": s_minus,
                "s_plus": s_plus,
                "w": w,
                "delta": delta,
                "tau": tau,
                "trajectory": trajectory,
                "positions": positions,
                "mesh_tau": mesh_tau,
                "mesh_positions": mesh_positions,
            }
        )

    defect_norms = []
    connection_gaps = []
    for left, right in zip(regions[:-1], regions[1:]):
        state_gap = left["s_plus"] - right["s_minus"]
        pos_gap = (
            dynamics.project_to_position(left["s_plus"]) -
            dynamics.project_to_position(right["s_minus"])
        )
        defect_norms.append(float(np.linalg.norm(state_gap)))
        connection_gaps.append(float(np.linalg.norm(pos_gap)))

    return {
        "iteration": iteration,
        "cost": np.nan if cost is None else float(cost),
        "regions": regions,
        "path_regions": list(path_regions),
        "defect_norms": defect_norms,
        "connection_gaps": connection_gaps,
        "max_defect": float(max(defect_norms)) if defect_norms else 0.0,
        "max_gap": float(max(connection_gaps)) if connection_gaps else 0.0,
    }


def parse_recorder_snapshots(
    recorder: ShootingIterationRecorder,
    path_regions: List[int],
    n_x: int,
    n_w: int,
    dynamics: DynamicsModel,
    control_param: ControlParameterization,
    F_endpoint: Callable,
    n_integration_steps: int,
    n_mesh_points: Optional[int] = None,
) -> List[Dict]:
    """Parse all raw snapshots held by a ShootingIterationRecorder."""
    parsed = []
    for raw in recorder.snapshots:
        parsed.append(
            parse_shooting_snapshot(
                np.asarray(raw["x"], dtype=float),
                path_regions,
                n_x,
                n_w,
                dynamics,
                control_param,
                F_endpoint,
                n_integration_steps,
                n_mesh_points=n_mesh_points,
                cost=float(raw["f"]),
                iteration=int(raw["iteration"]),
            )
        )
    return parsed


def _frame_indices(n_snapshots: int,
                   hold_first_frames: int,
                   hold_last_frames: int,
                   max_animation_frames: int) -> List[int]:
    if n_snapshots <= 0:
        return []

    hold_first_frames = max(0, int(hold_first_frames))
    hold_last_frames = max(0, int(hold_last_frames))
    max_animation_frames = max(2, int(max_animation_frames))
    core_count = max(2, max_animation_frames - hold_first_frames - hold_last_frames)
    core_count = min(core_count, n_snapshots)

    if core_count == n_snapshots:
        core = list(range(n_snapshots))
    else:
        core = np.linspace(0, n_snapshots - 1, core_count)
        core = np.unique(np.rint(core).astype(int)).tolist()
        if core[0] != 0:
            core.insert(0, 0)
        if core[-1] != n_snapshots - 1:
            core.append(n_snapshots - 1)

    return [0] * hold_first_frames + core + [n_snapshots - 1] * hold_last_frames


def create_shooting_convergence_animation(
    snapshots: List[Dict],
    graph: RegionGraph,
    workspace_bounds: Tuple[float, float, float, float],
    start_state: np.ndarray,
    goal_state: np.ndarray,
    filename: str,
    obstacles: Optional[List[object]] = None,
    fps: int = 12,
    hold_first_frames: int = 10,
    hold_last_frames: int = 20,
    max_animation_frames: int = 200,
) -> None:
    """Render a GIF showing fixed-path multiple-shooting convergence."""
    if not snapshots:
        raise ValueError("Cannot animate shooting convergence without snapshots")

    setup_plot_style()
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    path_regions = snapshots[-1].get("path_regions", [])
    frame_ids = _frame_indices(
        len(snapshots),
        hold_first_frames,
        hold_last_frames,
        max_animation_frames,
    )
    reference_gap = max(
        [snapshot.get("max_gap", 0.0) for snapshot in snapshots] +
        [1e-9]
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_environment(
        ax,
        workspace_bounds,
        graph.regions,
        start_state,
        goal_state,
        obstacles=obstacles,
        highlight_regions=path_regions,
        show_labels=True,
        alpha=0.22,
    )
    ax.set_title("Direct Multiple Shooting Convergence")

    metrics_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
        zorder=20,
    )
    dynamic_artists = []

    def clear_dynamic_artists() -> None:
        while dynamic_artists:
            artist = dynamic_artists.pop()
            try:
                artist.remove()
            except ValueError:
                pass

    def draw_frame(frame_idx: int):
        clear_dynamic_artists()
        snapshot = snapshots[frame_ids[frame_idx]]
        regions = snapshot["regions"]

        for region_data in regions:
            positions = np.asarray(region_data["positions"], dtype=float)
            line, = ax.plot(
                positions[:, 0],
                positions[:, 1],
                color="#1f77b4",
                linewidth=2.3,
                alpha=0.9,
                solid_capstyle="round",
                zorder=8,
            )
            entry = np.asarray(region_data["s_minus"], dtype=float)[:2]
            exit_point = np.asarray(region_data["s_plus"], dtype=float)[:2]
            entry_artist = ax.scatter(
                [entry[0]],
                [entry[1]],
                s=46,
                c="#d62728",
                edgecolors="white",
                linewidths=0.8,
                zorder=12,
            )
            exit_artist = ax.scatter(
                [exit_point[0]],
                [exit_point[1]],
                s=58,
                facecolors="none",
                edgecolors="#d62728",
                linewidths=1.4,
                zorder=12,
            )
            mesh_positions = np.asarray(
                region_data.get("mesh_positions", np.empty((0, 2))),
                dtype=float,
            )
            mesh_artist = None
            if mesh_positions.size:
                mesh_artist = ax.scatter(
                    mesh_positions[:, 0],
                    mesh_positions[:, 1],
                    s=34,
                    c="#7B1FA2",
                    marker="x",
                    linewidths=1.4,
                    alpha=0.95,
                    zorder=13,
                )
            dynamic_artists.extend([line, entry_artist, exit_artist])
            if mesh_artist is not None:
                dynamic_artists.append(mesh_artist)

        for left, right, gap in zip(
            regions[:-1],
            regions[1:],
            snapshot.get("connection_gaps", []),
        ):
            if gap < 1e-5:
                continue
            start = np.asarray(left["s_plus"], dtype=float)[:2]
            end = np.asarray(right["s_minus"], dtype=float)[:2]
            ratio = min(max(gap / reference_gap, 0.0), 1.0)
            color = plt.cm.RdYlGn_r(ratio)
            arrow = ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.6,
                    linestyle="--",
                    shrinkA=4,
                    shrinkB=4,
                ),
                zorder=11,
            )
            dynamic_artists.append(arrow)

        iter_label = snapshot.get("iteration")
        cost = snapshot.get("cost", np.nan)
        cost_label = "nan" if not np.isfinite(cost) else f"{cost:.4g}"
        metrics_text.set_text(
            "\n".join(
                [
                    f"Frame {frame_idx + 1}/{len(frame_ids)}",
                    f"IPOPT iter {iter_label if iter_label is not None else 'n/a'}",
                    f"Cost {cost_label}",
                    f"Max defect {snapshot.get('max_defect', 0.0):.2e}",
                    f"Max gap {snapshot.get('max_gap', 0.0):.2e}",
                ]
            )
        )
        return [metrics_text, *dynamic_artists]

    animation = FuncAnimation(
        fig,
        draw_frame,
        frames=len(frame_ids),
        interval=1000 / max(1, int(fps)),
        blit=False,
    )
    animation.save(filename, writer=PillowWriter(fps=max(1, int(fps))))
    clear_dynamic_artists()
    plt.close(fig)
    print(f"Shooting convergence animation saved to {filename}")
