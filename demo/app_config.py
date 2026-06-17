"""
app_config.py - Structured configuration loading for the demo app.

The goal of this module is to keep scenario resolution in one place so
adding a new scenario mostly means editing `config.yaml`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from problem_data import PROBLEM_PRESETS

def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update nested dictionaries."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _parse_state(raw_state: Dict[str, Any]) -> np.ndarray:
    """Parse a state dict into [px, py, theta]."""
    position = raw_state["position"]
    heading = raw_state.get("heading", 0.0)
    return np.array([position[0], position[1], heading], dtype=np.float64)


@dataclass(frozen=True)
class ResolvedScenario:
    """Fully resolved scenario settings used by the runtime pipeline."""
    key: str
    name: str
    description: str
    problem_preset: str
    active_region_indices: Optional[List[int]]
    start_state: np.ndarray
    goal_state: np.ndarray
    runtime_config: Dict[str, Any]


class DemoConfig:
    """Load and resolve the demo configuration file."""

    def __init__(self, config_path: str = "config.yaml"):
        path = Path(config_path)
        if not path.is_absolute() and not path.exists():
            path = Path(__file__).resolve().parent / config_path
        self.path = path
        with self.path.open("r") as handle:
            self.raw = yaml.safe_load(handle)

    @property
    def output_dir(self) -> str:
        output_dir = Path(self.raw.get("output", {}).get("results_dir", "results"))
        if not output_dir.is_absolute():
            output_dir = self.path.parent / output_dir
        return str(output_dir)

    @property
    def save_png(self) -> bool:
        return bool(self.raw.get("output", {}).get("save_png", True))

    @property
    def save_gif(self) -> bool:
        return bool(self.raw.get("output", {}).get("save_gif", True))

    @property
    def save_json(self) -> bool:
        return bool(self.raw.get("output", {}).get("save_json", True))

    @property
    def save_md(self) -> bool:
        return bool(self.raw.get("output", {}).get("save_md", True))

    @property
    def docs_dir(self) -> str:
        docs_dir = Path(self.raw.get("output", {}).get("docs_dir", "docs"))
        if not docs_dir.is_absolute():
            docs_dir = self.path.parent / docs_dir
        return str(docs_dir)

    def list_scenarios(self) -> List[str]:
        """Return scenario keys in config order."""
        return list(self.raw.get("scenarios", {}).keys())

    def default_scenario_name(self) -> str:
        """Return the first declared scenario."""
        scenario_names = self.list_scenarios()
        if not scenario_names:
            raise ValueError("No scenarios defined in config.")
        return scenario_names[0]

    def resolve_scenario(self, scenario_name: str) -> ResolvedScenario:
        """Merge top-level defaults with scenario-specific overrides."""
        scenarios = self.raw.get("scenarios", {})
        if scenario_name not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario_raw = scenarios[scenario_name]
        runtime_config = copy.deepcopy(self.raw)

        legacy_overrides: Dict[str, Any] = {}
        if "problem_preset" in scenario_raw:
            legacy_overrides.setdefault("problem", {})["default_preset"] = scenario_raw["problem_preset"]
        if "start_state" in scenario_raw:
            legacy_overrides["start_state"] = scenario_raw["start_state"]
        if "goal_state" in scenario_raw:
            legacy_overrides["goal_state"] = scenario_raw["goal_state"]
        if "n_mesh_points" in scenario_raw:
            legacy_overrides.setdefault("shooting", {})["n_mesh_points"] = scenario_raw["n_mesh_points"]
        if "n_control_segments" in scenario_raw:
            legacy_overrides.setdefault("control", {})["n_segments"] = scenario_raw["n_control_segments"]
        if "safety_margin" in scenario_raw:
            legacy_overrides.setdefault("shooting", {})["safety_margin"] = scenario_raw["safety_margin"]
        if "cost_a" in scenario_raw:
            legacy_overrides.setdefault("cost", {})["a"] = scenario_raw["cost_a"]
        if "cost_w_L" in scenario_raw:
            legacy_overrides.setdefault("cost", {})["w_L"] = scenario_raw["cost_w_L"]
        if "cost_w_E" in scenario_raw:
            legacy_overrides.setdefault("cost", {})["w_E"] = scenario_raw["cost_w_E"]

        _deep_update(runtime_config, legacy_overrides)
        _deep_update(runtime_config, scenario_raw.get("overrides", {}))

        active_regions = scenario_raw.get("active_regions", "all")
        if active_regions == "all":
            active_region_indices = None
        else:
            active_region_indices = list(active_regions)

        problem_preset = runtime_config.get("problem", {}).get("default_preset", "default")
        if problem_preset not in PROBLEM_PRESETS:
            available = ", ".join(PROBLEM_PRESETS.keys())
            raise ValueError(
                f"Unknown problem preset '{problem_preset}'. Available: {available}"
            )
        preset = PROBLEM_PRESETS[problem_preset]
        runtime_config.setdefault(
            "start_state",
            {
                "position": preset.default_start_state[:2],
                "heading": preset.default_start_state[2],
            },
        )
        runtime_config.setdefault(
            "goal_state",
            {
                "position": preset.default_goal_state[:2],
                "heading": preset.default_goal_state[2],
            },
        )

        return ResolvedScenario(
            key=scenario_name,
            name=scenario_raw.get("name", scenario_name),
            description=scenario_raw.get("description", ""),
            problem_preset=problem_preset,
            active_region_indices=active_region_indices,
            start_state=_parse_state(runtime_config["start_state"]),
            goal_state=_parse_state(runtime_config["goal_state"]),
            runtime_config=runtime_config,
        )
