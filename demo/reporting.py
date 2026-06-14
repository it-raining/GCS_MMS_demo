"""
reporting.py - Markdown report generation for GCS-DMS optimization results.

Generates per-scenario and benchmark summary Markdown files in docs/.
Import-safe: no circular dependencies with optimizer, experiments, or scenario_builder.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Big-M status tables keyed by formulation_mode string
# ---------------------------------------------------------------------------

_BIG_M_STATUS_BY_MODE: Dict[str, Dict[str, str]] = {
    "LEGACY_INTEGRATED_BIGM_RELAXATION_IPOPT": {
        "region_activation": (
            "**Big-M** — `|s| ≤ M·p`, `|w| ≤ M·p`, `Δ ≤ Δmax·p` force "
            "state/control/time to zero for inactive regions"
        ),
        "region_containment": (
            "**Big-M** — `A q − b ≤ M(1−p)` guards endpoint and mesh safety"
        ),
        "interface_containment": (
            "**Big-M** — `A z_pos − b ≤ M(1−y)` guards interface point in intersection"
        ),
        "interface_equality": (
            "**Big-M** — `s⁺[u] − z ≤ M(1−y)` and `s⁻[v] − z ≤ M(1−y)` for edge continuity"
        ),
        "dms_defect": (
            "**No Big-M** — defect `s⁺ = F_endpoint(s⁻, w, Δ)` applied unconditionally; "
            "trivially satisfied for inactive regions (activation forces all vars to zero)"
        ),
        "control_continuity": (
            "**Big-M** — `|u_exit − u_entry| ≤ M(1−y)` when `enforce_control_continuity=True`"
        ),
    },
    "TWO_STAGE_HEURISTIC_PATH_SELECTION_FIXED_PATH_NLP": {
        "region_activation": (
            "**None** — only active path regions have decision variables"
        ),
        "region_containment": (
            "**Direct inequality** — `A q ≤ b` applied directly (no Big-M)"
        ),
        "interface_containment": (
            "**Direct inequality** — handled by fixed-path coupling (no Big-M)"
        ),
        "interface_equality": (
            "**Direct equality** — `s⁺[i] = s⁻[i+1]` enforced without Big-M"
        ),
        "dms_defect": (
            "**No Big-M** — `s⁺ = F_endpoint(s⁻, w, Δ)` applied directly for each active region"
        ),
        "control_continuity": (
            "**Direct equality** — `u_exit = u_entry` when `enforce_control_continuity=True`, else disabled"
        ),
    },
    "FIXED_PATH_NLP": {
        "region_activation": "**None** — only active path regions have decision variables",
        "region_containment": "**Direct inequality** — `A q ≤ b` applied directly (no Big-M)",
        "interface_containment": "**Direct inequality** — handled by fixed-path coupling (no Big-M)",
        "interface_equality": "**Direct equality** — `s⁺[i] = s⁻[i+1]` enforced without Big-M",
        "dms_defect": "**No Big-M** — defect applied directly for each active region",
        "control_continuity": "**Direct equality** or disabled",
    },
}

_DEFAULT_BIG_M_STATUS: Dict[str, str] = {
    "region_activation": "N/A",
    "region_containment": "N/A",
    "interface_containment": "N/A",
    "interface_equality": "N/A",
    "dms_defect": "N/A",
    "control_continuity": "N/A",
}

_NONCONVEXITY_NOTE = """\
The unicycle dynamics `ẋ = [v cos θ, v sin θ, ω]` are **nonlinear** in `θ`.
The RK4 / CasADi endpoint map `F_endpoint(s⁻, w, Δ)` is nonlinear due to
`cos(θ)` and `sin(θ)`. With variable dwell time `Δ` as a decision variable,
the defect constraint `s⁺ − F_endpoint(s⁻, w, Δ) = 0` introduces additional
time-scaling nonconvexity.

**IPOPT finds a locally feasible NLP solution. No global optimality certificate
is provided.** Reporting the result as a "globally optimal" or "MISOCP" solution
would be incorrect.

Perspective / homogenized constraints (`A q̃ ≤ b p`) are appropriate for
convex polytope containment in GCS, but **cannot** be applied directly to the
nonlinear DMS defect equality. The defect is not a convex set containment
condition.

For the fixed-path NLP mode, no Big-M is needed in the geometry or coupling
layers because the discrete path is fixed before the NLP is assembled and only
active regions appear as decision variables. The DMS defect itself remains
nonconvex regardless of the path-selection strategy."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_docs_dir(repo_root: Optional[Path | str] = None) -> Path:
    """Create and return the ``docs/`` directory relative to the repo root."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    docs = Path(repo_root) / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    return docs


def _fmt(value: Any, fmt: str = ".4f", fallback: str = "N/A") -> str:
    """Format a numeric value; return *fallback* for NaN, inf, or None."""
    if value is None:
        return fallback
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return fallback
        return format(f, fmt)
    except (TypeError, ValueError):
        return fallback


# ---------------------------------------------------------------------------
# Markdown formatters
# ---------------------------------------------------------------------------

def format_optimization_result_markdown(exp_result: Any, prepared: Any) -> str:
    """
    Build a self-contained Markdown report string for one scenario run.

    Parameters
    ----------
    exp_result : ExperimentResult
    prepared   : PreparedScenario
    """
    opt = exp_result.optimization_result
    scenario_name = exp_result.scenario_name

    mode = getattr(opt, "formulation_mode", "") or "Unknown"
    big_m = _BIG_M_STATUS_BY_MODE.get(mode, _DEFAULT_BIG_M_STATUS)

    global_claim = getattr(
        opt,
        "global_optimality_claim",
        "No global certificate; local NLP solution only.",
    )

    total_duration = getattr(opt, "total_duration", float("nan"))
    if math.isnan(float(total_duration)) and opt.time_durations:
        total_duration = float(sum(opt.time_durations.values()))

    min_safety_margin = getattr(opt, "min_safety_margin", float("nan"))
    path_length_metric = getattr(opt, "path_length_metric", float("nan"))

    if "LEGACY" in mode:
        solver_label = "IPOPT (via CasADi) — continuous relaxation of integrated MIOCP"
        problem_class = (
            "Continuous relaxation of MIOCP (MINLP after DMS transcription). "
            "**Not** a MICP/MISOCP. Nonconvex due to unicycle dynamics."
        )
    elif "TWO_STAGE" in mode:
        solver_label = "IPOPT (via CasADi) — fixed-path nonlinear DMS NLP"
        problem_class = (
            "Continuous nonconvex NLP (fixed path, nonlinear DMS). "
            "Path selected by centroid-distance graph heuristic "
            "(not a certified GCS relaxation)."
        )
    else:
        solver_label = "IPOPT (via CasADi)"
        problem_class = "Nonlinear program (NLP). Nonconvex due to unicycle dynamics."

    path_str = " → ".join(opt.path) if opt.path else "N/A"
    path_regions_str = str(opt.path_regions) if opt.path_regions else "N/A"
    n_path_nodes = len(opt.path_regions)

    lines = [
        f"# GCS-DMS Optimization Report: `{scenario_name}`",
        "",
        "## 1. Scenario",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Preset | `{prepared.preset.key}` |",
        f"| Description | {prepared.resolved.description or 'N/A'} |",
        f"| Number of regions | {exp_result.n_regions} |",
        f"| Number of edges | {exp_result.n_edges} |",
        f"| Candidate paths | {exp_result.n_paths} |",
        f"| Start state | `{prepared.resolved.start_state.tolist()}` |",
        f"| Goal state | `{prepared.resolved.goal_state.tolist()}` |",
        "",
        "## 2. Solver Mode",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Mode | `{mode}` |",
        f"| Solver | {solver_label} |",
        f"| Problem class | {problem_class} |",
        f"| Global optimality claim | {global_claim} |",
        "",
        "## 3. Selected Path",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Path nodes | `{path_str}` |",
        f"| Path regions | `{path_regions_str}` |",
        f"| Path length (regions) | {n_path_nodes} |",
        "",
        "## 4. Objective and Timing",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total cost | {_fmt(opt.total_cost)} |",
        f"| Total duration | {_fmt(total_duration)} s |",
        f"| Setup time | {_fmt(exp_result.setup_time)} s |",
        f"| Solve time | {_fmt(opt.solve_time)} s |",
        f"| Paths evaluated | {opt.n_paths_evaluated} |",
        "",
        "## 5. Feasibility Diagnostics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Defect norm | {_fmt(opt.defect_norm, '.2e')} |",
        f"| Max connection gap | {_fmt(opt.max_connection_gap, '.2e')} |",
        f"| Max control jump | {_fmt(opt.max_control_jump, '.2e')} |",
        f"| Constraint violation | {_fmt(opt.constraint_violation, '.2e')} |",
        f"| Max integrality gap | {_fmt(opt.max_integrality_gap, '.2e')} |",
        f"| Min safety margin | {_fmt(min_safety_margin, '.2e')} |",
        f"| Path length metric | {_fmt(path_length_metric)} |",
        "",
        "## 6. Big-M / Perspective Status",
        "",
        "| Constraint type | Status |",
        "|-----------------|--------|",
        f"| Region activation | {big_m['region_activation']} |",
        f"| Region containment | {big_m['region_containment']} |",
        f"| Edge/interface containment | {big_m['interface_containment']} |",
        f"| Interface equality | {big_m['interface_equality']} |",
        f"| DMS defect | {big_m['dms_defect']} |",
        f"| Control continuity | {big_m['control_continuity']} |",
        "",
        "## 7. Nonconvexity Notes",
        "",
        _NONCONVEXITY_NOTE,
        "",
        "## 8. Generated Artifacts",
        "",
    ]

    if opt.success:
        prefix = f"results/{scenario_name}"
        lines += [
            f"- **JSON summary:** `{prefix}_summary.json`",
            f"- **PNG result:** `{prefix}_result.png`",
            "- **GIF animation:** `" + prefix + "_animation.gif` *(if generated)*",
        ]
    else:
        lines += [
            "- **Status:** Optimization failed",
            f"- **Solver status:** `{opt.solver_status}`",
            "- JSON / PNG / GIF may not have been generated.",
        ]

    lines.append("")
    return "\n".join(lines)


def write_markdown_report(
    exp_result: Any,
    prepared: Any,
    docs_dir: Path | str,
    filename: Optional[str] = None,
) -> Path:
    """
    Write a per-scenario Markdown report to ``docs/<scenario_name>_report.md``.

    Creates *docs_dir* if it does not exist. Does not raise on failed
    optimization — the report documents the failure instead.
    """
    docs_path = Path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{exp_result.scenario_name}_report.md"

    report_path = docs_path / filename
    try:
        content = format_optimization_result_markdown(exp_result, prepared)
    except Exception as exc:  # noqa: BLE001
        content = (
            f"# GCS-DMS Report: `{exp_result.scenario_name}`\n\n"
            f"Report generation failed: {exc}\n"
        )

    report_path.write_text(content, encoding="utf-8")
    print(f"Saved Markdown report: {report_path}")
    return report_path


def write_benchmark_markdown(
    results: Dict,
    docs_dir: Path | str,
    filename: str = "benchmark_summary.md",
) -> Path:
    """
    Write an aggregate benchmark summary to ``docs/benchmark_summary.md``.

    Parameters
    ----------
    results : Dict[str, Tuple[ExperimentResult, PreparedScenario]]
    """
    docs_path = Path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)

    lines = [
        "# GCS-DMS Benchmark Summary",
        "",
        "> **Important:** IPOPT provides a **locally optimal** NLP solution. "
        "No global optimality certificate is provided for any scenario.",
        "",
        "| Scenario | Status | Regions | Edges | Paths | Cost | Solve Time (s) | Mode |",
        "|----------|--------|---------|-------|-------|------|----------------|------|",
    ]

    for name, (exp_result, _prepared) in results.items():
        opt = exp_result.optimization_result
        status = "OK" if opt.success else "FAIL"
        cost = _fmt(opt.total_cost) if opt.success else "N/A"
        solve_t = _fmt(opt.solve_time, ".3f")
        mode = getattr(opt, "formulation_mode", "N/A") or "N/A"
        mode_short = mode if len(mode) <= 32 else mode[:30] + "..."
        lines.append(
            f"| [{name}]({name}_report.md) | {status} "
            f"| {exp_result.n_regions} | {exp_result.n_edges} "
            f"| {exp_result.n_paths} | {cost} | {solve_t} | `{mode_short}` |"
        )

    lines += [
        "",
        "## Individual Reports",
        "",
    ]
    for name in results:
        lines.append(f"- [{name} report]({name}_report.md)")

    lines += [
        "",
        "## Formulation Notes",
        "",
        "### Legacy integrated solver",
        "",
        "The legacy integrated solver is a **continuous relaxation of a nonconvex MINLP** "
        "solved by IPOPT. It is **not** a certified globally optimal MICP/MISOCP solver. "
        "Big-M constraints are used for region activation, region containment, "
        "edge/interface containment, interface equality, and optionally control continuity.",
        "",
        "### Two-stage solver",
        "",
        "The two-stage solver selects candidate paths using a centroid-distance graph "
        "heuristic (not a certified GCS convex relaxation) and then solves a "
        "**fixed-path nonlinear DMS NLP** with IPOPT. No Big-M activation, containment, "
        "or interface equality constraints are needed because the path is fixed before "
        "the NLP is assembled.",
        "",
        "### Why the DMS problem remains nonconvex",
        "",
        _NONCONVEXITY_NOTE,
        "",
    ]

    benchmark_path = docs_path / filename
    benchmark_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved benchmark Markdown: {benchmark_path}")
    return benchmark_path
