# GCS-MMS Demo

This demo solves motion planning as a single integrated one-phase MIOCP relaxation:

- graph flow variables `y_uv`, `p_v`
- multiple-shooting trajectory variables `s_v^-`, `s_v^+`, `w_v`, `Delta_v`
- interface variables `z_uv`
- local cost epigraphs `rho_v`

The runtime solver is:

- `IPOPT` on the integrated continuous relaxation
- optional fixed-path NLP polish when the extracted relaxed path is fractional or discontinuous

## Quick Start

From the repository root:

```bash
python3 -m venv demo/.venv
source demo/.venv/bin/activate
pip install -r requirements.txt
python3 demo/main_demo.py --scenario simple
```

If you only want to inspect the available configs without activating the demo environment:

```bash
python3 demo/main_demo.py --list-scenarios
python3 demo/main_demo.py --list-presets
```

## Structure

The code is organized so geometry, scenarios, and solver assembly are separate:

- `problem_data.py`: geometry presets for each environment/problem
- `config.yaml`: scenario list and parameter overrides
- `app_config.py`: config loading and scenario resolution
- `scenario_builder.py`: builds environment, regions, graph, dynamics, optimizer
- `optimizer.py`: integrated solver and fixed-path polish
- `experiments.py`: scenario runner and result saving
- `main_demo.py`: CLI entry point

## Run

```bash
cd demo
python main_demo.py
python main_demo.py --scenario simple
python main_demo.py --list-scenarios
python main_demo.py --list-presets
python main_demo.py --visualize-only --scenario medium
python main_demo.py --quick-test
```

## Configure Parameters

Global defaults live in `config.yaml`:

```yaml
cost:
  a: 1.0
  w_L: 1.0
  w_E: 1.0

shooting:
  n_integration_steps: 20
  n_mesh_points: 10
  safety_margin: 0.02
  safety_mode: "both"
  ctcs_tolerance: 1.0e-6
  dense_check_points: 1000
  dense_check_tolerance: 1.0e-4
  fail_on_dense_violation: true

optimizer:
  ipopt:
    max_iter: 3000
    tol: 1.0e-6
    print_level: 0
```

`shooting.safety_mode` controls local region containment:

- `mesh`: enforce `A*pi(x(tau_k)) <= b - safety_margin` at finite interior mesh points.
- `ctcs`: enforce the CTCS safety certificate, an RK4 approximation of the accumulated nonnegative path-constraint violation.
- `both`: enforce both transcriptions.

The CTCS certificate uses `eta_end ~= integral sum_i [A_i*pi(x(tau)) - b_i + safety_margin]_+^2 dtau` and constrains `eta_end <= ctcs_tolerance`. This is not an exact continuous-time proof under finite RK4, so solved trajectories are also densely post-checked and the summary reports `max_ctcs_integral` and `max_dense_region_violation`.

## Adjustable CTCS Maze Run

For CTCS experiments, start with a small generated maze and scale up gradually. This script writes a temporary config and runs one maze benchmark case; edit the shell variables at the top to trade safety resolution against solve time.

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/demo"

# Maze size knobs. The CTCS transcription can get expensive quickly on mazes
# with many ACD regions, so begin small before trying larger values.
MAZE_SIZE="${MAZE_SIZE:-3}"
MAZE_COUNT="${MAZE_COUNT:-1}"
MAZE_KNOCK_DOWNS="${MAZE_KNOCK_DOWNS:-0}"
MAZE_SEED="${MAZE_SEED:-4}"
MAZE_WALL_THICKNESS="${MAZE_WALL_THICKNESS:-0.08}"

# Safety transcription:
#   mesh: old finite mesh safety constraints
#   ctcs: RK4 approximation of accumulated path-constraint violation
#   both: mesh constraints plus CTCS certificate
SAFETY_MODE="${SAFETY_MODE:-both}"
N_INTEGRATION_STEPS="${N_INTEGRATION_STEPS:-5}"
N_MESH_POINTS="${N_MESH_POINTS:-3}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.02}"
CTCS_TOLERANCE="${CTCS_TOLERANCE:-1.0e-5}"
CTCS_ETA_BIG_M="${CTCS_ETA_BIG_M:-100.0}"
DENSE_CHECK_POINTS="${DENSE_CHECK_POINTS:-100}"
DENSE_CHECK_TOLERANCE="${DENSE_CHECK_TOLERANCE:-1.0e-4}"
FAIL_ON_DENSE_VIOLATION="${FAIL_ON_DENSE_VIOLATION:-false}"

# Dynamics/control and solver budget.
N_CONTROL_SEGMENTS="${N_CONTROL_SEGMENTS:-2}"
DELTA_MIN="${DELTA_MIN:-0.01}"
DELTA_MAX="${DELTA_MAX:-10.0}"
IPOPT_MAX_ITER="${IPOPT_MAX_ITER:-500}"
IPOPT_TOL="${IPOPT_TOL:-1.0e-6}"
IPOPT_PRINT_LEVEL="${IPOPT_PRINT_LEVEL:-0}"

# Artifact switches. Disable images/GIFs while tuning solver settings.
SAVE_PNG="${SAVE_PNG:-false}"
SAVE_GIF="${SAVE_GIF:-false}"
SAVE_JSON="${SAVE_JSON:-true}"

# Keep the generated config inside demo/ so relative results_dir resolves to
# demo/results instead of /tmp/results.
CONFIG_PATH="${CONFIG_PATH:-ctcs_maze.local.yaml}"

cat > "${CONFIG_PATH}" <<YAML
problem:
  default_preset: "maze"

start_state:
  position: [0.2, 0.2]
  heading: 0.785

goal_state:
  position: [4.8, 4.8]
  heading: 0.785

dynamics:
  model: "unicycle"
  v_min: -2.0
  v_max: 2.0
  omega_min: -3.14159
  omega_max: 3.14159
  delta_min: ${DELTA_MIN}
  delta_max: ${DELTA_MAX}

control:
  parameterization: "piecewise_constant"
  n_segments: ${N_CONTROL_SEGMENTS}

shooting:
  n_integration_steps: ${N_INTEGRATION_STEPS}
  n_mesh_points: ${N_MESH_POINTS}
  safety_margin: ${SAFETY_MARGIN}
  safety_mode: "${SAFETY_MODE}"
  ctcs_tolerance: ${CTCS_TOLERANCE}
  ctcs_penalty: "squared_hinge"
  ctcs_use_rk4_stages: true
  ctcs_eta_big_m: ${CTCS_ETA_BIG_M}
  dense_check_points: ${DENSE_CHECK_POINTS}
  dense_check_tolerance: ${DENSE_CHECK_TOLERANCE}
  fail_on_dense_violation: ${FAIL_ON_DENSE_VIOLATION}

cost:
  a: 1.0
  w_L: 1.0
  w_E: 1.0
  w_u_smooth: 0.2

graph:
  max_paths: 1500

optimizer:
  enforce_control_continuity: true
  path_polish_candidates: 3
  ipopt:
    max_iter: ${IPOPT_MAX_ITER}
    tol: ${IPOPT_TOL}
    print_level: ${IPOPT_PRINT_LEVEL}
  big_M:
    position: 20.0
    interface: 20.0
    time: 100.0

visualization:
  figsize: [10, 10]
  show_regions: true
  show_graph: false
  show_mesh_points: true
  animation:
    fps: 30
    duration: 5.0

scenarios:
  maze:
    name: "Generated Maze"
    description: "Runtime-generated maze benchmark"
    active_regions: "all"
    problem_preset: "maze"

output:
  results_dir: "results"
  save_png: ${SAVE_PNG}
  save_gif: ${SAVE_GIF}
  save_json: ${SAVE_JSON}
  verbosity: 1
YAML

../.venv/bin/python main_demo.py \
  --config "${CONFIG_PATH}" \
  --maze-benchmark \
  --maze-count "${MAZE_COUNT}" \
  --maze-size "${MAZE_SIZE}" \
  --maze-knock-downs "${MAZE_KNOCK_DOWNS}" \
  --maze-seed "${MAZE_SEED}" \
  --maze-wall-thickness "${MAZE_WALL_THICKNESS}" \
  --quiet
```

Examples:

```bash
# Fast CTCS sanity check on the smallest generated maze.
bash run_ctcs_maze.sh

# Baseline old behavior.
SAFETY_MODE=mesh bash run_ctcs_maze.sh

# Increase CTCS resolution after the small case runs.
MAZE_SIZE=5 MAZE_KNOCK_DOWNS=3 N_INTEGRATION_STEPS=10 DENSE_CHECK_POINTS=300 bash run_ctcs_maze.sh
```

Speed tuning notes:

- `dense_check_points` only changes post-solve verification cost. It does not reduce the IPOPT NLP size. Use `10` only for smoke tests; use `100-300` while tuning; use `500-1000` for final safety reporting.
- `n_integration_steps` changes the RK4 dynamics and CTCS NLP size. Try `5` for fast debugging, `10` for medium runs, and `20+` for final runs.
- `n_mesh_points` changes mesh safety constraints when `safety_mode` is `mesh` or `both`. Try `3-5` for speed and `10+` when checking final trajectories.
- `safety_mode: "mesh"` is fastest, `"ctcs"` is usually heavier, and `"both"` is the most conservative and most expensive.
- `dense_check_tolerance` should usually stay near `1.0e-4`. Lowering dense samples makes the check less reliable; it does not justify loosening the tolerance.
- If `dense_check_points` is very small, set `fail_on_dense_violation: false` and treat `max_dense_region_violation` as a quick diagnostic, not a safety decision.

Scenario-specific changes should go under `scenarios.<name>.overrides`:

```yaml
scenarios:
  high_resolution:
    name: "High Resolution"
    description: "All regions with increased mesh points"
    active_regions: "all"
    overrides:
      shooting:
        n_mesh_points: 15
      optimizer:
        ipopt:
          max_iter: 5000
```

You can also override `start_state`, `goal_state`, and `problem_preset` per scenario.

## Add a New Environment

1. Add a new preset entry in `problem_data.py`.
2. Define:
   - `workspace_vertices`
   - `obstacle_vertices`
   - `region_vertices`
   - `default_start_state`
   - `default_goal_state`
   - `region_buffer`
3. Reference that preset from `config.yaml`:

```yaml
problem:
  default_preset: "my_new_problem"
```

or per scenario:

```yaml
scenarios:
  my_case:
    problem_preset: "my_new_problem"
    active_regions: "all"
```

That is enough for the CLI and experiment runner to pick it up.

## Outputs

Results are saved under `results/`:

- `{scenario}_result.png`
- `{scenario}_animation.gif`
- `{scenario}_summary.json`
- `benchmark_summary.json`
