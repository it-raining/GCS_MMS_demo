# GCS-MMS Demo

Motion planning demo using Graph of Convex Sets (GCS) with Multiple-Shooting (MMS).
Supports three solver modes on top of a unicycle / double-integrator dynamics model.

## Solvers

| Mode | Description |
|------|-------------|
| `integrated_relaxation_legacy` | One-phase Big-M MIOCP relaxation solved with IPOPT (default) |
| `two_stage` | Heuristic centroid-cost path selection + fixed-path NLP polish |
| `centroid_refine_dms` | K-shortest Chebyshev-center search → interface QP → barrier-continuation DMS NLP |

---

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `networkx`, `shapely`, `casadi`.

---

## Quick Start

```bash
cd demo

# Run default scenario
python main_demo.py

# Run a named scenario
python main_demo.py --scenario default
python main_demo.py --scenario maze
```

---

## All CLI Options

```
python main_demo.py [OPTIONS]
```

### Core options

| Option | Default | Description |
|--------|---------|-------------|
| `--scenario NAME` | default | Run one named scenario from `config.yaml` |
| `--config PATH` | `config.yaml` | Path to YAML config file |
| `--quiet` | off | Reduce output verbosity |
| `--save-md` | off | Save Markdown report to `docs/` |

### Run modes

| Option | Description |
|--------|-------------|
| *(no flag)* | Run the default scenario end-to-end |
| `--scenario NAME` | Run one specific scenario |
| `--quick-test` | One fast integrated-solver sanity check |
| `--visualize-only` | Plot environment layout only — no solve |
| `--shooting-animation` | Create fixed-path shooting convergence GIF |
| `--formulation` | Print formulation summary to stdout |

### Inspection

| Option | Description |
|--------|-------------|
| `--list-scenarios` | List all named scenarios in the config |
| `--list-presets` | List all geometry presets in `problem_data.py` |

### Maze benchmark

| Option | Default | Description |
|--------|---------|-------------|
| `--maze-benchmark` | off | Benchmark randomly generated maze instances |
| `--maze-count N` | 5 | Number of random mazes |
| `--maze-size N` | 25 | Maze width/height in cells |
| `--maze-knock-downs N` | 25 | Extra wall removals after generation |
| `--maze-seed N` | 4 | Starting random seed |
| `--maze-wall-thickness F` | 0.08 | Interior wall thickness |
| `--maze-debug-geometry` | off | Save workspace + hole geometry plots before ACD |

---

## Scenarios

Defined in `demo/config.yaml` under `scenarios:`.

| Key | Name | Description |
|-----|------|-------------|
| `default` | Default Environment | Hand-crafted convex safe regions, integrated relaxation solver |
| `crd_demo` | CRD Demo — Overlapping Corridor | 5 overlapping rectangular regions; lightweight first-feasible CRD demo |
| `default_crd` | Default — Centroid-Refine-DMS | Default 5×5 workspace using CentroidRefineDMS |
| `maze_crd` | Maze — Centroid-Refine-DMS | Maze preset with CentroidRefineDMS solver |
| `maze_15_crd` | Maze 15×15 — CRD Zigzag Corridor | 8-region zigzag in a 15×15 workspace |
| `maze` | Maze Environment | ACD-decomposed maze map, integrated relaxation solver |

Run a specific scenario:

```bash
python main_demo.py --scenario default
python main_demo.py --scenario crd_demo
python main_demo.py --scenario default_crd
python main_demo.py --scenario maze --quiet
```

---

## Configure Parameters

Edit `demo/config.yaml`. Key sections:

```yaml
optimizer:
  solver_mode: "integrated_relaxation_legacy"
  # Options:
  #   "integrated_relaxation_legacy"  — Big-M MIOCP relaxation (IPOPT)
  #   "two_stage"                     — centroid path selection + NLP polish
  #   "centroid_refine_dms"           — Chebyshev-center + interface QP + barrier DMS

shooting:
  n_integration_steps: 20
  n_mesh_points: 10
  safety_margin: 0.01
  cbf_alpha: 0.0          # set > 0 to enable CBF safety layer

cost:
  a: 1.0
  w_L: 1.0
  w_E: 1.0
  w_u_smooth: 0.5
  w_theta_smooth: 0.0
```

### Centroid-Refine-DMS options

Add a `centroid_refine_dms:` block to override defaults:

```yaml
optimizer:
  solver_mode: "centroid_refine_dms"

centroid_refine_dms:
  # Graph search
  gamma_w: 1.0          # interface clearance bonus weight
  gamma_h: 0.64         # turn penalty weight (≈ v_nom/omega_max)

  # Geometric QP
  alpha_s: 0.0          # smoothness weight (0 = pure shortest polyline)
  delta_extra: 0.01     # strict-interior margin for QP

  # Warm-start
  v_nom_fraction: 0.5   # v_nom = v_nom_fraction * (v_max + |v_min|) / 2

  # Barrier schedule (Fiacco-McCormick)
  alpha_mu: 0.1         # mu_0 = alpha_mu * f0 / B0
  tau: 0.1              # barrier reduction factor per level
  mu_min: 1.0e-5        # stop threshold

  # NLP settings
  n_int: 20             # RK4 integration steps per segment
  delta_min: 0.01
  delta_max: 10.0

  # Cost weights
  w_T: 1.0              # time
  w_L: 1.0              # path length
  w_U: 1.0              # control effort
  w_S: 0.2              # smoothness

  # Stopping criteria
  epsilon_final: 1.0e-6
  epsilon_gap: 0.05
  time_limit_s: 60.0
  mode: "first_feasible"  # or "optimality_gap"

  # Ablation flags (all true = full method; use_log_barrier defaults false)
  use_centroid_cost: true
  use_interface_qp: true
  use_log_barrier: false
  use_barrier_continuation: true
  use_inexact_tolerance: true
```

### Per-scenario overrides

```yaml
scenarios:
  my_case:
    problem_preset: "default"
    active_regions: "all"
    overrides:
      shooting:
        n_mesh_points: 15
      optimizer:
        ipopt:
          max_iter: 5000
```

---

## Add a New Environment

1. Add a preset entry in `demo/problem_data.py`:
   - `workspace_vertices`
   - `obstacle_vertices`
   - `region_vertices`
   - `default_start_state` / `default_goal_state`
   - `region_buffer`

2. Reference it in `config.yaml`:

```yaml
problem:
  default_preset: "my_env"

scenarios:
  my_case:
    problem_preset: "my_env"
    active_regions: "all"
```

---

## Project Structure

```
demo/
├── main_demo.py          # CLI entry point
├── config.yaml           # All scenario/solver parameters
├── app_config.py         # Config loading and resolution
├── scenario_builder.py   # Assembles environment + graph + solver
├── problem_data.py       # Geometry presets per environment
├── dynamics.py           # UnicycleModel, DoubleIntegratorDynamics
├── environment.py        # Workspace + obstacle geometry
├── convex_regions.py     # Convex region abstraction
├── graph_types.py        # Graph node/edge type definitions
├── optimizer.py          # All solver classes + factory
├── graph_builder.py      # Region graph, Chebyshev center, k-shortest
├── geometric_refiner.py  # Interface QP (Centroid-Refine-DMS)
├── warmstart.py          # IVP warm-start generator
├── barrier_dms.py        # Barrier-continuation DMS NLP
├── constraint_layers.py  # Barrier log terms, safety gap
├── shooting.py           # Fixed-path multiple-shooting NLP
├── shooting_animation.py # Shooting convergence animation
├── maze_benchmark.py     # Random maze benchmark runner
├── experiments.py        # Scenario runner, result saving
├── reporting.py          # Markdown report generation
├── visualization.py      # Plotting helpers
└── results/              # Output directory

tests/
├── test_dynamics.py
├── test_graph_builder.py
├── test_geometric_refiner.py
├── test_warmstart.py
├── test_barrier_dms.py
├── test_centroid_refine_dms.py
├── test_constraint_layers.py
├── test_convex_regions.py
├── test_optimizer.py
├── test_edge_cases.py
├── test_regression.py
└── test_shooting_animation.py
```

---

## Run Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: 59 passed, 1 skipped.

---

## Outputs

Results are saved under `demo/results/`:

| File | Description |
|------|-------------|
| `{scenario}_result.png` | Trajectory visualization |
| `{scenario}_animation.gif` | Shooting convergence animation |
| `{scenario}_summary.json` | Numeric results |
| `benchmark_summary.json` | Maze benchmark aggregate |
