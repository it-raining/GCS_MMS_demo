# GCS-MMS Demo

This demo solves motion planning on a graph of safe convex regions using
**Centroid-Refine-DMS (CRD)**, the primary solver (`optimizer.solver_mode:
"centroid_refine_dms"` in `config.yaml`). Legacy single-phase
integrated-relaxation (`IntegratedNLPSolver`) and two-stage solver modes have
been removed; CRD is the only supported pipeline.

CRD runs five stages per candidate path:

- **A. Graph search** — Yen's k-shortest paths over the region graph, ranked
  by a heuristic that rewards interface clearance and penalizes heading change.
- **B. Interface QP** — places interface waypoints with a guaranteed
  `delta_safe + delta_extra` clearance from region boundaries.
- **C. Centroid warm-start** — builds a feasible initial guess for the
  multiple-shooting NLP from the QP waypoints.
- **D-E. Barrier DMS** — solves a log-barrier multiple-shooting NLP with a
  decreasing `barrier_levels` continuation schedule, certifying the safety
  margin of the final trajectory.

Math reference: `docs/superpowers/specs/2026-06-07-centroid-refine-dms-design.md`.

## Quick Start

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd demo
python main_demo.py --scenario default
```

If you only want to inspect the available configs without solving anything:

```bash
python main_demo.py --list-scenarios
python main_demo.py --list-presets
```

## Structure

- `problem_data.py`: geometry presets for each environment/problem
- `config.yaml`: scenario list and parameter overrides
- `app_config.py`: config loading and scenario resolution
- `scenario_builder.py`: builds environment, regions, graph, dynamics, optimizer
- `optimizer.py`: `CentroidRefineDMSSolver` (Stage A graph search + Stage E orchestration)
- `geometric_refiner.py`: Stage B Interface QP
- `barrier_dms.py`: Stage D-E barrier-method multiple shooting + safety certification
- `warmstart.py`: Stage C centroid warm-start construction
- `shooting_animation.py`: fixed-path NLP iteration recorder + convergence GIF
- `reporting.py`: Markdown report generation for per-scenario/benchmark results
- `experiments.py`: scenario runner and result saving
- `main_demo.py`: CLI entry point

## Running Tests

Tests use `pytest` and must run through the project's `.venv` interpreter
(from the repository root):

```bash
.venv/bin/python -m pytest tests/ -v
```

Quick subset while iterating on the safety-certification formula:

```bash
.venv/bin/python -m pytest tests/test_barrier_dms.py tests/test_centroid_refine_dms.py -v
```

The full default-scenario regression test is skipped by default because it
runs a real solve (~tens of seconds). Opt in explicitly:

```bash
RUN_DEMO_REGRESSION=1 .venv/bin/python -m pytest tests/test_regression.py -v
```

Standalone debug/sanity scripts (not collected by pytest, run from `demo/`):

```bash
cd demo
../.venv/bin/python test_crd_default.py   # end-to-end CRD run with verbose per-stage logging to /tmp/crd_debug.log
../.venv/bin/python ctcs_self_test.py     # RK4 accumulated path-constraint violation sanity checks
```

## Run the Model (`main_demo.py`)

Run from `demo/`:

```bash
cd demo
python main_demo.py \
  --config config.yaml \
  --scenario default \
  --quiet
```

Other entry points, runnable as-is (copy/paste):

```bash
python main_demo.py --list-scenarios
python main_demo.py --list-presets
python main_demo.py --formulation
python main_demo.py --visualize-only --scenario default
python main_demo.py --shooting-animation --scenario default
python main_demo.py --quick-test --scenario default
python main_demo.py --maze-benchmark \
  --maze-count 5 --maze-size 25 --maze-knock-downs 25 \
  --maze-seed 4 --maze-wall-thickness 0.08
```

### Argument reference

| Argument | Default | Effect on result |
| --- | --- | --- |
| `--config PATH` | `config.yaml` | Selects the YAML file all other settings are resolved from. |
| `--scenario NAME` | first scenario in config | Selects which `scenarios.<name>` block (geometry preset, start/goal, `overrides`) is solved. Omitting it with no other flag runs **all** scenarios. |
| `--quick-test` | off | Runs one scenario through `ExperimentRunner` and returns the raw `OptimizationResult` — useful for a fast pipeline sanity check without saving plots. |
| `--visualize-only` | off | Only renders the safe-region decomposition and region graph `G=(V,E)`; does not invoke the solver. |
| `--shooting-animation` | off | Solves the scenario, extracts the path, replays the fixed-path NLP with an iteration recorder, and writes a convergence GIF. Slower than a normal solve because it re-solves the fixed path with recording enabled. |
| `--formulation` | off | Prints the MIOCP/CRD formulation summary and exits; no solve. |
| `--list-presets` | off | Lists geometry presets defined in `problem_data.py`; no solve. |
| `--list-scenarios` | off | Lists scenario keys defined in `config.yaml`; no solve. |
| `--maze-benchmark` | off | Generates `--maze-count` random mazes and benchmarks CRD on each instead of running a configured scenario. |
| `--maze-count N` | `5` | Number of random maze instances to generate and solve. More instances give a more reliable success-rate/timing estimate but take longer overall. |
| `--maze-size N` | `25` | Maze width/height in cells. Larger mazes produce more ACD regions and longer paths, increasing NLP size and solve time, and stress-testing Stage A path search. |
| `--maze-knock-downs N` | `25` | Extra random interior wall removals after maze generation, widening some corridors and adding alternate routes. Higher values make the maze more open (fewer narrow-interface rejections in Stage B). |
| `--maze-seed N` | `4` | Starting RNG seed; each of the `--maze-count` instances increments from this seed. Fixing it makes a benchmark run reproducible. |
| `--maze-wall-thickness F` | `0.08` | Thickness of generated interior maze walls, in workspace units. Thicker walls shrink corridor width, making the `delta_safe`/`delta_extra` narrow-interface check (see below) reject more candidate paths. |
| `--maze-debug-geometry` | off | Saves workspace + hole geometry plots before ACD, for inspecting maze generation independent of the solve. |
| `--quiet` | off | Suppresses per-stage progress logging; does not change solver behavior or results. |

## Configure Parameters (`config.yaml`)

```yaml
region_decomposition:
  overlap_width: 0.5      # must satisfy overlap_width >= delta_safe + delta_extra

dynamics:
  model: "unicycle"
  delta_min: 0.1           # minimum dwell time per region [s]
  delta_max: 3.0           # bounds h_max = delta_max / n_integration_steps

control:
  n_segments: 2             # piecewise-constant control segments per region

shooting:
  n_integration_steps: 10   # RK4 steps per region; also sets h_max above

cost:
  a: 1.0          # time penalty w_T
  w_L: 1.0        # path-length penalty
  w_E: 1.0        # control-effort penalty w_U
  w_u_smooth: 0.5 # inter-segment control smoothness penalty w_S

optimizer:
  solver_mode: "centroid_refine_dms"
  max_paths: 1500            # upper bound on path enumeration in summaries

centroid_refine_dms:
  gamma_w: 1.0                      # Stage A: interface-clearance bonus
  gamma_h: 0.64                     # Stage A: heading-change penalty
  delta_safe: 0.02                  # Stage B/D-E: required safety clearance
  delta_extra: 0.01                 # hard floor added to the NLP safety slack
  epsilon_certificate_buffer: 0.0   # extra certified-margin headroom (NLP floor + pass/fail threshold only)
  alpha_s: 0.0                      # Stage B: QP smoothness weight
  v_nom_fraction: 0.5                # Stage C: warm-start nominal speed fraction of v_max
  barrier_levels: [1.0, 0.5, 0.1, 0.01]  # Stage D-E continuation schedule, large -> small mu
  epsilon_final: 1.0e-6              # final IPOPT KKT tolerance
  epsilon_gap: 0.05                  # near-optimality gap for "anytime" mode
  mode: "anytime"                    # "first_feasible" | "anytime"
```

Parameter effects worth knowing before tuning:

- `n_integration_steps` sets `h_max = delta_max / n_integration_steps`, which
  drives the per-segment Lipschitz/RK4 safety margin baked into the barrier
  NLP. Fewer steps means a larger required margin and tighter narrow-corridor
  rejections in Stage B; more steps shrinks the margin but grows the NLP.
- `delta_safe` and `delta_extra` gate **both** the Interface QP anchor
  placement (Stage B) and the live per-segment safety constraint inside the
  barrier NLP (Stage D-E). Raising either rejects more narrow interfaces but
  also raises `certified_safety_margin` on paths that do solve.
- `epsilon_certificate_buffer` only raises the NLP's safety-slack floor and
  the `certified_safety_margin` pass/fail threshold — it does not change
  Interface QP or path-search behavior. Use it to add headroom when a
  scenario reports `NOT_CERTIFIED` with a thin margin, without re-tuning
  `delta_safe`/`delta_extra` everywhere else.
- `barrier_levels` is the mu continuation schedule for the log-barrier NLP
  (Stage D-E): more/finer levels improve convergence robustness at the cost
  of more NLP solves per candidate path.
- `mode: "first_feasible"` stops at the first candidate path that solves and
  certifies; `mode: "anytime"` keeps searching until `epsilon_gap` or
  `time_limit_s` is hit, trading solve time for a better-cost guarantee.

Scenario-specific changes go under `scenarios.<name>.overrides`, deep-merged
onto the global defaults above:

```yaml
scenarios:
  default:
    problem_preset: "default"
    overrides:
      control:
        n_segments: 4
      shooting:
        n_integration_steps: 44
      centroid_refine_dms:
        delta_safe: 0.02
        barrier_levels: [0.05]
        mode: "first_feasible"
        epsilon_certificate_buffer: 0.001
```

You can also override `start_state`, `goal_state`, and `problem_preset` per
scenario. CRD-relevant override keys:

- `overrides.centroid_refine_dms.*` — solver hyperparameters
- `overrides.control.n_segments` — piecewise-constant segments
- `overrides.shooting.n_integration_steps` — RK4 steps (affects Lipschitz gap)
- `overrides.dynamics.delta_max` — max dwell time per region
- `overrides.cost.*` — cost weights
- `overrides.region_decomposition.overlap_width`

## Maze Benchmark Notes

Dense mazes have small regions and narrow interfaces, so the global
`delta_safe`/`delta_max` defaults can reject almost every candidate path (see
the `maze` scenario's `overrides` comments in `config.yaml` for the worked
example). When tuning a maze scenario:

- Shrink `dynamics.delta_max` (keep `n_integration_steps` fixed) to bring
  `h_max` down without growing the NLP.
- Raise `centroid_refine_dms.gamma_w` if Stage A keeps ranking narrow,
  cheap-looking corridors above a wider but longer route.
- Use `--maze-debug-geometry` to inspect generated geometry independently of
  the solve.

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

Results are saved under `results/` (or `output.results_dir` in
`config.yaml`):

- `{scenario}_result.png`
- `{scenario}_animation.gif`
- `{scenario}_summary.json`
- `benchmark_summary.json`

Markdown reports (`reporting.py`) are written under `output.docs_dir`
(default `docs/`) when `output.save_md` is true:

- `demo/docs/default_report.md`
- `demo/docs/benchmark_summary.md`
