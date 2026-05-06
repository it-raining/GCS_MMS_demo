# Multiple Shooting Convergence Animation

## Goal

Create an animation that visualizes the convergence of the Direct Multiple Shooting method on the 2D workspace map. The animation shows:

1. **Initial state**: Disconnected trajectory segments within each convex region, with visible defect gaps at interfaces
2. **Iteration progression**: Segments adjusting frame by frame as IPOPT iterates
3. **Final state**: All segments converge into one smooth, continuous trajectory satisfying boundary conditions

This is analogous to the reference image (shooting nodes W⁰, W¹, W², W³ converging across intervals P₁–P₄) but rendered directly on the 2D map with convex regions instead of a time-axis plot.

---

## CasADi Support Verification ✅

**Confirmed**: CasADi 3.7.2 (installed in `demo/.venv`) fully supports iteration callbacks.

```
nlpsol_n_out: 6
  nlpsol_out(0) = x    ← decision variables
  nlpsol_out(1) = f    ← objective value
  nlpsol_out(2) = g    ← constraint values
  nlpsol_out(3) = lam_x
  nlpsol_out(4) = lam_g
  nlpsol_out(5) = lam_p
```

A test callback successfully captured **7 iteration snapshots** from a simple NLP, including full `x` vectors and objective values at each IPOPT iteration.

**API pattern**:
```python
class MyCallback(ca.Callback):
    def __init__(self, name, nx, ng, opts={}):
        ca.Callback.__init__(self)
        self.nx, self.ng = nx, ng
        self.construct(name, opts)
    
    def get_n_in(self): return ca.nlpsol_n_out()    # 6
    def get_n_out(self): return 1
    def get_name_in(self, i): return ca.nlpsol_out(i)
    def get_name_out(self, i): return "ret"
    def get_sparsity_in(self, i):
        n = ca.nlpsol_out(i)
        if n == 'f': return ca.Sparsity.scalar()
        elif n in ('x', 'lam_x'): return ca.Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'): return ca.Sparsity.dense(self.ng)
        else: return ca.Sparsity(0, 0)
    def get_sparsity_out(self, i): return ca.Sparsity.scalar()
    
    def eval(self, arg):
        x = np.array(arg[0]).flatten()
        f = float(arg[1])
        # store snapshot ...
        return [0]
```

Pass to solver: `opts['iteration_callback'] = callback_instance`

---

## Current Codebase Context

### Variable layout in `PathNLPSolver._build_and_solve_nlp`

The fixed-path NLP (the one we'll animate) has this variable layout per region:

| Variable | Size | Description |
|----------|------|-------------|
| `s_minus` | `n_x = 3` | Entry state `[px, py, theta]` |
| `w` | `n_w = 4` | Control params (2 segments × 2 controls) |
| `delta` | `1` | Time duration |
| **Total** | **8** | **per region** |

`s_plus` is **not** a decision variable — it's computed as `F_endpoint(s_minus, w, delta)`.

For a path with K regions, the full decision vector `x` has size `K × 8`.

### Existing trajectory reconstruction

[_parse_solution](file:///home/dang-khoa/code/motion-planning/GCS-MMS/demo/optimizer.py#L599-L691) already:
1. Extracts `s_minus, w, delta` from `x_opt` using the stride `vars_per_region = n_x + n_w + 1`
2. Computes `s_plus = F_endpoint(s_minus, w, delta)` via CasADi
3. Generates trajectories via `RK4Integrator.integrate_with_trajectory()`
4. Computes defect norms and connection gaps

### Existing animation infrastructure

[visualization.py](file:///home/dang-khoa/code/motion-planning/GCS-MMS/demo/visualization.py) already has:
- `plot_environment()` — draws regions, obstacles, start/goal
- `plot_trajectory()` — draws trajectory segments, interface points, mesh points
- `create_animation()` — `FuncAnimation` + `PillowWriter` for GIF output
- `setup_plot_style()` — consistent styling

---

## Proposed Changes

### [NEW] [demo/shooting_animation.py](file:///home/dang-khoa/code/motion-planning/GCS-MMS/demo/shooting_animation.py)

New standalone module containing three components:

#### Component 1: `ShootingIterationRecorder` (CasADi Callback)

```python
class ShootingIterationRecorder(ca.Callback):
    """CasADi iteration callback that records IPOPT snapshots."""
```

**Responsibilities**:
- Subclass `ca.Callback` with correct sparsity definitions
- Store `(x_vector, f_value, iteration_index)` at each IPOPT iteration
- Optional: subsample every N iterations to limit memory (e.g., `record_every=1`)
- Optional: cap total stored snapshots (e.g., `max_snapshots=500`)

**Constructor args**: `nx` (total decision variables), `ng` (total constraints), `record_every`, `max_snapshots`

#### Component 2: `parse_shooting_snapshot()`

```python
def parse_shooting_snapshot(
    x_vec: np.ndarray,
    path_regions: List[int],
    n_x: int,
    n_w: int,
    dynamics: DynamicsModel,
    control_param: ControlParameterization,
    F_endpoint: Callable,
    n_integration_steps: int,
) -> Dict:
    """Parse one x-vector snapshot into per-region trajectory data."""
```

**Returns** a dict containing for each region:
- `s_minus`: entry state
- `s_plus`: computed exit state via `F_endpoint`
- `w`: control params
- `delta`: time duration
- `trajectory`: full RK4 trajectory `(N+1, n_x)` array
- `positions`: position-only trajectory `(N+1, 2)` array

Plus global metrics:
- `defect_norms`: per-region `||s_plus_computed - s_minus_next||`
- `connection_gaps`: per-interface position gap
- `max_defect`: scalar
- `max_gap`: scalar
- `cost`: objective value

This reuses the same stride logic as `_parse_solution` but operates on raw numpy arrays without creating `OptimizationResult`.

#### Component 3: `create_shooting_convergence_animation()`

```python
def create_shooting_convergence_animation(
    snapshots: List[Dict],       # parsed snapshots from Component 2
    graph: RegionGraph,
    workspace_bounds: Tuple,
    start_state: np.ndarray,
    goal_state: np.ndarray,
    filename: str,
    obstacles: Optional[List] = None,
    fps: int = 15,
    hold_first_frames: int = 10,
    hold_last_frames: int = 20,
    max_animation_frames: int = 200,
) -> None:
```

**Animation design**:

````carousel
### Frame Layout

```
┌─────────────────────────────────┐
│                                 │
│   2D Map with Convex Regions    │
│   + Trajectory Segments         │
│   + Defect Gap Arrows           │
│   + Shooting Nodes              │
│                                 │
│                                 │
│                                 │
│                                 │
├─────────────────────────────────┤
│ Iter: 42/300  Cost: 51.6        │
│ Max Defect: 2.3e-04             │
│ Max Gap: 1.1e-05                │
└─────────────────────────────────┘
```
<!-- slide -->
### Visual Elements Per Frame

| Element | Style | Description |
|---------|-------|-------------|
| Convex regions | Filled polygons, low alpha | Static background, highlight active path regions |
| Shooting nodes | Red filled circles (●) | `s_minus` entry points for each region |
| Exit points | Red open circles (○) | `F(s_minus, w, delta)` computed exit points |
| Trajectory arcs | Blue curves, per-region | RK4 trajectory within each region |
| Defect gap arrows | Red dashed arrows | From exit point of region i to entry point of region i+1 |
| Start/Goal | Green ●, Red ▲ | Fixed markers |
| Metrics overlay | Text box, top-left | Iteration number, cost, max defect, max gap |

<!-- slide -->
### Animation Phases

**Phase 1: Hold initial state** (`hold_first_frames`)
- Show the initial guess with large defect gaps
- Segments clearly disconnected
- Red gap arrows prominent

**Phase 2: Convergence progression** (main frames)
- Interpolate between IPOPT iteration snapshots
- Segments shift, rotate, stretch
- Gap arrows shrink progressively
- Color of gap arrows transitions: red → orange → green as defect → 0

**Phase 3: Hold final state** (`hold_last_frames`)
- Show converged smooth trajectory
- Gap arrows gone (below threshold)
- Clean continuous path
````

**Frame rendering logic**:

For each frame:
1. Clear dynamic artists (keep static background via `blit`)
2. Map frame index → snapshot index (with optional interpolation between adjacent snapshots for smoother animation)
3. For each active region in the path:
   - Draw the RK4 trajectory arc (positions[:, :2])
   - Draw the shooting node (s_minus) as a filled red circle
   - Draw the exit point (s_plus) as an open red circle
4. For each interface (between consecutive regions):
   - Compute gap = `s_plus[i][:2] - s_minus[i+1][:2]`
   - Draw a dashed arrow from exit point to next entry point
   - Color the arrow based on gap magnitude (red → green colormap)
   - If gap < threshold, hide the arrow
5. Update metrics text overlay

**Snapshot subsampling**: If IPOPT runs 300+ iterations, we subsample snapshots to fit `max_animation_frames`. Strategy:
- Always include first and last snapshot
- Use logarithmic spacing in early iterations (fast changes) and linear spacing later (fine convergence)
- Or adaptive: select snapshots where `|cost_change| > threshold` OR `|defect_change| > threshold`

---

### [MODIFY] [demo/optimizer.py](file:///home/dang-khoa/code/motion-planning/GCS-MMS/demo/optimizer.py)

#### In `PathNLPSolver._build_and_solve_nlp()`

Add an optional `iteration_recorder` parameter:

```python
def _build_and_solve_nlp(self, path_regions, start_state, goal_state,
                         warm_start=None,
                         iteration_recorder=None):  # NEW
```

Before creating the solver, if `iteration_recorder` is provided:
```python
if iteration_recorder is not None:
    opts['iteration_callback'] = iteration_recorder
```

This is a **minimal, non-breaking change** — the callback is only attached when explicitly requested.

#### In `PathNLPSolver.solve_path()`

Similarly, thread the `iteration_recorder` through:

```python
def solve_path(self, path, start_state, goal_state,
               warm_start=None,
               iteration_recorder=None):  # NEW
```

---

### [MODIFY] [demo/main_demo.py](file:///home/dang-khoa/code/motion-planning/GCS-MMS/demo/main_demo.py)

Add a new CLI command:

```
python main_demo.py --shooting-animation --scenario default
```

New function `run_shooting_animation()`:

```python
def run_shooting_animation(config_path, scenario_name, verbose=True):
    """Run a fixed-path NLP solve with iteration recording, then create animation."""
    from scenario_builder import prepare_scenario
    from shooting_animation import (
        ShootingIterationRecorder,
        parse_shooting_snapshot,
        create_shooting_convergence_animation,
    )
    # 1. Prepare scenario (regions, graph, dynamics, config)
    # 2. Run IntegratedMIOCPSolver first to get the discrete path
    # 3. Create ShootingIterationRecorder
    # 4. Re-solve with PathNLPSolver, passing the recorder
    # 5. Parse all recorded snapshots
    # 6. Create animation
```

> [!IMPORTANT]
> The animation targets the **PathNLPSolver** (fixed-path NLP), not the IntegratedMIOCPSolver. Reasons:
> - The fixed-path NLP has clean variable structure (only `s_minus, w, delta` per region)
> - No binary relaxation artifacts — all regions are active
> - The defect convergence is the core MSM story
> - The integrated solver's variable layout is more complex (includes `y, p, z, rho` variables)

**Workflow**:
1. First run the integrated solver to find the optimal discrete path
2. Then re-solve that fixed path with `PathNLPSolver`, this time with the iteration recorder attached
3. Parse all snapshots and produce the animation

---

### [NEW] [tests/test_shooting_animation.py](file:///home/dang-khoa/code/motion-planning/GCS-MMS/tests/test_shooting_animation.py)

Unit tests:
- `test_recorder_captures_snapshots`: Verify `ShootingIterationRecorder` captures ≥1 snapshot on a trivial NLP
- `test_parse_snapshot_matches_solution`: Verify `parse_shooting_snapshot` produces the same per-region data as `_parse_solution` on the final snapshot
- `test_animation_creates_gif`: Verify `create_shooting_convergence_animation` produces a valid GIF file (at least check file exists and is non-empty)

---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| [shooting_animation.py](file:///home/dang-khoa/code/motion-planning/GCS-MMS/demo/shooting_animation.py) | NEW | Recorder callback, snapshot parser, animation renderer |
| [optimizer.py](file:///home/dang-khoa/code/motion-planning/GCS-MMS/demo/optimizer.py) | MODIFY | Thread optional `iteration_recorder` through `solve_path` → `_build_and_solve_nlp` |
| [main_demo.py](file:///home/dang-khoa/code/motion-planning/GCS-MMS/demo/main_demo.py) | MODIFY | Add `--shooting-animation` CLI command |
| [test_shooting_animation.py](file:///home/dang-khoa/code/motion-planning/GCS-MMS/tests/test_shooting_animation.py) | NEW | Unit tests for recorder, parser, and animation |

---

## Open Questions

1. **Output format**: GIF only (current `create_animation` uses PillowWriter), or also MP4 (requires ffmpeg)? GIF keeps zero extra dependencies.

2. **Snapshot interpolation**: Should frames between IPOPT iterations be linearly interpolated for smoother animation, or should each frame correspond exactly to one IPOPT iteration? Linear interpolation looks smoother but may show non-physical intermediate states.

3. **Integrated solver animation**: Should we also support animating the `IntegratedMIOCPSolver` (showing `y_uv`/`p_v` relaxation converging alongside trajectory)? This would be a more complex but also more complete visualization. The plan currently targets only the `PathNLPSolver` for simplicity.

---

## Verification Plan

### Automated Tests
```bash
source demo/.venv/bin/activate
python -m unittest tests/test_shooting_animation.py -v
```

### Manual Verification
```bash
# Generate the animation
python demo/main_demo.py --shooting-animation --scenario default

# Check output
ls -la demo/results/default_shooting_convergence.gif
```

Visual checks:
- First frame shows clearly disconnected segments with red gap arrows
- Middle frames show progressive convergence
- Last frame shows smooth continuous trajectory
- Metrics overlay shows decreasing cost and defect values
- Animation is ≤ 10 seconds, ≤ 5MB file size
