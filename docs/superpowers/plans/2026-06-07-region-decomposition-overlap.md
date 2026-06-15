# Region Decomposition & Overlap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce ACD2D as the mandatory decomposition step for all solver modes by replacing `skip_acd=True` presets with real obstacle geometry and making `overlap_width` a configurable parameter that drives region buffer size.

**Architecture:** Remove the `skip_acd` branch from `scenario_builder.build_environment_and_regions`. Every preset goes through ACD2D → Shapely mitre buffer(`overlap_width`) → `build_region_graph`. The `maze_15` preset gets real obstacle geometry; `crd_demo` is redirected to the existing `default` preset workspace. Adjacency detection keeps using unbuffered ACD2D regions.

**Tech Stack:** Python 3.11, Shapely, ACD2D, dataclasses (frozen), PyYAML, pytest

---

## File Map

| File | Change |
|------|--------|
| `demo/config.yaml` | Add `region_decomposition.overlap_width: 0.5` block |
| `demo/problem_data.py` | Remove `skip_acd` field; update `maze_15` with obstacle geometry; clear `crd_demo` hand-crafted regions |
| `demo/scenario_builder.py` | Remove `skip_acd` branch; add `overlap_width` param; add startup validation |
| `tests/test_scenario_builder.py` (new) | Unit tests for `build_environment_and_regions` |
| `docs/superpowers/specs/2026-06-07-centroid-refine-dms-design.md` | Add Part 0 section |
| `docs/2026-06-07-implementation-gap-report.md` | Add 3 new gaps, mark them fixed |

---

## Task 1: Add `region_decomposition` config block

**Files:**
- Modify: `demo/config.yaml`

- [ ] **Step 1: Add block after the `shooting:` section (before `cost:`)**

```yaml
# -----------------------------------------------------------------------------
# Region Decomposition Configuration
# -----------------------------------------------------------------------------
# ACD2D decomposes the workspace into edge-touching convex polygons.
# Each polygon is then expanded outward by overlap_width (Shapely mitre buffer
# + workspace clip) so adjacent regions overlap by 2*overlap_width at shared edges.
#
# Hard constraint: overlap_width >= centroid_refine_dms.delta_safe + delta_extra
# (otherwise Interface QP will fail with NARROW_INTERFACE on all paths).
# Legacy and two_stage solvers are unaffected — extra feasible space at interfaces.
region_decomposition:
  overlap_width: 0.5
```

Global value 0.5 satisfies the most conservative CRD scenario (`delta_safe=0.3, delta_extra=0.15`, sum=0.45 < 0.5).

- [ ] **Step 2: Commit**

```bash
git add demo/config.yaml
git commit -m "config: add region_decomposition.overlap_width=0.5"
```

---

## Task 2: Update `problem_data.py` — remove `skip_acd`, redesign presets

**Files:**
- Modify: `demo/problem_data.py`

- [ ] **Step 1: Write a failing test**

Create `tests/test_scenario_builder.py`:

```python
"""Tests for build_environment_and_regions — ACD2D mandatory pipeline."""
import pytest
from problem_data import CRD_DEMO_PROBLEM, MAZE_15_PROBLEM, ProblemPresetSpec


def test_preset_spec_has_no_skip_acd_field():
    """ProblemPresetSpec must not have a skip_acd field after refactor."""
    assert "skip_acd" not in ProblemPresetSpec.__dataclass_fields__


def test_maze_15_has_obstacles():
    """maze_15 must define obstacle geometry for ACD2D to produce a corridor."""
    assert len(MAZE_15_PROBLEM.obstacle_vertices) > 0


def test_crd_demo_has_no_region_vertices():
    """crd_demo must not have hand-crafted region_vertices after refactor."""
    assert len(CRD_DEMO_PROBLEM.region_vertices) == 0
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/khoa/ws/GCS_MMS_demo && ../.venv/bin/pytest tests/test_scenario_builder.py -v 2>&1 | tail -20
```

Expected: all 3 FAIL.

- [ ] **Step 3: Remove `skip_acd` from `ProblemPresetSpec` and make `region_vertices` optional**

In `demo/problem_data.py`:

Change import line from:
```python
from dataclasses import dataclass
```
To:
```python
from dataclasses import dataclass, field
```

Remove line 24:
```python
skip_acd: bool = False  # bypass ACD; use region_vertices directly as safe regions
```

Change line 20 from:
```python
region_vertices: List[List[List[float]]]
```
To:
```python
region_vertices: List[List[List[float]]] = field(default_factory=list)
```

- [ ] **Step 4: Update `MAZE_15_PROBLEM` with real obstacle geometry**

The obstacle is a horizontal slab at y=6–9 from x=0 to x=9. This blocks direct vertical movement left/middle, forcing path: bottom strip → right corridor (x=9–15) → top strip.

Replace the full `MAZE_15_PROBLEM` definition:

```python
MAZE_15_PROBLEM = ProblemPresetSpec(
    key="maze_15",
    name="Maze 15x15 - Zigzag Corridor",
    description=(
        "15x15 workspace with a horizontal barrier (y=6-9, x=0-9) forcing a zigzag: "
        "bottom strip -> right-side passage (x=9-15) -> top strip. "
        "ACD2D decomposes free space; mitre buffer creates overlap for CentroidRefineDMS."
    ),
    workspace_vertices=[[0.0, 0.0], [15.0, 0.0], [15.0, 15.0], [0.0, 15.0]],
    obstacle_vertices=[
        [[0.0, 6.0], [9.0, 6.0], [9.0, 9.0], [0.0, 9.0]],
    ],
    default_start_state=[1.0, 1.0, 0.0],
    default_goal_state=[1.0, 14.0, 1.5708],
)
```

- [ ] **Step 5: Update `CRD_DEMO_PROBLEM` to use `default` workspace (no hand-crafted regions)**

Replace the full `CRD_DEMO_PROBLEM` definition:

```python
CRD_DEMO_PROBLEM = ProblemPresetSpec(
    key="crd_demo",
    name="CRD Demo - Default Workspace",
    description=(
        "Default 5x5 workspace with obstacles; ACD2D decomposition + mitre buffer "
        "provides overlapping regions for CentroidRefineDMS."
    ),
    workspace_vertices=DEFAULT_PROBLEM.workspace_vertices,
    obstacle_vertices=DEFAULT_PROBLEM.obstacle_vertices,
    default_start_state=[0.5, 0.5, 0.785],
    default_goal_state=[4.5, 4.5, 0.785],
)
```

- [ ] **Step 6: Run tests**

```bash
cd /home/khoa/ws/GCS_MMS_demo && ../.venv/bin/pytest tests/test_scenario_builder.py -v 2>&1 | tail -20
```

Expected: all 3 PASS.

- [ ] **Step 7: Commit**

```bash
git add demo/problem_data.py tests/test_scenario_builder.py
git commit -m "feat: remove skip_acd from presets; maze_15 uses obstacle geometry"
```

---

## Task 3: Update `scenario_builder.py` — remove `skip_acd` branch, add `overlap_width`

**Files:**
- Modify: `demo/scenario_builder.py`

- [ ] **Step 1: Add overlap and connectivity tests to `tests/test_scenario_builder.py`**

```python
import numpy as np
from scenario_builder import build_environment_and_regions
from problem_data import MAZE_15_PROBLEM, DEFAULT_PROBLEM
from convex_regions import compute_intersection


def test_adjacent_regions_overlap_after_buffer():
    """After mitre buffer, adjacent ACD2D regions must have positive-area intersection."""
    _, regions, _ = build_environment_and_regions(DEFAULT_PROBLEM, overlap_width=0.3)
    found_overlap = False
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            inter = compute_intersection(regions[i], regions[j])
            if inter is not None and inter.get_shapely_polygon().area > 1e-4:
                found_overlap = True
                break
        if found_overlap:
            break
    assert found_overlap, "No adjacent region pair has positive-area overlap after buffer"


def test_overlap_width_controls_interface_area():
    """Larger overlap_width produces larger interface area."""
    def max_interface_area(regions):
        best = 0.0
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                inter = compute_intersection(regions[i], regions[j])
                if inter is not None:
                    best = max(best, inter.get_shapely_polygon().area)
        return best

    _, regions_small, _ = build_environment_and_regions(DEFAULT_PROBLEM, overlap_width=0.1)
    _, regions_large, _ = build_environment_and_regions(DEFAULT_PROBLEM, overlap_width=0.5)
    assert max_interface_area(regions_large) > max_interface_area(regions_small)


def test_maze_15_produces_connected_graph():
    """maze_15 ACD2D decomposition must yield at least one SOURCE->TARGET path."""
    from graph_builder import build_region_graph, SOURCE, TARGET

    _, regions, adjacency_regions = build_environment_and_regions(
        MAZE_15_PROBLEM, overlap_width=0.5
    )
    start = np.array(MAZE_15_PROBLEM.default_start_state[:2])
    goal  = np.array(MAZE_15_PROBLEM.default_goal_state[:2])
    graph = build_region_graph(regions, start, goal, adjacency_regions=adjacency_regions)
    paths = list(graph.enumerate_simple_paths(SOURCE, TARGET, max_paths=5))
    assert len(paths) >= 1, (
        f"maze_15 has no path. Regions: {len(regions)}. "
        "Check obstacle geometry and overlap_width."
    )
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/khoa/ws/GCS_MMS_demo && ../.venv/bin/pytest tests/test_scenario_builder.py -v 2>&1 | tail -30
```

Expected: new tests FAIL — `build_environment_and_regions` does not accept `overlap_width`.

- [ ] **Step 3: Replace `build_environment_and_regions` in `scenario_builder.py`**

Remove the entire existing function body (the `if preset.skip_acd:` branch + `else:` block). Replace with:

```python
def build_environment_and_regions(
    preset: ProblemPresetSpec,
    overlap_width: float = 0.05,
) -> tuple[Environment, List[ConvexRegion], List[ConvexRegion]]:
    """Construct environment and convex regions from a preset via ACD2D.

    All presets go through ACD2D decomposition. Resulting edge-touching polygons
    are expanded outward by overlap_width (Shapely mitre buffer clipped to workspace)
    so adjacent regions overlap by 2*overlap_width at shared edges.

    Args:
        preset: Problem geometry specification.
        overlap_width: Outward buffer per polygon. Must satisfy
            overlap_width >= delta_safe + delta_extra for CRD solver mode.

    Returns:
        (environment, regions, adjacency_regions):
        - regions: mitre-buffered ConvexRegion objects used by the optimizer.
        - adjacency_regions: original ACD2D regions used only for adjacency detection.
    """
    environment = create_environment_from_vertices(
        preset.workspace_vertices,
        preset.obstacle_vertices,
    )

    try:
        region_vertices, _result = acd.decompose_to_polygons(
            environment.workspace,
            holes=environment.obstacles,
        )
    except Exception as exc:
        print(f"Warning: ACD decomposition failed: {exc}")
        region_vertices = []

    if not region_vertices:
        if not preset.region_vertices:
            raise RuntimeError(
                f"ACD decomposition produced no regions for preset '{preset.key}' "
                "and no fallback region_vertices are defined."
            )
        print(f"Warning: ACD failed for '{preset.key}', using fallback region_vertices.")
        region_vertices = [
            np.asarray(v, dtype=np.float64) for v in preset.region_vertices
        ]

    print(
        f"Built environment with {len(environment.obstacles)} obstacles "
        f"and {len(region_vertices)} regions"
    )

    adjacency_regions = create_regions_from_vertices_list(region_vertices)
    regions = create_buffered_regions_from_vertices_list(
        region_vertices,
        preset.workspace_vertices,
        buffer_size=overlap_width,
    )
    return environment, regions, adjacency_regions
```

- [ ] **Step 4: Update `prepare_scenario` to read `overlap_width` from config**

In `prepare_scenario`, add before the `build_environment_and_regions` call:

```python
import warnings

decomp_cfg    = resolved.runtime_config.get("region_decomposition", {})
overlap_width = float(decomp_cfg.get("overlap_width", 0.05))

solver_mode = resolved.runtime_config.get("optimizer", {}).get("solver_mode", "")
if solver_mode == "centroid_refine_dms":
    crd_cfg  = resolved.runtime_config.get("centroid_refine_dms", {})
    required = (float(crd_cfg.get("delta_safe", 0.0))
                + float(crd_cfg.get("delta_extra", 0.0)))
    if required > 0 and overlap_width < required:
        warnings.warn(
            f"[CRD] overlap_width={overlap_width} < "
            f"delta_safe+delta_extra={required:.3f}. "
            "Interface QP may fail with NARROW_INTERFACE on all paths.",
            stacklevel=2,
        )

environment, all_regions, all_adjacency_regions = build_environment_and_regions(
    preset, overlap_width=overlap_width
)
```

- [ ] **Step 5: Run all tests**

```bash
cd /home/khoa/ws/GCS_MMS_demo && ../.venv/bin/pytest tests/ -v --tb=short 2>&1 | tail -40
```

Expected: all tests pass. If `test_maze_15_produces_connected_graph` fails, run the diagnostic:

```bash
cd /home/khoa/ws/GCS_MMS_demo/demo && ../.venv/bin/python3 - <<'EOF'
from problem_data import MAZE_15_PROBLEM
from scenario_builder import build_environment_and_regions
from graph_builder import build_region_graph, SOURCE, TARGET
import numpy as np

_, regions, adj = build_environment_and_regions(MAZE_15_PROBLEM, overlap_width=0.5)
print(f"Regions: {len(regions)}")
for r in regions:
    print(f"  R{r.index}: centroid={r.get_centroid().round(2)}")
start = np.array([1.0, 1.0])
goal  = np.array([1.0, 14.0])
graph = build_region_graph(regions, start, goal, adjacency_regions=adj)
paths = list(graph.enumerate_simple_paths(SOURCE, TARGET, max_paths=3))
print(f"Paths found: {len(paths)}")
for p in paths:
    print(f"  {p}")
EOF
```

If 0 paths, adjust obstacle in `problem_data.py` (narrow the y-range or shrink x-extent) and rerun:

```python
# Try narrower barrier if original fails:
obstacle_vertices=[
    [[0.0, 6.5], [8.5, 6.5], [8.5, 8.5], [0.0, 8.5]],
],
```

- [ ] **Step 6: Commit**

```bash
git add demo/scenario_builder.py
git commit -m "feat: mandatory ACD2D pipeline; overlap_width replaces skip_acd buffer_size=0.001"
```

---

## Task 4: End-to-end smoke test

**Files:** No code changes — verification only.

- [ ] **Step 1: Run `crd_demo`**

```bash
cd /home/khoa/ws/GCS_MMS_demo/demo && ../.venv/bin/python main_demo.py --scenario crd_demo 2>&1 | tail -20
```

Expected:
```
Built environment with N obstacles and M regions
Optimization Status: SUCCESS
Solver: CRD: KKT solution
```

If FAIL with `NARROW_INTERFACE`: increase `overlap_width` in `config.yaml`.
If FAIL with `No path exists`: ACD2D regions disconnected — check obstacle geometry.

- [ ] **Step 2: Run `maze_15_crd`**

```bash
cd /home/khoa/ws/GCS_MMS_demo/demo && ../.venv/bin/python main_demo.py --scenario maze_15_crd 2>&1 | tail -20
```

Expected: `Optimization Status: SUCCESS`

- [ ] **Step 3: Run `default` and `default_crd` for regression**

```bash
cd /home/khoa/ws/GCS_MMS_demo/demo && ../.venv/bin/python main_demo.py --scenario default --scenario default_crd 2>&1 | tail -15
```

Expected: both SUCCESS.

- [ ] **Step 4: Commit smoke test confirmation**

```bash
git commit --allow-empty -m "verify: all scenarios pass with mandatory ACD2D pipeline"
```

---

## Task 5: Documentation updates

**Files:**
- Modify: `docs/superpowers/specs/2026-06-07-centroid-refine-dms-design.md`
- Modify: `docs/2026-06-07-implementation-gap-report.md`

- [ ] **Step 1: Add Part 0 to CRD design spec**

Find the line `## Part 1` in the spec and insert immediately before it:

```markdown
## Part 0 — Region Decomposition Pipeline

### 0.1 GCS goc vs. ky thuat hien tai

**GCS goc (Marcucci et al. 2022):** Xay dung do thi toi uu tren cac convex region *chong lan nhau*. Tai vung giao $C_i \cap C_j$, rang buoc lien tuc $x_i(1) = x_j(0)$ duoc dat truc tiep. Overlap la dieu kien can de rang buoc nay co nghiem khong tam thuong. Cac region thuong duoc tao bang IRIS (Deits & Tedrake 2015).

**Ky thuat hien tai:** ACD2D phan hoach workspace thanh cac da giac loi *khong chong lan* (edge-touching). De tao overlap can thiet, moi vung ACD2D duoc mo rong ra ngoai mot luong `overlap_width` bang Shapely mitre buffer + workspace clip:

```python
buffered = poly.buffer(overlap_width, join_style='mitre')  # giu loi
clipped  = buffered.intersection(workspace_polygon)         # clip tai bien
```

`join_style='mitre'` giu nguyen goc sac → polygon sau buffer van loi. Workspace clip loai bo phan mo rong tai bien workspace → chi canh chung giua cac vung lan can duoc mo rong thuc su. Giao $C_i \cap C_j$ la dai rong $2 \cdot \text{overlap\_width}$ doc canh chung.

### 0.2 Yeu cau cung

1. Moi solver mode (`integrated_relaxation_legacy`, `two_stage`, `centroid_refine_dms`) deu di qua pipeline: `ACD2D -> mitre buffer -> build_region_graph`.
2. `overlap_width >= delta_safe + delta_extra` de Interface QP luon co Chebyshev radius du lon.
3. Adjacency detection dung vung ACD2D goc (truoc buffer) de tranh canh gia.

### 0.3 Cau hinh

```yaml
region_decomposition:
  overlap_width: 0.5   # >= delta_safe + delta_extra cua moi CRD scenario
```

---
```

- [ ] **Step 2: Update gap report — append 3 gaps**

In `docs/2026-06-07-implementation-gap-report.md`, find the gaps table and append these rows:

```markdown
| `skip_acd=True` bypass ACD2D trong `maze_15`/`crd_demo` | Architectural gap | High | ✅ Fixed — Tasks 2+3 |
| `buffer_size=0.001` qua nho, khong tao overlap du cho CRD | Config gap | High | ✅ Fixed — Tasks 1+3 |
| Thieu block `region_decomposition` trong `config.yaml` | Missing config | Medium | ✅ Fixed — Task 1 |
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-06-07-centroid-refine-dms-design.md \
        docs/2026-06-07-implementation-gap-report.md
git commit -m "docs: add Part 0 region decomposition to CRD spec; update gap report"
```
