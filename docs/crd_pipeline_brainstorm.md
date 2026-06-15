# CRD Pipeline — Brainstorming & Root Cause Analysis

> **Date:** 2026-06-08  
> **Scope:** End-to-end analysis from ACD2D decomposition through Barrier-DMS NLP.  
> **Status:** Brainstorming only — NO implementation. All proposed fixes are hypotheses for discussion.

---

## 1. Situation Summary

| Scenario      | Status | Paths | Time   | Notes |
|---------------|--------|-------|--------|-------|
| `default_crd` | FAIL   | 68    | 1.402s | Was PASS before last commit |
| `crd_demo`    | FAIL   | —     | —      | Consistently failing |

The `1.402s / 68 paths ≈ 20ms per path` is a diagnostic fingerprint: the NLP is **never reached**. IPOPT takes seconds. Interface QP (Chebyshev LP + OSQP) takes milliseconds. All 68 paths are failing at Stage 2 (Interface QP) before the warm start or barrier solver is invoked.

---

## 2. Pipeline Stages

```
ACD2D decomposition
    → Region buffering + H-rep tightening
        → Graph construction (intersections, composite edge cost)
            → K-shortest paths (Yen's)
                → Interface QP (Chebyshev check + waypoint QP)
                    → Warm-start IVP (RK4 forward integration)
                        → Barrier-DMS NLP (IPOPT, optional log-barrier)
```

---

## 3. Stage 1: ACD2D Decomposition + Buffering

### 3.1 The V-rep / H-rep Inconsistency — Fundamental Structural Bug

After `_tighten_hrep_against_obstacles`, every buffered `ConvexRegion` has a split personality:

| Attribute | Content |
|-----------|---------|
| `vertices` | Buffered polygon vertices — **include obstacle-interior space** |
| `A, b` | Original halfplanes + obstacle face constraints — **exclude obstacle space** |

`get_shapely_polygon()` always returns `Polygon(self.vertices)`, i.e., the V-rep polygon. This polygon **includes obstacle space** even though the H-rep does not.

**Consequence chain:**

1. `compute_intersection(r1, r2)` calls `r1.get_shapely_polygon().intersection(r2.get_shapely_polygon())`
2. The resulting intersection polygon (its vertices) includes obstacle-interior points
3. `_tighten_intersection_hrep_against_obstacles` is called on this intersection
4. Inside: `poly = region.get_shapely_polygon()` → gets the poisoned polygon (includes obstacle space)
5. `poly.buffer(-1e-8)` → interior of the poisoned polygon also includes obstacle space
6. Obstacle face check: `face_seg.intersects(poly_interior)` → TRUE for far more faces than warranted
7. **Result: over-constrained intersection H-rep** with spurious halfplanes

This is the primary cause of the `default_crd` regression. The new tightening function (`_tighten_intersection_hrep_against_obstacles`) uses a better check than the old centroid method — but it operates on a poisoned input.

### 3.2 `_tighten_hrep_against_obstacles` — Centroid Limitation (Pre-existing)

The centroid-based guard:
```python
centroid_val = float(np.dot(n_in, acd_centroid))
if centroid_val >= face_val - 1e-6:
    continue   # skip this face
```

This works only because the ACD2D centroid is in free space. For any polygon whose centroid could be inside an obstacle (a buffer region that extends deeply into an obstacle), this silently skips all faces, leaving the H-rep uncorrected. The intersection tightening was added to fix this — but inherited the poisoned V-rep input problem.

### 3.3 Over-conservative tightening at obstacle corners

When an intersection polygon wraps around an obstacle corner, **both** adjacent obstacle faces may pass the interior check:

```
Obstacle corner:       Both face-A and face-B pass interior check.
      ┌── face-B       Both halfplanes added. Their normals conflict.
      │ obs            Intersection squeezed far below overlap_width.
─────┘
face-A
```

Result: Chebyshev radius drops well below `delta_safe + delta_extra = 0.03`.

### 3.4 ACD2D `tau=0.0` — Maximum decomposition

`tau=0.0` produces the maximum number of convex pieces. In a complex workspace (like `default` with 15 regions) this creates many narrow interfaces near concave corners, each more susceptible to tightening over-constraints. Higher tau would produce fewer, coarser regions with wider interfaces, at the cost of less precise free-space approximation.

---

## 4. Stage 2: Graph Construction

### 4.1 Chebyshev cost uses tightened H-rep

`composite_edge_cost()` computes `rho_ij = chebyshev_radius(intersection.A, intersection.b)`. After over-tightening, `rho_ij` is artificially small, distorting the graph cost function. Paths through less-tightened interfaces are falsely preferred.

### 4.2 Unhandled exception: infeasible intersection H-rep

If over-tightening produces conflicting halfplanes (no feasible point), `chebyshev_center()` calls `linprog` and gets `res.success=False`:
```python
raise ValueError(f"Chebyshev LP failed: {res.message}")
```
This exception propagates up and may crash graph construction entirely rather than gracefully removing the edge.

### 4.3 Path diversity when all intersections are similarly narrow

Yen's algorithm returns paths in non-decreasing cost order. If all intersections have similar artificially small Chebyshev radii, all 68 paths may share the same bottleneck interfaces. There is no mechanism to find paths that avoid the worst-tightened interfaces.

---

## 5. Stage 3: Interface QP (`solve_interface_refinement`)

### 5.1 NarrowInterfaceError — Primary cause of `default_crd` regression

```python
rho_ij = chebyshev_center(A_c, b_c)[1]   # uses tightened intersection H-rep
if rho_ij < margin:  # margin = delta_safe + delta_extra = 0.03
    raise NarrowInterfaceError(...)
```

With `overlap_width = 0.25`, an unobstructed intersection should have `rho ≈ 0.25`. The tightening must be reducing this to `< 0.03` — an 8× reduction — on the critical interfaces of **all 68 paths**. This is only consistent with the poisoned-V-rep hypothesis: the intersection polygon built from obstacle-including vertices creates a polygon whose interior includes obstacle space, matching far more obstacle faces than warranted.

### 5.2 InterfaceQPInfeasible — Secondary failure mode

Even if `rho > margin`, the OSQP waypoint placement may be infeasible if:
- The tightened intersection has conflicting constraints
- Start/goal endpoints force intermediate waypoints into an empty feasible region

### 5.3 Union H-rep fallback (when no cached intersection)

When no intersection is cached, `_interface_hrep` stacks both regions' H-reps:
```python
np.vstack([region[ri].A, region[ri1].A])
np.concatenate([region[ri].b, region[ri1].b])
```
This creates the intersection of both halfspace systems — correct geometrically, but if both regions have obstacle tightening constraints, the result is doubly constrained, further reducing the Chebyshev radius.

---

## 6. Stage 4: Warm-Start IVP

### 6.1 Warm start does not guarantee strict barrier interior

The interface QP guarantees `A·z[i] ≤ b - margin` at waypoints. Between waypoints, the RK4 trajectory may leave the feasible interior due to:
- Heading misalignment (unicycle cannot strafe)
- Short delta: time budget too small to traverse the arc without control saturation
- Trajectory curving outside region between two in-region waypoints

Interior nodes may violate `A·x[k] ≤ b - delta_safe`. When `use_log_barrier=True`:

`_eval_f0_B0` computes `B0 = -Σ h_k * log(max(s, 1e-9))`. With `s < 0`, clamping to `1e-9` gives `log(1e-9) ≈ -20.7`. Over `m * n_int` nodes with violations, `B0` becomes large in magnitude and negative-dominant.

Then:
- `f0 > 0, B0 < 0` → `f0/B0 < 0`
- `clip(negative, mu_min, 1.0) = mu_min = 1e-5`
- Only 1 barrier level scheduled with `mu = 1e-5` (extremely weak)
- With an already-violated warm start and a weak barrier, IPOPT gets poor initial scaling → restoration failure

### 6.2 Heading initialization in tight corridors

The warm start interpolates `theta` linearly between `theta_start` and `theta_goal` across waypoints. For a path requiring sharp direction changes (maze), interpolated heading at a waypoint may require `|omega| > omega_max`, causing the IVP to violate control bounds. The warm start positions states outside the region.

### 6.3 `delta` allocation for very short regions

Time per segment allocated proportionally to arc length. For tiny ACD2D sliver regions, `delta_min = 0.01` may bind, giving no time margin for control adjustments.

---

## 7. Stage 5: Barrier-DMS NLP

### 7.1 `use_log_barrier=False` — no safety guarantees

`crd_demo` has `use_log_barrier` commented out → False. Safety is enforced only by the explicit inequality constraints. IPOPT may place trajectories on the boundary (`s ≈ 0`, within IPOPT tolerance) or slightly outside. This is why earlier `default_crd` (with the YAML bug making log_barrier False) showed obstacle-penetrating trajectories.

### 7.2 Barrier log terms do not cover intersection constraints

`build_barrier_log_terms()` in `constraint_layers.py` computes barriers over `region.A, region.b` (individual region halfplanes) only. The intersection H-rep constraints at coupling nodes are enforced as **explicit inequalities** (`g ≤ 0`), not as barrier terms. This means:

- IPOPT may place coupling nodes right on the intersection boundary (`s ≈ 0`)
- After the first barrier level, this warm start for level 2 has `s ≈ 0`
- `log(0) = -∞` in the barrier → IPOPT restoration failure at level 2

### 7.3 Safety constraint at coupling node uses only intersection H-rep

At `k == n_int, i < m-1` in `_build_nlp`:
```python
isect = graph.intersections.get((node_i, node_i1))
if isect is not None:
    constraint: A_isect @ pos <= b_isect - delta_safe   ← over-constrained after tightening
    continue
```

After over-tightening, `b_isect - delta_safe` may have no feasible solution → NLP is infeasible by construction.

### 7.4 No safety constraint at entry node (k==0, i>0)

```python
if k == 0 and i > 0:
    continue  # assumed covered by exit constraint of previous segment
```

The exit of segment `i` uses the intersection H-rep. The entry of segment `i+1` uses no constraint (only coupling equality links it to the exit). The intersection H-rep ≠ region[i+1]'s H-rep: a point inside the intersection is not necessarily inside region[i+1] (the intersection is typically smaller than region[i+1]). So the entry node of segment `i+1` is geometrically unconstrained by region[i+1]'s safety bounds. If the warm start has segment `i+1`'s entry outside region[i+1]'s feasible set, the barrier log terms at that node yield negative slacks → barrier initialization failure.

### 7.5 `n_ctrl_per_seg=1` insufficient for tight corridors

`crd_demo` uses `n_ctrl_per_seg=1`. One constant `(v, omega)` over a full segment means the robot traces a circular arc. For a corridor requiring heading change while maintaining proximity to walls:

- Circular arc of radius `r = v / omega` must fit entirely in the corridor
- If corridor width < 2r at any point, the arc exits the corridor
- `n_ctrl_per_seg=2` (like `default_crd`) allows: slow+turn, then accelerate — much more flexible

### 7.6 IPOPT schedule exhaustion

```python
ipopt_max_iter_schedule: Tuple[int, ...] = (1000, 500, 300, 200, 200)
```

With 5 barrier levels and a complex multi-segment path, convergence is not guaranteed within these limits. Particularly at early barrier levels (`mu_0` large), IPOPT may spend many iterations on the strongly-regularized problem without converging.

---

## 8. Root Cause Hierarchy

```
ROOT: V-rep / H-rep inconsistency in ConvexRegion
│  vertices includes obstacle space; A,b excludes it; get_shapely_polygon() uses V-rep
│
└─▶ compute_intersection() builds intersection from poisoned V-rep polygons
    │  intersection.vertices includes obstacle space
    │
    └─▶ _tighten_intersection_hrep_against_obstacles() uses poisoned poly
        │  poly_interior includes obstacle space
        │  too many obstacle faces pass interior check
        │  over-constrained H-rep (many spurious halfplanes)
        │
        ├─▶ chebyshev_radius(isect.A, isect.b) << overlap_width
        │      rho < 0.03 on all interfaces of all 68 paths
        │      NarrowInterfaceError on every path
        │      default_crd FAIL in 1.4s  ← PRIMARY REGRESSION
        │
        └─▶ _build_nlp safety constraint at k==n_int uses over-constrained H-rep
               A_isect @ pos <= b_isect - delta_safe may be infeasible
               NLP infeasible by construction  ← SECONDARY FAILURE
```

### Independent secondary issues (not caused by the above)

```
_eval_f0_B0 corrupted when warm start violates constraints
   → B0 negative-dominant → mu_0 = mu_min → 1-level schedule → IPOPT failure

build_barrier_log_terms omits intersection constraints
   → coupling nodes placed on boundary → next level barrier explodes

Entry node k==0 i>0 has no safety constraint
   → entry can violate region[i+1] bounds undetected

n_ctrl_per_seg=1 (crd_demo) physically insufficient for tight turns
```

---

## 9. Proposed Fix Directions (Not Ranked — Brainstorming Only)

### Direction A: Recompute V-rep from tightened H-rep

After `_tighten_hrep_against_obstacles`, solve the H-rep polytope for new vertices using `scipy.spatial.HalfspaceIntersection` or `pycddlib`. Makes `get_shapely_polygon()` return the obstacle-free polygon everywhere.

**Pro:** Fixes the root cause at its source — every downstream function becomes correct automatically.  
**Con:** Vertex enumeration for 2D polytopes requires a robust library. May produce degenerate/empty polytopes for very tight tightening (needs graceful handling).

### Direction B: Use un-buffered adjacency regions as reference geometry

`adjacency_regions` (stored but currently used only for adjacency detection) represent the original ACD2D free-space polygons — no obstacle-including buffer. Pass them to `compute_intersection()` and use `adj_r1.get_shapely_polygon()` for the interior check instead of the buffered region's poisoned polygon.

**Sketch:**
```python
def compute_intersection(region1, region2, adj1=None, adj2=None, obs_polys=None):
    ...
    if obs_polys:
        ref_poly = adj1.get_shapely_polygon() if adj1 else None
        result = _tighten_intersection_hrep_v2(result, obs_polys, ref_poly)
```

**Pro:** Simple and architecturally clean. Un-buffered adjacency regions are correctly in free space.  
**Con:** Requires threading `adj1, adj2` through `compute_intersection` and `build_region_graph`. Also: the adjacency polygon may be smaller than the actual buffered intersection, so some valid obstacle constraints near the buffer extension may be missed.

### Direction C: Subtract obstacle space before interior check

Compute the free-space portion of the intersection polygon directly:
```python
from shapely.ops import unary_union
free_part = poly.difference(unary_union(obs_polys))
free_interior = free_part.buffer(-1e-8)
if free_interior.is_empty or not face_seg.intersects(free_interior):
    continue
```

**Pro:** Directly answers "does this obstacle face cut through the free-space part of the intersection" — the correct geometric question.  
**Con:** `poly.difference(obs)` is non-convex in general; resulting `free_interior` may be a MultiPolygon. Intersection check still works but is more expensive. Also: if the entire intersection is within an obstacle (degenerate case), `free_part` is empty — then no constraints are added, which is correct (the edge should not exist in the graph).

### Direction D: Limit to one constraint per obstacle face pair

For each obstacle, add at most the single most-restrictive face. When two faces from the same obstacle corner both pass the interior check, take only the one with smaller slack margin. This prevents conflicting constraints from corner cases.

**Pro:** Simple heuristic fix, no structural change needed.  
**Con:** Does not fix the poisoned V-rep root cause — still may add wrong constraints, just fewer of them. May miss necessary constraint if both faces need to be applied.

### Direction E: Soft fallback for NarrowInterfaceError

When `rho < margin`, instead of throwing immediately, try: relax `delta_extra` to 0 (just `delta_safe` margin), and if still failing, use a relaxed margin of 0. Allow the path to proceed to NLP.

**Pro:** More paths reach the NLP. NLP with explicit inequality constraints can still work even with a tight intersection.  
**Con:** The interface QP waypoints will be less strictly interior; warm start may violate safety. Risk of NLP restoration failure.

### Direction F: Revert intersection tightening; fix NLP coupling constraint differently

Temporarily skip `_tighten_intersection_hrep_against_obstacles`. This reverts to pre-fix behavior for interface QP (passes all paths). For the NLP coupling node, instead of using the intersection H-rep, use the intersection of **both regions'** H-reps explicitly at `k == n_int`:
```python
# At k==n_int, constrain pos to be in BOTH region[i] and region[i+1]
# (the actual H-rep intersection in free space, without the poisoned polygon)
A_both = np.vstack([region[i].A, region[i+1].A])
b_both = np.hstack([region[i].b, region[i+1].b])
constraint: A_both @ pos <= b_both - delta_safe
```

**Pro:** Immediate fix to the regression. No new library needed.  
**Con:** Union of two region H-reps may be over-constrained too (both regions have their own obstacle tightening). The coupling constraint becomes very tight for adjacent regions that are both near an obstacle.

### Direction G: Debug-first — no code change yet

Before any fix, add instrumentation:
1. Print `(u, v, rho_original, rho_tightened, n_added)` for each intersection during graph construction
2. Print which obstacle face(s) are being added to the worst-tightened intersections
3. Compare the tightened intersection polygon (H-rep vertices) against the true free-space intersection (adjacency region intersection)

This confirms or refutes each hypothesis with numbers before committing to a structural fix.

---

## 10. `crd_demo` — Specific Analysis

### Known config

- `n_ctrl_per_seg: 1` (single constant control per segment)
- `use_log_barrier: false` (commented out)
- `delta_safe=0.03, delta_extra=0.01` → margin=0.04
- `n_int=10`

### Failure tree

```
If intersection tightening over-constrains interfaces:
    → NarrowInterfaceError → FAIL before NLP (same as default_crd regression)

If interface QP passes (post-tightening-fix):
    → n_ctrl=1: single circular arc per segment
    → Tight passages require turning radius < corridor half-width
    → If v/omega > half_width, arc exits corridor → IPOPT infeasible
    → DURATION_SATURATION or CONTROL_SATURATION failure code

If NLP converges without barrier:
    → Trajectory may touch boundary (s ≈ 0)
    → Result check: s_min_sampled < 0 → classified as failure
    → Or: trajectory visually through obstacles (within IPOPT tolerance)
```

### Control insufficiency — quantitative

With `n_ctrl=1, n_int=10`, corridor width `w`:
- Unicycle traces arc radius `r = v/omega`
- Minimum turning radius at `omega_max = π ≈ 3.14 rad/s`, `v_nom ≈ 0.5 * 2 = 1.0 m/s`
- `r_min = 1.0 / π ≈ 0.32m`
- If corridor width < `2 * 0.32 = 0.64m` at a bend, the arc exits the corridor

With `n_ctrl=2`: the robot can reduce v during the turn → smaller arc radius → tighter turns feasible.

---

## 11. Investigation Priority (Before Any Fix)

1. **Add 5-line debug to `solve_interface_refinement`** (read-only): Print `rho` for each interface that fails the Chebyshev check on the first 3 failed paths of `default_crd`. This gives immediate numbers.

2. **Add 5-line debug after `build_region_graph`**: For each intersection in `graph.intersections`, print `(u, v, n_halfplanes_original, n_halfplanes_total)`. If `n_halfplanes_total >> n_halfplanes_original` near obstacles, poisoned-V-rep hypothesis is confirmed.

3. **Run `default_crd` with `_tighten_intersection_hrep_against_obstacles` disabled** (one-line change): If it passes, the regression is 100% caused by the new tightening function. Then fix the tightening input (Direction B or C), not the tightening logic itself.

4. **After fixing tightening**: test `crd_demo` with `n_ctrl_per_seg=2` (matches `default_crd`). If it passes, the original n_ctrl=1 config was the main blocker for crd_demo.

5. **Enable `use_log_barrier: true` for `crd_demo`** (comment it in config): The barrier prevents obstacle penetration and should improve solution quality once the interface QP regression is fixed.

---

## 13. Chẩn đoán xác nhận — maze_crd (2026-06-08)

> **Trạng thái:** Root cause xác nhận bằng `diag_a1.py` → `diag_a1d.py`.  
> **Kết quả:** 3240 paths thất bại, DMS chưa bao giờ được gọi (defect_norm=0, ~19ms/path).

### 13.1 Root cause — maze_crd

`_tighten_hrep_against_obstacles` clip từng buffered region về obstacle face và recompute V-rep qua `_vrep_from_hrep` (`convex_regions.py:470`). `region.vertices` là V-rep **tightened**, không phải raw buffered polygon.

Với hai region ở hai phía đối diện bức tường: mỗi bên bị clip về wall face tương ứng → gap = wall thickness → `poly_A.intersection(poly_B) = POLYGON EMPTY` → `compute_intersection` trả về `None` → `_interface_hrep` fallback stacked H-rep → LP infeasible → `NarrowInterfaceError`.

| Metric | Giá trị |
|--------|---------|
| Cạnh tightened-empty (72%) | 316 / 440 |
| Mean gap giữa tightened polygons | 0.13 đơn vị |
| Max gap | 2.57 đơn vị |
| Stacked H-rep feasible (sample 5) | 0 / 5 |
| Stored intersections (gap = 0) | 124 / 440 |

### 13.2 Ràng buộc cứng bổ sung (chưa implement)

1. **Point-contact edges**: `adj_i ∩ adj_j` là `Point` → không thêm vào graph.
2. **Tightened-empty edges**: `compute_intersection` trả về `None` → không thêm vào `region_edges`. Không fallback stacked H-rep.

Chi tiết: `docs/superpowers/specs/2026-06-07-region-decomposition-overlap-design.md` Section 1.3 items 4–5.

---

## 12. Summary

| Issue | Stage | Severity | Root Cause |
|-------|-------|----------|-----------|
| `default_crd` FAIL regression | Interface QP | **CRITICAL** | Poisoned V-rep → over-tightened intersection H-rep → NarrowInterfaceError on all 68 paths |
| Trajectories through obstacles (pre-fix) | NLP | HIGH | `use_log_barrier` silently False due to YAML indent bug |
| `crd_demo` fail | Multi-stage | HIGH | (1) Same tightening regression; (2) n_ctrl=1 insufficient; (3) no barrier |
| `B0` corruption when warm start violates constraints | Barrier init | MEDIUM | log(1e-9) clamp inflates \|B0\|; mu_0 → mu_min; 1-level schedule too weak |
| No barrier on intersection constraints | NLP | MEDIUM | `build_barrier_log_terms` uses region H-rep only, not isect H-rep |
| Missing safety at k==0, i>0 | NLP | LOW-MED | Entry node skipped; region[i+1] bounds not enforced at entry |
| Unhandled `ValueError` in `chebyshev_center` | Graph build | LOW | Infeasible H-rep → LP fails → uncaught exception crashes graph construction |
| ACD2D `tau=0.0` max decomposition | Decomp | INFO | Many narrow regions; wider interfaces possible with tau > 0 |
