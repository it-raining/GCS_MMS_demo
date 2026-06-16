# CRD Diagnostics & Safety-Certificate Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every diagnostic field reported by `CentroidRefineDMSSolver` (defect, coupling gap, lower bound, optimality gap, NLP iteration count, certified safety margin) reflect a real measurement instead of an unset dataclass default, and fix the certified-safety-margin formula so it matches design-spec §8.1/§8.2 exactly (it currently omits `δ_safe`, `ε_defect`, and `ε_int`, which can make an unsafe trajectory report as `"CERTIFIED"`).

**Architecture:** All numerical fixes live in `demo/barrier_dms.py` (where the NLP is built and solved) and are threaded up through `demo/optimizer.py`'s `CentroidRefineDMSSolver.solve()`. A small new helper in `demo/dynamics.py` supplies the RK4-truncation Lipschitz constant `L_f`. A small new helper in `demo/graph_builder.py` supplies the pure-geometric Dijkstra length needed for the lower bound. No constraint, no barrier, and no enforcement-side `δ_safe` logic changes — only measurement/reporting code changes (R1, R2, R13 untouched).

**Tech Stack:** Python, CasADi/IPOPT, NetworkX, NumPy, unittest (existing test style in `tests/` and `demo/`).

**Source spec:** `docs/superpowers/specs/2026-06-07-centroid-refine-dms-design.md` (§7.1–7.3, §8.1–8.3, Part 9, Part 10).
**Audit this plan implements:** `docs/superpowers/specs/2026-06-16-crd-diagnostics-audit.md` (items F1–F9 + UI step).

---

## File Structure

- Modify `demo/dynamics.py` — add abstract `f_lipschitz_bound()` + two concrete implementations (used for §8.2's `ε_int` term).
- Modify `demo/barrier_dms.py` — fix `_compute_min_slack` (F5), add `_compute_defect_norm` (F1), fix `certified_safety_margin` formula (F6), split NLP objective into `obj_base`/`barrier_term` so `total_cost` excludes the barrier term (F7), record `n_nlp_iterations` per barrier level (F8).
- Modify `demo/graph_builder.py` — record `geom_dist` edge attribute and add `dijkstra_geometric_length()` helper (needed for F3).
- Modify `demo/optimizer.py` — wire `defect_norm`/`max_connection_gap` into the `OptimizationResult` built by `CentroidRefineDMSSolver` (F1/F2), compute and set `lb_geometric`/`optimality_gap` (F3/F4), replace the disclaimer string with the exact spec text (F9).
- Modify `demo/experiments.py` — label CTCS/Dense fields as not-applicable when `safety_mode != "both"` instead of printing a misleading `0.00e+00` (UI step / F10 follow-through).
- Add tests to `tests/test_barrier_dms.py`, `tests/test_graph_builder.py`, and new files `tests/test_dynamics.py`, `tests/test_centroid_refine_dms.py`.

---

### Task 1: `f_lipschitz_bound()` on `DynamicsModel`

**Why:** §8.2 defines `ε_int ≈ L_f · h_max⁴ / 30`, the RK4 truncation-error term in the certified safety margin. `L_f` (a Lipschitz bound on `f(x,u)` w.r.t. `x`) does not exist anywhere in the codebase yet.

For the unicycle, `f = [v cosθ, v sinθ, ω]`; the only state-dependence is through `θ`, and `|∂(v cosθ)/∂θ| = |v sinθ| ≤ v_max`, `|∂(v sinθ)/∂θ| = |v cosθ| ≤ v_max`, so `L_f = v_max`.

For the double integrator, `f = [v_x, v_y, a_x, a_y]` is exactly linear in `x` with Jacobian rows `[0,0,1,0]` and `[0,0,0,1]` (constant, state-independent); its operator norm is `1.0`.

**Files:**
- Modify: `demo/dynamics.py:128` (after `compute_lipschitz_bound` abstract method, in `DynamicsModel`)
- Modify: `demo/dynamics.py:275` area (`UnicycleModel`, right after its `compute_lipschitz_bound`)
- Modify: `demo/dynamics.py:339` area (`DoubleIntegratorDynamics`, right after its `compute_lipschitz_bound`)
- Test: `tests/test_dynamics.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_dynamics.py`:

```python
from __future__ import annotations
import sys
import unittest
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from dynamics import UnicycleModel, DoubleIntegratorDynamics


class FLipschitzBoundTests(unittest.TestCase):
    def test_unicycle_bound_equals_v_max(self) -> None:
        dynamics = UnicycleModel(v_max=3.5)
        self.assertEqual(dynamics.f_lipschitz_bound(), 3.5)

    def test_double_integrator_bound_is_one(self) -> None:
        dynamics = DoubleIntegratorDynamics(v_max=3.5, a_max=1.0)
        self.assertEqual(dynamics.f_lipschitz_bound(), 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dynamics.py -v`
Expected: FAIL with `AttributeError: 'UnicycleModel' object has no attribute 'f_lipschitz_bound'`

- [ ] **Step 3: Add the abstract method to `DynamicsModel`**

In `demo/dynamics.py`, immediately after the `compute_lipschitz_bound` abstract method (ends at line 133 with `pass`), add:

```python

    @abstractmethod
    def f_lipschitz_bound(self) -> float:
        """
        Lipschitz bound L_f on f(x, u) with respect to x, used for the RK4
        truncation-error term in the safety certificate
        (design spec Sec 8.2: epsilon_int ~ L_f * h_max^4 / 30).
        """
        pass
```

- [ ] **Step 4: Implement for `UnicycleModel`**

In `demo/dynamics.py`, immediately after `UnicycleModel.compute_lipschitz_bound` (the method ending `return max_normal * self.v_max` around line 281), add:

```python

    def f_lipschitz_bound(self) -> float:
        """d/dtheta[v cos theta, v sin theta] has norm <= v_max (Sec 8.2)."""
        return self.v_max
```

- [ ] **Step 5: Implement for `DoubleIntegratorDynamics`**

In `demo/dynamics.py`, immediately after `DoubleIntegratorDynamics.compute_lipschitz_bound` (ending `return max_normal * self.v_max` around line 345), add:

```python

    def f_lipschitz_bound(self) -> float:
        """f is linear in x; the velocity-to-position Jacobian block has unit norm."""
        return 1.0
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_dynamics.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Commit**

```bash
git add tests/test_dynamics.py demo/dynamics.py
git commit -m "Add f_lipschitz_bound for the RK4 truncation-error term (spec Sec 8.2)"
```

---

### Task 2: Fix `_compute_min_slack` to subtract `δ_safe` (F5)

**Why:** Spec §2.1/§8.1: `s_{i,j,k} = b_{i,j} - δ_safe - aᵀPx`. The NLP barrier enforces this correctly, but `_compute_min_slack` (the function that produces the *reported* `min_safety_margin`) computes `region.b - region.A @ pos` — it omits `δ_safe` entirely, which inflates the reported sampled safety margin by exactly `δ_safe` and can make a path that is actually unsafe under the spec definition certify as `"CERTIFIED"`.

**Files:**
- Modify: `demo/barrier_dms.py:460-491` (`_compute_min_slack`)
- Modify: `demo/barrier_dms.py:136` (call site inside `solve()`)
- Test: `tests/test_barrier_dms.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_barrier_dms.py` (inside `BarrierDMSSolverTests`):

```python
    def test_compute_min_slack_subtracts_delta_safe(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(n_int=5, barrier_levels=[1.0])
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        result = BarrierPathResult(
            success=True, path_regions=[0], total_cost=0.0, solve_time=0.0,
            solver_status="ok",
            entry_states={0: np.array([0.5, 0.5, 0.0])},
            control_params={0: np.zeros(dynamics.n_u * solver.control_param.n_segments)},
            time_durations={0: 1.0},
        )
        slack_with_margin = solver._compute_min_slack(result, delta_safe=0.05)
        slack_without_margin = solver._compute_min_slack(result, delta_safe=0.0)
        self.assertAlmostEqual(slack_without_margin - slack_with_margin, 0.05, places=9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_barrier_dms.py::BarrierDMSSolverTests::test_compute_min_slack_subtracts_delta_safe -v`
Expected: FAIL with `TypeError: _compute_min_slack() got an unexpected keyword argument 'delta_safe'`

- [ ] **Step 3: Update `_compute_min_slack`**

In `demo/barrier_dms.py`, replace:

```python
    def _compute_min_slack(self, result: BarrierPathResult) -> float:
        min_slack = float('inf')
        for seg_idx, region_idx in enumerate(result.path_regions):
            if (
                region_idx not in result.entry_states
                or region_idx not in result.control_params
                or region_idx not in result.time_durations
            ):
                continue
            region = self.graph._regions_by_index[region_idx]
            x_k = np.asarray(result.entry_states[region_idx], dtype=float).copy()
            w = np.asarray(result.control_params[region_idx], dtype=float)
            delta = float(result.time_durations[region_idx])
            dt = 1.0 / self.config.n_int

            for k in range(self.config.n_int + 1):
                pos = self.dynamics.project_to_position(x_k)
                slacks = region.b - region.A @ pos
                min_slack = min(min_slack, float(np.min(slacks)))
```

with:

```python
    def _compute_min_slack(self, result: BarrierPathResult, delta_safe: float) -> float:
        min_slack = float('inf')
        for seg_idx, region_idx in enumerate(result.path_regions):
            if (
                region_idx not in result.entry_states
                or region_idx not in result.control_params
                or region_idx not in result.time_durations
            ):
                continue
            region = self.graph._regions_by_index[region_idx]
            x_k = np.asarray(result.entry_states[region_idx], dtype=float).copy()
            w = np.asarray(result.control_params[region_idx], dtype=float)
            delta = float(result.time_durations[region_idx])
            dt = 1.0 / self.config.n_int

            for k in range(self.config.n_int + 1):
                pos = self.dynamics.project_to_position(x_k)
                # Spec Sec 2.1/8.1: s_ijk = b - delta_safe - A @ pos.
                slacks = region.b - delta_safe - region.A @ pos
                min_slack = min(min_slack, float(np.min(slacks)))
```

(the rest of the method body — the RK4 stepping loop and final return — is unchanged).

- [ ] **Step 4: Confirm the file still parses**

Run: `python -c "import ast; ast.parse(open('demo/barrier_dms.py').read())"`
Expected: no output (valid syntax). Note: `solve()` still calls `_compute_min_slack(best_result)` with one positional arg at this point, which will raise `TypeError` if `solve()` is run end-to-end — that call site is fixed in Task 3 Step 5. Do not run `solve()` end-to-end until then; the unit test below calls `_compute_min_slack` directly and is unaffected.

- [ ] **Step 5: Run the unit test to verify it passes**

Run: `python -m pytest tests/test_barrier_dms.py::BarrierDMSSolverTests::test_compute_min_slack_subtracts_delta_safe -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_barrier_dms.py demo/barrier_dms.py
git commit -m "Subtract delta_safe in _compute_min_slack to match spec Sec 2.1/8.1"
```

---

### Task 3: Defect norm measurement + certified-margin formula (F1, F6)

**Why:** `defect_norm` is never computed for CRD (always the dataclass default `0.0`), and `certified_safety_margin` is computed as `min_slack - lip_gap`, omitting the `ε_defect` and `ε_int` terms required by spec §8.2:
`s_min_certified = s_min_sampled - L_s·h_max/2 - ε_defect - ε_int`.
This task adds a real defect measurement (the coupling+boundary residual at the solution, per audit F1) and uses it as `ε_defect`, plus uses Task 1's `f_lipschitz_bound()` for `ε_int`. This also finishes wiring `_compute_min_slack`'s new `delta_safe` parameter from Task 2.

**Files:**
- Modify: `demo/barrier_dms.py` — add `defect_norm` and `n_nlp_iterations` fields to `BarrierPathResult`; add `_compute_defect_norm` method; rewrite the post-processing block in `solve()`.
- Test: `tests/test_barrier_dms.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_barrier_dms.py` (inside `BarrierDMSSolverTests`):

```python
    def test_defect_norm_is_small_and_positive(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(
            n_int=20, barrier_levels=[1.0, 0.5, 0.1, 0.01], time_limit_s=30.0,
        )
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        result = solver.solve(
            [0, 1], anchor_points,
            start_state=np.array([0.1, 0.5, 0.0]),
            goal_state=np.array([1.9, 0.5, 0.0]),
        )
        self.assertTrue(result.success)
        self.assertGreater(result.defect_norm, 0.0)
        self.assertLess(result.defect_norm, cfg.epsilon_final * 10)

    def test_certified_margin_accounts_for_defect_and_integration_error(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(
            n_int=20, barrier_levels=[1.0, 0.5, 0.1, 0.01], time_limit_s=30.0,
        )
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        result = solver.solve(
            [0, 1], anchor_points,
            start_state=np.array([0.1, 0.5, 0.0]),
            goal_state=np.array([1.9, 0.5, 0.0]),
        )
        self.assertTrue(result.success)
        self.assertLessEqual(
            result.certified_safety_margin,
            result.min_safety_margin - result.lipschitz_gap,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_barrier_dms.py -k "defect_norm or certified_margin" -v`
Expected: FAIL — `AttributeError: 'BarrierPathResult' object has no attribute 'defect_norm'` (and/or `TypeError` from `_compute_min_slack`'s now-required `delta_safe` argument, left dangling since Task 2).

- [ ] **Step 3: Add fields to `BarrierPathResult`**

In `demo/barrier_dms.py`, in the `BarrierPathResult` dataclass, change:

```python
@dataclass
class BarrierPathResult:
    success: bool
    path_regions: List[int]
    total_cost: float
    solve_time: float
    solver_status: str
    min_safety_margin: float = float('nan')
    certified_safety_margin: float = float('nan')
    lipschitz_gap: float = float('nan')
    safety_certification: str = "NOT_SET"
    entry_states: Dict[int, np.ndarray] = field(default_factory=dict)
    exit_states: Dict[int, np.ndarray] = field(default_factory=dict)
    control_params: Dict[int, np.ndarray] = field(default_factory=dict)
    time_durations: Dict[int, float] = field(default_factory=dict)
    barrier_level_results: List[Dict] = field(default_factory=list)
```

to:

```python
@dataclass
class BarrierPathResult:
    success: bool
    path_regions: List[int]
    total_cost: float
    solve_time: float
    solver_status: str
    min_safety_margin: float = float('nan')
    certified_safety_margin: float = float('nan')
    lipschitz_gap: float = float('nan')
    safety_certification: str = "NOT_SET"
    defect_norm: float = float('nan')
    n_nlp_iterations: int = 0
    entry_states: Dict[int, np.ndarray] = field(default_factory=dict)
    exit_states: Dict[int, np.ndarray] = field(default_factory=dict)
    control_params: Dict[int, np.ndarray] = field(default_factory=dict)
    time_durations: Dict[int, float] = field(default_factory=dict)
    barrier_level_results: List[Dict] = field(default_factory=list)
```

- [ ] **Step 4: Add `_compute_defect_norm`**

In `demo/barrier_dms.py`, immediately after `_compute_min_slack` (the method whose body you edited in Task 2), add a new method:

```python

    def _compute_defect_norm(
        self,
        result: BarrierPathResult,
        start_state: Optional[np.ndarray],
        goal_state: Optional[np.ndarray],
    ) -> float:
        """
        Inf-norm of the coupling + boundary residuals at the NLP solution
        (spec Sec 6 Tier 1 / Part 9 Stage F: defect_norm, coupling_gap).
        These are hard equalities inside the NLP (R1) -- this only measures
        the residual at the reported solution, it does not change them.
        """
        path_regions = result.path_regions
        if not path_regions:
            return 0.0

        gaps = []
        if start_state is not None and path_regions[0] in result.entry_states:
            gaps.append(
                float(np.max(np.abs(
                    result.entry_states[path_regions[0]] - np.asarray(start_state, dtype=float)
                )))
            )
        if goal_state is not None and path_regions[-1] in result.exit_states:
            gaps.append(
                float(np.max(np.abs(
                    result.exit_states[path_regions[-1]] - np.asarray(goal_state, dtype=float)
                )))
            )
        for left_region, right_region in zip(path_regions[:-1], path_regions[1:]):
            if left_region in result.exit_states and right_region in result.entry_states:
                gaps.append(
                    float(np.max(np.abs(
                        result.exit_states[left_region] - result.entry_states[right_region]
                    )))
                )

        return float(max(gaps)) if gaps else 0.0
```

- [ ] **Step 5: Rewrite the post-processing block in `solve()`**

In `demo/barrier_dms.py`, inside `solve()`, replace:

```python
        min_slack = self._compute_min_slack(best_result)
        optimized_delta_arr = np.array(
            [best_result.time_durations[ri] for ri in path_regions], dtype=float
        )
        lip_gap = compute_lipschitz_safety_gap(
            self.dynamics, path_regions, self.graph,
            optimized_delta_arr, cfg.n_int,
        )
        best_result.lipschitz_gap = lip_gap
        best_result.min_safety_margin = float(min_slack)
        best_result.certified_safety_margin = float(min_slack - lip_gap)
        best_result.safety_certification = (
            "CERTIFIED" if best_result.certified_safety_margin > 0 else "NOT_CERTIFIED"
        )
```

with:

```python
        min_slack = self._compute_min_slack(best_result, delta_safe)
        optimized_delta_arr = np.array(
            [best_result.time_durations[ri] for ri in path_regions], dtype=float
        )
        lip_gap = compute_lipschitz_safety_gap(
            self.dynamics, path_regions, self.graph,
            optimized_delta_arr, cfg.n_int,
        )
        # Spec Sec 8.2: s_min_certified = s_min_sampled - L_s*h_max/2
        #                                 - epsilon_defect - epsilon_int.
        h_max = float(np.max(optimized_delta_arr)) / cfg.n_int
        defect_norm = self._compute_defect_norm(best_result, start_state, goal_state)
        eps_int = self.dynamics.f_lipschitz_bound() * (h_max ** 4) / 30.0
        best_result.defect_norm = defect_norm
        best_result.lipschitz_gap = lip_gap
        best_result.min_safety_margin = float(min_slack)
        best_result.certified_safety_margin = float(
            min_slack - lip_gap - defect_norm - eps_int
        )
        best_result.safety_certification = (
            "CERTIFIED" if best_result.certified_safety_margin > 0 else "NOT_CERTIFIED"
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_barrier_dms.py -v`
Expected: PASS (all tests, including the two new ones and the Task 2 test)

- [ ] **Step 7: Commit**

```bash
git add tests/test_barrier_dms.py demo/barrier_dms.py
git commit -m "Measure defect_norm and fix certified_safety_margin formula per spec Sec 8.2"
```

---

### Task 4: Report UB without the barrier term (F7)

**Why:** Spec §7.2 / Part 9 pseudocode line `J_star ← evaluate_cost(x_w, u_w, Δ_w, cfg, μ=0)`: the upper bound (and `total_cost`) must be the objective *without* the log-barrier term. Currently `total_cost = float(sol['f'])`, which includes `-μ·Σ h_k log(s)` — at `μ=0.01` this is not negligible and has a sign that depends on whether slacks are above or below 1. This also gives F4's `optimality_gap` a well-defined UB.

**Files:**
- Modify: `demo/barrier_dms.py` — `_build_nlp_symbols` (split `obj` into `obj_base` / `barrier_term`), `_solve_barrier_level` (build NLP from `obj_base + barrier_term`, but report `total_cost` from `obj_base` evaluated at the solution).
- Test: `tests/test_barrier_dms.py`

- [ ] **Step 1: Write the failing test**

Add near the top of `tests/test_barrier_dms.py` (with the other imports):

```python
import casadi as ca
from constraint_layers import compute_lipschitz_safety_gap
```

Add to `BarrierDMSSolverTests`:

```python
    def test_build_nlp_symbols_splits_barrier_from_base_cost(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(n_int=20, barrier_levels=[1.0])
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        delta_arr = np.array([1.0, 1.0])
        lip_gap = compute_lipschitz_safety_gap(dynamics, [0, 1], graph, delta_arr, cfg.n_int)
        delta_safe = max(cfg.delta_safe, lip_gap + cfg.epsilon_final)
        node_delta_safe = np.full((2, cfg.n_int + 1), delta_safe)

        mu = 1.0
        out1 = solver._build_nlp_symbols(
            [0, 1], anchor_points, delta_safe, mu, node_delta_safe=node_delta_safe,
        )
        x_sym, obj_base, barrier_term = out1[0], out1[1], out1[2]
        x0 = out1[7]

        out2 = solver._build_nlp_symbols(
            [0, 1], anchor_points, delta_safe, 2 * mu, node_delta_safe=node_delta_safe,
        )
        obj_base2, barrier_term2 = out2[1], out2[2]

        base_fn = ca.Function('base', [x_sym], [obj_base])
        barrier_fn = ca.Function('barrier', [x_sym], [barrier_term])
        base_fn2 = ca.Function('base2', [x_sym], [obj_base2])
        barrier_fn2 = ca.Function('barrier2', [x_sym], [barrier_term2])

        base_val = float(base_fn(x0))
        barrier_val = float(barrier_fn(x0))

        self.assertNotEqual(barrier_val, 0.0)
        # obj_base must not depend on mu; barrier_term must scale linearly with mu.
        self.assertAlmostEqual(float(base_fn2(x0)), base_val, places=8)
        self.assertAlmostEqual(float(barrier_fn2(x0)), 2 * barrier_val, places=8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_barrier_dms.py::BarrierDMSSolverTests::test_build_nlp_symbols_splits_barrier_from_base_cost -v`
Expected: FAIL — `_build_nlp_symbols` currently returns a single combined `obj`, so `out1[1]` is not a barrier-free cost and `out1[2]` is `g_sym`, not `barrier_term` (shape/type mismatch will raise, e.g. when constructing `ca.Function` with a non-scalar second argument, or the mu-scaling assertions will fail).

- [ ] **Step 3: Split the objective in `_build_nlp_symbols`**

In `demo/barrier_dms.py`, change the objective initialization (currently `obj = ca.MX(0.0)` right before the `for i in range(m):` loop) to:

```python
        obj_base = ca.MX(0.0)
        barrier_term = ca.MX(0.0)
```

Then, throughout the rest of the method, replace every `obj = obj + ...` / `obj = (obj + ...)` that is **not** the log-barrier accumulation with `obj_base = obj_base + ...` (these are: the `w_L`/`w_U` running-cost terms, the `w_T * dv` duration term, the two `w_S * ca.dot(...)` smoothness terms — one for in-segment control jump, one for inter-segment control jump). Concretely:

- `obj = (obj + h_k * cfg.w_L * ca.dot(vel_k, vel_k) + h_k * cfg.w_U * ca.dot(u_k, u_k))` → `obj_base = (obj_base + h_k * cfg.w_L * ca.dot(vel_k, vel_k) + h_k * cfg.w_U * ca.dot(u_k, u_k))`
- `obj = (obj + h_end * cfg.w_L * ca.dot(vel_end, vel_end) + h_end * cfg.w_U * ca.dot(u_end, u_end))` → same rename to `obj_base`
- `obj = obj + cfg.w_T * dv` → `obj_base = obj_base + cfg.w_T * dv`
- `obj = obj + cfg.w_S * ca.dot(u_right - u_left, u_right - u_left)` → `obj_base = obj_base + cfg.w_S * ca.dot(u_right - u_left, u_right - u_left)`
- `obj = obj + cfg.w_S * ca.dot(control_jump, control_jump)` → `obj_base = obj_base + cfg.w_S * ca.dot(control_jump, control_jump)`

Leave the log-barrier accumulation (`obj = (obj - mu * cfg.mu_weight * h_k * ca.sum1(ca.log(slack)))`) as the only place writing to `barrier_term`, renamed to:

```python
                barrier_term = (
                    barrier_term
                    - mu * cfg.mu_weight * h_k * ca.sum1(ca.log(slack))
                )
```

Finally, change the return statement from:

```python
        x_sym = ca.vertcat(*x_sym_list)
        g_sym = ca.vertcat(*g_list) if g_list else ca.MX(0, 1)

        return (x_sym, obj, g_sym, lbx, ubx, lbg, ubg, x0_list,
                s_minus_vars, w_vars, delta_vars, x_node_vars_list)
```

to:

```python
        x_sym = ca.vertcat(*x_sym_list)
        g_sym = ca.vertcat(*g_list) if g_list else ca.MX(0, 1)

        return (x_sym, obj_base, barrier_term, g_sym, lbx, ubx, lbg, ubg, x0_list,
                s_minus_vars, w_vars, delta_vars, x_node_vars_list)
```

- [ ] **Step 4: Update `_solve_barrier_level` to use the split objective and report the barrier-free cost**

In `demo/barrier_dms.py`, change:

```python
        (x_sym, obj, g_sym, lbx, ubx, lbg, ubg,
         x0_default, s_minus_vars, w_vars, delta_vars, x_node_vars_list) = self._build_nlp_symbols(
            path_regions, anchor_points, delta_safe, mu,
            start_state=start_state, goal_state=goal_state,
            node_delta_safe=node_delta_safe,
        )
```

to:

```python
        (x_sym, obj_base, barrier_term, g_sym, lbx, ubx, lbg, ubg,
         x0_default, s_minus_vars, w_vars, delta_vars, x_node_vars_list) = self._build_nlp_symbols(
            path_regions, anchor_points, delta_safe, mu,
            start_state=start_state, goal_state=goal_state,
            node_delta_safe=node_delta_safe,
        )
```

and change:

```python
        nlp = {'x': x_sym, 'f': obj, 'g': g_sym}
        opts = {
            'ipopt.max_iter': max_iter, 'ipopt.tol': tol,
            'ipopt.print_level': 0, 'print_time': 0,
        }
        solver = ca.nlpsol('barrier_dms', 'ipopt', nlp, opts)
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        stats = solver.stats()
        success = bool(stats.get('success', False))
        x_opt = np.array(sol['x']).flatten()

        m = len(path_regions)
        n_x = self.dynamics.n_x
        n_w = self.control_param.n_w
        vars_per_seg = n_x + n_w + 1

        result = BarrierPathResult(
            success=success, path_regions=path_regions, total_cost=float(sol['f']),
            solve_time=0.0, solver_status=stats.get('return_status', 'unknown'),
        )
```

to:

```python
        nlp = {'x': x_sym, 'f': obj_base + barrier_term, 'g': g_sym}
        opts = {
            'ipopt.max_iter': max_iter, 'ipopt.tol': tol,
            'ipopt.print_level': 0, 'print_time': 0,
        }
        solver = ca.nlpsol('barrier_dms', 'ipopt', nlp, opts)
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        stats = solver.stats()
        success = bool(stats.get('success', False))
        x_opt = np.array(sol['x']).flatten()

        # Spec Sec 7.2 / Part 9 Stage F: UB must be J|mu=0 (no barrier term).
        base_cost_fn = ca.Function('barrier_dms_base_cost', [x_sym], [obj_base])
        total_cost = float(base_cost_fn(x_opt))

        m = len(path_regions)
        n_x = self.dynamics.n_x
        n_w = self.control_param.n_w
        vars_per_seg = n_x + n_w + 1

        result = BarrierPathResult(
            success=success, path_regions=path_regions, total_cost=total_cost,
            solve_time=0.0, solver_status=stats.get('return_status', 'unknown'),
            n_nlp_iterations=int(stats.get('iter_count', 0)),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_barrier_dms.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add tests/test_barrier_dms.py demo/barrier_dms.py
git commit -m "Report total_cost without the log-barrier term (spec Sec 7.2 UB definition)"
```

---

### Task 5: Accumulate `n_nlp_iterations` across barrier levels (F8)

**Why:** Task 4 made `_solve_barrier_level` record iterations for *one* barrier level. The outer `solve()` continuation loop must sum these across all levels attempted for the winning path (Part 11 `crd_n_nlp_iterations`).

**Files:**
- Modify: `demo/barrier_dms.py` — `solve()` continuation loop.
- Test: `tests/test_barrier_dms.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_barrier_dms.py` (inside `BarrierDMSSolverTests`):

```python
    def test_n_nlp_iterations_recorded(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(
            n_int=20, barrier_levels=[1.0, 0.5, 0.1, 0.01], time_limit_s=30.0,
        )
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        result = solver.solve(
            [0, 1], anchor_points,
            start_state=np.array([0.1, 0.5, 0.0]),
            goal_state=np.array([1.9, 0.5, 0.0]),
        )
        self.assertTrue(result.success)
        self.assertGreater(result.n_nlp_iterations, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_barrier_dms.py::BarrierDMSSolverTests::test_n_nlp_iterations_recorded -v`
Expected: FAIL — `result.n_nlp_iterations == 0` (the per-level value from Task 4 is never copied onto the final `best_result`).

- [ ] **Step 3: Accumulate iterations in `solve()`**

In `demo/barrier_dms.py`, in `solve()`, change:

```python
        best_result = None
        level_results = []
        x_init = None
```

to:

```python
        best_result = None
        level_results = []
        x_init = None
        total_n_iter = 0
```

and change:

```python
            try:
                result, x_init = self._solve_barrier_level(
                    path_regions=path_regions,
                    anchor_points=anchor_points,
                    warm_start=warm_start,
                    delta_safe=delta_safe,
                    mu=mu,
                    x_init=x_init,
                    tol=tol,
                    start_state=start_state,
                    goal_state=goal_state,
                    node_delta_safe=node_delta_safe,
                )
                level_results.append({'mu': mu, 'success': result.success, 'cost': result.total_cost})
                if result.success:
                    best_result = result
            except Exception as e:
                level_results.append({'mu': mu, 'success': False, 'error': str(e)})
```

to:

```python
            try:
                result, x_init = self._solve_barrier_level(
                    path_regions=path_regions,
                    anchor_points=anchor_points,
                    warm_start=warm_start,
                    delta_safe=delta_safe,
                    mu=mu,
                    x_init=x_init,
                    tol=tol,
                    start_state=start_state,
                    goal_state=goal_state,
                    node_delta_safe=node_delta_safe,
                )
                total_n_iter += result.n_nlp_iterations
                level_results.append({'mu': mu, 'success': result.success, 'cost': result.total_cost})
                if result.success:
                    best_result = result
            except Exception as e:
                level_results.append({'mu': mu, 'success': False, 'error': str(e)})
```

Then, in the post-processing block you edited in Task 3 Step 5, add one line at the end so the final block reads:

```python
        best_result.defect_norm = defect_norm
        best_result.lipschitz_gap = lip_gap
        best_result.min_safety_margin = float(min_slack)
        best_result.certified_safety_margin = float(
            min_slack - lip_gap - defect_norm - eps_int
        )
        best_result.safety_certification = (
            "CERTIFIED" if best_result.certified_safety_margin > 0 else "NOT_CERTIFIED"
        )
        best_result.n_nlp_iterations = total_n_iter
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_barrier_dms.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_barrier_dms.py demo/barrier_dms.py
git commit -m "Accumulate n_nlp_iterations across barrier continuation levels"
```

---

### Task 6: Pure-geometric Dijkstra length on the centroid graph (F3 prerequisite)

**Why:** Spec §7.1: `LB_geom = D*_path / v_max`, where `D*_path` is the Dijkstra shortest polyline length using Euclidean edge weights on the centroid graph, **without** the composite-cost penalty (`-γ_w·ρ_interface`) that `add_composite_costs_to_graph` mixes in for path ranking. No such pure-geometric weight currently exists on the graph.

**Files:**
- Modify: `demo/graph_builder.py` — `add_composite_costs_to_graph` (store the raw Euclidean `dist` as a `geom_dist` edge attribute), add `dijkstra_geometric_length()`.
- Test: `tests/test_graph_builder.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_graph_builder.py`, change the import line `from graph_builder import SOURCE, TARGET, build_region_graph` to:

```python
from graph_builder import (
    SOURCE, TARGET, build_region_graph,
    add_composite_costs_to_graph, dijkstra_geometric_length,
)
```

Then add, at the end of the file:

```python
class DijkstraGeometricLengthTests(unittest.TestCase):
    def test_returns_pure_euclidean_path_length(self) -> None:
        graph = two_region_graph()
        add_composite_costs_to_graph(
            graph, start_pos=graph.start_pos, goal_pos=graph.goal_pos,
        )
        length = dijkstra_geometric_length(graph, SOURCE, TARGET)
        self.assertGreater(length, 0.0)
        direct = float(np.linalg.norm(graph.goal_pos - graph.start_pos))
        # Going through region centroids cannot be shorter than the
        # straight-line start-to-goal distance.
        self.assertGreaterEqual(length, direct - 1e-9)
```

(`two_region_graph()` is the existing helper at the top of the file; check it exposes `graph.start_pos`/`graph.goal_pos` on the returned `RegionGraph` — if those attribute names differ, use the same start/goal arrays already passed into `build_region_graph` inside `two_region_graph()` instead of reading them back off the graph object.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph_builder.py::DijkstraGeometricLengthTests -v`
Expected: FAIL with `ImportError: cannot import name 'dijkstra_geometric_length'`

- [ ] **Step 3: Add `geom_dist` edge attribute**

In `demo/graph_builder.py`, in `add_composite_costs_to_graph`, change:

```python
        dist = float(np.linalg.norm(p2 - p1))
        rho_interface = interface_radii.get((u, v), 0.0)
        cost = dist - gamma_w * rho_interface
        data['composite_cost'] = max(cost, 1e-9)
```

to:

```python
        dist = float(np.linalg.norm(p2 - p1))
        rho_interface = interface_radii.get((u, v), 0.0)
        cost = dist - gamma_w * rho_interface
        data['composite_cost'] = max(cost, 1e-9)
        # Pure-geometry weight for the Sec 7.1 lower bound -- no
        # composite-cost interface-radius credit.
        data['geom_dist'] = dist
```

- [ ] **Step 4: Add `dijkstra_geometric_length`**

In `demo/graph_builder.py`, immediately after `k_shortest_paths_generator` (which ends with `return nx.shortest_simple_paths(graph.graph, source, target, weight=weight)`), add:

```python


def dijkstra_geometric_length(
    graph: "RegionGraph",
    source: str,
    target: str,
) -> float:
    """
    Pure-geometry shortest path length on the centroid graph (spec Sec 7.1).
    Requires add_composite_costs_to_graph() to have been called first so
    edges carry the 'geom_dist' attribute.
    """
    return float(nx.shortest_path_length(graph.graph, source, target, weight='geom_dist'))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_graph_builder.py -v`
Expected: PASS (all tests, including the new one)

- [ ] **Step 6: Commit**

```bash
git add tests/test_graph_builder.py demo/graph_builder.py
git commit -m "Add dijkstra_geometric_length for the Sec 7.1 geometric lower bound"
```

---

### Task 7: Wire `defect_norm` / `max_connection_gap` into `CentroidRefineDMSSolver` (F1, F2)

**Why:** `OptimizationResult` for CRD never copies `defect_norm` from the (now-populated, Task 3) `BarrierPathResult`, and never calls the existing `_compute_connection_gap` helper (`optimizer.py:217`, already used by the legacy solvers) to populate `max_connection_gap`. Both currently report the dataclass default `0.0`.

**Files:**
- Modify: `demo/optimizer.py:2674-2692` (the `opt_result = OptimizationResult(...)` construction inside `CentroidRefineDMSSolver.solve()`)
- Test: `tests/test_centroid_refine_dms.py` (new file — also used by Tasks 8 and 9)

- [ ] **Step 1: Write the failing test**

Create `tests/test_centroid_refine_dms.py`:

```python
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
sys.path.insert(0, str(DEMO_DIR))

from convex_regions import create_regions_from_vertices_list
from graph_builder import build_region_graph
from dynamics import UnicycleModel
from optimizer import CentroidRefineDMSSolver, CentroidRefineDMSConfig


def _two_region_setup():
    regions = create_regions_from_vertices_list([
        np.array([[0, 0], [1.2, 0], [1.2, 1], [0, 1]], dtype=float),
        np.array([[0.8, 0], [2.0, 0], [2.0, 1], [0.8, 1]], dtype=float),
    ])
    start = np.array([0.1, 0.5])
    goal = np.array([1.9, 0.5])
    graph = build_region_graph(regions, start, goal)
    dynamics = UnicycleModel(v_max=2.0)
    return graph, dynamics


class CentroidRefineDMSDiagnosticsTests(unittest.TestCase):
    def _solve(self):
        graph, dynamics = _two_region_setup()
        cfg = CentroidRefineDMSConfig(
            n_int=20, barrier_levels=[1.0, 0.5, 0.1, 0.01], time_limit_s=30.0,
        )
        solver = CentroidRefineDMSSolver(graph, dynamics, cfg)
        start_state = np.array([0.1, 0.5, 0.0])
        goal_state = np.array([1.9, 0.5, 0.0])
        return solver.solve(start_state, goal_state)

    def test_defect_and_connection_gap_are_measured(self) -> None:
        result = self._solve()
        self.assertTrue(result.success)
        self.assertGreater(result.defect_norm, 0.0)
        self.assertLess(result.max_connection_gap, 1e-3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_centroid_refine_dms.py -v`
Expected: FAIL — `result.defect_norm == 0.0` (never wired) even though the underlying `BarrierPathResult` already has a real value after Task 3.

- [ ] **Step 3: Wire the fields**

In `demo/optimizer.py`, in `CentroidRefineDMSSolver.solve()`, change:

```python
            opt_result = OptimizationResult(
                success=barrier_result.success, path=path, path_regions=path_regions,
                total_cost=barrier_result.total_cost, solve_time=_time.time() - start_clock,
                n_paths_evaluated=n_evaluated, solver_status=barrier_result.solver_status,
                formulation_mode="centroid_refine_dms",
                safety_mode="log_barrier",
                global_optimality_claim=_KKT_DISCLAIMER,
                min_safety_margin=barrier_result.min_safety_margin,
                certified_safety_margin=barrier_result.certified_safety_margin,
                lipschitz_gap=barrier_result.lipschitz_gap,
                safety_certification=barrier_result.safety_certification,
                n_barrier_levels=len(barrier_result.barrier_level_results),
                failure_log=failure_log,
                entry_states=barrier_result.entry_states,
                exit_states=barrier_result.exit_states,
                control_params=barrier_result.control_params,
                time_durations=barrier_result.time_durations,
                pipeline_mode="centroid_refine_dms",
            )
```

to:

```python
            opt_result = OptimizationResult(
                success=barrier_result.success, path=path, path_regions=path_regions,
                total_cost=barrier_result.total_cost, solve_time=_time.time() - start_clock,
                n_paths_evaluated=n_evaluated, solver_status=barrier_result.solver_status,
                formulation_mode="centroid_refine_dms",
                safety_mode="log_barrier",
                global_optimality_claim=_KKT_DISCLAIMER,
                min_safety_margin=barrier_result.min_safety_margin,
                certified_safety_margin=barrier_result.certified_safety_margin,
                lipschitz_gap=barrier_result.lipschitz_gap,
                safety_certification=barrier_result.safety_certification,
                n_barrier_levels=len(barrier_result.barrier_level_results),
                failure_log=failure_log,
                entry_states=barrier_result.entry_states,
                exit_states=barrier_result.exit_states,
                control_params=barrier_result.control_params,
                time_durations=barrier_result.time_durations,
                pipeline_mode="centroid_refine_dms",
                defect_norm=barrier_result.defect_norm,
                max_connection_gap=_compute_connection_gap(
                    path_regions, barrier_result.entry_states, barrier_result.exit_states,
                    start_state, goal_state,
                ),
                n_nlp_iterations=barrier_result.n_nlp_iterations,
            )
```

(`_compute_connection_gap` is the module-level helper already defined at `optimizer.py:217`; `start_state`/`goal_state` are the parameters of the enclosing `solve()` method.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_centroid_refine_dms.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_centroid_refine_dms.py demo/optimizer.py
git commit -m "Wire defect_norm/max_connection_gap/n_nlp_iterations into CRD OptimizationResult"
```

---

### Task 8: `lb_geometric` and `optimality_gap` (F3, F4)

**Why:** Both fields are spec-required (§7.1, §7.2, Part 9 post-processing) and currently always default to `0.0` / `inf`. With Task 6's `dijkstra_geometric_length` and Task 4's barrier-free `total_cost` (UB), both can now be computed honestly.

**Files:**
- Modify: `demo/optimizer.py` — `CentroidRefineDMSSolver.solve()`: compute `lb_geometric` once after Stage A graph annotation, attach it to the `NO_PATH_FOUND` early return, and set both fields (plus `optimality_gap`) on the final `best_result` before returning.
- Test: `tests/test_centroid_refine_dms.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_centroid_refine_dms.py` (inside `CentroidRefineDMSDiagnosticsTests`):

```python
    def test_lower_bound_and_optimality_gap_are_populated(self) -> None:
        result = self._solve()
        self.assertTrue(result.success)
        self.assertGreater(result.lb_geometric, 0.0)
        self.assertTrue(np.isfinite(result.optimality_gap))
        self.assertGreaterEqual(result.optimality_gap, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_centroid_refine_dms.py::CentroidRefineDMSDiagnosticsTests::test_lower_bound_and_optimality_gap_are_populated -v`
Expected: FAIL — `result.lb_geometric == 0.0`, `result.optimality_gap == inf`

- [ ] **Step 3: Compute `lb_geometric` once, right after the `has_path` check**

In `demo/optimizer.py`, in `CentroidRefineDMSSolver.solve()`, change the import line:

```python
        from graph_builder import add_composite_costs_to_graph, k_shortest_paths_generator, SOURCE, TARGET
```

to:

```python
        from graph_builder import (
            add_composite_costs_to_graph, k_shortest_paths_generator,
            dijkstra_geometric_length, SOURCE, TARGET,
        )
```

Then, change:

```python
        import networkx as _nx
        if not _nx.has_path(self.graph.graph, SOURCE, TARGET):
            return OptimizationResult(
                success=False, path=[], path_regions=[], total_cost=float('inf'),
                solve_time=_time.time() - start_clock, n_paths_evaluated=0,
                solver_status="CentroidRefineDMS: graph has no path from source to target",
                formulation_mode="centroid_refine_dms",
                safety_mode="log_barrier",
            )

        gen = k_shortest_paths_generator(self.graph, SOURCE, TARGET)
```

to:

```python
        import networkx as _nx
        if not _nx.has_path(self.graph.graph, SOURCE, TARGET):
            return OptimizationResult(
                success=False, path=[], path_regions=[], total_cost=float('inf'),
                solve_time=_time.time() - start_clock, n_paths_evaluated=0,
                solver_status="CentroidRefineDMS: graph has no path from source to target",
                formulation_mode="centroid_refine_dms",
                safety_mode="log_barrier",
            )

        # Spec Sec 7.1: LB_geom = D*_path / v_max, pure-geometry Dijkstra on
        # the centroid graph -- computed once, independent of path candidate.
        d_star = dijkstra_geometric_length(self.graph, SOURCE, TARGET)
        lb_geometric = d_star / self.dynamics.v_max

        gen = k_shortest_paths_generator(self.graph, SOURCE, TARGET)
```

- [ ] **Step 4: Attach `lb_geometric` to the `NO_PATH_FOUND` return**

In `demo/optimizer.py`, change:

```python
        if best_result is None:
            return OptimizationResult(
                success=False, path=[], path_regions=[], total_cost=float('inf'),
                solve_time=_time.time() - start_clock, n_paths_evaluated=n_evaluated,
                solver_status="CentroidRefineDMS: no feasible path found",
                formulation_mode="centroid_refine_dms", failure_log=failure_log,
                safety_mode="log_barrier",
                global_optimality_claim=_KKT_DISCLAIMER,
            )
```

to:

```python
        if best_result is None:
            return OptimizationResult(
                success=False, path=[], path_regions=[], total_cost=float('inf'),
                solve_time=_time.time() - start_clock, n_paths_evaluated=n_evaluated,
                solver_status="CentroidRefineDMS: no feasible path found",
                formulation_mode="centroid_refine_dms", failure_log=failure_log,
                safety_mode="log_barrier",
                global_optimality_claim=_KKT_DISCLAIMER,
                lb_geometric=lb_geometric,
            )
```

- [ ] **Step 5: Set `lb_geometric`/`optimality_gap` on the final `best_result`**

In `demo/optimizer.py`, the method currently ends with:

```python
        if best_result.success and best_result.entry_states:
            control_param = ControlParameterization(
                n_segments=cfg.n_control_segments,
                n_u=self.dynamics.n_u,
                parameterization="piecewise_constant",
            )
            integrator = RK4Integrator(self.dynamics, control_param, cfg.n_int)
            for ri in best_result.path_regions:
                s0 = best_result.entry_states[ri]
                w = best_result.control_params[ri]
                delta = best_result.time_durations[ri]
                traj, tau = integrator.integrate_with_trajectory(s0, w, delta)
                best_result.trajectories.append((traj, tau, delta))
            for ri in best_result.path_regions[:-1]:
                s_exit = best_result.exit_states.get(ri)
                if s_exit is not None:
                    best_result.interface_points.append(
                        self.dynamics.project_to_position(s_exit)
                        if hasattr(self.dynamics, "project_to_position") else s_exit[:2]
                    )

        return best_result
```

Change the final lines (after the `interface_points.append(...)` block) to:

```python
                    best_result.interface_points.append(
                        self.dynamics.project_to_position(s_exit)
                        if hasattr(self.dynamics, "project_to_position") else s_exit[:2]
                    )

        # Spec Sec 7.1/7.2 + Part 9 post-processing.
        best_result.lb_geometric = lb_geometric
        denom = max(1.0, abs(best_result.total_cost))
        best_result.optimality_gap = (best_result.total_cost - cfg.w_T * lb_geometric) / denom

        return best_result
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_centroid_refine_dms.py -v`
Expected: PASS (all tests)

- [ ] **Step 7: Commit**

```bash
git add tests/test_centroid_refine_dms.py demo/optimizer.py
git commit -m "Populate lb_geometric and optimality_gap per spec Sec 7.1/7.2"
```

---

### Task 9: Mandatory disclaimer string (F9)

**Why:** Spec §7.3 mandates an exact 4-sentence disclaimer string for `global_optimality_claim`. The code currently uses an abridged, non-conforming string (`"KKT-feasible under LICQ+SOSC; no global optimality certificate."`).

**Files:**
- Modify: `demo/optimizer.py:2553-2555`
- Test: `tests/test_centroid_refine_dms.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_centroid_refine_dms.py` (inside `CentroidRefineDMSDiagnosticsTests`):

```python
    def test_global_optimality_claim_matches_mandatory_disclaimer(self) -> None:
        result = self._solve()
        expected = (
            "LB is a geometric lower bound on the time component only. "
            "gap_k is NOT a certificate for the full DMS objective. "
            "No global optimality claim is made. The reported solution is a "
            "KKT point of the fixed-path barrier-augmented NLP under LICQ + SOSC."
        )
        self.assertEqual(result.global_optimality_claim, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_centroid_refine_dms.py::CentroidRefineDMSDiagnosticsTests::test_global_optimality_claim_matches_mandatory_disclaimer -v`
Expected: FAIL — actual string is `"KKT-feasible under LICQ+SOSC; no global optimality certificate."`

- [ ] **Step 3: Replace the disclaimer string**

In `demo/optimizer.py`, change:

```python
        _KKT_DISCLAIMER = (
            "KKT-feasible under LICQ+SOSC; no global optimality certificate."
        )
```

to:

```python
        _KKT_DISCLAIMER = (
            "LB is a geometric lower bound on the time component only. "
            "gap_k is NOT a certificate for the full DMS objective. "
            "No global optimality claim is made. The reported solution is a "
            "KKT point of the fixed-path barrier-augmented NLP under LICQ + SOSC."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_centroid_refine_dms.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_centroid_refine_dms.py demo/optimizer.py
git commit -m "Use the exact spec Sec 7.3 mandatory disclaimer text"
```

---

### Task 10: Label CTCS/Dense fields as not-applicable for `log_barrier` mode (UI step, F10 follow-through)

**Why:** `visualization.py` already branches correctly on `safety_mode` (confirmed not broken). `experiments.py`'s `ExperimentResult.summary()` prints `Max CTCS Integral` / `Max Dense Region Violation` unconditionally — for CRD's `"log_barrier"` mode these fields are intentionally never populated (spec §8.3), so printing `0.00e+00` falsely implies "measured zero violation" instead of "not measured in this mode."

**Files:**
- Modify: `demo/experiments.py:30-65` (`ExperimentResult.summary()`)
- Test: manual verification (this is a string-formatting change in a human-readable report; no existing test covers `summary()`, and adding a full `ExperimentResult` fixture is disproportionate to a label change — verify by running the snippet in Step 2).

- [ ] **Step 1: Edit `summary()` to branch on `safety_mode`**

In `demo/experiments.py`, change:

```python
    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "===========================================",
            f"  Scenario: {self.scenario_name}",
            "===========================================",
            f"  Regions: {self.n_regions}",
            f"  Edges: {self.n_edges}",
            f"  Candidate Paths: {self.n_paths}",
            f"  Setup Time: {self.setup_time:.3f} s",
            "",
            f"  Optimization Status: {'SUCCESS' if self.optimization_result.success else 'FAILED'}",
            f"  Solver: {self.optimization_result.solver_status}",
            f"  Total Cost: {self.optimization_result.total_cost:.4f}",
            f"  Solve Time: {self.optimization_result.solve_time:.3f} s",
            f"  Paths Evaluated: {self.optimization_result.n_paths_evaluated}",
            f"  Safety Mode: {self.optimization_result.safety_mode}",
            f"  Max CTCS Integral: {self.optimization_result.max_continuous_violation_integral:.2e}",
            f"  Max Dense Region Violation: {self.optimization_result.max_dense_region_violation:.2e}",
        ]
        
        if self.optimization_result.success:
            lines.extend([
                "",
                f"  Path: {' -> '.join(self.optimization_result.path)}",
                f"  Total Duration: {sum(self.optimization_result.time_durations.values()):.3f} s",
                f"  Defect Norm: {self.optimization_result.defect_norm:.2e}",
                f"  Max Connection Gap: {self.optimization_result.max_connection_gap:.2e}",
                f"  Max Control Jump: {self.optimization_result.max_control_jump:.2e}",
            ])
            if self.optimization_result.max_integrality_gap > 0.0:
                lines.append(
                    f"  Max Integrality Gap: {self.optimization_result.max_integrality_gap:.2e}"
                )
        
        return "\n".join(lines)
```

to:

```python
    def summary(self) -> str:
        """Return human-readable summary."""
        result = self.optimization_result
        lines = [
            "===========================================",
            f"  Scenario: {self.scenario_name}",
            "===========================================",
            f"  Regions: {self.n_regions}",
            f"  Edges: {self.n_edges}",
            f"  Candidate Paths: {self.n_paths}",
            f"  Setup Time: {self.setup_time:.3f} s",
            "",
            f"  Optimization Status: {'SUCCESS' if result.success else 'FAILED'}",
            f"  Solver: {result.solver_status}",
            f"  Total Cost: {result.total_cost:.4f}",
            f"  Solve Time: {result.solve_time:.3f} s",
            f"  Paths Evaluated: {result.n_paths_evaluated}",
            f"  Safety Mode: {result.safety_mode}",
        ]
        if result.safety_mode == "both":
            lines.extend([
                f"  Max CTCS Integral: {result.max_continuous_violation_integral:.2e}",
                f"  Max Dense Region Violation: {result.max_dense_region_violation:.2e}",
            ])
        else:
            # Spec Sec 8.3: CTCS/dense fields are not populated outside
            # safety_mode="both" -- "n/a" avoids implying a measured zero.
            lines.append(f"  Max CTCS Integral: n/a ({result.safety_mode} mode)")
            lines.append(f"  Max Dense Region Violation: n/a ({result.safety_mode} mode)")

        if result.success:
            lines.extend([
                "",
                f"  Path: {' -> '.join(result.path)}",
                f"  Total Duration: {sum(result.time_durations.values()):.3f} s",
                f"  Defect Norm: {result.defect_norm:.2e}",
                f"  Max Connection Gap: {result.max_connection_gap:.2e}",
                f"  Max Control Jump: {result.max_control_jump:.2e}",
            ])
            if result.max_integrality_gap > 0.0:
                lines.append(
                    f"  Max Integrality Gap: {result.max_integrality_gap:.2e}"
                )

        return "\n".join(lines)
```

- [ ] **Step 2: Manually verify the formatting**

Run:

```bash
python - <<'EOF'
import sys
sys.path.insert(0, "demo")
from experiments import ExperimentResult
from optimizer import OptimizationResult

result = OptimizationResult(
    success=True, path=["s0", "t"], path_regions=[0], total_cost=1.0,
    solve_time=0.1, n_paths_evaluated=1, safety_mode="log_barrier",
)
exp = ExperimentResult(
    scenario_name="demo", optimization_result=result,
    n_regions=1, n_edges=1, n_paths=1, setup_time=0.01,
)
print(exp.summary())
EOF
```

Expected output includes the lines `Max CTCS Integral: n/a (log_barrier mode)` and `Max Dense Region Violation: n/a (log_barrier mode)`, and does NOT print `0.00e+00` for either.

- [ ] **Step 3: Run the full test suite to confirm no regressions**

Run: `python -m pytest tests/ -v`
Expected: PASS (all tests across the repo, including all tests added in Tasks 1–9)

- [ ] **Step 4: Commit**

```bash
git add demo/experiments.py
git commit -m "Label CTCS/Dense fields n/a outside safety_mode=both instead of printing 0.00e+00"
```

---

### Task 11: End-to-end sanity check against the default benchmark scenario

**Why:** `demo/test_crd_default.py` (referenced by the audit's "Kiểm chứng" section, and already the benchmark per commit `a846c2b`) exercises the full pipeline against the real default map/config. This is the final check that nothing upstream broke and that the audit's acceptance criteria hold end-to-end, not just in synthetic two-region unit tests.

**Files:**
- No code changes — verification only.

- [ ] **Step 1: Run the default-scenario smoke test**

Run: `cd demo && python test_crd_default.py`
Expected: exits without exception; log line `RESULT SUMMARY` printed; check `/tmp/crd_debug.log` if anything looks off.

- [ ] **Step 2: Inspect the printed diagnostics against the audit's acceptance criteria**

From the command output, confirm:
- `Safety: sampled=..., certified=...` — `certified` should now be `<= sampled - lipschitz_gap` (strictly less unless defect/eps_int are exactly 0, which they won't be).
- `Safety certification:` may now print `NOT_CERTIFIED` even though it printed `CERTIFIED` before this plan's changes — per the audit, **this is the correct, intended behavior change**, not a regression.
- (Optional manual check) temporarily add `print(f"defect_norm={result.defect_norm:.2e}")` after `result = prepared.optimizer.solve(...)` in `demo/test_crd_default.py` to see the raw value, then remove it — this is a manual inspection only, not a change to commit.

- [ ] **Step 3: Run the entire test suite one more time**

Run: `python -m pytest tests/ -v`
Expected: PASS

No commit for this task (verification only).

---

## Self-Review Notes

- **Spec coverage:** F1 (Task 3/7), F2 (Task 7), F3 (Task 6/8), F4 (Task 8), F5 (Task 2), F6 (Task 3), F7 (Task 4), F8 (Task 4/5), F9 (Task 9), F10/UI (Task 10). All nine audit findings plus the UI follow-through have a task.
- **R1 (hard equality, no penalty conversion):** Untouched — `_compute_defect_norm` only reads `entry_states`/`exit_states` after the solve; the NLP's `g_list` coupling/boundary constraints in `_build_nlp_symbols` are not modified.
- **R2 (barrier only on halfplane):** Untouched — Task 4 only renames/splits the *return value* of the barrier accumulation, the constraint structure and `lbx=[1e-10]` bound on slack variables are unchanged.
- **R13 (`δ_safe` enforcement):** Untouched — `delta_safe = max(cfg.delta_safe, lip_gap + cfg.epsilon_final)` at `barrier_dms.py:87` is not modified by any task; Task 2 only changes the separate *reporting* function `_compute_min_slack`.
- **R5 (always populate both margins):** Preserved — both `min_safety_margin` and `certified_safety_margin` are still set unconditionally in the Task 3 rewrite.
- **R4 (exact disclaimer):** Task 9.
