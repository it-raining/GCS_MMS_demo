# CRD Certificate-Fail Margin Relaxation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `CentroidRefineDMSSolver` reporting `CERTIFICATE_FAIL`/`NOT_CERTIFIED` on converged, non-crashing solves by embedding the Lipschitz-gap and RK4-truncation-error terms as live CasADi expressions of each region's own `Δ_i` decision variable (instead of a pre-solve numpy estimate), and by promoting `delta_extra` to a hard floor on the safety slack inside the NLP.

**Architecture:** `demo/barrier_dms.py`'s `_build_nlp_symbols` currently subtracts a precomputed scalar margin from each node's region-boundary distance. We replace that scalar with a CasADi expression built from the segment's own `delta_vars[i]`, so the constraint the solver actually satisfies already reflects whatever `Δ_i` it converges to. `_compute_min_slack` (used for post-hoc certification) is rewritten to use the identical formula evaluated at the optimized `Δ_i`, eliminating the old double-subtraction of `lipschitz_gap`/`eps_int`. `demo/optimizer.py`'s 4-iteration pre-solve warm-start loop collapses to a single pass (it's no longer correctness-critical) and a latent bug — passing the warm-start-inflated `delta_safe_barrier` into the live NLP's base margin instead of the raw `cfg.delta_safe` — is fixed as part of that collapse. A new `epsilon_certificate_buffer` config knob (default `0.0`) gives manual, config-only tuning headroom.

**Tech Stack:** Python, CasADi (MX symbolic expressions, IPOPT NLP), NumPy, unittest (existing test style in this repo).

**Spec:** `docs/superpowers/specs/2026-06-17-crd-certificate-margin-relaxation-design.md`

---

## File Structure

- Modify `demo/barrier_dms.py` — `BarrierDMSConfig` (new field), `_build_nlp_symbols` (live margin + hard floor), `_solve_barrier_level` (drop dead param), `_compute_min_slack` (rewritten formula), `solve()` (call sites + certification formula).
- Modify `demo/optimizer.py` — `CentroidRefineDMSConfig` (new field), `create_integrated_optimizer_from_config` (parse new key), pre-solve warm-start loop (collapse to single pass + base-margin bug fix), `BarrierDMSConfig(...)` construction (pass new field through).
- Modify `demo/config.yaml` — add `epsilon_certificate_buffer` under `centroid_refine_dms:` with documentation.
- Modify `tests/test_barrier_dms.py` — rewrite the three tests pinned to the old formula/signatures; add two new regression tests.
- Modify `tests/test_centroid_refine_dms.py` — add one regression test confirming the optimizer-level config plumbing and end-to-end certification.
- Modify `demo/test_crd_default.py` — fix the debug monkeypatch that calls `_solve_barrier_level` positionally (signature changes).

---

### Task 1: Add `epsilon_certificate_buffer` config field

**Files:**
- Modify: `demo/barrier_dms.py:14-29` (`BarrierDMSConfig`)
- Modify: `demo/optimizer.py:130-156` (`CentroidRefineDMSConfig`)
- Modify: `demo/optimizer.py:2780-2799` (`create_integrated_optimizer_from_config`)
- Modify: `demo/config.yaml:95-124` (`centroid_refine_dms:` section)
- Test: `tests/test_centroid_refine_dms.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_centroid_refine_dms.py`, inside `CentroidRefineDMSConfigTests`:

```python
    def test_epsilon_certificate_buffer_defaults_to_zero(self):
        cfg = CentroidRefineDMSConfig()
        self.assertEqual(cfg.epsilon_certificate_buffer, 0.0)

    def test_epsilon_certificate_buffer_parsed_from_config_dict(self):
        from optimizer import create_integrated_optimizer_from_config
        graph, dynamics, _, _ = _make_two_region_scenario()
        cfg_dict = {
            'optimizer': {'solver_mode': 'centroid_refine_dms'},
            'dynamics': {}, 'shooting': {}, 'cost': {}, 'control': {},
            'centroid_refine_dms': {'epsilon_certificate_buffer': 0.003},
        }
        solver = create_integrated_optimizer_from_config(graph, dynamics, cfg_dict)
        self.assertEqual(solver.config.epsilon_certificate_buffer, 0.003)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/khoa/ws/GCS_MMS_demo && .venv/bin/python -m pytest tests/test_centroid_refine_dms.py -k epsilon_certificate_buffer -v`
Expected: FAIL with `AttributeError: 'CentroidRefineDMSConfig' object has no attribute 'epsilon_certificate_buffer'`

- [ ] **Step 3: Add the field to `BarrierDMSConfig`**

In `demo/barrier_dms.py`, change:

```python
    n_control_segments: int = 2
    mu_weight: float = 1.0
```

to:

```python
    n_control_segments: int = 2
    epsilon_certificate_buffer: float = 0.0
    mu_weight: float = 1.0
```

- [ ] **Step 4: Add the field to `CentroidRefineDMSConfig`**

In `demo/optimizer.py`, change:

```python
    delta_safe: float = 0.02
    delta_extra: float = 0.01
    alpha_s: float = 0.0
```

to:

```python
    delta_safe: float = 0.02
    delta_extra: float = 0.01
    epsilon_certificate_buffer: float = 0.0
    alpha_s: float = 0.0
```

- [ ] **Step 5: Parse the new key in `create_integrated_optimizer_from_config`**

In `demo/optimizer.py`, change:

```python
            delta_safe=cr_cfg_dict.get('delta_safe', 0.02),
            delta_extra=cr_cfg_dict.get('delta_extra', 0.01),
            alpha_s=cr_cfg_dict.get('alpha_s', 0.0),
```

to:

```python
            delta_safe=cr_cfg_dict.get('delta_safe', 0.02),
            delta_extra=cr_cfg_dict.get('delta_extra', 0.01),
            epsilon_certificate_buffer=cr_cfg_dict.get('epsilon_certificate_buffer', 0.0),
            alpha_s=cr_cfg_dict.get('alpha_s', 0.0),
```

- [ ] **Step 6: Document the new knob in `config.yaml`**

In `demo/config.yaml`, change:

```yaml
  # Stage B -- Interface QP safety margins
  # delta_safe is raised at runtime to: max(this, L_s * h_max/2 + epsilon_final)
  delta_safe: 0.02
  delta_extra: 0.01
  alpha_s: 0.0    # QP smoothness weight (0 = pure shortest polyline)
```

to:

```yaml
  # Stage B -- Interface QP safety margins
  delta_safe: 0.02
  # delta_extra is a HARD lower bound on the safety slack inside the
  # barrier-DMS NLP (not just a warm-start sanity threshold): raising it
  # directly raises the guaranteed certified_safety_margin. It also still
  # affects Interface QP anchor-point clearance.
  delta_extra: 0.01
  # Extra certified-margin headroom, added only to the NLP's safety-slack
  # floor and the certified_safety_margin pass/fail threshold. Does NOT
  # affect Interface QP/path-search behavior (unlike delta_safe/delta_extra).
  # Raise this per-scenario if safety_certification reports NOT_CERTIFIED
  # with a thin margin.
  epsilon_certificate_buffer: 0.0
  alpha_s: 0.0    # QP smoothness weight (0 = pure shortest polyline)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd /home/khoa/ws/GCS_MMS_demo && .venv/bin/python -m pytest tests/test_centroid_refine_dms.py -k epsilon_certificate_buffer -v`
Expected: PASS (2 passed)

- [ ] **Step 8: Commit**

```bash
git add demo/barrier_dms.py demo/optimizer.py demo/config.yaml tests/test_centroid_refine_dms.py
git commit -m "feat(crd): add epsilon_certificate_buffer config knob"
```

---

### Task 2: Rewrite `_compute_min_slack` to use a live per-segment margin

**Files:**
- Modify: `demo/barrier_dms.py:482-514` (`_compute_min_slack`)
- Test: `tests/test_barrier_dms.py`

The current signature takes a flat `delta_safe: float` argument and subtracts it as one constant for every node. The new version computes, per segment, `margin = cfg.delta_safe + L_s * h/2 + f_lip * h**4/30` using that segment's own optimized `Δ_i` — eliminating the dependency on an externally-supplied scalar.

- [ ] **Step 1: Write the failing test**

Replace `test_compute_min_slack_subtracts_delta_safe` in `tests/test_barrier_dms.py` with:

```python
    def test_compute_min_slack_scales_with_configured_delta_safe(self) -> None:
        graph, dynamics = _two_region_setup()
        result = BarrierPathResult(
            success=True, path_regions=[0], total_cost=0.0, solve_time=0.0,
            solver_status="ok",
            entry_states={0: np.array([0.5, 0.5, 0.0])},
            control_params={0: np.zeros(dynamics.n_u * 2)},
            time_durations={0: 1.0},
        )
        cfg_low = BarrierDMSConfig(n_int=5, barrier_levels=[1.0], delta_safe=0.0)
        cfg_high = BarrierDMSConfig(n_int=5, barrier_levels=[1.0], delta_safe=0.05)
        slack_low = BarrierDMSSolver(graph, dynamics, cfg_low)._compute_min_slack(result)
        slack_high = BarrierDMSSolver(graph, dynamics, cfg_high)._compute_min_slack(result)
        # Only cfg.delta_safe differs between the two configs; L_s and eps_int
        # depend on dynamics/path/Delta_i, which are identical, so the gap
        # between the two slacks must equal exactly the delta_safe gap.
        self.assertAlmostEqual(slack_low - slack_high, 0.05, places=9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/khoa/ws/GCS_MMS_demo && .venv/bin/python -m pytest tests/test_barrier_dms.py -k test_compute_min_slack_scales_with_configured_delta_safe -v`
Expected: FAIL with `TypeError: _compute_min_slack() missing 1 required positional argument: 'delta_safe'`

- [ ] **Step 3: Rewrite `_compute_min_slack`**

In `demo/barrier_dms.py`, replace the entire method:

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
                if k == self.config.n_int:
                    break

                tau = k * dt
                u_k = self.control_param.evaluate(tau, w)
                k1 = delta * self.dynamics.f(x_k, u_k)
                u_mid = self.control_param.evaluate(tau + 0.5 * dt, w)
                k2 = delta * self.dynamics.f(x_k + 0.5 * dt * k1, u_mid)
                k3 = delta * self.dynamics.f(x_k + 0.5 * dt * k2, u_mid)
                u_end = self.control_param.evaluate(tau + dt, w)
                k4 = delta * self.dynamics.f(x_k + dt * k3, u_end)
                x_k = x_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return min_slack if np.isfinite(min_slack) else float('nan')
```

with:

```python
    def _compute_min_slack(self, result: BarrierPathResult) -> float:
        """
        Sampled safety margin at the optimized solution, using the SAME
        per-segment margin formula enforced live inside the NLP (spec
        2026-06-17-crd-certificate-margin-relaxation-design.md Sec 1/4):
        margin_i = delta_safe + L_s * h_i/2 + f_lip * h_i**4/30, with
        h_i = Delta_i / n_int computed from the OPTIMIZED Delta_i, not a
        pre-solve estimate.
        """
        cfg = self.config
        if not result.path_regions:
            return float('nan')
        A_list = [self.graph._regions_by_index[ri].A for ri in result.path_regions]
        L_s = self.dynamics.compute_lipschitz_bound(A_list)
        f_lip = self.dynamics.f_lipschitz_bound()

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
            dt = 1.0 / cfg.n_int
            h_i = delta / cfg.n_int
            margin = cfg.delta_safe + L_s * h_i / 2.0 + f_lip * (h_i ** 4) / 30.0

            for k in range(cfg.n_int + 1):
                pos = self.dynamics.project_to_position(x_k)
                slacks = region.b - margin - region.A @ pos
                min_slack = min(min_slack, float(np.min(slacks)))
                if k == cfg.n_int:
                    break

                tau = k * dt
                u_k = self.control_param.evaluate(tau, w)
                k1 = delta * self.dynamics.f(x_k, u_k)
                u_mid = self.control_param.evaluate(tau + 0.5 * dt, w)
                k2 = delta * self.dynamics.f(x_k + 0.5 * dt * k1, u_mid)
                k3 = delta * self.dynamics.f(x_k + 0.5 * dt * k2, u_mid)
                u_end = self.control_param.evaluate(tau + dt, w)
                k4 = delta * self.dynamics.f(x_k + dt * k3, u_end)
                x_k = x_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return min_slack if np.isfinite(min_slack) else float('nan')
```

- [ ] **Step 4: Update the call site in `solve()` (will be finished in Task 4, but must compile now)**

In `demo/barrier_dms.py`, temporarily change line 143 from:

```python
        min_slack = self._compute_min_slack(best_result, delta_safe)
```

to:

```python
        min_slack = self._compute_min_slack(best_result)
```

(The rest of the certification formula in `solve()` is finished in Task 4 — this step only keeps the file import-able and the new test runnable in isolation.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/khoa/ws/GCS_MMS_demo && .venv/bin/python -m pytest tests/test_barrier_dms.py -k test_compute_min_slack_scales_with_configured_delta_safe -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add demo/barrier_dms.py tests/test_barrier_dms.py
git commit -m "refactor(crd): compute certification margin from live per-segment Delta_i"
```

---

### Task 3: Embed the live margin and hard `delta_extra` floor in `_build_nlp_symbols`

**Files:**
- Modify: `demo/barrier_dms.py:215-419` (`_build_nlp_symbols`)
- Modify: `demo/barrier_dms.py:421-480` (`_solve_barrier_level`)
- Modify: `demo/barrier_dms.py:71-130` (`solve()` call sites)
- Test: `tests/test_barrier_dms.py`

This is the core fix: the per-node slack constraint stops using the precomputed numpy `node_delta_safe[i, k]` and instead uses a CasADi expression of `delta_vars[i]`, and the slack variable's lower bound becomes `delta_extra + epsilon_certificate_buffer` instead of `1e-10`. The now-dead `delta_safe` scalar parameter is dropped from `_build_nlp_symbols`/`_solve_barrier_level` (it was already unused in the constraint — only threaded through).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_barrier_dms.py`:

```python
    def test_safety_slack_lower_bound_is_delta_extra_plus_buffer(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(
            n_int=5, barrier_levels=[1.0], delta_extra=0.02,
            epsilon_certificate_buffer=0.01,
        )
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        node_delta_safe = np.full((2, cfg.n_int + 1), cfg.delta_safe)
        out = solver._build_nlp_symbols(
            [0, 1], anchor_points, mu=1.0, node_delta_safe=node_delta_safe,
        )
        lbx = out[5]
        expected_floor = cfg.delta_extra + cfg.epsilon_certificate_buffer
        # lbx layout per segment: [state_lb..., w_lb..., delta_min, then
        # one entry per safety-slack component per node]. Every safety-slack
        # lower bound must equal delta_extra + epsilon_certificate_buffer,
        # never the old hardcoded 1e-10.
        self.assertNotIn(1e-10, lbx)
        self.assertTrue(any(abs(v - expected_floor) < 1e-12 for v in lbx))

    def test_live_margin_keeps_min_slack_above_delta_extra(self) -> None:
        # The defined_slack expression must depend on delta_vars[i]: once
        # the per-segment Lipschitz/RK4 terms are embedded live, a converged
        # solve's sampled min_safety_margin (which nets out that live
        # margin) must still clear the hard delta_extra floor enforced on
        # the slack variable.
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(n_int=10, barrier_levels=[1.0, 0.1, 0.01], time_limit_s=30.0)
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        result = solver.solve(
            [0, 1], anchor_points,
            start_state=np.array([0.1, 0.5, 0.0]),
            goal_state=np.array([1.9, 0.5, 0.0]),
        )
        self.assertTrue(result.success)
        self.assertGreaterEqual(
            result.min_safety_margin, cfg.delta_extra - 1e-6,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/khoa/ws/GCS_MMS_demo && .venv/bin/python -m pytest tests/test_barrier_dms.py -k "test_safety_slack_lower_bound_is_delta_extra_plus_buffer or test_live_margin_keeps_min_slack_above_delta_extra" -v`
Expected: FAIL — `_build_nlp_symbols` still requires the now-removed `delta_safe` positional arg in its current form and still hardcodes `1e-10`.

- [ ] **Step 3: Rewrite `_build_nlp_symbols`**

In `demo/barrier_dms.py`, change the signature:

```python
    def _build_nlp_symbols(
        self, path_regions, anchor_points, delta_safe, mu,
        start_state=None, goal_state=None, node_delta_safe=None,
    ):
```

to:

```python
    def _build_nlp_symbols(
        self, path_regions, anchor_points, mu,
        start_state=None, goal_state=None, node_delta_safe=None,
    ):
```

Then, change the existing setup block:

```python
        m = len(path_regions)
        n_x = self.dynamics.n_x
        n_w = self.control_param.n_w
        cfg = self.config
        dt = 1.0 / cfg.n_int
```

to:

```python
        m = len(path_regions)
        n_x = self.dynamics.n_x
        n_w = self.control_param.n_w
        cfg = self.config
        dt = 1.0 / cfg.n_int
        A_list = [self.graph._regions_by_index[ri].A for ri in path_regions]
        L_s = self.dynamics.compute_lipschitz_bound(A_list)
        f_lip = self.dynamics.f_lipschitz_bound()
```

Next, in the safety-slack construction loop, change:

```python
        for i, region_idx in enumerate(path_regions):
            region = self.graph._regions_by_index[region_idx]
            for k, x_k in enumerate(x_node_vars_list[i]):
                slack = ca.MX.sym(
                    f'safety_slack_{i}_{k}', region.A.shape[0]
                )
                x_sym_list.append(slack)
                lbx.extend([1e-10] * region.A.shape[0])
                ubx.extend([np.inf] * region.A.shape[0])

                tau = k / cfg.n_int
                warm_pos = (
                    (1.0 - tau) * anchor_points[i]
                    + tau * anchor_points[i + 1]
                )
                warm_slack = (
                    region.b - node_delta_safe[i, k]
                    - region.A @ warm_pos
                )
                x0_list.extend(np.maximum(warm_slack, cfg.delta_extra).tolist())

                pos_k = self.dynamics.project_to_position_casadi(x_k)
                defined_slack = (
                    ca.DM(region.b)
                    - float(node_delta_safe[i, k])
                    - ca.mtimes(ca.DM(region.A), pos_k)
                )
                g_list.append(slack - defined_slack)
                lbg.extend([0.0] * region.A.shape[0])
                ubg.extend([0.0] * region.A.shape[0])

                delta_i = delta_vars[i]
                h_k = (
                    delta_i / (2.0 * cfg.n_int)
                    if k in (0, cfg.n_int)
                    else delta_i / cfg.n_int
                )
                barrier_term = (
                    barrier_term
                    - mu * cfg.mu_weight * h_k * ca.sum1(ca.log(slack))
                )
```

to:

```python
        for i, region_idx in enumerate(path_regions):
            region = self.graph._regions_by_index[region_idx]
            delta_i = delta_vars[i]
            # Live margin (spec Sec 1): grows with this segment's OWN
            # optimized Delta_i, so the constraint the solver satisfies
            # always matches the true post-hoc Lipschitz/RK4 requirement
            # for whatever Delta_i it converges to -- no pre-solve estimate.
            h_i = delta_i / cfg.n_int
            margin_i = cfg.delta_safe + L_s * h_i / 2.0 + f_lip * (h_i ** 4) / 30.0
            slack_floor = cfg.delta_extra + cfg.epsilon_certificate_buffer

            for k, x_k in enumerate(x_node_vars_list[i]):
                slack = ca.MX.sym(
                    f'safety_slack_{i}_{k}', region.A.shape[0]
                )
                x_sym_list.append(slack)
                # Hard floor (spec Sec 2): guarantees certified_safety_margin
                # >= delta_extra + epsilon_certificate_buffer - defect_norm
                # for any successful solve, by construction.
                lbx.extend([slack_floor] * region.A.shape[0])
                ubx.extend([np.inf] * region.A.shape[0])

                tau = k / cfg.n_int
                warm_pos = (
                    (1.0 - tau) * anchor_points[i]
                    + tau * anchor_points[i + 1]
                )
                warm_slack = (
                    region.b - node_delta_safe[i, k]
                    - region.A @ warm_pos
                )
                x0_list.extend(np.maximum(warm_slack, slack_floor).tolist())

                pos_k = self.dynamics.project_to_position_casadi(x_k)
                defined_slack = (
                    ca.DM(region.b)
                    - margin_i
                    - ca.mtimes(ca.DM(region.A), pos_k)
                )
                g_list.append(slack - defined_slack)
                lbg.extend([0.0] * region.A.shape[0])
                ubg.extend([0.0] * region.A.shape[0])

                h_k = (
                    delta_i / (2.0 * cfg.n_int)
                    if k in (0, cfg.n_int)
                    else delta_i / cfg.n_int
                )
                barrier_term = (
                    barrier_term
                    - mu * cfg.mu_weight * h_k * ca.sum1(ca.log(slack))
                )
```

Note `delta_i = delta_vars[i]` is hoisted out of the `for k, ...` loop (computed once per segment instead of redundantly per node — same value either way).

- [ ] **Step 4: Drop the dead `delta_safe` parameter from `_solve_barrier_level`**

In `demo/barrier_dms.py`, change:

```python
    def _solve_barrier_level(self, path_regions, anchor_points, warm_start,
                              delta_safe, mu, x_init, tol,
                              start_state=None, goal_state=None,
                              node_delta_safe=None):
        (x_sym, obj_base, barrier_term, g_sym, lbx, ubx, lbg, ubg,
         x0_default, s_minus_vars, w_vars, delta_vars, x_node_vars_list) = self._build_nlp_symbols(
            path_regions, anchor_points, delta_safe, mu,
            start_state=start_state, goal_state=goal_state,
            node_delta_safe=node_delta_safe,
        )
```

to:

```python
    def _solve_barrier_level(self, path_regions, anchor_points, warm_start,
                              mu, x_init, tol,
                              start_state=None, goal_state=None,
                              node_delta_safe=None):
        (x_sym, obj_base, barrier_term, g_sym, lbx, ubx, lbg, ubg,
         x0_default, s_minus_vars, w_vars, delta_vars, x_node_vars_list) = self._build_nlp_symbols(
            path_regions, anchor_points, mu,
            start_state=start_state, goal_state=goal_state,
            node_delta_safe=node_delta_safe,
        )
```

- [ ] **Step 5: Update the call site in `solve()`**

In `demo/barrier_dms.py`, change:

```python
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
```

to:

```python
                result, x_init = self._solve_barrier_level(
                    path_regions=path_regions,
                    anchor_points=anchor_points,
                    warm_start=warm_start,
                    mu=mu,
                    x_init=x_init,
                    tol=tol,
                    start_state=start_state,
                    goal_state=goal_state,
                    node_delta_safe=node_delta_safe,
                )
```

- [ ] **Step 6: Update the other test that calls `_build_nlp_symbols` directly**

In `tests/test_barrier_dms.py`, `test_build_nlp_symbols_splits_barrier_from_base_cost` currently calls:

```python
        out1 = solver._build_nlp_symbols(
            [0, 1], anchor_points, delta_safe, mu, node_delta_safe=node_delta_safe,
        )
```
and
```python
        out2 = solver._build_nlp_symbols(
            [0, 1], anchor_points, delta_safe, 2 * mu, node_delta_safe=node_delta_safe,
        )
```

Change both to drop the now-removed `delta_safe` positional argument:

```python
        out1 = solver._build_nlp_symbols(
            [0, 1], anchor_points, mu, node_delta_safe=node_delta_safe,
        )
```
and
```python
        out2 = solver._build_nlp_symbols(
            [0, 1], anchor_points, 2 * mu, node_delta_safe=node_delta_safe,
        )
```

The local `delta_safe` variable computed earlier in that test (`delta_safe = max(cfg.delta_safe, lip_gap + cfg.epsilon_final)`) and `node_delta_safe` built from it can stay — they're still valid inputs for the `node_delta_safe` warm-start parameter, just no longer also passed as the removed scalar argument.

- [ ] **Step 7: Run the full barrier_dms test file**

Run: `cd /home/khoa/ws/GCS_MMS_demo && .venv/bin/python -m pytest tests/test_barrier_dms.py -v`
Expected: All tests pass except possibly `test_certified_margin_accounts_for_defect_and_integration_error` (fixed in Task 4) — confirm every other test, including the two new ones from Step 1, is green.

- [ ] **Step 8: Commit**

```bash
git add demo/barrier_dms.py tests/test_barrier_dms.py
git commit -m "feat(crd): embed live per-segment safety margin and hard delta_extra floor in NLP"
```

---

### Task 4: Fix the certification formula in `solve()` (stop double-subtracting)

**Files:**
- Modify: `demo/barrier_dms.py:71-179` (`solve()`)
- Test: `tests/test_barrier_dms.py`

- [ ] **Step 1: Write the failing test**

Replace `test_certified_margin_accounts_for_defect_and_integration_error` in `tests/test_barrier_dms.py` with:

```python
    def test_certified_margin_matches_new_formula(self) -> None:
        graph, dynamics = _two_region_setup()
        cfg = BarrierDMSConfig(
            n_int=40, barrier_levels=[1.0, 0.5, 0.1, 0.01], time_limit_s=30.0,
        )
        solver = BarrierDMSSolver(graph, dynamics, cfg)
        anchor_points = np.array([[0.1, 0.5], [1.0, 0.5], [1.9, 0.5]])
        result = solver.solve(
            [0, 1], anchor_points,
            start_state=np.array([0.1, 0.5, 0.0]),
            goal_state=np.array([1.9, 0.5, 0.0]),
        )
        self.assertTrue(result.success)
        expected = (
            result.min_safety_margin
            - cfg.delta_extra
            - cfg.epsilon_certificate_buffer
            - result.defect_norm
        )
        self.assertAlmostEqual(result.certified_safety_margin, expected, places=9)
        self.assertEqual(result.safety_certification, "CERTIFIED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/khoa/ws/GCS_MMS_demo && .venv/bin/python -m pytest tests/test_barrier_dms.py -k test_certified_margin_matches_new_formula -v`
Expected: FAIL — `certified_safety_margin` still computed with the old `min_slack - lip_gap - defect_norm - eps_int` formula, so it won't equal `expected`.

- [ ] **Step 3: Rewrite the certification block in `solve()`**

In `demo/barrier_dms.py`, change:

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

to:

```python
        # min_slack already embeds the Lipschitz-gap/RK4-truncation terms
        # for the OPTIMIZED Delta_i (Task 2/3) -- do not subtract lip_gap or
        # eps_int again here, that would double-count them.
        min_slack = self._compute_min_slack(best_result)
        optimized_delta_arr = np.array(
            [best_result.time_durations[ri] for ri in path_regions], dtype=float
        )
        # lip_gap/eps_int below are PATH-WIDE WORST-CASE diagnostics only
        # (reported on BarrierPathResult for backward-compatible
        # logging/reporting) -- the per-segment values actually enforced
        # live inside the NLP are already folded into min_slack.
        lip_gap = compute_lipschitz_safety_gap(
            self.dynamics, path_regions, self.graph,
            optimized_delta_arr, cfg.n_int,
        )
        h_max = float(np.max(optimized_delta_arr)) / cfg.n_int
        defect_norm = self._compute_defect_norm(best_result, start_state, goal_state)
        eps_int = self.dynamics.f_lipschitz_bound() * (h_max ** 4) / 30.0
        best_result.defect_norm = defect_norm
        best_result.lipschitz_gap = lip_gap
        best_result.min_safety_margin = float(min_slack)
        # Spec Sec 4: certified margin above the hard floor enforced live in
        # the NLP (delta_extra + epsilon_certificate_buffer), corrected only
        # for the hard-equality residual (defect_norm) at solver tolerance.
        best_result.certified_safety_margin = float(
            min_slack - cfg.delta_extra - cfg.epsilon_certificate_buffer - defect_norm
        )
        best_result.safety_certification = (
            "CERTIFIED" if best_result.certified_safety_margin > 0 else "NOT_CERTIFIED"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/khoa/ws/GCS_MMS_demo && .venv/bin/python -m pytest tests/test_barrier_dms.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add demo/barrier_dms.py tests/test_barrier_dms.py
git commit -m "fix(crd): stop double-subtracting Lipschitz gap in certification formula"
```

---

### Task 5: Collapse the optimizer.py pre-solve loop to a single pass and fix the base-margin bug

**Files:**
- Modify: `demo/optimizer.py:2617-2678`
- Test: `tests/test_centroid_refine_dms.py`

This also fixes a bug uncovered by the redesign: the existing code passes the *warm-start-inflated* `delta_safe_barrier` into `BarrierDMSConfig(delta_safe=...)`, which would now double-count against the live per-segment margin added inside the NLP (Task 3). The NLP's base margin must be the raw `cfg.delta_safe`; the inflated estimate is only for the Interface QP's anchor-point placement.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_centroid_refine_dms.py`, inside `CentroidRefineDMSSolverTests`:

```python
    def test_barrier_nlp_uses_raw_delta_safe_not_inflated_estimate(self):
        # Regression test: BarrierDMSConfig.delta_safe passed to the NLP
        # must be the raw config value, not the warm-start-inflated
        # delta_safe_barrier estimate computed for Interface QP placement
        # (that estimate already gets re-added live inside the NLP per
        # Delta_i, so reusing it as the NLP's base would double-count it).
        import optimizer as _opt
        graph, dynamics, x_start, x_goal = _make_two_region_scenario()
        cfg = CentroidRefineDMSConfig(n_int=8, epsilon_final=1e-3, delta_safe=0.02)
        solver = _opt.CentroidRefineDMSSolver(graph, dynamics, cfg)

        seen_delta_safe = []
        OrigBarrierDMSConfig = _opt.BarrierDMSConfig

        def _spy(*args, **kwargs):
            seen_delta_safe.append(kwargs.get('delta_safe'))
            return OrigBarrierDMSConfig(*args, **kwargs)

        _opt.BarrierDMSConfig = _spy
        try:
            solver.solve(x_start, x_goal)
        finally:
            _opt.BarrierDMSConfig = OrigBarrierDMSConfig

        self.assertTrue(seen_delta_safe)
        for value in seen_delta_safe:
            self.assertEqual(value, cfg.delta_safe)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/khoa/ws/GCS_MMS_demo && .venv/bin/python -m pytest tests/test_centroid_refine_dms.py -k test_barrier_nlp_uses_raw_delta_safe_not_inflated_estimate -v`
Expected: FAIL — `seen_delta_safe` contains the inflated `delta_safe_barrier` value, not `cfg.delta_safe` (0.02), for any path where the warm-start loop actually raised the estimate above the raw value.

- [ ] **Step 3: Collapse the loop and fix the base-margin bug**

In `demo/optimizer.py`, change:

```python
                delta_safe_barrier = cfg.delta_safe
                for _ in range(4):
                    # Clamp to delta_max: the NLP variable bound (Part 2.4)
                    # means Delta_i can never exceed delta_max once solved (see
                    # the x0 clamp a few lines below), so the safety-gap
                    # pre-check (R13/Part 8.2's h_max) must use the same bound
                    # -- not the raw, unclamped warm-start estimate, which can
                    # be arbitrarily large for elongated regions (Part 4.1
                    # defines Delta_i^0 with no upper clamp) and would
                    # otherwise inflate delta_safe far past what the solve
                    # will actually need.
                    delta_arr = np.array(
                        [warm_start[ri]['delta'] for ri in path_regions], dtype=float
                    )
                    delta_arr = np.minimum(delta_arr, cfg.delta_max)
                    lip_gap = compute_lipschitz_safety_gap(
                        self.dynamics, path_regions, self.graph, delta_arr, cfg.n_int
                    )
                    required_delta_safe = max(
                        cfg.delta_safe, lip_gap + cfg.epsilon_final
                    )
                    if (
                        not cfg.use_interface_qp
                        or required_delta_safe <= delta_safe_barrier + 1e-9
                    ):
                        delta_safe_barrier = required_delta_safe
                        break

                    delta_safe_barrier = required_delta_safe
                    barrier_qp_cfg = InterfaceQPConfig(
                        delta_safe=delta_safe_barrier,
                        delta_extra=cfg.delta_extra,
                        lambda_s=cfg.alpha_s,
                    )
                    z = solve_interface_refinement(
                        self.graph, path_regions, x_start, x_goal, barrier_qp_cfg
                    )
                    warm_start = build_centroid_warmstart(
                        graph=self.graph, path_regions=path_regions, anchor_points=z,
                        dynamics=self.dynamics, config=ws_cfg,
                    )
            except NarrowInterfaceError as e:
                failure_log.append({'path': path, 'stage': 'B2', 'reason': str(e)})
                continue
            except Exception as e:
                failure_log.append({'path': path, 'stage': 'C', 'reason': str(e)})
                continue

            barrier_cfg = BarrierDMSConfig(
                n_int=cfg.n_int, delta_safe=delta_safe_barrier, delta_extra=cfg.delta_extra,
                barrier_levels=cfg.barrier_levels if cfg.use_barrier_continuation else [cfg.barrier_levels[-1]],
                epsilon_final=cfg.epsilon_final,
                time_limit_s=max(1.0, cfg.time_limit_s - (_time.time() - start_clock)),
                delta_min=cfg.delta_min, delta_max=cfg.delta_max,
                n_control_segments=cfg.n_control_segments,
                w_T=cfg.w_T, w_L=cfg.w_L, w_U=cfg.w_U, w_S=cfg.w_S,
            )
```

to:

```python
                # Single-pass warm-start clearance estimate (spec Sec 5):
                # this only sizes the Interface QP anchor-point placement so
                # the warm start passes the R3 strict-interior check and
                # gives IPOPT a reasonable first iterate. It is NOT
                # load-bearing for safety correctness anymore -- the live
                # per-segment margin inside BarrierDMSSolver (Task 3) is
                # what actually guarantees the certified safety margin,
                # regardless of how good this estimate is.
                delta_arr = np.array(
                    [warm_start[ri]['delta'] for ri in path_regions], dtype=float
                )
                delta_arr = np.minimum(delta_arr, cfg.delta_max)
                lip_gap = compute_lipschitz_safety_gap(
                    self.dynamics, path_regions, self.graph, delta_arr, cfg.n_int
                )
                h_max = float(np.max(delta_arr)) / cfg.n_int
                eps_int = self.dynamics.f_lipschitz_bound() * (h_max ** 4) / 30.0
                required_delta_safe = max(
                    cfg.delta_safe, lip_gap + eps_int + cfg.epsilon_final
                )
                if cfg.use_interface_qp and required_delta_safe > cfg.delta_safe + 1e-9:
                    barrier_qp_cfg = InterfaceQPConfig(
                        delta_safe=required_delta_safe,
                        delta_extra=cfg.delta_extra,
                        lambda_s=cfg.alpha_s,
                    )
                    z = solve_interface_refinement(
                        self.graph, path_regions, x_start, x_goal, barrier_qp_cfg
                    )
                    warm_start = build_centroid_warmstart(
                        graph=self.graph, path_regions=path_regions, anchor_points=z,
                        dynamics=self.dynamics, config=ws_cfg,
                    )
            except NarrowInterfaceError as e:
                failure_log.append({'path': path, 'stage': 'B2', 'reason': str(e)})
                continue
            except Exception as e:
                failure_log.append({'path': path, 'stage': 'C', 'reason': str(e)})
                continue

            # IMPORTANT: delta_safe here is the RAW config value, not the
            # warm-start-inflated required_delta_safe estimate above. The
            # live per-segment margin inside BarrierDMSSolver already adds
            # the Lipschitz-gap/RK4-truncation terms on top of this base
            # using the NLP's own Delta_i -- reusing the inflated estimate
            # here would double-count those terms.
            barrier_cfg = BarrierDMSConfig(
                n_int=cfg.n_int, delta_safe=cfg.delta_safe, delta_extra=cfg.delta_extra,
                epsilon_certificate_buffer=cfg.epsilon_certificate_buffer,
                barrier_levels=cfg.barrier_levels if cfg.use_barrier_continuation else [cfg.barrier_levels[-1]],
                epsilon_final=cfg.epsilon_final,
                time_limit_s=max(1.0, cfg.time_limit_s - (_time.time() - start_clock)),
                delta_min=cfg.delta_min, delta_max=cfg.delta_max,
                n_control_segments=cfg.n_control_segments,
                w_T=cfg.w_T, w_L=cfg.w_L, w_U=cfg.w_U, w_S=cfg.w_S,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/khoa/ws/GCS_MMS_demo && .venv/bin/python -m pytest tests/test_centroid_refine_dms.py -v`
Expected: All tests pass, including the new `test_barrier_nlp_uses_raw_delta_safe_not_inflated_estimate`.

- [ ] **Step 5: Commit**

```bash
git add demo/optimizer.py tests/test_centroid_refine_dms.py
git commit -m "fix(crd): pass raw delta_safe to barrier NLP, collapse warm-start loop to single pass"
```

---

### Task 6: Fix the debug repro script's monkeypatch signature

**Files:**
- Modify: `demo/test_crd_default.py:128-144`

`patched_build_level` wraps `BarrierDMSSolver._solve_barrier_level`, which dropped its `delta_safe` positional parameter in Task 3. This script isn't part of the pytest suite but is the exact repro used in the original bug report, so it must keep working.

- [ ] **Step 1: Update the monkeypatch wrapper**

In `demo/test_crd_default.py`, change:

```python
    OrigBuildLevel = _bdms.BarrierDMSSolver._solve_barrier_level

    def patched_build_level(self, path_regions, anchor_points, warm_start,
                             delta_safe, mu, x_init, tol, **kwargs):
        log.debug(f"    _solve_barrier_level mu={mu:.4f}, tol={tol:.2e}, "
                  f"delta_safe={delta_safe:.4f}, "
                  f"warm={'from_prev' if x_init is not None else 'default'}")
        t0 = time.time()
        result, x_opt = OrigBuildLevel(self, path_regions, anchor_points, warm_start,
                                        delta_safe, mu, x_init, tol, **kwargs)
        elapsed = time.time() - t0
        log.debug(f"    Level done: success={result.success}, "
                  f"status={result.solver_status!r}, time={elapsed:.3f}s, "
                  f"cost={result.total_cost:.4f}")
        return result, x_opt
```

to:

```python
    OrigBuildLevel = _bdms.BarrierDMSSolver._solve_barrier_level

    def patched_build_level(self, path_regions, anchor_points, warm_start,
                             mu, x_init, tol, **kwargs):
        log.debug(f"    _solve_barrier_level mu={mu:.4f}, tol={tol:.2e}, "
                  f"warm={'from_prev' if x_init is not None else 'default'}")
        t0 = time.time()
        result, x_opt = OrigBuildLevel(self, path_regions, anchor_points, warm_start,
                                        mu, x_init, tol, **kwargs)
        elapsed = time.time() - t0
        log.debug(f"    Level done: success={result.success}, "
                  f"status={result.solver_status!r}, time={elapsed:.3f}s, "
                  f"cost={result.total_cost:.4f}")
        return result, x_opt
```

- [ ] **Step 2: Run the repro script to verify it still executes and now certifies**

Run: `cd /home/khoa/ws/GCS_MMS_demo/demo && ../.venv/bin/python test_crd_default.py`
Expected: Script completes without exception. Final log lines show `Safety certification: CERTIFIED` (previously `CERTIFICATE_FAIL`/`NOT_CERTIFIED`). If it still reports `NOT_CERTIFIED`, that means the `default` scenario's configured `delta_extra`/`epsilon_certificate_buffer` is too thin for its geometry — per the spec's config-tuning table, that is resolved by raising those two values for the `default` scenario in `config.yaml`, not by further code changes. Report the actual `certified_safety_margin` value back so the user can decide how much to raise `delta_extra`/`epsilon_certificate_buffer`.

- [ ] **Step 3: Commit**

```bash
git add demo/test_crd_default.py
git commit -m "fix(crd): update debug repro script for _solve_barrier_level signature change"
```

---

### Task 7: Full regression sweep

**Files:** none (verification only)

- [ ] **Step 1: Run the full unit test suite**

Run: `cd /home/khoa/ws/GCS_MMS_demo && .venv/bin/python -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 2: Run the full demo benchmark entry point used in the original bug report**

Run: `cd /home/khoa/ws/GCS_MMS_demo/demo && ../.venv/bin/python main_demo.py --scenario default`
Expected: Runs to completion. Compare `Optimization Status` and `Solver` in the output against the original report (`FAILED` / `CERTIFICATE_FAIL`) — confirm it now reports success/`CERTIFIED`, and note the new `Solve Time` relative to the original `65.868 s` (the single-pass warm-start loop from Task 5 should reduce it, since it removes up to 3 redundant Interface-QP solves per candidate path).

- [ ] **Step 3: Report results to the user**

Summarize, in chat: pass/fail status of `tests/`, the `default` scenario's final `safety_certification` and `certified_safety_margin`, and the before/after solve time. If `default` (or `maze`, if also re-run) still reports `NOT_CERTIFIED`, state the exact `certified_safety_margin` value and recommend a `delta_extra`/`epsilon_certificate_buffer` value for that scenario's `config.yaml` override — do not add any auto-tuning code to address it.

---

## Self-Review Notes (completed during planning, recorded for the implementer)

- **Spec coverage:** Sec 1 → Task 3 Step 3; Sec 2 → Task 3 Step 3 (`slack_floor`); Sec 3 → Task 1; Sec 4 → Task 4; Sec 5 → Task 5 (also fixes the base-margin double-count bug the spec's Sec 5 implies but doesn't spell out — called out explicitly in Task 5's intro and Step 3 comment).
- **Type/signature consistency:** `_build_nlp_symbols(path_regions, anchor_points, mu, ...)` and `_solve_barrier_level(path_regions, anchor_points, warm_start, mu, x_init, tol, ...)` — both lose the `delta_safe` positional argument together (Task 3 Steps 3-5); every call site (`solve()`, `tests/test_barrier_dms.py`, `demo/test_crd_default.py`) is updated in the same or a later task before the suite is expected to pass.
- **No placeholders:** every step shows the literal before/after code or the literal shell command and expected output.
