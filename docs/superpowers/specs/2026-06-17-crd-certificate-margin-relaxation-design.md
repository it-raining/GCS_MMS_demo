# Design: live Lipschitz/RK4 safety margin to fix CRD `CERTIFICATE_FAIL`

Companion to the root-cause analysis in
`docs/superpowers/specs/2026-06-17-crd-certificate-fail-phan-tich.md`. That
doc identifies the bug as an architectural gap: the per-node safety margin
used inside the barrier-DMS NLP is a **scalar estimated before solving**
from a warm-start guess of each region's dwell time `Δ_i`, but `Δ_i` is
itself a free NLP variable. When the solver converges to a different `Δ_i`
than the warm-start assumed, the post-hoc certificate (which recomputes the
Lipschitz-gap/RK4-truncation terms from the *true* `Δ_i`) can find that more
margin was required than was reserved, and reports `CERTIFICATE_FAIL` on an
otherwise well-converged, non-crashing solve.

## Goal

Relax the *estimation* (no longer guess a single safety margin from
warm-start data before solving) while *guaranteeing* the converged solution
never violates the certified safety margin — without adding a re-solve /
feedback loop after the main solve, and without materially increasing solve
time. Tunable safety/conservatism knobs are exposed in `config.yaml` for the
user to adjust by hand per scenario; no auto-tuning logic is added.

## Fix: embed the margin as a live function of the decision variable

Instead of computing the Lipschitz-gap and RK4-truncation-error terms from a
pre-solve estimate of `Δ_i` and baking them into a constant `delta_safe`,
compute them **symbolically, per segment, from that segment's own `Δ_i`
NLP variable** (`delta_vars[i]`, already in scope in
`_build_nlp_symbols`). Whatever `Δ_i` IPOPT converges to, the constraint it
satisfied during the solve already used the correct margin for that exact
`Δ_i` — the "estimate vs. actual" gap is eliminated by construction, not by
detecting and retrying after the fact.

### 1. Per-node margin as a CasADi expression of `Δ_i`

File: `demo/barrier_dms.py`, `_build_nlp_symbols` (currently lines ~373-413).

Today the slack constraint subtracts a precomputed numpy scalar
`node_delta_safe[i, k]`. Replace with, per segment `i`:

```python
L_s   = dynamics.compute_lipschitz_bound(A_list_for_path)   # scalar, computed once per path
f_lip = dynamics.f_lipschitz_bound()                        # scalar
h_i        = delta_vars[i] / cfg.n_int                      # CasADi expr
lip_gap_i  = L_s * h_i / 2.0                                 # CasADi expr
eps_int_i  = f_lip * h_i**4 / 30.0                           # CasADi expr
margin_i   = cfg.delta_safe + lip_gap_i + eps_int_i          # CasADi expr, varies with Δ_i
defined_slack = region.b - margin_i - A @ pos_k
```

`L_s` and `f_lip` are path-level constants (independent of `Δ_i`), matching
what `compute_lipschitz_safety_gap` already computes in
`constraint_layers.py` — but evaluated per-segment with that segment's own
`h_i = Δ_i / n_int` instead of the path-wide `max(Δ_i)/n_int`. This is not
an approximation: `L_s * h/2` and `f_lip * h^4/30` are bounds on the
discretization error accrued *within one shooting interval* of step size
`h`; each segment's bound only depends on its own `h_i`, so using the
path-wide worst-case `h` (as today's `compute_lipschitz_safety_gap` does for
the *summary* metric) is strictly more conservative than necessary for the
per-node constraint.

### 2. `delta_extra` becomes a hard floor on the slack variable

Today the safety-slack variable's box bound is `lbx=1e-10` — only the soft
log-barrier term discourages the trajectory from approaching the boundary,
so cost pressure can push the converged slack arbitrarily close to 0
relative to whatever `delta_safe` was used. Change `lbx` for every safety
slack variable (in `_build_nlp_symbols`) to:

```python
cfg.delta_extra + cfg.epsilon_certificate_buffer
```

This guarantees, for any successful solve, that the true distance to the
region boundary is at least `margin_i(Δ_i) + delta_extra +
epsilon_certificate_buffer`, up to IPOPT's own primal-feasibility tolerance
(≈1e-8 by default — negligible next to `delta_extra=0.01`). `delta_extra`
already exists in config and is already documented as a per-scenario knob
(`config.yaml`'s `maze` override); this gives it real teeth in the NLP
itself, not just the warm-start sanity check (`_log_warm_start_slacks`,
"R3 strict-interior invariant").

### 3. New config knob: `epsilon_certificate_buffer` (default `0.0`)

Added to `centroid_refine_dms:` in `config.yaml`, `CentroidRefineDMSConfig`
(`demo/optimizer.py`), and `BarrierDMSConfig` (`demo/barrier_dms.py`).

Kept separate from `delta_safe`/`delta_extra` deliberately: those two also
size the Interface-QP anchor-point clearance check
(`geometric_refiner.py:34`, `margin = delta_safe + delta_extra + 5e-4`) and
the R3 warm-start rejection threshold — changing them perturbs path
search/feasibility, not just the certificate. `epsilon_certificate_buffer`
only raises the in-NLP slack floor and the certified-margin pass/fail
threshold (point 4), giving a single dial purely for "how much certified
headroom do I want," independent of path-search/geometry behavior.

### 4. Post-hoc certification: stop double-subtracting

`_compute_min_slack` (`demo/barrier_dms.py`) is rewritten to use the same
`margin_i(Δ_i)` formula as the live constraint, evaluated numerically with
the **optimized** `Δ_i` (`result.time_durations[region_idx]`), instead of
taking a flat `delta_safe` scalar argument. Since the Lipschitz/RK4 terms
are now already inside `margin_i`, the final formula in `solve()` drops the
redundant external subtraction:

```python
# old: certified = min_slack - lip_gap - defect_norm - eps_int
#      (double-counts once margin_i already embeds lip_gap/eps_int)
# new:
certified_safety_margin = min_slack_full - cfg.delta_extra - cfg.epsilon_certificate_buffer - defect_norm
```

`lipschitz_gap` and the path-wide `eps_int` are still computed and reported
on `BarrierPathResult` for diagnostics and existing report formatting — they
are simply no longer subtracted a second time.

### 5. Simplify the pre-solve warm-start refinement loop

File: `demo/optimizer.py`, lines ~2622-2662 (inside
`CentroidRefineDMSSolver.solve`).

The existing 4-iteration fixed-point loop re-estimates `delta_safe_barrier`
from the warm-start `Δ_i`, re-solves the interface QP with that estimate,
and rebuilds the warm start — repeated up to 4 times per candidate path.
Its only remaining purpose after this fix is producing a *reasonable*
initial guess (good enough that `_log_warm_start_slacks`'s R3 check and
IPOPT's first iterate are sane); it is no longer load-bearing for
correctness, since correctness now lives entirely inside the NLP (points
1-2). Collapse it to a single estimate-then-refine pass: compute
`delta_arr` from the initial warm start, compute `lip_gap` (still using the
path-wide max, for an initial-guess estimate, including the `eps_int` term
for consistency with the new per-node formula), and re-solve the interface
QP **at most once** if the required clearance exceeds `cfg.delta_safe`.
This removes up to 3 redundant interface-QP solves per candidate path,
directly reducing solve time without weakening any guarantee.

## Config surface (what you tune by hand)

| Knob | File | Effect |
|---|---|---|
| `delta_safe` | `config.yaml: centroid_refine_dms.delta_safe` | Baseline geometric clearance from region boundary (unchanged meaning). |
| `delta_extra` | `config.yaml: centroid_refine_dms.delta_extra` | **New teeth**: hard lower bound on safety slack inside the NLP. Raising it directly raises the guaranteed certified margin. Also still affects Interface-QP anchor clearance and the R3 warm-start check (unchanged). |
| `epsilon_certificate_buffer` | `config.yaml: centroid_refine_dms.epsilon_certificate_buffer` (new) | Extra headroom added only to the NLP slack floor and the certified-margin pass/fail threshold. Does not affect Interface-QP/path-search behavior. Default `0.0`. |

If a scenario reports `NOT_CERTIFIED` (or a thin `certified_safety_margin`)
after this fix, the fix is: raise `delta_extra` and/or
`epsilon_certificate_buffer` for that scenario in `config.yaml`, the same
way `maze` already overrides `delta_extra`/`gamma_w`/`dynamics.delta_max`
today. No code changes or auto-tuning required for that adjustment.

## Test impact

`tests/test_barrier_dms.py` has tests pinned to the old formula's exact
shape:

- `test_compute_min_slack_subtracts_delta_safe` passes a flat `delta_safe`
  kwarg directly to `_compute_min_slack` — signature changes since the
  margin is now computed internally per-segment from `cfg.delta_safe` plus
  the live `Δ_i`-dependent terms. Rewritten to vary `BarrierDMSConfig.delta_safe`
  and assert the same "min_slack decreases by exactly the delta" relationship.
- `test_certified_margin_accounts_for_defect_and_integration_error` asserts
  `certified_safety_margin <= min_safety_margin - lipschitz_gap`, which
  encodes the old double-subtraction. Rewritten to assert the new contract:
  `certified_safety_margin == min_safety_margin - delta_extra -
  epsilon_certificate_buffer - defect_norm` (within floating-point
  tolerance), and that a successful solve on the existing two-region
  fixture is `CERTIFIED`.
- `test_build_nlp_symbols_splits_barrier_from_base_cost` calls
  `_build_nlp_symbols` directly with a `delta_safe`/`node_delta_safe`
  argument pair — signature/semantics updated to match the new internal
  margin computation (these arguments are now only used for the x0 initial
  guess construction, not the live constraint).

New tests to add:
- A test asserting that two solves with different `Δ_i` outcomes (e.g.
  forced via different `w_T`/`w_U` cost weights) each get a constraint
  consistent with *their own* converged `Δ_i`, not a shared pre-solve
  estimate — i.e. the regression test for the actual bug.
- A test that raising `delta_extra` strictly increases
  `certified_safety_margin` for a fixed solve, confirming it is a real
  tunable knob.

`demo/test_crd_default.py` (the script used to reproduce the original bug
report on the `default` scenario) becomes the primary end-to-end regression
check — it should report `safety_certification: CERTIFIED` after the fix,
without needing any change to `config.yaml`'s existing `delta_safe`/
`delta_extra` values for that scenario (if it still fails to certify, that's
the signal to raise `delta_extra`/`epsilon_certificate_buffer` for that
scenario, per the table above, not a sign the code fix is wrong).

## Out of scope

- Any automatic re-solve / feedback loop after the main solve converges
  (explicitly excluded — relaxation is handled by construction during the
  solve, and remaining headroom decisions are manual config knobs).
- Tightening `delta_max` or `n_integration_steps` per scenario (already a
  documented manual lever via existing config, e.g. the `maze` scenario's
  override comments) — orthogonal to this fix.
