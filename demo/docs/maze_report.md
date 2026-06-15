# GCS-DMS Optimization Report: `maze`

## 1. Scenario

| Field | Value |
|-------|-------|
| Preset | `maze` |
| Description | Maze map with all active regions |
| Number of regions | 171 |
| Number of edges | 442 |
| Candidate paths | 768 |
| Start state | `[0.2, 0.2, 0.785]` |
| Goal state | `[4.8, 4.8, 0.785]` |

## 2. Solver Mode

| Field | Value |
|-------|-------|
| Mode | `LEGACY_INTEGRATED_BIGM_RELAXATION_IPOPT` |
| Solver | IPOPT (via CasADi) — continuous relaxation of integrated MIOCP |
| Problem class | Continuous relaxation of MIOCP (MINLP after DMS transcription). **Not** a MICP/MISOCP. Nonconvex due to unicycle dynamics. |
| Global optimality claim | No global certificate; IPOPT continuous relaxation of MIOCP. |

## 3. Selected Path

| Field | Value |
|-------|-------|
| Path nodes | `source → R56 → R73 → R99 → R119 → R120 → R139 → R138 → R121 → R100 → R97 → R98 → R96 → R118 → R136 → R137 → R156 → R135 → R113 → R93 → R76 → R117 → R116 → R95 → R159 → R160 → R143 → R141 → R142 → R122 → R140 → R157 → R124 → R123 → R78 → R104 → target` |
| Path regions | `[56, 73, 99, 119, 120, 139, 138, 121, 100, 97, 98, 96, 118, 136, 137, 156, 135, 113, 93, 76, 117, 116, 95, 159, 160, 143, 141, 142, 122, 140, 157, 124, 123, 78, 104]` |
| Path length (regions) | 35 |

## 4. Objective and Timing

| Metric | Value |
|--------|-------|
| Total cost | 8.1764 |
| Total duration | 1.8322 s |
| Setup time | 33.2489 s |
| Solve time | 284.0005 s |
| Paths evaluated | 1 |

## 5. Feasibility Diagnostics

| Metric | Value |
|--------|-------|
| Defect norm | 1.17e-06 |
| Max connection gap | 3.57e+00 |
| Max control jump | 8.85e-01 |
| Constraint violation | 6.87e-03 |
| Max integrality gap | 5.00e-01 |
| Min safety margin | N/A |
| Path length metric | N/A |

## 6. Big-M / Perspective Status

| Constraint type | Status |
|-----------------|--------|
| Region activation | **Big-M** — `|s| ≤ M·p`, `|w| ≤ M·p`, `Δ ≤ Δmax·p` force state/control/time to zero for inactive regions |
| Region containment | **Big-M** — `A q − b ≤ M(1−p)` guards endpoint and mesh safety |
| Edge/interface containment | **Big-M** — `A z_pos − b ≤ M(1−y)` guards interface point in intersection |
| Interface equality | **Big-M** — `s⁺[u] − z ≤ M(1−y)` and `s⁻[v] − z ≤ M(1−y)` for edge continuity |
| DMS defect | **No Big-M** — defect `s⁺ = F_endpoint(s⁻, w, Δ)` applied unconditionally; trivially satisfied for inactive regions (activation forces all vars to zero) |
| Control continuity | **Big-M** — `|u_exit − u_entry| ≤ M(1−y)` when `enforce_control_continuity=True` |

## 7. Nonconvexity Notes

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
nonconvex regardless of the path-selection strategy.

## 8. Generated Artifacts

- **Status:** Optimization failed
- **Solver status:** `IntegratedMIOCP relaxation: Infeasible_Problem_Detected`
- JSON / PNG / GIF may not have been generated.
