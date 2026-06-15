# GCS-DMS Optimization Report: `default`

## 1. Scenario

| Field | Value |
|-------|-------|
| Preset | `default` |
| Description | Default map with hand-crafted convex safe regions |
| Number of regions | 15 |
| Number of edges | 44 |
| Candidate paths | 68 |
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
| Path nodes | `source → R9 → R13 → R4 → R7 → R6 → R10 → R11 → R0 → target` |
| Path regions | `[9, 13, 4, 7, 6, 10, 11, 0]` |
| Path length (regions) | 8 |

## 4. Objective and Timing

| Metric | Value |
|--------|-------|
| Total cost | 48.9455 |
| Total duration | 25.6704 s |
| Setup time | 0.3156 s |
| Solve time | 15.0511 s |
| Paths evaluated | 1 |

## 5. Feasibility Diagnostics

| Metric | Value |
|--------|-------|
| Defect norm | 0.00e+00 |
| Max connection gap | 1.13e-10 |
| Max control jump | 0.00e+00 |
| Constraint violation | 4.18e-09 |
| Max integrality gap | 4.45e-01 |
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

- **JSON summary:** `results/default_summary.json`
- **PNG result:** `results/default_result.png`
- **GIF animation:** `results/default_animation.gif` *(if generated)*
