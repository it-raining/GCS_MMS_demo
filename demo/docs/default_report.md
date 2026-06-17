# GCS-DMS Optimization Report: `default`

## 1. Scenario

| Field | Value |
|-------|-------|
| Preset | `default` |
| Description | 5x5 workspace with 6 obstacles, ACD2D convex decomposition |
| Number of regions | 15 |
| Number of edges | 51 |
| Candidate paths | 532 |
| Start state | `[0.2, 0.2, 0.785]` |
| Goal state | `[4.8, 4.8, 0.785]` |

## 2. Solver Mode

| Field | Value |
|-------|-------|
| Mode | `centroid_refine_dms` |
| Solver | IPOPT (via CasADi) |
| Problem class | Nonlinear program (NLP). Nonconvex due to unicycle dynamics. |
| Global optimality claim | LB is a geometric lower bound on the time component only. gap_k is NOT a certificate for the full DMS objective. No global optimality claim is made. The reported solution is a KKT point of the fixed-path barrier-augmented NLP under LICQ + SOSC. |

## 3. Selected Path

| Field | Value |
|-------|-------|
| Path nodes | `source → R9 → R13 → R14 → R4 → R7 → R6 → R10 → R11 → R0 → target` |
| Path regions | `[9, 13, 14, 4, 7, 6, 10, 11, 0]` |
| Path length (regions) | 9 |

## 4. Objective and Timing

| Metric | Value |
|--------|-------|
| Total cost | 58.3189 |
| Total duration | 17.4311 s |
| Setup time | 0.2205 s |
| Solve time | 14.6019 s |
| Paths evaluated | 17 |

## 5. Feasibility Diagnostics

| Metric | Value |
|--------|-------|
| Defect norm | 5.58e-07 |
| Max connection gap | 5.89e-07 |
| Max control jump | 0.00e+00 |
| Constraint violation | 0.00e+00 |
| Max integrality gap | 0.00e+00 |
| Min safety margin | 1.10e-02 |
| Path length metric | N/A |

## 6. Big-M / Perspective Status

| Constraint type | Status |
|-----------------|--------|
| Region activation | N/A |
| Region containment | N/A |
| Edge/interface containment | N/A |
| Interface equality | N/A |
| DMS defect | N/A |
| Control continuity | N/A |

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
