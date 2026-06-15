# GCS-DMS Optimization Report: `crd_demo`

## 1. Scenario

| Field | Value |
|-------|-------|
| Preset | `crd_demo` |
| Description | 5 overlapping rectangular regions; lightweight first-feasible CRD demo |
| Number of regions | 15 |
| Number of edges | 44 |
| Candidate paths | 68 |
| Start state | `[0.2, 0.2, 0.785]` |
| Goal state | `[4.8, 4.8, 0.785]` |

## 2. Solver Mode

| Field | Value |
|-------|-------|
| Mode | `CENTROID_REFINE_DMS` |
| Solver | IPOPT (via CasADi) |
| Problem class | Nonlinear program (NLP). Nonconvex due to unicycle dynamics. |
| Global optimality claim | LB is a geometric lower bound on time component only. No global optimality claim. The solution is a KKT point of the barrier-augmented NLP under LICQ+SOSC. |

## 3. Selected Path

| Field | Value |
|-------|-------|
| Path nodes | `source → R9 → R13 → R12 → R2 → R1 → R8 → R11 → R0 → target` |
| Path regions | `[9, 13, 12, 2, 1, 8, 11, 0]` |
| Path length (regions) | 8 |

## 4. Objective and Timing

| Metric | Value |
|--------|-------|
| Total cost | 45.4273 |
| Total duration | 22.8459 s |
| Setup time | 0.2158 s |
| Solve time | 1.4516 s |
| Paths evaluated | 2 |

## 5. Feasibility Diagnostics

| Metric | Value |
|--------|-------|
| Defect norm | 5.24e-10 |
| Max connection gap | 0.00e+00 |
| Max control jump | 0.00e+00 |
| Constraint violation | 0.00e+00 |
| Max integrality gap | 0.00e+00 |
| Min safety margin | -4.56e-10 |
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

- **JSON summary:** `results/crd_demo_summary.json`
- **PNG result:** `results/crd_demo_result.png`
- **GIF animation:** `results/crd_demo_animation.gif` *(if generated)*
