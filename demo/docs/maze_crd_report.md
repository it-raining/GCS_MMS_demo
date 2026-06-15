# GCS-DMS Optimization Report: `maze_crd`

## 1. Scenario

| Field | Value |
|-------|-------|
| Preset | `maze` |
| Description | Maze preset with CentroidRefineDMS solver |
| Number of regions | 171 |
| Number of edges | 442 |
| Candidate paths | 1500 |
| Start state | `[0.4, 2.0, 0.0]` |
| Goal state | `[4.8, 4.8, 0.785]` |

## 2. Solver Mode

| Field | Value |
|-------|-------|
| Mode | `Unknown` |
| Solver | IPOPT (via CasADi) |
| Problem class | Nonlinear program (NLP). Nonconvex due to unicycle dynamics. |
| Global optimality claim | LB is a geometric lower bound on time component only. No global optimality claim. The solution is a KKT point of the barrier-augmented NLP under LICQ+SOSC. |

## 3. Selected Path

| Field | Value |
|-------|-------|
| Path nodes | `N/A` |
| Path regions | `N/A` |
| Path length (regions) | 0 |

## 4. Objective and Timing

| Metric | Value |
|--------|-------|
| Total cost | N/A |
| Total duration | N/A s |
| Setup time | 18.1345 s |
| Solve time | 62.7619 s |
| Paths evaluated | 3240 |

## 5. Feasibility Diagnostics

| Metric | Value |
|--------|-------|
| Defect norm | 0.00e+00 |
| Max connection gap | 0.00e+00 |
| Max control jump | 0.00e+00 |
| Constraint violation | 0.00e+00 |
| Max integrality gap | 0.00e+00 |
| Min safety margin | N/A |
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

- **Status:** Optimization failed
- **Solver status:** `CRD: no feasible path found`
- JSON / PNG / GIF may not have been generated.
