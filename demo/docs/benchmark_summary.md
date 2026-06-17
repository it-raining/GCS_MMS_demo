# GCS-DMS Benchmark Summary

> **Important:** IPOPT provides a **locally optimal** NLP solution. No global optimality certificate is provided for any scenario.

| Scenario | Status | Regions | Edges | Paths | Cost | Solve Time (s) | Mode |
|----------|--------|---------|-------|-------|------|----------------|------|
| [default](default_report.md) | OK | 15 | 51 | 532 | 58.3189 | 14.602 | `centroid_refine_dms` |

## Individual Reports

- [default report](default_report.md)

## Formulation Notes

### Legacy integrated solver

The legacy integrated solver is a **continuous relaxation of a nonconvex MINLP** solved by IPOPT. It is **not** a certified globally optimal MICP/MISOCP solver. Big-M constraints are used for region activation, region containment, edge/interface containment, interface equality, and optionally control continuity.

### Two-stage solver

The two-stage solver selects candidate paths using a centroid-distance graph heuristic (not a certified GCS convex relaxation) and then solves a **fixed-path nonlinear DMS NLP** with IPOPT. No Big-M activation, containment, or interface equality constraints are needed because the path is fixed before the NLP is assembled.

### Why the DMS problem remains nonconvex

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
