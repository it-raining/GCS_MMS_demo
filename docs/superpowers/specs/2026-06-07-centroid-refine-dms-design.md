# Centroid–Refine–DMS Loop — Full Design Specification

**Date:** 2026-06-07  
**Status:** Design — awaiting implementation plan  
**Solver mode key:** `"centroid_refine_dms"` — dispatched via a dedicated `CentroidRefineDMSConfig` class; selected by passing a `CentroidRefineDMSConfig` to the solver entry point (see §1.1)  
**Primary objective:** First-feasible safe solution as fast as possible  
**Dynamics scope:** Unicycle (existing) + Double integrator (new)

---

## Design Decisions (from clarification)

| Dimension | Decision |
|---|---|
| Primary objective | First-feasible — return as soon as one safe, KKT-feasible solution is found |
| Dynamics | Unicycle (nonlinear, existing) + Double integrator (linear, new in `dynamics.py`) |
| GCS-Bézier baseline | Comparison slots only in `experiments.py`; no implementation required |
| Lower bound | Geometric LB (`path_length / v_max`) + explicit disclaimer in report |
| Graph reweighting | Path blacklist only; no adaptive edge-penalty |
| Mesh points | Removed from default path; barrier evaluated at RK4 integration nodes (zero overhead) |
| Barrier schedule | Full continuation with theoretically grounded μ₀, τ, μ_min |

---

## Assumptions

| ID | Statement |
|---|---|
| **A1** | All halfplane normals satisfy ‖a_{i,j}‖₂ = 1. Enforced by `ConvexRegion.__post_init__` which normalizes normals from V-rep. Under A1, s_{i,j,k} > 0 directly measures Euclidean distance to the boundary. |
| **A2** | The path (C_1,...,C_m) is fixed by graph preprocessing (Stages A–B). The DMS NLP is a fixed-path formulation; path selection is external to the optimization problem. |
| **A3** | The warm-start (x⁰, u⁰, Δ⁰) is a **feasible initial guess for the NLP**: all state nodes satisfy s_{i,j,k}(x⁰) ≥ δ_extra > 0 (proven in Part 4). It is NOT claimed to be IVP-feasible; the continuous trajectory from integrating u⁰ may deviate from the linear interpolation. |

---

## Part 0 — Region Decomposition Pipeline

### 0.1 GCS gốc vs. kỹ thuật hiện tại

**GCS gốc (Marcucci et al. 2022):** Xây dựng đồ thị tối ưu trên các convex region *chồng lấn nhau*. Tại vùng giao $C_i \cap C_j$, ràng buộc liên tục $x_i(1) = x_j(0)$ được đặt trực tiếp. Overlap là điều kiện **cần** để ràng buộc này có nghiệm không tầm thường. Các region trong GCS gốc thường được xây dựng bằng IRIS (Deits & Tedrake 2015) — thuật toán tự nhiên sinh ra các polytope chồng lấn vì nó inflation từ seed point cho đến khi chạm obstacle.

**Kỹ thuật hiện tại:** ACD2D phân hoạch workspace thành các đa giác lồi *không chồng lấn* (edge-touching partition). Để tạo overlap cần thiết, mỗi vùng ACD2D được mở rộng ra ngoài một lượng `overlap_width` bằng Shapely mitre buffer + workspace clip:

```python
buffered = poly.buffer(overlap_width, join_style='mitre')  # giu loi
clipped  = buffered.intersection(workspace_polygon)         # clip tai bien
```

`join_style='mitre'` giữ nguyên các góc sắc → polygon sau buffer vẫn lồi. Workspace clip loại bỏ phần mở rộng tại biên workspace → chỉ cạnh chung giữa các vùng lân cận được mở rộng thực sự. Giao $C_i \cap C_j$ là dải rộng $2 \cdot \text{overlap\_width}$ dọc cạnh chung.

### 0.2 Yêu cầu cứng

1. **Mọi solver mode** (`integrated_relaxation_legacy`, `two_stage`, `centroid_refine_dms`) đều đi qua pipeline: `ACD2D -> mitre buffer -> build_region_graph`. Không có code path bypass ACD2D.
2. `overlap_width >= delta_safe + delta_extra` để Interface QP luôn có Chebyshev radius đủ lớn. Startup warning được emit nếu vi phạm.
3. Adjacency detection dùng vùng ACD2D gốc (trước buffer) để tránh cạnh giả.
4. **Cạnh Point-contact bị loại bỏ cứng.** `_build_graph` phải kiểm tra `orig_inter.geom_type` sau `regions_intersect`; nếu là `'Point'` thì bỏ qua cạnh. Vertex-only contact là artifact ACD2D tại reflex vertex, không phải cửa đi thực sự — không có độ rộng để đặt waypoint.
5. **Cạnh có tightened intersection rỗng bị loại bỏ cứng.** Nếu `compute_intersection(ri, rj)` trả về `None`, cạnh không được thêm vào `region_edges`. Không có fallback stacked H-rep: stacked H-rep của hai vùng tách nhau là infeasible by construction, không có waypoint hợp lệ nào tồn tại. *(Xác nhận bằng diagnostic maze_crd 2026-06-08: 316/440 cạnh vi phạm với mean gap 0.13, max gap 2.57 — khiến toàn bộ 3240 paths thất bại tại Interface QP pre-check trước khi DMS được gọi.)*

### 0.3 Cấu hình

```yaml
region_decomposition:
  overlap_width: 0.5   # >= delta_safe + delta_extra cua moi CRD scenario
```

**Known risk:** ACD2D có thể tạo ra các cuts chéo (diagonal) tại reflex vertex của free space, dẫn đến region không theo trục (non-axis-aligned). Với region chéo, `delta_safe` cần giảm tương ứng để tránh KKT degenerate trong DMS NLP.

---

## Part 1 — Architecture and Data Flow

```
INPUT: RegionGraph G, x_start, x_goal, CentroidRefineDMSConfig
        │
        ▼
┌──────────────────────────────┐
│  STAGE A: CENTROID K-SHORTEST │  graph_builder.py
│  Yen's algorithm on G         │  edge cost = centroid-dist
│  cost += composite penalties  │            - γ_w · ρ_interface
│  blacklist enforced           │            + γ_h · |Δθ|²
└──────────────┬───────────────┘
               │ path P = (C₁,...,Cₘ)
        ▼
┌──────────────────────────────┐
│  STAGE B: INTERFACE QP        │  geometric_refiner.py  [NEW]
│  min ‖polyline‖² + λ_s·‖κ‖²  │
│  s.t. zᵢ ∈ int(Cᵢ ∩ Cᵢ₊₁)   │  ← strict interior guaranteed
│  solver: CasADi qpsol/osqp   │  typically <5 ms
└──────────────┬───────────────┘
               │ z₀,...,zₘ₊₁ with sᵢⱼ ≥ δ_extra > 0 (proven)
        ▼
┌──────────────────────────────┐
│  STAGE C: INITIAL GUESS GEN   │  warmstart.py  [NEW]
│  linear interp per segment    │
│  unicycle: θ from Δz, ω=Δθ/Δ │
│  double-int: const velocity   │
│  Δᵢ⁰ ∝ ‖zᵢ - zᵢ₋₁‖ / v_nom  │
│  NLP initial guess (not IVP)  │  see Assumption A3
└──────────────┬───────────────┘
               │ (x⁰, u⁰, Δ⁰) — state nodes in strict interior
        ▼
┌──────────────────────────────┐
│  STAGE D: BARRIER SCHEDULE    │  barrier_dms.py  [NEW]
│  Fixed levels: {1.0,0.5,     │
│    0.1, 0.01}                 │
│  Large μ first → small μ last │
└──────────────┬───────────────┘
               │
        ▼
┌──────────────────────────────────────────────┐
│  STAGE E: BARRIER CONTINUATION LOOP           │  barrier_dms.py
│  for r = 0, 1, ..., N_μ − 1:                │
│    μᵣ = μ₀ · τʳ                              │
│    ε(r) = max(ε_final, μᵣ)   ← inexact tol  │
│    solve DMS-barrier(μᵣ) via IPOPT            │
│    warm-start level r+1 from level r solution │
│    if IPOPT diverges → classify failure, break│
└──────────────┬───────────────────────────────┘
               │ (x*, u*, Δ*, KKT residuals)
        ▼
┌──────────────────────────────┐
│  STAGE F: DIAGNOSTICS         │
│  defect_norm, coupling_gap    │
│  KKT_residual                 │
│  s_min_sampled                │
│  certified_s_min (Lipschitz)  │
│  LB (geometric), UB update    │
│  gap = (UB−LB)/max(1,|UB|)   │
└──────────────┬───────────────┘
               │
     success? ──yes──► record UB → return immediately (first-feasible)
          │ no
     ┌────▼───────────────────┐
     │  STAGE G: FAILURE+RETRY │
     │  classify failure type  │
     │  blacklist path P        │
     │  next K-shortest path   │
     └─────────────────────────┘
```

### 1.1 Integration with existing codebase

`CentroidRefineDMSSolver` is a **new class** added to `optimizer.py` alongside the existing `PathNLPSolver` and integrated relaxation logic. Existing classes are untouched. The new mode is selected by passing a `CentroidRefineDMSConfig` (a separate dataclass, already defined in `optimizer.py`) to the solver entry point:

```python
# Caller passes CentroidRefineDMSConfig, not OptimizationConfig
if isinstance(config, CentroidRefineDMSConfig):
    solver = CentroidRefineDMSSolver(graph, dynamics, config)
    return solver.solve(x_start, x_goal)
```

`OptimizationResult` already has all needed diagnostic fields (`defect_norm`, `min_safety_margin`, etc.) — they are populated by the new solver, not modified in the dataclass. The solver also sets `result.safety_mode = "log_barrier"` to distinguish CRD results from legacy solvers that use `"both"` (CTCS + dense check).

---

## Part 2 — Mathematical Formulation

### 2.1 Notation

- Path: $(C_1, \ldots, C_m)$, each $C_i = \{q \mid A_i q \le b_i\}$, $A_i \in \mathbb{R}^{n_{H_i} \times n_{\text{pos}}}$
- Position extractor: $P \in \mathbb{R}^{n_{\text{pos}} \times n_x}$ (selects position indices from state)
- Normalized time on segment $i$: $\tau \in [0,1]$, real time $t = \Delta_i \cdot \tau$
- RK4 nodes on segment $i$: $\tau_{i,k} = k/n_{\text{int}}$, $k = 0, \ldots, n_{\text{int}}$
- Safety slack at node $(i, j, k)$:

$$s_{i,j,k} = b_{i,j} - \delta_{\text{safe}} - a_{i,j}^\top P\, x_i(\tau_{i,k})$$

The definition subtracts $\delta_{\text{safe}}$ from the halfplane offset so that $s_{i,j,k} > 0$ means the trajectory is inside the safety-shrunken region $C_i \ominus \delta_{\text{safe}}$.

**Velocity notation — critical distinction:**

| Symbol | Definition | Used in |
|---|---|---|
| $v_{q,i}(\tau) = P f(x_i(\tau), u_i(\tau))$ | Physical position velocity in real time $t$ | Objective cost |
| $\dot{x}_i(\tau) = \Delta_i \cdot f(x_i(\tau), u_i(\tau))$ | State derivative in normalized time $\tau$ | RK4 integration |

These differ by a factor of $\Delta_i$. **Only $v_{q,i}$ appears in the cost.** Equivalently, $\int_0^1 \Delta_i \|v_{q,i}\|^2 d\tau = \int_0^{\Delta_i} \|P f(x(t),u(t))\|^2 dt$ by the substitution $dt = \Delta_i d\tau$.

> **Paper/thesis note:** Express the velocity cost in physical time $t$ as $\int_0^{\Delta_i} w_V \|P f(x(t), u(t))\|^2\, dt$ to avoid confusion with the normalized-time state derivative.

### 2.2 Barrier-augmented objective

$$\boxed{J_\mu = w_T \sum_{i=1}^m \Delta_i
+ \sum_{i=1}^m \Delta_i \int_0^1 \left[ w_V \|v_{q,i}(\tau)\|_2^2 + w_U \|u_i(\tau)\|_2^2 \right] d\tau
+ w_S \sum_{i=1}^{m-1} \|u_i(1) - u_{i+1}(0)\|_2^2
- \mu \sum_{i=1}^m \sum_{j=1}^{n_{H_i}} \sum_{k=0}^{n_{\text{int}}} h_k \log s_{i,j,k}}$$

where $v_{q,i}(\tau) = P f(x_i(\tau), u_i(\tau))$ is the physical velocity (see notation table in §2.1). The weight $w_V$ corresponds to `w_L` in `OptimizationConfig`. Equivalent physical-time form: $\int_0^{\Delta_i} [w_V \|P f\|^2 + w_U \|u\|^2]\, dt$.

**Trapezoidal quadrature weights on RK4 nodes** (exact for the barrier integral approximation):

$$h_k = \begin{cases}
\Delta_i / (2\, n_{\text{int}}) & k \in \{0,\; n_{\text{int}}\} \\
\Delta_i / n_{\text{int}} & \text{otherwise}
\end{cases}$$

Note: $\Delta_i$ appears in $h_k$ because the quadrature is over real time $t \in [0, \Delta_i]$, not normalized $\tau$.

**Barrier evaluation domain:** Log-barrier is only defined when $s_{i,j,k} > 0$. The design guarantees this at initialization (see Part 4). During the NLP solve, IPOPT's interior-point method maintains $s > 0$ automatically via its own barrier. If IPOPT's restoration phase is triggered because $s \le 0$ was reached, this is classified as a solver failure.

### 2.3 Equality constraints (defect and coupling)

**Defect — never softened or penalized:**

$$r_{i,k}^{\text{def}} := x_i(\tau_{i,k+1}) - \Phi_{\Delta_i}(x_i(\tau_{i,k}),\, u_i) = 0$$

where $\Phi_{\Delta_i}$ is the RK4 flow map. These are hard equality constraints passed to IPOPT. Penalty-based relaxation of defects is explicitly excluded — it destroys dynamical feasibility.

**Coupling:**

$$x_i(1) = x_{i+1}(0), \quad i = 1,\ldots,m-1$$
$$x_1(0) = x_{\text{start}}, \quad x_m(1) = x_{\text{goal}}$$

### 2.4 Inequality constraints

Control and duration bounds are enforced as IPOPT **variable bounds** (not constraint rows) for numerical efficiency:

$$u_{\min} \le u_i(\tau) \le u_{\max}, \quad \Delta_{\min} \le \Delta_i \le \Delta_{\max}$$

Geometric halfplane constraints are handled entirely by the log-barrier — no explicit inequality rows for geometry. This eliminates the constraint-count overhead of the legacy mesh safety constraints.

### 2.5 NLP structure per barrier level

At level $r$:

$$\min_{x, u, \Delta} \; J_{\mu_r}(x, u, \Delta) \quad \text{s.t.} \quad r^{\text{def}} = 0,\; r^{\text{coup}} = 0,\; u \in \mathcal{U},\; \Delta \in \mathcal{D}$$

This is a smooth NLP with:
- $n_{\text{var}} = m \cdot (n_x \cdot (n_{\text{int}}+1) + n_u \cdot N_{\text{seg}} + 1)$ decision variables
- $n_{\text{eq}} = m \cdot n_x \cdot n_{\text{int}} + (m-1) \cdot n_x + 2 \cdot n_x$ equality constraints

For unicycle ($n_x=3, n_u=2$), $m=6$ regions, $n_{\text{int}}=20$, $N_{\text{seg}}=2$:  
$n_{\text{var}} \approx 6 \cdot (63 + 4 + 1) = 408$, $n_{\text{eq}} \approx 6 \cdot 60 + 5 \cdot 3 + 6 = 381$.  
Medium-scale NLP — well within IPOPT's efficient range.

---

## Part 3 — Graph Search and Geometric Refinement

### 3.1 Chebyshev center (region centroid)

For $C_i = \{q \mid A_i q \le b_i\}$, the Chebyshev center $c_i^*$ and radius $\rho_i^*$ are found via LP:

$$\max_{q,\, \rho} \;\rho \quad \text{s.t.} \quad a_{i,j}^\top q + \|a_{i,j}\|_2\, \rho \le b_{i,j} \quad \forall j, \quad \rho \ge 0$$

Solved once per region at graph construction time using `scipy.optimize.linprog` or CasADi LP. Result cached in `RegionGraph`.

**Why Chebyshev center over vertex mean:** For a non-regular polytope, the vertex mean lies near the boundary when the polytope is elongated. The Chebyshev center is the unique maximally interior point; its radius $\rho_i^*$ is directly interpretable as the safety clearance of the centroid — the natural metric for safety-aware graph search.

### 3.2 Composite edge cost

$$c(i \to j) = \underbrace{\|c_j^* - c_i^*\|_2}_{\text{centroid distance}} - \underbrace{\gamma_w\, \rho_{ij}^*}_{\text{interface bonus}} + \underbrace{\gamma_h\, |\Delta\theta_{ij}|^2}_{\text{turn penalty}}$$

where $\rho_{ij}^*$ is the Chebyshev radius of $C_i \cap C_{i+1}$ (the interface polytope — also an LP).

**Parameter derivations:**

| Param | Default | Derivation |
|---|---|---|
| $\gamma_w = 1.0$ | (dimensionless: same units as length) | Interface 1 unit wider reduces cost by 1; dimensional balance |
| $\gamma_h = v_{\text{nom}}/\omega_{\max}$ | Unicycle: $2.0/\pi \approx 0.64$ | Extra arc-length to execute heading change $|\Delta\theta|$ at speed $v_{\text{nom}}$ with max yaw $\omega_{\max}$: $|\Delta\theta| \cdot v_{\text{nom}}/\omega_{\max}$ |

**Source/target edge cost:** For edge from source to $C_1$: cost = distance from $q_{\text{start}}$ to $c_1^*$. Interface bonus uses $\rho_1^*$ (clearance of $q_{\text{start}}$ from $C_1$ boundary, or $\rho_1^*$ if inside).

### 3.3 K-shortest paths with blacklist

**Algorithm:** Yen's K-shortest simple paths — `networkx.shortest_simple_paths(G, source, target, weight='composite_cost')`. Returns paths in non-decreasing cost order. All are simple (no repeated nodes).

```python
tried_paths: set[tuple[str, ...]] = set()

def next_candidate(generator, tried_paths):
    for path in generator:
        key = tuple(path)
        if key not in tried_paths:
            tried_paths.add(key)
            return path
    return None  # exhausted
```

**Finite termination:** The set of simple paths in a finite directed graph is finite. The blacklist is monotone-increasing. The generator terminates in finite calls.

**Blacklist granularity:** Exact node-sequence tuples. Two paths visiting the same region set in different order are distinct candidates and may have different feasibility.

### 3.4 Interface refinement QP

For a path with $m$ regions there are exactly $m-1$ free interface points:

**Fixed:** $z_0 = q_{\text{start}}$, $\quad z_m = q_{\text{goal}}$

**Variables:** $z_1, \ldots, z_{m-1} \in \mathbb{R}^{n_{\text{pos}}}$, where $z_i \in C_i \cap C_{i+1}$, $\;i = 1,\ldots,m-1$

Segment $i$ (for $i = 1,\ldots,m$) connects $z_{i-1}$ to $z_i$ through region $C_i$; both endpoints lie in $C_i$.

$$\min_{z_1,\ldots,z_{m-1}} \sum_{i=0}^{m-1} \|z_{i+1} - z_i\|_2^2 + \lambda_s \sum_{i=1}^{m-2} \|z_{i+1} - 2z_i + z_{i-1}\|_2^2$$

**Constraints — strictly interior to intersection:**

$$a_{i,j}^\top z_i \le b_{i,j} - \delta_{\text{safe}} - \delta_{\text{extra}}, \quad \forall j,\quad i=1,\ldots,m-1 \quad\text{(region }C_i\text{)}$$
$$a_{i+1,j}^\top z_i \le b_{i+1,j} - \delta_{\text{safe}} - \delta_{\text{extra}}, \quad \forall j,\quad i=1,\ldots,m-1 \quad\text{(region }C_{i+1}\text{)}$$

with $\delta_{\text{extra}} = \tfrac{1}{2}\delta_{\text{safe}}$.

**QP output:** array $z \in \mathbb{R}^{(m+1) \times n_{\text{pos}}}$ — $z[0]=q_{\text{start}}$, $z[1],\ldots,z[m-1]$ optimized, $z[m]=q_{\text{goal}}$.

**Smoothness parameter:**

$$\lambda_s = \alpha_s \left(\frac{v_{\min}}{\omega_{\max}}\right)^2$$

Derivation: the curvature term $\|z_{i+1} - 2z_i + z_{i-1}\|^2 \approx (\kappa \cdot \bar{d})^2 \cdot \bar{d}^2$ where $\bar{d}$ is mean segment length and $\kappa$ is local curvature. Balance with length term $\bar{d}^2$ at $\kappa = \kappa_{\max} = \omega_{\max}/v_{\min}$ gives $\lambda_s = (v_{\min}/\omega_{\max})^2$.

**Default:** $\alpha_s = 0$ for first-feasible (pure shortest interface polyline). Set $\alpha_s \in [0.1, 1.0]$ when DMS failures are attributed to large heading changes.

**Pre-solve check:** Before calling the QP solver, compute $\rho_{ij}^*$ for each interface. If $\rho_{ij}^* < \delta_{\text{safe}} + \delta_{\text{extra}}$ for any $i$: infeasible by construction. Classify as `NARROW_INTERFACE`, skip to next path candidate.

**Solver:** CasADi `qpsol` with `'osqp'` backend. Problem size $\le m \cdot n_{\text{pos}} \approx 30$ variables. Runtime $< 5$ ms.

**Convexity:** The QP objective is strictly convex (positive definite Hessian $H = L^\top L + \lambda_s D^\top D$ where $L$, $D$ are finite-difference matrices, full-rank for $m \ge 2$). Unique global solution — no local optima.

---

## Part 4 — State/Control Initial Guess Generation and Strict-Interior Proof

### 4.1 Duration allocation

$$\Delta_i^0 = \max\!\left(\Delta_{\min},\; \frac{\|z_i - z_{i-1}\|_2}{v_{\text{nom}}}\right), \quad v_{\text{nom}} = \frac{v_{\max} + |v_{\min}|}{2} \cdot 0.5$$

Conservative $v_{\text{nom}}$ (half of mean speed range) avoids control saturation in the initial guess. If $\omega_i^0 = \Delta\theta_i^0 / \Delta_i^0 > \omega_{\max}$, increase $\Delta_i^0$ until $\omega_i^0 = \omega_{\max}$.

### 4.2 State and control initialization

**Unicycle** ($x = [p_x, p_y, \theta]^\top$, $u = [v, \omega]^\top$):

$$q_i^0(\tau) = (1-\tau)\, z_{i-1} + \tau\, z_i$$
$$\theta_i^0 = \text{atan2}(z_i^y - z_{i-1}^y,\; z_i^x - z_{i-1}^x)$$
$$x_i^0(\tau) = [q_i^0(\tau);\; \theta_i^0]$$
$$u_i^0 = \left[\frac{\|z_i - z_{i-1}\|_2}{\Delta_i^0};\; \frac{\theta_i^0 - \theta_{i-1}^0}{\Delta_i^0}\right] \quad \text{(clamped to control bounds)}$$

**Note (A3):** This is an initial guess for the NLP, not an IVP-feasible trajectory. When $\omega_i^0 \ne 0$, the trajectory integrated from $u_i^0$ is not a straight line and may deviate from $q_i^0(\tau)$. The strict-interior guarantee (Part 4.3) applies to the linear-interpolation state nodes $q_i^0(\tau_{i,k})$, not to the RK4-integrated trajectory.

**Double integrator** ($x = [p_x, p_y, v_x, v_y]^\top$, $u = [a_x, a_y]^\top$):

$$q_i^0(\tau) = (1-\tau)\, z_{i-1} + \tau\, z_i$$
$$\dot{q}_i^0 = (z_i - z_{i-1}) / \Delta_i^0 \quad \text{(constant velocity)}$$
$$x_i^0(\tau) = [q_i^0(\tau);\; \dot{q}_i^0], \quad u_i^0 = \mathbf{0}$$

### 4.3 Strict-interior guarantee (formal proof)

**Claim:** For all RK4 nodes $\tau_{i,k} \in [0,1]$: $s_{i,j,k}(x_i^0) \ge \delta_{\text{extra}} > 0$.

**Proof:**  
The QP constraint enforces: $b_{i,j} - a_{i,j}^\top z_\ell \ge \delta_{\text{safe}} + \delta_{\text{extra}}$ for $z_\ell \in \{z_{i-1}, z_i\}$ and all halfplane indices $j$ of region $C_i$.

The warm-start position is $q_i^0(\tau) = (1-\tau)z_{i-1} + \tau z_i$, a convex combination. The set $\{q : b_{i,j} - a_{i,j}^\top q \ge \delta_{\text{safe}} + \delta_{\text{extra}}\}$ is convex (halfspace). Therefore:

$$b_{i,j} - a_{i,j}^\top q_i^0(\tau) \ge \delta_{\text{safe}} + \delta_{\text{extra}} \quad \forall \tau \in [0,1]$$

The safety slack is:

$$s_{i,j,k} = b_{i,j} - \delta_{\text{safe}} - a_{i,j}^\top q_i^0(\tau_{i,k}) \ge \delta_{\text{extra}} > 0 \quad \square$$

This proves $B_0 = -\sum h_k \log s_{i,j,k} < +\infty$ (finite), so $\mu_0 = \alpha f_0 / B_0$ is well-defined.

---

## Part 5 — Barrier Schedule

### 5.1 Fixed barrier levels

The barrier parameter $\mu$ follows a fixed decreasing schedule. Interior-point methods begin with a large $\mu$ (loose safety enforcement, fast convergence) and decrease $\mu$ progressively, using each solution as the warm-start for the next level:

$$\mu \in \{1.0,\; 0.5,\; 0.1,\; 0.01\}$$

Reduction factor between consecutive levels is approximately $5\text{–}10\times$. The final level $\mu_{\text{final}} = 0.01$ balances safety enforcement against optimization quality.

**Why a fixed schedule instead of adaptive $\mu_0$:** Computing $\mu_0 = \alpha f_0 / B_0$ (Fiacco-McCormick) requires $B_0 > 0$, which is not guaranteed — when some slacks $s_{i,j,k} > 1$, the term $-\log s < 0$ makes $B_0$ potentially negative. The fixed schedule avoids this dependence entirely and is robust across problem scales.

### 5.2 Barrier continuation protocol

```
for μ in {1.0, 0.5, 0.1, 0.01}:
    ε_r = max(ε_final, μ)          # inexact tolerance (Nocedal & Wright §19.5)
    solve DMS-barrier NLP at μ, warm-starting from previous solution
    compute s_min_certified (Part 8)
    if s_min_certified > 0:
        ACCEPT → continue to next μ level (or stop if at μ_final)
    else:
        CONTINUE to next μ level (smaller μ may give more margin)

if certificate fails at μ_final = 0.01:
    DECLARE PATH INFEASIBLE → blacklist → try next K-shortest candidate
```

**Inexact tolerance per level:**

| Level | $\mu$ | $\varepsilon_{\text{KKT}}$ | IPOPT `max_iter` |
|---|---|---|---|
| 0 | 1.0 | 1.0 (loose) | 300 |
| 1 | 0.5 | 0.5 | 200 |
| 2 | 0.1 | 0.1 | 150 |
| 3 (final) | 0.01 | $\varepsilon_{\text{final}} = 10^{-6}$ | 300 |

### 5.3 KKT convergence claim

At the final level $\mu = 0.01$, convergence is to a **KKT point of the barrier-augmented NLP**, not of the original hard-constrained NLP. The barrier perturbation of magnitude $\mu = 0.01$ remains in the solution. This is the intended behavior — the barrier acts as a safety enforcement mechanism that persists at the final solution.

**Claim:** The final solution $(x^*, u^*, \Delta^*)$ satisfies the KKT conditions of $\min J_{\mu=0.01}$ subject to defect, coupling, and bound constraints, under LICQ + SOSC. This is a standard NLP result (Nocedal & Wright, Theorem 12.1). No global optimality claim is made.

**No claim about the original problem:** Setting $\mu \to 0$ would recover a KKT point of the original hard-constrained NLP, but this is not the goal of this formulation. The barrier at $\mu = 0.01$ is intentionally kept to enforce the interior-domain property.

### 5.4 Complete schedule summary

```python
BARRIER_LEVELS = [1.0, 0.5, 0.1, 0.01]
EPSILON_FINAL  = 1e-6

def ipopt_tol(mu: float) -> float:
    return max(EPSILON_FINAL, mu)

def ipopt_max_iter(mu: float) -> int:
    return 300 if mu >= 1.0 or mu <= 0.01 else 200 if mu >= 0.5 else 150
```

---

## Part 6 — Three-Tier Stopping Criteria

### Tier 1: Inner NLP convergence (per barrier level)

Passed directly to IPOPT at level $r$:

$$\|r^{\text{def}}\|_\infty \le \varepsilon_d(r), \quad \|r^{\text{coup}}\|_\infty \le \varepsilon_c(r), \quad \|r^{\text{KKT}}\|_\infty \le \varepsilon_{\text{KKT}}(r)$$

with $\varepsilon_d(r) = \varepsilon_c(r) = \varepsilon_{\text{KKT}}(r) = \max(\varepsilon_{\text{final}}, \mu_r)$ and $\varepsilon_{\text{final}} = 10^{-6}$.

**Claim:** Local KKT convergence of the barrier-augmented NLP under LICQ + SOSC. This is a standard NLP result (Nocedal & Wright, Theorem 12.1). It is **not** global optimality.

**Regularity assumption:** LICQ (linear independence of active constraint gradients) is expected to hold generically. If IPOPT reports degenerate convergence, classify as `KKT_DEGENERATE` failure.

### Tier 2: Outer barrier loop (per path candidate)

A path is declared **successful** when all $N_\mu$ inner NLPs converge. Update:

$$\text{UB}_k = \min\!\left(\text{UB}_{k-1},\; J^*(x^*, u^*, \Delta^*)\big|_{\mu=0}\right)$$

$\text{UB}_k$ is non-increasing by construction. The upper bound sequence $\{\text{UB}_k\}$ is monotone non-increasing and bounded below by 0 — it converges.

### Tier 3: Graph search termination

Stop when any of the following:

1. **First-feasible achieved:** `UB_k < +∞` and `config.mode == "first_feasible"` → return immediately
2. **Optimality gap sufficient:** $({\text{UB}_k - \text{LB}_k}) / \max(1, |\text{UB}_k|) \le \varepsilon_{\text{gap}}$ → return best
3. **Paths exhausted:** `next_candidate()` returns `None` (all simple paths tried) → return best found (possibly `None` if UB = +∞)
4. **Time budget exceeded:** wall-clock time > `config.time_limit_s` → return best found (anytime mode)

**No `max_iter` as convergence criterion:** The outer loop terminates by structural properties of the path space, not by an arbitrary iteration count. This distinguishes finite-search convergence from optimality convergence.

---

## Part 7 — Lower Bound, Upper Bound, and Gap

### 7.1 Geometric lower bound

$$\text{LB}_{\text{geom}} = \frac{D^*_{\text{path}}}{v_{\max}}$$

where $D^*_{\text{path}}$ is the shortest obstacle-free polyline length from $q_{\text{start}}$ to $q_{\text{goal}}$ through any sequence of convex regions (computed as Dijkstra with Euclidean edge weights on the centroid graph, without composite penalties — pure geometry).

**What LB covers:** Only the time component $w_T \sum_i \Delta_i$ of the objective, since $D^*_{\text{path}} / v_{\max}$ is the minimum possible travel time ignoring dynamics, control effort, and inertia.

**What LB does NOT cover:** Velocity cost $w_L \|\dot{q}\|^2$, control effort $w_U \|u\|^2$, smoothness cost $w_S J_{\text{smooth}}$. The full DMS objective is strictly greater than $w_T \cdot \text{LB}_{\text{geom}}$ for any real trajectory.

### 7.2 Optimality gap

$$\text{gap}_k = \frac{\text{UB}_k - w_T \cdot \text{LB}_{\text{geom}}}{\max(1, |\text{UB}_k|)}$$

### 7.3 Mandatory disclaimer in all reports

Every `OptimizationResult` must include (as `global_optimality_claim`):

> "LB is a geometric lower bound on the time component only. gap_k is NOT a certificate for the full DMS objective. No global optimality claim is made. The reported solution is a KKT point of the fixed-path barrier-augmented NLP under LICQ + SOSC."

This text is set by `CentroidRefineDMSSolver` and is not user-configurable.

---

## Part 8 — Safety Certification

### 8.1 Sampled safety

After solve:

$$s_{\min}^{\text{sampled}} = \min_{i,j,k} s_{i,j,k}(x^*)$$

This is the minimum safety slack at the $n_{\text{int}}+1$ RK4 nodes across all segments and halfplanes.

### 8.2 Continuous Lipschitz certificate

Between consecutive RK4 nodes $\tau_{i,k}$ and $\tau_{i,k+1}$ (separated by $h_{\text{rk4}} = \Delta_i / n_{\text{int}}$ in real time), the rate of change of $s_{i,j}(t)$ is bounded:

$$|\dot{s}_{i,j}(t)| = |a_{i,j}^\top P f(x(t), u(t))| \le \|a_{i,j}\|_2 \cdot \|P f(x,u)\|_2$$

**For unicycle:** $\|P f(x,u)\|_2 = |v| \le v_{\max}$, so $|\dot{s}_{i,j}| \le \|a_{i,j}\|_2 \cdot v_{\max}$.

**For double integrator:** $\|P f(x,u)\|_2 = \|[v_x, v_y]\|_2 \le v_{\max}$, same bound.

Define the global Lipschitz constant for safety slack:

$$L_s = \max_{i,j} \|a_{i,j}\|_2 \cdot v_{\max}$$

By the mean value theorem applied to $s_{i,j}(t)$ on $[t_k, t_{k+1}]$, the worst-case deviation from any sampled value is $L_s \cdot h_{\text{rk4}}/2$ (maximum at the midpoint).

**Certified continuous minimum:**

$$s_{\min}^{\text{certified}} = s_{\min}^{\text{sampled}} - L_s \cdot \frac{h_{\text{max}}}{2} - \varepsilon_{\text{defect}} - \varepsilon_{\text{int}}$$

where:
- $h_{\text{max}} = \max_i \Delta_i / n_{\text{int}}$ — worst-case RK4 step size across all segments (uses per-segment $\Delta_i$, not a single $h$)
- $\varepsilon_{\text{defect}}$ — defect/coupling residual at the IPOPT solution ($\approx$ IPOPT `tol` setting)
- $\varepsilon_{\text{int}}$ — RK4 integration error ($\approx L_f \cdot h_{\text{max}}^4 / 30$, typically negligible vs. $\varepsilon_{\text{defect}}$)

**Requirement for certificate to hold:** $\delta_{\text{safe}}$ must satisfy

$$\delta_{\text{safe}} \ge L_s \cdot \frac{h_{\text{max}}}{2} + \varepsilon_{\text{defect}}$$

This is computed at runtime from `dynamics.compute_lipschitz_bound()` and the config. See Rule R13.

**Interpretation:**
- $s_{\min}^{\text{certified}} > 0$: trajectory is certifiably safe for all $t$ (not just at nodes)
- $s_{\min}^{\text{certified}} \le 0$: certificate fails for this path — declare infeasible, blacklist, try next K-shortest candidate (Q7 decision)

Report both values. Never conflate `sampled_safe` with `certified_safe`.

### 8.3 Certificate in `OptimizationResult`

```python
result.safety_mode             = "log_barrier"       # distinguishes CRD from legacy "both"
result.min_safety_margin       = s_min_sampled       # existing field
result.certified_safety_margin = s_min_certified     # new field
result.lipschitz_gap           = L_s * h_rk4 / 2    # new field
result.safety_certification    = (
    "CERTIFIED_CONTINUOUS_SAFE" if s_min_certified > 0
    else "SAMPLED_SAFE_ONLY"
)
```

**Note:** CTCS integrals (`max_continuous_violation_integral`, `continuous_violation_integrals`) and `max_dense_region_violation` are legacy fields for the `"both"` safety mode and are **not populated** by `CentroidRefineDMSSolver`. Visualization and reporting code must branch on `result.safety_mode` to display the correct diagnostics.

---

## Part 9 — Pseudocode

```
ALGORITHM: CentroidRefineDMSSolver.solve(x_start, x_goal)

INPUT:   RegionGraph G, dynamics model f, CentroidRefineDMSConfig cfg
OUTPUT:  OptimizationResult (best feasible found, or failure)

── INITIALIZATION ────────────────────────────────────────────────
tried_paths ← ∅
UB ← +∞
best_result ← None
LB_geom ← dijkstra_path_length(G, q_start, q_goal) / v_max
path_gen ← k_shortest_paths_generator(G, source, target, weight='composite_cost')

── OUTER LOOP (over path candidates) ─────────────────────────────
LOOP:
  P ← next_candidate(path_gen, tried_paths)
  if P is None: BREAK  ← exhausted all simple paths

  tried_paths.add(P)
  (C₁,...,Cₘ) ← regions_from_path(P, G)

  ── STAGE B: NARROW INTERFACE CHECK ────────────────────────────
  for i = 1,...,m-1:
    ρ_ij ← chebyshev_radius(C_i ∩ C_{i+1})
    if ρ_ij < δ_safe + δ_extra:
      log_failure(P, NARROW_INTERFACE, i)
      continue LOOP

  ── STAGE B: GEOMETRIC QP ──────────────────────────────────────
  z ← solve_interface_qp(P, q_start, q_goal, δ_safe, δ_extra, λ_s)
  if QP_INFEASIBLE:
    log_failure(P, QP_INFEASIBLE)
    continue LOOP

  ── STAGE C: INITIAL GUESS ─────────────────────────────────────
  (x⁰, u⁰, Δ⁰) ← generate_initial_guess(z, dynamics, cfg)
  assert all s_ijk(x⁰) ≥ δ_extra  ← guaranteed for state nodes (A3)

  ── STAGE D: BARRIER INIT ──────────────────────────────────────
  ← Fixed schedule: μ ∈ {1.0, 0.5, 0.1, 0.01}

  ── STAGE E: BARRIER CONTINUATION ──────────────────────────────
  x_w, u_w, Δ_w ← x⁰, u⁰, Δ⁰
  path_ok ← False
  fail_type ← None

  for μ_r in {1.0, 0.5, 0.1, 0.01}:
    ε_r ← max(ε_final, μ_r)
    ipopt_cfg ← {tol: ε_r, max_iter: ipopt_iters(μ_r)}

    nlp_result ← solve_dms_barrier_nlp(
      path=P, μ=μ_r,
      warm_start=(x_w, u_w, Δ_w),
      ipopt_cfg=ipopt_cfg
    )

    if nlp_result.status in {SOLVE_SUCCEEDED, ACCEPTABLE}:
      x_w, u_w, Δ_w ← nlp_result.solution
      ── CERTIFICATE CHECK (every level) ─────────────────────────
      s_cert ← compute_s_min_certified(x_w, Δ_w, G, dynamics)
      if s_cert > 0 and μ_r == 0.01:  ← final level passed
        path_ok ← True
        BREAK
      ← if certificate fails at intermediate level: continue to smaller μ
      ← if certificate fails at final level (μ=0.01): path_ok stays False
    else:
      fail_type ← classify_dms_failure(nlp_result)
      BREAK  ← do not continue barrier levels on this path

  if not path_ok and fail_type is None:  ← all levels ran, certificate never passed
    fail_type ← CERTIFICATE_FAIL  ← same as infeasible: blacklist

  ── STAGE F: DIAGNOSTICS AND UB UPDATE ─────────────────────────
  if path_ok:
    J_star ← evaluate_cost(x_w, u_w, Δ_w, cfg, μ=0)
    diag ← compute_diagnostics(x_w, u_w, Δ_w, P, G, dynamics)
    ← defect_norm, coupling_gap, KKT_residual
    ← s_min_sampled, s_min_certified (Lipschitz)

    if J_star < UB:
      UB ← J_star
      best_result ← build_result(x_w, u_w, Δ_w, P, diag, LB_geom, UB)

    gap ← (UB − LB_geom * w_T) / max(1, |UB|)

    ── TERMINATION CHECKS ─────────────────────────────────────
    if cfg.mode == FIRST_FEASIBLE: BREAK  ← primary mode
    if gap ≤ ε_gap: BREAK
    if wall_clock > cfg.time_limit_s: BREAK

  else:
    log_failure(P, fail_type)
    ← blacklist enforced via tried_paths (no adaptive penalty)

── END OUTER LOOP ─────────────────────────────────────────────────

── POST-PROCESSING ────────────────────────────────────────────────
if best_result is not None:
  best_result.global_optimality_claim ← MANDATORY_DISCLAIMER
  best_result.lb_geometric ← LB_geom
  best_result.optimality_gap ← gap
  best_result.n_paths_evaluated ← len(tried_paths)
  best_result.n_barrier_levels ← N_μ

return best_result  ← None if no feasible path found
```

---

## Part 10 — Failure Taxonomy and Fallback Hierarchy

### 10.1 Failure taxonomy

| Code | Description | Diagnosis | Fallback |
|---|---|---|---|
| `NARROW_INTERFACE` | Chebyshev radius of $C_i \cap C_{i+1} < \delta_{\text{safe}} + \delta_{\text{extra}}$ | Graph edge physically too narrow for safe passage | Blacklist path; next K-shortest |
| `QP_INFEASIBLE` | Interface QP has no feasible point | Should be caught by narrow interface check; if triggered, geometry data inconsistent | Blacklist; check graph builder |
| `WARM_START_VIOLATION` | $\exists (i,j,k): s_{i,j,k}(x^0) \le 0$ | Assertion failure — strict-interior proof violated; indicates a bug | **Bug** — halt and report |
| `IPOPT_RESTORATION` | IPOPT entered restoration phase (equality feasibility recovery) | Defect or coupling constraint far from feasibility; warm-start quality poor at this level | Blacklist path; next K-shortest |
| `IPOPT_MAX_ITER` | Reached `max_iter` without convergence at some level $r$ | NLP too hard at this barrier level; warm-start was far | Blacklist; next K-shortest |
| `CONTROL_SATURATION` | $u^*$ hits bounds throughout; objective is dominated by constraint activity | Path requires more agility than vehicle can provide | Blacklist; try path with fewer turns |
| `DURATION_SATURATION` | $\Delta_i^* = \Delta_{\max}$ for some $i$ | Path too long for region; $\Delta_{\max}$ should be increased in config | Blacklist; increase $\Delta_{\max}$ in config and retry (not automatic) |
| `CONNECTION_GAP_LARGE` | $\|x_i(1) - x_{i+1}(0)\|_\infty > \varepsilon_c$ post-solve | Coupling constraints not satisfied; IPOPT converged to locally infeasible point | Blacklist |
| `KKT_DEGENERATE` | IPOPT reports "Converged to a locally infeasible point" or "Problem may be infeasible" | LICQ likely violated; degenerate geometry | Blacklist |
| `NO_PATH_FOUND` | `next_candidate()` exhausted | All simple paths tried, none feasible | Return `None` |

### 10.2 Fallback hierarchy

```
Primary:   blacklist P, try next K-shortest candidate (always)
Secondary: if DURATION_SATURATION: notify user to increase Δ_max
Tertiary:  if NO_PATH_FOUND and problem is known to have a solution:
             → fall back to solver_mode = "integrated_relaxation_legacy"
             → this is a manually configured fallback, not automatic
```

No automatic parameter tuning (μ, mesh, etc.) across failures. Each failure is fully described in `result.failure_log` (list of dicts with path, failure code, level at failure, residuals at failure).

---

## Part 11 — Comparison with GCS-Bézier (Experimental Slots)

In `experiments.py`, the `ScenarioResult` dataclass adds comparison fields:

```python
@dataclass
class ScenarioResult:
    # existing fields ...
    
    # Centroid-Refine-DMS diagnostics
    crd_solve_time: float
    crd_time_to_first_feasible: float
    crd_success: bool
    crd_objective: float
    crd_path_length: float
    crd_travel_time: float
    crd_defect_norm: float
    crd_connection_gap: float
    crd_s_min_sampled: float
    crd_s_min_certified: float
    crd_n_paths_tried: int
    crd_n_barrier_levels: int
    crd_n_nlp_iterations: int
    crd_optimality_gap: float       # (UB - LB_geom*w_T) / max(1, UB)
    crd_lb_geometric: float
    
    # GCS-Bézier baseline (filled from external source / paper)
    gcs_bezier_solve_time: float    = float("nan")
    gcs_bezier_objective: float     = float("nan")
    gcs_bezier_path_length: float   = float("nan")
    gcs_bezier_travel_time: float   = float("nan")
    gcs_bezier_s_min: float         = float("nan")
    gcs_bezier_degree: int          = -1
    gcs_bezier_source: str          = "not_set"  # e.g., "Drake GCS", "paper Table 2"
```

The `gcs_bezier_*` fields are `nan` / sentinel by default. They are filled by manual assignment or a thin loader function reading a JSON reference file. **No GCS-Bézier solver is implemented in this repo.**

---

## Part 12 — Ablation Study Design

Five components to isolate, each with a binary on/off flag in `CentroidRefineDMSConfig`:

| Ablation | Flag | Baseline condition | Ablated condition |
|---|---|---|---|
| A1: Centroid K-shortest | `use_centroid_cost` | Chebyshev-center composite cost | Uniform edge cost (Dijkstra, no penalties) |
| A2: Geometric QP refinement | `use_interface_qp` | QP-refined interface points | Naive: $z_i = c_i^* \cap c_{i+1}^*$ midpoint (no QP) |
| A3: Log-barrier safety | `use_log_barrier` | Log-barrier in DMS objective | Hard inequality constraints (explicit, no barrier) |
| A4: Barrier continuation | `use_barrier_continuation` | Full $N_\mu$ levels with $\tau=0.1$ | Single solve at $\mu = \mu_{\min}$ (no continuation) |
| A5: Inexact tolerance schedule | `use_inexact_tolerance` | $\varepsilon(r) = \max(\varepsilon_{\text{final}}, \mu_r)$ | Fixed $\varepsilon = \varepsilon_{\text{final}}$ at all levels |

**Metrics per ablation run:**

- Time to first feasible (wall clock)
- Success rate (over N scenarios)
- Final objective value $J^*$
- $s_{\min}^{\text{sampled}}$ and $s_{\min}^{\text{certified}}$
- Number of paths tried before success
- Total NLP iterations across all barrier levels
- Defect norm at final solution

**Protocol:**
- Same environment, partition, start/goal for all ablation runs
- 5 random seeds per scenario × 3 environments = 15 runs per ablation configuration
- All runs with `mode = "first_feasible"` (primary objective)
- Time budget: 60 seconds per run (report success if found, otherwise `None`)

**Expected findings:**
- A1 (centroid cost) reduces paths tried before success
- A2 (QP refinement) reduces `IPOPT_RESTORATION` failures and improves $s_{\min}$
- A3 (log-barrier) vs hard constraints: comparable success rate, but barrier gives smoother convergence and better warm-starting for continuation
- A4 (continuation) vs single-level: continuation more robust at extreme safety margins; single-level faster but fragile if $\mu_{\min}$ is too small for IPOPT to handle cold
- A5 (inexact tolerance): reduces total NLP iterations ~2–3× at early levels with negligible impact on final quality

---

## Part 13 — File Mapping and Implementation Scope

### New files

| File | Role | Key classes/functions |
|---|---|---|
| `demo/geometric_refiner.py` | Interface QP solver | `InterfaceQP`, `solve_interface_refinement()`, `chebyshev_center()` |
| `demo/warmstart.py` | IVP generation from interfaces | `generate_warm_start_ivp()` (dispatches on `DynamicsModel` type) |
| `demo/barrier_dms.py` | Barrier continuation core | `BarrierSchedule`, `BarrierDMSSolver`, `classify_dms_failure()` |

### Modified files

| File | Changes |
|---|---|
| `demo/optimizer.py` | Add `CentroidRefineDMSSolver`; add `solver_mode == "centroid_refine_dms"` dispatch; add `lb_geometric`, `certified_safety_margin`, `lipschitz_gap`, `safety_certification` fields to `OptimizationResult` |
| `demo/constraint_layers.py` | Add `build_barrier_log_terms()`; add `compute_lipschitz_safety_gap()` |
| `demo/dynamics.py` | Add `DoubleIntegratorDynamics` subclass; add `compute_lipschitz_bound()` method to `DynamicsModel` |
| `demo/graph_builder.py` | Add `chebyshev_center()`, `composite_edge_cost()`, `k_shortest_paths_generator()` |
| `demo/experiments.py` | Add `ScenarioResult` comparison slots (Part 11); add ablation runner |
| `demo/config.yaml` | Add `centroid_refine_dms:` block (see below) |

### New config block

```yaml
centroid_refine_dms:
  # Graph search
  gamma_w: 1.0                  # interface clearance bonus weight
  gamma_h: 0.64                 # heading-change penalty weight (v_nom/omega_max)

  # Geometric QP
  alpha_s: 0.0                  # smoothness weight (0 = pure shortest polyline)
  delta_extra: 0.01             # strict-interior margin = δ_safe / 2 (auto-computed if null)

  # Warm-start
  v_nom_fraction: 0.5           # v_nom = v_nom_fraction * (v_max + |v_min|) / 2

  # Barrier schedule — fixed levels, large-to-small (interior-point style)
  # Production: BarrierSchedule.fixed() uses barrier_levels only.
  # alpha_mu / tau / mu_min are legacy parameters for BarrierSchedule.from_warm_start()
  # (ablation-only; not used in the production path).
  barrier_levels: [1.0, 0.5, 0.1, 0.01]  # production schedule
  alpha_mu: 0.1                 # ablation only: μ₀ = alpha_mu * f₀ / B₀
  tau: 0.1                      # ablation only: reduction factor
  mu_min: 1.0e-5                # ablation only: terminal level for from_warm_start

  # Stopping criteria
  epsilon_final: 1.0e-6         # final IPOPT KKT tolerance (at mu=0.01 level)
  epsilon_gap: 0.05             # near-optimality gap threshold (anytime mode only)
  time_limit_s: 60.0            # wall-clock budget (anytime mode)

  # Solver mode
  mode: "first_feasible"        # "first_feasible" | "anytime"

  # Safety certificate — on failure at mu_final: declare infeasible, blacklist path
  # delta_safe is auto-raised to: max(config value, L_s * h_max/2 + epsilon_final)

  # Ablation flags (all true = full proposed method)
  use_centroid_cost: true
  use_interface_qp: true
  use_log_barrier: true
  use_barrier_continuation: true
  use_inexact_tolerance: true
```

---

## Implementation Rules and Design Constraints

These rules apply during all future implementation work on this module. They take precedence over general repo conventions where they conflict.

### R1 — Never soften defect constraints

Defect equalities $r^{\text{def}} = 0$ must be passed to IPOPT as hard equalities, not as penalties or augmented Lagrangian terms. Any implementation that converts defects to penalty terms (e.g., $\min J + \rho \|r^{\text{def}}\|^2$) violates dynamical feasibility and is forbidden.

### R2 — Barrier only on safety halfplanes

The log-barrier is exclusively for the geometric safety constraints $s_{i,j,k} \ge 0$. Control bounds, duration bounds, and coupling constraints must remain as hard constraints (variable bounds or equality rows). Do not introduce barrier terms for any other constraint class.

### R3 — Strict-interior invariant at initialization

At the start of barrier level $r=0$, the assertion `all(s_ijk(x0) >= delta_extra)` must hold. This is verified programmatically (not just assumed). If the assertion fails, it indicates a bug in the QP or warm-start generator — halt with a clear error, not a silent fallback.

### R4 — No global optimality claims

`global_optimality_claim` must always be set to the mandatory disclaimer string (Part 7.3). The solver result must never use the words "globally optimal", "optimal", or "certified optimal" in any field or log message. Permitted: "KKT-feasible", "locally optimal under LICQ+SOSC", "best found".

### R5 — Sampled vs. certified safety are always both reported

Both `min_safety_margin` (sampled) and `certified_safety_margin` (Lipschitz-corrected) must be populated in every successful `OptimizationResult`. Reporting only one is an error.

### R6 — Barrier parameters are derived, not hardcoded

$\mu_0$ is computed from $(f_0, B_0)$ at runtime. $\mu_{\min}$ is computed from $(\varepsilon_{\text{KKT}}, \delta_{\text{extra}})$. $N_\mu$ is derived from $(\mu_0, \mu_{\min}, \tau)$. The values in `config.yaml` are overrides for expert use; the default path uses the derived values. Document any override clearly.

### R7 — Failure classification is mandatory

Every path candidate that fails must produce a failure record in `result.failure_log` with: path sequence, failure code, barrier level at failure, IPOPT status string, defect norm, and coupling gap at the point of failure. Silent failures are not permitted.

### R8 — QP and NLP solvers are independent

The interface QP (CasADi qpsol/osqp) and the DMS NLP (CasADi nlpsol/ipopt) must not share state or variable symbols. They are constructed independently for each path candidate. This enables clean teardown and re-initialization per candidate without residual CasADi graph state.

### R9 — DoubleIntegratorDynamics must implement compute_lipschitz_bound()

The `DynamicsModel` abstract class must add an abstract method `compute_lipschitz_bound(u_max)` returning $L_s = \max_{i,j} \|a_{i,j}\|_2 \cdot v_{\max}$. Both `UnicycleDynamics` and `DoubleIntegratorDynamics` must implement it. `constraint_layers.py`'s `compute_lipschitz_safety_gap()` calls this method — not a hardcoded formula.

### R10 — Ablation flags are first-class

`CentroidRefineDMSConfig.use_*` flags must be respected throughout the implementation — they are not dead code. `use_interface_qp=False` must fall through to the naive midpoint initialization. `use_log_barrier=False` must fall through to explicit inequality constraints via `constraint_layers.build_fixed_path_geometry_constraints()`. Each ablation path must produce a valid `OptimizationResult`.

### R11 — No mesh points in default path

`n_mesh_points` from the existing config must not be read by `CentroidRefineDMSSolver`. If a developer mistakenly adds mesh-point-based constraints to the barrier DMS code path, the code review should catch this as a violation. Mesh points remain available in `PathNLPSolver` (existing code) and the `"barrier_at_mesh"` safety mode (optional, separate implementation).

### R12 — GCS-Bézier slots are data fields, not solver calls

The `gcs_bezier_*` fields in `ScenarioResult` must never be populated by calling any internal solver. They are populated by reading a reference JSON file or by direct assignment from the experiment script. Do not create a GCS-Bézier solver class.

### R13 — δ_safe is computed from the Lipschitz bound, not hardcoded

`delta_safe` is not a magic number. It must satisfy $\delta_{\text{safe}} \ge L_s \cdot h_{\text{max}} / 2 + \varepsilon_{\text{defect}}$. Compute it at solver initialization:

```python
h_max = config.delta_max / config.n_integration_steps
L_s   = dynamics.compute_lipschitz_bound(all_region_A_matrices)
delta_safe_min = L_s * h_max / 2 + config.tol
config.delta_safe = max(config.delta_safe, delta_safe_min)
```

If the user-provided `delta_safe` is smaller than `delta_safe_min`, emit a warning and raise it automatically. Document the computed value in `OptimizationResult` diagnostics.

---

## Open Questions for Implementation Phase

1. **IPOPT warm-start format:** Verify that CasADi's IPOPT interface supports passing `lam_x0` and `lam_g0` (dual variable warm-start) across barrier levels. If not, primal-only warm-starting is used (acceptable — theory still holds, just fewer iterations saved).

2. **QP feasibility when `delta_extra` is large:** For narrow-corridor environments (e.g., the maze preset), `δ_safe + δ_extra = 1.5 · δ_safe` may render many interfaces infeasible. Consider a fallback: if QP infeasible with `δ_extra`, retry with `δ_extra = 0.1 · δ_safe` and flag `warm_start_margin = "reduced"` in diagnostics.

3. **Heading initialization at start/goal:** `θ_start` and `θ_goal` from config are full state boundary conditions. The warm-start sets `θ_i^0` from the interface geometry, which may conflict with `θ_start`. Handle by: segment 1 warm-start uses `θ_start` (not computed from $z_0 \to z_1$ direction) and applies heading change gradually over the segment duration.

4. **Double integrator velocity bounds:** `v_max` for double integrator must be read from `dynamics.v_max` (a new field), not from the unicycle angular velocity config. Confirm config schema extension.

5. **`compute_lipschitz_bound()` for compound regions:** When the trajectory passes through multiple halfplanes with different normal magnitudes, $L_s$ uses the max over all active halfplanes and all segments. This is conservative but safe. A tighter per-segment bound is possible but adds complexity — defer to post-implementation optimization.
