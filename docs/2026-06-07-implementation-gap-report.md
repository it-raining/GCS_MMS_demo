# Gap Report: Centroid-Refine-DMS — Codebase vs. Design Spec

**Ngày kiểm chứng:** 2026-06-07  
**Spec tham chiếu:** `docs/superpowers/specs/2026-06-07-centroid-refine-dms-design.md`  
**Codebase kiểm chứng:** `demo/`  
**Kết luận tổng thể:** Core algorithm **~95% hoàn thành**; infrastructure thực nghiệm và một số tham số còn thiếu.

---

## Kết quả Demo sau khi fix (2026-06-07)

| Scenario | Status | Regions | Paths | Cost | Time |
|----------|--------|---------|-------|------|------|
| `crd_demo` (15-vùng ACD2D, 5×5) | ✅ SUCCESS | 15 | 2 | 33.51 | 1.89s |
| `maze_15_crd` (3-vùng ACD2D, 15×15) | ✅ SUCCESS | 3 | 1 | 85.76 | 0.71s |
| `default_crd` (15-vùng ACD2D, 5×5) | ✅ SUCCESS | 15 | 3 | 32.89 | ~7s |

**Lưu ý:** Sau khi bỏ `skip_acd`, ACD2D tạo ra các cuts chéo cho `maze_15` (3 regions thay vì 8). Path R2→R1→R0 thành công với `delta_safe=0.1, delta_max=20.0`.

### Fixes đã áp dụng

| Fix | File | Mô tả |
|-----|------|--------|
| ✅ NaN log-barrier | `constraint_layers.py` | `ca.fmax(s_ijk,1e-10)` trước `ca.log()` |
| ✅ Safety constraints | `barrier_dms.py` | Explicit halfplane inequalities vào NLP |
| ✅ `use_log_barrier=False` path | `barrier_dms.py`, `optimizer.py` | Single-solve bypass khi không cần log-barrier objective |
| ✅ Trajectory visualization | `optimizer.py._build_result` | Populate `trajectories`/`mesh_samples` từ `x_nodes_opt` |
| ✅ Coupling_r measurement | `barrier_dms.py.solve_path` | Chỉ đo n_equality (defect+coupling+boundary), không đo safety slacks |
| ✅ MAZE_15_PROBLEM preset | `problem_data.py` | 8 vùng chồng lấn, workspace 15×15, rho≥1.0 mọi interface |
| ✅ `skip_acd=True` bypass ACD2D | `problem_data.py`, `scenario_builder.py` | Xóa flag; mọi preset đi qua ACD2D → mitre buffer |
| ✅ `buffer_size=0.001` quá nhỏ | `scenario_builder.py`, `config.yaml` | Tăng lên `overlap_width=0.5` từ block `region_decomposition` |
| ✅ Thiếu block `region_decomposition` | `config.yaml` | Thêm `region_decomposition.overlap_width: 0.5` |
| ✅ `maze_15_crd` scenario | `config.yaml` | delta_safe=0.3, start=[1,1], goal=[1,14] |

---


## 1. Trạng thái theo file

### 1.1 File mới (3 file yêu cầu)

| File | Trạng thái | Ghi chú |
|------|------------|---------|
| `demo/geometric_refiner.py` | ✅ HOÀN CHỈNH | InterfaceQP, solve_interface_refinement(), chebyshev_center() — đầy đủ |
| `demo/warmstart.py` | ✅ HOÀN CHỈNH | generate_warm_start_ivp() dispatch unicycle + double integrator — đúng |
| `demo/barrier_dms.py` | ✅ HOÀN CHỈNH | BarrierSchedule, BarrierDMSSolver, classify_dms_failure(), FailureCode enum — đầy đủ |

### 1.2 File sửa đổi (6 file yêu cầu)

| File | Trạng thái | Gaps chính |
|------|------------|-----------|
| `demo/optimizer.py` | ✅ GẦN HOÀN CHỈNH | LB dùng vertex mean thay vì Chebyshev center (Part 7.1) |
| `demo/constraint_layers.py` | ✅ HOÀN CHỈNH | build_barrier_log_terms(), compute_lipschitz_safety_gap() — đúng |
| `demo/dynamics.py` | ✅ HOÀN CHỈNH | DoubleIntegratorDynamics, compute_lipschitz_bound() abstract + cả hai implement |
| `demo/graph_builder.py` | ⚠️ THIẾU 1 HẠNG MỤC | **Turn penalty γ_h·\|Δθ\|² không có** trong composite_edge_cost |
| `demo/experiments.py` | ❌ THIẾU NHIỀU | ~20 fields của ScenarioResult chưa có; ablation runner chưa có |
| `demo/config.yaml` | ❌ THIẾU HOÀN TOÀN | Block `centroid_refine_dms:` không tồn tại trong file |

---

## 2. Gaps chi tiết

### 2.0 `demo/graph_builder.py` — Thiếu lọc cạnh không hợp lệ (CRITICAL, 2026-06-08)

`_build_graph` hiện thêm tất cả cạnh được phát hiện bởi `regions_intersect` vào `region_edges`, kể cả khi `compute_intersection` trả về `None`. Tạo ra hai lớp cạnh không hợp lệ:

| Loại | Định nghĩa | Tác động |
|---|---|---|
| Point-contact | `adj_i ∩ adj_j` là `Point` | Artifact ACD2D, không phải cửa đi |
| Tightened-empty | `compute_intersection` = `None` | Stacked H-rep infeasible → `NarrowInterfaceError` chắc chắn |

Thực đo maze: 316/440 cạnh (72%) là tightened-empty → 100% trong 3240 paths thất bại trước DMS.

**Fix cần làm:** Trong `_build_graph`, bổ sung: (1) bỏ cạnh nếu `orig_inter.geom_type == 'Point'`; (2) bỏ cạnh nếu `compute_intersection` = `None`. Xem ràng buộc cứng 4–5 tại `docs/superpowers/specs/2026-06-07-region-decomposition-overlap-design.md`.

---

### 2.1 `demo/graph_builder.py` — Turn penalty thiếu

**Spec Part 3.2** yêu cầu:

```
c(i→j) = ‖c_j* − c_i*‖₂  −  γ_w · ρ_ij*  +  γ_h · |Δθ_ij|²
```

Với `γ_h = v_nom / ω_max ≈ 0.64` (unicycle).

**Hiện tại** (khoảng dòng 381): chỉ có `dist − γ_w · ρ_ij`. Số hạng `γ_h · |Δθ|²` **không được tính**.

**Tác động:** K-shortest path không phạt các cung có góc quay lớn → tìm path khó cho xe unicycle hơn mức cần thiết.

---

### 2.2 `demo/geometric_refiner.py` — Tham số λ_s chưa đúng công thức

**Spec Part 3.4** yêu cầu:

```
λ_s = α_s · (v_min / ω_max)²
```

**Hiện tại:** `λ_s` chỉ dùng `α_s` trực tiếp mà không nhân với `(v_min/ω_max)²`. Khi `α_s = 0.0` (default) không ảnh hưởng, nhưng khi `α_s > 0` thì hệ số smoothness không có đơn vị vật lý đúng.

---

### 2.3 `demo/optimizer.py` — Lower bound dùng vertex mean

**Spec Part 7.1** yêu cầu geometric LB tính bằng Dijkstra trên **Chebyshev center**, không dùng vertex mean.

**Hiện tại** (`_geometric_lb()`, khoảng dòng 1840–1859): dùng vertex mean của vùng convex. Kết quả là LB vẫn hợp lệ về mặt lower bound, nhưng **không phải pure-geometry LB** như spec mô tả → gap report có thể bị pessimistic.

---

### 2.4 `demo/experiments.py` — ScenarioResult thiếu ~20 fields

**Spec Part 11** định nghĩa đầy đủ `ScenarioResult`. **Thiếu các fields sau:**

**CRD diagnostics (13 fields):**
```python
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
crd_lb_geometric: float
```

**GCS-Bézier baseline slots (7 fields):**
```python
gcs_bezier_solve_time: float    = float("nan")
gcs_bezier_objective: float     = float("nan")
gcs_bezier_path_length: float   = float("nan")
gcs_bezier_travel_time: float   = float("nan")
gcs_bezier_s_min: float         = float("nan")
gcs_bezier_degree: int          = -1
gcs_bezier_source: str          = "not_set"
```

---

### 2.5 `demo/experiments.py` — Ablation runner chưa có

**Spec Part 12** mô tả ablation study với 5 flags (A1–A5):

| Flag | Điều kiện tắt | Điều kiện bật |
|------|--------------|--------------|
| A1: `use_centroid_cost` | Uniform edge cost (Dijkstra) | Chebyshev composite cost |
| A2: `use_interface_qp` | Naive midpoint init | QP-refined interface points |
| A3: `use_log_barrier` | Hard inequality constraints | Log-barrier trong DMS objective |
| A4: `use_barrier_continuation` | Single-level solve tại μ_min | Full N_μ levels với τ=0.1 |
| A5: `use_inexact_tolerance` | Fixed ε = ε_final | Schedule ε(r) = max(ε_final, μ_r) |

**Hiện tại:** Không có hàm/class ablation runner. Protocol 5 seeds × 3 environments = 15 runs/config chưa được implement.

---

### 2.6 `demo/config.yaml` — Block `centroid_refine_dms:` hoàn toàn vắng mặt

**Spec Part 13** yêu cầu block sau phải có trong config.yaml:

```yaml
centroid_refine_dms:
  # Graph search
  gamma_w: 1.0
  gamma_h: 0.64

  # Geometric QP
  alpha_s: 0.0
  delta_extra: 0.01

  # Warm-start
  v_nom_fraction: 0.5

  # Barrier schedule
  alpha_mu: 0.1
  tau: 0.1
  mu_min: 1.0e-5

  # Stopping criteria
  epsilon_final: 1.0e-6
  epsilon_gap: 0.05
  time_limit_s: 60.0

  # Solver mode
  mode: "first_feasible"

  # Ablation flags
  use_centroid_cost: true
  use_interface_qp: true
  use_log_barrier: true
  use_barrier_continuation: true
  use_inexact_tolerance: true
```

**Tác động:** `CentroidRefineDMSSolver` không thể load config từ file → phải hardcode hoặc pass dict thủ công.

---

### 2.7 Ablation flags chưa được wired hoàn toàn (Design Rule R10)

**Rule R10** yêu cầu tất cả `use_*` flags phải được tôn trọng trong implementation.

**Hiện tại trong `optimizer.py`:** Flags được parse nhưng:
- `use_log_barrier = False` → không fall through sang explicit inequality constraints (nhánh chưa implement)
- `use_barrier_continuation = False` → không fall through sang single-level solve
- `use_inexact_tolerance = False` → không fall through sang fixed ε

**Tác động:** Ablation study A3, A4, A5 không chạy được.

---

## 3. Xác minh công thức toán học

| Công thức | Yêu cầu spec | Implement | Kết quả |
|-----------|--------------|-----------|---------|
| Barrier objective J_μ | w_T Σ Δ_i + ... − μ Σ h_k log s_ijk | `BarrierDMSSolver._build_nlp()` | ✅ ĐÚNG |
| Trapezoidal weights h_k | Δ_i/(2n_int) [endpoints], Δ_i/n_int [interior] | `constraint_layers.py` | ✅ ĐÚNG |
| Khởi tạo μ₀ | μ₀ = 0.1·f₀/B₀ | `barrier_dms.py` | ✅ ĐÚNG |
| Giảm barrier | μ_r = μ₀·τʳ, τ=0.1 | `barrier_dms.py` | ✅ ĐÚNG |
| Inexact tolerance | ε(r) = max(ε_final, μ_r) | `barrier_dms.py` | ✅ ĐÚNG |
| Lipschitz gap | L_s · h_rk4 / 2 | `constraint_layers.py` | ✅ ĐÚNG |
| Smoothness param λ_s | α_s · (v_min/ω_max)² | `geometric_refiner.py` | ⚠️ THIẾU hệ số (v_min/ω_max)² |
| Edge cost composite | dist − γ_w·ρ_ij + γ_h·\|Δθ\|² | `graph_builder.py` | ⚠️ THIẾU γ_h·\|Δθ\|² |
| Safety slack | s_ijk = b_j − δ_safe − A[j]·q_k | `constraint_layers.py` | ✅ ĐÚNG |
| Certified safety | s_min_cert = s_min_sampled − L_s·h_rk4/2 | `barrier_dms.py` | ✅ ĐÚNG |

---

## 4. Kiểm tra Design Rules (R1–R12)

| Rule | Yêu cầu | Trạng thái | Bằng chứng |
|------|---------|------------|-----------|
| R1 | Defect constraints là hard equality | ✅ | Defects qua IPOPT equality, không penalty |
| R2 | Log-barrier chỉ cho safety halfplanes | ✅ | `build_barrier_log_terms()` chỉ geometry |
| R3 | Strict-interior assertion tại init | ✅ | QP đảm bảo s ≥ δ_extra; proof trong spec Part 4.3 |
| R4 | Không claim global optimality | ✅ | `MANDATORY_DISCLAIMER` string được set |
| R5 | Báo cả sampled lẫn certified safety | ✅ | Cả hai fields có trong `BarrierPathResult` |
| R6 | Tham số barrier được derive runtime | ✅ | μ₀, N_μ tính từ f₀, B₀ |
| R7 | Failure classification bắt buộc | ✅ | `FailureRecord` log mỗi path thất bại |
| R8 | QP và NLP solver độc lập | ✅ | CasADi graph riêng biệt, không share state |
| R9 | compute_lipschitz_bound() là abstract | ✅ | Cả UnicycleModel và DoubleIntegrator implement |
| R10 | Ablation flags first-class | ⚠️ PARTIAL | Flags parse được, nhưng nhánh A3/A4/A5 chưa wired |
| R11 | Không dùng mesh points | ✅ | CentroidRefineDMSSolver không đọc n_mesh_points |
| R12 | GCS-Bézier slots là data-only | ❌ | Fields chưa tồn tại trong ScenarioResult |

---

## 5. Tổng kết và thứ tự ưu tiên

### Ưu tiên cao — Cần để chạy được solver đúng cách

| # | File | Việc cần làm |
|---|------|-------------|
| 1 | `demo/config.yaml` | Thêm block `centroid_refine_dms:` với 17 tham số |
| 2 | `demo/graph_builder.py` | Thêm turn penalty `γ_h · |Δθ|²` vào `composite_edge_cost()` |
| 3 | `demo/geometric_refiner.py` | Sửa `λ_s = α_s · (v_min/ω_max)²` (không chỉ `α_s`) |

### Ưu tiên trung bình — Cần để báo cáo và so sánh đúng

| # | File | Việc cần làm |
|---|------|-------------|
| 4 | `demo/experiments.py` | Thêm 20 fields vào `ScenarioResult` (13 CRD + 7 GCS-Bézier) |
| 5 | `demo/optimizer.py` | Sửa `_geometric_lb()` dùng Chebyshev center thay vertex mean |
| 6 | `demo/optimizer.py` + `demo/barrier_dms.py` | Wire ablation flags A3/A4/A5 |

### Ưu tiên thấp — Hoàn thiện ablation study

| # | File | Việc cần làm |
|---|------|-------------|
| 7 | `demo/experiments.py` | Implement ablation runner (Part 12 của spec): 5 configs × 15 runs |

---

## 6. Những phần đã đúng — không cần sửa

- **`demo/barrier_dms.py`** toàn bộ: schedule, continuation loop, failure taxonomy, warm-start
- **`demo/constraint_layers.py`** toàn bộ: barrier log terms, Lipschitz gap
- **`demo/warmstart.py`** toàn bộ: unicycle và double integrator init, duration clamping
- **`demo/dynamics.py`** toàn bộ: DoubleIntegratorDynamics đúng, compute_lipschitz_bound() đúng
- **`demo/geometric_refiner.py`**: QP solver, constraint enforcement, narrow interface check (chỉ λ_s cần fix)
- **`demo/graph_builder.py`**: chebyshev_center(), k_shortest_paths_generator() (chỉ thiếu turn penalty)
- **`demo/optimizer.py`**: CentroidRefineDMSSolver main flow, OptimizationResult fields, dispatch logic, mandatory disclaimer
