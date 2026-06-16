# Đối chiếu Code ↔ Lý thuyết — Diagnostics của Centroid-Refine-DMS

**Date:** 2026-06-16
**Spec đối chiếu:** `docs/superpowers/specs/2026-06-07-centroid-refine-dms-design.md`
**Phạm vi:** Các đại lượng *báo cáo* (gap, max violation, max control jump, defect/coupling residual, certified/sampled safety margin) của `CentroidRefineDMSSolver`.
**Kết luận ngắn:** Nghi ngờ "sai số số học = 0 là vô lý" **được xác nhận là đúng**. Phần lớn các sai số được report bằng **0 vì chúng chưa bao giờ được tính** — chúng là giá trị mặc định của dataclass mà nhánh CRD không bao giờ ghi đè. Ngoài ra certified safety margin đang đo **sai đại lượng** so với spec (vấn đề liên quan an toàn).

> **LƯU Ý CHO AGENT SỬA:** Đây là báo cáo audit. **Chưa sửa gì cả.** Hãy đọc kỹ phần "Prompt cho agent sửa" ở cuối. Mọi thay đổi **không được vi phạm** các ràng buộc R1–R13 trong design doc, và code sau khi sửa **sẽ được Codex review** → cần đúng về toán học, có dẫn chiếu spec, và có test.

---

## Bối cảnh: tại sao "= 0" lại đáng ngờ

Một nghiệm NLP do IPOPT trả về **không bao giờ** có defect/coupling residual đúng bằng `0.0` — nó nằm trong khoảng `~1e-9 … 1e-6` (theo `ipopt.tol`). Khi report thấy đúng `0.00e+00`, đó là dấu hiệu của một **giá trị mặc định chưa được tính**, không phải một phép đo.

`OptimizationResult` khai báo các default này (`demo/optimizer.py:181`, `187`, `210`, `211`, `214`):

```python
defect_norm: float = 0.0
max_connection_gap: float = 0.0
lb_geometric: float = 0.0
optimality_gap: float = float("inf")
n_nlp_iterations: int = 0
```

Trong `optimizer.py`, các trường này **chỉ** được gán bởi solver legacy (`PathNLPSolver` @ `1109`, `935`; `IntegratedMIOCPSolver` @ `2463`). `CentroidRefineDMSSolver.solve()` (`optimizer.py:2522–2734`) **không gán bất kỳ trường nào trong số đó**. Vì vậy mọi `OptimizationResult` mode `centroid_refine_dms` luôn có `defect_norm=0`, `max_connection_gap=0`, `lb_geometric=0`, `optimality_gap=inf`, `n_nlp_iterations=0`.

Và `visualization.py:670` / `:922` lại **in trực tiếp** `Defect {result.defect_norm:.2e}` cho mode `log_barrier` → màn hình luôn hiện `Defect 0.00e+00`. Đây chính là con số "vô lý" đã quan sát thấy.

---

## Bảng đối chiếu chi tiết

| # | Đại lượng | Spec yêu cầu | Code thực tế | File:line | Mức độ |
|---|---|---|---|---|---|
| F1 | `defect_norm` (residual ràng buộc continuity/coupling) | Part 6 Tier 1: `‖r_def‖∞ ≤ ε_d`; phải report (R7, Part 9 Stage F) | Không bao giờ tính cho CRD → mặc định `0.0`, được in ra như sai số thực | `optimizer.py:2674–2692` (thiếu); `visualization.py:670,922` (in) | **Cao** |
| F2 | `max_connection_gap` (coupling gap `‖x_i(1)−x_{i+1}(0)‖`) | §2.3 coupling; Part 10 `CONNECTION_GAP_LARGE` so với `ε_c`; report | Không tính cho CRD → `0.0`. Helper `_compute_connection_gap` (`optimizer.py:217`) **đã có sẵn** và đủ dữ liệu (`entry_states`/`exit_states`/start/goal) nhưng không được gọi | `optimizer.py:2674–2692` | **Cao** |
| F3 | `lb_geometric` = `D*_path / v_max` | §7.1, Part 9 init `LB_geom ← dijkstra/v_max` | Không tính → `0.0` | `optimizer.py:2522–2734` | **Cao** |
| F4 | `optimality_gap` = `(UB − w_T·LB)/max(1,|UB|)` | §7.2, Part 9 Stage F | Không tính → `inf` | `optimizer.py:2522–2734` | **Cao** |
| F5 | `min_safety_margin` (sampled `s_min`) | §8.1: `s_{i,j,k} = b − δ_safe − aᵀPx` (CÓ trừ `δ_safe`) | `_compute_min_slack` dùng `region.b − region.A @ pos` — **KHÔNG trừ `δ_safe`** | `barrier_dms.py:477` | **Cao (an toàn)** |
| F6 | `certified_safety_margin` | §8.2: `s_min_sampled − L_s·h_max/2 − ε_defect − ε_int`, với `s` theo định nghĩa §8.1 | `min_slack − lip_gap`; (a) `min_slack` không trừ `δ_safe` (xem F5); (b) bỏ hẳn `ε_defect`, `ε_int` | `barrier_dms.py:140–149`, `constraint_layers.py:448–459` | **Cao (an toàn)** |
| F7 | `total_cost` dùng làm UB | §7.2 Tier 2 + pseudocode dòng 626: `J* = evaluate_cost(..., μ=0)` (KHÔNG có barrier) | `result.total_cost = float(sol['f'])` — là `J_μ` **đã gồm** số hạng barrier `−μ Σ h_k log s` | `barrier_dms.py:442` | **Trung bình** |
| F8 | `n_nlp_iterations` | Part 11 `crd_n_nlp_iterations` | Không tính → `0` | `optimizer.py:2674–2692` | Thấp |
| F9 | `global_optimality_claim` | R4 + §7.3: phải là chuỗi disclaimer chuẩn (4 câu) | Dùng chuỗi rút gọn khác: `"KKT-feasible under LICQ+SOSC; no global optimality certificate."` | `optimizer.py:2553–2555` | Thấp |
| F10 | `max_dense_region_violation`, `max_continuous_violation_integral` | §8.3: **không** populate ở CRD (đúng); report phải branch theo `safety_mode` | Giữ default `0.0` (đúng spec). **Rủi ro:** nếu UI hiển thị "violation = 0" mà không nói rõ "không đo ở mode này" sẽ gây hiểu nhầm | `visualization.py:682–694` (đã branch — OK) | Thông tin |

---

## Phân tích sâu các điểm quan trọng

### F1–F4: "Sai số = 0" — nguyên nhân gốc

Đây là cụm lỗi chính trả lời trực tiếp câu hỏi. Trong `CentroidRefineDMSSolver.solve()`, `OptimizationResult` được dựng tại `optimizer.py:2674–2692` chỉ copy các trường từ `BarrierPathResult` (success, total_cost, các safety margin). **Không có** bước Stage F "DIAGNOSTICS" như pseudocode Part 9 (dòng 624–635) mô tả:

```
── STAGE F: DIAGNOSTICS AND UB UPDATE ──
  J_star ← evaluate_cost(x_w, u_w, Δ_w, cfg, μ=0)
  diag ← compute_diagnostics(...)   ← defect_norm, coupling_gap, KKT_residual
                                       s_min_sampled, s_min_certified
  ...
  gap ← (UB − LB_geom * w_T) / max(1, |UB|)
```

Toàn bộ khối này **chưa được hiện thực**. Hệ quả: các con số defect/coupling/gap mà người dùng nhìn thấy không phản ánh nghiệm thực — chúng là hằng số mặc định.

`BarrierPathResult` (`barrier_dms.py:32–47`) thậm chí **không có trường `defect_norm`/`coupling_gap`** để mang dữ liệu ra. Cần bổ sung phép đo trong `barrier_dms` (nơi có sẵn `entry_states`/`exit_states`/`x_node_vars_list`) rồi truyền lên.

Lưu ý cấu trúc: mỗi region là **một** khoảng shooting (RK4 nội suy trong segment, chỉ ràng endpoint), nên "defect" thực chất chính là **coupling residual giữa các segment**: `g = s_plus − s_minus_next` (`barrier_dms.py:316`) cộng 2 ràng buộc biên start/goal (`:349`, `:352`). Đo `‖g‖∞` tại nghiệm chính là defect/coupling cần report. **Không được** đổi các ràng buộc này thành penalty (R1).

### F5–F6: Certified safety đo SAI đại lượng (liên quan an toàn)

Spec §2.1 / §8.1 định nghĩa slack an toàn **có trừ `δ_safe`**:

```
s_{i,j,k} = b_{i,j} − δ_safe − a_{i,j}ᵀ P x_i(τ_{i,k})
```

Trong NLP, barrier **đúng** với spec: biến slack bị ràng `slack = b − node_delta_safe − A·pos` (`barrier_dms.py:378–382`) và `lbx = 1e-10` → IPOPT giữ trajectory trong vùng đã co `δ_safe`. **Nhưng** hàm report `_compute_min_slack` lại tính:

```python
slacks = region.b - region.A @ pos       # barrier_dms.py:477  — THIẾU − δ_safe
```

→ `min_safety_margin` (sampled) bị **thổi phồng đúng bằng `δ_safe`** so với định nghĩa spec, và `certified_safety_margin = min_slack − lip_gap` (`barrier_dms.py:146`) kế thừa sai lệch đó. Hệ quả nghiêm trọng: có thể report `safety_certification = "CERTIFIED"` (`barrier_dms.py:147–149`) trong khi giá trị đúng theo spec đã `≤ 0`. Tức là **chứng nhận an toàn lạc quan hơn lý thuyết cho phép**.

Ngoài ra §8.2 yêu cầu trừ thêm `ε_defect` (≈ `ipopt.tol`) và `ε_int` (RK4): `s_cert = s_sampled − L_s·h_max/2 − ε_defect − ε_int`. Code chỉ trừ `L_s·h_max/2` (`constraint_layers.py:459`). Vì defect không được đo (F1), thành phần `ε_defect` mặc nhiên bị coi = 0 — vòng luẩn quẩn với F1.

> Phần *enforcement* (`δ_safe = max(cfg.delta_safe, lip_gap + epsilon_final)`, `barrier_dms.py:87`) thỏa R13 và đúng. Chỉ phần *certification/report* sai. Khi sửa, giữ nguyên enforcement.

### F7: UB chứa số hạng barrier

`total_cost = float(sol['f'])` (`barrier_dms.py:442`) là `J_μ`, gồm `−μ Σ h_k log s`. Tại `μ=0.01` số hạng này không nhỏ và dấu phụ thuộc `s` (≷1). Spec §7.2 + pseudocode dòng 626 yêu cầu UB là `J|_{μ=0}` (chỉ `w_T ΣΔ + w_L∫‖v‖² + w_U∫‖u‖² + w_S·jumps`). Cần đánh giá lại cost **không có** barrier để dùng cho UB và cho `optimality_gap` (F4).

---

## Tổng hợp tác động

- Câu hỏi gốc: **đúng** — defect/coupling/gap = 0 vì *chưa từng được tính*, không phải vì nghiệm hoàn hảo.
- `max violation` ở mode CRD: các trường CTCS/dense **cố ý** không dùng (§8.3, đúng) — đại lượng an toàn hợp lệ là `s_min`/`s_cert`, nhưng cả hai đang **đo sai** (F5/F6).
- `optimality_gap`/`lb_geometric`: không dùng được (luôn `inf`/`0`).
- Rủi ro an toàn cao nhất: F5/F6 có thể chứng nhận CERTIFIED sai.

Không có lỗi nào vi phạm R1 (defect vẫn là hard equality) hay R2 (barrier vẫn chỉ trên halfplane an toàn) — đây thuần là lỗi **đo lường & báo cáo**, cộng một lỗi **định nghĩa đại lượng** (F5/F6).

---

## Prompt cho agent sửa (paste nguyên văn)

> **Nhiệm vụ:** Sửa tầng *diagnostics/báo cáo* của `CentroidRefineDMSSolver` trong repo `GCS_MMS_demo` để các đại lượng report khớp với lý thuyết trong `docs/superpowers/specs/2026-06-07-centroid-refine-dms-design.md`. Hiện tại defect/coupling/gap/control jump report = 0 vì **chưa bao giờ được tính** (chỉ là default của dataclass), và certified safety margin **đo sai đại lượng**. Chi tiết bằng chứng: `docs/superpowers/specs/2026-06-16-crd-diagnostics-audit.md`.
>
> **Ràng buộc bắt buộc (không được vi phạm):**
> - R1: defect/coupling vẫn là **hard equality** truyền cho IPOPT — chỉ *đo* residual tại nghiệm, **tuyệt đối không** chuyển sang penalty.
> - R2: barrier chỉ trên halfplane an toàn — không đụng tới.
> - R5: cả `min_safety_margin` (sampled) và `certified_safety_margin` luôn được populate.
> - R13: giữ nguyên cách tính `δ_safe` enforcement ở `barrier_dms.py:87` (đã đúng).
> - R4: `global_optimality_claim` phải là **đúng chuỗi disclaimer chuẩn** ở §7.3 của design doc.
> - Code sẽ được **Codex review** → mỗi thay đổi cần: đúng toán học, kèm comment dẫn chiếu §spec, và có unit test.
>
> **Việc cần làm (theo thứ tự ưu tiên):**
> 1. **F5/F6 (an toàn, ưu tiên cao nhất):** Sửa `_compute_min_slack` (`barrier_dms.py:460–491`) để slack = `region.b − delta_safe − region.A @ pos`, đúng định nghĩa §2.1/§8.1 (delta_safe = giá trị `delta_safe` đã dùng để build NLP, tức `node_delta_safe`/`delta_safe_barrier`). Sửa `certified_safety_margin` (`barrier_dms.py:140–149`) thành `s_min_sampled − L_s·h_max/2 − ε_defect − ε_int` theo §8.2, trong đó `ε_defect` = defect_norm đo được (xem mục 2), `ε_int` theo công thức RK4 §8.2 (có thể xấp xỉ). Đảm bảo `safety_certification` đổi theo giá trị đúng.
> 2. **F1/F2 (defect & coupling):** Trong `barrier_dms.py`, tại nghiệm cuối, đo `‖g‖∞` của các ràng buộc coupling+biên (`s_plus − s_minus_next`, `s_minus[0]−start`, `endpoint−goal`) → thêm trường `defect_norm`/`coupling_gap` vào `BarrierPathResult` (`:32–47`). Ở `optimizer.py:2674–2692` gán `result.defect_norm` và gọi `_compute_connection_gap(...)` (đã có sẵn, `optimizer.py:217`) để set `result.max_connection_gap`.
> 3. **F7:** Thêm hàm đánh giá cost **không có** số hạng barrier (`μ=0`) và dùng nó cho `total_cost`/UB (spec §7.2, pseudocode dòng 626). Có thể tách `obj` thành `obj_base + barrier_term` khi build NLP để tái dùng.
> 4. **F3/F4:** Tính `lb_geometric = D*_path / v_max` (Dijkstra Euclid trên centroid graph, §7.1) và `optimality_gap = (UB − w_T·LB)/max(1,|UB|)` (§7.2); gán vào `OptimizationResult`. (Nếu graph helper chưa có, thêm trong `graph_builder.py`.)
> 5. **F8:** Lấy số vòng lặp IPOPT từ `solver.stats()` (cộng dồn qua các barrier level) → `n_nlp_iterations`.
> 6. **F9:** Thay `_KKT_DISCLAIMER` (`optimizer.py:2553–2555`) bằng đúng chuỗi 4 câu §7.3.
> 7. **UI:** Kiểm tra `visualization.py:670,922` và `experiments.py:56` hiển thị defect đúng cho mode `log_barrier`; với các trường CTCS/dense ở CRD, label rõ "n/a ở mode log_barrier" thay vì in `0.00e+00` (§8.3).
>
> **Kiểm chứng:** Chạy `demo/test_crd_default.py` (default case là benchmark, xem commit `a846c2b`). Sau khi sửa, `defect_norm` phải nằm trong `~[1e-9, 1e-6]` (KHÔNG đúng `0.0`); `s_cert ≤ s_min`; nếu trajectory sát biên thì `safety_certification` có thể chuyển sang `NOT_CERTIFIED` — đó là hành vi đúng. Thêm unit test khẳng định: (a) defect_norm > 0 và ≤ ipopt.tol·10; (b) `min_safety_margin` nhỏ hơn giá trị cũ đúng ~`δ_safe`; (c) `optimality_gap` hữu hạn.

---

## Phụ lục — vị trí code tham chiếu nhanh

- CRD solver build result (thiếu Stage F): `demo/optimizer.py:2674–2692`
- `OptimizationResult` defaults: `demo/optimizer.py:181–214`
- `_compute_connection_gap` (có sẵn, chưa dùng cho CRD): `demo/optimizer.py:217–249`
- Sampled slack tính sai (thiếu δ_safe): `demo/barrier_dms.py:460–491`
- Certified margin: `demo/barrier_dms.py:136–163`
- Lipschitz gap: `demo/constraint_layers.py:448–459`
- Barrier slack trong NLP (đúng định nghĩa, có δ_safe): `demo/barrier_dms.py:356–396`
- `total_cost = sol['f']` (gồm barrier): `demo/barrier_dms.py:442`
- UI in defect cho log_barrier: `demo/visualization.py:670, 918–929`
