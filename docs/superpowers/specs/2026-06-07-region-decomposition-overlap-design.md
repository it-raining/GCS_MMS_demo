# Region Decomposition & Overlap Design

**Ngày:** 2026-06-07
**Liên quan:** `docs/superpowers/specs/2026-06-07-centroid-refine-dms-design.md`
**Trạng thái:** Approved — chờ implementation plan

---

## 1. Bối cảnh & Yêu cầu cứng

### 1.1 Nguyên lý GCS gốc

Graph of Convex Sets (Marcucci et al. 2022) xây dựng đồ thị tối ưu trên các convex region **chồng lấn nhau**. Tại vùng giao `C_i ∩ C_j`, ràng buộc liên tục:

```
x_i(1) = x_j(0)
```

được đặt trực tiếp. Overlap là điều kiện **cần** để ràng buộc này có nghiệm không tầm thường (nếu giao rỗng hoặc chỉ là một điểm/cạnh, bài toán infeasible tại interface).

Trong GCS gốc, các region thường được xây dựng bằng IRIS (Iterative Regional Inflation by Semidefinite programming — Deits & Tedrake 2015), thuật toán tự nhiên sinh ra các polytope chồng lấn vì nó inflation từ seed point cho đến khi chạm obstacle.

### 1.2 Kỹ thuật hiện tại trong repo này

Repo dùng **ACD2D** (Approximate Convex Decomposition 2D) để phân hoạch workspace. ACD2D sinh ra các đa giác lồi **không chồng lấn** (edge-touching partition). Hai vùng lân cận chỉ tiếp giáp dọc một cạnh chung — giao là một đoạn thẳng (1D), Chebyshev radius = 0.

Để tạo overlap cần thiết, mỗi vùng ACD2D được **mở rộng ra ngoài** (outward buffer) một lượng `overlap_width` bằng Shapely mitre buffer + workspace clip:

```python
buffered = poly.buffer(overlap_width, join_style='mitre')
clipped  = buffered.intersection(workspace_polygon)
```

**Tại sao `join_style='mitre'`:** Mitre join giữ nguyên các góc sắc của polygon — vùng sau buffer vẫn lồi. Round join tạo cung tròn, không dùng được với H-rep.

**Tại sao workspace clip hiệu quả cho "shared edge only":** Các cạnh tại biên workspace sau khi buffer bị clip lại về workspace boundary. Chỉ phần buffer tại cạnh chung giữa hai vùng lân cận có tác dụng thực sự. Behavior này tương đương "shared edge only buffer" mà không cần shared edge detection thủ công.

Kết quả: giao `C_i ∩ C_j` là dải rộng `2·overlap_width` dọc cạnh chung — đủ để Interface QP của CRD tìm điểm khả thi.

### 1.3 Yêu cầu cứng (Hard Requirements)

1. **Mọi solver mode** (`integrated_relaxation_legacy`, `two_stage`, `centroid_refine_dms`) **đều phải đi qua pipeline ACD2D → mitre buffer → `build_region_graph`**. Không có code path nào bypass ACD2D.

2. **`skip_acd` flag bị xóa** khỏi `ProblemPresetSpec`. Các preset hiện dùng `skip_acd=True` (`maze_15`, `crd_demo`) phải được refactor để dùng ACD2D trên workspace tương ứng.

3. **`overlap_width ≥ delta_safe + delta_extra`** để Interface QP của CRD luôn có Chebyshev radius đủ lớn. Nếu vi phạm: log warning tại startup.

4. **Cạnh Point-contact bị loại bỏ cứng khỏi graph.** Trong `_build_graph`, sau khi `regions_intersect(adj_i, adj_j)` trả về True, phải kiểm tra thêm:

   ```python
   orig_inter = adj_i.get_shapely_polygon().intersection(adj_j.get_shapely_polygon())
   if orig_inter.geom_type == 'Point':
       continue  # bỏ qua cạnh này
   ```

   **Lý do:** Khi hai vùng ACD2D chỉ tiếp xúc tại một đỉnh (vertex contact), `regions_intersect` với `tolerance=0.005` vẫn phát hiện là adjacent nhờ dilation. Nhưng về mặt tôpô, một điểm duy nhất không phải cửa đi thực sự — robot không thể đi qua một điểm không có độ rộng. Đây là artifact của quá trình phân hoạch ACD2D khi hai đa giác lân cận gặp nhau tại reflex vertex. Loại bỏ là bắt buộc.

5. **Cạnh có tightened intersection rỗng bị loại bỏ cứng khỏi graph.** Khi `compute_intersection(ri, rj)` trả về `None` (do tightened Shapely intersection là empty), cạnh (i→j) **không được thêm vào** `region_edges`. Không có fallback stacked H-rep.

   **Lý do (xác nhận bằng diagnostic maze_crd, 2026-06-08):** `_tighten_hrep_against_obstacles` clip từng vùng về phía obstacle face của nó. Hai vùng ở hai phía đối diện của bức tường bị tighten về hai wall face khác nhau, tạo khoảng cách = wall thickness (thực đo: mean gap 0.13, max gap 2.57 đơn vị). Stacked H-rep của hai vùng đã tách nhau là infeasible → Interface QP luôn thất bại với `NarrowInterfaceError`. Với `maze_crd`: 316/440 cạnh (72%) rơi vào trường hợp này, khiến toàn bộ 3240 paths thất bại trước khi DMS được gọi.

   Nếu hai vùng ACD2D chia sẻ cạnh 1D trong không gian tự do (cửa đi thực sự) thì sau tightening, các vùng tightened vẫn chồng lấn nhau qua cửa đi → `compute_intersection` trả về non-None. Cạnh tightened-empty chứng tỏ "cạnh chung" nằm trên wall face, không phải trong free space — robot không thể đi qua.

---

## 2. Pipeline tổng thể

```
build_environment_and_regions(preset)
│
├─ ACD2D(workspace, obstacles)
│      └─ [polygon list]                      ← vùng gốc, edge-touching
│
├─ adjacency_regions = ConvexRegion(original)  ← dùng để detect adjacency
│
├─ Shapely mitre buffer(overlap_width) + workspace clip
│      └─ regions = ConvexRegion(A, b_expanded)   ← dùng bởi optimizer
│
└─ return (environment, regions, adjacency_regions)

build_region_graph(regions, start, goal, adjacency_regions=adjacency_regions)
│
├─ adjacency: regions_intersect(adj_i, adj_j)    ← trên vùng gốc
└─ intersection: compute_intersection(r_i, r_j)  ← trên vùng expanded (overlap 2w)

Solver
├─ integrated_relaxation_legacy : Big-M containment trên expanded regions
├─ two_stage                    : mesh containment trên expanded regions
└─ centroid_refine_dms          : Interface QP trên C_i∩C_j (rộng 2w ≥ δ_safe+δ_extra)
```

**Bất biến:** Adjacency detection luôn dùng vùng ACD2D gốc (không buffer) để tránh tạo cạnh giả giữa các vùng không thực sự lân cận.

---

## 3. Thay đổi config

Thêm block `region_decomposition` vào `config.yaml`:

```yaml
region_decomposition:
  # Outward buffer applied to each ACD2D polygon after decomposition.
  # Creates overlap of 2*overlap_width at each shared edge between adjacent regions.
  # Constraint: overlap_width >= centroid_refine_dms.delta_safe + delta_extra
  # Legacy and two_stage solvers are unaffected (extra feasible space at interfaces).
  overlap_width: 0.3
```

`scenario_builder.py` đọc giá trị này và validate:

```python
decomp_cfg    = runtime_config.get("region_decomposition", {})
overlap_width = decomp_cfg.get("overlap_width", 0.05)
regions = create_buffered_regions_from_vertices_list(
    region_vertices, preset.workspace_vertices, buffer_size=overlap_width
)

# Startup validation cho CRD
if solver_mode == "centroid_refine_dms":
    crd_cfg  = runtime_config.get("centroid_refine_dms", {})
    required = crd_cfg.get("delta_safe", 0.0) + crd_cfg.get("delta_extra", 0.0)
    if overlap_width < required:
        warnings.warn(
            f"overlap_width={overlap_width} < delta_safe+delta_extra={required:.3f}. "
            "Interface QP may fail with NARROW_INTERFACE."
        )
```

---

## 4. Thay đổi code

| File | Thay đổi |
|------|----------|
| `config.yaml` | Thêm block `region_decomposition.overlap_width: 0.3` |
| `scenario_builder.py` | Đọc `overlap_width` từ config; truyền vào `create_buffered_regions_from_vertices_list`; xóa nhánh `if preset.skip_acd`; thêm startup validation |
| `problem_data.py` | Xóa field `skip_acd` khỏi `ProblemPresetSpec`; xóa `region_vertices` thủ công và `skip_acd=True` khỏi `crd_demo` và `maze_15`; thêm `obstacle_vertices` thực cho `maze_15` tạo hành lang zigzag |
| `convex_regions.py` | Không đổi — `create_buffered_regions_from_vertices_list` đã đúng |
| `graph_builder.py` | Không đổi |

---

## 5. Thay đổi documentation

### 5.1 Design spec CRD (`2026-06-07-centroid-refine-dms-design.md`)

Thêm **"Part 0 — Region Decomposition Pipeline"** trước Part 1, nội dung tóm tắt từ Section 1–2 của doc này.

Cập nhật bảng "Known Implementation Risks":
> Nếu `overlap_width < delta_safe + delta_extra`, Interface QP infeasible trên toàn bộ paths — kiểm tra bằng startup warning.

### 5.2 Gap report (`2026-06-07-implementation-gap-report.md`)

Thêm 3 gaps vào bảng gaps:

| Gap | Loại | Priority |
|-----|------|----------|
| `skip_acd=True` bypass ACD2D trong `maze_15`/`crd_demo` | Architectural gap | High |
| `buffer_size=0.001` quá nhỏ — không tạo overlap đủ cho CRD | Config gap | High |
| Thiếu block `region_decomposition` trong `config.yaml` | Missing config | Medium |

---

## 6. Preset `maze_15` sau refactor

Preset `maze_15` hiện dùng `skip_acd=True` với 8 vùng thủ công. Sau khi xóa `skip_acd`, preset cần:

- `workspace_vertices`: `[[0,0],[15,0],[15,15],[0,15]]` (giữ nguyên)
- `obstacle_vertices`: định nghĩa các vách ngăn tạo hành lang zigzag thay thế cho 8 vùng thủ công
- `region_vertices` và `skip_acd`: xóa

ACD2D sẽ phân hoạch workspace với obstacles đó thành các vùng lồi; mitre buffer tạo overlap `2·overlap_width`. Nếu ACD2D không sinh path liên thông START→GOAL, cần điều chỉnh obstacle geometry.

> **Lưu ý:** Obstacle geometry cho `maze_15` cần được verify bằng cách chạy ACD2D thực và kiểm tra region graph connectivity. Đây là bước cần thực nghiệm — không thể xác định tĩnh chỉ từ thiết kế.

---

## 7. Giải pháp đã xem xét và loại bỏ

| Giải pháp | Lý do loại bỏ |
|-----------|--------------|
| IRIS inflation (Deits & Tedrake 2015) | Cần SDP solver, không có trong deps; overkill |
| Bridge/interface region nodes trong graph | Breaking change quá lớn cho graph structure hiện tại |
| Halfspace relaxation thủ công (shared edge detection) | Không cần — Shapely mitre buffer + workspace clip đã cho behavior tương đương |
| Giữ `skip_acd` dưới tên `use_iris_regions` | Vi phạm yêu cầu cứng: mọi solver phải qua ACD2D |
