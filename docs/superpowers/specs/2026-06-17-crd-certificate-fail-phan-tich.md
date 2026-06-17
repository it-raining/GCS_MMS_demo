# Phân tích nguyên nhân: CentroidRefineDMS báo `CERTIFICATE_FAIL` ở scenario `default`

## 1. Hiện tượng

Chạy `python main_demo.py --scenario default` (sau khi đã sửa lỗi `AttributeError:
'DemoConfig' object has no attribute 'docs_dir'` do rebase hỏng), pipeline chạy
hết, không crash, nhưng kết quả tối ưu hoá báo **thất bại**:

```
Optimization Status: FAILED
Solver: CERTIFICATE_FAIL
Total Cost: 54.9383
Solve Time: 65.868 s
Paths Evaluated: 8
```

Chạy lại bằng `test_crd_default.py` (có log chi tiết từng bước) cho số liệu cụ
thể hơn:

```
Safety: sampled=0.0000, certified=-0.0682, lipschitz_gap=0.0682
Safety certification: NOT_CERTIFIED
```

Đây **không phải** lỗi crash/exception — optimizer chạy đúng, NLP hội tụ, nhưng
bước "chứng nhận an toàn" (safety certification) sau cùng kết luận quỹ đạo
không đủ an toàn theo định nghĩa trong spec.

## 2. Công thức đang dùng (đã được audit ở commit `2b8b0b3`)

File `demo/barrier_dms.py:143-164`:

```python
min_slack = self._compute_min_slack(best_result, delta_safe)      # sampled margin
lip_gap   = compute_lipschitz_safety_gap(..., optimized_delta_arr, cfg.n_int)
h_max     = max(optimized_delta_arr) / cfg.n_int
eps_int   = dynamics.f_lipschitz_bound() * h_max**4 / 30.0

certified_safety_margin = min_slack - lip_gap - defect_norm - eps_int
safety_certification = "CERTIFIED" if certified_safety_margin > 0 else "NOT_CERTIFIED"
```

`_compute_min_slack` (`barrier_dms.py:482-501`) tính:

```python
slacks = region.b - delta_safe - region.A @ pos     # spec Sec 2.1/8.1
min_slack = min(slacks tại mọi điểm lưới RK4)
```

Tức là `min_slack` đo phần dư geometric margin **so với `delta_safe` đã dùng
trong ràng buộc của NLP** — không phải so với 0.

## 3. Nguyên nhân gốc rễ nhiều khả năng nhất: lệch giữa "ước lượng trước khi giải" và "giá trị thật sau khi giải"

`delta_safe` được truyền vào `_compute_min_slack`/NLP **không phải** giá trị
thô `centroid_refine_dms.delta_safe` (0.02 trong config), mà là một giá trị
đã được "bơm thêm" trước khi giải, gọi là `delta_safe_barrier`
(`demo/optimizer.py:2622-2650`):

```python
delta_safe_barrier = cfg.delta_safe
for _ in range(4):
    delta_arr = [warm_start[ri]['delta'] for ri in path_regions]   # <-- ước lượng warm-start
    delta_arr = min(delta_arr, cfg.delta_max)
    lip_gap = compute_lipschitz_safety_gap(dynamics, path_regions, graph, delta_arr, cfg.n_int)
    required_delta_safe = max(cfg.delta_safe, lip_gap + cfg.epsilon_final)
    ...
    delta_safe_barrier = required_delta_safe
```

Vòng lặp này dùng **thời gian lưu trú mỗi vùng (`Delta_i`) do warm-start ước
lượng** (warm-start giả định tốc độ không đổi `v_nom_fraction=0.5*v_max`) để
tính trước một `lip_gap` "dự kiến", rồi bơm vào `delta_safe_barrier` làm ràng
buộc cứng cho NLP.

Nhưng sau khi NLP (`BarrierDMSSolver`) giải xong, `lip_gap` ở bước chứng nhận
cuối (mục 2) được **tính lại bằng `Delta_i` thật sự mà NLP chọn**
(`optimized_delta_arr = best_result.time_durations[...]`), không phải
`Delta_i` của warm-start. NLP có cost gồm cả time penalty (`w_T`), control
effort (`w_U`) và độ trơn điều khiển (`w_S`) — nó **được tự do chọn
`Delta_i` khác với giả định tốc độ không đổi của warm-start**, miễn là tổng
chi phí thấp hơn.

Vì hàm mục tiêu luôn có động cơ "ăn mòn" ràng buộc an toàn đến sát biên (giảm
chi phí), `min_slack` (đo so với `delta_safe_barrier` đã bơm) luôn tiến về
**0** — đây không phải dấu hiệu lỗi, mà gần như là hệ quả tất nhiên của một
NLP cực tiểu hoá chi phí có ràng buộc bất đẳng thức. Do đó:

```
certified_safety_margin ≈ 0 - lip_gap_thật_sau_khi_giải - defect_norm - eps_int
```

**Không có vòng lặp phản hồi (feedback loop)** để, sau khi biết `Delta_i`
thật, kiểm tra lại "liệu `delta_safe_barrier` đã bơm có >= `lip_gap` thật hay
chưa", và re-solve nếu chưa đủ — giống cách vòng lặp 4 bước ở trên đã làm
*trước* khi giải, nhưng chỉ dựa trên warm-start. Đây là khoảng trống thiết kế
(design gap) khiến certification gần như chắc chắn fail mỗi khi `Delta_i`
thực tế > `Delta_i` warm-start ước lượng.

Số liệu khớp với giả thuyết này: `sampled=0.0000` (ràng buộc luôn active —
đúng như dự đoán) và `certified=-0.0682` xấp xỉ đúng bằng
`-lipschitz_gap=-0.0682` (vì `defect_norm`, `eps_int` rất nhỏ) — nghĩa là toàn
bộ phần thiếu hụt đến từ `lip_gap` tính sau khi giải, lớn hơn phần đã được bơm
trước khi giải.

## 4. Nguyên nhân góp phần: hình học `default` preset khá "hẹp" so với margin cấu hình

Log debug (`test_crd_default.py`, stage `DE`) cho thấy **12/13 lần thử bị từ
chối ngay ở bước warm-start** vì không thoả điều kiện R3 (strict-interior):

```
R3 strict-interior invariant failed: s_ijk=0.007002 < delta_extra=0.010000
   (region=3, node=44)
... (nhiều dòng tương tự, s_ijk dao động 0.007–0.0099)
```

Tức là biên độ dư (`s_ijk`) của rất nhiều ứng viên đường đi chỉ nhích hơn
`delta_extra=0.01` một chút hoặc thấp hơn — cho thấy phân rã ACD2D của
workspace 5×5 + 6 vật cản (`default` preset) tạo ra một số interface khá hẹp,
tương tự vấn đề đã được ghi nhận và xử lý riêng cho `maze` (xem comment dài
trong `config.yaml` về `delta_max`, `gamma_w`, `delta_extra` cho maze). Scenario
`default` **chưa có** điều chỉnh tương tự — nó vẫn dùng
`delta_safe=0.02 / delta_extra=0.01` khá nhỏ trong khi `dynamics.delta_max`
toàn cục là 3.0.

Đây là yếu tố làm cho khoảng trống ở mục 3 dễ bộc lộ thành fail: với margin
cấu hình đã mỏng, chỉ cần `lip_gap` thật sau khi giải lệch khỏi ước lượng một
chút là certification rơi xuống âm.

## 5. Đây có phải "bug" do rebase hỏng không?

Không. Khác với lỗi `docs_dir`/`reporting.py` (code bị rebase làm rụng mất),
phần này hoạt động đúng như code hiện có được viết — và đúng như mục đích của
commit audit `2b8b0b3` ("Fix CRD diagnostics/safety-certificate reporting per
spec Sec 7-8 audit"): trước đây `certified_safety_margin` **thiếu** số hạng
`lip_gap`/`defect_norm`/`eps_int` nên có thể báo "CERTIFIED" sai cho một
đường đi không an toàn theo đúng định nghĩa spec Sec 8.2. Sau khi vá đúng công
thức, scenario `default` với cấu hình hiện tại **lộ ra** rằng nó chưa từng
thực sự đạt ngưỡng chứng nhận an toàn chặt — chỉ là trước đây bị báo sai
(false positive) nên "trông như" thành công.

## 6. Hướng khắc phục khả thi (đề xuất, chưa triển khai)

1. **Vòng lặp re-certify sau khi giải**: sau khi `BarrierDMSSolver.solve()`
   hội tụ, nếu `lip_gap` (tính từ `Delta_i` thật) > phần đã bơm vào
   `delta_safe_barrier` trước đó, re-solve thêm 1-2 lần với
   `delta_safe_barrier` mới = `lip_gap_thật + epsilon_final`, giống cách
   `optimizer.py:2622-2650` đã làm cho warm-start — chỉ khác là lặp lại trên
   chính kết quả NLP, không chỉ trên warm-start.
2. **Tăng `n_int` (n_integration_steps)** cho scenario `default` (hiện 44) để
   giảm `h_max`, giảm `lip_gap` — đổi lại NLP lớn hơn, giải lâu hơn.
3. **Tăng `delta_safe`/`delta_extra`** cho `default` giống cách đã làm riêng
   cho `maze`, để margin có sẵn đủ "đệm" cho phần Lipschitz gap phát sinh sau
   khi giải.
4. **Ràng buộc trực tiếp `Delta_i` trong NLP** theo giá trị đã dùng để tính
   `delta_safe_barrier` (ví dụ thêm bound `Delta_i <= delta_arr_warmstart_i *
   hệ số`) để đảm bảo `Delta_i` thật không vượt quá giả định ban đầu — đánh
   đổi: có thể làm mất nghiệm tốt hơn mà NLP có thể tìm ra.

Các hướng trên đều là *thay đổi hành vi/tham số*, không phải sửa lỗi cú pháp,
nên cần thảo luận và đo lại trên benchmark trước khi áp dụng.
