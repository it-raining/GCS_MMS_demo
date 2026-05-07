# GCS-MMS Demo

Tài liệu này hướng dẫn cách chạy demo GCS-MMS từ repo hiện tại, bao gồm chạy scenario có sẵn, chạy maze benchmark, bật/tắt CTCS safety certificate, xuất PNG/GIF, và chọn bộ config phù hợp cho từng mục tiêu.

## 1. Tổng Quan

Demo này giải bài toán motion planning bằng pipeline:

- Chia free space thành các convex region bằng ACD.
- Xây graph of convex sets trên các region.
- Giải continuous relaxation bằng IPOPT với biến graph flow `y_uv`, `p_v`.
- Mỗi active region có một local multiple-shooting segment với `s_v^-`, `s_v^+`, `w_v`, `Delta_v`.
- Optional fixed-path NLP polish sau khi trích path từ relaxation.
- Safety có thể chạy bằng mesh constraints, CTCS integral constraints, hoặc cả hai.

Lưu ý quan trọng: solver hiện tại dùng IPOPT continuous relaxation, không phải true mixed-integer solver.

## 2. Cài Đặt

Chạy từ root của repo:

```bash
cd /home/khoa/ws/GCS_MMS_demo
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Kiểm tra nhanh:

```bash
.venv/bin/python demo/main_demo.py --list-scenarios
.venv/bin/python demo/main_demo.py --list-presets
```

Nếu đang đứng trong folder `demo/`, dùng:

```bash
cd /home/khoa/ws/GCS_MMS_demo/demo
../.venv/bin/python main_demo.py --list-scenarios
```

## 3. Các Lệnh Chạy Nhanh

### Chạy scenario đơn giản

Từ root:

```bash
.venv/bin/python demo/main_demo.py --scenario simple
```

Từ folder `demo/`:

```bash
../.venv/bin/python main_demo.py --scenario simple
```

### Chạy default scenario

```bash
.venv/bin/python demo/main_demo.py --scenario default
```

### Chạy maze scenario trong `config.yaml`

```bash
.venv/bin/python demo/main_demo.py --scenario maze --quiet
```

### Chạy maze benchmark cụ thể

Đây là lệnh thường dùng nhất khi bạn muốn test maze tự sinh:

```bash
.venv/bin/python demo/main_demo.py \
  --maze-benchmark \
  --maze-count 1 \
  --maze-size 10 \
  --maze-knock-downs 10 \
  --maze-seed 4 \
  --maze-wall-thickness 0.08 \
  --quiet
```

Ý nghĩa:

- `--maze-count 1`: chạy 1 maze case.
- `--maze-size 10`: maze 10x10.
- `--maze-knock-downs 10`: phá thêm 10 tường ngẫu nhiên để mở thêm đường.
- `--maze-seed 4`: seed để tái lập kết quả.
- `--maze-wall-thickness 0.08`: độ dày tường.
- `--quiet`: giảm log console.

### Chạy benchmark nhỏ để debug nhanh

```bash
.venv/bin/python demo/main_demo.py \
  --maze-benchmark \
  --maze-count 1 \
  --maze-size 3 \
  --maze-knock-downs 0 \
  --maze-seed 4 \
  --maze-wall-thickness 0.08 \
  --quiet
```

## 4. Output Ở Đâu?

Mặc định output nằm trong:

```text
demo/results/
```

Với maze benchmark:

```text
demo/results/maze_benchmarks/
demo/results/maze_benchmarks/case_results/
```

Các file quan trọng:

- `*_summary.json`: thông số nghiệm, path, cost, safety diagnostics.
- `*_result.png`: hình tổng hợp environment, trajectory, safety, route.
- `*_animation.gif`: animation quá trình robot đi theo trajectory.
- `maze_benchmark_*.json`: summary tổng hợp benchmark.

Nếu bạn dùng config tạm nằm ngoài `demo/`, ví dụ `/tmp/my_config.yaml`, thì `results_dir: "results"` có thể resolve thành `/tmp/results`. Muốn chắc output nằm trong `demo/results`, hãy chạy với config nằm trong folder `demo/` hoặc đặt `output.results_dir` thành absolute path.

## 5. Config Quan Trọng

Các config chính nằm trong `demo/config.yaml`.

### Safety mode

```yaml
shooting:
  safety_mode: "both"   # "mesh" | "ctcs" | "both"
```

Ý nghĩa:

- `mesh`: chỉ enforce an toàn tại finite mesh points. Nhanh nhất.
- `ctcs`: enforce CTCS safety certificate, tức RK4 approximation của accumulated path-constraint violation.
- `both`: enforce cả mesh constraints và CTCS integral constraints. Chặt hơn nhưng chậm nhất.

CTCS trong repo này không phải full CT-SCVX solver. Nó là continuous-time constraint satisfaction penalty integral được discretize bằng RK4:

```text
eta_end ~= integral sum_i [A_i*pi(x(tau)) - b_i + safety_margin]_+^2 dtau
eta_end <= ctcs_tolerance
```

Sau solve vẫn cần dense post-check.

### Dense check

```yaml
shooting:
  dense_check_points: 100
  dense_check_tolerance: 1.0e-4
  fail_on_dense_violation: true
```

`dense_check_points` chỉ ảnh hưởng hậu kiểm sau solve. Nó không làm bài toán IPOPT nhỏ đi nhiều.

Gợi ý:

- Smoke test rất nhanh: `dense_check_points: 10`, `fail_on_dense_violation: false`
- Tuning: `dense_check_points: 100`
- Báo cáo/final: `dense_check_points: 500` hoặc `1000`
- Giữ `dense_check_tolerance: 1.0e-4` trong đa số trường hợp.

### Các knob ảnh hưởng tốc độ solve mạnh hơn

```yaml
shooting:
  n_integration_steps: 5
  n_mesh_points: 3

control:
  n_segments: 2

optimizer:
  ipopt:
    max_iter: 500
```

Tác động:

- `n_integration_steps`: tăng thì dynamics/CTCS chính xác hơn nhưng NLP lớn hơn.
- `n_mesh_points`: tăng thì thêm nhiều mesh safety constraints.
- `control.n_segments`: tăng thì trajectory linh hoạt hơn nhưng thêm biến.
- `ipopt.max_iter`: giới hạn thời gian chờ, nhưng có thể dừng trước khi nghiệm tốt.

## 6. Bộ Config Gợi Ý

Bạn có thể sửa trực tiếp `demo/config.yaml`, hoặc tạo file YAML riêng rồi chạy bằng `--config`.

### A. Debug nhanh nhất

Dùng để kiểm tra pipeline có chạy không.

```yaml
shooting:
  n_integration_steps: 5
  n_mesh_points: 3
  safety_margin: 0.02
  safety_mode: "mesh"
  ctcs_tolerance: 1.0e-5
  ctcs_penalty: "squared_hinge"
  ctcs_use_rk4_stages: true
  ctcs_eta_big_m: 100.0
  dense_check_points: 10
  dense_check_tolerance: 1.0e-4
  fail_on_dense_violation: false

control:
  n_segments: 2

optimizer:
  enforce_control_continuity: false
  ipopt:
    max_iter: 300
    tol: 1.0e-5
    print_level: 0
```

Lệnh chạy:

```bash
.venv/bin/python demo/main_demo.py \
  --maze-benchmark \
  --maze-count 1 \
  --maze-size 3 \
  --maze-knock-downs 0 \
  --maze-seed 4 \
  --quiet
```

### B. Baseline nhanh bằng mesh

Dùng để so sánh với behavior cũ.

```yaml
shooting:
  n_integration_steps: 10
  n_mesh_points: 5
  safety_margin: 0.02
  safety_mode: "mesh"
  dense_check_points: 100
  dense_check_tolerance: 1.0e-4
  fail_on_dense_violation: false

control:
  n_segments: 2

optimizer:
  enforce_control_continuity: false
  ipopt:
    max_iter: 1000
    tol: 1.0e-6
    print_level: 0
```

Lệnh chạy maze 10x10:

```bash
.venv/bin/python demo/main_demo.py \
  --maze-benchmark \
  --maze-count 1 \
  --maze-size 10 \
  --maze-knock-downs 10 \
  --maze-seed 4 \
  --maze-wall-thickness 0.08 \
  --quiet
```

### C. CTCS-only nhẹ để hội tụ nhanh hơn

Dùng khi maze lớn và `safety_mode: "both"` tạo quá nhiều ràng buộc do vừa có mesh constraints vừa có CTCS constraints. Chế độ này bỏ mesh interior constraints khỏi NLP, chỉ giữ CTCS integral và dense post-check sau solve. Đây là cấu hình tốt để thử tốc độ, nhưng không nên xem là chứng nhận safety cuối cùng nếu `dense_check_tolerance` được nới quá rộng.

```yaml
shooting:
  n_integration_steps: 10
  n_mesh_points: 3
  safety_margin: 0.02
  safety_mode: "ctcs"
  ctcs_tolerance: 1.0e-4
  ctcs_penalty: "squared_hinge"
  ctcs_integral_mode: "normalized"
  ctcs_use_rk4_stages: true
  ctcs_eta_big_m: 100.0
  dense_check_points: 10
  dense_check_tolerance: 5.0e-3
  fail_on_dense_violation: false

control:
  n_segments: 2

optimizer:
  enforce_control_continuity: false
  path_polish_candidates: 8
  ipopt:
    max_iter: 1200
    tol: 1.0e-6
    print_level: 0
```

Lệnh chạy maze 10x10:

```bash
.venv/bin/python demo/main_demo.py \
  --maze-benchmark \
  --maze-count 1 \
  --maze-size 10 \
  --maze-knock-downs 10 \
  --maze-seed 4 \
  --maze-wall-thickness 0.08 \
  --quiet
```

Nếu chỉ muốn smoke-test rất nhanh, có thể nới thêm:

```yaml
shooting:
  ctcs_tolerance: 1.0e-3
  dense_check_tolerance: 2.0e-2
optimizer:
  path_polish_candidates: 3
  ipopt:
    max_iter: 800
```

### D. CTCS + mesh nhanh để thử nghiệm

Dùng khi muốn bật CTCS nhưng vẫn giữ mesh constraints ở mức thấp.

```yaml
shooting:
  n_integration_steps: 5
  n_mesh_points: 3
  safety_margin: 0.02
  safety_mode: "both"
  ctcs_tolerance: 1.0e-5
  ctcs_penalty: "squared_hinge"
  ctcs_use_rk4_stages: true
  ctcs_eta_big_m: 100.0
  dense_check_points: 100
  dense_check_tolerance: 1.0e-4
  fail_on_dense_violation: false

control:
  n_segments: 2

optimizer:
  enforce_control_continuity: false
  ipopt:
    max_iter: 500
    tol: 1.0e-6
    print_level: 0
```

Nên bắt đầu với maze nhỏ:

```bash
.venv/bin/python demo/main_demo.py \
  --maze-benchmark \
  --maze-count 1 \
  --maze-size 3 \
  --maze-knock-downs 0 \
  --maze-seed 4 \
  --maze-wall-thickness 0.08 \
  --quiet
```

Sau đó tăng dần:

```bash
.venv/bin/python demo/main_demo.py \
  --maze-benchmark \
  --maze-count 1 \
  --maze-size 5 \
  --maze-knock-downs 3 \
  --maze-seed 4 \
  --maze-wall-thickness 0.08 \
  --quiet
```

### E. CTCS kỹ hơn để báo cáo

Dùng khi cần kết quả có safety diagnostics đáng tin hơn. Chạy sẽ lâu hơn rõ rệt.

```yaml
shooting:
  n_integration_steps: 20
  n_mesh_points: 10
  safety_margin: 0.02
  safety_mode: "both"
  ctcs_tolerance: 1.0e-6
  ctcs_penalty: "squared_hinge"
  ctcs_use_rk4_stages: true
  ctcs_eta_big_m: 100.0
  dense_check_points: 1000
  dense_check_tolerance: 1.0e-4
  fail_on_dense_violation: true

control:
  n_segments: 2

optimizer:
  enforce_control_continuity: false
  ipopt:
    max_iter: 3500
    tol: 1.0e-6
    print_level: 0
```

Lệnh chạy:

```bash
.venv/bin/python demo/main_demo.py \
  --maze-benchmark \
  --maze-count 1 \
  --maze-size 10 \
  --maze-knock-downs 10 \
  --maze-seed 4 \
  --maze-wall-thickness 0.08 \
  --quiet
```

Nếu case này quá lâu, giảm trước:

```yaml
shooting:
  n_integration_steps: 10
  dense_check_points: 300
optimizer:
  ipopt:
    max_iter: 1500
```

## 7. Tạo Config Riêng Và Chạy Luôn

Ví dụ tạo file `demo/config_fast_ctcs.yaml`:

```yaml
problem:
  default_preset: "maze"

start_state:
  position: [0.2, 0.2]
  heading: 0.785

goal_state:
  position: [4.8, 4.8]
  heading: 0.785

dynamics:
  model: "unicycle"
  v_min: -2.0
  v_max: 2.0
  omega_min: -3.14159
  omega_max: 3.14159
  delta_min: 0.01
  delta_max: 10.0

control:
  parameterization: "piecewise_constant"
  n_segments: 2

shooting:
  n_integration_steps: 5
  n_mesh_points: 3
  safety_margin: 0.02
  safety_mode: "both"
  ctcs_tolerance: 1.0e-5
  ctcs_penalty: "squared_hinge"
  ctcs_use_rk4_stages: true
  ctcs_eta_big_m: 100.0
  dense_check_points: 100
  dense_check_tolerance: 1.0e-4
  fail_on_dense_violation: false

cost:
  a: 1.0
  w_L: 1.0
  w_E: 1.0
  w_u_smooth: 0.2

graph:
  max_paths: 1500

optimizer:
  enforce_control_continuity: false
  path_polish_candidates: 3
  ipopt:
    max_iter: 500
    tol: 1.0e-6
    print_level: 0
  big_M:
    position: 20.0
    interface: 20.0
    time: 100.0

visualization:
  figsize: [10, 10]
  show_mesh_points: true
  animation:
    fps: 15
    duration: 4.0

scenarios:
  maze:
    name: "Maze Fast CTCS"
    description: "Small/medium generated maze with fast CTCS settings"
    active_regions: "all"
    problem_preset: "maze"

output:
  results_dir: "results"
  save_png: true
  save_gif: true
  save_json: true
  verbosity: 1
```

Chạy bằng config riêng:

```bash
.venv/bin/python demo/main_demo.py \
  --config demo/config_fast_ctcs.yaml \
  --maze-benchmark \
  --maze-count 1 \
  --maze-size 10 \
  --maze-knock-downs 10 \
  --maze-seed 4 \
  --maze-wall-thickness 0.08 \
  --quiet
```

## 8. Bật PNG/GIF

Trong config:

```yaml
output:
  results_dir: "results"
  save_png: true
  save_gif: true
  save_json: true

visualization:
  animation:
    fps: 15
    duration: 4.0
```

Lưu ý:

- GIF có thể làm runtime lâu hơn sau khi solver đã xong.
- Khi đang debug solver, có thể đặt `save_gif: false`.
- Khi cần báo cáo, bật `save_png: true`, `save_gif: true`, `save_json: true`.

## 9. Đọc Kết Quả

Trong `*_summary.json`, các trường nên xem:

- `success`: solver và post-check có được coi là thành công không.
- `solver_status`: trạng thái IPOPT, polish, warning dense violation nếu có.
- `total_cost`: tổng cost.
- `solve_time`: thời gian solve.
- `path`: path theo region graph.
- `path_regions`: danh sách region active.
- `defect_norm`: lỗi dynamics defect.
- `constraint_violation`: max NLP constraint residual.
- `max_ctcs_integral`: max RK4 approximation của accumulated path-constraint violation.
- `max_dense_region_violation`: max signed region violation từ dense post-check.
- `time_durations`: duration từng local segment.

Diễn giải safety:

- `max_dense_region_violation <= dense_check_tolerance`: dense post-check đạt theo tolerance.
- `max_dense_region_violation > dense_check_tolerance`: trajectory có thể rời assigned convex region tại các điểm dense sample.
- `max_ctcs_integral` nhỏ không phải proof safety liên tục tuyệt đối; nó là RK4 approximation và vẫn cần dense post-check.

## 10. Khi Chạy Quá Lâu

Nếu lệnh maze 10x10 chờ quá lâu, giảm theo thứ tự:

1. `safety_mode: "mesh"` để lấy baseline path.
2. `n_integration_steps: 5`.
3. `n_mesh_points: 3`.
4. `dense_check_points: 10` hoặc `100`.
5. `optimizer.ipopt.max_iter: 300` hoặc `500`.
6. `output.save_gif: false`.
7. Maze nhỏ hơn, ví dụ `--maze-size 3` hoặc `--maze-size 5`.

Ví dụ command debug nhanh:

```bash
.venv/bin/python demo/main_demo.py \
  --maze-benchmark \
  --maze-count 1 \
  --maze-size 3 \
  --maze-knock-downs 0 \
  --maze-seed 4 \
  --maze-wall-thickness 0.08 \
  --quiet
```

Khi đã ổn, tăng dần:

```bash
.venv/bin/python demo/main_demo.py \
  --maze-benchmark \
  --maze-count 1 \
  --maze-size 5 \
  --maze-knock-downs 3 \
  --maze-seed 4 \
  --maze-wall-thickness 0.08 \
  --quiet
```

Rồi mới chạy:

```bash
.venv/bin/python demo/main_demo.py \
  --maze-benchmark \
  --maze-count 1 \
  --maze-size 10 \
  --maze-knock-downs 10 \
  --maze-seed 4 \
  --maze-wall-thickness 0.08 \
  --quiet
```

## 11. Các Lệnh Hữu Ích Khác

List scenario:

```bash
.venv/bin/python demo/main_demo.py --list-scenarios
```

List geometry preset:

```bash
.venv/bin/python demo/main_demo.py --list-presets
```

Chỉ visualize environment:

```bash
.venv/bin/python demo/main_demo.py --visualize-only --scenario default
```

In formulation summary:

```bash
.venv/bin/python demo/main_demo.py --formulation
```

Chạy CTCS self-test:

```bash
.venv/bin/python demo/ctcs_self_test.py
```
