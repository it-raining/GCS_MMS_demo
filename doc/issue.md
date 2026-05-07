### Issue A: transition point không bị ép nằm trong “intersection polygon” explicit, mà chỉ qua hai closure constraints

Trong fixed-path NLP, điểm nối không có biến intersection riêng. Nó chỉ được suy ra từ:

[
s_u^+ = s_v^-,
\quad
s_u^+ \in \overline{Q_u},
\quad
s_v^- \in \overline{Q_v}.
]

Về toán học, nếu H-rep chính xác thì điều này đủ để điểm nối nằm trong (\overline{Q_u} \cap \overline{Q_v}). Nhưng về số học, `boundary_tolerance` cho phép lệch nhẹ; nếu hai region chỉ tiếp xúc ở một cạnh rất mỏng hoặc một điểm, tolerance và Big-M có thể làm interface “gần đúng” nằm ngoài giao thật một chút. Điều này thường nhỏ, nhưng trong maze hoặc vùng hẹp, nó có thể tạo connection trông hợp lệ trong NLP nhưng sát obstacle/biên.

### Issue B: vùng giao không phải một shooting region riêng, nên không có dynamics/cost riêng trong phần overlap

Nếu robot đi từ `R_u` sang `R_v`, phần trajectory trước interface nằm trong `R_u`, sau interface nằm trong `R_v`. Vùng giao chỉ đóng vai trò là cửa nối, không phải một convex set riêng với segment động lực học riêng. Điều này không sai nếu mục tiêu chỉ cần một điểm chuyển vùng. Nhưng nếu giao có diện tích đáng kể, robot có thể đi một đoạn dài trong overlap; formulation hiện tại không mô hình hoá overlap như một “mode” riêng. Nó bắt buộc mỗi đoạn shooting được gán cho một region gốc, và điểm đổi region là một instant.

Hệ quả: nghiệm có thể kém linh hoạt hơn so với formulation cho phép overlap là một region độc lập, vì không thể “ở trong giao” như một stage riêng trừ khi nó vẫn được tính là nằm trong một trong hai region cha.

### Issue C: có thể “nhảy cóc” qua vùng lồi trung gian nếu graph có cạnh do intersection không rỗng

Graph thêm cạnh cho mọi cặp region có giao không rỗng, không chỉ các region kề theo cấu trúc decomposition cục bộ. Nếu ACD2D sinh các region chồng lấn rộng hoặc buffer làm nhiều vùng giao nhau, graph có thể có cạnh xa về mặt topology trực giác. Khi đó solver có thể chọn `R_i -> R_j` trực tiếp nếu hai H-polytopes giao nhau, bỏ qua region trung gian. Về mặt union-of-convex-sets, điều này không nhất thiết sai: nếu hai set thật sự overlap, chuyển trực tiếp là hợp lệ. Nhưng về mặt diễn giải “đi qua các cell kề nhau” thì nó có thể trông như nhảy cóc.

Điều cần phân biệt: “nhảy cóc” là vấn đề nếu cạnh đó sinh ra do artifact của region construction/buffering/tolerance; còn nếu hai convex sets thật sự overlap trong free space, thì direct transition là hợp lệ và còn có thể tốt hơn.

### Issue D: interior safety bỏ endpoint, nên ngay sát interface không có margin

Endpoint/interface chỉ closure, không strict margin. Interior mesh points mới có safety margin. Vì vậy đoạn rất gần interface có thể bắt đầu từ biên rồi đi vào trong; nếu dynamics cong ra phía ngoài giữa endpoint và interior sample đầu tiên, sampled constraints có thể bỏ sót. Đây là vấn đề đã thấy ở logic `n_mesh_points`: càng ít mesh point thì khoảng giữa endpoint và interior sample càng dài.

## 6. So sánh “loại bỏ vùng giao giữa 2 cạnh” và “không loại bỏ”

Ở đây cần nói rõ “vùng giao” có thể hiểu theo hai cách:

1. **Không tạo region riêng cho intersection** — hiện tại đang làm như vậy.
2. **Loại phần overlap khỏi các region cha để tạo partition disjoint** — tức các region không còn chồng lấn.

### Trường hợp hiện tại: không loại bỏ overlap khỏi region cha

Ưu điểm:

* Graph dễ nối hơn vì chỉ cần hai region có giao không rỗng.
* Transition qua shared boundary/overlap tự nhiên hơn.
* Fixed-path NLP có thể chọn interface ở bất kỳ điểm nào thuộc giao, không cần tạo thêm cell.
* Số lượng region thường ít hơn so với việc split thành partition disjoint.

Nhược điểm:

* Một điểm trong overlap thuộc nhiều region, nên discrete path không còn unique.
* Relaxation có thể fractional hơn vì nhiều path tương đương cùng mô tả một vùng không gian.
* Có thể sinh cạnh “tắt” qua overlap nhỏ, làm path extraction chọn transition kém robust.
* Không có chi phí/động học riêng cho overlap như một phase độc lập.

### Nếu loại bỏ overlap để các region gần như disjoint

Ưu điểm:

* Discrete path rõ ràng hơn; ít ambiguity.
* Giảm số cạnh do overlap artifact.
* Path extraction có thể ổn định hơn vì ít path tương đương.
* Dễ diễn giải theo cell decomposition cổ điển.

Nhược điểm lớn:

* Nếu loại bỏ overlap quá mạnh, hai region chỉ còn tiếp xúc qua boundary rất mỏng hoặc không còn giao; bài toán có thể mất feasibility.
* Với unicycle và multiple shooting, transition qua một cạnh mỏng yêu cầu endpoint đúng trên boundary; điều này numerically khó hơn so với có overlap có diện tích.
* Không gian feasible bị thu hẹp, có thể làm cost xấu hơn hoặc solver khó hội tụ hơn.
* Nếu có gap nhỏ do numerical geometry, graph có thể mất cạnh dù về hình học liên tục đáng lẽ nối được.

### Nhận định cho code hiện tại

Với implementation này, **không nên đơn giản loại bỏ vùng giao** nếu chưa thay bằng một interface formulation tốt hơn. Overlap đang đóng vai trò “numerical slack hình học” cho transition. Nếu xoá overlap, bạn làm bài toán discrete rõ hơn nhưng có nguy cơ làm NLP khó hơn, đặc biệt vì endpoint chỉ được phép ở closure và dynamics unicycle không thể đổi hướng tùy ý.

Một hướng tốt hơn là giữ overlap nhưng kiểm soát nó:

* chỉ tạo cạnh nếu intersection có diện tích/chiều dài vượt ngưỡng, không chỉ non-empty;
* lưu intersection polygon và dùng nó làm interface set rõ ràng;
* thêm penalty/regularizer cho interface nằm sâu trong intersection thay vì sát biên;
* tăng validation mesh quanh interface;
* nếu muốn partition disjoint, tạo thêm “portal/interface constraints” thay vì để solver tự suy ra từ hai closure constraints.

## 7. Kết luận ngắn

Trong code hiện tại, CasADi + IPOPT luôn cho **NLP continuous relaxation**, không cho mixed-integer/global optimality. Các biến `y,p` chỉ là binary-like trong formulation nhưng được relax trong `[0,1]`. Vì vậy path extraction là heuristic: greedy nếu nghiệm gần nhị phân, weighted shortest path nếu fractional. Sau đó fixed-path NLP polish là bước cần thiết để biến path rời rạc đã chọn thành trajectory liên tục và vật lý hơn.

Entry/exit được xử lý bằng closure:

[
A_v p \le b_v + \text{boundary_tolerance},
]

còn interior mesh mới bị ép:

[
A_v p_k \le b_v - \text{safety_margin}.
]

Khi hai region giao nhau, integrated solver dùng `z_uv` nằm trong closure của cả hai region và ép `s_u^+ = z_uv = s_v^-`; fixed-path solver bỏ `z_uv` và ép trực tiếp `s_u^+ = s_v^-`, còn membership trong giao được suy ra từ endpoint closure của hai region. Logic này hợp lý, nhưng có rủi ro numerical và modelling: overlap gây ambiguity/fractionality, intersection nhỏ gây transition mong manh, và endpoint/interface không có safety margin nên không phải chứng chỉ an toàn liên tục.
