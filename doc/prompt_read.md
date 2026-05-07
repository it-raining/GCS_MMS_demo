Dưới đây là **3 prompt có thể copy trực tiếp** cho AI Agent. Tôi thiết kế chúng dựa trên ba trục tài liệu bạn nêu: bài GCS dùng vùng an toàn lồi và Bézier để bảo đảm toàn bộ đoạn quỹ đạo nằm trong vùng lồi, thay vì chỉ kiểm tra tại vài điểm; report_v4 đang ghép GCS/network flow với direct multiple shooting nhưng phần an toàn hiện vẫn dựa vào kiểm tra trên lưới hữu hạn; còn ct-scvx gợi ý hướng “tích lũy mức vi phạm liên tục” để tránh hiện tượng quỹ đạo văng giữa các điểm kiểm tra mà không cần tăng số nút.   

---

## Prompt 1 — Nhóm bài toán hình học: đường đi ngắn nhất và chuyển vùng

```text
Bạn là một AI Agent nghiên cứu về motion planning, tối ưu hình học và Graphs of Convex Sets. Hãy đọc các tài liệu đã được cung cấp, đặc biệt là:
- An Iterative Algorithm for Computing Shortest Paths Through Line Segments in 3D;
- các phần liên quan đến Shortest Path Problem trong Graphs of Convex Sets;
- Motion Planning around Obstacles with Convex Optimization;
- report_v4.

Mục tiêu của bạn là giúp tôi hiểu nhóm bài toán “đường đi ngắn nhất qua các đối tượng hình học trung gian” và liên hệ nó với bài Motion Planning using GCS.

Yêu cầu phân tích:

1. Tóm tắt cốt lõi bài toán hình học:
   - Bài toán đang tối ưu biến gì?
   - Các điểm chuyển tiếp nằm trên đoạn thẳng, mặt giao, hay vùng giao đóng vai trò gì?
   - Vì sao đường đi ngắn nhất thường có cấu trúc đường gấp khúc hoặc ghép từ các đoạn cục bộ?
   - Cơ chế giảm chiều dài qua từng vòng lặp là gì?
   - Tính chất “chi phí giảm đơn điệu” hoặc “đường đi được rút ngắn dần” có thể hiểu thế nào ở mức trực giác học thuật?

2. Rút ra các ý tưởng có thể chuyển sang GCS:
   - Điểm chuyển tiếp trong bài shortest path tương ứng với biến interface z_uv trong report_v4 như thế nào?
   - Điều kiện “đi qua một dãy đoạn thẳng” tương ứng với điều kiện “đi qua một chuỗi vùng lồi” như thế nào?
   - Điều kiện điểm nối thuộc giao của hai vùng lồi Q_u ∩ Q_v có đủ để bảo đảm cả đoạn quỹ đạo nằm trong vùng lồi hay không?
   - Nếu không đủ, cần thêm điều kiện lý thuyết nào?

3. Tập trung vào lỗi quỹ đạo “văng” ra ngoài vùng lồi:
   - Phân biệt rõ ba mức kiểm tra:
     a) chỉ kiểm tra entry/exit point;
     b) kiểm tra một số điểm mẫu trên đoạn;
     c) chứng minh toàn bộ đoạn liên tục nằm trong vùng.
   - Đề xuất các hướng khắc phục không dựa chủ yếu vào tăng số điểm ràng buộc:
     a) dùng tính chất bao lồi của đường cong Bézier/Bernstein;
     b) dùng ràng buộc tích lũy mức vi phạm liên tục theo thời gian;
     c) co vùng lồi lại bằng một biên an toàn có chứng minh sai số;
     d) thêm điều kiện hướng chuyển động tại mặt giao hoặc nón tiếp tuyến của vùng lồi.

4. Liên hệ với report_v4:
   - Kiểm tra xem biến z_uv, s_u^+, s_v^- trong report_v4 đã mô hình hóa đúng “điểm chuyển vùng” chưa.
   - Chỉ ra trường hợp mô hình có thể chạy sai dù z_uv ∈ Q_u ∩ Q_v.
   - Kiểm tra xem formulation hiện tại có đang nhầm giữa “điểm nối hợp lệ” và “đoạn quỹ đạo hợp lệ liên tục” hay không.

5. Tạo đầu ra theo cấu trúc sau:
   - Tóm tắt 10 dòng về bài toán hình học.
   - Bảng 4 cột: “Ý tưởng trong shortest path” | “Đối ứng trong GCS” | “Rủi ro khi áp dụng vào report_v4” | “Cách sửa lý thuyết”.
   - Danh sách 8 câu hỏi nghiên cứu cần tự trả lời khi đọc bài Motion Planning using GCS.
   - Danh sách 5 kiểm tra cụ thể để phát hiện vì sao quỹ đạo trong report_v4 có thể văng khỏi vùng lồi.
   - Đề xuất 3 hướng sửa mô hình ưu tiên, sắp xếp theo mức ít làm tăng kích thước bài toán nhất.

Văn phong: học thuật, rõ ràng, không dùng thuật ngữ quá chuyên ngành mà không giải thích. Mỗi thuật ngữ như “interface”, “convex hull”, “path constraint”, “shooting state” phải được diễn giải bằng tiếng Việt tương đương.
```

---

## Prompt 2 — Nhóm OCP dùng multiple shooting / MSM / MMS

```text
Bạn là một AI Agent chuyên về optimal control problem, direct multiple shooting và kiểm định mô hình tối ưu điều khiển. Hãy đọc các tài liệu đã được cung cấp, đặc biệt là:
- Using a Direct Multiple Shooting Method to Control a Quadrotor;
- A Convergence Guaranteed Multiple-Shooting DDP Method for Optimization-Based Robot Motion Planning;
- Continuous-time successive convexification for constrained trajectory optimization;
- report_v4.

Mục tiêu của bạn là giúp tôi hiểu sâu hơn về multiple shooting nói riêng và optimal control problem nói chung, nhằm áp dụng đúng vào bài toán đồ thị của các tập lồi.

Yêu cầu phân tích:

1. Giải thích bản chất của direct multiple shooting:
   - Vì sao phải chia quỹ đạo thành nhiều đoạn?
   - “Shooting state” là gì, khác gì với trạng thái sinh ra bằng tích phân từ điểm đầu?
   - “Defect constraint” hoặc “điều kiện nối khớp” có vai trò gì?
   - Multiple shooting giúp gì so với single shooting khi hệ dài, phi tuyến, hoặc nhạy với điều kiện đầu?
   - Multiple shooting khác direct collocation ở điểm nào?

2. Viết lại OCP tổng quát trong report_v4:
   - state x(t), control u(t), final time T;
   - terminal cost và running cost;
   - dynamics x_dot = f(x,u);
   - path constraints x(t) ∈ Q, u(t) ∈ U;
   - cách chuẩn hóa thời gian về τ ∈ [0,1].

3. Audit logic multiple shooting trong report_v4:
   - Kiểm tra vai trò của s_v^-, s_v^+, Δ_v, w_v, F_v.
   - Kiểm tra ràng buộc s_v^+ = F_v(s_v^-, w_v, Δ_v) có đúng nghĩa local IVP không.
   - Kiểm tra việc bật/tắt local block bằng p_v có hợp lý không.
   - Kiểm tra xem các vùng inactive có bị “trôi biến” hay đã bị khóa đúng cách.
   - Kiểm tra objective dạng p_v J_v có nên thay bằng biến epigraph ρ_v và indicator/disjunction không.

4. Phân tích phần an toàn hình học:
   - report_v4 hiện đang dùng sampling trên lưới hữu hạn để xấp xỉ điều kiện x_v(τ) ∈ Q_v.
   - Hãy chỉ ra vì sao cách này có thể dẫn đến hiện tượng “văng” giữa hai điểm kiểm tra.
   - So sánh với hướng continuous-time constraint satisfaction: tích phân mức vi phạm không âm bằng 0.
   - Đề xuất cách nhúng ý tưởng này vào local multiple-shooting block mà không tăng số điểm ràng buộc.

5. Liên hệ với GCS:
   - Trong GCS-Bézier, vì sao ép các điểm điều khiển nằm trong vùng lồi có thể bảo đảm toàn bộ đường cong nằm trong vùng?
   - Nếu thay Bézier bằng nghiệm ODE từ local IVP, bảo đảm bao lồi còn đúng không?
   - Nếu không, cần dùng chứng nhận nào thay thế?
   - Có thể kết hợp multiple shooting với Bézier/hàm cơ sở điều khiển để giữ bài toán nhỏ hơn không?

6. Tối ưu kích thước mô hình:
   - Biến p_v có cần giữ riêng không hay có thể suy ra từ y_uv?
   - Biến z_uv có cần cho mọi cạnh hay chỉ cho cạnh active theo perspective/homogenization?
   - Có thể giảm số biến điều khiển w_v bằng điều khiển hằng, piecewise constant ít đoạn, hoặc basis bậc thấp không?
   - Có thể bỏ một số biến entry/exit nếu dùng edge-based segment thay vì vertex-based segment không?
   - Phần nào nên giữ lồi, phần nào bắt buộc là phi lồi?

7. Tạo đầu ra theo cấu trúc sau:
   - Một “bản đồ khái niệm” giữa OCP, multiple shooting, GCS và report_v4.
   - Bảng audit report_v4 gồm: “thành phần mô hình” | “đúng/sai/nghi ngờ” | “lý do” | “cách sửa”.
   - Danh sách các lỗi mô hình có thể làm solver trả nghiệm nhìn hợp lệ nhưng quỹ đạo thật không hợp lệ.
   - 10 câu hỏi đào sâu để tôi tự kiểm tra hiểu biết về MSM/MMS.
   - 5 đề xuất cụ thể để giảm kích thước bài toán mà vẫn giữ ý nghĩa multiple shooting.

Văn phong: học thuật, có giải thích trực giác. Không viết như văn nói. Khi dùng thuật ngữ tiếng Anh, hãy kèm diễn giải tiếng Việt trong ngoặc.
```

---

## Prompt 3 — Nhóm Motion planning / quỹ đạo hình học dùng đồ thị các tập lồi

```text
Bạn là một AI Agent chuyên đọc bài Motion Planning around Obstacles with Convex Optimization và đối chiếu nó với mô hình trong report_v4. Hãy tập trung vào cách bài GCS biến motion planning quanh vật cản thành bài toán đường đi ngắn nhất trên đồ thị các tập lồi.

Tài liệu cần đọc:
- Motion Planning around Obstacles with Convex Optimization;
- A Biconvex Method for Minimum-Time Motion Planning Through Sequences of Convex Sets;
- Continuous-time successive convexification for constrained trajectory optimization;
- report_v4.

Mục tiêu của bạn:
Tóm tắt, ghi chú và tạo câu hỏi để tôi hiểu rõ cách áp dụng GCS vào bài toán có multiple shooting, đồng thời tìm cách sửa hiện tượng quỹ đạo văng ra khỏi vùng lồi và tối ưu kích thước mô hình.

Yêu cầu phân tích:

1. Tóm tắt bài Motion Planning using GCS:
   - Không gian tự do được chia thành các vùng an toàn lồi như thế nào?
   - Đồ thị được xây dựng ra sao: đỉnh, cạnh, source, target?
   - Mỗi vùng lồi được gắn với biến liên tục nào?
   - Vai trò của đường cong Bézier và time-scaling là gì?
   - Điều kiện nào bảo đảm đoạn quỹ đạo nằm hoàn toàn trong vùng lồi?
   - Điều kiện nào bảo đảm nối trơn giữa hai vùng liên tiếp?
   - Hàm mục tiêu gồm thời gian, chiều dài, năng lượng/vận tốc được biểu diễn như thế nào?
   - Vì sao bài toán có thể đưa về mixed-integer convex program rồi thường giải bằng relaxation + rounding?

2. Đối chiếu GCS-Bézier với report_v4:
   - GCS-Bézier là bài toán hình học/quỹ đạo hình học có chứng nhận lồi mạnh.
   - report_v4 đang mở rộng sang optimal control bằng local IVP và endpoint map F_v.
   - Hãy chỉ ra chính xác phần nào còn giữ được tính lồi và phần nào trở thành phi lồi.
   - Kiểm tra nhận định: mô hình report_v4 không còn là MISOCP như GCS-Bézier mà đúng hơn là MIOCP ở continuous-time level và MINLP sau khi rời rạc hóa.
   - Kiểm tra việc dùng perspective: perspective xử lý tốt membership của tập lồi khi bật/tắt biến, nhưng không tự động xử lý được dynamics phi tuyến và defect constraints.

3. Tập trung vào hiện tượng quỹ đạo “văng” ra ngoài vùng lồi:
   - Liệt kê các nguyên nhân có thể xảy ra:
     a) chỉ ràng buộc entry/exit;
     b) chỉ ràng buộc mesh points;
     c) nội suy hoặc nghiệm ODE cong ra ngoài giữa các điểm;
     d) điểm chuyển vùng đúng nhưng vector vận tốc/gia tốc đẩy quỹ đạo ra ngoài;
     e) safe region bị hiểu là điều kiện trên state x thay vì phần hình học π(x).
   - Đề xuất các sửa lý thuyết không ưu tiên tăng số điểm ràng buộc:
     a) nếu dùng Bézier/Bernstein cho quỹ đạo hình học, ép toàn bộ control points nằm trong Q_v;
     b) dùng auxiliary state tích lũy mức vi phạm liên tục của A_v x(τ) ≤ b_v;
     c) dùng vùng lồi co lại Q_v^δ để hấp thụ sai số giữa mô hình và thực thi;
     d) dùng điều kiện bất biến thuận hoặc nón tiếp tuyến tại biên vùng lồi;
     e) tách trajectory-shape layer bằng GCS/SCS rồi dùng OCP/MSM như post-processing có chứng nhận safety.

4. Check tính đúng đắn model trong report_v4:
   - Đọc formulation hiện tại và đánh dấu các điểm có thể chạy sai.
   - Kiểm tra các điều kiện flow có loại chu trình đủ không.
   - Kiểm tra active/inactive block có đủ chặt không.
   - Kiểm tra boundary conditions với source/target.
   - Kiểm tra interface constraints trên z_uv.
   - Kiểm tra safety transcription: mesh-based, margin, hay continuous-time certificate.
   - Kiểm tra objective và các biến cost ρ_v.

5. Tối ưu kích thước bài toán:
   - Đề xuất cách giảm biến nhị phân, biến interface, biến activation, biến điều khiển.
   - Đề xuất khi nào nên dùng vertex-based segment và khi nào nên dùng edge-based segment.
   - Đề xuất khi nào nên fix thời gian local, khi nào cho Δ_v là biến.
   - Đề xuất cách giữ phần hình học của GCS là convex, còn phần dynamics xử lý bằng một lớp refine riêng.
   - Đề xuất pipeline nhỏ nhất nhưng vẫn an toàn: GCS chọn chuỗi vùng → chứng nhận containment → multiple shooting refine → continuous-time safety check.

6. Tạo đầu ra theo cấu trúc sau:
   - Executive summary 12 dòng.
   - Bảng “GCS-Bézier vs report_v4-MSM”.
   - Bảng “nguyên nhân quỹ đạo văng” | “dấu hiệu trong nghiệm” | “cách sửa lý thuyết” | “tác động lên kích thước bài toán”.
   - Checklist 20 mục để audit report_v4.
   - 12 câu hỏi nghiên cứu liên kết tới hiện thực, ví dụ:
     + Khi nào GCS chỉ cần geometric trajectory, khi nào bắt buộc OCP?
     + Nếu local IVP không có tính bao lồi, chứng nhận safety nào thay thế Bézier?
     + Có thể biến continuous-time violation integral thành một ràng buộc nhỏ gọn cho từng vùng không?
     + Có nên để GCS chọn chuỗi vùng trước rồi MSM tối ưu sau, hay tối ưu đồng thời?
     + Tối ưu đồng thời có đáng với chi phí MINLP tăng không?

Văn phong: học thuật, mạch lạc, ưu tiên giải thích bằng tiếng Việt. Không dùng thuật ngữ quá chuyên ngành mà không giải thích.
```

---