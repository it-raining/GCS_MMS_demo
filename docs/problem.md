- tại sao overlap_width yêu cầu một đại lượng quá lớn như vậy (tối thiểu 0.25 để chạy được cho demo crd hiện tại, trong khi về lý thuyết chỉ cần 0.075-0.1 là đủ để chạy) --> Cần xử lý các thông số sao cho vừa chạy được giải thuật vừa không bị overlap quá mức (mức chấp nhận overlap của mỗi vùng phình theo cạnh bao nhiêu cũng được, miễn là không đi xuyên hay chạm vào vật cản)

- vùng overlap đang bị dính vào obstacle, khiến hiện thực robot đi thẳng vào obstacle; lý do chính là buffer clip sai domain 

Trong convex_regions.py:337-340, create_buffered_regions_from_vertices_list clip vào workspace_polygon (boundary hình chữ nhật) chứ không clip vào free space (workspace − obstacles):

workspace = Polygon(workspace_vertices)          # chỉ là bounding box
buffered  = poly.buffer(buffer_size, join_style='mitre')
clipped   = buffered.intersection(workspace)     # obstacle interior không bị loại trừ

ACD2D đảm bảo các polygon gốc nằm trong free space. Nhưng khi buffer mở rộng ra ngoài, nó vượt qua ranh giới obstacle mà không bị ngăn — vì workspace clip không biết về obstacles. Kết quả: ConvexRegion sau buffer có thể xâm phạm obstacle interior.

Vùng màu cam (interface points) được tính trong C_i ∩ C_j — nếu giao đó extend vào obstacle, QP sẽ đặt điểm trong obstacle.

---
Các vấn đề cần xem xét theo mức độ quan trọng

1. Clip buffer vào free space thay vì workspace (root cause)

# Cần thay thế:
free_space = workspace_poly.difference(unary_union(obstacle_polys))
clipped = buffered.intersection(free_space)

create_buffered_regions_from_vertices_list cần nhận thêm obstacle_vertices — hiện nó không có parameter này. scenario_builder.py:122–125 gọi hàm này nhưng không truyền obstacles dù đã có preset.obstacle_vertices.

2. overlap_width so với khoảng cách đến obstacle

Nếu một vùng ACD2D nằm gần obstacle với khoảng cách d < overlap_width, buffer sẽ luôn xâm phạm obstacle dù clip đúng hay sai. Cần đảm bảo overlap_width nhỏ hơn khoảng cách tối thiểu từ bất kỳ polygon ACD2D nào đến obstacle gần nhất.

3. Adjacency detection trên buffered vs original regions

Design doc quy định adjacency phải dùng vùng gốc (trước buffer) để tránh tạo cạnh giả qua obstacle. Nếu regions_intersect trong graph_builder.py được gọi trên buffered regions, hai vùng không thực sự lân cận (nhưng buffer của chúng chồng lên nhau qua obstacle) có thể bị nối thành cạnh trong graph — tạo path đi qua obstacle.
