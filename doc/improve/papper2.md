- Nhận diện khuôn mặt an toàn thông qua Ước lượng Bản chất Vật lý" (Secure Face Recognition via Physical Essence Estimation)

  - **Lập luận cốt lõi:** Một hệ thống Nhận diện khuôn mặt (FR) xuất sắc không chỉ phân biệt giữa Người A và Người B (Identity), mà trước hết phải phân biệt được giữa **Thực thể 3D thực tế (Physical 3D Entity)** và  **Bản sao 2D giả mạo (2D Spoof Replica)** .
  - **Sự nhỉnh hơn của MagFace:** Bạn hãy nhấn mạnh MagFace vốn là SOTA của Nhận diện khuôn mặt (Face Recognition). Việc bạn dùng MagFace đồng nghĩa với việc hệ thống này  **vừa trích xuất đặc trưng nhận diện định danh cực tốt, vừa tự động gán luôn bộ lọc chất lượng chống giả mạo thông qua Magnitude** .
- Ý nghĩa

  - Chuyển đổi RGB -> Normalmap: triệt tiêu nhiễu về màu sắc, ánh sáng. Bẻ gãy cấu trúc phẳng của ảnh giả mạo nhận diện.
  - Chuyển đổi RGB -> Albedo: Cung cấp thêm thông tin về màu sắc
- Thực nghiệm

  - Chỉ dùng RGB + Softmax (Baseline)
  - *Dùng Normal Map + Softmax*
  - Dùng Normal Map + MTL (Khử nhiễu râu, kính, pose)
  - Dùng Normal Map + Albedo + MTL (Thêm thông tin màu sắc)
  - Mô hình hoàn chỉnh: Normal Map + MTL + MagFace.
- Chứng minh học MTL face giúp phân rã đặc trưng embedding bằng

  - t-SNE Plot (Không gian đặc trưng): Vẽ biểu đồ t-SNE của vector đặc trưng (Embedding) từ MagFace. Chứng minh rằng khi có Normal Map và MTL, các mẫu Live và Spoof tách rời nhau hoàn toàn thành 2 cụm rõ rệt, ngay cả khi người dùng đeo kính hoặc đổi góc mặt (Pose lớn).
  - Grad-CAM (Bản đồ nhiệt kích hoạt): Xem mô hình tập trung vào đâu. Chứng minh rằng mô hình của bạn không nhìn vào cái kính hay bộ râu (nhờ nhánh MTL đã "gánh" các tác vụ đó), mà nó tập trung nhìn vào viền màn hình, độ lóa, hoặc vết gập của giấy (Spoof cues).
- Khả năng chống giả mạo của magface ?

  - Ảnh giả mạo thường có chất lượng cực kém sau khi đi qua mô hình normal map và albedo -> Magnitude nhỏ
  - Ảnh thật -> Magnitude lớn.
- Dataset train

  - **CelebA-Spoof (Khuyến nghị số 1):** Đây là tập dữ liệu hoàn hảo nhất cho bạn. Nó không chỉ là bộ dữ liệu FAS siêu lớn (600.000 ảnh) mà còn  **có sẵn nhãn (labels) về: Góc mặt (Pose), Vật che khuất (Occlusion), Kính (Glasses), và Điều kiện ánh sáng (Illumination)** **. Bạn chỉ cần dùng thêm một tool nhận diện thuộc tính khuôn mặt nhỏ để gán thêm nhãn Râu (Beard) và Biểu cảm (Emotion) là đủ bộ cho nhánh MTL.**
  - **OULU-NPU hoặc CASIA-FASD:** Nếu máy tính của bạn không đủ mạnh để train tập CelebA-Spoof khổng lồ, hãy chọn OULU-NPU hoặc CASIA-FASD. Chúng nhỏ gọn hơn, chuẩn mực và bắt buộc phải có nếu bạn muốn chứng minh tính hiệu quả của mô hình với giới nghiên cứu.
- Viết công thức về loss
