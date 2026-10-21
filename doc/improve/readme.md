# Kiến trúc bài báo

- Thay đổi cách tiếp cận khoa học: nhấn mạnh vào  **"Đóng góp về mặt Cơ chế" (Mechanism/Theoretical Contribution)**:
- Đóng góp

  * **Khai thác sâu vào Normal Map:** Đừng coi Normal Map chỉ là một tập dữ liệu đầu vào. Hãy chứng minh rằng:  *Normal Map đóng vai trò như một bộ lọc không gian (Geometric Prior) giúp triệt tiêu nhiễu kết cấu (Texture Bias) của ảnh RGB* **.**
  * **Lý giải toán học cho MTL trong FAS:** Bạn cần chứng minh việc học song song các tác vụ (Pose, Kính, Râu) giúp mô hình đạt được trạng thái **Disentangled Representation Learning** (Học phân rã đặc trưng). Cụ thể: Mạng Neural tách biệt được đâu là "Đặc trưng định danh/ngoại quan" (Identity/Appearance - kính, râu, biểu cảm) và đâu là "Đặc trưng trình diện giả mạo" (Presentation Attack cues).
    Bổ sung "Trực quan hóa" (Visualization) - Vũ khí tối thượng hạ gục Reviewer Q1
    Các tạp chí Q1 yêu cầu bạn phải giải thích được bên trong "hộp đen" AI đang làm gì. Bạn bắt buộc phải bổ sung:
    - t-SNE Plot (Không gian đặc trưng): Vẽ biểu đồ t-SNE của vector đặc trưng (Embedding) từ MagFace. Chứng minh rằng khi có Normal Map và MTL, các mẫu Live và Spoof tách rời nhau hoàn toàn thành 2 cụm rõ rệt, ngay cả khi người dùng đeo kính hoặc đổi góc mặt (Pose lớn).
    - Grad-CAM (Bản đồ nhiệt kích hoạt): Xem mô hình tập trung vào đâu. Chứng minh rằng mô hình của bạn không nhìn vào cái kính hay bộ râu (nhờ nhánh MTL đã "gánh" các tác vụ đó), mà nó tập trung nhìn vào viền màn hình, độ lóa, hoặc vết gập của giấy (Spoof cues).
- Bổ sung thực nghiệm
- * **CelebA-Spoof (Khuyến nghị số 1):** Đây là tập dữ liệu hoàn hảo nhất cho bạn. Nó không chỉ là bộ dữ liệu FAS siêu lớn (600.000 ảnh) mà còn  **có sẵn nhãn (labels) về: Góc mặt (Pose), Vật che khuất (Occlusion), Kính (Glasses), và Điều kiện ánh sáng (Illumination)** **. Bạn chỉ cần dùng thêm một tool nhận diện thuộc tính khuôn mặt nhỏ để gán thêm nhãn Râu (Beard) và Biểu cảm (Emotion) là đủ bộ cho nhánh MTL.**
  * **OULU-NPU hoặc CASIA-FASD:** Nếu máy tính của bạn không đủ mạnh để train tập CelebA-Spoof khổng lồ, hãy chọn OULU-NPU hoặc CASIA-FASD. Chúng nhỏ gọn hơn, chuẩn mực và bắt buộc phải có nếu bạn muốn chứng minh tính hiệu quả của mô hình với giới nghiên cứu.
