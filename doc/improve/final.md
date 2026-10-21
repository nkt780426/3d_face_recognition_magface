1. Ý tưởng và tính mới (Novelty & Originality)

   1. Xác định khoảng trống trong nghiên cứu: *Nghiên cứu trước đây chưa giải quyết được gì, và bài viết này đóng góp điều gì mới?*
   2. Đóng góp rõ ràng (Contributions)
      1. Sử dụng thêm thông tin 3D (normalmap + albedo): để cải thiện nhiễu do ánh sáng và cung cấp thêm thông tin không gian 3 chiều
      2. Cải thiện thêm công thức loss MTL face (suy nghĩ thêm).
2. Dataset train

   1. PhotoFace: 453 người.
   2. FaceVerse3D_Render: 120 người, mỗi người có 20 ảnh.
   3. dataset: 50000 id
   4. VGface2:

      1. Nhiễu
      2. Chưa có ảnh normal map + deepmap -> Sử dụng mô hình gen ra.
   5. CelebA-Spoof: Đây là tập dữ liệu hoàn hảo nhất cho bạn. Nó không chỉ là bộ dữ liệu FAS siêu lớn (600.000 ảnh) mà còn  có sẵn nhãn (labels) về: Góc mặt (Pose), Vật che khuất (Occlusion), Kính (Glasses), và Điều kiện ánh sáng (Illumination) .

      1. Ứng dụng trong FAS là chính.
   6. OULU-NPU hoặc CASIA-FASD
3. Bổ sung đánh giá

   1. t-SNE Plot (Không gian đặc trưng): Vẽ biểu đồ t-SNE của vector đặc trưng (Embedding) từ MagFace. Chứng minh rằng khi có Normal Map và MTL, các mẫu Live và Spoof tách rời nhau hoàn toàn thành 2 cụm rõ rệt, ngay cả khi người dùng đeo kính hoặc đổi góc mặt (Pose lớn).
   2. Grad-CAM (Bản đồ nhiệt kích hoạt): Xem mô hình tập trung vào đâu. Chứng minh rằng mô hình của bạn không nhìn vào cái kính hay bộ râu (nhờ nhánh MTL đã "gánh" các tác vụ đó), mà nó tập trung nhìn vào viền màn hình, độ lóa, hoặc vết gập của giấy (Spoof cues).
   3. ... Dựa vào research thêm đánh giá ...
4. Ý tưởng

   1. Đầu vào ảnh 2D -> Mô hình gen ra ảnh 3D + ánh sáng -> train theo MTL face.
   2. Servey 2D ->3D để cải thiện độ chính xác.
