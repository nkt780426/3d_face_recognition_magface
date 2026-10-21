# Hướng tiếp cận

- Đóng góp:
  - Đóng góp về mặt cơ chế:

    - Chứng minh Normal map là 1 bộ lọc không gian  *(Geometric Prior) giúp triệt tiêu nhiễu kết cấu (Texture Bias) của ảnh RGB. Train bằng normal map giúp cải thiện[`]()*

    * Chứng minh việc học multi-task giúp học phân rã các đặc trưng.
      * Đâu là đặc trưng ngoại quan: kính, râu, biểu cảm
      * Đặc trưng giả mạo:
      * Trực quan hóa bằng
        * biểu đồ t-SNE Plot (Không gian đặc trưng)
        * Grad-CAM (Bản đồ nhiệt kích hoạt): Chứng minh rằng mô hình của bạn không nhìn vào cái kính hay bộ râu (nhờ nhánh MTL đã "gánh" các tác vụ đó), mà nó tập trung nhìn vào viền màn hình, độ lóa, hoặc vết gập của giấy
- Thực nghiệm
  - Trên dataset lớn quốc tế

    - **Protocol 4 của OULU-NPU:** Đây là bài test khắc nghiệt nhất (kiểm tra khả năng tổng quát hóa khi đổi camera và đổi môi trường).
    - **Cross-Dataset Testing:** Train trên tập *CASIA-FASD* nhưng mang sang Test trên tập *Replay-Attack* (hoặc ngược lại). Nếu mô hình của bạn đạt chỉ số **HTER (Half Total Error Rate)**hoặc **ACER** thấp vượt trội so với các nghiên cứu SOTA (State-of-the-art) năm 2024-2026
  - Xây dựng bảng Ablation Study. Chứng minh các chỉ số

    - Chỉ dùng RGB + Softmax (Baseline)

    * *Dùng Normal Map + Softmax*
    * Dùng Normal Map + MTL (Khử nhiễu râu, kính, pose)
    * Mô hình hoàn chỉnh: Normal Map + MTL + MagFace (Tối ưu nhất)
