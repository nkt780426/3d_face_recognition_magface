1. Vấn đề: Tại sao iresnet18 cho kết quả tốt hơn các mạng khác dù tiêu chí của iresnet là mạng càng sâu sẽ cho accuracy càng tốt ?
    - Dataset rất nhỏ. Training set chỉ có 203 người, mỗi người có từ 3-10 ảnh. Testset có tổng cộng tất cả 237 người (khác với 203 người kia)
    - Không sử dụng mô hình pre-trained: Do các mô hình Face Recognition được train trên dataset lớn thường là ảnh RGB và phục vụ cho việc nhận dạng 2D. Trong khi dữ liệu đồ án của em là ảnh mang thông tin 3D như normal map, depth map và rất ít mô hình đã được pre-trained với tập dữ liệu dạng này.

2. Kết luận: 
- Backbone size ảnh hưởng rất lớn đến accuracy của hệ thống. Không phải lúc nào mạng càng sâu sẽ càng cho ra kết quả tốt. Kích thước của mạng phù hợp với dataset sẽ cho ra accuracy tốt nhất, đồng thời giảm bớt chi phí tính toán và thời gian train.
- Chọn iresnet18 là backbone cho hệ thống nhận diện khuôn mặt.