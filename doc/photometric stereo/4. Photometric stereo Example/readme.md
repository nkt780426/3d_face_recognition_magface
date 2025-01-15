# Nguồn:
https://www.youtube.com/watch?v=pW_AhEh6VpI
# Mục đích
Hướng dẫn cách tạo 3D Objects bằng phương pháp photometric stereo.
Điều kiện:
- Phải có nhiều image được chụp từ nhiều nguồn sáng khác nhau, các nguồn sáng phải độc lập tuyến tính
- Có ít nhất 3 image và phải đảm bảo mọi điểm trên surface đều được chiếu sáng trong 3 ảnh. Nếu không phải thực hiện chụp thêm nhiều image.
- Các pixel trên các ảnh phải hoàn toàn giống nhau: Thực tế trong quá trình thu thập dataset của 1 vật thể có khả năng di chuyển như con vật/mặt người, thường các pixel của mặt người sẽ bị lệch 1 vài pixel so với ảnh gốc và điều này ảnh hưởng cực mạnh đến quá trình tính toán normal map của phương pháp. Do đó cần phải đảm bảo mỗi điểm trên pixel của vật thể là giống nhau với mọi image.

Ở bài thực hành này các ảnh đã được tiền xử lý thành công và ở dạng pmp format, các nguồn sáng sẽ là các file .txu sau đó họ sẽ tạo 3D mesh file ở dạng .ply

# Nhắc lại
Photometric stereo là phương pháp để estimate (ước tính) các normal (pháp tuyến) của từng điểm trên vật thể từ nhiều ảnh chụp ở nhiều điều kiện chiếu sáng khác nhau. Bằng cách phân tích độ sáng của mỗi pixels chain trên image, chúng ta có thể suy ra normal của vật thể ứng với pixel đó từ đó reconstruct 1 3D shape.
