0. Lời mở đầu
Kính thưa thầy cô, em tên là ... sinh viên khoa kỹ thuật máy tính k65. Hôm nay, em xin trình bày báo cáo kết quả đồ án tốt nghiệp của em ... Bằng sự nỗ lực của bản thân và sự hướng dẫn chỉ hướng nhiệt tình của Tiến sĩ Ngô Thành Trung. Em rất vui được có mặt ở đây ngày hôm nay. Em rất mong nhận được góp ý, những ý kiến đóng góp tích cực trong suất quá trình báo cáo để kết quả báo cáo ngày hôm nay trở nên tốt đẹp và hoàn thiện hơn.

Sau đây em xin phép bắt đầy trình bày. Nội dung của buổi báo cáo có 6 phần....

1. Giới thiệu đề tài
- Đầu tiên là lý do em chọn đề tài này:
    +) Em hiện là sinh viên năm cuối, đồ án tốt nghiệp là môn học cuối cùng của em trong ngôi trường này. Em rất ấn tượng với cách giảng dạy của Đại học Bách Khoa Hà Nội và lượng kiến thức mà em tiếp thu được trong thời gian em theo học ở đây. Một lối học chú trọng vào thực hành và đề cao tính tự học, tìm tòi và khám phá của mỗi người.
    +) Vì vậy, em mong muốn tận dụng tối đa thời gian còn lại ở đây để tiếp tục học được nhiều điều hơn nữa. Tự thách thức bản thân trong 1 lĩnh vực hoàn toán mới không phải sở trường của em. Dưới sự dẫn dắt của thầy Ngô Thành Trung, em được tiếp cận đến lĩnh vực nhận diện khuôn mặt, 1 trong những lĩnh vực nổi bật của computer vision và ai nói chung. Em bị thu hút bởi lịch sử hình thành, những khó khăn thách thức mà vấn đề này gặp phải và cách các công nghệ được sinh ra để giảm thiểu, khắc phục những thách thực này.
    +) Đến nay trong thực tế, hầu hết các công ng

# 2. Các giải pháp
## 2.1. Các giải pháp, hướng tiêp cận truyền thống



Kể từ khi việc học feature vector và phân loại class được tách rời. Các thuật toán dựa trên Support vector machine ra đời và sử dụng nó cho việc phân loại. Mục tiêu là tìm 1 siêu phẳng - hyperplane để phân loại các feature vector.


Tổng kết:
Không xử lý được biến đổi phong phú: Các phương pháp LBP truyền thống không có khả năng xử lý các biến đổi phức tạp của hình ảnh như xoay, phóng to, thu nhỏ, hoặc thay đổi góc nhìn, điều này làm hạn chế khả năng áp dụng trong các tình huống thực tế.

Khó khăn trong việc xác định kích thước và hình dáng vùng: LBP chia hình ảnh thành các vùng nhỏ để phân tích, nhưng việc xác định kích thước vùng và cách chia nhỏ hình ảnh một cách hợp lý là không dễ dàng, đặc biệt là khi các đối tượng trong hình ảnh có độ phân giải và kích thước khác nhau.

**các phương pháp truyền thống như SVM và LBP có thể hiệu quả trong nhiều trường hợp, nhưng cũng có các hạn chế khi đối mặt với dữ liệu phức tạp, nhiều nhiễu hoặc yêu cầu tính toán tài nguyên lớn. Các phương pháp học sâu (deep learning) hiện đại đã giúp khắc phục nhiều vấn đề này, nhưng vẫn cần cân nhắc khi sử dụng trong các bài toán cụ thể.**

## 2.2. Các giải pháp hiện Đại
Với sự phát triển ngày càng mạng của tài nguyên tính toán, mainstream của Face Recognition system đã sử dụng deep learning làm backbone để giải quyết vấn đề này. Như mọi người có thể thấy trên hình


L-softmax: là hàm loss đầu tiên dựa trên margin được đo bằng góc giữa các feature vector.  Đầu tiên, phương pháp này loại bỏ độ chệch (bias) của mỗi lớp 𝑏𝑗 và thay đổi tích vô hướng giữa đặc trưng 𝑥𝑖 và trọng số 
Wj thành dạng mới 
∥
𝑊
𝑗
∥
∥
𝑥
𝑖
∥
cos
⁡
(
𝜃
𝑗
)
∥W 
j
​
 ∥∥x 
i
​
 ∥cos(θ 
j
​
 ), trong đó 
𝜃
𝑗
θ 
j
​
  là góc giữa 
𝑥
𝑖
x 
i
​
  và trọng số 
𝑊
𝑗
W 
j
​
  của lớp 
𝑗
j.
A-softmax: normalize each class weight Wj trước khi tính toán loss

# Câu hỏi
1. Tại sao lại dùng phương pháp photometric stereo ?
