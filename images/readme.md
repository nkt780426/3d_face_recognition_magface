albedo:
    - hoang: 0.9580, 0.9575
    - lâm: 0.9485 0.9502
normalmap:
    - hoàng: 0.9386, 0.9394
    - lâm: 0.8976, 0.9177
deopthmap:
    - hoàng: 0.8733, 0.8825
concat:
    - hoàng: 0.9737, 0.9747
    - lâm: 0.8999, 0.9271

concat-unfreeze:
    - epoch cuối:
        - cosine: 0,9493 (210)
        - euclidean: 0,9607 (357)
    - best:
        - cosine: 0,9659 (210)
        - euclidean: 0,9686 (327)
    - Giả định: mỗi mạng con được tối ưu để học 1 embedding chứa thông tin xác định identity của khuôn mặt với từng chiều khuôn mặt. Việc unfreeze lại làm cho trọng số mạng này bị thay đổi theo hướng.
        - Tích cực (mạng concat cải thiện): Việc unfreeze sẽ giúp,
            - Các mạng backbone bổ sung thông tin và khả năng nhận diện thông tin khuôn mặt tốt hơn, qua đó gián tiếp làm backbone cải thiện hơn, embedding chất lượng hơn.
            - Mạng concat tìm ra tổ hợp embedding tối ưu giữa 3 mạng.
        - Tiêu cực (mạng concat tệ hơn): Việc unfreeze làm cho
            - Backbone đang được tối ưu để học embedding sẽ bị gây rối bởi thông tin từ các mạng backbone khác gây suy giảm chất lượng embedding mà backbone này tạo ra. Các backbone ảnh hưởng qua lại lẫn nhau làm cho embedding sinh ra của toàn mạng backbone kém đi => concat tệ đi
        - Thục nghiệm chứng minh concat khi unfreeze tệ hơn => Phải freeze lại.


iResnet:
1. Giới thiệu
- Resnet vẫn gặp phải 'gradient vanishment' khi tăng layer mạn từ 152 lên 200 trên imagenet cho kết quả tệ hơn đáng kể. Điều này cho thấy resnet vẫn làm suy giảm thông tin loss trong quá trình probation.
- Khi dimention của building block không khớp với building block kế tiếp, 1 project shorcut phải được sử dụng. Project shortcut này không tối ưu cho vấn đề degradation problem.
    Tuy nhiên đường này lại quan trọng, vì nó nằm trên đường main information của mạng.
- Trong resnet gốc bottleneck building block được giới thiệu để kiểm soát số lượng tham số và chi phí tính toán khi độ sâu tăng đáng kể.
    Tuy nhiên trong bottleneck block này, lớp Conv phụ trách học spatial information nhận được ít số lượng input/output channels.

- iresnet đề xuất kiến trúc mới mà độ phức tạp tương đương.
    - Mạng mới dựa trên resnet nguyên bản (mạng sâu hơn bằng cách xếp chồng các block, phân tách mạng thành các giái đoạn và áp dụng các khối block khác nhau tại mỗi giai đoạn). Hướng tiếp cận của mạng này nhằm dữ 
    - project shorcut mới với không tham số cần train giúp cải thiện.
    - building block mới tập trung vào spatial convolution. Nó gồm nheieuf spatial channels hơn bottleneck building block.
    - Đề xuất này không tăng độ phức tạp của mạng và họ đã đạt được kết quả đáng kế.
=> Kết quả học được mạng cực kỳ sâu mà không gặp phải khó khăn optimization khi mạng tăng lên.
