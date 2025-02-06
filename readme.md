# Đóng góp đồ án

1. Xây dựng bộ nhận diện khuôn mặt 3D sử dụng dữ liệu 2D mang thông tin 3D từ phương pháp Photometric Stereo.
2. Sử dụng Multi task để học tách biệt feature map của ảnh sao cho không liên quan đến kính, râu, pose, emotion, giới tính. Từ đó thu được feature map tốt hơn cho bài toán nhận diện danh tính.
    - Kết quả 2 task pose và emotion không được tốt do cách đánh label không thể hiện được tính chất của lớp đó (train kết quả cao nhưng khi test vẫn sai). Lý do: 
        - Với task Pose, mình đánh lable là 0 (nhìn trực diện) và 1 (nghiêng 1 chút, nghiêng 30-45 độ, nghiêng 90 độ). Điều này khiến task có label 1 học không được tốt, do các ảnh trong label này khác biệt rất nhiều về tính chất. Tương tự với task emotion. 
3. Sử dụng focal loss để cân bằng dữ liệu trong mỗi task.

# Cấu trúc Project

```plaintext
3d_face_recognition_magface/
├── checkpoint/                     # các experments (jupyter) và tensorboard logs + models
│   ├── concat2/                    # experment concat đôi một normal map, depthmap, albedo
│   ├── ├── logs/
│   ├── ├── models/
│   ├── ├── experments.ipynb
│   ├── concat3/                    # experment concat cả 3 loại dữ liệu
│   └── multi/                      # experment chỉ có 1 loại dữ liệu
├── doc/                            # Slide + Kiến thức
├── Dataset/                        # Dataset sau tiền xử lý
│   ├── Albedo/                     
│   ├── Depth_Map/
│   ├── Normal_Map/
│   └── train_set.csv               # metadata train set
│   └── test_set.csv                # metadata validate set
│   └── gallery_set.csv             # metadata test set (gallery set + probe set)
├── going_modular/                  # package multi task + magface để viết các experments đơn giản hơn
│   └── dataloader/                 # dataloader từng loại dữ liệu và data prefetch
│   └── loss/                       # cách tính multi task toàn mạng (focal loss + magface)
│   └── model/                      # kiến trúc mạng multi task
│   └── train_eval/                 # train loop + eval loop
│   └── utils/                      # các hàm phụ phục vụ huấn luyện như tính auc, acc, model checkpoint, early stopping, ...
├── preprocess/                     # tiền xử lý và phân tích dữ liệu từ dataset gốc (không quan tâm nếu đã có thư mục Dataset)
├── test_models/
│   ├── multi/                      # experment test dữ liệu (gallery+probe) với bộ nhận diện đơn và concat
│   ├── └── gallery_db.csv          # vector database chứa dữ liệu gallery set
│   ├── └── gallery_remaining.csv   # metadata probe set
│   ├── └── gallery.csv             # metadata gallery set
│   ├── └── multi_model.ipynb       # expertment test dữ liệu với mạng concat
│   ├── └── single_model.ipynb      # expertment test dữ liệu với mạng đơn
│   └── triplet/                    # experment test dữ liệu với bộ nhận diện triplet loss ở project khác.
│   └── test.ipynb                  # experment đọc tensorboard log
├── .gitignore
└── README.md
```

Dataset download tại [đây](https://www.kaggle.com/datasets/blueeyewhitedaragon/hoangvn-3dmultitask/versions/1) (sử dụng version 1, không dùng version 2)

# Cách chạy project

**Đưa các file jupyter (experment) muốn chạy vào thư mục root của project này và chạy bình thường**
- Code có thể có 1 chút bug khi chạy, do trong quá trình làm đồ án mình đã sửa đổi rât nhiều để phù hợp với tính huống gần nhất. (chủ yếu nằm ở phần dataloader và trong jupyter, còn lại code bình thường)
- Nếu muốn tính thêm chỉ số accuracy, chỉnh lại phần comment ở file utils/roc_auc.py (nên làm với model thu được sau cùng chứ ko nên làm trong quá trình train)
- Chú ý cần đọc kỹ cẩn thận lại các đường dẫn lưu log và models.
- File requirements.txt ko hoàn chỉnh
- Muốn code nhanh, hay chạy trên máy cá nhân trước (wsl hoặc ubuntu) rồi mới chạy trên kaggle.

# Hướng cải thiện

1. Kiến trúc mạng
- Với mạng concat 2 hoặc 3 loại dữ liệu, train sẽ rất lâu do mình thêm bộ nhận diện fully connected 512 sau khi concat các embedding của các backbone. Ví dụ concat 3 embedđing được 1536 -> đi vào fully connected 512. Train đoạn này rất lâu. Có thể thay thế lớp fully connected layer đó bằng Conv 1x1 hoặc self-attention, hoặc gì đó để giảm chi phí tính toán các mạng concat giúp train nhanh hơn và có thể tốt hơn.
- Học tách biệt từng task: thay vì train cùng 1 lúc nhiều task như kiểu mình. Hãy train 1 task, 1 thời điểm, khi nào task đó tốt rồi thì freeze nhánh chứa task đó lại rồi tiếp tục học các task khác.

2. Task emotion + pose.
- Trong dataset gốc, 2 task này không được đánh nhị phân mà có khoảng 4-5 label. Có thể xem xét dữ nguyên các label này thay vì gộp lại thành task nhị phân như của mình. *Lý do mình gộp 2 task này lại thành task nhị phân là do mình không tinh chỉnh được tham số alpha, gamma của focal loss với task có nhiều label, loss càng học càng tăng.*
- *Mình đã thử lọc lại dataset sao cho chỉ có pose trực diện và emotion nhìn thẳng, qua đó bỏ được 2 task này ra khỏi mạng multi task từ đó cho kết quả tốt hơn (là cái dataset kaggle version 2, mất 1000k phiên chụp so với dataset gốc). Các mạng đơn đều cho 97 -> 99% khi test, các mạng concat thì chưa thử nhưng chắc tốt hơn. Nhược điểm khi bỏ 2 task là bộ nhận diện không thể nhận diện được khuôn mặt có pose và emotion thay đổi.*

3. Train lại với dữ liệu 2D gốc (4 ảnh .bmp) để làm căn cứ so sánh với bộ nhận diện sử dụng dữ liệu Photometric Stereo.

4. Triển khai bộ nhận diện trên thiết bị thật.

5. Tái tạo dataset mới từ embedding đã học được. Chi tiết đọc research MTL Face.
