# Đóng góp đồ án

1. Xây dựng bộ nhận diện khuôn mặt sử dụng dữ liệu 2D và 3D tăng cường từ phương pháp Photometric Stereo.
2. Sử dụng Multi task để học tách biệt feature map của ảnh sao cho không liên quan đến kính, râu, pose, emotion, giới tính. Từ đó thu được feature map tốt hơn cho bài toán nhận diện danh tính.
    - Kết quả pose và emotion không được tốt do cách đánh label không thể hiện được tính chất của lớp đó.
    - Ví dụ: Để phân loại Pose tốt, cần đánh lable theo độ thay vì 0 (nhìn chính diện) và 1 (chứa các pose khác và lớp này sẽ rất khó học do các mẫu trong lớp này có các pose khác nhau). Tương tự, học task emotion cũng ko tốt.
    - Lược bỏ dataset để ko học 2 task này. Mô hình sẽ có hiệu suất tốt hơn nhưng không có khả năng học độc lập về Pose và emotion
3. Sử dụng focal loss để cân bằng dữ liệu trong mỗi task.

# Cấu trúc Project

```plaintext
3d_face_recognition_magface/
├── checkpoint/                     # các experments (jupyter) và tensorboard logs + models
│   ├── concat2/                    # experment concat đôi một normal map, depthmap, albedo
│       ├── logs/
│       ├── models/
│       ├── experments.ipynb
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
│   └── multi/                      # experment test dữ liệu (gallery+probe) với bộ nhận diện đơn và concat
│       └── gallery_db.csv          # vector database chứa dữ liệu gallery set
│       └── gallery_remaining.csv   # metadata probe set
│       └── gallery.csv             # metadata gallery set
│       └── multi_model.ipynb       # expertment test dữ liệu với mạng concat
│       └── single_model.ipynb      # expertment test dữ liệu với mạng đơn
│   └── triplet/                    # experment test dữ liệu với bộ nhận diện triplet loss ở project khác.
│   └── test.ipynb                  # experment đọc tensorboard log
├── .gitignore
└── README.md
```

Dataset download tại: https://www.kaggle.com/datasets/blueeyewhitedaragon/hoangvn-3dmultitask (sử dụng version 1, không dùng version 2)

# Cách chạy project

**Đưa các file jupyter (experment) muốn chạy vào thư mục root của project này và chạy bình thường**
- Code có thể có 1 chút bug khi chạy, do trong quá trình làm đồ án mình đã sửa đổi rât nhiều để phù hợp với tính huống gần nhất. (chủ yếu nằm ở phần dataloader và trong jupyter, còn lại code bình thường)
- Nếu muốn tính thêm chỉ số accuracy, chỉnh lại phần comment ở file utils/roc_auc.py (nên làm với model thu được sau cùng chứ ko nên làm trong quá trình train)
- Chú ý cần đọc kỹ cẩn thận lại các đường dẫn lưu log và models.
- File requirements.txt ko hoàn chỉnh
- Muốn code nhanh, hay chạy trên máy cá nhân trước (wsl hoặc ubuntu) rồi mới chạy trên kaggle.

# Hướng cải thiện

1. Kiến trúc mạng
- Trong project này ở mỗi head mình sử dụng fully connected 512 neutron để lưu trữ thông tin các embedding với mỗi task. Điều này chỉ phù hợp với task cuối (nhận diện danh tính), còn các task còn lại như nhận diện kính, râu, giới tính, ... rất lãng phí. Có thể xem xét giảm kích thước embedding cho các task này cho phù hợp.
- Với mạng concat 2 hoặc 3 loại dữ liệu, train sẽ rất lâu do mình thêm bộ nhận diện fully connected 512 sau khi concat các embedding của các backbone. Ví dụ concat 3 embedđing được 1536 -> đi vào fully connected 512. Train đoạn này rất lâu. Có thể thay thế lớp fully connected layer đó bằng Conv 1x1 hoặc self-attention, hoặc gì đó để giảm chi phí tính toán các mạng concat giúp train nhanh hơn và có thể tốt hơn.

2. Bỏ task emotion + pose.
- Dựa vào file metadata, lọc lại dataset bằng folder preprocess, chỉ giữ lại ảnh có thuộc tính emotion là nhìn trực diện và pose nhìn thẳng (mất khoảng 1k ảnh)
- Tạo dataset mới dựa trên dataset đã lọc
- *Mình đã thử và kết quả cho rất tốt (là cái dataset kaggle version 2). Các mạng đơn đều cho 97 -> 99%, các mạng concat thì chưa thử*
- Lý do học 2 task này ko tốt: label gán nhãn ko đủ tốt. Ví dụ có rất nhiều emotion như cười, nói, ... đều được chia thành nhìn trục diện và không nhìn trực diện (cái ko nhìn này học ko tốt)

3. Train lại với dữ liệu 2D gốc (4 ảnh .bmp) để làm căn cứ so sánh với bộ nhận diện sử dụng dữ liệu Photometric Stereo.

4. Triển khai bộ nhận diện trên thiết bị thật.

5. Tái tạo dataset mới từ mạng embedding đã học được. Chi tiết đọc research MTL Face.
