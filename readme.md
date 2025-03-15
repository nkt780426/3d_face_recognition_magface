# Vấn đề

Các bộ nhận diện khuôn mặt hiện nay chủ yếu được huấn luyện dựa trên mô hình và dữ liệu 2D. Tuy nhiên các bộ nhận diện khuôn mặt sử dụng dữ liệu 2D thường gặp phải việc suy giảm hiệu suất nghiệm trọng khi do ảnh hưởng của điều kiện chiếu sáng (illumination), góc chụp (pose), các điều kiện che khuất (kính, khẩu trang, ...) và cảm xúc (emotion). Điều này là do

1. Điều kiện chiếu sáng: Bản chất ảnh 2D chỉ cung cấp cho máy tính thông tin về màu sắc của vật thể. Tuy nhiên thông tin này bị ảnh hưởng mạnh bởi điều kiện ánh sáng như thông lượng ánh sáng, khoảng cách từ người đến camera, độ bao phủ của nguồn sáng lên vật thể, ... Do đó cùng 1 người trong các điều kiện ánh sáng khác nhau sẽ cho ra ảnh 2D khác nhau. Điều này ảnh hưởng đến quá trình học của máy tính.
2. Góc chụp và cảm xúc: Ảnh 2D không cung câp thông tin chiều sâu của vật thể như ảnh 3D. Do đó cùng 1 người chụp dưới các điều kiện góc chụp và cảm xúc khác nhau sẽ cho các hình dạng khác nhau gây khó khăn trong quá trình học của máy tính.
3. Các điều kiện che khuất: Khi huấn luyện bộ nhận diện, embedding được trích xuất ra từ ảnh chưa thực sự tối ưu cho tác vụ nhận diện khuôn mặt. Nó có thể chứa các yếu tố như môi trường, các vật thể che khuất như kính, khẩu trang, ... Đặc biệt là khi các yếu tố này quá lớn có thể che khuất đi các đặc điểm đặc trưng để nhận diện khuôn mặt, tệ hơn bộ nhận diện có thể nhận diện nhầm các yếu tố che khuất của ảnh là đặc điểm nhận dạng của khuôn mặt.

Trong khi đó các bộ nhận diện 3D thường cho hiệu suất nhận diện tốt hơn do dữ liệu 3D không bị ảnh hưởng bởi điều kiện chiếu sáng, cung cấp thông tin 3 chiều nên không bị ảnh hưởng bởi góc chụp và cảm xúc. Ngoài ra các bộ nhận diện 3D có khả năng chống giả mạo tốt hơn. Tuy nhiên việc huấn luyện dữ liệu 3D thường rất đắt đỏ do yêu cầu dữ liệu 3D, tài nguyên tính toán lớn và tốc độ nhận diện chậm hơn so với các bộ nhận diện 2D

# Đóng góp đồ án

1. Xây dựng **bộ nhận diện khuôn mặt 3D** bằng việc sử dụng **dữ liệu 2D mang thông tin 3D từ phương pháp Photometric Stereo**.
2. Sử dụng các công nghệ tiên tiến như **MagFace loss** để cải thiện hiệu suất của bộ nhận diện, **MTLFace (một mô hình multitask) để học tách biệt feature map** của ảnh sao cho thu được feature map không liên quan đến kính, râu, pose, emotion, giới tính. Từ đó thu được embedding tối ưu hơn phục vụ bài toán nhận diện khuôn mặt.
3. Sử dụng **focal loss** giải quyết vấn đề dataset mất cần bằng trong mỗi tác vụ

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
