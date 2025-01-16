# Kiến thức

1. Kiến thức
- Phương án 1: Triplet loss + iresnet 18:
    - Xử lý dữ liệu và chia dataset
        - Photometric Stereo: điều kiện tạo ảnh, 
        - MTCNN
        - Under sample thế nào
    - Mạng và loss:
        - Mạng iresnet18 so với resnet truyền thống
        - Triplet loss.
    - Các kết luận rút ra được từ thực nghiệm:
    - Đọc lại quyển
    - Tham số đánh giá AUC và điểm yếu của nó với triplet loss và phương án Magface.

- Phương án 2: Multi task
    - Mạng iresnet 18: cải tiến bằng Prelu thay vì relu => Ý nghĩa học số âm tốt hơn không bị ignore như relu
    - 

# Slide
1. Giới thiệu đề tài: 
    - Ứng dụng thực tiễn của nhận diện khuôn mặt, nó quan trọng nhiều nghiên cứu
    - Các nhược điểm của hệ thống FR ảnh 2D như ánh sáng, pose, emotion, ... 3D là gì

2. Các hướng tiếp cận
    - 4 hướng, đề cập đến phương pháp của mình

3. Đề xuất
    - Gồm 3 bước tiền xử lý, huấn luyện và chạy thực nghiệm
        - Để loại bỏ ảnh hưởng của ánh sáng => Photometric stereo => Cần ít nhất 3 ảnh vs 3 nguồn sáng khác nhau => Tính ra normalmap kèm theo các ảnh dephmap và albedo
        - Để giảm thiểu tác động của ngoại cảnh => MTCNN để crop mặt
    - Phân tích sự phân bổ của dataset => Để việc tối ưu embedding bằng multi task với 5 task để chia embedding thành 6 thành phần
        - Sử dụng chiến lược ... để chia cân bằng theo tỷ lệ các task.
        - Xử lý việc mất cân bằng dữ liệu các task bằng focal loss.
        - Đề xuất sử dụng MagFace loss, adaptive angular margin tiên tiến (dẫn chứng research nó tốt hơn arcface - 1 tiêu chuẩn của hệ thống nhận diện khuôn mặt hiện nay) để học embedding về id. Kết hợp với weight class để giảm thiểu mất cân bằng dữ liệu.
    - Mạng backbone: iresnet (ưu điểm so với resnet truyền thống) và thay PRelu
    - Đề xuất kiến trúc mạng MTL Face học tách biệt các task
        - Ví dụ tách biệt spectacles (kính)
        - Module attention (concat thế nào giữa 2 chiều C và spatial)
        - Sử dụng GRL học đối kháng
    - Giới thiệu các mạng concat 2 và 3 để cải thiện hiệu suất
    - Data Augmentation giảm thiểu overfit
        - Random Noise và RandomCrop, kiểm soát sự random giống nhau để mạng concat có hiệu suất tốt hơn.
    - Các tham số huấn luyện.

4. Kết quả thực nghiệm
    - Các kết quả auc multitask.
    - Code phần gallery để lấy được ảnh so sánh với gallery, chứng minh nó ko ảnh hưởng bởi kính và emotion.

# Research
- MagFace: https://arxiv.org/abs/2103.06627
- MTLface: https://arxiv.org/pdf/2103.01520
- Iresnet: https://arxiv.org/pdf/2004.04989


https://drive.google.com/drive/folders/1-1KWxxoID1SrKzs1BYvEj6whqBzz1LtW?usp=sharing

concat-albedo-depthmap (bản chất là normalmap-albedo): 3 - ngonluahoangkim
concat-albedo-depthmap: 1 - ádfsadfsdfs
concat-normalmap-depthmap: 2 - Sullyvan
concat-all: 5 (xong)

albedo: 6 - Sullyvan2002(fix focal loss) -
normalmap: 5 - BlueEyeWhiteDragon        -
depthmap: 1 - ádfsadfsdfs                -

normalmap-albedo: 3 - ngonluahoangkim
normalmap-depthmap: 2 - Sullyvan         -
albedo-depthmap: 1- ádfsadfsdfs          -

all: 5 