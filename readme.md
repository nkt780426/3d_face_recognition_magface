1. Kiến thức
- Phương án 1: Triplet loss + iresnet 18:
    - Xử lý dữ liệu và chia dataset
        - Photometric Stereo
        - MTCNN
        - Under sample thế nào
    - Mạng và loss:
        - Mạng iresnet18 so với resnet truyền thống
        - Triplet loss.
    - Các kết luận rút ra được từ thực nghiệm:
    - Đọc lại quyển
    - Tham số đánh giá AUC và điểm yếu của nó với triplet loss và phương án Magface.

- Phương án 2: Multi task
    - Mạng iresnet 18: cải tiến bằng Prelu thay vì relu => Ý nghĩa
    - 
https://drive.google.com/drive/folders/1-1KWxxoID1SrKzs1BYvEj6whqBzz1LtW?usp=sharing

concat-albedo-depthmap (bản chất là normalmap-albedo): 3 - ngonluahoangkim
concat-albedo-depthmap: 1 - ádfsadfsdfs
concat-normalmap-depthmap: 2 - Sullyvan
concat-all: 5 (xong)

albedo: 6 - Sullyvan2002(fix focal loss)
normalmap: 5 - BlueEyeWhiteDragon (10p -8 epoch) => 2,5 tiếng
depthmap: 1 - ádfsadfsdfs (13p-12 epoch) => 

normalmap-albedo: 3 - ngonluahoangkim
normalmap-depthmap: 2 - Sullyvan
albedo-depthmap: 1- ádfsadfsdfs

all: 5 

# Khác concatall và concat-2
hàm tạo dataloader
Mạng V2 và V3