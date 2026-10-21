import torch
import torch.nn.functional as F

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha_weights=None, gamma_weights=None, num_classes=None):
        super(FocalLoss, self).__init__()

        self.num_classes = num_classes

        # Khởi tạo alpha tensor với num_classes, nếu không có trong alpha_weights, sử dụng giá trị mặc định là 1.0
        self.alpha = torch.tensor(
            [alpha_weights.get(i, 1.0) for i in range(self.num_classes)],
            dtype=torch.float32
        )
        
        # Khởi tạo gamma tensor với num_classes, nếu không có trong gamma_weights, sử dụng giá trị mặc định là 1.0
        self.gamma = torch.tensor(
            [gamma_weights.get(i, 1.0) for i in range(self.num_classes)],
            dtype=torch.float32
        )

    # def _compute_alpha(self, file_path, label_name):
    #     # Đọc dữ liệu từ file CSV
    #     self.data = pd.read_csv(file_path)
        
    #     # Đếm số lượng mẫu của từng nhãn
    #     label_counts = self.data[label_name].value_counts().sort_index()
        
    #     # Tổng số mẫu
    #     total_samples = label_counts.sum()
        
    #     # Tính tần suất ngược
    #     alpha = 1 - (label_counts.values / total_samples)
        
    #     # Tổng chuẩn hóa alpha (đảm bảo tổng alpha = 1)
    #     alpha_sum = alpha.sum()
    #     alpha = alpha / alpha_sum
        
    #     # Kiểm tra và điều chỉnh alpha theo điều kiện
    #     max_alpha = alpha.max()
    #     if max_alpha < 0.2:
    #         alpha *= 5
    #     elif max_alpha < 0.3:
    #         alpha *= 3
    #     elif max_alpha < 0.5:
    #         alpha *= 2
            
    #     return torch.tensor(alpha, dtype=torch.float32)


    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_true = y_true.long()
        y_pred_correct_class = y_pred.gather(1, y_true.unsqueeze(1)).squeeze(1)
        
        # Lấy alpha cho từng nhãn
        alpha = self.alpha.to(y_true.device)[y_true]  # Kích thước: (batch_size,)
        
        # Lấy gamma cho từng nhãn
        gamma = self.gamma.to(y_true.device)[y_true]  # Kích thước: (batch_size,)
        
        # Tính Focal Loss
        focal_weight = alpha * (1 - y_pred_correct_class) ** gamma
        log_prob = torch.log(y_pred_correct_class + 1e-8)  # Thêm epsilon để tránh log(0)
        focal_loss = -focal_weight * log_prob

        # Trả về trung bình của Focal Loss trong batch
        return focal_loss.mean()

    
if __name__ == '__main__':
    # batch_size = 4
    # num_classes = 3
    # y_pred = torch.tensor([[2.0, 1.0, 0.1],  # Logits cho mẫu 1
    #                     [0.5, 2.5, 0.3],  # Logits cho mẫu 2
    #                     [1.0, 0.2, 3.0],  # Logits cho mẫu 3
    #                     [0.3, 0.7, 1.5]]) # Logits cho mẫu 4
    # y_true = torch.tensor([0, 1, 2, 1])  # Nhãn thực của batch

    file_path = '../../../Dataset/train_set.csv'
    # gamma_weights = {0: 2.0, 1: 2.0, 2: 2.0}

    # focal_loss = FocalLoss(file_path=file_path, label_name="Pose", gamma_weights=gamma_weights)

    # # Tính Focal Loss
    # loss = focal_loss(y_pred, y_true)
    # print(f"Focal Loss: {loss.item()}")
    
    # Tốt không cần chỉnh
    # Gender: tensor([0.9160, 0.0840])
    # Facial_Hair: tensor([0.2989, 0.7011])
    # Spectacles: tensor([0.2770, 0.7230])
    
    # Cần cải thiện
    # Emotion: tensor([0.2316, 0.9124, 0.8560])
    # Pose: tensor([0.1414, 0.8858, 0.9729])
    # Occlusion: tensor([0.9963, 0.9828, 0.0209])
    
    focal_loss = FocalLoss(file_path=file_path, label_name="Pose")
    print(focal_loss.alpha)
    