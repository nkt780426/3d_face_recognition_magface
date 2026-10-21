import torch
import torch.nn.functional as F
import pandas as pd

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)


class IdFocalLoss(torch.nn.Module):
    
    def __init__(self, file_path, gamma_thresholds=((0.7, 2.0), (0.4, 1.0), (0.2, 0.25))):
        super(IdFocalLoss, self).__init__()
        self.alpha = self._compute_alpha(file_path)

        self.gamma = self._compute_gamma(gamma_thresholds)
    
        
    def _compute_alpha(self, file_path):
        # Đọc dữ liệu từ file CSV
        self.data = pd.read_csv(file_path)
        
        # Đếm số lượng mẫu của từng nhãn
        label_counts = self.data['id'].value_counts().sort_index()
        
        # Tổng số mẫu
        total_samples = label_counts.sum()
        
        # Tính tần suất ngược
        alpha = 1 - (label_counts.values / total_samples)
        
        # Chuẩn hóa alpha về khoảng [0, 1]
        alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-8)
        
        # Giới hạn alpha trong khoảng [0.2, 0.8]
        min_alpha, max_alpha = 0.2, 0.8
        alpha = alpha * (max_alpha - min_alpha) + min_alpha
        
        # Chuyển alpha thành tensor
        return torch.tensor(alpha, dtype=torch.float32)


    def _compute_gamma(self, gamma_thresholds):
        gamma_values = []
        for alpha_value in self.alpha:
            gamma = self._get_gamma_for_alpha(alpha_value.item(), gamma_thresholds)
            gamma_values.append(gamma)
        return torch.tensor(gamma_values, dtype=torch.float32)


    def _get_gamma_for_alpha(self, alpha, gamma_thresholds):
        for threshold, gamma_value in gamma_thresholds:
            if alpha > threshold:
                return gamma_value
        return 0.0


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

# Hàm main để test
if __name__ == '__main__':
    # # Tạo dữ liệu giả lập (sử dụng số lớp là 3, ví dụ)
    # batch_size = 5
    # num_classes = 3
    # y_true = torch.tensor([0, 1, 2, 0, 1])  # Chỉ số lớp giả lập (có 3 lớp: 0, 1, 2)
    # y_pred = torch.randn(batch_size, num_classes)  # Đầu ra của mô hình (chưa qua softmax)

    # # Tạo đối tượng IdFocalLoss (giả sử bạn đã có đường dẫn đến file CSV)
    # focal = IdFocalLoss('../../../Dataset/train_set.csv')

    # # Tính loss
    # loss = focal(y_true, y_pred)
    
    # print(f"Focal Loss: {loss.item()}")
    file_path = '../../../Dataset/train_set.csv'
    focal_loss = IdFocalLoss(file_path=file_path)
    print(focal_loss.gamma)