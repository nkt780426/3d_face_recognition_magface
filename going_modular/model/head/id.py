import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class MagLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, easy_margin=True, l_margin=0.45, u_margin=0.8, l_a=10, u_a=110):
        super(MagLinear, self).__init__()
        # Tạo ra ma trận cần train của lớp FullyConnected này
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        # Can thiệp vào quá trình khởi tạo trọng số: 
        # uniform_(-1,1): đảm bảo phân phối đồng đều (uniform) trong khoảng [-1,1]
        # renorm_(2, 1, 1e-5): chuẩn hóa lại các giá trị theo 1 chiều cụ thể. 
        #   2: Chuẩn hóa theo chuẩn L2(Euclidean norm).
        #   1: Thực hiện chuẩn hóa trên chiều thứ nhất của tensor (các hàng của ma trận self.weight)
        #   1e-5: Giá trị ngưỡng (epsilon) để tránh chia cho số 0
        # Khi dùng chuẩn hóa này, mỗi hàng của ma trận trọng số sẽ được điều chỉnh sao cho norm của chúng không vượt quá một giá trị cụ thể (giới hạn bởi epsilon).
        # Bản thân mỗi cột của ma trận này là ma trận weight của 1 neutron hay vector tâm của class identity.
        self.weight.data.uniform_(-1,1).renorm_(2, 1, 1e-5).mul_(1e5)
        
        self.easy_margin = easy_margin
        self.l_margin, self.u_margin = l_margin, u_margin
        self.l_a, self.u_a = l_a, u_a
    
    # Generate adaptive margin
    def _margin(self, x):
        margin = (self.u_margin-self.l_margin) / \
            (self.u_a-self.l_a)*(x-self.l_a) + self.l_margin
        return margin
    
    def forward(self, x):
        # normalize lại để tính cosine ổn định
        # clamp(min, max) giới hạn giá trị trong tensor sao cho tất cả các phần tử nhỏ hơn min sẽ được thay thế bằng min, và tất cả các phần tử lớn hơn max sẽ được thay thế bằng max.
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        
        # adaptive margin
        ada_margin = self._margin(x_norm)
        
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)
        
        # Tính theta yi
        # dim bằng 0 tức là chuẩn hóa ma trận theo chiều batch
        weight_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(F.normalize(x), weight_norm)
        # Đảm bảo giá trị cosin của các vector không vượt quá giới hạn => Thừa
        cos_theta = cos_theta.clamp(-1, 1)
        # sin sẽ luôn dương
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        # cos(a+b) = cos(a)*cos(b) - sin(a)*sin(b)
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

        # easy_margin = True để điều chỉnh cos_theta_m, nó sẽ bằng công thức trên nếu cos_theta>0 và không điều chỉnh gì nếu cos_theta<0
        # Cách này làm đơn giản việc tính toán
        # Nếu muốn điều chỉnh thì cần thêm vào threshold nào đó (code của họ để mặc định easy_margin=False)
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            mm = torch.sin(math.pi - ada_margin) * ada_margin
            threshold = torch.cos(math.pi - ada_margin)
            cos_theta_m = torch.where(cos_theta > threshold, cos_theta_m, cos_theta - mm)
            
        # Trả về các giá trị này để tính loss và accuracy
        # [cos_theta, cos_theta_m] là logits của lớp này. Lý do thêm x_norm để phục vụ MagFace+ => Train không ?
        # cos_theta chính là accuracy, nó đo cosine similarity giữa vector tâm và feature vector.
        return [cos_theta, cos_theta_m], x_norm
 
    
class IdRecognitionModule(nn.Module):
    def __init__(self, num_classes):
        super(IdRecognitionModule, self).__init__()
        self.id_embedding = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.maglinear = MagLinear(512, num_classes)

    def forward(self, x_id):
        x_id = self.id_embedding(x_id)
        logits, x_norm = self.maglinear(x_id)
        return logits, x_norm
    