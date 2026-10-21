import torch

from .focalloss.FocalLoss import FocalLoss
from .WeightClassMagLoss import WeightClassMagLoss

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class ConcatMultiTaskLoss(torch.nn.Module):
    
    def __init__(self, metadata_path:str, loss_weight:dict):
        super(ConcatMultiTaskLoss, self).__init__()
        
        self.id_loss = WeightClassMagLoss(metadata_path)
        # 0: female (235), 1: male (2579)
        self.gender_loss = FocalLoss(alpha_weights={0:0.916, 1:0.084}, gamma_weights={0:2, 1:0}, num_classes=2)
        # 0: không đeo kính (2026), 1: đeo kính (788)
        self.spectacles_loss = FocalLoss(alpha_weights={0: 0.28, 1: 0.72}, gamma_weights={0:0, 1:1}, num_classes=2)
        # 0: không râu (1965), 1: có râu (849)
        self.facial_hair_loss = FocalLoss(alpha_weights={0:0.3, 1:0.7}, gamma_weights={0:0, 1:1}, num_classes=2)
        # 0: nhìn trực diện (2740), 1: nhìn nghiêng 1 chút (74)
        self.pose_loss = FocalLoss(alpha_weights={0: 0.0263, 1: 0.9737}, gamma_weights={0:0, 1:2.5}, num_classes=2)
        # 0: nhìn trực diện (2162), 1: các cảm xúc khác (652)
        self.emotion_loss = FocalLoss(alpha_weights={0:0.232, 1:0.768}, gamma_weights={0:0, 1:1}, num_classes=2)
        
        # hyper parameter
        self.spectacles_weight = loss_weight['loss_spectacles_weight']
        self.facial_hair_weight = loss_weight['loss_facial_hair_weight']
        self.pose_weight = loss_weight['loss_pose_weight']
        self.gender_weight = loss_weight['loss_gender_weight']
        self.emotion_weight = loss_weight['loss_emotion_weight']
        
        
    def forward(self, logits, y):
        (
            x_spectacles,
            x_facial_hair,
            x_pose,
            x_emotion,
            x_gender,
            x_id_logits, x_id_norm
        ) = logits
        
        id, gender, spectacles, facial_hair, pose, emotion = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5]
        
        loss_spectacles = self.spectacles_loss(x_spectacles, spectacles)
        
        loss_facial_hair = self.facial_hair_loss(x_facial_hair, facial_hair)
        
        loss_pose = self.pose_loss(x_pose, pose)
        
        loss_emotion = self.emotion_loss(x_emotion, emotion)
        
        loss_gender = self.gender_loss(x_gender, gender)
        
        loss_id = self.id_loss(x_id_logits, id, x_id_norm)
        
        total_loss =    loss_id + \
                        loss_gender * self.gender_weight + \
                        loss_emotion * self.emotion_weight + \
                        loss_pose * self.pose_weight + \
                        loss_facial_hair * self.facial_hair_weight + \
                        loss_spectacles * self.spectacles_weight
        
        return (
            total_loss,
            loss_id,
            loss_gender,
            loss_emotion,
            loss_pose,
            loss_facial_hair,
            loss_spectacles
        )
    