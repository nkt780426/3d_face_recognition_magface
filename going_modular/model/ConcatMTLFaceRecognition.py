import torch
from torch.nn import Linear

from .MTLFaceRecognition import MTLFaceRecognition
from .head.id import MagLinear

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class ConcatMTLFaceRecognitionV3(torch.nn.Module):


    def __init__(self, mtl_normalmap: MTLFaceRecognition, mtl_albedo:MTLFaceRecognition, mtl_depthmap:MTLFaceRecognition, num_classes):
        super(ConcatMTLFaceRecognitionV3, self).__init__()
        self.mtl_normalmap = mtl_normalmap
        self.mtl_albedo = mtl_albedo
        self.mtl_depthmap = mtl_depthmap
       
        # concat head
        self.id_head = MagLinear(1536, num_classes)
        self.gender_head = Linear(1536, 2)
        self.emotion_head = Linear(1536, 2)
        self.facial_hair_head = Linear(1536, 2)
        self.pose_head = Linear(1536, 2)
        self.spectacles_head = Linear(1536, 2)
  
        
    def forward(self, x):
        x_normalmap = x[:, 0, :, :, :]
        x_albedo = x[:, 1, :, :, :]
        x_depthmap = x[:, 2, :, :, :]
        
        x_normalmap_spectacles, x_normalmap_facial_hair, x_normalmap_pose, x_normalmap_emotion, x_normalmap_gender, x_normalmap_id = self.mtl_normalmap.get_embedding(x_normalmap)
        
        x_albedo_spectacles, x_albedo_facial_hair, x_albedo_pose, x_albedo_emotion, x_albedo_gender, x_albedo_id = self.mtl_albedo.get_embedding(x_albedo)
        
        x_depthmap_spectacles, x_depthmap_facial_hair, x_depthmap_pose, x_depthmap_emotion, x_depthmap_gender, x_depthmap_id = self.mtl_depthmap.get_embedding(x_depthmap)
            
        # Concatenate embeddings from all modalities (normalmap, albedo, depthmap)
        spectacles_embedding = torch.cat([x_normalmap_spectacles, x_albedo_spectacles, x_depthmap_spectacles], dim=1)
        facial_hair_embedding = torch.cat([x_normalmap_facial_hair, x_albedo_facial_hair, x_depthmap_facial_hair], dim=1)
        pose_embedding = torch.cat([x_normalmap_pose, x_albedo_pose, x_depthmap_pose], dim=1)
        emotion_embedding = torch.cat([x_normalmap_emotion, x_albedo_emotion, x_depthmap_emotion], dim=1)
        gender_embedding = torch.cat([x_normalmap_gender, x_albedo_gender, x_depthmap_gender], dim=1)
        id_embedding = torch.cat([x_normalmap_id, x_albedo_id, x_depthmap_id], dim=1)

        # Pass concatenated embeddings through their respective linear layers
        x_id_logits, x_id_norm = self.id_head(id_embedding)
        x_gender = self.gender_head(gender_embedding)
        x_emotion = self.emotion_head(emotion_embedding)
        x_facial_hair = self.facial_hair_head(facial_hair_embedding)
        x_pose = self.pose_head(pose_embedding)
        x_spectacles = self.spectacles_head(spectacles_embedding)

        logits = (
                    x_spectacles,
                    x_facial_hair,
                    x_pose,
                    x_emotion,
                    x_gender,
                    x_id_logits, x_id_norm
                )
        return logits
  
  
    # Trả về các task khác như bình thường trừ id chỉ trả về embedding
    def get_result(self, x):
        x_normalmap = x[:, 0, :, :, :]
        x_albedo = x[:, 1, :, :, :]
        x_depthmap = x[:, 2, :, :, :]
        
        x_normalmap_spectacles, x_normalmap_facial_hair, x_normalmap_pose, x_normalmap_emotion, x_normalmap_gender, x_normalmap_id = self.mtl_normalmap.get_embedding(x_normalmap)
        
        x_albedo_spectacles, x_albedo_facial_hair, x_albedo_pose, x_albedo_emotion, x_albedo_gender, x_albedo_id = self.mtl_albedo.get_embedding(x_albedo)
        
        x_depthmap_spectacles, x_depthmap_facial_hair, x_depthmap_pose, x_depthmap_emotion, x_depthmap_gender, x_depthmap_id = self.mtl_depthmap.get_embedding(x_depthmap)
            
        # Concatenate embeddings from all modalities (normalmap, albedo, depthmap)
        spectacles_embedding = torch.cat([x_normalmap_spectacles, x_albedo_spectacles, x_depthmap_spectacles], dim=1)
        facial_hair_embedding = torch.cat([x_normalmap_facial_hair, x_albedo_facial_hair, x_depthmap_facial_hair], dim=1)
        pose_embedding = torch.cat([x_normalmap_pose, x_albedo_pose, x_depthmap_pose], dim=1)
        emotion_embedding = torch.cat([x_normalmap_emotion, x_albedo_emotion, x_depthmap_emotion], dim=1)
        gender_embedding = torch.cat([x_normalmap_gender, x_albedo_gender, x_depthmap_gender], dim=1)
        id_embedding = torch.cat([x_normalmap_id, x_albedo_id, x_depthmap_id], dim=1)
        
        x_gender = self.gender_head(gender_embedding)
        x_emotion = self.emotion_head(emotion_embedding)
        x_facial_hair = self.facial_hair_head(facial_hair_embedding)
        x_pose = self.pose_head(pose_embedding)
        x_spectacles = self.spectacles_head(spectacles_embedding)

        return id_embedding, x_gender, x_pose, x_emotion, x_facial_hair, x_spectacles
    
class ConcatMTLFaceRecognitionV2(torch.nn.Module):


    def __init__(self, mtl_backbone1: MTLFaceRecognition, mtl_backbone2:MTLFaceRecognition, num_classes):
        super(ConcatMTLFaceRecognitionV2, self).__init__()
        self.mtl_backbone1 = mtl_backbone1
        self.mtl_backbone2 = mtl_backbone2
       
        # concat head
        self.id_head = MagLinear(1024, num_classes)
        self.gender_head = Linear(1024, 2)
        self.emotion_head = Linear(1024, 2)
        self.facial_hair_head = Linear(1024, 2)
        self.pose_head = Linear(1024, 2)
        self.spectacles_head = Linear(1024, 2)
  
        
    def forward(self, x):
        x_backbone1 = x[:, 0, :, :, :]
        x_backbone2 = x[:, 1, :, :, :]
        
        x_backbone1_spectacles, x_backbone1_facial_hair, x_backbone1_pose, x_backbone1_emotion, x_backbone1_gender, x_backbone1_id = self.mtl_backbone1.get_embedding(x_backbone1)
        
        x_backbone2_spectacles, x_backbone2_facial_hair, x_backbone2_pose, x_backbone2_emotion, x_backbone2_gender, x_backbone2_id = self.mtl_backbone2.get_embedding(x_backbone2)
        
        # Concatenate embeddings from all modalities (normalmap, albedo, depthmap)
        spectacles_embedding = torch.cat([x_backbone1_spectacles, x_backbone2_spectacles], dim=1)
        facial_hair_embedding = torch.cat([x_backbone1_facial_hair, x_backbone2_facial_hair], dim=1)
        pose_embedding = torch.cat([x_backbone1_pose, x_backbone2_pose], dim=1)
        emotion_embedding = torch.cat([x_backbone1_emotion, x_backbone2_emotion], dim=1)
        gender_embedding = torch.cat([x_backbone1_gender, x_backbone2_gender], dim=1)
        id_embedding = torch.cat([x_backbone1_id, x_backbone2_id], dim=1)

        # Pass concatenated embeddings through their respective linear layers
        x_id_logits, x_id_norm = self.id_head(id_embedding)
        x_gender = self.gender_head(gender_embedding)
        x_emotion = self.emotion_head(emotion_embedding)
        x_facial_hair = self.facial_hair_head(facial_hair_embedding)
        x_pose = self.pose_head(pose_embedding)
        x_spectacles = self.spectacles_head(spectacles_embedding)

        logits = (
                    x_spectacles,
                    x_facial_hair,
                    x_pose,
                    x_emotion,
                    x_gender,
                    x_id_logits, x_id_norm
                )
        return logits
  
  
    # Trả về các task khác như bình thường trừ id chỉ trả về embedding
    def get_result(self, x):
        x_backbone1 = x[:, 0, :, :, :]
        x_backbone2 = x[:, 1, :, :, :]
        
        x_backbone1_spectacles, x_backbone1_facial_hair, x_backbone1_pose, x_backbone1_emotion, x_backbone1_gender, x_backbone1_id = self.mtl_backbone1.get_embedding(x_backbone1)
        
        x_backbone2_spectacles, x_backbone2_facial_hair, x_backbone2_pose, x_backbone2_emotion, x_backbone2_gender, x_backbone2_id = self.mtl_backbone2.get_embedding(x_backbone2)
        
        # Concatenate embeddings from all modalities (normalmap, albedo, depthmap)
        spectacles_embedding = torch.cat([x_backbone1_spectacles, x_backbone2_spectacles], dim=1)
        facial_hair_embedding = torch.cat([x_backbone1_facial_hair, x_backbone2_facial_hair], dim=1)
        pose_embedding = torch.cat([x_backbone1_pose, x_backbone2_pose], dim=1)
        emotion_embedding = torch.cat([x_backbone1_emotion, x_backbone2_emotion], dim=1)
        gender_embedding = torch.cat([x_backbone1_gender, x_backbone2_gender], dim=1)
        id_embedding = torch.cat([x_backbone1_id, x_backbone2_id], dim=1)
        
        x_gender = self.gender_head(gender_embedding)
        x_emotion = self.emotion_head(emotion_embedding)
        x_facial_hair = self.facial_hair_head(facial_hair_embedding)
        x_pose = self.pose_head(pose_embedding)
        x_spectacles = self.spectacles_head(spectacles_embedding)

        return id_embedding, x_gender, x_pose, x_emotion, x_facial_hair, x_spectacles
        