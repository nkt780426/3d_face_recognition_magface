import torch
from torch.nn import Linear

from .MTLFaceRecognition import MTLFaceRecognition

from .grl import GradientReverseLayer

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class ConcatMTLFaceRecognition(torch.nn.Module):


    def __init__(self, mtl_normalmap: MTLFaceRecognition, mtl_albedo:MTLFaceRecognition, mtl_depthmap:MTLFaceRecognition):
        super(ConcatMTLFaceRecognition, self).__init__()
        self.mtl_normalmap = mtl_normalmap
        self.mtl_albedo = mtl_albedo
        self.mtl_depthmap = mtl_depthmap
       
        # concat head
        self.id_head = IdRecognitionModule(num_classes)
        self.gender_head = nn.
        self.emotion_head = EmotionDetectModule()
        self.facial_hair_head = FacialHairDetectModule()
        self.pose_head = PoseDetectModule()
        self.spectacles_head = SpectacleDetectModule()
        
        # da_discriminator (domain adaptation)
        self.da_gender_head = GenderDetectModule()
        self.da_emotion_head = EmotionDetectModule()
        self.da_facial_hair_head = FacialHairDetectModule()
        self.da_pose_head = PoseDetectModule()
        self.da_spectacles_head = SpectacleDetectModule()
        
        # grl
        self.grl_gender = GradientReverseLayer()
        self.grl_emotion = GradientReverseLayer()
        self.grl_facial_hair = GradientReverseLayer()
        self.grl_pose = GradientReverseLayer()
        self.grl_spectacles = GradientReverseLayer()
        
    def forward(self, x):
        x_normalmap = x[:, 0, :, :, :]
        x_albedo = x[:, 1, :, :, :]
        x_depthmap = x[:, 2, :, :, :]
        
        (
            (x_normalmap_spectacles, x_normalmap_da_spectacles), 
            (x_normalmap_facial_hair, x_normalmap_da_facial_hair),
            (x_normalmap_pose, x_normalmap_da_pose),
            (x_normalmap_emotion, x_normalmap_da_emotion),
            (x_normalmap_gender, x_normalmap_da_gender),
            x_normalmap_id_logits, x_normalmap_id_norm
        ) = self.mtl_normalmap(x_normalmap)
        
        (
            (x_albedospectacles, x_albedoda_spectacles), 
            (x_albedofacial_hair, x_albedoda_facial_hair),
            (x_albedopose, x_albedoda_pose),
            (x_albedoemotion, x_albedoda_emotion),
            (x_albedogender, x_albedoda_gender),
            x_albedoid_logits, x_albedoid_norm
        ) = self.mtl_albedo(x_albedo)
        
        (
            (x_depthmap_spectacles, x_depthmap_da_spectacles), 
            (x_depthmap_facial_hair, x_depthmap_da_facial_hair),
            (x_depthmap_pose, x_depthmap_da_pose),
            (x_depthmap_emotion, x_depthmap_da_emotion),
            (x_depthmap_gender, x_depthmap_da_gender),
            x_depthmap_id_logits, x_depthmap_id_norm
        ) = self.mtl_depthmap(x_depthmap)
    
    def get_embedding(self, x):
        (
            (x_spectacles, x_non_spectacles),
            (x_facial_hair, x_non_facial_hair),
            (x_emotion, x_non_emotion),
            (x_pose, x_non_pose),
            (x_gender, x_id)
        ) = self.backbone(x)
        x_id = self.id_head.id_ouput_layer(x_id)
        x_gender = self.gender_head.gender_output_layer(x_gender)
        x_pose = self.pose_head.pose_ouput_layer(x_pose)
        x_emotion = self.emotion_head.emotion_ouput_layer(x_emotion)
        x_facial_hair = self.facial_hair_head.facial_hair_ouput_layer(x_facial_hair)
        x_spectacles = self.spectacles_head.spectacle_ouput_layer(x_spectacles)
        return x_id, x_gender, x_pose, x_emotion, x_facial_hair, x_spectacles
