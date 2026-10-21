import torch
import torch.nn as nn
import timm 

class SPPModule(nn.Module):
    def __init__(self, pool_mode='avg', sizes=(1, 2, 3, 6)):
        super().__init__()
        pool_layer = nn.AdaptiveAvgPool2d if pool_mode == 'avg' else nn.AdaptiveMaxPool2d
        self.pool_blocks = nn.ModuleList([
            nn.Sequential(pool_layer(size), nn.Flatten()) for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.pool_blocks]
        x = torch.cat(xs, dim=1)
        return x.view(x.size(0), x.size(1), 1, 1)

class AttentionModule(nn.Module):
    def __init__(self, channels=512, reduction=16):
        super(AttentionModule, self).__init__()
        kernel_size = 7
        pool_size = (1, 2, 3)
        self.avg_spp = SPPModule('avg', pool_size)
        self.max_spp = SPPModule('max', pool_size)
        
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        _channels = channels * int(sum([x ** 2 for x in pool_size]))
        self.channel = nn.Sequential(
            nn.Conv2d(_channels, _channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(_channels // reduction, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):   
        channel_scale = self.channel(self.avg_spp(x) + self.max_spp(x))
        spatial_input = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        spatial_scale = self.spatial(spatial_input)
        x_non_id = (x * channel_scale + x * spatial_scale) * 0.5
        return x - x_non_id, x_non_id

class MIConvNeXtV2(nn.Module):
    def __init__(self, model_name='convnextv2_tiny', pretrained=True, **kwargs):
        super(MIConvNeXtV2, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        
        # Tự động lấy out_channels để tránh lỗi NameError
        dummy_input = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            out_channels = self.backbone(dummy_input)[-1].shape[1]
        
        self.target_channels = 512
        self.adapter_conv = nn.Conv2d(out_channels, self.target_channels, kernel_size=1, bias=False)
        self.adapter_bn = nn.BatchNorm2d(self.target_channels)
        self.adapter_act = nn.PReLU(self.target_channels)
        
        self.spectacles_fsm = AttentionModule(channels=self.target_channels)
        self.facial_hair_fsm = AttentionModule(channels=self.target_channels)
        self.emotion_fsm = AttentionModule(channels=self.target_channels)
        self.pose_fsm = AttentionModule(channels=self.target_channels)
        self.gender_fsm = AttentionModule(channels=self.target_channels)

    def forward(self, x):
        x = self.adapter_act(self.adapter_bn(self.adapter_conv(self.backbone(x)[-1])))
        x_non_spec, x_spec = self.spectacles_fsm(x)
        x_non_fh, x_fh = self.facial_hair_fsm(x_non_spec)
        x_non_emot, x_emot = self.emotion_fsm(x_non_fh)
        x_non_pose, x_pose = self.pose_fsm(x_non_emot)
        x_id, x_gen = self.gender_fsm(x_non_pose)
        return ((x_spec, x_non_spec), (x_fh, x_non_fh), (x_emot, x_non_emot), (x_pose, x_non_pose), (x_gen, x_id))

def create_miconvnextv2(model_name='convnextv2_tiny', **kwargs):
    return MIConvNeXtV2(model_name, **kwargs)