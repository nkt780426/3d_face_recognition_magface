import torch
from torch import nn

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class GenderDetectModule(nn.Module):
    def __init__(self):
        super(GenderDetectModule, self).__init__()
        out_neurons = 2
        self.gender_output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, out_neurons),
        )

    def forward(self, x_gender):
        x_gender = self.gender_output_layer(x_gender)
        return x_gender