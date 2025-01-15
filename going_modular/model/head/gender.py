import torch
from torch import nn

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class GenderDetectModule(nn.Module):
    def __init__(self):
        super(GenderDetectModule, self).__init__()
        out_neurons = 2
        self.gender_embedding = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.gender_linear = nn.Linear(512, out_neurons)

    def forward(self, x_gender):
        x_gender = self.gender_embedding(x_gender)
        x_gender = self.gender_linear(x_gender)
        return x_gender