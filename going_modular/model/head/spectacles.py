import torch
from torch import nn

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class SpectacleDetectModule(nn.Module):
    def __init__(self):
        super(SpectacleDetectModule, self).__init__()
        out_neurons = 2
        self.spectacle_ouput_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, out_neurons)
        )

    def forward(self, x_spectacle):
        x_spectacle = self.spectacle_ouput_layer(x_spectacle)
        return x_spectacle