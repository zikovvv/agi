# file: cnn_model.py
import torch
from torch import nn

class CNNModel(nn.Module):
    def __init__(self, height: int, width: int, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=1),  # final projection to logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B,1,H,W)
