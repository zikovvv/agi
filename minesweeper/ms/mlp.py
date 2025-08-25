# file: mlp_model.py
import torch
from torch import nn

class MLPModel(nn.Module):
    """
    Simple MLP mapping (B,1,H,W) → (B,1,H,W) bomb logits.
    """
    def __init__(self, height: int, width: int, hidden_dim: int = 512):
        super().__init__()
        self.H = height
        self.W = width
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(height * width, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, height * width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,H,W)
        B = x.shape[0]
        out = self.flatten(x)           # → (B, H*W)
        out = self.net(out)             # → (B, H*W)
        out = out.view(B, 1, self.H, self.W)
        return out  # logits (trainer expects from_logits=True)
