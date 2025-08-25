# file: transformer_model.py
import torch
from torch import nn
import math

class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, height: int, width: int):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(d_model, height, width)
        y_pos = torch.arange(0, height, dtype=torch.float32).unsqueeze(1)
        x_pos = torch.arange(0, width, dtype=torch.float32).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[0::2, :, :] = torch.sin(y_pos * div_term).unsqueeze(-1).repeat(1,1,width)
        pe[1::2, :, :] = torch.cos(x_pos * div_term).unsqueeze(-2).repeat(1,height,1)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, D, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe  # (B, D, H, W)

class TransformerModel(nn.Module):
    """
    Full-attention encoder transformer taking categorical field.
    """

    def __init__(self, height: int, width: int,
                 emb_dim: int = 64,
                 nhead: int = 4,
                 num_layers: int = 3,
                 ff_dim: int = 256,
                 hidden_mlp: int = 512,
                 categories: int = 10):
        """
        categories: number of distinct categorical values (e.g., -1 → 0, 0→1, .. 8→9)
        """
        super().__init__()
        self.H = height
        self.W = width
        self.seq_len = height * width
        self.emb = nn.Embedding(categories, emb_dim)
        self.pos_enc = PositionalEncoding2D(emb_dim, height, width)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=ff_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # MLP projection to single logit per token
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,H,W), int-encoded categorical values
        B = x.shape[0]
        x = x.squeeze(1).long()  # → (B, H, W)
        x = self.emb(x)          # → (B, H, W, emb_dim)
        x = x.permute(0,3,1,2)   # → (B, emb_dim, H, W)
        x = self.pos_enc(x)      # → (B, emb_dim, H, W)
        x = x.flatten(2).permute(2, 0, 1)  # → (seq_len, B, emb_dim)
        out = self.transformer(x)         # → (seq_len, B, emb_dim)
        out = out.permute(1,0,2).contiguous()  # (B, seq_len, emb_dim)
        out = self.mlp(out)               # → (B, seq_len, 1)
        out = out.view(B, 1, self.H, self.W)
        return out  # logits

