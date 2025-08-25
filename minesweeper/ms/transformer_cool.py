# pip install torch>=2.1 rotary-embedding-torch
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 2D / axial RoPE for vision
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb


# -------------------------
#  Basics: MLP + norm block
# -------------------------
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ------------------------------------------------------
#  Full-attention MHA with 2D RoPE injected into Q and K
# ------------------------------------------------------
class MultiHeadSelfAttention2DRoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dropout: float = 0.0,
        rope_dim_per_head: Optional[int] = None,
        rope_max_freq: int = 256,
    ):
        """
        dim: model width (must be divisible by heads)
        heads: number of attention heads
        dropout: attention dropout (pre-output projection)
        rope_dim_per_head: how many dims per head to rotate with RoPE (<= dim/head). If None, uses full head dim.
        rope_max_freq: max spatial frequency for axial/pixel RoPE (tuned around image sizes)
        """
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5

        # qkv + out projection
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

        # 2D axial rotary embedding (vision-friendly RoPE)
        # We allow partial rotary (common in practice): rotate only a slice of the head dims
        rope_use_dim = self.dim_head if rope_dim_per_head is None else min(rope_dim_per_head, self.dim_head)
        self.rope = RotaryEmbedding(dim=rope_use_dim, freqs_for="pixel", max_freq=rope_max_freq)
        self.rope_use_dim = rope_use_dim

    @torch.no_grad()
    def _axial_freqs(self, H: int, W: int) -> torch.Tensor:
        # Returns (H, W, rope_dims_for_H_plus_W) and will be broadcast by apply_rotary_emb
        return self.rope.get_axial_freqs(H, W)

    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        """
        x: (b, L, dim) where L = H * W
        hw: (H, W)
        """
        b, L, d = x.shape
        H, W = hw
        assert L == H * W, f"Sequence length {L} must equal H*W={H*W}"

        qkv = self.to_qkv(x)  # (b, L, 3*dim)
        q, k, v = qkv.chunk(3, dim=-1)  # (b, L, dim) each

        # reshape to heads
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # (b, L, dim) -> (b, heads, L, dim_head)
            return t.view(b, L, self.heads, self.dim_head).transpose(1, 2).contiguous()

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # reshape to 2D grid for 2D RoPE
        # shapes: (b, heads, H, W, dim_head)
        q2d = q.view(b, self.heads, H, W, self.dim_head)
        k2d = k.view(b, self.heads, H, W, self.dim_head)

        # axial 2D RoPE: rotate only first rope_use_dim of last channel
        freqs = self._axial_freqs(H, W)  # (H, W, rope_dim_*2)
        if self.rope_use_dim < self.dim_head:
            # split: [rotary part | passthrough part]
            q_rot, q_pass = q2d[..., : self.rope_use_dim], q2d[..., self.rope_use_dim :]
            k_rot, k_pass = k2d[..., : self.rope_use_dim], k2d[..., self.rope_use_dim :]

            q_rot = apply_rotary_emb(freqs, q_rot)
            k_rot = apply_rotary_emb(freqs, k_rot)

            q2d = torch.cat([q_rot, q_pass], dim=-1)
            k2d = torch.cat([k_rot, k_pass], dim=-1)
        else:
            q2d = apply_rotary_emb(freqs, q2d)
            k2d = apply_rotary_emb(freqs, k2d)

        # back to (b, heads, L, dim_head)
        q = q2d.view(b, self.heads, L, self.dim_head)
        k = k2d.view(b, self.heads, L, self.dim_head)

        # PyTorch 2.x fused attention (full attention, not causal)
        # shapes must be (b*heads, L, dim_head) for the SDPA convenience API with batch_first=True
        q = q.transpose(1, 2)  # (b, L, heads, dim_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = q.reshape(b, L, self.heads * self.dim_head)
        k = k.reshape(b, L, self.heads * self.dim_head)
        v = v.reshape(b, L, self.heads * self.dim_head)

        # recover heads for SDPA
        q = q.view(b, L, self.heads, self.dim_head).transpose(1, 2)  # (b, heads, L, dim_head)
        k = k.view(b, L, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, L, self.heads, self.dim_head).transpose(1, 2)

        # scale queries
        q = q * self.scale

        # SDPA expects (b*heads, L, d)
        q = q.reshape(b * self.heads, L, self.dim_head)
        k = k.reshape(b * self.heads, L, self.dim_head)
        v = v.reshape(b * self.heads, L, self.dim_head)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        attn = attn.reshape(b, self.heads, L, self.dim_head).transpose(1, 2).contiguous()  # (b, L, dim)
        attn = attn.view(b, L, self.dim)

        out = self.proj(attn)
        out = self.drop(out)
        return out


# ---------------------------------------
#  Transformer encoder block (Pre-LN)
# ---------------------------------------
class EncoderBlock2D(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        rope_dim_per_head: Optional[int] = None,
        rope_max_freq: int = 256,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention2DRoPE(
            dim=dim,
            heads=heads,
            dropout=dropout,
            rope_dim_per_head=rope_dim_per_head,
            rope_max_freq=rope_max_freq,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), hw)
        x = x + self.mlp(self.norm2(x))
        return x


# -------------------------------------------------------
#  Full model: categorical grid -> per-cell scalar output
# -------------------------------------------------------
@dataclass
class GridTransformer2DConfig:
    num_categories: int
    width: int
    height: int
    emb_dim: int = 256
    depth: int = 6
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    rope_dim_per_head: Optional[int] = None   # e.g., 32 for partial rotary; None = full head dim
    rope_max_freq: int = 256                  # tune roughly around max(H, W)


class GridTransformer2D(nn.Module):
    """
    Integer grid (b, W, H) -> embeddings (b, W, H, D) -> 2D RoPE -> flatten (b, W*H, D)
    -> Transformer encoder (full attention) -> MLP head to (b, W*H, 1) -> reshape (b, W, H)
    """
    def __init__(self, cfg: GridTransformer2DConfig):
        super().__init__()
        self.cfg = cfg
        self.W = cfg.width
        self.H = cfg.height
        self.D = cfg.emb_dim

        # Categorical token embedding at each cell
        self.token_embed = nn.Embedding(cfg.num_categories, cfg.emb_dim)

        # Stack of encoder blocks
        self.blocks = nn.ModuleList([
            EncoderBlock2D(
                dim=cfg.emb_dim,
                heads=cfg.heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                rope_dim_per_head=cfg.rope_dim_per_head,
                rope_max_freq=cfg.rope_max_freq,
            )
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.emb_dim)

        # Per-token head -> scalar
        self.head = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.emb_dim),
            nn.GELU(),
            nn.Linear(cfg.emb_dim, 1),
        )

    def forward(self, x_idx: torch.Tensor) -> torch.Tensor:
        """
        x_idx: (b, W, H) integer categories
        returns: (b, W, H) float predictions
        """
        b, W, H = x_idx.shape
        assert W == self.W and H == self.H, f"Expected input (b,{self.W},{self.H}), got (b,{W},{H})"

        # (b, W, H, D)
        x = self.token_embed(x_idx)  # nn.Embedding works on (...,) so broadcast ok: (b, W, H, D)

        # flatten to (b, L, D) with L = W*H; keep (W,H) for 2D RoPE
        x = x.view(b, W * H, self.D)

        # transformer encoder with 2D RoPE in every attention
        for blk in self.blocks:
            x = blk(x, hw=(H, W))  # note: RoPE expects (H, W)

        x = self.norm(x)  # (b, L, D)

        # per-token scalar
        y = self.head(x)              # (b, L, 1)
        y = y.view(b, W, H, 1).squeeze(-1)  # (b, W, H)
        return y


# -----------------
#  Quick smoke test
# -----------------
if __name__ == "__main__":
    cfg = GridTransformer2DConfig(
        num_categories=11,
        width=10,
        height=10,
        emb_dim=256,
        depth=6,
        heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        rope_dim_per_head=64,   # rotate a subset of dims per head (often enough & efficient)
        rope_max_freq=256,
    )
    model = GridTransformer2D(cfg)

    x_idx = torch.randint(0, cfg.num_categories, (2, cfg.width, cfg.height))  # (b, W, H)
    y = model(x_idx)  # (b, W, H)
    print(y.shape)
