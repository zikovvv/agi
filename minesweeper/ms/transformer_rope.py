import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding
import math

class SimpleRoPETransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1, max_seq_len=512):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_scale = math.sqrt(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prepare RoPE embedding for head dimension
        head_dim = d_model // nhead
        self.rotary_emb = RotaryEmbedding(dim=head_dim)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src shape: (batch, seq_len)
        x = self.embedding(src) * self.pos_scale  # (batch, seq_len, d_model)
        # For simplicity, fallback to sequential application
        return self.encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

# Alternative: simpler custom single-layer with RoPE and MultiheadAttention
class RoPEEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.rotary_emb = RotaryEmbedding(dim=d_model // nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        qkv = src
        # Compute self-attention manually to insert RoPE
        q = k = v = src  # (batch, seq_len, d_model)
        attn_output, attn_weights = self.self_attn(q, k, v,
                                                  attn_mask=src_mask,
                                                  key_padding_mask=src_key_padding_mask,
                                                  need_weights=False)
        # But we want to apply RoPE before attention; so instead:
        B, S, _ = src.size()
        H = self.self_attn.num_heads
        D = self.self_attn.embed_dim // H
        q = src.view(B, S, H, D).transpose(1, 2)
        k = q.clone()
        v = q.clone()
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        q = q.transpose(1, 2).reshape(B, S, -1)
        k = k.transpose(1, 2).reshape(B, S, -1)
        attn_output, _ = self.self_attn(q, k, v.transpose(1,2).reshape(B, S, -1),
                                        attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask,
                                        need_weights=False)
        src2 = attn_output
        src = src + src2
        src = self.norm1(src)
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + ff
        src = self.norm2(src)
        return src

class SimpleRoPEEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = self.embedding(src)
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return x
