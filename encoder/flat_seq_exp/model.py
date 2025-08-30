from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Literal, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder.flat_seq_exp.config import EncoderConfig

class EncoderModel(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg

        d = cfg.d_model

        # Embeddings
        self.token_embed = nn.Embedding(cfg.vocab_size, d)
        self.pos_emb = nn.Embedding(cfg.max_len, d)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        self.embed_ln = nn.LayerNorm(d, eps=cfg.layer_norm_eps)
        self.embed_drop = nn.Dropout(cfg.dropout)

        # Transformer Encoder (bidirectional)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_head,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=cfg.norm_first
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=cfg.num_layers,
            norm=nn.LayerNorm(d, eps=cfg.layer_norm_eps)
        )

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, L]
    ) -> torch.Tensor:
        # print(input_ids,rows, cols, token_type_ids)
        x_tok = self.token_embed(input_ids)                  # [B,L,D]
        # print(f'{x_tok.shape=}')
        # print(x_tok.shape, x_tok.dtype)
        if self.cfg.embeddings_type == 'learnable':
            x_pos = self.pos_emb(torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0))  # [1,L,D]
        else : 
            raise NotImplementedError("Only 'learnable' embeddings are implemented")
        x = x_tok + x_pos
        x = self.embed_ln(x)
        h = self.encoder(x)  # [B,L,D]
        return h


class EncoderForCLS(nn.Module):
    """
    Adds a tied-weight LM head to predict the true token at positions marked "missing".
    If labels is provided, computes cross-entropy loss (ignore_index = -100).
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = EncoderModel(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)
        
    def forward(
        self,
        input_ids: torch.Tensor,        # [B, L]
        labels: torch.Tensor,           # [B, L], -100 to ignore
    ) -> Dict[str, torch.Tensor]:
        h : torch.Tensor = self.encoder(input_ids)  # [B,L,D]
        logits : torch.Tensor = self.lm_head(h)     # [B,L,V]
        out = {"logits": logits, "hidden_states": h}
        loss = F.cross_entropy(
            logits.view(-1, self.cfg.vocab_size),
            labels.view(-1),
            ignore_index=self.cfg.ignore_id,
        )
        out["loss"] = loss
        return out

