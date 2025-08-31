from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Literal, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder.hrm_cnn.components import TransformerBlockHRM
from encoder.hrm_cnn.config import EncoderConfig

class EncoderModel(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg

        d = cfg.d_model

        # # Embeddings
        # self.token_embed = nn.Embedding(cfg.vocab_size, d)
        # nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)

        self.embed_ln = nn.LayerNorm(d, eps=cfg.layer_norm_eps)
        self.embed_drop = nn.Dropout(cfg.dropout)
        self.seq_len = cfg.max_len

        self.encoder = nn.Sequential(
            *[TransformerBlockHRM(self.cfg) for _ in range(self.cfg.num_layers)]
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,        # [B, L]
    ) -> torch.Tensor:
        # print(input_ids.shape)
        # exit()
        # x = self.token_embed(input_ids)  # [B,L,D]
        # x = self.embed_ln(x)
        x = input_ids
        h = self.encoder(x)
        if self.cfg.nb_refinement_steps > 1:
            for i in range(self.cfg.nb_refinement_steps - 1):
                h = self.encoder(h)
        return h


class CNNFeatureExtractor(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)

        self.encoder = nn.Sequential(
            nn.Conv2d(cfg.d_model, cfg.d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cfg.d_model, cfg.d_model, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(cfg.d_model, cfg.d_model, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tok_embeddings(x)
        x = x.permute(0, 3, 1, 2)  # [B, D, H, W]
        x = self.encoder(x)
        return x

class EncoderForCLSWithCNNFeatureExtraction(nn.Module):
    """
    Adds a tied-weight LM head to predict the true token at positions marked "missing".
    If labels is provided, computes cross-entropy loss (ignore_index = -100).
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = CNNFeatureExtractor(cfg)
        self.encoder = EncoderModel(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)
        
    def forward(
        self,
        input_ids: torch.Tensor,        # [B, H, W]
        labels: torch.Tensor,           # [B, H, W], -100 to ignore
    ) -> Dict[str, torch.Tensor]:
        B, H, W, D = (*input_ids.size(), self.cfg.d_model)
        features : torch.Tensor = self.feature_extractor(input_ids)  # [B, D, H, W]
        features = features.permute(0, 2, 3, 1)  # [B, H, W, D]
        assert features.shape == (B, H, W, D)
        h = features.view(B, -1, D) # [B, H*W, D]
        assert h.shape == (B, H * W, D)

        h : torch.Tensor = self.encoder(h)  # [B,H*W,D]
        logits : torch.Tensor = self.lm_head(h)     # [B,H*W,V]
        assert logits.shape == (B, H * W, self.cfg.vocab_size)
        assert h.shape == (B, H * W, D)
        out = {"logits": logits, "hidden_states": h}
        loss = F.cross_entropy(
            logits.view(-1, self.cfg.vocab_size),
            labels.view(-1),
            ignore_index=self.cfg.ignore_id,
        )
        out["loss"] = loss
        h_map = h.view(B, H, W, D)
        out["hidden_states_map"] = h_map
        assert h_map.shape == (B, H, W, D)
        logits_map  = logits.view(B, H, W, self.cfg.vocab_size)
        assert logits_map.shape == (B, H, W, self.cfg.vocab_size)
        out["logits_map"] = logits_map
        return out

