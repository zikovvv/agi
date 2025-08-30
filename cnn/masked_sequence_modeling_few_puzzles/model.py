from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn.masked_sequence_modeling_few_puzzles.cfg import ARCCNNConfig
from cnn.masked_sequence_modeling_few_puzzles.components import RotaryEmbedding, TransformerBlockHRM
from common import *




class ConvAttBlock(nn.Module) :
    def __init__(
        self,
        cfg: ARCCNNConfig,
        rope_cos_sin: torch.Tensor,
        kernel_size: int,
        padding : int,
        use_transformer: bool = True,
        is_testing: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv2d(cfg.d_model, cfg.d_model, kernel_size=kernel_size, padding=padding)
        self.transformer = TransformerBlockHRM(cfg, rope_cos_sin)
        self.use_transformer = use_transformer
        self.is_testing = is_testing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D = x.shape
        x = x.permute(0, 3, 1, 2)  # [B, D, H, W]
        if self.is_testing:    
            assert x.shape == (B, D, H, W), f"Unexpected shape: {x.shape}"
        x = self.conv(x)
        if self.is_testing:
            assert x.shape == (B, D, H, W), f"Unexpected shape: {x.shape}"
        x = x.permute(0, 2, 3, 1)  # [B, H, W, D]
        if self.is_testing:
            assert x.shape == (B, H, W, D), f"Unexpected shape: {x.shape}"
        x_after_cnn = x.clone()
        if self.use_transformer :
            x_B_SEQLEN_D = x.view(x.size(0), -1, x.size(-1)).squeeze()  # [B, H*W, D]
            if self.is_testing:
                assert x_B_SEQLEN_D.shape == (B, H * W, D), f"Unexpected shape: {x_B_SEQLEN_D.shape}"
            
            
            x = self.transformer(x_B_SEQLEN_D)
            if self.is_testing:
                x = x_B_SEQLEN_D
                assert x.shape == (B, H * W, D), f"Unexpected shape: {x.shape}"
            x = x.view(B, H, W, D)  # [B, H, W, D]
            if self.is_testing:
            
                assert x.shape == (B, H, W, D), f"Unexpected shape: {x.shape}"
                assert x_after_cnn.isclose(x).all(), f"Output mismatch after CNN: {x_after_cnn} vs {x}"
        return x


class ARCCNNModel(nn.Module):
    def __init__(self, cfg: ARCCNNConfig):
        super().__init__()
        self.cfg = cfg
        self.rotary_emb = RotaryEmbedding(
            dim = self.cfg.d_model // self.cfg.n_head,
            max_position_embeddings = int((self.cfg.max_height * self.cfg.max_width) * 2.5),
            base = self.cfg.rope_theta
        )
        self.rope_cos_sin : torch.Tensor = self.rotary_emb()
        # print(self.rope_mat[0])
        # print(self.rope_mat[0].shape)
        
        # print(self.rope_mat[1])
        # print(self.rope_mat[1].shape)
        # # exit()
        d = cfg.d_model

        # Embeddings
        self.token_embed = nn.Embedding(cfg.vocab_size, d)

        self.encoder = nn.Sequential(
            ConvAttBlock(cfg, self.rope_cos_sin, kernel_size=3, padding=1),
            ConvAttBlock(cfg, self.rope_cos_sin, kernel_size=5, padding=2),
            ConvAttBlock(cfg, self.rope_cos_sin, kernel_size=7, padding=3),
        )

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, H, W]
    ) -> torch.Tensor:
        x : torch.Tensor = self.token_embed(input_ids)  # [B, H, W, D]
        log_debug('after embeddings', x.shape)
        # x = x.permute(0, 3, 1, 2)  # [B, D, H, W]
        # log_debug('after permutation', x.shape)
        h = self.encoder(x)
        return h
    
    
    

class ARCCNNForMaskedSequenceModelling(nn.Module):
    def __init__(self, cfg: ARCCNNConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ARCCNNModel(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)
        

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, H, W]
        labels: torch.Tensor,           # [B, H, W], -100 to ignore
    ) -> Dict[str, torch.Tensor]:
        h : torch.Tensor = self.encoder(input_ids)  # [B,D,H,W]
        # h = h.permute(0, 2, 3, 1) # [B, H, W, D]
        logits : torch.Tensor = self.lm_head(h) # [B, H, W, V]
        log_debug('in ARCCNNForMaskedSequenceModelling.forward after logits')
        log_debug(f'{logits.shape = }')
        log_debug(f'{input_ids.shape = }')
        log_debug(f'{labels.shape = }')
        log_debug(f'{h.shape = }')
        out = {"logits": logits, "hidden_states": h}
        loss = F.cross_entropy(
            logits.view(-1, self.cfg.vocab_size),
            labels.view(-1),
            ignore_index=self.cfg.ignore_id,
        )
        out["loss"] = loss
        return out


def ex1() :
    cfg = ARCCNNConfig(
        vocab_size=20,
    )
    model_encoder = ARCCNNModel(cfg)
    picture = torch.randint(1, 10, (16, 10, 10)) # [B, w, h]

    output = model_encoder(picture)
    log_debug(output.shape)


    model_cls = ARCCNNForMaskedSequenceModelling(cfg)
    picture_labels = picture.clone()
    output = model_cls(picture, picture_labels)
    log_debug(output['loss'])
    # print(output['logits'].shape)

if __name__ == "__main__":
    ex1()