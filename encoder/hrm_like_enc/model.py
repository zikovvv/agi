from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Literal, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder.hrm_like_enc.components import TransformerBlockHRM
from encoder.hrm_like_enc.config import EncoderConfig
from common import *


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
        self.seq_len = cfg.max_len

        # self.encoder = nn.Sequential(
        #     *[TransformerBlockHRM(self.cfg) for _ in range(self.cfg.num_layers)]
        # )
        
        self.blocks = nn.ModuleList(
            [TransformerBlockHRM(self.cfg) for _ in range(self.cfg.num_layers)]
        )

    def encoder(self, h : torch.Tensor) -> torch.Tensor:
        first_half_copy = h[:, :self.seq_len // 2].clone()
        if self.cfg.enable_pseudo_diffusion_inner :
            for block in self.blocks :
                h = h - block(h)
                if self.cfg.feed_first_half :
                    h[:, :first_half_copy.shape[1]] = first_half_copy
            return h
        else :
            hidden = torch.zeros_like(h)    
            for block in self.blocks :
                hidden = block(hidden + h)
            return hidden


    def forward(
        self,
        input_ids: torch.Tensor,        # [B, L]
    ) -> torch.Tensor:
        inps = self.token_embed(input_ids)  # [B,L,D]
        if self.cfg.use_emb_norm :
            inps = self.embed_ln(inps)
        if self.cfg.enable_pseudo_diffusion_outer :
            assert self.cfg.nb_refinement_steps > 0
            assert self.cfg.nb_last_trained_steps > 0
            assert self.cfg.nb_last_trained_steps <= self.cfg.nb_refinement_steps
            nb_steps_no_grad = self.cfg.nb_refinement_steps - self.cfg.nb_last_trained_steps
            h = inps
            first_half_copy = h[:, :self.seq_len // 2].clone()
            with torch.no_grad():
                for i in range(nb_steps_no_grad):
                    h = h - self.encoder(h)
                    if self.cfg.feed_first_half :
                        h[:, :first_half_copy.shape[1]] = first_half_copy
            for i in range(self.cfg.nb_last_trained_steps):
                h = h - self.encoder(h)
            return h
        else :                
            assert self.cfg.nb_refinement_steps > 0
            assert self.cfg.nb_last_trained_steps > 0
            assert self.cfg.nb_last_trained_steps <= self.cfg.nb_refinement_steps
            nb_steps_no_grad = self.cfg.nb_refinement_steps - self.cfg.nb_last_trained_steps
            h = torch.zeros_like(inps)
            with torch.no_grad():
                for i in range(nb_steps_no_grad):
                    h = self.encoder(h + inps)
            for i in range(self.cfg.nb_last_trained_steps):
                h = self.encoder(h + inps)
            return h


class EncoderForCLS(nn.Module):
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

class EncoderForValidation(nn.Module) :
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = EncoderModel(cfg)
        self.lm_head = nn.Linear(cfg.d_model, 2, bias=True)
        
    def forward(
        self,
        input_ids: torch.Tensor,        # [B, L]
        labels: torch.Tensor,           # [B, L], -100 to ignore, values are 1 or 0
    ) -> Dict[str, torch.Tensor]:
        log_debug(f'{input_ids.shape = }, {labels.shape = }, {input_ids.dtype = }, {labels.dtype = }')
        h : torch.Tensor = self.encoder(input_ids)  # [B,L,D]
        logits : torch.Tensor = self.lm_head(h)     # [B,L,V]
        out = {"logits": logits, "hidden_states": h}
        log_debug(f'{logits.shape = }, {labels.shape = }, {logits.dtype = }, {labels.dtype = }')
        
        
        # Optional sanity checks (helpful during dev)
        assert logits.dim() == 3 and logits.size(-1) == 2, f"Unexpected logits shape: {logits.shape}"
        assert labels.shape == logits.shape[:2], f"Labels {labels.shape} must match logits[:2] {logits.shape[:2]}"
        assert isinstance(self.cfg.ignore_id, int), f"ignore_id must be int, got {type(self.cfg.ignore_id)}"
        assert self.cfg.ignore_id == -100, f"Expected ignore_id -100, got {self.cfg.ignore_id}"

        loss = F.cross_entropy(
            logits.view(-1, 2),
            labels.view(-1),
            ignore_index=self.cfg.ignore_id,
        )
        out["loss"] = loss
        return out
