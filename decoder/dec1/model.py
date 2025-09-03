from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Literal, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder.dec1.components import DecoderBlock, TransformerBlockHRM
from decoder.dec1.config import ModelConfig


# class EncoderModel(nn.Module):
    #     def __init__(self, cfg: EncoderConfig):
    #         super().__init__()
    #         self.cfg = cfg

    #         d = cfg.d_model

    #         # Embeddings
    #         self.token_embed = nn.Embedding(cfg.vocab_size, d)
    #         self.pos_emb = nn.Embedding(cfg.max_len, d)
    #         nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
    #         nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    #         self.embed_ln = nn.LayerNorm(d, eps=cfg.layer_norm_eps)
    #         self.embed_drop = nn.Dropout(cfg.dropout)
    #         self.seq_len = cfg.max_len

    #         # self.encoder = nn.Sequential(
    #         #     *[TransformerBlockHRM(self.cfg) for _ in range(self.cfg.num_layers)]
    #         # )
            
    #         self.blocks = nn.ModuleList(
    #             [TransformerBlockHRM(self.cfg) for _ in range(self.cfg.num_layers)]
    #         )

    #     def encoder(self, h : torch.Tensor) -> torch.Tensor:
    #         first_half_copy = h[:, :self.seq_len // 2].clone()
    #         if self.cfg.enable_pseudo_diffusion_inner :
    #             for block in self.blocks :
    #                 h = h - block(h)
    #                 if self.cfg.feed_first_half :
    #                     h[:, :first_half_copy.shape[1]] = first_half_copy
    #             return h
    #         else :    
    #             for block in self.blocks :
    #                 h = block(h)
    #             return h


    #     def forward(
    #         self,
    #         input_ids: torch.Tensor,        # [B, L]
    #     ) -> torch.Tensor:
    #         inps = self.token_embed(input_ids)  # [B,L,D]
    #         h = h - self.encoder(h)
            
    #         if self.cfg.enable_pseudo_diffusion_outer :
    #             assert self.cfg.nb_refinement_steps > 0
    #             assert self.cfg.nb_last_trained_steps > 0
    #             assert self.cfg.nb_last_trained_steps <= self.cfg.nb_refinement_steps
    #             nb_steps_no_grad = self.cfg.nb_refinement_steps - self.cfg.nb_last_trained_steps
    #             h = inps
    #             first_half_copy = h[:, :self.seq_len // 2].clone()
    #             with torch.no_grad():
    #                 for i in range(nb_steps_no_grad):
    #                     h = h - self.encoder(h)
    #                     if self.cfg.feed_first_half :
    #                         h[:, :first_half_copy.shape[1]] = first_half_copy
    #             for i in range(self.cfg.nb_last_trained_steps):
    #                 h = h - self.encoder(h)
    #             return h
    #         else :                
    #             assert self.cfg.nb_refinement_steps > 0
    #             assert self.cfg.nb_last_trained_steps > 0
    #             assert self.cfg.nb_last_trained_steps <= self.cfg.nb_refinement_steps
    #             nb_steps_no_grad = self.cfg.nb_refinement_steps - self.cfg.nb_last_trained_steps
    #             h = torch.zeros_like(inps)
    #             with torch.no_grad():
    #                 for i in range(nb_steps_no_grad):
    #                     h = self.encoder(h + inps)
    #             for i in range(self.cfg.nb_last_trained_steps):
    #                 h = self.encoder(h + inps)
    #             return h


class DecoderModel(nn.Module) :
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            [DecoderBlock(self.cfg) for _ in range(self.cfg.num_layers)]
        )
    
    def forward(
        self,
        h: torch.Tensor,        # [B, L, D]
    ) -> torch.Tensor:
        for block in self.blocks:
            h = block(h)
        return h



class ARCPureDecoder(nn.Module):
    """
    Adds a tied-weight LM head to predict the true token at positions marked "missing".
    If labels is provided, computes cross-entropy loss (ignore_index = -100).
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.decoder = DecoderModel(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)
        
    def forward(
        self,
        input_ids: torch.Tensor,      # [B, L]
        labels: torch.Tensor,       # [B, L]
    ) -> Dict[str, torch.Tensor]:
        embeds = self.decoder.token_embed(input_ids)
        h : torch.Tensor = self.decoder(embeds)  # [B,C,D]
        logits : torch.Tensor = self.lm_head(h)     # [B,C,V]
        logits_shifted = logits[:, :-1, :].contiguous()
        input_ids_shifted = input_ids[:, 1:].contiguous()
        out = {"logits": logits, "hidden_states": h}
        loss = F.cross_entropy(
            logits_shifted.view(-1, self.cfg.vocab_size),
            input_ids_shifted.view(-1),
            ignore_index=self.cfg.ignore_id,
        )
        out["loss"] = loss
        return out
