from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Literal, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder.enc_dec.components import TransformerBlockHRM
from encoder.enc_dec.config import EncoderConfig


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
            for block in self.blocks :
                h = block(h)
            return h


    def forward(
        self,
        inps: torch.Tensor,
    ) -> torch.Tensor:
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



class EncoderDecoder(nn.Module):
    def __init__(self, cfg_enc: EncoderConfig, cfg_dec: EncoderConfig):
        super().__init__()
        self.cfg = cfg_enc # for compatibility
        self.cfg_enc = cfg_enc
        self.cfg_dec = cfg_dec
        self.encoder = EncoderModel(cfg_enc)
        self.decoder = EncoderModel(cfg_dec)
        self.lm_head = nn.Linear(cfg_dec.d_model, cfg_dec.vocab_size, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, L]
        labels: torch.Tensor,           # [B, L], -100 to ignore
    ) : 
        B = input_ids.shape[0]
        V = self.cfg_dec.vocab_size
        D = self.cfg_dec.d_model
        L = input_ids.shape[1]
        I = self.cfg_enc.nb_info_tokens
        # print(f'{B = }, {L = }, {D = }, {V = }, {I = }')
        learning_tokens_ids = torch.full(
            (B, I),
            self.cfg_enc.info_token_id,
            dtype=torch.long,
            device=input_ids.device
        ) # [B, I]
        assert learning_tokens_ids.shape == (B, I)
        inputs_with_info_token_ids = torch.cat([learning_tokens_ids, input_ids], dim=1) # [B, I+L]
        assert inputs_with_info_token_ids.shape == (B, I + L)
        inputs_embedded = self.encoder.token_embed(inputs_with_info_token_ids) # [B, I+L, D]
        assert inputs_embedded.shape == (B, I + L, D)
        h_enc : torch.Tensor = self.encoder(inputs_embedded) # [B, I+L, D]
        assert h_enc.shape == (B, I + L, D)
        info_token_embeddings = h_enc[:, :I, :] # [B, I, D]
        assert info_token_embeddings.shape == (B, I, D)
        masked_tokens : torch.Tensor = torch.full(
            (B, L),
            self.cfg_dec.masked_token_id,
            dtype=torch.long,
            device=input_ids.device
        ) # [B, L]
        assert masked_tokens.shape == (B, L)
        masked_tokens_embedded = self.decoder.token_embed(masked_tokens) # [B, L, D]
        assert masked_tokens_embedded.shape == (B, L, D)
        concat = torch.cat([info_token_embeddings, masked_tokens_embedded], dim=1) # [B, I+L, D]
        assert concat.shape == (B, I + L, D)
        h_dec : torch.Tensor = self.decoder(concat) # [B, I+L, D]
        assert h_dec.shape == (B, I + L, D)
        h_dec_strp = h_dec[:, I:, :] # [B, L, D]
        assert h_dec_strp.shape == (B, L, D)
        logits : torch.Tensor = self.lm_head(h_dec_strp)     # [B,L,V]
        assert logits.shape == (B, L, V)
        # preds : torch.Tensor = logits.argmax(dim=-1) # [B, L]
        # assert preds.shape == (B, L)
        loss = F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            labels.reshape(-1),
            ignore_index=self.cfg_dec.ignore_id,
        )
        return {'loss' : loss, 'logits' : logits} 
