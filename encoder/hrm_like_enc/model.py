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
from x_transformers import Encoder as XEncoder


class EncoderModel(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg : EncoderConfig = cfg

        d = cfg.d_model

        # Embeddings
        self.token_embed = nn.Embedding(cfg.vocab_size, d)
        self.pos_emb = nn.Embedding(cfg.nb_max_rope_positions, d)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        self.embed_ln = nn.LayerNorm(d, eps=cfg.layer_norm_eps)
        self.embed_drop = nn.Dropout(cfg.dropout)
        self.seq_len = cfg.nb_max_rope_positions

        
        if self.cfg.use_x_encoder :
            self.blocks = nn.ModuleList(
                [
                    XEncoder(
                        dim = self.cfg.d_model,
                        depth = self.cfg.num_layers,
                        heads = self.cfg.n_head,
                        causal = False,
                        cross_attend = False,
                        only_cross = False,
                        use_scalenorm = False,
                        use_rmsnorm = True,# default is False
                        use_dynamic_tanh = False,
                        dynamic_tanh_init_alpha = 1.,
                        use_simple_rmsnorm = False,
                        use_adaptive_layernorm = False,
                        use_adaptive_rmsnorm = False,
                        use_adaptive_layerscale = False, # paired with use_adaptive_layernorm for ada-ln-zero from DiT paper
                        norm_add_unit_offset = True,
                        dim_condition = None,
                        adaptive_condition_mlp = False,
                        adaptive_condition_mlp_expansion = 4,
                        alibi_pos_bias = False,
                        alibi_num_heads = None,
                        rel_pos_bias = False,
                        rel_pos_num_buckets = 32,
                        rel_pos_max_distance = 128,
                        dynamic_pos_bias = False,
                        dynamic_pos_bias_log_distance = False,
                        dynamic_pos_bias_mlp_depth = 2,
                        dynamic_pos_bias_norm = False,
                        rotary_pos_emb = False,
                        rotary_emb_dim = None,
                        rotary_xpos = False,
                        rotary_interpolation_factor = 1.,
                        rotary_xpos_scale_base = 512,
                        rotary_base_rescale_factor = 1.,
                        rotate_num_heads = None,
                        weight_tie_layers = False,
                        # custom_layers: tuple[str, ...] | None = None,
                        # layers_execute_order: tuple[int, ...] | None = None,
                        # sandwich_coef = None,
                        # par_ratio = None,
                        # residual_attn = False,
                        # cross_residual_attn = False,
                        # macaron = False,
                        pre_norm = True,
                        pre_norm_has_final_norm = True,
                        # gate_residual = False,
                        # scale_residual = False,
                        # scale_residual_constant = 1.,
                        # shift_tokens = 0,
                        # sandwich_norm = False,
                        # softclamp_output = False,
                        # softclamp_output_value = 30.,
                        # zero_init_branch_output = False,
                        # layer_dropout = 0.,
                        # cross_attn_tokens_dropout = 0.,
                        # disable_abs_pos_emb = None,
                        # use_layerscale = False,
                        # layerscale_init_value = 0.,
                        # unet_skips = False,
                        # integrate_layers = False,
                        # layer_integrate_use_softmax = True,
                        num_residual_streams = 1,
                        # qkv_receive_diff_residuals = False,
                        reinject_input = False,              # seen first in DEQ paper https://arxiv.org/abs/1909.01377, but later used in a number of papers trying to achieve depthwise generalization https://arxiv.org/abs/2410.03020v1
                        learned_reinject_input_gate = False,
                        add_value_residual = True,          # resformer from Zhou et al - https://arxiv.org/abs/2410.17897v1 - further corroboration by https://arxiv.org/abs/2412.15113 (faster emergence of ICL) - looks like this setting may becoming a necessity for every transformer soon
                        learned_value_residual_mix = True,   # seeing big improvements when the value residual mix value is learned per token - credit goes to @faresobeid for taking the first step with learned scalar mix, then @Blinkdl for taking it a step further with data dependent. here we will use per token learned
                        rel_pos_kwargs = dict(),
                    ) for _ in range(self.cfg.num_layers)
                ]
            )
        else :
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
            if self.cfg.init_hidden_state_to_zero :
                h = torch.zeros_like(inps)
            else :
                h = torch.randn_like(inps) * self.cfg.init_hidden_state_std
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
