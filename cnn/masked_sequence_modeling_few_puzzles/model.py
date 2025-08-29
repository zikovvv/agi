from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'

@dataclass
class ARCCNNConfig:
    # Model dims
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    norm_first: bool = True
    layer_norm_eps: float = 1e-5

    # Grid limits
    max_height: int = 40
    max_width: int = 40

    # Vocabulary
    vocab_size: int = 32  # colors + specials (you can raise this)
    pad_token_id: int = 15
    newline_token_id: int = 14
    qa_sep_token_id: int = 13
    example_sep_token_id: int = 12
    masked_token_id: int = 11  # the "hole" token to predict
    
    # how to mark something as ignored
    ignore_id: int = -100

    # Token types (learnable)
    # We keep it simple: QUESTION=0, ANSWER=1, SPECIAL=2
    num_token_types: int = 3
    token_type_question: int = 0
    token_type_answer: int = 1
    token_type_special: int = 2

    # Packing
    # If True, we prepend [EXAMPLE_SEP], add [NEWLINE] after each row,
    # and insert [Q_A_SEP] between question and answer.
    use_separators: bool = False

    # Weight tying for LM head
    tie_word_embeddings: bool = True

class Learned2DPositionalEmbedding(nn.Module):
    """
    Learnable 2-D positional embeddings over a HxW grid.
    Maps (row, col) -> R^D with a single embedding table of size (H*W, D).
    Special tokens (row == -1 or col == -1) get a zero vector.
    Shared between question and answer grids (you just call it on their coords).
    """
    def __init__(self, max_height: int, max_width: int, d_model: int, std: float = 0.02):
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.d_model = d_model

        self.emb = nn.Embedding(max_height * max_width, d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=std)

        # Precompute flat indices grid (H,W) -> idx
        rows = torch.arange(max_height).view(max_height, 1).expand(max_height, max_width)
        cols = torch.arange(max_width).view(1, max_width).expand(max_height, max_width)
        flat = rows * max_width + cols  # [H,W]
        self.register_buffer("_flat_idx", flat, persistent=False)  # not a parameter

    def forward(self, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        """
        rows, cols: [B, L] with -1 marking "no position" (special tokens).
        returns: [B, L, D]
        """
        # assert rows.shape == cols.shape
        B, L = rows.shape

        # Mask for specials (no pos)
        special_mask = (rows < 0) | (cols < 0)

        # asser that all valid positions are in range
        if not torch.all((rows[special_mask == 0] < self.max_height) &
                         (cols[special_mask == 0] < self.max_width)):
            raise ValueError("row/col values out of range")
        # print(torch.min(rows), torch.max(rows), torch.min(cols), torch.max(cols))
        flat_idx = (rows * self.max_width + cols).view(B, L)  # [B,L]
        flat_idx = flat_idx.clamp(min=0)  # so that -1 (specials) become 0; we'll mask them out later
        # print(torch.max(flat_idx), torch.min(flat_idx))
        pos = self.emb(flat_idx)  # [B,L,D]
        pos[special_mask] = 0.0
        # print(pos.shape)/
        # print(pos)
        # exit()
        return pos

class TokenTypeEmbedding(nn.Module):
    """
    Learnable token-type embeddings (e.g., QUESTION, ANSWER, SPECIAL).
    """
    def __init__(self, num_types: int, d_model: int, std: float = 0.02):
        super().__init__()
        self.emb = nn.Embedding(num_types, d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=std)

    def forward(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        skip_mask = token_type_ids < 0
        token_type_ids_filled = token_type_ids.masked_fill(skip_mask, 0)
        embs = self.emb(token_type_ids_filled)
        embs[skip_mask] = 0.0
        return embs

class ARCCNNModel(nn.Module):
    def __init__(self, cfg: ARCCNNConfig):
        super().__init__()
        self.cfg = cfg

        d = cfg.d_model

        # Embeddings
        self.token_embed = nn.Embedding(cfg.vocab_size, d)
        self.encoder = nn.Sequential(
            nn.Conv2d(d, d * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d * 2, d * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(d * 2, d, kernel_size=7, padding=3),
            nn.ReLU(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, w, h]
    ) -> torch.Tensor:
        x : torch.Tensor = self.token_embed(input_ids)  # [B, w, h, D]
        log_debug('after embeddings', x.shape)
        x = x.permute(0, 3, 1, 2)  # [B, D, w, h]
        log_debug('after permutation', x.shape)
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
        h = h.permute(0, 2, 3, 1) # [B, H, W, D]
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
        d_model=128,
        vocab_size=20,
    )
    model_encoder = ARCCNNModel(cfg)
    picture = torch.randint(1, 10, (16, 30, 30)) # [B, w, h]

    output = model_encoder(picture)
    log_debug(output.shape)


    model_cls = ARCCNNForMaskedSequenceModelling(cfg)
    picture_labels = picture.clone()
    output = model_cls(picture, picture_labels)
    log_debug(output['loss'])
    # print(output['logits'].shape)

if __name__ == "__main__":
    ex1()