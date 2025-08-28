# arc_encoder.py
# 4 spaces per indent, one big block.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'


# ----------------------------
# Config
# ----------------------------

@dataclass
class ARCEncoderConfig:
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


# ----------------------------
# Learnable 2-D Positional Embedding
# ----------------------------

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


# ----------------------------
# Token-Type Embedding
# ----------------------------

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

# ----------------------------
# Base Encoder (bidirectional)
# ----------------------------

class ARCEncoderModel(nn.Module):
    """
    Base encoder that returns hidden states only.

    Inputs (all tensors are [B, L]):
      input_ids: token ids (colors/specials)
      rows, cols: 2-D coords for grid cells; use -1 for specials (no pos)
      token_type_ids: QUESTION/ANSWER/SPECIAL
      attention_mask: 1 for tokens to keep, 0 for pad

    Embeddings summed:
      token_embed(input_ids) + pos2d(rows, cols) + tokentype(token_type_ids)
    """
    def __init__(self, cfg: ARCEncoderConfig):
        super().__init__()
        self.cfg = cfg

        d = cfg.d_model

        # Embeddings
        self.token_embed = nn.Embedding(cfg.vocab_size, d)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)

        self.pos2d = Learned2DPositionalEmbedding(cfg.max_height, cfg.max_width, d)
        self.ttype = TokenTypeEmbedding(cfg.num_token_types, d)

        self.embed_ln = nn.LayerNorm(d, eps=cfg.layer_norm_eps)
        self.embed_drop = nn.Dropout(cfg.dropout)

        # Transformer Encoder (bidirectional)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.nhead,
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
        rows: torch.Tensor,             # [B, L] (grid row or -1)
        cols: torch.Tensor,             # [B, L] (grid col or -1)
        token_type_ids: torch.Tensor,   # [B, L]
    ) -> torch.Tensor:
        # print(input_ids,rows, cols, token_type_ids)
        x_tok = self.token_embed(input_ids)                  # [B,L,D]
        # print(x_tok.shape, x_tok.dtype)
        x_pos = self.pos2d(rows, cols)                       # [B,L,D]
        # print(x_pos.shape, x_pos.dtype)
        x_typ = self.ttype(token_type_ids)                   # [B,L,D]
        # print(x_typ.shape, x_typ.dtype)

        x = x_tok + x_pos + x_typ
        # print('after embedding', x.shape, x.dtype)
        x = self.embed_ln(x)
        # print('after layer norm', x.shape)
        # x = self.embed_drop(x)

        h = self.encoder(x)  # [B,L,D]
        # print('after encoder', h.shape, h.dtype)
        return h


# ----------------------------
# Task Head: Masked Grid Filling (tied weights)
# ----------------------------

class ARCEncoderForMaskedSequenceModelling(nn.Module):
    """
    Adds a tied-weight LM head to predict the true token at positions marked "missing".
    If labels is provided, computes cross-entropy loss (ignore_index = -100).
    """
    def __init__(self, cfg: ARCEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ARCEncoderModel(cfg)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.encoder.token_embed.weight  # tie
        self.lm_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, L]
        rows: torch.Tensor,             # [B, L]
        cols: torch.Tensor,             # [B, L]
        token_type_ids: torch.Tensor,   # [B, L]
        labels: torch.Tensor,           # [B, L], -100 to ignore
    ) -> Dict[str, torch.Tensor]:
        h : torch.Tensor = self.encoder(input_ids, rows, cols, token_type_ids)  # [B,L,D]
        logits : torch.Tensor = self.lm_head(h) + self.lm_bias                                   # [B,L,V]
        # print(f'{logits.shape = }')
        # print(f'{input_ids.shape = }')
        # print(f'{labels.shape = }')
        # print(f'{h.shape = }')
        out = {"logits": logits, "hidden_states": h}
        loss = F.cross_entropy(
            logits.view(-1, self.cfg.vocab_size),
            labels.view(-1),
            ignore_index=self.cfg.ignore_id,
        )
        out["loss"] = loss
        return out




# # ----------------------------
# # Minimal example
# # ----------------------------

# if __name__ == "__main__":
#     # Dummy batch: two tiny grids (values must be valid token ids < vocab_size)
#     cfg : ARCEncoderConfig = ARCEncoderConfig(
#         d_model=256,
#         nhead=8,
#         num_layers=6,
#         dim_feedforward=1024,
#         vocab_size=20,            # adjust to your palette + specials
#         pad_token_id=0,
#         newline_token_id=1,
#         qa_sep_token_id=2,
#         example_sep_token_id=3,
#         masked_token_id=11,
#         max_height=40, max_width=40,
#     )

#     # Build random toy grids (question & answer), sizes <= 5x5 just for demo.
#     rng = np.random.default_rng(0)
#     q1 = rng.integers(low=4, high=10, size=(4, 5), dtype=np.int64)  # skip 0..3 reserved specials
#     a1 = rng.integers(low=4, high=10, size=(3, 4), dtype=np.int64)
#     q2 = rng.integers(low=4, high=10, size=(5, 5), dtype=np.int64)
#     a2 = rng.integers(low=4, high=10, size=(4, 3), dtype=np.int64)
#     batch_np: List[Tuple[np.ndarray, np.ndarray]] = [(q1, a1), (q2, a2)]
#     print(f'{batch_np = }')

#     packed = collate_arc_batch(batch_np, cfg, DEVICE)
#     print("Packed shapes:",
#           {k: tuple(v.shape) for k, v in packed.items()})
#     # Base encoder (hidden states only)
#     base = ARCEncoderModel(cfg).to(device=DEVICE)
#     h = base(**packed)  # [B, L, D]
#     print("Hidden states:", h.shape)
#     # exit()

#     # Task model: masked filling
#     model = ARCEncoderForMaskedSequenceModelling(cfg)
#     # out = model(**packed, labels=None)
#     # print("Logits:", out["logits"].shape)

#     # Example loss (toy): pretend we want to predict every position that equals 'missing_token_id'
#     # In practice, you'd set input grids with missing_token_id at holes and pass true target labels.
#     batch_np_masked = prepare_batch_for_masked_modeling(
#         batch_raw = batch_np,
#         masked_token_id=cfg.masked_token_id,
#         nb_masked_tokens=3,
#         mask_also_task=False
#     )
#     packed_masked = collate_arc_batch(batch_np_masked, cfg, DEVICE)
#     packed = collate_arc_batch(batch_np, cfg, DEVICE)
    
#     labels = packed["input_ids"].clone() # we want to konw ground truth tokens
#     labels[packed_masked["input_ids"] != cfg.masked_token_id] = -100  # mask out everything except masked positions
    
    
#     # packed["input_ids"] = input_ids
#     out2 = model(**packed_masked, labels=labels)
#     print("Loss:", float(out2["loss"]))
