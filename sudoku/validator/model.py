import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SudokuTransformerConfig:
    vocab_size: int = 11                # tokens 0..10 inclusive (0=empty, 10=reserved)
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    norm_first: bool = True
    layer_norm_eps: float = 1e-5
    max_positions: int = 81                 # 9x9 Sudoku
    num_classes: int = 9                    # digits 1..9
    pad_token_id: Optional[int] = None      # not used; grid is always length 81
    nb_refinement_steps: int = 0 
    
class SudokuTransformer(nn.Module):
    """
    Encoder-only, bidirectional Transformer for Sudoku token classification.

    Input : input_ids LongTensor (B, 81) with values in [0..10].
    Output: logits FloatTensor (B, 81, 9) for classes 1..9 per cell.
    """
    def __init__(self, cfg: SudokuTransformerConfig = SudokuTransformerConfig()):
        super().__init__()
        self.cfg = cfg

        d = cfg.d_model

        # --- Embeddings ---
        self.token_embed = nn.Embedding(cfg.vocab_size, d)
        self.pos_embed = nn.Embedding(cfg.max_positions, d)

        # if cfg.use_structural_embeddings:
        #     self.row_embed = nn.Embedding(9, d)
        #     self.col_embed = nn.Embedding(9, d)
        #     self.box_embed = nn.Embedding(9, d)
        # else:
        #     self.row_embed = self.col_embed = self.box_embed = None

        self.embed_ln = nn.LayerNorm(d, eps=cfg.layer_norm_eps)
        self.embed_drop = nn.Dropout(cfg.dropout)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,         # (B, S, D)
            norm_first=cfg.norm_first
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
            norm=nn.LayerNorm(d, eps=cfg.layer_norm_eps)
        )

        # --- LM Head (token classification head) ---
        self.lm_head = nn.Linear(d, cfg.num_classes)
        self.confidence_head = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            # nn.Sigmoid()
        )

    def forward(
        self,
        input_ids: torch.LongTensor,          # (B, 81), tokens 0..10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert input_ids.dim() == 2 and input_ids.size(1) == self.cfg.max_positions, \
            f"expected (B, {self.cfg.max_positions}), got {tuple(input_ids.shape)}"

        B, S = input_ids.shape
        device = input_ids.device

        # --- Build embeddings ---
        tok = self.token_embed(input_ids)                             # (B, S, D)
        pos = self.pos_embed(torch.arange(S, device=device).unsqueeze(0).expand(B, S))  # (B, S, D)
        x = tok + pos

        x = self.embed_ln(x)
        x : torch.Tensor = self.embed_drop(x)
            
        hidden = torch.zeros_like(x)
        
        with torch.no_grad():
            if self.cfg.nb_refinement_steps:
                for _ in range(self.cfg.nb_refinement_steps):
                    hidden = self.encoder(
                        hidden + x.clone().detach(),
                        src_key_padding_mask=None,
                        is_causal=False
                    )  # (B, S, D)
        hidden = self.encoder(hidden + x, src_key_padding_mask=None, is_causal=False)  # (B, S, D)

        # --- Project to per-cell 9-class logits ---
        token_logits = self.lm_head(hidden)                                 # (B, S, 9)
        confidence_logits = self.confidence_head(hidden).squeeze(-1)        # (B, S,)
        return token_logits, confidence_logits


# ----------------
# Minimal usage
# ----------------
if __name__ == "__main__":
    cfg = SudokuTransformerConfig(
        d_model=256, nhead=8, num_layers=6, dim_feedforward=1024,
    )
    model = SudokuTransformer(cfg)

    B = 2
    # Example batch: tokens in [0..10]; 0 = empty, 1..9 digits, 10 = reserved token if you need it
    x = torch.randint(low=0, high=11, size=(B, 81))  # (B, 81)
    logits = model(x)                                 # (B, 81, 9)

    # # Example targets if you have ground-truth digits 0..9 (0 = empty)
    # ground_truth_digits = torch.randint(0, 10, (B, 81))
    # labels = SudokuTransformer.labels_from_digits(ground_truth_digits)  # (-100 or 0..8)

    # loss = F.cross_entropy(
    #     logits.view(-1, cfg.num_classes),
    #     labels.view(-1),
    #     ignore_index=-100
    # )
    # loss.backward()

    # print("Logits shape:", logits.shape)
    # print("Loss:", float(loss))
