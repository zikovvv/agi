# file: build_dataset_and_train.py
from __future__ import annotations

import math
from dataclasses import dataclass
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================
# Import your Sudoku engine (categorical features)
# Expected API (from previous script):
#   generate_dataset(n_samples: int, n_missing: int, seed: Optional[int]) -> SudokuDataset
#   where SudokuDataset has fields: puzzles (N,9,9) uint8 [0..9], solutions (N,9,9) uint8 [1..9],
#                                   x_onehot (N,81,9) uint8 one-hot, y_onehot (N,81,9) uint8 one-hot
# ============================================================
import engine as e  # <- make sure the filename matches
from torch.functional import F

from model import SudokuTransformer, SudokuTransformerConfig
from common import *

# ============================================================
# Config
# ============================================================

@dataclass
class DatasetConfig:
    n_missing: int = 25
    n_missing_max: Optional[int] = 40
    num_samples: int = 600    # dataset size
    seed: int = 123
    val_frac: float = 0.1
    test_frac: float = 0.0    # set >0 for a held-out test split


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    loss_type: str = "ce"  # 'ce' (recommended) or 'bce'
    # CrossEntropyLoss expects logits over classes with target as class indices 0..C-1 (not one-hot).  # docs: pytorch  # noqa


# ============================================================
# Torch Dataset wrapper
# ============================================================

class SudokuTorchDataset(Dataset):
    def __init__(self,
                 puzzles: np.ndarray,       # (N,9,9) uint8 [0..9]
                 solutions: np.ndarray,     # (N,9,9) uint8 [1..9]
    ):
        assert solutions.ndim == 3 and solutions.shape[1:] == (9, 9)
        self.puzzles = puzzles.astype(np.uint8)
        self.solutions = solutions.astype(np.uint8)

    def __len__(self) -> int:
        return self.solutions.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.puzzles[idx], self.solutions[idx]

# ============================================================
# Train / Eval
# ============================================================

INPUT_VOCAB = [i for i in range(0, 10 + 1)] # 0 for empty field, 10 for thinking token, 1 - 9 numbers, total of 11 entries
OUTPUT_VOCAB = [i for i in range(1, 9 + 1)] # 1 - 9 numbers, total of 9 entries



def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: str = "cpu") -> Dict[str, float]:
    model.train()
    # loss_fn = nn.CrossEntropyLoss()
    total_comb_loss = 0.0
    total_cls_loss = 0.0
    total_certainty_loss = 0.0
    correct = 0
    # correct_new = 0
    # total_new = 0
    total = 0

    for x, t in loader:
        # x.size = (B, 9, 9)
        # t.size = (B, 9, 9)
        x : torch.Tensor = x.to(device)
        t : torch.Tensor = t.to(device)        
        x_flat : torch.Tensor = x.view(x.size(0), -1).long() # (B, 81)
        t_flat : torch.Tensor = t.view(t.size(0), -1).long() # (B, 81)
        t_onehot_flat : torch.Tensor = F.one_hot(t_flat.to(device).long() - 1, num_classes=9).to(torch.float32) # (B, 81, 9) 

        optimizer.zero_grad(set_to_none=True)
        logits, certainty_logits = model(x_flat)
        logits : torch.Tensor
        certainty_logits : torch.Tensor
        log_debug(f'{logits.shape=}, {certainty_logits.shape=}')


        visible_cells = (x_flat != 0).to(torch.float32)
        large_neg = -1e6
        mask = visible_cells * large_neg
        certainty_logits += mask
        most_certain_idx = certainty_logits.argmax(dim=-1)
        log_debug(f'{most_certain_idx.shape=}, {most_certain_idx=}')
        
        
        row_idx = torch.arange(x.shape[0], device=logits.device)
        cls_logits_chosen = logits[row_idx, most_certain_idx]            # (B, 9)
        cls_onehot_chosen = t_onehot_flat[row_idx, most_certain_idx]       # (B,)
        log_debug(f'{cls_logits_chosen.shape=}, {cls_logits_chosen=}')
        log_debug(f'{cls_onehot_chosen.shape=}, {cls_onehot_chosen=}')
        cls_chosen_pred = cls_logits_chosen.argmax(dim=-1) + 1
        log_debug(f'{cls_chosen_pred.shape=}, {cls_chosen_pred=}')
        cls_chosen_truth = cls_onehot_chosen.argmax(dim=-1) + 1
        log_debug(f'{cls_chosen_truth.shape=}, {cls_chosen_truth=}')
        loss_cls_logits_chosen = nn.CrossEntropyLoss()(cls_logits_chosen, cls_onehot_chosen)
        log_debug(f'{loss_cls_logits_chosen.shape=}, {loss_cls_logits_chosen=}')
        certainty_result = (cls_chosen_pred == cls_chosen_truth).to(torch.float32)
        log_debug(f'{certainty_result.shape=}, {certainty_result=}')
        certainty_loss = nn.BCEWithLogitsLoss()(certainty_logits[row_idx, most_certain_idx], certainty_result)
        log_debug(f'{certainty_loss.shape=}, {certainty_loss=}')
        combined_loss = loss_cls_logits_chosen + certainty_loss
        log_debug(f'{combined_loss.shape=}, {combined_loss=}')
        
        combined_loss.backward()
        optimizer.step()

        total_comb_loss += combined_loss.item() * x.size(0)
        total_cls_loss += loss_cls_logits_chosen.item() * x.size(0)
        total_certainty_loss += certainty_loss.item() * x.size(0)


        correct += (cls_chosen_pred == cls_chosen_truth).sum().item()

        total += x.shape[0]



    avg_comb_loss = total_comb_loss / len(loader.dataset)
    avg_cls_loss = total_cls_loss / len(loader.dataset)
    avg_certainty_loss = total_certainty_loss / len(loader.dataset)
    acc = correct / max(total, 1)

    return {"comb_loss": avg_comb_loss, "cls_loss": avg_cls_loss, "certainty_loss": avg_certainty_loss, "acc": acc}

@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: str = "cpu") -> Dict[str, float]:
    model.eval()
    # loss_fn = nn.CrossEntropyLoss()
    total_comb_loss = 0.0
    total_cls_loss = 0.0
    total_certainty_loss = 0.0
    correct = 0
    # correct_new = 0
    # total_new = 0
    total = 0

    for x, t in loader:
        # x.size = (B, 9, 9)
        # t.size = (B, 9, 9)
        x : torch.Tensor = x.to(device)
        t : torch.Tensor = t.to(device)        
        x_flat : torch.Tensor = x.view(x.size(0), -1).long() # (B, 81)
        t_flat : torch.Tensor = t.view(t.size(0), -1).long() # (B, 81)
        t_onehot_flat : torch.Tensor = F.one_hot(t_flat.to(device).long() - 1, num_classes=9).to(torch.float32) # (B, 81, 9) 

        # optimizer.zero_grad(set_to_none=True)
        logits, certainty_logits = model(x_flat)
        logits : torch.Tensor
        certainty_logits : torch.Tensor
        log_debug(f'{logits.shape=}, {certainty_logits.shape=}')


        visible_cells = (x_flat != 0).to(torch.float32)
        large_neg = -1e6
        mask = visible_cells * large_neg
        certainty_logits += mask
        most_certain_idx = certainty_logits.argmax(dim=-1)
        log_debug(f'{most_certain_idx.shape=}, {most_certain_idx=}')
        
        
        row_idx = torch.arange(x.shape[0], device=logits.device)
        cls_logits_chosen = logits[row_idx, most_certain_idx]            # (B, 9)
        cls_onehot_chosen = t_onehot_flat[row_idx, most_certain_idx]       # (B,)
        log_debug(f'{cls_logits_chosen.shape=}, {cls_logits_chosen=}')
        log_debug(f'{cls_onehot_chosen.shape=}, {cls_onehot_chosen=}')
        cls_chosen_pred = cls_logits_chosen.argmax(dim=-1) + 1
        log_debug(f'{cls_chosen_pred.shape=}, {cls_chosen_pred=}')
        cls_chosen_truth = cls_onehot_chosen.argmax(dim=-1) + 1
        log_debug(f'{cls_chosen_truth.shape=}, {cls_chosen_truth=}')
        loss_cls_logits_chosen = nn.CrossEntropyLoss()(cls_logits_chosen, cls_onehot_chosen)
        log_debug(f'{loss_cls_logits_chosen.shape=}, {loss_cls_logits_chosen=}')
        certainty_result = (cls_chosen_pred == cls_chosen_truth).to(torch.float32)
        log_debug(f'{certainty_result.shape=}, {certainty_result=}')
        certainty_loss = nn.BCEWithLogitsLoss()(certainty_logits[row_idx, most_certain_idx], certainty_result)
        log_debug(f'{certainty_loss.shape=}, {certainty_loss=}')
        combined_loss = loss_cls_logits_chosen + certainty_loss
        log_debug(f'{combined_loss.shape=}, {combined_loss=}')
        
        # combined_loss.backward()
        # optimizer.step()

        total_comb_loss += combined_loss.item() * x.size(0)
        total_cls_loss += loss_cls_logits_chosen.item() * x.size(0)
        total_certainty_loss += certainty_loss.item() * x.size(0)


        correct += (cls_chosen_pred == cls_chosen_truth).sum().item()

        total += x.shape[0]



    avg_comb_loss = total_comb_loss / len(loader.dataset)
    avg_cls_loss = total_cls_loss / len(loader.dataset)
    avg_certainty_loss = total_certainty_loss / len(loader.dataset)
    acc = correct / max(total, 1)

    return {"comb_loss": avg_comb_loss, "cls_loss": avg_cls_loss, "certainty_loss": avg_certainty_loss, "acc": acc}

# ============================================================
# Main
# ============================================================

import torch
from torch import nn


def main(
    dcfg: DatasetConfig,
    tcfg: TrainConfig,
    mcfg: SudokuTransformerConfig
):

    
    print("Generating Sudoku dataset...")
    ds = e.generate_dataset(dcfg.num_samples, dcfg.n_missing, dcfg.n_missing_max, seed=dcfg.seed)

    print(ds.puzzles.shape, ds.solutions.shape)

    # 2) Wrap into Torch Dataset
    print(f"Using device: {tcfg.device}")
    # use_onehot_targets = (tcfg.loss_type.lower() == "bce")
    full = SudokuTorchDataset(ds.puzzles, ds.solutions)

    # 3) Split
    N = len(full)
    n_val = int(round(dcfg.val_frac * N))
    n_test = int(round(dcfg.test_frac * N))
    n_train = N - n_val - n_test
    gen = torch.Generator().manual_seed(dcfg.seed)
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=gen)  # returns Subset
    print(f"Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # 4) Loaders
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, pin_memory=True)
    # test_loader = DataLoader(test_ds, batch_size=tcfg.batch_size, shuffle=False, pin_memory=True)

    # 5) Model

    model = SudokuTransformer(mcfg).to(tcfg.device)

    # 6) Loss & Optimizer
    # loss_fn = make_loss(tcfg.loss_type).to(tcfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr)

    # 7) Train loop
    best_val = math.inf
    for epoch in range(1, tcfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device=tcfg.device)
        l  = f"Epoch {epoch:02d} | "
        for k, v in tr.items() : 
            l += f"{k} {v:.4f} | "
        print(f"{l}")
        
        l  = f"Epoch {epoch:02d} | "
        with torch.no_grad():
            va = evaluate(model, val_loader, device=tcfg.device)
        for k, v in va.items() : 
            l += f"{k} {v:.4f} | "
        print(f"{l}")

    print("Done.")


if __name__ == "__main__":
    logger.remove() #remove the old handler. Else, the old one will work along with the new one you've added below'
    logger.add(sys.stderr, level="INFO") 
    
    dcfg = DatasetConfig(n_missing=25, num_samples=600, seed=123, val_frac=0.1, test_frac=0.0, n_missing_max=40)
    tcfg = TrainConfig(batch_size=16, lr=3e-4)
    mcfg = SudokuTransformerConfig(
        d_model=256, nhead=8, num_layers=8, dim_feedforward=1024,
    )
    
    main(
        dcfg = dcfg,
        tcfg = tcfg,
        mcfg = mcfg
    )
