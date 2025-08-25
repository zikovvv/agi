# file: build_dataset_and_train.py
from __future__ import annotations

import math
from dataclasses import dataclass
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


# ============================================================
# Config
# ============================================================

@dataclass
class DatasetConfig:
    n_missing: int = 40       # exactly this many blanks per puzzle
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
# Model: small CNN baseline producing 9 logits per cell
# (You can swap this with your own MLP/Transformer/CNN that maps (B,9,9,9)->(B,9,9,9) logits.)
# Conv2d uses NCHW: (batch, channels, height, width).
# ============================================================

class SudokuCNN(nn.Module):
    def __init__(self, channels: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, channels, 1)  # logits over 9 classes per cell
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B,9,9,9) logits


# ============================================================
# Train / Eval
# ============================================================

INPUT_VOCAB = [i for i in range(0, 10 + 1)] # 0 for empty field, 10 for thinking token, 1 - 9 numbers, total of 11 entries
OUTPUT_VOCAB = [i for i in range(1, 9 + 1)] # 1 - 9 numbers, total of 9 entries



def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    # loss_fn: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: str = "cpu") -> Dict[str, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    correct_new = 0
    total_new = 0
    total = 0

    for x, t in loader:
        # x.size = (B, 9, 9)
        # t.size = (B, 9, 9)
        x : torch.Tensor = x.to(device)
        t : torch.Tensor = t.to(device)
        # t_onehot : torch.Tensor = F.one_hot(t.to(device).long() - 1, num_classes=9)#.permute(0, 3, 1, 2)
        # print(x.size(), t.size(), t_onehot.size())
        # for B in range(len(x)) :
        #     for X in range(9) : 
        #         for Y in range(9) :
        #             if t_onehot[B, X, Y].sum() != 1:
        #                 print(f"Empty cell found at ({X}, {Y})")
        #             n = t_onehot[B, X, Y].argmax() + 1
        #             assert n == t[B, X, Y].item(), f"Expected {t[B, X, Y].item()} but got {n} at ({X}, {Y})"

        # exit()
        
        x_flat : torch.Tensor = x.view(x.size(0), -1).long() # (B, 81)
        t_flat : torch.Tensor = t.view(t.size(0), -1).long() # (B, 81)
        t_onehot_flat : torch.Tensor = F.one_hot(t_flat.to(device).long() - 1, num_classes=9).to(torch.float32) # (B, 81, 9) 
        # print(x_flat.size(), t_flat.size(), t_onehot_flat.size())

        optimizer.zero_grad(set_to_none=True)
        logits : torch.Tensor = model(x_flat)

        loss = loss_fn(logits, t_onehot_flat)
        pred = logits.argmax(dim=-1) + 1 # sudoku is from 1 to 9
        # print(logits.size(), pred.size(), t_flat.size())
        correct += (pred == t_flat).sum().item()
        total += t_flat.numel()
        
        # only check fields that are 0 in x_flat
        correct_new += ((x_flat == 0) * (pred == t_flat)).sum().item()
        total_new += (x_flat == 0).sum().item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(loader.dataset)
    acc = correct / max(total, 1)
    acc_new = correct_new / max(total_new, 1)
    return {"loss": avg_loss, "acc": acc, "acc_new": acc_new}


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: str = "cpu") -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    full_grid_correct = 0
    n_batches = 0
    correct_new = 0
    total_new = 0

    loss_fn = nn.CrossEntropyLoss()
    for x, t in loader:
        x : torch.Tensor = x.to(device)
        t : torch.Tensor = t.to(device)
        x_flat : torch.Tensor = x.view(x.size(0), -1).long() # (B, 81)
        t_flat : torch.Tensor = t.view(t.size(0), -1).long() # (B, 81)
        t_onehot_flat : torch.Tensor = F.one_hot(t_flat.to(device).long() - 1, num_classes=9).to(torch.float32) # (B, 81, 9) 


        with torch.no_grad():
            logits : torch.Tensor = model(x_flat)


        loss = loss_fn(logits, t_onehot_flat)
        pred = logits.argmax(dim=-1) + 1 # sudoku is from 1 to 9
        correct += (pred == t_flat).sum().item()
        total += t_flat.numel()

        full_grid_correct += (pred == t_flat).flatten().sum().item() == 81

        total_loss += loss.item() * x.size(0)
        n_batches += 1
        
        correct_new += ((x_flat == 0) * (pred == t_flat)).sum().item()
        total_new += (x_flat == 0).sum().item()


    avg_loss = total_loss / len(loader.dataset)
    token_acc = correct / max(total, 1)
    grid_acc = full_grid_correct / (len(loader.dataset))
    acc_new = correct_new / max(total_new, 1)

    return {"loss": avg_loss, "token_acc": token_acc, "grid_acc": grid_acc, "acc_new": acc_new}


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
    ds = e.generate_dataset(dcfg.num_samples, dcfg.n_missing, seed=dcfg.seed)
    
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
        va = evaluate(model, val_loader, device=tcfg.device)

        print(f"Epoch {epoch:02d} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} acc_new {tr['acc_new']:.4f} "
              f"| val loss {va['loss']:.4f} token_acc {va['token_acc']:.4f} grid_acc {va['grid_acc']:.4f} acc_new {va['acc_new']:.4f}")


    print("Done.")


if __name__ == "__main__":
    dcfg = DatasetConfig(n_missing=40, num_samples=600, seed=123, val_frac=0.1, test_frac=0.0)
    tcfg = TrainConfig(batch_size=16, lr=3e-4)
    mcfg = SudokuTransformerConfig(
        d_model=256, nhead=8, num_layers=8, dim_feedforward=1024,
    )
    
    main(
        dcfg = dcfg,
        tcfg = tcfg,
        mcfg = mcfg
    )
