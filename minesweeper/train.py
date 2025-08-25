# file: build_dataset_and_train.py
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

# ---- import your engine (the previous script you saved) ----
# Expected symbols: GameState, generate_episode
import engine as e

from models.mlp import MLPModel
# or
from models.transformer import TransformerModel
# or
from models.cnn import CNNModel


# ------------------------------
# Config & helpers
# ------------------------------
HIDDEN_VALUE = -1  # mask for invisible cells in input

@dataclass
class DatasetConfig:
    height: int = 10
    width: int = 14
    n_bombs: int = 20
    num_games: int = 500
    seed: int = 42
    val_frac: float = 0.1
    test_frac: float = 0.0  # set >0 if you want a held-out test split
    use_frontier_policy: bool = True  # engine already does frontier-first in your version


def mask_numbers(numbers: np.ndarray, visible: np.ndarray, hidden_value: int = HIDDEN_VALUE) -> np.ndarray:
    """
    numbers: (H, W) uint8 0..8
    visible: (H, W) bool
    returns float32 masked numbers with hidden_value in hidden positions
    """
    masked = numbers.astype(np.int16).copy()
    masked[~visible] = hidden_value
    return masked.astype(np.float32)


def collect_dataset(cfg: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build dataset by generating episodes and collecting (masked_numbers, bombs) pairs
    for every intermediate snapshot in each episode.
    Returns:
        X: (N, H, W) float32   masked numbers, invisible=-1
        y: (N, H, W) float32   bombs in {0,1}
    """
    rng = np.random.default_rng(cfg.seed)
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for i in range(cfg.num_games):
        seed_i = int(cfg.seed + i)
        episode = e.generate_episode(cfg.height, cfg.width, cfg.n_bombs, seed=seed_i)
        for state in episode:
            x = mask_numbers(state.numbers, state.visible, hidden_value=HIDDEN_VALUE)
            y = state.bombs.astype(np.float32)
            X_list.append(x)
            y_list.append(y)

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, H, W)
    y = np.stack(y_list, axis=0).astype(np.float32)  # (N, H, W)
    return X, y


class MinesweeperTorchDataset(Dataset):
    """
    Wraps NumPy arrays into a Torch Dataset.
    - Inputs are (1, H, W) float tensors (masked numbers; invisible = -1).
    - Targets are (1, H, W) float tensors in {0,1}.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3 and y.ndim == 3
        assert X.shape == y.shape
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).unsqueeze(0).float()  # (1,H,W)
        t = torch.from_numpy(self.y[idx]).unsqueeze(0).float()  # (1,H,W)
        return x, t


def train_val_test_split(X: np.ndarray, y: np.ndarray, val_frac: float, test_frac: float, seed: int):
    ds = MinesweeperTorchDataset(X, y)
    N = len(ds)
    n_val = int(round(val_frac * N))
    n_test = int(round(test_frac * N))
    n_train = N - n_val - n_test
    gen = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val, n_test], generator=gen)


def estimate_pos_weight(dataset: Dataset) -> torch.Tensor:
    """
    Compute pos_weight = (#neg / #pos) over all pixels in the (training) dataset.
    Suitable for nn.BCEWithLogitsLoss to mitigate class imbalance. :contentReference[oaicite:1]{index=1}
    """
    pos = 0
    total = 0
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    for x, t in loader:
        pos += t.sum().item()
        total += t.numel()
    pos = max(pos, 1e-8)
    neg = total - pos
    pw = neg / pos
    return torch.tensor([pw], dtype=torch.float32)  # shape (1,) for single-channel


# ------------------------------
# Training / evaluation loop
# ------------------------------
@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    from_logits: bool = True  # If True, expect logits and use BCEWithLogitsLoss; else BCELoss (probabilities)
    use_pos_weight: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def make_loss(from_logits: bool, pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    """
    - If you make your model return logits, prefer BCEWithLogitsLoss (better numerical stability). :contentReference[oaicite:2]{index=2}
    - If your model returns probabilities in [0,1], use BCELoss. :contentReference[oaicite:3]{index=3}
    """
    if from_logits:
        if pos_weight is not None:
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return nn.BCEWithLogitsLoss()
    else:
        return nn.BCELoss()


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    loss_fn: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: str = "cpu") -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for x, t in loader:
        print(x)
        exit()
        x = x.to(device)
        t = t.to(device)

        optimizer.zero_grad(set_to_none=True)
        y = model(x)  # shape (B,1,H,W); logits if from_logits else probabilities

        loss = loss_fn(y, t)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(y) if isinstance(loss_fn, nn.BCEWithLogitsLoss) else y
            pred = (probs >= 0.5).float()
            correct = (pred == t).sum().item()
            total = t.numel()

        total_loss += loss.item() * x.size(0)
        total_acc += correct
        total_n += total

    return {"loss": total_loss / (len(loader.dataset)),
            "acc": total_acc / total_n}


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             loss_fn: nn.Module,
             device: str = "cpu") -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for x, t in loader:
        x = x.to(device)
        t = t.to(device)
        y = model(x)
        loss = loss_fn(y, t)

        probs = torch.sigmoid(y) if isinstance(loss_fn, nn.BCEWithLogitsLoss) else y
        pred = (probs >= 0.5).float()
        correct = (pred == t).sum().item()
        total = t.numel()

        total_loss += loss.item() * x.size(0)
        total_acc += correct
        total_n += total

    return {"loss": total_loss / (len(loader.dataset)),
            "acc": total_acc / total_n}


# ------------------------------
# Model placeholder
# ------------------------------
class YourModel(nn.Module):
    """
    Replace with your own architecture.
    Must map (B,1,H,W) -> (B,1,H,W), returning LOGITS if TrainConfig.from_logits=True,
    or PROBABILITIES if from_logits=False.
    """
    def __init__(self, height: int, width: int):
        super().__init__()
        # Example minimal baseline (commented): a tiny conv stack producing logits
        # self.net = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 32, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 1, 1)  # logits
        # )

        # Leave unimplemented so you plug in your model:
        self.net = None
        self.height = height
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            raise NotImplementedError("Implement YourModel.net or replace YourModel with your own model.")
        return self.net(x)


# ------------------------------
# Main: build dataset, train, eval
# ------------------------------
def main():
    # print(torc0h.cuda.is_available())
    # exit()
    
    # 1) Dataset generation
    dcfg = DatasetConfig(
        height=10, width=14, n_bombs=20,
        num_games=600, seed=123, val_frac=0.1, test_frac=0.0
    )
    print("Generating dataset...")
    X, y = collect_dataset(dcfg)
    print(f"Dataset: X={X.shape}, y={y.shape}  (per-sample shape={X.shape[1:]})")

    # 2) Split & loaders
    train_ds, val_ds, test_ds = train_val_test_split(X, y, dcfg.val_frac, dcfg.test_frac, dcfg.seed)
    print(f"Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    tcfg = TrainConfig(epochs=5, batch_size=32, lr=1e-3, from_logits=True, use_pos_weight=True)
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, pin_memory=True)

    # 3) Model
    # model = CNNModel(dcfg.height, dcfg.width).to(tcfg.device)
    # model = MLPModel(dcfg.height, dcfg.width).to(tcfg.device)
    model = TransformerModel(dcfg.height, dcfg.width).to(tcfg.device)
    
    

    # 4) Loss
    pos_weight = None
    if tcfg.from_logits and tcfg.use_pos_weight:
        pos_weight = estimate_pos_weight(train_ds).to(tcfg.device)
    loss_fn = make_loss(tcfg.from_logits, pos_weight=pos_weight)

    # 5) Optimizer (Adam) :contentReference[oaicite:4]{index=4}
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr)

    # 6) Train/Eval loop
    best_val = math.inf
    for epoch in range(1, tcfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, loss_fn, optimizer, device=tcfg.device)
        va = evaluate(model, val_loader, loss_fn, device=tcfg.device)
        print(f"Epoch {epoch:02d} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} "
              f"| val loss {va['loss']:.4f} acc {va['acc']:.4f}")

        # (Optional) checkpoint best
        if va["loss"] < best_val:
            best_val = va["loss"]
            # torch.save(model.state_dict(), "best_minesweeper_model.pt")

    print("Done.")


if __name__ == "__main__":
    main()
