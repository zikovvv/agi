import gc
import json
import traceback
import tqdm
from gen_simple_ds import gen_examples, PuzzleNames
from get_dataloader_for_model_for_task import get_dataloaders_for_cnn_masked_modeling
from get_ds_for_task import get_ds_for_masked_modeling_only_answer, get_ds_for_masked_modeling_only_answer_only_foreground_items
from model import *
import math
from dataclasses import dataclass
import sys
from typing import Callable, Dict, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.functional import F # type: ignore
from common import *
from datasets import Dataset
import torch
from torch import nn
import atexit
import weakref
import matplotlib.pyplot as plt



class SimpleARCDataset(Dataset):
    def __init__(self,
        examples: List[Tuple[np.ndarray, np.ndarray]],       # (N,9,9) uint8 [0..9]
    ):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.examples[idx]

@dataclass
class DatasetConfig:
    num_samples: int = 600    # dataset size
    seed: int = 123
    val_frac: float = 0.1
    test_frac: float = 0.0    # set >0 for a held-out test split
    
    # for masked modelling
    percentage_masked : float = 0.2


@dataclass
class TrainConfig:
    nb_plots_on_val : int = 2
    epochs: int = 50
    batch_size: int = 16
    lr: float = 3e-4
    device: str = "cpu"

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: str = "cpu") -> Dict[str, float]:
    ignored_id  : int = model.cfg.ignore_id # type: ignore

    model.train()
    total = 0
    total_loss = 0.0
    total_correct = 0
    total_classified = 0
    for batch in tqdm.tqdm(loader, total=len(loader)):
        input_ids, labels = batch['input_ids'], batch['labels']
        optimizer.zero_grad(set_to_none=True)
        res = model(input_ids, labels)
        # print(res)
        loss = res['loss']
        logits = res['logits']
        log_debug(f'{logits.shape = }')

        loss.backward()
        optimizer.step()
        
        
        preds = logits.argmax(dim=-1)
        log_debug(f'{preds.shape = }')
        nb_corr_cur = (preds == labels).sum().item()
        nb_masked = (labels != ignored_id).sum().item()
        total_correct += nb_corr_cur
        total_classified += nb_masked
        
        total_loss += loss.item() * input_ids.size(0)
        total += input_ids.shape[0]



    avg_loss = total_loss / max(total, 1)
    acc = total_correct / total_classified
    return {"loss": avg_loss, "acc": acc}




# def plot_eval_batch(
#     input_ids : torch.Tensor,
#     labels : torch.Tensor,
#     labels_predicted : torch.Tensor,
#     masked_token_id : int,
# ) -> None:
#     B, H, W = input_ids.shape
#     assert ((input_ids == masked_token_id) == (labels != -100)).sum().item() == B * H * W, f'{input_ids.shape = }, {labels.shape = }, {labels_predicted.shape = }'
#     print(input_ids.shape)
#     MAX_COLOR = 30
#     # get predicted values of labels_predicted where labels are masked_token_id
#     # then place them in a new tensor with the same shape as input_ids
#     labels_predicted_masked = labels_predicted[input_ids == masked_token_id]
#     input_ids_with_predictions = input_ids.clone()
#     # place values of labels_predicted_masked inside of input_ids_with_predictions where input_ids == masked_token_id
#     input_ids_with_predictions[input_ids == masked_token_id] = labels_predicted_masked

#     fig, axs = plt.subplots(4, min(B, 5), figsize=(6, 12))
#     for i in range(min(B, 5)):
#         axs[0, i].imshow(input_ids[i], cmap='tab20', vmin=0, vmax=MAX_COLOR)
#         axs[0, i].set_title(f'Input ids')

#         axs[1, i].imshow(labels[i], cmap='tab20', vmin=0, vmax=MAX_COLOR)
#         axs[1, i].set_title(f'Actual labels')

#         axs[2, i].imshow(labels_predicted[i], cmap='tab20', vmin=0, vmax=MAX_COLOR)
#         axs[2, i].set_title(f'Predicted labels')
        
#         axs[3, i].imshow(input_ids_with_predictions[i], cmap='tab20', vmin=0, vmax=MAX_COLOR)
#         axs[3, i].set_title(f'Predicted labels on inputs')
#     plt.show()


import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_eval_batch(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    labels_predicted: torch.Tensor,
    masked_token_id: int,
    *,
    max_color: int = 30,
    cmap: str = "tab20",
    cell_inches: float = 2.0,      # size of each small image (inches)
    dpi: int = 200,                # higher -> sharper
    show_row_labels: bool = True,  # overlay row labels instead of titles (no extra space)
) -> None:
    """
    4 x B grid with NO whitespace.
    Rows: [Input ids, Actual labels, Predicted labels, Predicted on inputs]
    Columns: all batch elements.
    """
    B, H, W = input_ids.shape
    assert ((input_ids == masked_token_id) == (labels != -100)).sum().item() == B * H * W, \
        f"{input_ids.shape = }, {labels.shape = }, {labels_predicted.shape = }"

    # Fill masked tokens with predictions
    labels_predicted_masked = labels_predicted[input_ids == masked_token_id]
    input_ids_with_predictions = input_ids.clone()
    input_ids_with_predictions[input_ids == masked_token_id] = labels_predicted_masked

    # Build figure sized exactly to the grid; no margins, no gaps
    nrows, ncols = 4, B
    fig = plt.figure(figsize=(cell_inches * ncols, cell_inches * nrows), dpi=dpi)
    gs = fig.add_gridspec(nrows, ncols, wspace=0.0, hspace=0.0)

    # Helper to render one small image without axes chrome
    def _show(ax, arr):
        # Ensure numpy
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        ax.imshow(arr, cmap=cmap, vmin=0, vmax=max_color, interpolation="nearest")
        ax.set_axis_off()
        # Keep pixels square & tight
        ax.set_aspect("equal", adjustable="box")

    # Fill grid
    for j in range(ncols):
        _show(fig.add_subplot(gs[0, j]), input_ids[j])
        _show(fig.add_subplot(gs[1, j]), labels[j])
        _show(fig.add_subplot(gs[2, j]), labels_predicted[j])
        _show(fig.add_subplot(gs[3, j]), input_ids_with_predictions[j])

    # Absolutely no outer padding
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Optional overlay row labels (doesn't add layout space)
    if show_row_labels and ncols > 0:
        row_names = ["Input ids", "Actual labels", "Predicted labels", "Predicted on inputs"]
        for r, name in enumerate(row_names):
            ax0 = fig.axes[r * ncols + 0]  # leftmost axis of the row
            ax0.text(
                0.02, 0.02, name,
                transform=ax0.transAxes,
                ha="left", va="bottom",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
            )

    plt.show()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    nb_plots_on_val: int = 2
) -> Dict[str, float]:
    ignored_id  : int = model.cfg.ignore_id # type: ignore
    masked_token_id : int = model.cfg.masked_token_id # type: ignore
    plotted_examples = 0
    model.eval()
    total = 0
    total_loss = 0.0
    total_correct = 0
    total_classified = 0
    for batch in loader:
        input_ids : torch.Tensor = batch['input_ids']
        labels : torch.Tensor = batch['labels']
        res = model(input_ids, labels)
        loss = res['loss']
        logits : torch.Tensor = res['logits']
        preds = logits.argmax(dim=-1)
        nb_corr_cur = (preds == labels).sum().item()
        nb_masked = (labels != ignored_id).sum().item()
        total_correct += nb_corr_cur
        total_classified += nb_masked
        
        total_loss += loss.item() * input_ids.size(0)

        total += input_ids.shape[0]
        if plotted_examples < nb_plots_on_val:
            plot_eval_batch(
                input_ids.cpu().detach(),
                labels.cpu().detach(),
                preds.cpu().detach(),
                masked_token_id=masked_token_id
            )
            plotted_examples += 1

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / total_classified
    return {"loss": avg_loss, "acc": acc}


def _get_model(mcfg : ARCCNNConfig, tcfg : TrainConfig) -> ARCCNNForMaskedSequenceModelling :
    model = ARCCNNForMaskedSequenceModelling(mcfg).to(tcfg.device)
    return model


def _get_optimizer(model, tcfg : TrainConfig) :
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    return optimizer

logger.remove()
logger.add(sys.stderr, level="INFO")


def main(
    dcfg : Optional[DatasetConfig] = None,
    tcfg : Optional[TrainConfig] = None,
    mcfg : Optional[ARCCNNConfig] = None
):
    dcfg = DatasetConfig(
        seed=123,
        val_frac=0.1,
        test_frac=0.0,
        percentage_masked=0.2,
        num_samples=12
    ) if dcfg is None else dcfg
    tcfg = TrainConfig(
        batch_size=4,
        lr=3e-4
    ) if tcfg is None else tcfg
    mcfg = ARCCNNConfig(
        d_model=256,
        n_head=8,
        num_layers=6,
        dim_feedforward=1024,
        vocab_size=20,            # adjust to your palette + specials
        max_height=40,
        max_width=40,
    ) if mcfg is None else mcfg

    
    model = _get_model(mcfg, tcfg)
    optimizer = _get_optimizer(model, tcfg)
    atexit.register(get_cleanup_function(model, optimizer=optimizer))

    
    raw_ds = get_ds_for_masked_modeling_only_answer(
        gen_examples(
            name = PuzzleNames.FIll_SIMPLE_OPENED_SHAPE,
            nb_examples = dcfg.num_samples,
            augment_colors = False,
            do_shuffle = False
        ),
        dcfg.percentage_masked,
        mcfg.masked_token_id
    )
    train_dl, val_dl = get_dataloaders_for_cnn_masked_modeling(
        raw_ds,
        masked_token_id = mcfg.masked_token_id,
        ignore_label_id = mcfg.ignore_id,
        newline_token_id = mcfg.newline_token_id,
        max_grid_width = mcfg.max_width,
        max_grid_height = mcfg.max_height,
        pad_token_id = mcfg.pad_token_id,
        split_ratio = 1 - dcfg.val_frac,
        batch_size_train = tcfg.batch_size,
        batch_size_eval = tcfg.batch_size,
        only_answer=False, # add question to input or not (only answer will be masked anyways)
        pad_to_max=True, # because now we have rotary embeddings and we need to pad this to the max
        device = tcfg.device
    )

    for epoch in range(1, tcfg.epochs + 1):
        tr = train_one_epoch(model, train_dl, optimizer, device=tcfg.device)
        l  = f"Epoch {epoch:02d} | "
        for k, v in tr.items() : 
            l += f"{k} {v:.4f} | "
        log(f"{l}")
        
        l  = f"Epoch {epoch:02d} | "
        with torch.no_grad():
            va = evaluate(model, val_dl, nb_plots_on_val = tcfg.nb_plots_on_val)
        for k, v in va.items() : 
            l += f"{k} {v:.4f} | "
        log(f"{l}")

    log("Done.")

if __name__ == "__main__":
    main()