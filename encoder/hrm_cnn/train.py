import atexit
import json
import tqdm
from encoder.hrm_cnn.config import DatasetConfig, TrainConfig, EncoderConfig            
from gen_simple_arc_ds import PuzzleNames, gen_arc_puzzle_ex
from get_dataloader_for_model_for_task import get_dataloaders_for_2d_full_pred, get_dataloaders_for_flat_seq_cls, get_dataloaders_for_encoder_masked_modeling
from get_ds_for_task import get_arc_puzzle_ds_as_flat_ds, get_ds_1d_seq_for_random_input_with_some_transformation_for_output, get_ds_arc_for_1d, get_ds_for_masked_modeling_only_answer, get_ds_for_masked_modeling_only_answer_only_foreground_items
from model import *
import math
from dataclasses import dataclass
import sys
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.functional import F # type: ignore
from common import *
from datasets import Dataset
import torch
from torch import nn
import matplotlib.pyplot as plt




def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    ignored_id: int = model.cfg.ignore_id  # type: ignore

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
        loss : torch.Tensor = res['loss']

        loss.backward()
        optimizer.step()



        # print(logits.shape, labels.shape, logits_map.shape)
        logits_map: torch.Tensor = res['logits_map']
        preds = logits_map.argmax(dim=-1)
        nb_corr_cur = (preds == labels).sum().item()
        nb_masked = (labels != ignored_id).sum().item()
        total_correct += nb_corr_cur
        total_classified += nb_masked
        total_loss += loss.item() * input_ids.size(0)
        total += input_ids.shape[0]



    avg_loss = total_loss / max(total, 1)
    acc = total_correct / total_classified
    return {"loss": avg_loss, "acc": acc}



import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_eval_batch(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    max_field_width : int = 40,
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

    nrows, ncols = 3, B
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
        _show(fig.add_subplot(gs[2, j]), preds[j])

    # Absolutely no outer padding
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    plt.show()

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    max_field_width : int,
    show_nb_first_preds : int
) -> Dict[str, float]:
    ignored_id: int = model.cfg.ignore_id  # type: ignore
    model.eval()
    total = 0
    total_loss = 0.0
    total_correct = 0
    total_classified = 0
    
    showed = 0
    for batch in loader:
        input_ids, labels = batch['input_ids'], batch['labels']
        res = model(input_ids, labels)
        loss = res['loss']
        # print(logits.shape, labels.shape, logits_map.shape)
        logits_map: torch.Tensor = res['logits_map']
        preds = logits_map.argmax(dim=-1)
        nb_corr_cur = (preds == labels).sum().item()
        nb_masked = (labels != ignored_id).sum().item()
        total_correct += nb_corr_cur
        total_classified += nb_masked
        total_loss += loss.item() * input_ids.size(0)
        total += input_ids.shape[0]

        if showed < show_nb_first_preds:
            plot_eval_batch(input_ids, labels, preds, max_field_width=max_field_width) # type: ignore
            showed += 1

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / total_classified
    return {"loss": avg_loss, "acc": acc}


def _get_model(mcfg : EncoderConfig, tcfg : TrainConfig) -> EncoderForCLSWithCNNFeatureExtraction :
    model = EncoderForCLSWithCNNFeatureExtraction(mcfg).to(tcfg.device)
    return model


def _get_optimizer(model : EncoderForCLSWithCNNFeatureExtraction, tcfg : TrainConfig) :
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    return optimizer
    

logger.remove() #remove the old handler. Else, the old one will work along with the new one you've added below'
logger.add(sys.stderr, level="INFO") 

def main(
    dcfg : Optional[DatasetConfig] = None,
    tcfg : Optional[TrainConfig] = None,
    mcfg : Optional[EncoderConfig] = None,
):
    dcfg = DatasetConfig(
        seed=123,
        val_frac=0.1,
        test_frac=0.0,
        num_samples=20,
        seq_len=30,
        max_width=40
    ) if dcfg is None else dcfg
    tcfg = TrainConfig(
        batch_size=2,
        lr=3e-4
    ) if tcfg is None else tcfg
    mcfg = EncoderConfig(
        d_model=64,
        n_head=4,
        num_layers=1,
        nb_refinement_steps=8,
        dim_feedforward=256,
        vocab_size=200,
        max_len=4000
    ) if mcfg is None else mcfg

    # device = tcfg.device
    model = _get_model(mcfg, tcfg)
    optimizer = _get_optimizer(model, tcfg)
    atexit.register(get_cleanup_function(model, optimizer=optimizer))
    
    def get_train_val_dls() :
        ds_raw = gen_arc_puzzle_ex(
            name = PuzzleNames.FILL_SIMPLE_OPENED_SHAPE,
            nb_examples = dcfg.num_samples,
            augment_colors = False,
            do_shuffle = False
        )
        train_dl, val_dl = get_dataloaders_for_2d_full_pred(
            ds_raw,
            ignore_label_id = mcfg.ignore_id,
            pad_token_id = mcfg.pad_token_id,
            split_ratio = 1 - dcfg.val_frac,
            batch_size_train = tcfg.batch_size,
            batch_size_eval = tcfg.batch_size,
            max_grid_width=dcfg.max_width,
            add_input_to_labels=True,
            device = tcfg.device,
        )
        return train_dl, val_dl
    train_dl, val_dl = get_train_val_dls()
    for epoch in range(1, tcfg.epochs + 1):
        tr = train_one_epoch(model, train_dl, optimizer)
        l  = f"Epoch {epoch:02d} | "
        for k, v in tr.items() : 
            l += f"{k} {v:.4f} | "
        log(f"{l}")
        
        l  = f"Epoch {epoch:02d} | "
        with torch.no_grad():
            va = evaluate(model, val_dl, show_nb_first_preds=4, max_field_width=dcfg.max_width)
        for k, v in va.items() : 
            l += f"{k} {v:.4f} | "
        log(f"{l}")
        train_dl, _ = get_train_val_dls()

    log("Done.")



if __name__ == "__main__":
    main()
