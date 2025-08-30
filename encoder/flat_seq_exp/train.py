import atexit
import json
import tqdm
from encoder.flat_seq_exp.config import DatasetConfig, TrainConfig, EncoderConfig            
from get_dataloader_for_model_for_task import get_dataloaders_for_flat_seq_cls, get_dataloaders_for_encoder_masked_modeling
from get_ds_for_task import get_ds_for_simple_input_output_flat_seuqences, get_ds_for_masked_modeling_only_answer, get_ds_for_masked_modeling_only_answer_only_foreground_items
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
        logits : torch.Tensor = res['logits']

        loss.backward()
        optimizer.step()
        
        
        preds = logits.argmax(dim=-1)
        nb_corr_cur = (preds == labels).sum().item()
        nb_masked = (labels != ignored_id).sum().item()
        total_correct += nb_corr_cur
        total_classified += nb_masked
        
        total_loss += loss.item() * input_ids.size(0)
        total += input_ids.shape[0]



    avg_loss = total_loss / max(total, 1)
    acc = total_correct / total_classified
    return {"loss": avg_loss, "acc": acc}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
) -> Dict[str, float]:
    ignored_id: int = model.cfg.ignore_id  # type: ignore
    model.eval()
    total = 0
    total_loss = 0.0
    total_correct = 0
    total_classified = 0
    for batch in loader:
        input_ids, labels = batch['input_ids'], batch['labels']
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

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / total_classified
    return {"loss": avg_loss, "acc": acc}


def _get_model(mcfg : EncoderConfig, tcfg : TrainConfig) -> EncoderForCLS :
    model = EncoderForCLS(mcfg).to(tcfg.device)
    return model


def _get_optimizer(model : EncoderForCLS, tcfg : TrainConfig) :
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
        num_samples=250,
        seq_len=600,
    ) if dcfg is None else dcfg
    tcfg = TrainConfig(
        batch_size=4,
        lr=3e-4
    ) if tcfg is None else tcfg
    mcfg = EncoderConfig(
        d_model=256,
        n_head=8,
        num_layers=6,
        dim_feedforward=1024,
        vocab_size=200,
        max_len=2000
    ) if mcfg is None else mcfg

    # device = tcfg.device
    model = _get_model(mcfg, tcfg)
    optimizer = _get_optimizer(model, tcfg)
    atexit.register(get_cleanup_function(model, optimizer=optimizer))
    
    ds_raw = get_ds_for_simple_input_output_flat_seuqences(
        seq_len=dcfg.seq_len,
        nb_samples=dcfg.num_samples,
        nb_cls=10,
        reverse_labels=True
    )
    train_dl, val_dl = get_dataloaders_for_flat_seq_cls(
        ds_raw,
        masked_token_id=mcfg.masked_token_id,
        ignore_label_id = mcfg.ignore_id,
        sep_token_id = mcfg.qa_sep_token_id,
        pad_token_id = mcfg.pad_token_id,
        split_ratio = 1 - dcfg.val_frac,
        batch_size_train = tcfg.batch_size,
        batch_size_eval = tcfg.batch_size,
        device = tcfg.device
    )

    for epoch in range(1, tcfg.epochs + 1):
        tr = train_one_epoch(model, train_dl, optimizer)
        l  = f"Epoch {epoch:02d} | "
        for k, v in tr.items() : 
            l += f"{k} {v:.4f} | "
        log(f"{l}")
        
        l  = f"Epoch {epoch:02d} | "
        with torch.no_grad():
            va = evaluate(model, val_dl)
        for k, v in va.items() : 
            l += f"{k} {v:.4f} | "
        log(f"{l}")

    log("Done.")



if __name__ == "__main__":
    main()
