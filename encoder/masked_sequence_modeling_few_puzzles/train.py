import json
import tqdm
from gen_simple_ds import gen_simple_ds_fill_simple_shape
from get_dataloader_for_model_for_task import get_dataloaders_for_encoder_masked_modeling
from get_ds_for_task import get_ds_for_masked_modeling_only_answer
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
    epochs: int = 50
    batch_size: int = 16
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    loss_type: str = "ce"  # 'ce' (recommended) or 'bce'
    # CrossEntropyLoss expects logits over classes with target as class indices 0..C-1 (not one-hot).  # docs: pytorch  # noqa

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
        input_ids, rows, cols, token_types_ids, labels = batch['input_ids'], batch['rows'], batch['cols'], batch['token_type_ids'], batch['labels']
        optimizer.zero_grad(set_to_none=True)
        res = model(input_ids, rows, cols, token_types_ids, labels)
        # print(res)
        loss = res['loss']
        logits = res['logits']

        loss.backward()
        optimizer.step()
        
        
        preds = logits.argmax(dim=-1)
        nb_corr_cur = (preds == labels).sum().item()
        nb_masked = (labels == ignored_id).sum().item()
        total_correct += nb_corr_cur
        total_classified += nb_masked
        
        total_loss += loss.item() * input_ids.size(0)

        total += 1



    avg_loss = total_loss / max(total, 1)
    acc = total_correct / total_classified
    return {"loss": avg_loss, "acc": acc}

@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: str = "cpu") -> Dict[str, float]:
    ignored_id  : int = model.cfg.ignore_id # type: ignore

    model.eval()
    total = 0
    total_loss = 0.0
    total_correct = 0
    total_classified = 0
    for batch in loader:
        input_ids, rows, cols, token_types_ids, labels = batch['input_ids'], batch['rows'], batch['cols'], batch['token_type_ids'], batch['labels']
        res = model(input_ids, rows, cols, token_types_ids, labels)
        loss = res['loss']
        logits = res['logits']
        preds = logits.item().argmax(dim=-1)
        nb_corr_cur = (preds == labels).sum().item()
        nb_masked = (labels == ignored_id).sum().item()
        total_correct += nb_corr_cur
        total_classified += nb_masked
        
        total_loss += loss.item() * input_ids.size(0)

        total += 1

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / total_classified
    return {"loss": avg_loss, "acc": acc}


def _get_model(mcfg : ARCEncoderConfig, tcfg : TrainConfig) -> ARCEncoderForMaskedSequenceModelling :
    model = ARCEncoderForMaskedSequenceModelling(mcfg).to(tcfg.device)
    return model


def _get_optimizer(model, tcfg : TrainConfig) :
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    return optimizer
    

def main(
    dcfg: DatasetConfig,
    tcfg: TrainConfig,
    mcfg: ARCEncoderConfig,
):
    tcfg.device = 'cuda'
    # device = tcfg.device
    model = _get_model(mcfg, tcfg)
    optimizer = _get_optimizer(model, tcfg)
    
    raw_ds = get_ds_for_masked_modeling_only_answer(
        gen_simple_ds_fill_simple_shape(dcfg.num_samples),
        dcfg.percentage_masked,
        mcfg.masked_token_id
    )
    train_dl, val_dl = get_dataloaders_for_encoder_masked_modeling(
        raw_ds,
        masked_token_id = mcfg.masked_token_id,
        ignore_label_id = mcfg.ignore_id,
        newline_token_id = mcfg.newline_token_id,
        max_grid_width = mcfg.max_width,
        max_grid_height = mcfg.max_height,
        question_token_type_id = mcfg.token_type_question,
        answer_token_type_id = mcfg.token_type_answer,
        pad_token_id = mcfg.pad_token_id,
        pad_row_id = mcfg.ignore_id,
        pad_col_id = mcfg.ignore_id,
        pad_token_type_id = mcfg.ignore_id,
        split_ratio = 1 - dcfg.val_frac,
        batch_size_train = tcfg.batch_size,
        batch_size_eval = tcfg.batch_size,
        device = tcfg.device
    )

    for epoch in range(1, tcfg.epochs + 1):
        tr = train_one_epoch(model, train_dl, optimizer, device=tcfg.device)
        l  = f"Epoch {epoch:02d} | "
        for k, v in tr.items() : 
            l += f"{k} {v:.4f} | "
        print(f"{l}")
        
        l  = f"Epoch {epoch:02d} | "
        with torch.no_grad():
            va = evaluate(model, val_dl, device=tcfg.device)
        for k, v in va.items() : 
            l += f"{k} {v:.4f} | "
        print(f"{l}")

    print("Done.")






if __name__ == "__main__":
    logger.remove() #remove the old handler. Else, the old one will work along with the new one you've added below'
    logger.add(sys.stderr, level="INFO") 
    
    _dcfg = DatasetConfig(
        seed=123,
        val_frac=0.1,
        test_frac=0.0,
        percentage_masked=0.2,
        num_samples=1000
    )
    _tcfg = TrainConfig(
        batch_size=2,
        lr=3e-4
    )
    _mcfg : ARCEncoderConfig = ARCEncoderConfig(
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        vocab_size=20,            # adjust to your palette + specials
        max_height=40,
        max_width=40,
    )
    
    main(
        dcfg = _dcfg,
        tcfg = _tcfg,
        mcfg = _mcfg
    )
