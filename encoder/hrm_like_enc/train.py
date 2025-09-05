import atexit
import json
import tqdm
import os
import wandb
from dotenv import load_dotenv
from encoder.hrm_like_enc.config import DatasetConfig, TrainConfig, EncoderConfig            
from encoder.hrm_like_enc.hrm_lucidrains import HRM
from encoder.hrm_like_enc.inference import augment_colors_batch, augmented_inference_batched, augmented_inference_batched_with_voting
from gen_simple_arc_ds import PuzzleNames
from get_dataloader_for_model_for_task import get_dataloaders_for_flat_seq_cls, get_dataloaders_for_encoder_masked_modeling
from get_ds_for_task import get_arc_puzzle_ds_as_flat_ds, get_ds_1d_seq_for_random_input_with_some_transformation_for_output, get_custom_ds_arc, get_ds_for_masked_modeling_only_answer, get_ds_for_masked_modeling_only_answer_only_foreground_items
from model import *
import math
from dataclasses import dataclass
import sys
from typing import Dict, Optional, Tuple, Union
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
from show import plot_batch

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mcfg : EncoderConfig,
    dcfg : DatasetConfig,
    tcfg : TrainConfig,
) -> Dict[str, float]:
    model.train()
    nb_total_steps = 0
    total_loss = 0.0
    nb_total_correct = 0
    nb_total_classified = 0
    
    nb_total_last_steps = 0
    total_loss_last = 0.0
    nb_last_correct = 0
    nb_last_classified = 0


    for bid, batch in enumerate(tqdm.tqdm(loader, total=len(loader))):
        input_ids, labels = batch['input_ids'], batch['labels']
        
        if bid < tcfg.t_show_nb_first_preds:
            log(f'{input_ids.shape = }, {labels.shape = }')
            plot_batch(
                data=[
                    input_ids[:10, :],
                    labels[:10, :],
                ],
                height=dcfg.max_width * (2 if dcfg.expand else 1),
                width=dcfg.max_width,
                show_to_window=tcfg.t_show_in_window,
            )
        
        colors_orig, colors_perms, input_ids, labels = augment_colors_batch(
            input_ids,
            labels,
            max_permutations=tcfg.t_max_nb_aug,
            ignore_label_id=mcfg.ignore_id,
            pad_token_id=mcfg.pad_token_id,
            add_orig=True
        )

        optimizer.zero_grad(set_to_none=True)
        for self_correct_step in range(max(1, tcfg.t_nb_max_self_correction)):
            res = model(input_ids, labels)
            # print(res)
            loss : torch.Tensor = res['loss']
            logits : torch.Tensor = res['logits']

            loss.backward()
            
            with torch.no_grad():
                preds = logits.detach().argmax(dim=-1)
                nb_corr_cur = (preds == labels).sum().item()
                nb_masked = (labels != mcfg.ignore_id).sum().item()
                nb_total_correct += nb_corr_cur
                nb_total_classified += nb_masked
                
                total_loss += loss.item()
                nb_total_steps += 1
                
                input_ids = preds.detach().clone()
        optimizer.step()
        
        nb_last_correct += nb_corr_cur
        nb_last_classified += nb_masked
        total_loss_last += loss.item() 
        nb_total_last_steps += 1

    avg_loss = total_loss / max(nb_total_steps, 1)
    acc = nb_total_correct / nb_total_classified
    avg_loss_last = total_loss_last / max(nb_total_last_steps, 1)
    acc_last = nb_last_correct / max(nb_last_classified, 1)
    return {"loss": avg_loss, "acc": acc, "acc_last": acc_last, 'avg_loss_last': avg_loss_last}



def self_correction_val(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    ignored_id: int,
    nb_steps: int = 4
) :
    total_loss = 0.0
    nb_total_correct = 0
    nb_total_classified = 0
    nb_steps = 0
    
    for self_correct_step in range(nb_steps):
        res = model(input_ids, labels)
        # print(res)
        loss : torch.Tensor = res['loss']
        logits : torch.Tensor = res['logits']
        
        with torch.no_grad():
            preds = logits.detach().argmax(dim=-1)
            nb_corr_cur = (preds == labels).sum().item()
            nb_masked = (labels != ignored_id).sum().item()
            nb_total_correct += nb_corr_cur
            nb_total_classified += nb_masked
            
            total_loss += loss.item()
            input_ids = preds.detach().clone()
            nb_steps += 1
            if nb_masked == nb_corr_cur:
                log_warn("All classified correctly, stopping self-correction! wow!")
                break
    return {
        "total_loss": total_loss,
        "nb_total_correct": nb_total_correct,
        "nb_total_classified": nb_total_classified,
        'loss_last': loss.item(),
        'nb_corr_last': nb_corr_cur,
        'nb_class_last': nb_masked,
        'nb_steps': nb_steps
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    mcfg : EncoderConfig,
    dcfg : DatasetConfig,
    tcfg : TrainConfig,
) -> Dict[str, float]:
    def simple_val(
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) :
        with torch.no_grad():
            input_ids, labels = input_ids.to(tcfg.device), labels.to(tcfg.device)
            res = model(input_ids, labels)
            logits = res['logits']
            loss = res['loss'].item()
            preds = logits.argmax(dim=-1)
            return loss, preds

    ignored_id: int = mcfg.ignore_id  # type: ignore
    pad_token_id: int = mcfg.pad_token_id # type: ignore
    model.eval()
    total_steps = 0


    total_loss = 0.0
    nb_total_correct = 0
    nb_total_labels = 0

    total_loss = 0.0
    nb_total_correct = 0
    nb_total_labels = 0

    nb_total_labels_aug = 0
    nb_total_labels_vote = 0
    total_loss_aug = 0.0

    nb_total_cor_vote = 0
    nb_total_cor_aug = 0
    
    
    sc_total_loss = 0.0
    sc_nb_total_correct = 0
    sc_nb_total_labels = 0
    sc_nb_last_correct = 0
    sc_nb_last_labels = 0
    sc_total_steps = 0
    sc_total_loss_last = 0.0
    sc_last_steps = 0


    expand_coeff = 1 if not dcfg.expand else 2
    for bit, batch in enumerate(loader):
        input_ids, labels = batch['input_ids'].to(tcfg.device), batch['labels'].to(tcfg.device)
        loss, preds = simple_val(
            input_ids=input_ids,
            labels=labels
        )
        nb_correct = (preds == labels).sum().item()
        nb_labels = (labels != ignored_id).sum().item()
        nb_total_correct += nb_correct
        nb_total_labels += nb_labels
        total_loss += loss
        
        
        if tcfg.v_nb_max_self_correction > 1:
            res : Dict[str, Union[float, int]] = self_correction_val(
                model=model,
                input_ids=input_ids,
                labels=labels,
                nb_steps=tcfg.v_nb_max_self_correction,
                ignored_id=ignored_id
            )
            sc_total_loss += res['total_loss']
            sc_nb_total_correct += res['nb_corr_last']
            sc_nb_total_labels += res['nb_class_last']
            sc_nb_last_correct += res['nb_corr_last']
            sc_nb_last_labels += res['nb_class_last']
            sc_total_steps += res['nb_steps']
            sc_total_loss_last += res['loss_last']
            sc_last_steps += 1

        if tcfg.v_do_augmented_inference:
            loss_aug, preds_voted, nb_labels_aug, nb_cor_aug, nb_labels_vote, nb_cor_voting = augmented_inference_batched_with_voting(
                model = model,
                input_ids = input_ids,
                labels = labels,
                ignore_label_id = ignored_id,
                pad_token_id = pad_token_id,
                vocab_size=mcfg.vocab_size,
                show_to_window=tcfg.v_show_in_window,
                max_aug=tcfg.v_max_nb_aug,
                debug=bit < tcfg.v_show_nb_first_preds
            )
            nb_total_cor_aug += nb_cor_aug
            nb_total_labels_aug += nb_labels_aug
            total_loss_aug += loss_aug

            nb_total_cor_vote += nb_cor_voting
            nb_total_labels_vote += nb_labels_vote

        if bit < tcfg.v_show_nb_first_preds:
            plot_batch(
                data=[
                    input_ids[:10, :],
                    labels[:10, :],
                    # preds[:10, field_area:],
                    preds[:10, :],
                ] + ([preds_voted[:10, :]] if tcfg.v_do_augmented_inference else []),
                height=dcfg.max_width * expand_coeff,
                width=dcfg.max_width,
                show_to_window=tcfg.v_show_in_window,
            )

        total_steps += 1
        
        
    acc = nb_total_correct / max(nb_total_labels, 1)
    acc_aug = nb_total_cor_aug / max(nb_total_labels_aug, 1)
    acc_vote = nb_total_cor_vote / max(nb_total_labels_vote, 1)
    sc_acc = sc_nb_total_correct / max(sc_nb_total_labels, 1)
    sc_acc_last = sc_nb_last_correct / max(sc_nb_last_labels, 1)
    
    return {
        "loss":  total_loss / max(total_steps, 1),
        "loss_1": total_loss / max(total_steps, 1),
        'loss_aug': total_loss_aug / max(total_steps, 1),
        'acc': acc,
        'acc_aug': acc_aug,
        'acc_vote': acc_vote,
        'sc_loss': sc_total_loss / max(sc_total_steps, 1),
        'sc_loss_last': sc_total_loss_last / max(sc_last_steps, 1),
        'sc_acc': sc_acc,
        'sc_acc_last': sc_acc_last,
    }


def _get_model(mcfg : EncoderConfig, tcfg : TrainConfig) -> EncoderForCLS :
    model = EncoderForCLS(mcfg).to(tcfg.device)
    return model


def _get_optimizer(model : nn.Module, tcfg : TrainConfig) :
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    return optimizer
    

logger.remove() #remove the old handler. Else, the old one will work along with the new one you've added below'
# logger.add(sys.stderr, level="DEBUG") 
logger.add(sys.stderr, level="INFO") 


def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize wandb
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
    
    field_width = 9
    seq_len = -100
    expand = False
    dcfg = DatasetConfig(
        seed=123,
        val_frac=0.1,
        test_frac=0.0,
        num_samples=300,
        seq_len=seq_len,
        max_width=field_width,
        expand=expand,
    )

    tcfg = TrainConfig(
        lr=1e-4,

        # train cfg
        t_batch_size=4,
        t_show_nb_first_preds=0,
        t_nb_max_self_correction=1,
        t_show_in_window=False,
        t_max_nb_aug=5,

        # val cfg
        v_batch_size=3,
        v_nb_max_self_correction=1,
        v_do_augmented_inference=False,
        v_show_in_window=False,
        v_max_nb_aug=20,
        v_show_nb_first_preds=1
    )
    
    mcfg = EncoderConfig(
        d_model=128,
        n_head=8,
        d_head=64,
        num_layers=8,
        dim_feedforward=128,
        vocab_size=200,
        
        nb_max_rope_positions=4000,
        
        nb_refinement_steps=1,
        nb_last_trained_steps=1,
        
        enable_pseudo_diffusion_inner=False,
        enable_pseudo_diffusion_outer=True,
        feed_first_half=False,

        # use_transposed_rope_for_2d_vertical_orientation=False,
        # field_width=field_width,
        # field_height=field_width * 2,

        use_emb_norm = False,
        
        use_axial_rope = True,
        
        # learned pos emb
        use_learned_pos_emb=False,
        use_custom_learned_pos_emb_per_head=True,
        
        # learned pos emb with custo dim
        use_projection_for_learned_pos_embs=False,        
        learned_pos_embs_dim=64,
        
        # if use ready made implementation from x_transformers
        use_x_encoder=False,

        # neccessary for 2d learned pos embedding
        field_width=field_width,
        field_height=field_width * (2 if expand else 1),
        
        # cnn in each attention layer 
        use_cnn=True,

    )

    # Initialize wandb run
    wandb.init(
        project="hrm-encoder-training",
        config={
            "dataset": dcfg.__dict__,
            "training": tcfg.__dict__,
            "model": mcfg.__dict__,
            "field_width": field_width,
            "expand": expand
        }
    )

    # tcfg.device = 'cpu'
    model = _get_model(mcfg, tcfg)
    from x_transformers import Encoder

    optimizer = _get_optimizer(model, tcfg)
    atexit.register(get_cleanup_function(model, optimizer=optimizer))
    
    task_name = 'sudoku'
    ds_raw = get_custom_ds_arc(
        seq_len=dcfg.seq_len,
        nb_samples=dcfg.num_samples,
        nb_cls=10,
        task=task_name,

        field_width = field_width,


        do_2d = True,
        do_transpose = True,
        
        
        nb_missing_min = 20,
        nb_missing_max = 21,
        masked_token_id = mcfg.masked_token_id,
    )
    train_dl, val_dl = get_dataloaders_for_flat_seq_cls(
        ds_raw,
        ignore_label_id = mcfg.ignore_id,
        sep_token_id = mcfg.qa_sep_token_id,
        pad_token_id = mcfg.pad_token_id,
        split_ratio = 1 - dcfg.val_frac,
        batch_size_train = tcfg.t_batch_size,
        batch_size_eval = tcfg.v_batch_size,
        device = tcfg.device,
        add_labels_to_inputs=dcfg.add_labels_to_inputs,
        add_sep=dcfg.add_sep,
        expand=dcfg.expand,
    )
    epoch = 0
    while 1 :
        tr = train_one_epoch(
            model = model,
            loader = train_dl,
            optimizer = optimizer,
            mcfg = mcfg,
            dcfg = dcfg,
            tcfg = tcfg,
        )
        l  = f"TRAIN Epoch {epoch:02d} | "
        for k, v in tr.items() : 
            l += f"{k} {v:.4f} | "
        log(f"{l}")
        
        # Log training metrics to wandb
        train_metrics = {f"train/{k}": v for k, v in tr.items()}
        train_metrics["epoch"] = epoch
        wandb.log(train_metrics)

        l  = f"VAL Epoch {epoch:02d} | "
        with torch.no_grad():
            va = evaluate(
                model = model,
                loader = val_dl,
                mcfg = mcfg,
                dcfg = dcfg,
                tcfg = tcfg,
            )
        for k, v in va.items() : 
            l += f"{k} {v:.4f} | "
        log(f"{l}")
        
        # Log validation metrics to wandb
        val_metrics = {f"val/{k}": v for k, v in va.items()}
        wandb.log(val_metrics)

        epoch += 1

    wandb.finish()
    log("Done.")

if __name__ == "__main__":
    main()
