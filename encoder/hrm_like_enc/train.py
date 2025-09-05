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
    device : str = 'cpu',
    max_nb_aug : int = 5,
    nb_iterative_self_correction_steps : int = 1
) -> Dict[str, float]:
    ignored_id: int = mcfg.ignore_id  # type: ignore
    pad_token_id: int = mcfg.pad_token_id  # type: ignore

    model.train()
    nb_total_steps = 0
    total_loss = 0.0
    nb_total_correct = 0
    nb_total_classified = 0
    
    nb_total_last_steps = 0
    total_loss_last = 0.0
    nb_last_correct = 0
    nb_last_classified = 0
    
    
    for batch in tqdm.tqdm(loader, total=len(loader)):
        input_ids, labels = batch['input_ids'], batch['labels']
        colors_orig, colors_perms, input_ids, labels = augment_colors_batch(
            input_ids,
            labels,
            max_permutations=max_nb_aug,
            ignore_label_id=ignored_id,
            pad_token_id=pad_token_id,
            add_orig=True
        )

        optimizer.zero_grad(set_to_none=True)
        for self_correct_step in range(max(1, nb_iterative_self_correction_steps)):
            res = model(input_ids, labels)
            # print(res)
            loss : torch.Tensor = res['loss']
            logits : torch.Tensor = res['logits']

            loss.backward()
            
            with torch.no_grad():
                preds = logits.detach().argmax(dim=-1)
                nb_corr_cur = (preds == labels).sum().item()
                nb_masked = (labels != ignored_id).sum().item()
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
    nb_iterative_self_correction_steps : int = 4
) :
    total_loss = 0.0
    nb_total_correct = 0
    nb_total_classified = 0
    nb_steps = 0
    
    for self_correct_step in range(nb_iterative_self_correction_steps):
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
    max_field_width : int,
    show_nb_first_preds : int,
    mcfg : EncoderConfig,
    expand : bool,
    device : str = 'cpu',
    nb_iterative_self_correction_steps : int = 1,
    do_augmented_inference : bool = True,
    show_or_just_log_to_wandb : bool = True
) -> Dict[str, float]:
    def simple_val(
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) :
        with torch.no_grad():
            input_ids, labels = input_ids.to(device), labels.to(device)
            res = model(input_ids, labels)
            logits = res['logits']
            loss = res['loss'].item()
            preds = logits.argmax(dim=-1)
            return loss, preds
            # nb_right  = (preds == labels).sum().item()
            # nb_classified = (labels != ignored_id).sum().item()
            # return loss, preds, logits, nb_right / nb_classified

    ignored_id: int = mcfg.ignore_id  # type: ignore
    pad_token_id: int = mcfg.pad_token_id # type: ignore
    model.eval()
    total_steps = 0


    total_loss = 0.0
    nb_total_correct = 0
    nb_total_labels = 0

    total_loss_1 = 0.0
    nb_total_correct_1 = 0
    nb_total_labels_1 = 0


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


    expand_coeff = 1 if not expand else 2
    showed = 0
    for it, batch in enumerate(loader):
        input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
        
        # if showed < show_nb_first_preds:
        #     inp = input_ids[0]
        #     inp_fields = inp.view(-1, max_field_width, max_field_width).cpu().numpy()
        #     lab = labels[0]
        #     lab_fields = lab.view(-1, max_field_width, max_field_width * expand_coeff).cpu().numpy()
        #     print(inp_fields)
        #     print(lab_fields)
        #     exit()

        # res = model(input_ids, labels)
        # loss = res['loss']
        # logits : torch.Tensor = res['logits']
        # preds = logits.argmax(dim=-1)
        # nb_corr_cur = (preds == labels).sum().item()
        # nb_masked = (labels != ignored_id).sum().item()
        # total_correct += nb_corr_cur
        # total_classified += nb_masked
        # log_debug(f'{input_ids.shape = }, {labels.shape = }, {ignored_id = }')
        
        
        
        
        # loss, voted_preds, acc_global, acc_local = augmented_inference_batched_with_voting(
        #     model = model,
        #     input_ids = input_ids,
        #     labels = labels,
        #     ignore_label_id = ignored_id,
        #     debug = it < show_nb_first_preds
        # )
        # acc_global_sum += acc_global
        # acc_local_sum += acc_local
        


        # loss, preds = augmented_inference_batched(
        #     model = model,
        #     input_ids = input_ids,
        #     labels = labels,
        #     ignore_label_id = ignored_id,
        #     pad_token_id = pad_token_id,
        #     debug = it < show_nb_first_preds
        # )
        # nb_correct = (preds == labels).sum().item()
        # nb_labels = (labels != ignored_id).sum().item()
        # total_correct += nb_correct
        # total_labels += nb_labels
        # total_loss += loss


        loss_1, preds_1 = simple_val(
            input_ids=input_ids,
            labels=labels
        )
        nb_correct = (preds_1 == labels).sum().item()
        nb_labels = (labels != ignored_id).sum().item()
        nb_total_correct_1 += nb_correct
        nb_total_labels_1 += nb_labels
        total_loss_1 += loss_1
        
        
        if nb_iterative_self_correction_steps > 1:
            res : Dict[str, Union[float, int]] = self_correction_val(
                model=model,
                input_ids=input_ids,
                labels=labels,
                nb_iterative_self_correction_steps=nb_iterative_self_correction_steps,
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

        if do_augmented_inference:
            loss_aug, preds_voted, nb_labels_aug, nb_cor_aug, nb_labels_vote, nb_cor_voting = augmented_inference_batched_with_voting(
                model = model,
                input_ids = input_ids,
                labels = labels,
                ignore_label_id = ignored_id,
                pad_token_id = pad_token_id,
                vocab_size=mcfg.vocab_size,
                debug = False
            )
            nb_total_cor_aug += nb_cor_aug
            nb_total_labels_aug += nb_labels_aug
            total_loss_aug += loss_aug

            nb_total_cor_vote += nb_cor_voting
            nb_total_labels_vote += nb_labels_vote

        if showed < show_nb_first_preds:
            field_area = max_field_width * max_field_width
            plot_batch(
                data=[
                    input_ids[:10, :],
                    labels[:10, :],
                    # preds[:10, field_area:],
                    preds_1[:10, :],
                ] + ([preds_voted[:10, :]] if do_augmented_inference else []),
                height=max_field_width * expand_coeff,
                width=max_field_width,
                show=show_or_just_log_to_wandb,
            )

            showed += 1

        total_steps += 1
        
        
    acc = nb_total_correct / max(nb_total_labels, 1)
    acc_1 = nb_total_correct_1 / max(nb_total_labels_1, 1)
    acc_aug = nb_total_cor_aug / max(nb_total_labels_aug, 1)
    acc_vote = nb_total_cor_vote / max(nb_total_labels_vote, 1)
    sc_acc = sc_nb_total_correct / max(sc_nb_total_labels, 1)
    sc_acc_last = sc_nb_last_correct / max(sc_nb_last_labels, 1)
    
    return {
        "loss":  total_loss / max(total_steps, 1),
        "loss_1": total_loss_1 / max(total_steps, 1),
        'loss_aug': total_loss_aug / max(total_steps, 1),
        'acc': acc,
        'acc_1': acc_1,
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


def main(
    dcfg : Optional[DatasetConfig] = None,
    tcfg : Optional[TrainConfig] = None,
    mcfg : Optional[EncoderConfig] = None,
):
    # Load environment variables
    load_dotenv()
    
    # Initialize wandb
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
    
    field_width = 15
    seq_len = -100
    expand = False
    dcfg = DatasetConfig(
        seed=123,
        val_frac=0.1,
        test_frac=0.0,
        num_samples=1000,
        seq_len=seq_len,
        max_width=field_width
    ) if dcfg is None else dcfg
    tcfg = TrainConfig(
        batch_size=4,
        lr=3e-4
    ) if tcfg is None else tcfg
    mcfg = EncoderConfig(
        d_model=64,
        n_head=8,
        d_head=64,
        num_layers=8,
        dim_feedforward=128,
        vocab_size=200,
        
        max_len=4000,
        
        nb_refinement_steps=1,
        nb_last_trained_steps=1,
        
        use_cnn=False,
        enable_pseudo_diffusion_inner=False,
        enable_pseudo_diffusion_outer=True,
        feed_first_half=False,

        use_transposed_rope_for_2d_vertical_orientation=False,
        field_width=field_width,
        field_height=field_width * 2,

        use_emb_norm = False,

        use_axial_rope = True,
    ) if mcfg is None else mcfg

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
    
    hrm = HRM(
        networks = [
            Encoder(
                dim = mcfg.d_model,
                depth = 8,
                attn_dim_head = mcfg.d_head,
                heads = mcfg.n_head,
                use_rmsnorm = True,
                rotary_pos_emb = True,
                pre_norm = False
            ),
            Encoder(
                dim = mcfg.d_model,
                depth = 8,
                attn_dim_head = mcfg.d_head,
                heads = mcfg.n_head,
                use_rmsnorm = True,
                rotary_pos_emb = True,
                pre_norm = False
            )
        ],
        causal = False,
        num_tokens = 256,
        dim = mcfg.d_model,
        reasoning_steps = 2,
        ignore_index=mcfg.ignore_id
    ).to(tcfg.device)
    # model = hrm
    # model.encoder = Encoder(
    #     dim = mcfg.d_model,
    #     depth = 8,
    #     attn_dim_head = mcfg.d_head,
    #     heads = mcfg.n_head,
    #     use_rmsnorm = True,
    #     rotary_pos_emb = True,
    #     pre_norm = False
    # )
    optimizer = _get_optimizer(model, tcfg)
    atexit.register(get_cleanup_function(model, optimizer=optimizer))
    
    ds_raw = get_custom_ds_arc(
        seq_len=dcfg.seq_len,
        nb_samples=dcfg.num_samples,
        nb_cls=10,
        task='fill_squares_2d',

        field_width = field_width,


        do_2d = True,
        do_transpose = True,
        
        
        nb_missing_min = 20,
        nb_missing_max = 21,
        masked_token_id = mcfg.masked_token_id,
    )
    # ds_raw = get_arc_puzzle_ds_as_flat_ds(
    #     puzzle_name=PuzzleNames.FILL_SIMPLE_OPENED_SHAPE,
    #     nb_samples=dcfg.num_samples,
    #     max_field_width=dcfg.max_width,
    #     pad_token_id=mcfg.pad_token_id,
    #     ignore_label_id=mcfg.ignore_id
    # )
    train_dl, val_dl = get_dataloaders_for_flat_seq_cls(
        ds_raw,
        ignore_label_id = mcfg.ignore_id,
        sep_token_id = mcfg.qa_sep_token_id,
        pad_token_id = mcfg.pad_token_id,
        split_ratio = 1 - dcfg.val_frac,
        batch_size_train = tcfg.batch_size,
        batch_size_eval = tcfg.batch_size,
        device = tcfg.device,
        add_labels_to_inputs=False,
        add_sep=False,
        expand=expand,
    )
    epoch = 0
    while 1 :
        tr = train_one_epoch(
            model,
            train_dl,
            optimizer, 
            device = tcfg.device,
            mcfg=mcfg,
            max_nb_aug = 0,
            
            nb_iterative_self_correction_steps = 7,
        )
        l  = f"Epoch {epoch:02d} | "
        for k, v in tr.items() : 
            l += f"{k} {v:.4f} | "
        log(f"{l}")
        
        # Log training metrics to wandb
        train_metrics = {f"train/{k}": v for k, v in tr.items()}
        train_metrics["epoch"] = epoch
        wandb.log(train_metrics)

        l  = f"Epoch {epoch:02d} | "
        with torch.no_grad():
            va = evaluate(
                model,
                val_dl,
                show_nb_first_preds=1,
                max_field_width=dcfg.max_width,
                mcfg=mcfg,
                device = tcfg.device,
                do_augmented_inference = False,
                expand = expand,
                nb_iterative_self_correction_steps = 15,
                show_or_just_log_to_wandb = False
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
