import atexit
import json
from pyexpat import model
import tqdm
from encoder.enc_dec.config import DatasetConfig, TrainConfig, EncoderConfig            
from encoder.enc_dec.inference import augment_colors_batch, augmented_inference_batched, augmented_inference_batched_with_voting
from gen_simple_arc_ds import PuzzleNames
from get_dataloader_for_model_for_task import get_dataloaders_for_flat_seq_cls, get_dataloaders_for_encoder_masked_modeling
from get_ds_for_task import get_arc_puzzle_ds_as_flat_ds, get_ds_1d_seq_for_random_input_with_some_transformation_for_output, get_custom_ds_arc, get_ds_for_masked_modeling_only_answer, get_ds_for_masked_modeling_only_answer_only_foreground_items
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
from show import plot_eval_batch

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    max_field_width: int,
    device: str = 'cpu',
) -> Dict[str, float]:
    ignored_id: int = model.cfg.ignore_id  # type: ignore
    pad_token_id: int = model.cfg.pad_token_id  # type: ignore

    model.train()
    total = 0
    total_loss = 0.0
    total_correct = 0
    total_classified = 0

    field_area = max_field_width * max_field_width

    for batch in tqdm.tqdm(loader, total=len(loader)):
        input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
        input_ids = input_ids[:, :field_area]
        labels = labels[:, field_area:]

        # colors_orig, colors_perms, input_ids, labels = augment_colors_batch(
        #     input_ids,
        #     labels,
        #     max_permutations=5,
        #     ignore_label_id=ignored_id,
        #     pad_token_id=pad_token_id,
        #     add_orig=True
        # )

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
    max_field_width : int,
    show_nb_first_preds : int,
    device : str = 'cpu',
) -> Dict[str, float]:
    def simple_inference(
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

    ignored_id: int = model.cfg.ignore_id  # type: ignore
    pad_token_id: int = model.cfg.pad_token_id # type: ignore
    model.eval()
    total_steps = 0


    total_loss = 0.0
    total_correct = 0
    total_labels = 0

    total_loss_1 = 0.0
    total_correct_1 = 0
    total_labels_1 = 0


    total_labels_aug = 0
    total_labels_vote = 0
    total_loss_aug = 0.0

    total_cor_vote = 0
    total_cor_aug = 0

    field_area = max_field_width * max_field_width

    showed = 0
    for it, batch in enumerate(loader):
        input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
        input_ids = input_ids[:, :field_area]
        labels = labels[:, field_area:]


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


        loss_1, preds_1 = simple_inference(
            input_ids=input_ids,
            labels=labels
        )
        nb_correct = (preds_1 == labels).sum().item()
        nb_labels = (labels != ignored_id).sum().item()
        total_correct_1 += nb_correct
        total_labels_1 += nb_labels
        total_loss_1 += loss_1

        # loss_aug, preds_voted, nb_labels_aug, nb_cor_aug, nb_labels_vote, nb_cor_voting = augmented_inference_batched_with_voting(
        #     model = model,
        #     input_ids = input_ids,
        #     labels = labels,
        #     ignore_label_id = ignored_id,
        #     pad_token_id = pad_token_id,
        #     debug = it < show_nb_first_preds
        # )
        # total_cor_aug += nb_cor_aug
        # total_labels_aug += nb_labels_aug
        # total_loss_aug += loss_aug

        # total_cor_vote += nb_cor_voting
        # total_labels_vote += nb_labels_vote

        if showed < show_nb_first_preds:
            plot_eval_batch(
            [
                input_ids[:10, :],
                labels[:10, :],
                # preds[:10, field_area:],
                preds_1[:10, :],
                # preds_voted[:10, field_area:]
            ], max_field_width=max_field_width) # type: ignore
            showed += 1

        total_steps += 1
        
        
    acc = total_correct / max(total_labels, 1)
    acc_1 = total_correct_1 / max(total_labels_1, 1)
    acc_aug = total_cor_aug / max(total_labels_aug, 1)
    acc_vote = total_cor_vote / max(total_labels_vote, 1)
    return {
        "loss":  total_loss / max(total_steps, 1),
        "loss_1": total_loss_1 / max(total_steps, 1),
        'loss_aug': total_loss_aug / max(total_steps, 1),
        'acc': acc,
        'acc_1': acc_1,
        'acc_aug': acc_aug,
        'acc_vote': acc_vote
    }


def _get_model(mcfg_e : EncoderConfig, mcfg_d : EncoderConfig, tcfg : TrainConfig) -> EncoderDecoder :
    model = EncoderDecoder(mcfg_e, mcfg_d).to(tcfg.device)
    return model


def _get_optimizer(model : EncoderDecoder, tcfg : TrainConfig) :
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    return optimizer
    

logger.remove() #remove the old handler. Else, the old one will work along with the new one you've added below'
# logger.add(sys.stderr, level="DEBUG") 
logger.add(sys.stderr, level="INFO") 


def main(
):
    field_width = 15
    seq_len = -100
    dcfg = DatasetConfig(
        seed=123,
        val_frac=0.1,
        test_frac=0.0,
        num_samples=200,
        seq_len=seq_len,
        max_width=field_width
    )
    tcfg = TrainConfig(
        batch_size=8,
        lr=3e-4
    )
    mcfg_e = EncoderConfig(
        d_model=256,
        n_head=8,
        d_head=64,
        num_layers=4,
        nb_refinement_steps=1,
        nb_last_trained_steps=1,
        dim_feedforward=512,
        vocab_size=200,
        max_len=4000,
        use_cnn=False,
        enable_pseudo_diffusion_inner=False,
        enable_pseudo_diffusion_outer=False,
        feed_first_half=False,

        is_encoder=True,
        nb_info_tokens=50,

        use_transposed_rope_for_2d_vertical_orientation=False,
        field_width_for_t_rope=field_width,
        field_height_for_t_rope=field_width * 2,
    )
    mcfg_d = EncoderConfig(
        d_model=256,
        n_head=8,
        d_head=64,
        num_layers=8,
        nb_refinement_steps=1,
        nb_last_trained_steps=1,
        dim_feedforward=512,
        vocab_size=200,
        max_len=4000,
        use_cnn=False,
        enable_pseudo_diffusion_inner=False,
        enable_pseudo_diffusion_outer=False,
        feed_first_half=False,

        is_encoder=False,

        use_transposed_rope_for_2d_vertical_orientation=False,
        field_width_for_t_rope=field_width,
        field_height_for_t_rope=field_width * 2,
    )
    # device = tcfg.device
    model : EncoderDecoder = _get_model(mcfg_e, mcfg_d, tcfg)
    optimizer = _get_optimizer(model, tcfg)
    atexit.register(get_cleanup_function(model, optimizer=optimizer))
    
    ds_raw = get_custom_ds_arc(
        seq_len=dcfg.seq_len,
        nb_samples=dcfg.num_samples,
        nb_cls=10,
        task='fill_squares_2d',
        field_width = field_width,
    )
    train_dl, val_dl = get_dataloaders_for_flat_seq_cls(
        ds_raw,
        ignore_label_id = mcfg_e.ignore_id,
        sep_token_id = mcfg_e.qa_sep_token_id,
        pad_token_id = mcfg_e.pad_token_id,
        split_ratio = 1 - dcfg.val_frac,
        batch_size_train = tcfg.batch_size,
        batch_size_eval = tcfg.batch_size,
        device = tcfg.device,
        add_sep=False
    )
    epoch = 0
    while 1 :
        tr = train_one_epoch(model, train_dl, optimizer, max_field_width=field_width, device=tcfg.device)
        l  = f"Epoch {epoch:02d} | "
        for k, v in tr.items() : 
            l += f"{k} {v:.4f} | "
        log(f"{l}")

        # nb_refinement_steps_back = mcfg.nb_refinement_steps
        # mcfg.nb_refinement_steps = 4
        l  = f"Epoch {epoch:02d} | "
        with torch.no_grad():
            va = evaluate(model, val_dl, show_nb_first_preds=1, max_field_width=dcfg.max_width, device = tcfg.device)
        for k, v in va.items() : 
            l += f"{k} {v:.4f} | "
        log(f"{l}")
        # mcfg.nb_refinement_steps = nb_refinement_steps_back

        epoch += 1

    log("Done.")



if __name__ == "__main__":
    main()
    