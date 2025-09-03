import atexit
import json
import tqdm
from encoder.hrm_like_enc.config import DatasetConfig, TrainConfig, EncoderConfig            
from encoder.hrm_like_enc.inference import augment_colors_batch, augmented_inference_batched, augmented_inference_batched_with_voting
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
from show import plot_batch


def train_batch(
    model : nn.Module,
    optimizer : Optional[torch.optim.Optimizer],
    bid : int,
    batch : Dict[str, torch.Tensor],
    nb_steps_to_visualize : int,
    ignored_id : int,
    pad_token_id : int ,
    nb_colors_potential : int, 
    add_orig_to_color_perms : bool,
    no_grad : bool = False,
    nb_max_augments: int = 4,
) :
    assert optimizer is not None or no_grad, "If optimizer is None, no_grad must be True"

    total_correct = 0
    total_classified = 0
    total_loss = 0
    total_steps = 0

    add_orig_to_color_perms =  add_orig_to_color_perms
    color_input_ids : torch.Tensor = batch['input_ids']
    color_labels : torch.Tensor = batch['labels']
    B_ = color_input_ids.shape[0]

    colors_orig, colors_perms, color_input_ids, color_labels = augment_colors_batch(
        color_input_ids,
        color_labels,
        max_permutations=nb_max_augments,
        ignore_label_id=ignored_id,
        pad_token_id=pad_token_id,
        add_orig=add_orig_to_color_perms
    )

    
    B, L = color_input_ids.shape
    D = model.cfg.d_model # type: ignore
    V = 2
    assert color_labels.shape == color_input_ids.shape == (B, L)
    FW = FH = int((color_labels.shape[1] // 2) ** 0.5) # field width, field height (only labels or only inputs)
    FA = FW * FH # field area
    W, H =  FW, FH * 2 # total tensor in 2d height and width
    CP = len(colors_perms) + int(add_orig_to_color_perms)
    NC = len(colors_orig)
    log_debug(f'{B = }, {L = }, {D = }, {V = }, {W = }, {H = }, {FA = }, {FH = }, {FW = }, {B_ = }, {CP = }, {NC = }')
    assert B_ * CP == B


    if bid < nb_steps_to_visualize :
        plot_batch(
            data = [
                color_input_ids[:10, :],
                color_labels[:10, :],
            ],
            height=H,
            width=W
        )
    
    
    main_cls = torch.ones_like(color_input_ids, device=color_labels.device) # 1 is classified right or ignored,0 is bad and needs refinement
    color_labels_ignored_mask = color_labels == ignored_id
    main_cls[~color_labels_ignored_mask] = 0 # we set all to wrong in the beginning
    main_cls_is_wrong_mask = main_cls == 0
    main_color_guess = color_input_ids.clone()

    if bid < nb_steps_to_visualize :
        plot_batch(
            data = [
                main_cls[:10, :],
                main_color_guess[:10, :],
            ],
            height=H,
            width=W
        )
    

    for it in range(NC) : # iterate for nb colors in original sample because we need to fill all of them

        # GENERATING ASSUMPTIONS ABOUT COLORS
        color_guesses, cls_labels_list = [], []
        for color in range(nb_colors_potential) :
            colors_guess = main_color_guess.clone() # we need input to the puzzle to be in context
            colors_guess[main_cls_is_wrong_mask] = color # where we have wrong color in previous step we try to set new color
            assert colors_guess.shape == color_labels.shape == color_input_ids.shape == main_color_guess.shape == (B, L)
            color_guesses.append(colors_guess)
            cls_labels = (colors_guess == color_labels).to(dtype=torch.long) # where guess equals labels bool map 1, 0
            cls_labels[color_labels_ignored_mask] = ignored_id # also set to -100 where input ids are located like in plain color class labels
            assert cls_labels.shape == color_labels.shape == color_input_ids.shape == main_color_guess.shape == (B, L)
            cls_labels_list.append(cls_labels)
        
        # concat for training in single pass in parallel on gpu
        color_guess_cat = torch.cat(color_guesses, dim=0)
        cls_labels_cat = torch.cat(cls_labels_list, dim=0)
        assert color_guess_cat.shape == cls_labels_cat.shape == (B * 10, L)

        # defining the best color for each sample
        counter : torch.Tensor = torch.zeros((B, 10), dtype=torch.int32, device=color_guess_cat.device)
        for color in range(nb_colors_potential) :
            cls_labels : torch.Tensor = cls_labels_list[color].clone()
            cls_labels[cls_labels == ignored_id] = 0
            s = cls_labels.count_nonzero(dim=-1) # [B], count number of color matches where cls is 1
            log_debug(s)
            counter[:, color] += s # add count of matching items in each sample for this color
            log_debug(counter)

        # setting defined color to all bad cells in main guess
        best_colors = list(counter.argmax(dim=-1))# [B] do we care about color that matched the most bad cells or about the less cells, argmax for the max bad cells
        assert len(best_colors) == B
        for j, best_color in enumerate(best_colors):
            main_color_guess[j, main_cls_is_wrong_mask[j]] = best_color # set the best color for the wrong masked positions in this current sample because this particular color fills the most gaps for this particulat sample

        # calculate new main_cls and get new bad cells mask
        main_cls = (main_color_guess == color_labels).int()
        main_cls[color_labels_ignored_mask] = 1 # where original labels are ignoring we dont care
        main_cls_is_wrong_mask = main_cls == 0


        if bid < nb_steps_to_visualize :
            plot_batch(
                data = [
                    main_color_guess[:10, :],
                    main_cls[:10, :],
                    # main_cls_is_wrong_mask[:10, :],
                ],
                height=H,
                width=W
            )
        




        # TRAINING
        # FORWARD PASS
        if not no_grad :
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            
        with torch.set_grad_enabled(not no_grad) :
            res = model(color_guess_cat, cls_labels_cat) # models that classifies if color is correct for the given position

        loss : torch.Tensor = res['loss']
        logits : torch.Tensor = res['logits']
        assert logits.shape == (B * 10, L, V)

        # BACKWARD PASS AND TRAINING
        if not no_grad :
            assert optimizer is not None
            # assert logits.grad is not None
            loss.backward()
            optimizer.step()
        
        
        # METRICS
        cls_preds = logits.argmax(dim=-1)
        assert cls_preds.shape == cls_labels_cat.shape == (B * 10, L)
        nb_corr_cur = (cls_preds == cls_labels_cat).sum().item()
        nb_active = (cls_labels_cat != ignored_id).sum().item()
        total_correct += nb_corr_cur
        total_classified += nb_active
        total_loss += loss.item()
        total_steps += 1

        # VISUALIZE
        if bid < nb_steps_to_visualize :
            plot_batch(
                data = [
                    color_guess_cat[::B, :],
                    cls_labels_cat[::B, :],
                    cls_preds[::B, :],
                ],
                height=H,
                width=W
            )
    return {
        "total_correct": total_correct,
        "total_classified": total_classified,
        "total_loss": total_loss,
        "total_steps": total_steps
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    nb_steps_to_visualize: int = 0,
    nb_colors_potential: int = 10,
    nb_max_augments: int = 4,
    device : str = 'cpu',
) -> Dict[str, float]:
    ignored_id: int = model.cfg.ignore_id  # type: ignore
    pad_token_id: int = model.cfg.pad_token_id  # type: ignore
    ADD_ORIG_TO_COLOR_PERMS = True
    model.train()
    total_steps = 0
    total_loss = 0.0
    total_correct = 0
    total_classified = 0
    for bid, batch in enumerate(tqdm.tqdm(loader, total=len(loader))):
        res = train_batch(
            model = model,
            optimizer = optimizer,
            bid = bid,
            batch = batch,
            nb_steps_to_visualize = nb_steps_to_visualize,
            ignored_id = ignored_id,
            pad_token_id = pad_token_id,
            nb_colors_potential = nb_colors_potential,
            add_orig_to_color_perms = ADD_ORIG_TO_COLOR_PERMS,
            nb_max_augments = nb_max_augments,
            no_grad=False,
        )
        total_correct += res['total_correct']
        total_classified += res['total_classified']
        total_loss += res['total_loss']
        total_steps += res['total_steps']
    avg_loss = total_loss / max(total_steps, 1)
    acc = total_correct / total_classified
    return {"loss": avg_loss, "acc": acc}



@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    nb_steps_to_visualize: int = 0,
    nb_colors_potential: int = 10,
    nb_max_augments: int = 4,
    device : str = 'cpu',
) -> Dict[str, float]:
    ignored_id: int = model.cfg.ignore_id  # type: ignore
    pad_token_id: int = model.cfg.pad_token_id  # type: ignore
    ADD_ORIG_TO_COLOR_PERMS = True
    model.train()
    total_steps = 0
    total_loss = 0.0
    total_correct = 0
    total_classified = 0
    for bid, batch in enumerate(tqdm.tqdm(loader, total=len(loader))):
        res = train_batch(
            model = model,
            optimizer = None,
            bid = bid,
            batch = batch,
            nb_steps_to_visualize = nb_steps_to_visualize,
            ignored_id = ignored_id,
            pad_token_id = pad_token_id,
            nb_colors_potential = nb_colors_potential,
            add_orig_to_color_perms = ADD_ORIG_TO_COLOR_PERMS,
            no_grad=True,
            nb_max_augments = nb_max_augments,
        )
        total_correct += res['total_correct']
        total_classified += res['total_classified']
        total_loss += res['total_loss']
        total_steps += res['total_steps']
    avg_loss = total_loss / max(total_steps, 1)
    acc = total_correct / total_classified
    return {"loss": avg_loss, "acc": acc}
    
def _get_model(mcfg : EncoderConfig, tcfg : TrainConfig) -> EncoderForValidation :
    model = EncoderForValidation(mcfg).to(tcfg.device)
    return model


def _get_optimizer(model : EncoderForValidation, tcfg : TrainConfig) :
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
    field_width = 15
    seq_len = -100
    dcfg = DatasetConfig(
        seed=123,
        val_frac=0.1,
        test_frac=0.0,
        num_samples=6,
        seq_len=seq_len,
        max_width=field_width
    ) if dcfg is None else dcfg
    tcfg = TrainConfig(
        batch_size=2,
        lr=3e-4
    ) if tcfg is None else tcfg
    mcfg = EncoderConfig(
        d_model=128,
        n_head=8,
        d_head=32,
        num_layers=4,
        nb_refinement_steps=1,
        nb_last_trained_steps=1,
        dim_feedforward=256,
        vocab_size=200,
        max_len=4000,
        
        use_cnn=False,
        enable_pseudo_diffusion_inner=False,
        enable_pseudo_diffusion_outer=False,
        feed_first_half=False,

        use_transposed_rope_for_2d_vertical_orientation=False,
        field_width_for_t_rope=field_width,
        field_height_for_t_rope=field_width * 2,
    ) if mcfg is None else mcfg

    # device = tcfg.device
    model = _get_model(mcfg, tcfg)
    optimizer = _get_optimizer(model, tcfg)
    atexit.register(get_cleanup_function(model, optimizer=optimizer))
    
    ds_raw = get_custom_ds_arc(
        seq_len=dcfg.seq_len,
        nb_samples=dcfg.num_samples,
        nb_cls=10,
        task='fill_squares_2d',
        # do_2d = True,
        field_width = field_width,
        # do_transpose = True,
    )
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
        add_sep=False
    )
    epoch = 0
    while 1 :
        tr = train_one_epoch(model, train_dl, optimizer, device = tcfg.device, nb_steps_to_visualize=0, nb_colors_potential=10, nb_max_augments = 4,)
        l  = f"Train Epoch {epoch:02d} | "
        for k, v in tr.items() : 
            l += f"{k} {v:.4f} | "
        log(f"{l}")

        nb_refinement_steps_back = mcfg.nb_refinement_steps
        mcfg.nb_refinement_steps = 4
        l  = f"Val Epoch {epoch:02d} | "
        with torch.no_grad():
            va = evaluate_one_epoch(model, val_dl, nb_steps_to_visualize=1, nb_colors_potential=10, nb_max_augments = 4,device = tcfg.device)
        for k, v in va.items() : 
            l += f"{k} {v:.4f} | "
        log(f"{l}")
        mcfg.nb_refinement_steps = nb_refinement_steps_back

        epoch += 1

    log("Done.")



if __name__ == "__main__":
    main()
    