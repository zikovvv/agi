import itertools
import random
from matplotlib import pyplot as plt
import tqdm
import collections
from typing import List, Set, Tuple
import torch
from common import *
from show import plot_batch

all_permutations = [
    list(itertools.permutations([_ for _ in range(nb)], nb)) for nb in range(1, 11)
]


def get_batch_colors(
    input_ids: torch.Tensor,  # [B, H*W, D]
    labels: torch.Tensor,  # [B, H*W, D]
    ignore_label_id: int,
    pad_token_id: int
) -> Tuple[int, ...]:
    # unique values from labels
    colors = set(color.item() for color in labels.unique().flatten())
    colors |= set(color.item() for color in input_ids.unique().flatten())
    colors.discard(ignore_label_id)
    colors.discard(pad_token_id)
    assert pad_token_id not in colors
    return tuple(colors)


def get_perms_for_batch(
    colors : Tuple[int, ...],
    max_perms : int
) -> List[Tuple[int, ...]]:
    perms = all_permutations[len(colors) - 1]
    if len(perms) > max_perms:
        perms = random.sample(perms, max_perms)
    return perms


def transform_tensor(
    ids: torch.Tensor,  # [B, H*W, D]
    colors: Tuple[int, ...],
    perm: Tuple[int, ...],
    inverse: bool = False
) -> torch.Tensor: 
    assert len(colors) == len(perm)
    ip = ids.clone().to(ids.device)
    for new_color, old_color in zip(*((colors, perm) if inverse else (perm, colors))):
        ip[ids == old_color] = new_color
    return ip

def augment_colors_batch(
    input_ids: torch.Tensor, # [B, H*W, D]
    labels: torch.Tensor, # [B, H*W, D]
    max_permutations: int,
    ignore_label_id : int,
    pad_token_id: int,
    add_orig : bool = False
) -> Tuple[Tuple[int, ...], List[Tuple[int, ...]], torch.Tensor, torch.Tensor]:
    colors = get_batch_colors(input_ids, labels, ignore_label_id, pad_token_id)
    # pop ignore_label_id
    perms = get_perms_for_batch(colors, max_permutations)
    inp_perms, l_perms = [], []
    if add_orig:
        inp_perms, l_perms = [input_ids], [labels]
    for perm in perms:
        inp_perms.append(transform_tensor(input_ids, colors, perm))
        l_perms.append(transform_tensor(labels, colors, perm))

        # assert transform_tensor(transform_tensor(labels, colors, perm), colors, perm, inverse=True).equal(labels)
        # assert transform_tensor(transform_tensor(input_ids, colors, perm), colors, perm, inverse=True).equal(input_ids)

    return colors, perms, torch.concatenate(inp_perms, dim=0), torch.concatenate(l_perms, dim=0)



def vote_for_actual_answer(
    colors_orig : Tuple[int, ...],
    colors_perms: List[Tuple[int, ...]],
    labels_orig: torch.Tensor,
    preds_permutated: List[torch.Tensor],
    vocab_size: int,
    debug : bool = False
) -> torch.Tensor:
    log_debug(f'{labels_orig.shape = }, {preds_permutated[0].shape = }, {vocab_size = }, {len(colors_orig) = }, {len(colors_perms) = }')
    # preds_counter : torch.Tensor = torch.zeros((*labels_orig.shape, vocab_size), dtype=torch.int32)
    # log_debug(f'{preds_counter.shape = }')
    restored_preds = [transform_tensor(p, colors_orig, perm, inverse=True) for p, perm in zip(preds_permutated, colors_perms)]
    # print(f'{restored_preds[0].shape = }')
    restored_preds_one_hot_encoding = [torch.nn.functional.one_hot(p, num_classes=vocab_size) for p in restored_preds]
    # print(f'{restored_preds_one_hot_encoding[0].shape = }')
    preds_counter : torch.Tensor = sum(restored_preds_one_hot_encoding) # type: ignore
    # print(f'{preds_counter.shape = }')
    
    field_width = int((labels_orig.shape[1] // 2)**0.5)
    # print(field_width)
    # exit()
    if debug:
        plot_batch(
            height=field_width,
            width=field_width,
            data = [
                restored_preds[0][:10,field_width ** 2:],
                restored_preds[1][:10,field_width ** 2:],
                restored_preds[2][:10,field_width ** 2:],
            ],
            show_row_labels=True
        )

    preds_voted = preds_counter.argmax(dim=-1)
    return preds_voted

def augmented_inference_batched_with_voting(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    ignore_label_id: int,
    pad_token_id : int,
    debug: bool = False
):
    device = input_ids.device
    log_debug(f'{input_ids.shape = }, {labels.shape = }, {ignore_label_id = }')
    small_batch_size = input_ids.shape[0]
    colors_orig, colors_perms, input_ids_p, labels_p = augment_colors_batch(
        input_ids.cpu(),
        labels.cpu(),
        max_permutations=20,
        ignore_label_id=ignore_label_id,
        pad_token_id=pad_token_id,
        add_orig=False
    )
    input_ids_p = input_ids_p.to(device)
    labels_p = labels_p.to(device)
    log_debug(f'{input_ids_p.shape = }, {labels_p.shape = }, {len(colors_orig) = }, {len(colors_perms) = }')
    res = model(input_ids_p, labels=labels_p)
    loss = res['loss'].item()
    preds_all = res['logits'].argmax(dim=-1)


    field_width = int((labels.shape[1] // 2)**0.5)
    if debug :
        plot_batch(
            height=field_width,
            width=field_width,
            data=[
                input_ids_p[:small_batch_size, :field_width ** 2],
                # input_ids_p[:small_batch_size, field_width ** 2:],
                input_ids_p[small_batch_size:small_batch_size*2, :field_width ** 2],
                # input_ids_p[small_batch_size:small_batch_size*2, field_width ** 2:],
                labels_p[:small_batch_size, field_width ** 2:],
                labels_p[small_batch_size:small_batch_size*2, field_width ** 2:],
                preds_all[:small_batch_size, field_width ** 2:],
                preds_all[small_batch_size:small_batch_size*2, field_width ** 2:],
                # preds_all[:small_batch_size, :field_width ** 2],
                # preds_all[small_batch_size:small_batch_size*2, :field_width ** 2],
            ],
            show_row_labels=True
        )
    assert input_ids_p[:small_batch_size, field_width ** 2:][0][0] == pad_token_id, input_ids_p[:small_batch_size, field_width ** 2:][0][0] 
    assert input_ids_p[small_batch_size:small_batch_size*2, field_width ** 2:][0][0] == pad_token_id, input_ids_p[small_batch_size:small_batch_size*2, field_width ** 2:][0][0]
    assert input_ids_p[:small_batch_size, field_width ** 2:][0][0] == input_ids_p[small_batch_size:small_batch_size*2, field_width ** 2:][0][0]



    #global acc
    nb_l_aug = (labels_p != ignore_label_id).sum().item()
    nb_cor_aug = (preds_all == labels_p).sum().item()
    log_debug(f'{preds_all.shape = }, {labels_p.shape = }, {nb_l_aug = }, {nb_cor_aug = }')

    # split preds
    preds_p_split : List[torch.Tensor] = list(torch.split(preds_all, small_batch_size, dim=0))
    log_debug(f'{len(preds_p_split) = }, {preds_p_split[0].shape = }')
    preds_voted : torch.Tensor = vote_for_actual_answer(
        colors_orig=colors_orig,
        colors_perms=colors_perms,
        labels_orig=labels,
        preds_permutated=preds_p_split,
        vocab_size=model.cfg.vocab_size,
        debug = debug
    )
    nb_l = (labels != ignore_label_id).sum().item()
    nb_voted = (preds_voted == labels).sum().item()

    return loss, preds_voted, nb_l_aug, nb_cor_aug, nb_l, nb_voted




def augmented_inference_batched(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    ignore_label_id: int,
    pad_token_id : int,
    debug: bool = False
):
    log_debug(f'{input_ids.shape = }, {labels.shape = }, {ignore_label_id = }')
    small_batch_size = input_ids.shape[0]
    colors_orig, colors_perms, input_ids_p, labels_p = augment_colors_batch(
        input_ids,
        labels,
        max_permutations=20,
        ignore_label_id=ignore_label_id,
        pad_token_id=pad_token_id,
        add_orig=True
    )
    log_debug(f'{input_ids_p.shape = }, {labels_p.shape = }, {len(colors_orig) = }, {len(colors_perms) = }')
    res = model(input_ids_p, labels=labels_p)
    loss = res['loss'].item()
    preds_all = res['logits'].argmax(dim=-1)


    field_width = int((labels.shape[1] // 2)**0.5)
    if debug :
        plot_batch(
            height=field_width,
            width=field_width,
            data=[
                input_ids_p[:small_batch_size, :field_width ** 2],
                input_ids_p[:small_batch_size, field_width ** 2:],
                input_ids_p[small_batch_size:small_batch_size*2, :field_width ** 2],
                input_ids_p[small_batch_size:small_batch_size*2, field_width ** 2:],
                labels_p[:small_batch_size, field_width ** 2:],
                labels_p[small_batch_size:small_batch_size*2, field_width ** 2:],
                preds_all[:small_batch_size, field_width ** 2:],
                preds_all[small_batch_size:small_batch_size*2, field_width ** 2:],
                # preds_all[:small_batch_size, :field_width ** 2],
                # preds_all[small_batch_size:small_batch_size*2, :field_width ** 2],
            ],
            show_row_labels=True
        )
    assert input_ids_p[:small_batch_size, field_width ** 2:][0][0] == pad_token_id, input_ids_p[:small_batch_size, field_width ** 2:][0][0] 
    assert input_ids_p[small_batch_size:small_batch_size*2, field_width ** 2:][0][0] == pad_token_id, input_ids_p[small_batch_size:small_batch_size*2, field_width ** 2:][0][0]
    assert input_ids_p[:small_batch_size, field_width ** 2:][0][0] == input_ids_p[small_batch_size:small_batch_size*2, field_width ** 2:][0][0]


    # #global acc
    # nb_labels = (labels != ignore_label_id).sum().item()
    # nb_correct_pred = (preds_all[:small_batch_size] == labels).sum().item()
    # accuracy = nb_correct_pred / nb_labels if nb_labels > 0 else 0
    # log_debug(f'{logits.shape = }, {preds_all.shape = }, {labels.shape = }, {accuracy = }, {nb_labels = }, {nb_correct_pred = }')
    
    return loss, preds_all[:small_batch_size]