from typing import List, Tuple
import numpy as np
import tqdm

def get_ds_for_masked_modeling_only_answer(
    ds : List[Tuple[np.ndarray, np.ndarray]],
    percentage_masked : float,
    masked_token_id : int,
) -> List[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] :
    new_ds = []
    for q, a in tqdm.tqdm(ds) :
        a_copy = a.copy()
        mask = np.random.rand(*a.shape) < percentage_masked
        a_copy[mask] = masked_token_id
        new_ds.append(((q, a_copy), a))
    return new_ds


def get_ds_for_masked_modeling_only_answer_only_foreground_items(
    ds : List[Tuple[np.ndarray, np.ndarray]],
    percentage_masked : float,
    masked_token_id : int,
) -> List[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] :
    new_ds = []
    for q, a in tqdm.tqdm(ds) :
        a_copy = a.copy()
        # the most common background color
        background_color = np.bincount(a_copy.flatten(), minlength=256).argmax()
        mask_foreground = (a_copy != background_color)
        # select random pixels from mask_foreground
        mask = np.random.rand(*a_copy.shape) < percentage_masked
        mask = mask & mask_foreground
        a_copy[mask] = masked_token_id
        new_ds.append(((q, a_copy), a))
    return new_ds

def get_ds_for_copy_question(
    ds : List[Tuple[np.ndarray, np.ndarray]],
    masked_token_id : int,
) -> List[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] :
    new_ds = []
    for q, _ in tqdm.tqdm(ds) :
        empty_a = np.full_like(q, masked_token_id)
        new_ds.append(((q, empty_a), q))
    return new_ds



def get_ds_for_simple_input_output_flat_seuqences(
    seq_len : int,
    nb_samples : int,
    nb_cls : int,
    reverse_labels : bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]] :
    seq = np.random.randint(0, nb_cls, size=(nb_samples, seq_len))
    if reverse_labels:
        return [(s, np.array(list(s)[::-1])) for s in seq]
    return [(s, s) for s in seq]