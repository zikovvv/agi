


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
