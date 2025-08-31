import random
from typing import Dict, List, Literal, Optional, Tuple
from matplotlib import pyplot as plt
import numpy as np
import tqdm

from gen_simple_arc_ds import PuzzleNames, gen_arc_puzzle_ex

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



def get_ds_1d_seq_for_random_input_with_some_transformation_for_output(
    seq_len : int,
    nb_samples : int,
    nb_cls : int,
    task: Literal['copy', 'reverse', 'random', 'color_permutation', 'split_in_8_pieces_and_recombine'],
    color_permutation : Optional[Dict[int, int]] = None,
) -> List[Tuple[np.ndarray, np.ndarray]] :
    seq = np.random.randint(0, nb_cls, size=(nb_samples, seq_len))
    if task == 'random':
        rl = np.random.randint(0, nb_cls, size=(nb_samples, seq_len))
        return [(s, r) for s, r in zip(seq, rl)]
    if task == 'reverse':
        return [(s, np.array(list(s)[::-1])) for s in seq]
    if task == 'color_permutation':
        if color_permutation is None:
            raise ValueError("color_permutation must be provided for color_permutation task")
        return [(s, np.vectorize(color_permutation.get)(s)) for s in seq]
    if task == 'split_in_8_pieces_and_recombine':
        res = []
        for s in seq:
            order = [0, 5, 4, 1, 6, 2, 7, 3, 3, 3, 4, 7, 4]
            splits = np.split(s, 8)
            splits = [splits[i] for i in order]
            res.append((s, np.concatenate(splits)))
        return res

    return [(s, s) for s in seq]



def get_ds_arc_for_1d(
    seq_len : int,
    nb_samples : int,
    nb_cls : int,
    task: Literal['fill_between_pieces'],
    # color_permutation : Optional[Dict[int, int]] = None,
) -> List[Tuple[np.ndarray, np.ndarray]] :
    match task:
        case 'fill_between_pieces':
            bg_colors = [0, 4, 6]
            figure_to_filling_color :Dict[int, int] = {
                1:3, 2:5, 3:7, 5:9
            }
            figure_colors = [1, 2, 3, 5]

            fig_shape = np.array([1, 0, 0, 0, 1])
            in_seq = np.zeros((nb_samples, seq_len))
            out_seq = np.zeros((nb_samples, seq_len))
            for i in range(nb_samples):
                bg_color = random.choice(bg_colors)
                in_seq[i] = np.full((seq_len,), bg_color)
                out_seq[i] = np.full((seq_len,), bg_color)
                offset = 0
                while 1 : 
                    fig_color = random.choice(figure_colors)
                    filling = figure_to_filling_color[fig_color]
                    fig_in = fig_shape.copy() * fig_color
                    fig_out = fig_in.copy()
                    fig_out[1:-1] = filling
                    fig_in[1:-1] = bg_color
                    offset += random.randint(1, seq_len // 5)
                    if offset < seq_len - fig_in.shape[0]:
                        in_seq[i, offset:offset + fig_in.shape[0]] = fig_in
                        out_seq[i, offset:offset + fig_in.shape[0]] = fig_out
                        offset += len(fig_in)
                    else:
                        break
            return [(i, o) for i, o in zip(in_seq, out_seq)]
    raise ValueError("Unknown task")

def get_arc_puzzle_ds_as_flat_ds(
    puzzle_name : PuzzleNames,
    nb_samples : int,
    max_field_width : int,
    pad_token_id : int,
    ignore_label_id : int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Get the dataset for the specified puzzle
    ds = gen_arc_puzzle_ex(puzzle_name, nb_examples=nb_samples, augment_colors=False, do_shuffle=False)
    res = []
    for q, a in ds :
        input_ids_field = np.full((max_field_width, max_field_width), pad_token_id, dtype=np.int32)
        labels_field = np.full((max_field_width, max_field_width), ignore_label_id, dtype=np.int32)
        input_ids_field[:q.shape[0], :q.shape[1]] = q
        labels_field[:a.shape[0], :a.shape[1]] = a
        # input_ids_field = np.full((max_field_width * 2, max_field_width), pad_token_id, dtype=np.int32)
        # input_ids_field[:q_field.shape[0], :q_field.shape[1]] = q_field
        # labels_field = np.full((max_field_width * 2, max_field_width), ignore_label_id, dtype=np.int32)
        # labels_field[max_field_width:max_field_width + a_field.shape[0], :a_field.shape[1]] = a_field
        assert input_ids_field.shape == labels_field.shape
        # assert they are flat vectors both
        assert input_ids_field.flatten().shape == labels_field.flatten().shape

        res.append((input_ids_field.flatten(), labels_field.flatten()))
        
        # print(labels_field.flatten().shape)
        # plt.subplot(1, 2, 1)
        # plt.imshow(input_ids_field, aspect='auto')
        # plt.title("Input IDs")
        # plt.subplot(1, 2, 2)
        # plt.imshow(labels_field, aspect='auto')
        # plt.title("Labels")
        # plt.show()
    return res


# get_arc_puzzle_ds_as_flat_ds(
#     PuzzleNames.FILL_SIMPLE_OPENED_SHAPE,
#     nb_samples=10,
#     max_field_width=40,
#     pad_token_id=16,
#     ignore_label_id=16
# )

# res = get_ds_arc_for_1d(
#     seq_len=20,
#     nb_samples=30,
#     nb_cls=10,
#     task='fill_between_pieces'
# )
# print(res)