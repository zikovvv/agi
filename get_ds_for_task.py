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



def get_custom_ds_arc(
    seq_len : int,
    nb_samples : int,
    nb_cls : int,
    task: Literal[
        'fill_between_pieces',
        'fill_between_pieces_with_color_from_example',
        'fill_squares_2d',
        'sudoku'
    ],
    **kwargs,
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
        case 'fill_between_pieces_with_color_from_example': 
            do_2d = False
            if 'do_2d' in kwargs :
                do_2d = True
            field_width = None
            do_transpose = False
            if do_2d :
                field_width = kwargs.get('field_width', 15)
                seq_len = field_width * field_width
                do_transpose = kwargs.get('do_transpose', False)
            bg_colors = [0, 4, 6, 9]
            figure_colors = [1, 2, 3, 5, 7]

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
                    filling = random.choice(figure_colors)
                    fig_in = fig_shape.copy() * fig_color
                    fig_out = fig_in.copy()
                    fig_out[1:-1] = filling
                    fig_in[1:-1] = bg_color
                    example_pos_in_in = random.randint(0, fig_shape.shape[0] - 3)
                    fig_in[example_pos_in_in + 1] = filling  # add example of filling color in the input
                    offset += random.randint(1, seq_len // 5)
                    if offset < seq_len - fig_in.shape[0]:
                        in_seq[i, offset:offset + fig_in.shape[0]] = fig_in
                        out_seq[i, offset:offset + fig_in.shape[0]] = fig_out
                        offset += len(fig_in)
                    else:
                        break
                if do_transpose:
                    in_seq[i] = in_seq[i].reshape(field_width, field_width).T.flatten()
                    out_seq[i] = out_seq[i].reshape(field_width, field_width).T.flatten()
            return [(i, o) for i, o in zip(in_seq, out_seq)]
        case 'fill_squares_2d' :
            seq_len = kwargs.get('field_width', None) # type: ignore
            assert seq_len is not None, "field_width must be specified"
            def get_np_square(size: int) : 
                top_line = np.array([1] * size)
                middle = np.array([1] + [0] * (size - 2) + [1])
                return np.vstack([top_line, *(middle for _ in range(size - 2)), top_line])

            def fill_shape(shape, color, bg_color) :
                # shape is 1 for borders and 0 in the middle
                filled_shape = shape.copy()
                filled_shape[shape == bg_color] = color
                return filled_shape
            
            def add_example_color_to_shape_in_random_place(shape, example_color, bg_color):
                # pick a random point where color is bg_color
                empty_positions = np.argwhere(shape == bg_color)
                if empty_positions.size > 0:
                    pos = random.choice(empty_positions)
                    shape[tuple(pos)] = example_color
                return shape

            def get_next_pos_for_shape(fig_width, taken_coords) :
                # randomly choose coords until find free space
        
                def overlap(a1, a2, b1, b2) :
                    assert a1 < a2
                    assert b1 < b2
                    # l = [a1, a2, b1, b2]
                    # l.sort()
                    return (a1 <= b1 <= a2 <= b2) or \
                            (a1 <= b1 <= b2 <= a2) or \
                            (b1 <= a1 <= a2 <= b2) or \
                            (b1 <= a1 <= b2 <= a2)

                for trial in range(3) :
                    x, y = random.randint(0, seq_len - fig_width), random.randint(0, seq_len - fig_width)
                    if not taken_coords :
                        return x, y
                    # else :
                        # print(taken_coords)
                    ov = False
                    for (x_taken, y_taken), fig_w in taken_coords :
                        # print(fig_w, fig_width)
                        x_overlap : bool = overlap(x, x + fig_width, x_taken, x_taken + fig_w)
                        y_overlap : bool = overlap(y, y + fig_width, y_taken, y_taken + fig_w)
                        if x_overlap and y_overlap :
                            ov = True
                            break
                    if not ov :
                        return x, y

                return None, None
                # raise ValueError("No valid position found")

            fig_shapes = [
                # get_np_square(3),
                get_np_square(4),
                get_np_square(5),
                get_np_square(6),
            ]
            # bg_colors = [0, 4, 6, 9]/
            # figure_colors = [1, 2, 3, 5, 7]
            bg_colors = list(range(1, 10))
            figure_colors = list(range(1, 10))

            # print(f'{fig_shapes = }')
            res = []
            for i in range(nb_samples):
                # print(i)
                taken_coords : List[Tuple[Tuple[int, int], int]] = []# (x, y), fig_w
                bg_color = random.choice(bg_colors)
                in_field = np.full((seq_len, seq_len), bg_color)
                out_field = np.full((seq_len, seq_len), bg_color)
                in_f_copy = in_field.copy()
                offset = 0
                while 1 : 
                    while 1 :
                        fig_color = random.choice(figure_colors)
                        if fig_color != bg_color:
                            break
                    assert fig_color != bg_color
                    filling = random.choice(figure_colors)
                    fig_shape = random.choice(fig_shapes)
                    fig_width = fig_shape.shape[0]

                    fig_in = fig_shape.copy() * fig_color
                    assert fig_in[0][0] != bg_color
                    
                    fig_in = add_example_color_to_shape_in_random_place(fig_in, filling, 0)
                    assert fig_in[0][0] != bg_color
                    
                    fig_in = fill_shape(fig_in, bg_color, 0)
                    assert fig_in[0][0] != bg_color
                    
                    
                    fig_out = fig_in.copy()
                    fig_out = fill_shape(fig_out, filling, bg_color)
                    assert fig_out[0][0] == fig_in[0][0] 
                    assert fig_out[0][0] != bg_color
                    assert fig_in[0][0] != bg_color
                    assert fig_in[0][0] != in_f_copy[0][0]
                    assert fig_out[0][0] != in_f_copy[0][0]
                    
                    newx, newy = get_next_pos_for_shape(fig_width, taken_coords)
                    if newx is not None  and newy is not None:
                        taken_coords.append(((newx, newy), fig_width))
                        in_field[newx:newx + fig_width, newy:newy + fig_width] = fig_in
                        out_field[newx:newx + fig_width, newy:newy + fig_width] = fig_out
                    else :
                        break
                res.append((in_field.flatten(), out_field.flatten()))
            return res
        case 'sudoku' :
            from sudoku.engine import _complete_solution
            inuts = [_complete_solution() - 1 for _ in range(nb_samples)] # 0-8 instead of 1-9
            nb_missing_min, nb_missing_max = kwargs.get('nb_missing_min', None), kwargs.get('nb_missing_max', None)
            masked_token_id = kwargs.get('masked_token_id', None)
            assert masked_token_id is not None, "masked_token_id must be specified for sudoku task"
            assert nb_missing_min is not None, "n_missing must be specified for sudoku task"
            assert nb_missing_max is not None, "n_missing_max must be specified for sudoku task"
            assert nb_missing_min <= nb_missing_max, f'{nb_missing_min = }, {nb_missing_max = }]' 
            masked = []
            for p in inuts :
                p_copy = p.copy()
                # remove some numbers
                n_remove = random.randint(nb_missing_min, nb_missing_max)
                mask = np.random.choice(81, n_remove, replace=False)
                mask_2d = np.unravel_index(mask, (9, 9))
                p_copy[mask_2d] = masked_token_id
                masked.append(p_copy)
            #
            # print(inuts[0], masked[0])
            # print(inuts[1], masked[1])
            
            # exit()
            return [(m.flatten(), s.flatten()) for m, s in zip(masked, inuts)]

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

if 0 :
    w = 20
    res = get_custom_ds_arc(
        seq_len=w,
        nb_samples=30,
        nb_cls=10,
        task='fill_squares_2d',
        # do_2d = True,
        field_width = w,
        # do_transpose = True
    )
    def print_fields(q, a) :
        q_f, a_f = q.reshape(w, w), a.reshape(w, w)
        plt.subplot(1, 2, 1)
        plt.imshow(q_f, aspect='auto')
        plt.title("Input")
        plt.subplot(1, 2, 2)
        plt.imshow(a_f, aspect='auto')
        plt.title("Output")
        plt.show()
        
    for q, a in res[:3]:
        # print(q)
        print_fields(q, a)