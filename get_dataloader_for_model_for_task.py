from typing import Dict, List, Set, Tuple
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from gen_simple_arc_ds import gen_arc_puzzle_ex, PuzzleNames
from get_ds_for_task import get_ds_1d_seq_for_random_input_with_some_transformation_for_output, get_ds_for_masked_modeling_only_answer, get_ds_for_masked_modeling_only_answer_only_foreground_items

class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

def get_dataloaders_for_encoder_masked_modeling(
    ds : List[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]],
    masked_token_id : int,
    ignore_label_id : int,
    newline_token_id : int,
    max_grid_width : int,
    max_grid_height : int,
    question_token_type_id : int,
    answer_token_type_id : int,
    pad_token_id : int,
    pad_row_id : int,
    pad_col_id : int,
    pad_token_type_id : int,
    split_ratio : float,
    batch_size_train : int,
    batch_size_eval : int,
    device : str
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    def grid_to_seq(grid : np.ndarray, current_token_type_id : int) -> Tuple[List[int], List[int], List[int], List[int]] :
        H, W = grid.shape
        assert H <= max_grid_height
        assert W <= max_grid_width
        seq_ids = []
        seq_rows = []
        seq_cols = []
        seq_types = []
        for r in range(H) :
            for c in range(W) :
                tok = int(grid[r, c])
                seq_ids.append(tok)
                seq_rows.append(r)
                seq_cols.append(c)
                seq_types.append(current_token_type_id)
            # newline has to also be embedded with position
            seq_ids.append(newline_token_id)
            seq_rows.append(r)
            seq_cols.append(W)
            seq_types.append(current_token_type_id)  # SPECIAL
        return seq_ids, seq_rows, seq_cols, seq_types
    
    def input_as_seq(
        q : np.ndarray,
        a : np.ndarray,
        l : np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] :
        q_ids, q_rows, q_cols, q_types = grid_to_seq(q, question_token_type_id)
        a_ids, a_rows, a_cols, a_types = grid_to_seq(a, answer_token_type_id)
        label_ids, *_ = torch.tensor(grid_to_seq(l, -1))
        input_ids = torch.tensor(q_ids + a_ids, dtype=torch.long)
        rows = torch.tensor(q_rows + a_rows, dtype=torch.long)
        cols = torch.tensor(q_cols + a_cols, dtype=torch.long)
        types = torch.tensor(q_types + a_types, dtype=torch.long)
        # label ids are -100 everywhere except where input_ids is pad_token
        # count number of masked_token_id in a and in a_ids
        # nb_a = (a == masked_token_id).sum()
        # nb_a_ids = (torch.a_ids == masked_token_id).sum()
        # print(nb_a, nb_a_ids)
        # print(a_ids != masked_token_id)
        label_ids[torch.tensor(a_ids) != masked_token_id] = -100
        label_ids_complete = torch.full_like(input_ids, ignore_label_id)
        label_ids_complete[len(q_ids):] = label_ids
        return (
            input_ids,
            rows,
            cols,
            types,
            label_ids_complete,
        )
    
    def sample_as_sequences(
        q : np.ndarray,
        a : np.ndarray,
        l : np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids, rows, cols, types, labels = input_as_seq(q, a, l)
        return input_ids, rows, cols, types, labels

    def get_as_sequences(
        # ds : List[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] :
        """
        Convert each ( (q_grid, a_grid), ans ) into (token_seq, row_seq, col_seq, type_seq, label_seq) of same length
        """
        out_ds = []
        for (q, a), l in ds :
            out_ds.append(sample_as_sequences(q, a, l))
        return out_ds
    
    def collate_fn(
        batch : List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        # do_assert : bool = True
    ) -> Dict[str, torch.Tensor] :
        B = len(batch)
        #assert same lengths
        if True :
            for x in batch :
                tl = x[0].shape[0]
                for t in x : 
                    assert t.shape[0] == tl
            
        max_len = max(x[0].shape[0] for x in batch)
        token_ids_pad : torch.Tensor = torch.full((B, max_len), pad_token_id, dtype=torch.long)
        rows_pad = torch.full((B, max_len), pad_row_id, dtype=torch.long)
        cols_pad = torch.full((B, max_len), pad_col_id, dtype=torch.long)
        types_pad = torch.full((B, max_len), pad_token_type_id, dtype=torch.long)
        labels_pad = torch.full((B, max_len), ignore_label_id, dtype=torch.long) # padding labels and ignore labels is literally the same because loss function will treat them equally

        for i, (token_ids, rows, cols, types, labels) in enumerate(batch):
            seq_len = token_ids.shape[0]
            token_ids_pad[i, :seq_len] = token_ids
            rows_pad[i, :seq_len] = rows
            cols_pad[i, :seq_len] = cols
            types_pad[i, :seq_len] = types
            labels_pad[i, :seq_len] = labels

        return {
            "input_ids": token_ids_pad.to(device),
            "rows": rows_pad.to(device),
            "cols": cols_pad.to(device),
            "token_type_ids": types_pad.to(device),
            "labels": labels_pad.to(device)
        }
    
    ds_seq = get_as_sequences()

    dataset = SimpleDataset(ds_seq)

    # Create train/validation split
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create dataloaders with custom collate function
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        collate_fn=collate_fn
    )

    def test_dataloader(dl) :
        for el in dl:
            print(el)
            return
    # test_dataloader(train_dataloader)
    
    return train_dataloader, val_dataloader

def ex1 ():
    def plot_batch_data(batch : Dict[str, torch.Tensor]) :
        import matplotlib.pyplot as plt
        B, L = batch['input_ids'].shape
        MAX_COLOR = 30
        for i in range(B) :
            input_ids = batch['input_ids'][i].numpy()
            rows = batch['rows'][i].numpy()
            cols = batch['cols'][i].numpy()
            types = batch['token_type_ids'][i].numpy()
            labels = batch['labels'][i].numpy()
            # reconstruct grids from sequences
            # max_row = rows.max() + 1
            # max_col = cols.max() + 1
            q_grid = np.full((40, 40), MAX_COLOR)
            a_grid = np.full((40, 40), MAX_COLOR)
            l_grid = np.full((40, 40), MAX_COLOR)
            cols_grid = np.full((40, 40), MAX_COLOR)
            rows_grid = np.full((40, 40), MAX_COLOR)
            types_grid = np.full((40, 40), MAX_COLOR)
            # assert labels has somtehing except -100
            assert (labels != -100).sum() > 0
            for inp, r, c, t, l in zip(input_ids, rows, cols, types, labels):
                if r != -100  and c != -100 :
                    if t == 0 :
                        q_grid[r, c] = inp
                    else :
                        a_grid[r, c] = inp
                        if l != -100 :
                            l_grid[r, c] = l
                    cols_grid[r, c] = c
                    rows_grid[r, c] = r
                    types_grid[r, c] = t

            fig, axs = plt.subplots(3, 2, figsize=(6, 12))

            axs[0, 0].imshow(q_grid, cmap='tab20', vmin=0, vmax=MAX_COLOR)
            axs[0, 0].set_title(f'Example {i} - Questions')

            axs[1, 0].imshow(a_grid, cmap='tab20', vmin=0, vmax=MAX_COLOR)
            axs[1, 0].set_title(f'Example {i} - Answers')

            axs[2, 0].imshow(l_grid, cmap='tab20', vmin=0, vmax=MAX_COLOR)
            axs[2, 0].set_title(f'Example {i} - Labels')

            axs[0, 1].imshow(types_grid, cmap='tab20', vmin=0, vmax=MAX_COLOR)
            axs[0, 1].set_title(f'Example {i} - Types')

            axs[1, 1].imshow(rows_grid, cmap='tab20', vmin=0, vmax=MAX_COLOR)
            axs[1, 1].set_title(f'Example {i} - Rows')

            axs[2, 1].imshow(cols_grid, cmap='tab20', vmin=0, vmax=MAX_COLOR)
            axs[2, 1].set_title(f'Example {i} - Cols')

            plt.show()

    masked_token_id = 15
    newline_token_id = 12
    pad_token_id = 18
    raw_ds = get_ds_for_masked_modeling_only_answer_only_foreground_items(
        gen_arc_puzzle_ex(
            name = PuzzleNames.FILL_SIMPLE_OPENED_SHAPE,
            nb_examples = 100,
            augment_colors = False,
            do_shuffle = False
        ),
        0.15,
        masked_token_id
    )
    # _l = raw_ds[0][1]
    # print(_l)
    # assert (_l != masked_token_id).sum() > 0
    train_dl, val_dl = get_dataloaders_for_encoder_masked_modeling(
        raw_ds,
        masked_token_id = masked_token_id,
        ignore_label_id = -100,
        newline_token_id = newline_token_id,
        max_grid_width = 40,
        max_grid_height = 40,
        question_token_type_id = 0,
        answer_token_type_id = 1,
        pad_token_id = pad_token_id,
        pad_row_id = -100,
        pad_col_id = -100,
        pad_token_type_id = -100,
        split_ratio = 0.8,
        batch_size_train = 4,
        batch_size_eval = 4,
        device = "cpu"# if torch.cuda.is_available() else "cpu"
    )
    for batch in val_dl : 
        print(batch)
        assert len(batch[next(iter(batch.keys()))]) == 4
        plot_batch_data(batch)
        return


def get_dataloaders_for_cnn_masked_modeling(
    ds_raw : List[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]],
    masked_token_id : int,
    ignore_label_id : int,
    newline_token_id : int,
    max_grid_width : int,
    max_grid_height : int,
    pad_token_id : int,
    split_ratio : float,
    batch_size_train : int,
    batch_size_eval : int,
    only_answer : bool,
    pad_to_max: bool,
    device : str
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:


    def sample_as_input_output_pair(q : torch.Tensor, a : torch.Tensor, l : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] :
        if only_answer :
            l_copy = l.clone()
            l_mask = a != masked_token_id
            l_copy[l_mask] = ignore_label_id
            return a, l_copy
        
        else :
            qa_field = torch.full((max_grid_height * 2, max_grid_width), pad_token_id, dtype=torch.long)
            labels_field = torch.full((max_grid_height * 2, max_grid_width), ignore_label_id, dtype=torch.long)

            # Copy question and answer into the fields
            qa_field[:q.shape[0], :q.shape[1]] = q
            qa_field[max_grid_height:max_grid_height + a.shape[0], :a.shape[1]] = a
            labels_field[max_grid_height:max_grid_height + l.shape[0], :l.shape[1]] = l
            l_copy = labels_field.clone()
            l_mask = qa_field != masked_token_id
            l_copy[l_mask] = ignore_label_id
            return qa_field, l_copy

    def get_as_input_and_labels_pairs() -> List[Tuple[torch.Tensor, torch.Tensor]] :
        res = []
        for (q, a), l in ds_raw :
            res.append(sample_as_input_output_pair(
                torch.tensor(q, dtype=torch.long),
                torch.tensor(a, dtype=torch.long),
                torch.tensor(l, dtype=torch.long)
            ))
        return res
    
    def collate_fn(
        batch : List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor] :
        B = len(batch)

        max_h = max(x[0].shape[0] for x in batch)
        max_w = max(x[0].shape[1] for x in batch)

        height_koef = 1 if only_answer else 2
        assert max_h <= max_grid_height * height_koef
        assert max_w <= max_grid_width
        if pad_to_max :
            max_h = max_grid_height * height_koef
            max_w = max_grid_width
        input_ids_base : torch.Tensor = torch.full((B, max_h, max_w), pad_token_id, dtype=torch.long)
        labels_base = torch.full((B, max_h, max_w), ignore_label_id, dtype=torch.long) # padding labels and ignore labels is literally the same because loss function will treat them equally

        for i, (token_ids, labels) in enumerate(batch):
            h, w = token_ids.shape
            input_ids_base[i, :h, :w] = token_ids
            labels_base[i, :h, :w] = labels

        return {
            "input_ids": input_ids_base.to(device),
            "labels": labels_base.to(device)
        }

    ds_torch = get_as_input_and_labels_pairs()

    dataset = SimpleDataset(ds_torch)

    # Create train/validation split
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create dataloaders with custom collate function
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader

def ex2() :
    def plot_batch_data(
        batch : Dict[str, torch.Tensor]
    ) -> None:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        B, H, W = input_ids.shape
        MAX_COLOR = 30
        # input_ids (B, H, W)
        fig, axs = plt.subplots(2, B, figsize=(6, 12))
        for i in range(B):
            axs[0, i].imshow(input_ids[i], cmap='tab20', vmin=0, vmax=MAX_COLOR)
            axs[0, i].set_title(f'Input ids')

            axs[1, i].imshow(labels[i], cmap='tab20', vmin=0, vmax=MAX_COLOR)
            axs[1, i].set_title(f'Actual labels')

        plt.show()
    
    masked_token_id = 15
    newline_token_id = 12
    pad_token_id = 18
    raw_ds = get_ds_for_masked_modeling_only_answer_only_foreground_items(
        gen_arc_puzzle_ex(
            name = PuzzleNames.FILL_SIMPLE_OPENED_SHAPE,
            nb_examples = 100,
            augment_colors = True,
            do_shuffle = False
        ),
        0.2,
        masked_token_id
    )
    
    
    
    # _l = raw_ds[0][1]
    # print(_l)
    # assert (_l != masked_token_id).sum() > 0
    train_dl, val_dl = get_dataloaders_for_cnn_masked_modeling(
        raw_ds,
        masked_token_id = masked_token_id,
        ignore_label_id = -100,
        newline_token_id = newline_token_id,
        max_grid_width = 40,
        max_grid_height = 40,
        pad_token_id = pad_token_id,
        split_ratio = 0.8,
        batch_size_train = 4,
        batch_size_eval = 4,
        only_answer=False,
        pad_to_max=True,
        device = "cpu"# if torch.cuda.is_available() else "cpu"
    )
    
    
    background_colors_labels = []
    background_colors_input_ids = []
    
    for b in train_dl:
        input_ids, labels = b["input_ids"], b["labels"]
        for inp, l in zip(input_ids, labels):
            most_cmmon_color_inp : int = int(torch.bincount(inp[inp != pad_token_id]).argmax().item())
            try : 
                most_cmmon_color_label : int = int(torch.bincount(l[l != -100]).argmax().item())
            except BaseException :
                print('DADKASJDLKAS')
                # unique_colors_inp :Set[int] = set(int(x.item()) for x in inp.flatten() if x.item() != pad_token_id)
                
            background_colors_input_ids.append(most_cmmon_color_inp)
            background_colors_labels.append(most_cmmon_color_label)

    labels_histogram = torch.bincount(torch.tensor(background_colors_labels))
    input_ids_histogram = torch.bincount(torch.tensor(background_colors_input_ids))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].bar(range(len(labels_histogram)), labels_histogram.numpy(), color='blue')
    axs[0].set_title('Background Colors - Labels')
    axs[1].bar(range(len(input_ids_histogram)), input_ids_histogram.numpy(), color='orange')
    axs[1].set_title('Background Colors - Input IDs')
    plt.show()

    for batch in val_dl : 
        print(batch)
        assert len(batch[next(iter(batch.keys()))]) == 4
        plot_batch_data(batch)
        return






def get_dataloaders_for_flat_seq_cls(
    ds_raw : List[Tuple[np.ndarray, np.ndarray]],
    masked_token_id : int,
    ignore_label_id : int,
    sep_token_id : int,
    pad_token_id : int,
    split_ratio : float,
    batch_size_train : int,
    batch_size_eval : int,
    device : str
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    def get_as_input_and_labels_pairs() -> List[Tuple[torch.Tensor, torch.Tensor]]:
        res = []
        for input_seq, output_seq in ds_raw:
            # print(input_seq, output_seq)
            input_tensor = torch.tensor(input_seq, dtype=torch.long)
            output_tensor = torch.tensor(output_seq, dtype=torch.long)
            
            # Create sequence: input + sep + masked_tokens
            total_len = input_tensor.shape[0] + 1 + output_tensor.shape[0]
            input_ids = torch.full((total_len,), pad_token_id, dtype=torch.long)
            labels = torch.full((total_len,), ignore_label_id, dtype=torch.long)
            
            # Fill input part
            input_ids[:input_tensor.shape[0]] = input_tensor
            # Add separator
            input_ids[input_tensor.shape[0]] = sep_token_id
            # Fill masked tokens for output part
            input_ids[input_tensor.shape[0] + 1:] = masked_token_id
            # Set labels for output part only
            labels[input_tensor.shape[0] + 1:] = output_tensor
            
            res.append((input_ids, labels))
        return res

    def collate_fn(
        batch : List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor] :
        B = len(batch)
        max_len = max(x.shape[0] for x, _ in batch)

        input_ids_base : torch.Tensor = torch.full((B, max_len), pad_token_id, dtype=torch.long)
        labels_base = torch.full((B, max_len), ignore_label_id, dtype=torch.long) # padding labels and ignore labels is literally the same because loss function will treat them equally

        for i, (token_ids, labels) in enumerate(batch):
            l = token_ids.shape[0]
            input_ids_base[i, :l] = token_ids
            labels_base[i, :l] = labels

        return {
            "input_ids": input_ids_base.to(device),
            "labels": labels_base.to(device)
        }

    ds_torch = get_as_input_and_labels_pairs()

    dataset = SimpleDataset(ds_torch)

    # Create train/validation split
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(len(train_dataset), len(val_dataset))
    # Create dataloaders with custom collate function
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader



def ex3():
    ds_raw = get_ds_1d_seq_for_random_input_with_some_transformation_for_output(
        seq_len = 32, 
        nb_samples = 100,
        nb_cls = 10,
        task='split_in_8_pieces_and_recombine'
    )
    masked_token_id = 100
    ignore_label_id = -1
    sep_token_id = 101
    pad_token_id = 102
    split_ratio = 0.8
    batch_size_train = 2
    batch_size_eval = 2
    device = "cpu"

    train_dl, val_dl = get_dataloaders_for_flat_seq_cls(
        ds_raw,
        masked_token_id,
        ignore_label_id,
        sep_token_id,
        pad_token_id,
        split_ratio,
        batch_size_train,
        batch_size_eval,
        device
    )

    for batch in train_dl:
        print(batch)
        break







def get_dataloaders_for_2d_full_pred(
    ds_raw : List[Tuple[np.ndarray, np.ndarray]],
    ignore_label_id : int,
    pad_token_id : int,
    split_ratio : float,
    batch_size_train : int,
    batch_size_eval : int,
    max_grid_width : int,
    device : str,
    add_input_to_labels : bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    def sample_as_input_output_pair(inp : torch.Tensor, l : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] :
        i_field = torch.full((max_grid_width * 2, max_grid_width), pad_token_id, dtype=torch.long)
        l_field = torch.full((max_grid_width * 2, max_grid_width), ignore_label_id, dtype=torch.long)
        # print(inp.shape, l.shape)
        i_field[:inp.shape[0], :inp.shape[1]] = inp
        l_field[max_grid_width:max_grid_width + l.shape[0], :l.shape[1]] = l
        if add_input_to_labels :
            l_field[:inp.shape[0], :inp.shape[1]] = inp
        return i_field, l_field

    def get_as_input_and_labels_pairs() -> List[Tuple[torch.Tensor, torch.Tensor]] :
        res = []
        for i, l in ds_raw :
            res.append(sample_as_input_output_pair(
                torch.tensor(i, dtype=torch.long),
                torch.tensor(l, dtype=torch.long),
            ))
        return res
    
    def collate_fn(
        batch : List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor] :
        B = len(batch)
        max_h = max_grid_width * 2
        max_w = max_grid_width
        input_ids_base : torch.Tensor = torch.full((B, max_h, max_w), pad_token_id, dtype=torch.long)
        labels_base = torch.full((B, max_h, max_w), ignore_label_id, dtype=torch.long) # padding labels and ignore labels is literally the same because loss function will treat them equally

        for i, (token_ids, labels) in enumerate(batch):
            h, w = token_ids.shape
            input_ids_base[i, :h, :w] = token_ids
            labels_base[i, :h, :w] = labels

        return {
            "input_ids": input_ids_base.to(device),
            "labels": labels_base.to(device)
        }


    ds_torch = get_as_input_and_labels_pairs()

    dataset = SimpleDataset(ds_torch)

    # Create train/validation split
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # print(len(train_dataset), len(val_dataset))
    # Create dataloaders with custom collate function
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader


def ex4() :
    def plot_batch_data(
        batch : Dict[str, torch.Tensor]
    ) -> None:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        B, H, W = input_ids.shape
        MAX_COLOR = 30
        # input_ids (B, H, W)
        fig, axs = plt.subplots(2, B, figsize=(6, 12))
        for i in range(B):
            axs[0, i].imshow(input_ids[i], cmap='tab20', vmin=0, vmax=MAX_COLOR)
            axs[0, i].set_title(f'Input ids')

            axs[1, i].imshow(labels[i], cmap='tab20', vmin=0, vmax=MAX_COLOR)
            axs[1, i].set_title(f'Actual labels')

        plt.show()
        
    ds_raw = gen_arc_puzzle_ex(
        name = PuzzleNames.FILL_SIMPLE_OPENED_SHAPE,
        nb_examples = 100,
        augment_colors = False,
        do_shuffle = False
    )
    ignore_label_id = -100
    pad_token_id = 102
    split_ratio = 0.8
    batch_size_train = 4
    batch_size_eval = 2
    device = "cpu"

    train_dl, val_dl = get_dataloaders_for_2d_full_pred(
        ds_raw,
        ignore_label_id,
        pad_token_id,
        split_ratio,
        batch_size_train,
        batch_size_eval,
        max_grid_width=40,
        device=device
    )

    for batch in train_dl:
        # print(batch)
        plot_batch_data(batch)
        break



if __name__ == "__main__":
    # ex1()
    # ex2()
    # ex3()
    ex4()
