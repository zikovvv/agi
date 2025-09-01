from dataclasses import dataclass
from typing import Literal
import torch


@dataclass
class EncoderConfig:
   # Model dims
    d_model: int = 512
    n_head: int = 8
    d_head: int = 64    
    num_layers: int = 2
    nb_refinement_steps: int = 8
    nb_last_trained_steps: int = 1  # how many of the last steps to train. 1 means only the last step, all means all steps
    dim_feedforward: int = 1024
    dropout: float = 0.1
    norm_first: bool = True
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # 2d rope
    # use_2d_rope_attn: bool = False
    # rope_2d_attn_row_width: int = 40
    use_transposed_rope_for_2d_vertical_orientation: bool = False
    field_width_for_t_rope: int = 40
    field_height_for_t_rope: int = 80

    # Grid limits
    max_len: int = 5000

    # Vocabulary
    vocab_size: int = 200
    pad_token_id: int = 115
    newline_token_id: int = 114
    qa_sep_token_id: int = 113
    example_sep_token_id: int = 112
    masked_token_id: int = 111  # the "hole" token to predict

    # how to mark something as ignored
    ignore_id: int = -100

    embeddings_type : Literal['learnable', 'rope'] = 'learnable'  # 'learnable' or 'rope'


@dataclass
class DatasetConfig:
    num_samples: int = 600    # dataset size
    seed: int = 123
    val_frac: float = 0.1
    test_frac: float = 0.0    # set >0 for a held-out test split
    seq_len : int = 15
    max_width :int = 40

@dataclass
class TrainConfig:
    nb_plots_on_val : int = 2
    epochs: int = 50
    batch_size: int = 16
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
