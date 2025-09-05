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
    dropout: float = 0.1
    norm_first: bool = True
    use_emb_norm: bool = False
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    
    nb_refinement_steps: int = 8
    nb_last_trained_steps: int = 1  # how many of the last steps to train. 1 means only the last step, all means all steps
    dim_feedforward: int = 1024
    use_cnn: bool = False
    enable_pseudo_diffusion_inner: bool = False
    enable_pseudo_diffusion_outer: bool = False
    feed_first_half: bool = False

    # 2d rope
    # use_2d_rope_attn: bool = False
    # rope_2d_attn_row_width: int = 40
    use_transposed_rope_for_2d_vertical_orientation: bool = False
    use_axial_rope: bool = False

    field_width: int = -1
    field_height: int = -1

    # Grid limits
    nb_max_rope_positions: int = 5000

    # Vocabulary
    vocab_size: int = 200
    pad_token_id: int = 115
    newline_token_id: int = 114
    qa_sep_token_id: int = 113
    example_sep_token_id: int = 112
    masked_token_id: int = 111  # the "hole" token to predict

    # how to mark something as ignored
    ignore_id: int = -100

    use_x_encoder: bool = False
    
    init_hidden_state_to_zero: bool = True
    init_hidden_state_std: float = 0.02

    # learned pos embdeddings
    use_learned_pos_emb: bool = False
    learned_pos_embs_dim: int = 32  # if used, must be <= d_head
    use_projection_for_learned_pos_embs: bool = False  # whether to project the learned pos emb to d_head dim
    use_custom_learned_pos_emb_per_head: bool = False  # whether to have a separate learned pos emb for each head (increases param count a lot)

@dataclass
class DatasetConfig:
    num_samples: int = 600    # dataset size
    seed: int = 123
    val_frac: float = 0.1
    test_frac: float = 0.0    # set >0 for a held-out test split
    seq_len : int = 15
    max_width :int = 40
    expand: bool = False
    add_labels_to_inputs: bool = False
    add_sep: bool = False  # whether to add a sep token between input and labels

@dataclass
class TrainConfig:
    nb_plots_on_val : int = 2
    nb_max_epochs: int = 50
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    t_batch_size: int = 32
    t_show_nb_first_preds: int = 2
    t_nb_max_self_correction: int = 15
    t_show_in_window: bool = False
    t_max_nb_aug: int = 5
    
    v_batch_size: int = 8
    v_nb_max_self_correction : int = 10
    v_do_augmented_inference : bool = True
    v_show_in_window : bool = False
    v_max_nb_aug : int = 5
    v_show_nb_first_preds : int = 2
    
    
    

