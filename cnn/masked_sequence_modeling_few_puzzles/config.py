from dataclasses import dataclass
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'

@dataclass
class ARCCNNConfig:
    # Model dims
    d_model: int = 512
    n_head: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    norm_first: bool = True
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Grid limits
    max_height: int = 40
    max_width: int = 40

    # Vocabulary
    vocab_size: int = 32  # colors + specials (you can raise this)
    pad_token_id: int = 15
    newline_token_id: int = 14
    qa_sep_token_id: int = 13
    example_sep_token_id: int = 12
    masked_token_id: int = 11  # the "hole" token to predict
    
    # how to mark something as ignored
    ignore_id: int = -100

    # Token types (learnable)
    # We keep it simple: QUESTION=0, ANSWER=1, SPECIAL=2
    num_token_types: int = 3
    token_type_question: int = 0
    token_type_answer: int = 1
    token_type_special: int = 2

    # Packing
    # If True, we prepend [EXAMPLE_SEP], add [NEWLINE] after each row,
    # and insert [Q_A_SEP] between question and answer.
    use_separators: bool = False

    # Weight tying for LM head
    tie_word_embeddings: bool = True
    