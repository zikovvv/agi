import math
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention as sdpa
from common import *
from encoder.hrm_like_enc.config import EncoderConfig

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    sin = sin[:q.size(1)]
    cos = cos[:q.size(1)]

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
    
class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached, self.sin_cached # type: ignore

class LearnedROPE(nn.Module) :

    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        
        # now fill freqs with random nubers for experiment
        freq_mean = freqs.mean()
        freqs = torch.randn_like(freqs) * freq_mean
        
        self.initial_freqs_backup = freqs.detach().clone()
        self.freq = nn.Parameter(
            freqs, requires_grad=True
        )

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.freq.cos()
        sin = self.freq.sin()
        # print(self.freq.isclose(self.initial_freqs_backup).sum())
        return cos, sin

class Attention(nn.Module):
    def __init__(
        self,
        cfg : EncoderConfig,
        
        # hidden_size,
        # head_dim,
        # num_heads,
        # num_key_value_heads,
        # use_transposed_rope_for_2d_vertical_orientation: bool = False,
        # field_width_for_t_rope: int = 40,
        # field_height_for_t_rope: int = 80,
        # use_custom_learned_rope: bool = False,
    ):
        super().__init__()

        self.hidden_size = cfg.d_model
        self.head_dim = cfg.d_head
        self.output_size = self.head_dim * cfg.n_head
        self.num_heads = cfg.n_head
        self.num_key_value_heads = cfg.n_head

        self.use_transposed_rope_for_2d_vertical_orientation = cfg.use_transposed_rope_for_2d_vertical_orientation
        self.field_width_for_t_rope = cfg.field_width_for_t_rope
        self.field_height_for_t_rope = cfg.field_height_for_t_rope

        # self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        # self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

        self.qkv_proj = nn.Linear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.output_size, self.hidden_size, bias=False)
        
    def apply_rope_and_permute(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        transposed: bool = False,
        field_w : int = -1,
        field_h : int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if transposed :
            # it has to be like we are iterating
            assert key.shape[1] == field_h * field_w
            indexes = torch.arange(field_w * field_h, device=key.device)
            indexes_mat = indexes.view(field_w, field_h) # IMPORTANT W THEN H!!! hard to explain just believe me
            indexes_mat_t_flat = indexes_mat.T.flatten()
            cos = cos[indexes_mat_t_flat]
            sin = sin[indexes_mat_t_flat]

        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        q = query.permute(0, 2, 1, 3)  # [B,H,S,D]
        k = key.permute(0, 2, 1, 3)    # [B,H_kv,S,D] (GQA OK)
        v = value.permute(0, 2, 1, 3)  # [B,H_kv,S,D]
        return q, k, v
        

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        log_debug(f'{hidden_states.shape = }')

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv : torch.Tensor = self.qkv_proj(hidden_states)
        log_debug(f'{qkv.shape = }')

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        log_debug(f'{qkv.shape = }, {query.shape = }, {key.shape = }, {value.shape = }')


        q, k, v = self.apply_rope_and_permute(query, key, value, cos_sin[0], cos_sin[1])
        log_debug(f'{q.shape = }, {k.shape = }, {v.shape = }')

        attn = sdpa(q, k, v, attn_mask=None, is_causal=False)
        log_debug(f'{attn.shape = }')
        attn_output = attn.permute(0, 2, 1, 3)  # [B,S,H,D]
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        log_debug(f'{attn_output.shape = }')


        if self.use_transposed_rope_for_2d_vertical_orientation:
            q, k, v = self.apply_rope_and_permute(
                query,
                key,
                value,
                cos_sin[0], 
                cos_sin[1],
                transposed=True,
                field_w=self.field_width_for_t_rope,
                field_h=self.field_height_for_t_rope
            )
            attn_vertical = sdpa(q, k, v, attn_mask=None, is_causal=False)
            attn_output_vertical = attn_vertical.permute(0, 2, 1, 3)  # [B,S,H,D]
            attn_output_vertical = attn_output_vertical.view(batch_size, seq_len, self.output_size)  # type: ignore
            attn_output = (attn_output + attn_output_vertical) / 2

        return self.o_proj(attn_output)


    


class TransformerBlockHRM(nn.Module):
    def __init__(
        self,
        cfg : EncoderConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.rotary_emb = RotaryEmbedding(
            dim = self.cfg.d_head,
            max_position_embeddings = self.cfg.max_len,
            base = self.cfg.rope_theta
        )
        # self.rotary_emb = LearnedROPE(
        #     dim=self.cfg.d_head,
        #     max_position_embeddings=self.cfg.max_len,
        #     base=self.cfg.rope_theta
        # )
        # self.rope_cos, self.rope_sin = self.rotary_emb()
        # self.register_buffer('rope_cos', self.rope_cos, persistent=False)
        # self.register_buffer('rope_sin', self.rope_sin, persistent=False)

        self.self_attn = Attention(
            cfg=cfg
        )
        self.mlp = SwiGLU(
            hidden_size=cfg.d_model,
            expansion=cfg.dim_feedforward / cfg.d_model,
        )
        self.norm_eps = cfg.layer_norm_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: # type: ignore
        # Post Norm
        # Self Attention
        log_debug(f'{hidden_states.shape = }')
        # cos_sin : torch.Tensor = self.rope_cos_sin # type: ignore
        # rope_cos : torch.Tensor = self.rope_cos # type: ignore
        # rope_sin : torch.Tensor = self.rope_sin # type: ignore
        cos_sin : CosSin = self.rotary_emb()
        cos_sin : CosSin = (cos_sin[0].to(hidden_states.device), cos_sin[1].to(hidden_states.device))
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states
