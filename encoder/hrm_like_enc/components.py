import math
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention as sdpa
from common import *
from encoder.hrm_like_enc.config import EncoderConfig
from x_transformers import Encoder as XEncoder


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


# def rotate_half(x: torch.Tensor):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
#     # q, k: [bs, seq_len, num_heads, head_dim]
#     # cos, sin: [seq_len, head_dim]
#     orig_dtype = q.dtype
#     q = q.to(cos.dtype)
#     k = k.to(cos.dtype)
#     sin = sin[:q.size(1)]
#     cos = cos[:q.size(1)]

#     q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
#     k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

#     return q_embed.to(orig_dtype), k_embed.to(orig_dtype)



def apply_rotary_pos_emb(
    x: torch.Tensor,          # [B, L, H, D] (works for Q or K)
    cos: torch.Tensor,        # [S,D] or [S,H,D] or [S,D] or [S,D/2]
    sin: torch.Tensor,        # same shape rules as cos
    inplace: bool = False,    # if True, write result into x (slightly faster)
) -> torch.Tensor:    
    assert x.dim() == 4, f"x must be [B,L,H,D], got {tuple(x.shape)}"
    B, L, H, D = x.shape
    s = f'{B= }, {L = }, {H= }, {D= }'
    assert sin.dim() == 2, (sin.shape, sin.dim())
    assert (D % 2) == 0, f"D must be even, got {D}"
    assert sin.shape == cos.shape == (L, D)

    # --- split x into (even, odd) channel pairs and rotate ---
    x_even = x[..., ::2]                    # [B, L, H, D/2]
    x_odd  = x[..., 1::2]                   # [B, L, H, D/2]
    assert x_even.shape == x_odd.shape == (B, L, H, D // 2)
    cos_unsqueezed = cos.unsqueeze(0).unsqueeze(2)  # [1, L, 1, D/2]
    assert cos_unsqueezed.shape == (1, L, 1, D)
    sin_unsqueezed = sin.unsqueeze(0).unsqueeze(2)  # [1, L, 1, D/2]
    assert sin_unsqueezed.shape == (1, L, 1, D)

    cos_upper = cos_unsqueezed[..., :D//2] # [1, L, 1, D//2]
    cos_lower = cos_unsqueezed[..., D//2:] # [1, L, 1, D//2]
    sin_upper = sin_unsqueezed[..., :D//2] # [1, L, 1, D//2]
    sin_lower = sin_unsqueezed[..., D//2:] # [1, L, 1, D//2]
    assert cos_upper.shape == (1, L, 1, D // 2), (cos_upper.shape, (1, L, 1, D // 2), s)
    assert cos_lower.shape == (1, L, 1, D // 2), (cos_lower.shape, (1, L, 1, D // 2), s)
    assert sin_upper.shape == (1, L, 1, D // 2), (sin_upper.shape, (1, L, 1, D // 2), s)
    assert sin_lower.shape == (1, L, 1, D // 2), (sin_lower.shape, (1, L, 1, D // 2), s)

    # Broadcast cos/sin over B; they are already [1,L,(1 or H),D/2]
    # Rotation: (x_even,x_odd) -> (x_even*cos - x_odd*sin, x_even*sin + x_odd*cos)
    x[..., ::2] = x_even * cos_upper - x_odd * sin_upper
    x[..., 1::2] = x_even * sin_lower + x_odd * cos_lower
    return x


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = nn.Linear(hidden_size, inter * 2, bias=False)
        self.down_proj    = nn.Linear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
    
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # log(f'{freqs.shape = }, {t.shape = }, {inv_freq.shape = }, {emb.shape = }')        
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
        # log(f'{freqs.shape = }, {self.cos_cached.shape = }, {self.sin_cached.shape = }')

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached, self.sin_cached # type: ignore


class Attention(nn.Module):
    def __init__(
        self,
        cfg : EncoderConfig,
    ):
        super().__init__()

        self.hidden_size = cfg.d_model
        self.head_dim = cfg.d_head
        self.output_size = self.head_dim * cfg.n_head
        self.num_heads = cfg.n_head
        self.num_key_value_heads = cfg.n_head

        self.use_transposed_rope_for_2d_vertical_orientation = cfg.use_transposed_rope_for_2d_vertical_orientation
        self.field_width_for_t_rope = cfg.field_width
        self.field_height_for_t_rope = cfg.field_height

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

        query = apply_rotary_pos_emb(query, cos, sin)
        key = apply_rotary_pos_emb(key, cos, sin)
        q = query.permute(0, 2, 1, 3)  # [B,H,S,D]
        k = key.permute(0, 2, 1, 3)    # [B,H_kv,S,D] (GQA OK)
        v = value.permute(0, 2, 1, 3)  # [B,H_kv,S,D]
        return q, k, v
        

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        log_debug(f'{hidden_states.shape = }')
        batch_size, seq_len, _ = hidden_states.shape

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




class Attention2DROPEAxial(nn.Module):
    def __init__(
        self,
        cfg : EncoderConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.hidden_size = cfg.d_model
        self.head_dim = cfg.d_head
        self.output_size = self.head_dim * cfg.n_head
        self.num_heads = cfg.n_head
        self.num_key_value_heads = cfg.n_head

        self.use_transposed_rope_for_2d_vertical_orientation = cfg.use_transposed_rope_for_2d_vertical_orientation
        self.width = cfg.field_width
        self.height = cfg.field_height

        self.qkv_proj = nn.Linear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.output_size, self.hidden_size, bias=False)

        assert cfg.learned_pos_embs_dim == cfg.d_head or cfg.use_projection_for_learned_pos_embs, "learned_pos_embs_dim must be equal to d_head if not using projection"

        per_head_coeff = cfg.n_head if cfg.use_custom_learned_pos_emb_per_head else 1
        self.l_pos_emb_global_flat = nn.ModuleList([nn.Embedding(cfg.nb_max_rope_positions, cfg.learned_pos_embs_dim) for _ in range(per_head_coeff)])
        self.l_pos_emb_local_row = nn.ModuleList([nn.Embedding(cfg.nb_max_rope_positions, cfg.learned_pos_embs_dim) for _ in range(per_head_coeff)])
        self.l_pos_emb_local_col = nn.ModuleList([nn.Embedding(cfg.nb_max_rope_positions, cfg.learned_pos_embs_dim) for _ in range(per_head_coeff)])

        
        different_embs_coeff = 3
        self.pos_embeddings_projection = nn.Linear(cfg.learned_pos_embs_dim * different_embs_coeff + self.head_dim, self.head_dim)

    def apply_rope_flat(
        self,
        query: torch.Tensor,  # [B, S, H, D]
        key: torch.Tensor,    # [B, S, H, D]
        cos: torch.Tensor,    # expected broadcastable to [1, S, 1, D] or shaped [S, D]
        sin: torch.Tensor,    # same as cos
    ) :
        # 1-D fallback
        query = apply_rotary_pos_emb(query, cos, sin)
        key = apply_rotary_pos_emb(key, cos, sin)
        return query, key

    def apply_rope_axial(
        self,
        query: torch.Tensor,  # [B, S, H, D]
        key: torch.Tensor,    # [B, S, H, D]
        cos: torch.Tensor,    # expected broadcastable to [1, S, 1, D] or shaped [S, D]
        sin: torch.Tensor,    # same as cos
    ) :
        B, L, H, D = query.shape
        
        cos, sin = cos[:L, :], sin[:L, :]
        field_h, field_w = self.cfg.field_height, self.cfg.field_width
        assert field_h > 0, field_w > 0
        half = D // 2
        dev = query.device
        dtype = query.dtype

        # Indices mapping flattened pos -> (x, y)
        idx = torch.arange(L, device=dev)
        x_idx = idx % field_w               # [L]
        y_idx = idx // field_w              # [L]

        cos_x = cos[x_idx, :half].to(dtype) # [L, D/2]
        sin_x = sin[x_idx, :half].to(dtype) # [L, D/2]
        cos_y = cos[y_idx, :half].to(dtype) # [L, D/2]
        sin_y = sin[y_idx, :half].to(dtype) # [L, D/2]


        # Split Q/K along the head dimension and apply RoPE per axis
        qx, qy = query[..., :half], query[..., half:] # [B,L,H,D/2]
        kx, ky = key  [..., :half], key  [..., half:] # [B,L,H,D/2]

        qx = apply_rotary_pos_emb(qx, cos_x, sin_x) # [B,L,H,D/2]
        qy = apply_rotary_pos_emb(qy, cos_y, sin_y) # [B,L,H,D/2]

        kx = apply_rotary_pos_emb(kx, cos_x, sin_x) # [B,L,H,D/2]
        ky = apply_rotary_pos_emb(ky, cos_y, sin_y) # [B,L,H,D/2]

        query = torch.cat([qx, qy], dim=-1)
        key   = torch.cat([kx, ky], dim=-1)

        # Note: V is intentionally NOT rotated (standard RoPE usage). :contentReference[oaicite:1]{index=1}
        return query, key
    
    def get_idx(self, H : int, W : int, device: torch.device) :
        L = H * W
        idx = torch.arange(L, device=device)
        idx_rows = idx // W
        idx_cols = idx % W
        idx_g_h = idx
        return idx_g_h, idx_rows, idx_cols

    
    def get_pos_embs(self, device : torch.device) :
        L, H, D = self.cfg.field_height * self.cfg.field_width, self.num_heads, self.cfg.learned_pos_embs_dim if self.cfg.use_learned_pos_emb else self.cfg.d_head
        idx_g_h, idx_rows, idx_cols = self.get_idx(self.cfg.field_height, self.cfg.field_width, device) # [L]
        assert idx_g_h.shape == idx_rows.shape == idx_cols.shape == (L,)
        
        custom_coef = 1 if not self.cfg.use_custom_learned_pos_emb_per_head else self.cfg.n_head

        global_embs = torch.stack([pe(idx_g_h) for pe in self.l_pos_emb_global_flat]).unsqueeze(0) # [1, custom_coef, L, D]
        row_embs = torch.stack([pe(idx_rows) for pe in self.l_pos_emb_local_row]).unsqueeze(0) # [1, custom_coef, L, D]
        col_embs = torch.stack([pe(idx_cols) for pe in self.l_pos_emb_local_col]).unsqueeze(0) # [1, custom_coef, L, D]
        assert global_embs.shape == row_embs.shape == col_embs.shape == (1, custom_coef, L, D), (global_embs.shape, row_embs.shape, col_embs.shape, (1, custom_coef, L, D))
        
        if not self.cfg.use_custom_learned_pos_emb_per_head:
            # repear along head dim
            global_embs = global_embs.repeat(1, self.num_heads, 1, 1) # [1, H, L, D]
            row_embs = row_embs.repeat(1, self.num_heads, 1, 1) # [1, H, L, D]
            col_embs = col_embs.repeat(1, self.num_heads, 1, 1) # [1, H, L, D]
            assert global_embs.shape == row_embs.shape == col_embs.shape == (1, H, L, D), (global_embs.shape, row_embs.shape, col_embs.shape, (1, H, L, D))

        #permute to [1, L, H, D]
        global_embs = global_embs.permute(0, 2, 1, 3) # [1, L, H, D]
        row_embs = row_embs.permute(0, 2, 1, 3) # [1, L, H, D]
        col_embs = col_embs.permute(0, 2, 1, 3) # [1, L, H, D]
        assert global_embs.shape == row_embs.shape == col_embs.shape == (1, L, H, D), (global_embs.shape, row_embs.shape, col_embs.shape, (1, L, H, D))
        
        return global_embs, row_embs, col_embs

    def apply_learned_pos_emb(
        self,
        query: torch.Tensor,  # [B, L, H, D] 
        key: torch.Tensor,    # [B, L, H, D]
    ) :
        B, L, H, D = query.shape
        global_embs, row_embs, col_embs = self.get_pos_embs(query.device) # [1, L, H, D or pos_emb_dim] each
        assert global_embs.shape == row_embs.shape == col_embs.shape

        if self.cfg.use_projection_for_learned_pos_embs:
            raise NotImplementedError('Not implemented yet')
            # Combine all position embeddings and project to head_dim
        else :
            # Combine all position embeddings
            pos_emb = global_embs + row_embs + col_embs
            query += pos_emb
            key += pos_emb
            return query, key


            
    def apply_pos_emb(
        self,
        query: torch.Tensor,  # [B, L, H, D]
        key: torch.Tensor,    # [B, L, H, D]
        cos: torch.Tensor,    # expected broadcastable to [1, L, 1, D] or shaped [L, D]
        sin: torch.Tensor,    # same as cos
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Axial RoPE on a flattened (field_h x field_w) grid:
        - first half of dim D is rotated by x-positions,
        - second half by y-positions.
        Falls back to 1-D RoPE if grid info is invalid.
        """
        B, L, H, D = query.shape
        if self.cfg.use_learned_pos_emb:
            query, key = self.apply_learned_pos_emb(query, key)
        else:
            if self.cfg.use_axial_rope:
                query, key = self.apply_rope_axial(query, key, cos, sin)
            else:
                query, key = self.apply_rope_flat(query, key, cos, sin)
        return query, key

        

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        B, L, D = hidden_states.shape # [B, L, D]
        log_debug(f'{hidden_states.shape = }')

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv : torch.Tensor = self.qkv_proj(hidden_states) # [B, L, H * HD]
        log_debug(f'{qkv.shape = }')

        # Split head
        qkv = qkv.view(B, L, self.num_heads + 2 * self.num_key_value_heads, self.head_dim) # [B, L, (num_heads + 2 * num_key_value_heads), head_dim]
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        log_debug(f'{qkv.shape = }, {query.shape = }, {key.shape = }, {value.shape = }')

        query, key = self.apply_pos_emb(query, key, cos_sin[0], cos_sin[1])

        log_debug(f'{query.shape = }, {key.shape = }, {value.shape = }')

        # Final permutation to [B, H, S, D]
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)


        log_debug(f'{q.shape = }, {k.shape = }, {v.shape = }')

        attn = sdpa(q, k, v, attn_mask=None, is_causal=False)
        log_debug(f'{attn.shape = }')
        attn_output = attn.permute(0, 2, 1, 3)  # [B,S,H,D]
        attn_output = attn_output.view(B, L, self.output_size)  # type: ignore
        log_debug(f'{attn_output.shape = }')
        
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
            max_position_embeddings = self.cfg.nb_max_rope_positions,
            base = self.cfg.rope_theta
        )
        self.self_attn = Attention2DROPEAxial(
            cfg=cfg
        )
        self.mlp = SwiGLU(
            hidden_size=cfg.d_model,
            expansion=cfg.dim_feedforward / cfg.d_model,
        )
        self.norm_eps = cfg.layer_norm_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: # type: ignore
        log_debug(f'{hidden_states.shape = }')
        cos_sin : CosSin = self.rotary_emb()
        cos_sin : CosSin = (cos_sin[0].to(hidden_states.device), cos_sin[1].to(hidden_states.device))
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states
