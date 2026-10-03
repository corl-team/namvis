"""
Definitions of blocks of VAR transformer model.
"""

import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from torch.utils.checkpoint import checkpoint

# Import flash_attn's attention
from flash_attn import flash_attn_func                  # q, k, or v: BLHc, ret: BLHc
from flash_attn import flash_attn_varlen_kvpacked_func  # qkv: N3Hc, ret: NHc
try:
    from flash_attn import flash_attn_varlen_func       # q, k, v: total_tokens,H,c
except ImportError:
    flash_attn_varlen_func = None

from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
from infinity.models.flex_attn import (
    SWITTI_FLEX_KERNEL_OPTIONS,
    create_switti_block_mask,
    flex_attention_available,
    get_compiled_flex_attention,
)

# Import flash_attn's fused ops

try:
    from flash_attn.ops.fused_dense import fused_mlp_func
    flash_fused_op_installed = True
except ImportError:
    flash_fused_op_installed = False
    fused_mlp_func = None

try:
    from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
except ImportError:
    def rms_norm_impl(x, weight, epsilon):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon))) * weight


MAX_D_F = 10.0
MAX_ASINH_D_F = math.asinh(MAX_D_F)


class FastRMSNorm(nn.Module):
    def __init__(self, C, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.C = C
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(C))
        else:
            self.register_buffer('weight', torch.ones(C))

    def forward(self, x):
        src_type = x.dtype
        return rms_norm_impl(x.float(), self.weight, epsilon=self.eps).to(src_type)

    def extra_repr(self) -> str:
        return f'C={self.C}, eps={self.eps:g}, elementwise_affine={self.elementwise_affine}'


def get_dropout_layer(p):
    return nn.Dropout(p, inplace=True) if p > 0 else nn.Identity()


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_mlp=False):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_mlp else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = get_dropout_layer(drop)
        self.heuristic = 0

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x,
                weight1=self.fc1.weight,
                weight2=self.fc2.weight,
                bias1=self.fc1.bias,
                bias2=self.fc2.bias,
                activation='gelu_approx',
                save_pre_act=self.training,
                return_residual=False,
                checkpoint_lvl=0,
                heuristic=self.heuristic,
                process_group=None,
            ))
        else:
            return self.drop(self.fc2( self.act(self.fc1(x)) ))

    def extra_repr(self) -> str:
        return f'fused_mlp={self.fused_mlp_func is not None}'


class FFNSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, drop=0., fused_mlp=False):
        super().__init__()
        self.fused_mlp_func = None
        hidden_features = round(2 * hidden_features / 3 / 256) * 256

        out_features = out_features or in_features
        self.fcg = nn.Linear(in_features, hidden_features, bias=False)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = get_dropout_layer(drop)

    def forward(self, x):
        return self.drop(self.fc2( F.silu(self.fcg(x), inplace=True).mul_(self.fc1(x)) ))

    def extra_repr(self) -> str:
        return f'fused_mlp={self.fused_mlp_func is not None}'

@staticmethod
def _lift_K(Ks):
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out

@staticmethod
def _invert_K(Ks):
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out

@staticmethod
def _invert_SE3(transforms):
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out

def get_prope_matrices(poses_c2w, intrs):
    """
    Calculates P and P_inv.
    Matches Original: input `poses` is converted to `inv(poses)` (w2c) logic.
    P = Lift(K) @ inv(poses)  (World -> Screen)
    P_inv = poses @ Lift(K)^-1 (Screen -> World)
    """
    poses_w2c = _invert_SE3(poses_c2w) # World-to-Camera (assuming poses is Camera-to-World)

    if intrs is not None:
        Ks_norm = intrs.clone()
        Ks_norm[..., 0, 2] -= 0.5
        Ks_norm[..., 1, 2] -= 0.5

        lifted_K = _lift_K(Ks_norm)
        # P = K @ w2c
        P = torch.einsum("...ij,...jk->...ik", lifted_K, poses_w2c)

        # P_inv
        lifted_K_inv = _lift_K(_invert_K(Ks_norm))

        # P_inv = c2w @ K_inv
        P_inv = torch.einsum("...ij,...jk->...ik", poses_c2w, lifted_K_inv)

        P_T = P.transpose(-1,-2)
    else:
        P = poses_w2c
        P_inv = poses_c2w
        P_T = P.transpose(-1,-2)

    return P, P_T, P_inv


class SelfAttention(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, customized_flash_attn=True, use_flex_attn=False,
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0, N_views=1
    ):
        """
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        :param customized_flash_attn:
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.using_flash = customized_flash_attn

        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.tau, self.cos_attn = tau, cos_attn
        if self.cos_attn:
            self.scale = 1
            size = (1, 1, self.num_heads, 1) if self.using_flash else (1, self.num_heads, 1, 1)
            # size: 11H1 or 1H11
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau

        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)

        self.caching = False    # kv caching: only used during inference
        self.cached_k = None    # kv caching: only used during inference
        self.cached_v = None    # kv caching: only used during inference

        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier

        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw

    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = None
        self.cached_v = None

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self,
                x,
                attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]],
                attn_fn=None,
                scale_schedule=None,
                rope2d_freqs_grid=None,
                scale_ind=0):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        # x: fp32
        B, L, C = x.shape

        # qkv: amp, bf16
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)  # BL3Hc
        if self.using_flash: q, k, v = qkv.unbind(dim=2); L_dim = 1           # q or k or v: all are shaped in (B:batch_size, L:seq_len, H:heads, c:head_dim)
        else: q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); L_dim = 2   # q or k or v: all are shaped in (B:batch_size, H:heads, L:seq_len, c:head_dim)

        if self.cos_attn:   # always True
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
            q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()   # fp32
            k = F.normalize(k, dim=-1, eps=1e-12).contiguous()                  # fp32
            v = v.contiguous()                                                  # bf16
        else:   # be contiguous, to make kernel happy
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16

        if self.caching:    # kv caching: only used during inference
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=L_dim); v = self.cached_v = torch.cat((self.cached_v, v), dim=L_dim)


        if self.using_flash:
            if attn_bias_or_two_vector is not None: # training
                kw = dict(VAR_visible_kvlen=attn_bias_or_two_vector[0], VAR_invisible_qlen=attn_bias_or_two_vector[1])
            else:                                   # inference (autoregressive sampling)
                kw = dict()
            oup = flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, dropout_p=0, softmax_scale=self.scale, **kw).view(B, L, C)
        else:
            # if self.cos_attn: q, k are in fp32; v is in bf16
            # else: q, k, v are in bf16
            if self.use_flex_attn and attn_fn is not None:
                oup = attn_fn(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
            else:
                oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C)
            # oup: bf16

        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        tail = ''
        return f'using_flash={self.using_flash}, tau={self.tau}, cos_attn={self.cos_attn}{tail}'


@torch.compile
def apply_cos_attn_fused(q, k, scale_mul, target_dtype):
    # Compute mathematically stable fp32 norm, scale, then cast down
    q_norm = F.normalize(q.to(torch.float32), dim=-1, eps=1e-12) * scale_mul
    k_norm = F.normalize(k.to(torch.float32), dim=-1, eps=1e-12)
    return q_norm.to(target_dtype), k_norm.to(target_dtype)


def _build_switti_varlen_metadata(
    scale_schedule,
    batch_size,
    n_views,
    sequence_length,
    device,
):
    """Build packed-sequence offsets for batch-major, scale-major SWITTI tokens."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if n_views < 1:
        raise ValueError(f"n_views must be positive, got {n_views}")
    if not scale_schedule:
        raise ValueError("scale_schedule must contain at least one scale")

    scale_lengths = [
        n_views * math.prod(int(dim) for dim in scale)
        for scale in scale_schedule
    ]
    expected_length = sum(scale_lengths)
    if expected_length != sequence_length:
        raise ValueError(
            "Packed SWITTI attention requires an unpadded scale-major sequence: "
            f"expected {expected_length} tokens from scale_schedule and n_views={n_views}, "
            f"but received {sequence_length}."
        )

    segment_lengths = torch.tensor(
        scale_lengths * batch_size,
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens = torch.zeros(
        segment_lengths.numel() + 1,
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens[1:] = torch.cumsum(segment_lengths, dim=0, dtype=torch.int32)
    return cu_seqlens, max(scale_lengths)

class SelfAttentionPropeOptimized(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, customized_flash_attn=True, use_flex_attn=False,
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0, N_views=1,
        switti_attn_backend="sdpa",
    ):
        """
        Implements PRope
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        :param customized_flash_attn:
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.using_flash = customized_flash_attn

        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.tau, self.cos_attn = tau, cos_attn
        if self.cos_attn:
            self.scale = 1
            size = (1, 1, self.num_heads, 1) if self.using_flash else (1, self.num_heads, 1, 1)
            # size: 11H1 or 1H11
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau

        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)

        self.caching = False    # kv caching: only used during inference
        self.cached_k = None    # kv caching: only used during inference
        self.cached_v = None    # kv caching: only used during inference


        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier
        if switti_attn_backend not in {"sdpa", "flash_varlen", "flex"}:
            raise ValueError(
                "switti_attn_backend must be 'sdpa',  "
                "'flash_varlen', or 'flex', got "
                f"{switti_attn_backend!r}."
            )
        if switti_attn_backend == "flash_varlen" and customized_flash_attn:
            raise ValueError(
                "switti_attn_backend='flash_varlen' requires the customized dense "
                "FlashAttention path to be disabled (--flash=0)."
            )
        if switti_attn_backend == "flash_varlen" and use_flex_attn:
            raise ValueError(
                "switti_attn_backend='flash_varlen' cannot be combined with FlexAttention."
            )
        if switti_attn_backend == "flash_varlen" and pad_to_multiplier > 1:
            raise ValueError(
                "switti_attn_backend='flash_varlen' requires an unpadded sequence; "
                f"got pad_to_multiplier={pad_to_multiplier}."
            )
        if switti_attn_backend == "flash_varlen" and flash_attn_varlen_func is None:
            raise RuntimeError(
                "switti_attn_backend='flash_varlen' requires a FlashAttention build "
                "that exposes flash_attn_varlen_func."
            )
        if switti_attn_backend == "flash_varlen" and self.head_dim > 256:
            raise ValueError(
                "FlashAttention varlen supports head dimensions up to 256, got "
                f"head_dim={self.head_dim}."
            )
        if switti_attn_backend == "flex" and customized_flash_attn:
            raise ValueError(
                "switti_attn_backend='flex' requires the customized dense "
                "FlashAttention path to be disabled (--flash=0)."
            )
        if switti_attn_backend == "flex" and use_flex_attn:
            raise ValueError(
                "switti_attn_backend='flex' is separate from the legacy AR "
                "FlexAttention path; set use_flex_attn=False."
            )
        if switti_attn_backend == "flex" and pad_to_multiplier > 1:
            raise ValueError(
                "switti_attn_backend='flex' currently requires an unpadded "
                f"sequence; got pad_to_multiplier={pad_to_multiplier}."
            )
        if switti_attn_backend == "flex" and not flex_attention_available:
            raise RuntimeError(
                "switti_attn_backend='flex' requires PyTorch FlexAttention "
                "(torch>=2.5.1)."
            )
        self.switti_attn_backend = switti_attn_backend
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw

    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = None
        self.cached_v = None


    def prope_dot_product_vectorized(
        self, q, k, v,
        poses_c2w, Ks,
        scale_schedule,
        scale_ind=None,
        attn_mask=None,
        prope_cache=None,
        **kwargs
    ):
        """
        Vectorized implementation that avoids Python loops over sequence chunks.
        """
        B, num_heads, seqlen, head_dim = q.shape
        N_views = poses_c2w.shape[1]
        device = q.device

        cache_key = f"tgt_scale_{scale_ind}" if scale_ind is not None else "tgt_full"
        cached_geom = None


        if prope_cache is not None and cache_key in prope_cache:
            cached_geom = prope_cache.get(cache_key)

        else:
            # 1. Select schedule (Single step vs Full sequence)
            if scale_ind is not None:
                current_schedule = [scale_schedule[scale_ind]]
            else:
                current_schedule = scale_schedule

            # 2. Compute Camera Matrices
            P, P_T, P_inv = get_prope_matrices(poses_c2w=poses_c2w, intrs=Ks)

            pos_x_list, pos_y_list = [], []
            P_T_list, P_inv_list, P_list = [], [], []
            is_single_scale = (len(current_schedule) == 1)

            for _, px, py in current_schedule:
                num_repeats = px * py

                x_grid = torch.arange(px, device=device).repeat(py * N_views)
                y_grid = torch.arange(py, device=device).repeat_interleave(px).repeat(N_views)

                pos_x_list.append(x_grid)
                pos_y_list.append(y_grid)

                if not is_single_scale:
                    P_T_list.append(P_T.repeat_interleave(num_repeats, dim=1))
                    P_inv_list.append(P_inv.repeat_interleave(num_repeats, dim=1))
                    P_list.append(P.repeat_interleave(num_repeats, dim=1))

            pos_x_total = torch.cat(pos_x_list, dim=0)
            pos_y_total = torch.cat(pos_y_list, dim=0)
            coeffs_x = self._rope_precompute_coeffs(pos_x_total, 100.0, 1.0, head_dim // 4)
            coeffs_y = self._rope_precompute_coeffs(pos_y_total, 100.0, 1.0, head_dim // 4)

            if is_single_scale:
                # FAST PATH: Keep matrices compact (B, V, 4, 4)
                P_T_seq, P_inv_seq, P_seq = P_T, P_inv, P
            else:
                # SLOW PATH: Concatenate expanded matrices (B, L, 4, 4)
                P_T_seq = torch.cat(P_T_list, dim=1)
                P_inv_seq = torch.cat(P_inv_list, dim=1)
                P_seq = torch.cat(P_list, dim=1)

            cached_geom = (P_T_seq, P_inv_seq, P_seq, coeffs_x, coeffs_y)

            if prope_cache is not None:
                prope_cache[cache_key] = cached_geom

        if scale_ind is not None:
            scale_schedule = [scale_schedule[scale_ind]]

        P_T_seq, P_inv_seq, P_seq, coeffs_x, coeffs_y = cached_geom

        # Preserve the projected autocast dtype for backends that require
        # fp16/bf16. ProPE coefficients promote the transformed tensors to
        # fp32; dense SDPA and SWITTI FlexAttention intentionally retain that
        # post-ProPE precision.
        projected_attention_dtype = v.dtype

        # Apply transforms to Q, K, V
        q = self._apply_transform(q, P_T_seq, coeffs_x, coeffs_y, inverse_rope=False)
        k = self._apply_transform(k, P_inv_seq, coeffs_x, coeffs_y, inverse_rope=False)
        v = self._apply_transform(v, P_inv_seq, coeffs_x, coeffs_y, inverse_rope=False)

        # --- ATTENTION PHASE ---
        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=2)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=2)

        # Teacher-forced SWITTI packs every (batch item, scale) pair as an
        # independent sequence. Progressive inference keeps using SDPA because
        # scale_ind is set and its Q/K/V are already current-scale-only.
        if self.switti_attn_backend == "flash_varlen" and scale_ind is None:
            if self.caching:
                raise RuntimeError(
                    "Packed SWITTI FlashAttention cannot be combined with KV caching."
                )
            if q.device.type != "cuda":
                raise RuntimeError(
                    "Packed SWITTI FlashAttention requires CUDA tensors, got "
                    f"device={q.device}."
                )
            if projected_attention_dtype not in {torch.float16, torch.bfloat16}:
                raise RuntimeError(
                    "Packed SWITTI FlashAttention requires fp16 or bf16 QKV from "
                    f"autocast, got dtype={projected_attention_dtype}."
                )

            metadata_key = (
                "switti_varlen",
                B,
                N_views,
                tuple(tuple(int(dim) for dim in scale) for scale in scale_schedule),
                q.device.type,
                q.device.index,
            )
            metadata = None if prope_cache is None else prope_cache.get(metadata_key)
            if metadata is None:
                metadata = _build_switti_varlen_metadata(
                    scale_schedule=scale_schedule,
                    batch_size=B,
                    n_views=N_views,
                    sequence_length=seqlen,
                    device=q.device,
                )
                if prope_cache is not None:
                    prope_cache[metadata_key] = metadata
            cu_seqlens, max_seqlen = metadata

            q_attention = q.to(projected_attention_dtype).contiguous()
            k_attention = k.to(projected_attention_dtype).contiguous()
            v_attention = v.to(projected_attention_dtype).contiguous()

            q_packed = q_attention.permute(0, 2, 1, 3).contiguous().view(-1, num_heads, head_dim)
            k_packed = k_attention.permute(0, 2, 1, 3).contiguous().view(-1, num_heads, head_dim)
            v_packed = v_attention.permute(0, 2, 1, 3).contiguous().view(-1, num_heads, head_dim)
            out = flash_attn_varlen_func(
                q=q_packed,
                k=k_packed,
                v=v_packed,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                softmax_scale=kwargs.get("scale"),
                causal=False,
            )
            out = out.view(B, seqlen, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        elif self.switti_attn_backend == "flex" and scale_ind is None:
            if self.caching:
                raise RuntimeError(
                    "SWITTI FlexAttention cannot be combined with KV caching."
                )
            if q.device.type != "cuda":
                raise RuntimeError(
                    "SWITTI FlexAttention requires CUDA tensors, got "
                    f"device={q.device}."
                )
            if any(tensor.dtype != torch.float32 for tensor in (q, k, v)):
                raise RuntimeError(
                    "SWITTI FlexAttention expects ProPE-transformed Q/K/V in "
                    "float32, got dtypes="
                    f"{(q.dtype, k.dtype, v.dtype)}."
                )
            if attn_mask is not None:
                raise RuntimeError(
                    "SWITTI FlexAttention received a dense attention mask; "
                    "the caller must skip dense mask allocation."
                )

            metadata_key = (
                "switti_flex_block_mask",
                N_views,
                tuple(tuple(int(dim) for dim in scale) for scale in scale_schedule),
                seqlen,
                q.device.type,
                q.device.index,
            )
            block_mask = None if prope_cache is None else prope_cache.get(metadata_key)
            if block_mask is None:
                block_mask = create_switti_block_mask(
                    block_scales=scale_schedule,
                    n_views=N_views,
                    sequence_length=seqlen,
                    device=q.device,
                )
                if prope_cache is not None:
                    prope_cache[metadata_key] = block_mask

            # Retain the post-ProPE FP32 tensors. This makes the candidate a
            # precision-matched comparison with the existing dense SDPA path
            # and avoids BF16 rounding of the unusually large ProPE logits.

            q_attention = q.to(projected_attention_dtype).contiguous()
            k_attention = k.to(projected_attention_dtype).contiguous()
            v_attention = v.to(projected_attention_dtype).contiguous()


            out = get_compiled_flex_attention()(
                q_attention,
                k_attention,
                v_attention,
                block_mask=block_mask,
                scale=kwargs.get("scale"),
                kernel_options=SWITTI_FLEX_KERNEL_OPTIONS,
            )
        else:
            # Global dense/current-scale Attention call.
            q_attention, k_attention, v_attention = q, k, v
            out = F.scaled_dot_product_attention(
                query=q_attention,
                key=k_attention,
                value=v_attention,
                attn_mask=attn_mask,
                **kwargs
            )


        # Apply output transform
        out = self._apply_transform(out, P_seq, coeffs_x, coeffs_y, inverse_rope=True)

        return out

    # Helper to apply the 3-part block diagonal transform efficiently
    def _apply_transform(self, x, mat_seq_or_cameras, coeffs_x, coeffs_y, inverse_rope=False):
        # Split features: [Half (Proj), Quarter (RoPE X), Quarter (RoPE Y)]

        B, H, L, D = x.shape

        # x: (B, H, L, D)
        d_half = D // 2
        d_quart = D // 4

        # split into projective/ropex/ropey
        x_proj, x_rope_x, x_rope_y = torch.split(x, [d_half, d_quart, d_quart], dim=-1)

        # if single_scale(i.e. inference)
        if (mat_seq_or_cameras.dim() == 4 and
            mat_seq_or_cameras.shape[1] < L and
            L % mat_seq_or_cameras.shape[1] == 0):

            # Fast Path: Broadcast over patches
            mat = mat_seq_or_cameras
            cameras = mat.shape[1]
            patches_per_camera = L // cameras

            x_proj_c = x_proj.contiguous()
            x_proj_reshaped = x_proj_c.view(B, H, cameras, patches_per_camera, d_half // 4, 4)

            # Einsum: bcij (Mat) * bncpkj (Vec) -> bncpki
            proj_out = torch.einsum("bcij,bncpkj->bncpki", mat, x_proj_reshaped)
            x_proj_out = proj_out.reshape(B, H, L, d_half).contiguous()

        # if all scales(i.e. train)
        else:
            mat = mat_seq_or_cameras
            x_proj_c = x_proj.contiguous()
            x_proj_reshaped = x_proj_c.view(B, H, L, -1, 4)
            x_proj_out = torch.einsum("blij, bhlkj -> bhlki", mat, x_proj_reshaped).reshape(B, H, L, d_half).contiguous()

        # 2. Apply RoPE
        x_rope_x_out = self._rope_apply_coeffs(x_rope_x, coeffs_x, inverse=inverse_rope)
        x_rope_y_out = self._rope_apply_coeffs(x_rope_y, coeffs_y, inverse=inverse_rope)

        return torch.cat([x_proj_out, x_rope_x_out, x_rope_y_out], dim=-1)

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self,
                x,
                attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]],
                attn_fn=None,
                scale_schedule=None,
                rope2d_freqs_grid=None,
                scale_ind=None,
                poses=None,
                intrs=None,
                input_size=None):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        # x: fp32
        B, L, C = x.shape

        # qkv: amp, bf16
        # qkv: [B,L,3,16,128] [2,3,3,16,128]
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)  # BL3Hc

        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        if self.cos_attn:   # always True


            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
            q,k = apply_cos_attn_fused(q,k,scale_mul,v.dtype)

            # q
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16                                             # bf16
        else:   # be contiguous, to make kernel happy
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16


        oup = self.prope_dot_product_vectorized(q,k,v,
                                                poses_c2w=poses,
                                                Ks=intrs,
                                                scale_schedule=scale_schedule,
                                                scale_ind=scale_ind,
                                                attn_mask=attn_bias_or_two_vector,
                                                prope_cache=rope2d_freqs_grid,
                                                scale=self.scale)
        oup = oup.transpose(1,2).reshape(B,L,C)

        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        tail = ''
        return (
            f'using_flash={self.using_flash}, switti_attn_backend={self.switti_attn_backend}, '
            f'tau={self.tau}, cos_attn={self.cos_attn}{tail}'
        )

    # --- Helper Static Methods (Inlined for speed/simplicity) ---
    @staticmethod
    def _rope_precompute_coeffs(positions, freq_base, freq_scale, feat_dim):
        num_freqs = feat_dim // 2
        freqs = freq_scale * (freq_base ** (-torch.arange(num_freqs, device=positions.device) / num_freqs))
        angles = positions[:, None] * freqs[None, :] # (Seq, Freqs)
        # Reshape for broadcasting: (1, 1, Seq, Freqs)
        angles = angles.view(1, 1, positions.shape[0], num_freqs)
        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    def _rope_apply_coeffs(feats, coeffs, inverse=False):
        cos, sin = coeffs

        x_in = feats[..., : feats.shape[-1] // 2]
        y_in = feats[..., feats.shape[-1] // 2 :]

        if not inverse:
            return torch.cat([cos * x_in + sin * y_in, -sin * x_in + cos * y_in], dim=-1)
        else:
            return torch.cat([cos * x_in - sin * y_in, sin * x_in + cos * y_in], dim=-1)


class CrossAttention(nn.Module):
    def __init__(
        self, for_attn_pool=False, embed_dim=768, kv_dim=4096, num_heads=12,
        proj_drop=0., cos_attn=False,
    ):
        """
        :param for_attn_pool: only used in VAR.text_proj_for_sos
        :param embed_dim: Q's dim
        :param kv_dim: K's and V's dim
        :param num_heads: num heads of multi-head attention
        :param proj_drop: proj drop out
        :param cos_attn: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        """
        cos_attn = False    # TODO: never use cos attn in cross attention with T5 kv
        super().__init__()
        self.for_attn_pool = for_attn_pool
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads  # =64
        self.cos_attn = cos_attn
        if self.cos_attn:
            self.scale = 1
            self.scale_mul_1H1 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim)

        if for_attn_pool:
            q = torch.empty(1, self.num_heads, self.head_dim)
            nn.init.trunc_normal_(q, mean=0, std=math.sqrt(1 / embed_dim / 3))
            self.mat_q = nn.Parameter(q)
        else:
            self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mat_kv = nn.Linear(kv_dim, embed_dim*2, bias=False)
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)

    def forward(self, q, ca_kv):
        """
        :param q: shaped as (batch, seq_len, Q_dim)
        :param ca_kv: contains several vectors, each of which is shaped as (len_i, KV_dim). We have [len_1xKV_dim, len_2xKV_dim, len_3xKV_dim, ...] and lens == [len_1, len_2, len_3, ...]
            - kv_compact: shaped as (sum(lens), KV_dim)
            - cu_seqlens_k: cumulated sum of lens
            - max_seqlen_k: int, max(lens)
        NOTE: seq_len (num of Qs) can reach 10k;  but len_i (num of KVs) must <= 256
        
        :return: shaped as (batch, seq_len, Q_dim)
        """
        kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
        N = kv_compact.shape[0]


        # kv_compact: [256, 2, 16, 128] ,[n_tokens, 2(k,v), n_heads, head_dim]
        kv_compact = F.linear(kv_compact,
                              weight=self.mat_kv.weight,
                              bias=torch.cat((self.zero_k_bias, self.v_bias))).view(N, 2, self.num_heads, self.head_dim) # NC => N2Hc


        if not self.for_attn_pool:
            B, Lq = q.shape[:2]
            q_compact = self.mat_q(q).view(-1, self.num_heads, self.head_dim)
        else:
            B = cu_seqlens_k.shape[0] - 1
            Lq = 1
            q_compact = self.mat_q.repeat(B, 1, 1).to(dtype=kv_compact.dtype)
        if self.cos_attn:   # always False
            scale_mul = self.scale_mul_1H1.clamp_max(self.max_scale_mul).exp()
            k, v = kv_compact.unbind(dim=1)
            q_compact = F.normalize(q_compact, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
            kv_compact = torch.stack((k, v), dim=1)

        q_compact = q_compact.contiguous()
        kv_compact = kv_compact.contiguous()

        cu_seqlens_q = torch.arange(0, Lq * (B+1), Lq, dtype=torch.int32, device=q_compact.device)
        if q_compact.dtype == torch.float32:    # todo: fp16 or bf16?
            oup = flash_attn_varlen_kvpacked_func(q=q_compact.to(dtype=torch.bfloat16), kv=kv_compact.to(dtype=torch.bfloat16), cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
            oup = oup.float()
        else:
            oup = flash_attn_varlen_kvpacked_func(q=q_compact, kv=kv_compact, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)

        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        return f'Cq={self.embed_dim}, Ckv={self.kv_dim}, cos_attn={self.cos_attn}'


class CrossAttentionPropeOptimized(nn.Module):

    def __init__(
        self,
        embed_dim=768,
        kv_dim=4096,
        num_heads=12,
        proj_drop=0.,
        cos_attn=False,
        N_views_src=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads

        self.scale = 1 / math.sqrt(self.head_dim)

        self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mat_kv = nn.Linear(kv_dim, embed_dim*2, bias=False)

        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        self.N_views_src = N_views_src

    def forward(self, q, ca_kv,
                poses, poses_src,
                intrs, intrs_src,
                scale_schedule,
                scale_ind=None,
                rope2d_freqs_grid=None):
        B, L, C = q.shape
        device = q.device

        N_views_src = poses_src.shape[1]
        N_views_tgt = poses.shape[1]

        # [1] PREPARE SOURCE (K, V) ---
        kv_compact, _, _ = ca_kv
        kv_compact = kv_compact.reshape(B, -1, self.kv_dim)
        patches_per_view_src = kv_compact.shape[1] // N_views_src

        # Apply Linear + Bias
        # [B, L_src, 2*D]
        bias_kv = torch.cat((self.zero_k_bias, self.v_bias))
        kv_projected = F.linear(kv_compact, self.mat_kv.weight, bias=bias_kv)

        # Split into K, V and reshape to [B, H, L_src, D]
        kv_projected = kv_projected.view(B, -1, 2, self.embed_dim)
        k, v = kv_projected.unbind(dim=2)
        k = k.reshape(B, N_views_src*patches_per_view_src, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v.reshape(B, N_views_src*patches_per_view_src, self.num_heads, self.head_dim).permute(0,2,1,3)

        src_cache_key = "src_full"
        tgt_cache_key = f"tgt_scale_{scale_ind}" if scale_ind is not None else "tgt_full"
        cached_src = None
        if rope2d_freqs_grid is not None:
             cached_src = rope2d_freqs_grid.get(src_cache_key)


        if cached_src is None:

            # [2] PREPARE SOURCE MATRICES [Proj, rope_x, rope_y]
            P_src, P_T_src, P_inv_src = get_prope_matrices(poses_c2w=poses_src, intrs=intrs_src)


            # prepare matrices for src tokens(single scale)
            _, px_last, py_last = scale_schedule[-1]
            px_last,py_last = int(math.sqrt(patches_per_view_src)), int(math.sqrt(patches_per_view_src))

            num_repeats = px_last * py_last
            x_grid = torch.arange(px_last, device=device).repeat(py_last * N_views_src)
            y_grid = torch.arange(py_last, device=device).repeat_interleave(px_last).repeat(N_views_src)


            pos_x_src = x_grid
            pos_y_src = y_grid

            coeffs_x_src = self._rope_precompute_coeffs(pos_x_src, 100.0, 1.0, self.head_dim // 4)
            coeffs_y_src = self._rope_precompute_coeffs(pos_y_src, 100.0, 1.0, self.head_dim // 4)

            cached_src = (P_inv_src, coeffs_x_src, coeffs_y_src)
            if rope2d_freqs_grid is not None:
                rope2d_freqs_grid[src_cache_key] = cached_src

        P_inv_src, coeffs_x_src, coeffs_y_src = cached_src


        # [3] apply transform to source KV
        k = self._apply_transform(k, P_inv_src, coeffs_x_src, coeffs_y_src, inverse_rope=False)
        v = self._apply_transform(v, P_inv_src, coeffs_x_src, coeffs_y_src, inverse_rope=False)

        # [4] Q projection
        # Project Q: [B, L, C] -> [B, H, L, D]
        q = self.mat_q(q).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)


        cached_tgt = None
        if rope2d_freqs_grid is not None:
            cached_tgt = rope2d_freqs_grid.get(tgt_cache_key)


        if cached_tgt is None:
            if scale_ind is not None:
                current_schedule = [scale_schedule[scale_ind]]
            else:
                current_schedule = scale_schedule

            # prepare tgt matrices
            P_tgt, P_T_tgt, P_inv_tgt = get_prope_matrices(poses_c2w=poses, intrs=intrs)


            is_single_scale = (len(current_schedule) == 1)
            pos_x_tgt, pos_y_tgt = [], []
            P_T_list, P_list = [], [] # We only need P_T (for Q) and P (for Out) in CA


            # Build Multi-scale Target Sequences
            for _, px, py in current_schedule:
                num_repeats = px * py

                x_grid = torch.arange(px, device=device).repeat(py * N_views_tgt)
                y_grid = torch.arange(py, device=device).repeat_interleave(px).repeat(N_views_tgt)


                pos_x_tgt.append(x_grid)
                pos_y_tgt.append(y_grid)

                if not is_single_scale:
                    P_T_list.append(P_T_tgt.repeat_interleave(num_repeats, dim=1))
                    P_list.append(P_tgt.repeat_interleave(num_repeats, dim=1))


            pos_x_tgt = torch.cat(pos_x_tgt, dim=0)
            pos_y_tgt = torch.cat(pos_y_tgt, dim=0)
            coeffs_x_tgt = self._rope_precompute_coeffs(pos_x_tgt, 100.0, 1.0, self.head_dim // 4)
            coeffs_y_tgt = self._rope_precompute_coeffs(pos_y_tgt, 100.0, 1.0, self.head_dim // 4)

            if is_single_scale:
                # FAST PATH
                P_T_expanded_tgt = P_T_tgt
                P_expanded_tgt = P_tgt
            else:
                # SLOW PATH
                P_T_expanded_tgt = torch.cat(P_T_list, dim=1)
                P_expanded_tgt = torch.cat(P_list, dim=1)


            cached_tgt = (P_T_expanded_tgt, None, P_expanded_tgt, coeffs_x_tgt, coeffs_y_tgt)
            if rope2d_freqs_grid is not None:
                rope2d_freqs_grid[tgt_cache_key] = cached_tgt

        P_T_expanded_tgt, _, P_expanded_tgt, coeffs_x_tgt, coeffs_y_tgt = cached_tgt


        q = self._apply_transform(q, P_T_expanded_tgt, coeffs_x_tgt, coeffs_y_tgt, inverse_rope=False)

        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
            )
        # --- 3. ATTENTION ---

        # --- 4. OUTPUT TRANSFORM ---
        out = self._apply_transform(out, P_expanded_tgt, coeffs_x_tgt, coeffs_y_tgt, inverse_rope=True)

        out = out.transpose(1, 2).reshape(B, L, C)
        return self.proj_drop(self.proj(out))

    # --- Helper Methods ---

    def _apply_transform(self, x, mat_seq_or_cameras, coeffs_x, coeffs_y, inverse_rope=False):
        """
        x: (B, H, L, D)
        mat: (B, L, 4, 4) or (B, Cams, 4, 4)
        Applies M @ x (Matrix-Vector product). 
        """
        B, H, L, D = x.shape
        d_half = D // 2
        d_quart = D // 4

        x_proj, x_rope_x, x_rope_y = torch.split(x, [d_half, d_quart, d_quart], dim=-1)


        # --- Projection block ---
        # Efficient Tiling check: if mat is per-camera, use broadcast einsum
        if mat_seq_or_cameras.dim() == 4 and mat_seq_or_cameras.shape[1] <= L and L % mat_seq_or_cameras.shape[1] == 0:

            mat = mat_seq_or_cameras
            cameras = mat.shape[1]
            patches_per_camera = L // cameras

            x_proj_c = x_proj.contiguous()
            x_proj_reshaped = x_proj_c.view(B, H, cameras, patches_per_camera, d_half // 4, 4)

            # einsum: ...ij, ...kj -> ...ki (Standard Matrix-Vector: M @ x)
            # mat: (B, Cams, 4, 4) -> ij
            # x:   (..., 4) -> kj (treated as column vector j)
            proj_out = torch.einsum("bcij,bncpkj->bncpki", mat, x_proj_reshaped)
            x_proj_out = proj_out.reshape(B, H, L, d_half).contiguous()
        else:
            mat = mat_seq_or_cameras
            x_proj_c = x_proj.contiguous()
            x_proj_reshaped = x_proj_c.view(B, H, L, -1, 4)

            # einsum: ...ij, ...kj -> ...ki (Standard Matrix-Vector: M @ x)
            x_proj_out = torch.einsum("blij, bhlkj -> bhlki", mat, x_proj_reshaped).reshape(B, H, L, d_half).contiguous()

        # --- RoPE blocks ---
        x_rope_x_out = self._rope_apply_coeffs(x_rope_x, coeffs_x, inverse=inverse_rope)
        x_rope_y_out = self._rope_apply_coeffs(x_rope_y, coeffs_y, inverse=inverse_rope)

        return torch.cat([x_proj_out, x_rope_x_out, x_rope_y_out], dim=-1).contiguous()

    @staticmethod
    def _rope_precompute_coeffs(positions, freq_base, freq_scale, feat_dim):
        num_freqs = feat_dim // 2
        freqs = freq_scale * (freq_base ** (-torch.arange(num_freqs, device=positions.device) / num_freqs))
        angles = positions[:, None] * freqs[None, :]
        angles = angles.view(1, 1, positions.shape[0], num_freqs)
        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    def _rope_apply_coeffs(feats, coeffs, inverse=False):
        cos, sin = coeffs
        if cos.shape[2] != feats.shape[2]:
            pass

        x_in = feats[..., : feats.shape[-1] // 2]
        y_in = feats[..., feats.shape[-1] // 2 :]
        if not inverse:
            return torch.cat((cos * x_in + sin * y_in, -sin * x_in + cos * y_in), dim=-1)
        else:
            return torch.cat((cos * x_in - sin * y_in, sin * x_in + cos * y_in), dim=-1)


class CrossAttnBlock(nn.Module):
    def __init__(
        self,
        embed_dim, kv_dim, cross_attn_layer_scale, cond_dim, act: bool, shared_aln: bool, norm_layer: partial,
        num_heads, mlp_ratio=4., drop=0., drop_path=0., tau=1, cos_attn=False,
        swiglu=False, customized_flash_attn=False, fused_mlp=False, fused_norm_func=None, checkpointing_sa_only=False,
        use_flex_attn=False, batch_size=2, pad_to_multiplier=1, apply_rope2d=False, rope2d_normalized_by_hw=False,
        N_views_src=1, N_views_tgt=1, use_prope=False,
        switti_attn_backend="sdpa",
    ):
        super(CrossAttnBlock, self).__init__()
        self.C, self.D = embed_dim, cond_dim
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if use_prope:

            self.sa = SelfAttentionPropeOptimized(
                embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn, customized_flash_attn=customized_flash_attn,
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                N_views=N_views_tgt, switti_attn_backend=switti_attn_backend,
            )

        else:
            self.sa = SelfAttention(
                embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn, customized_flash_attn=customized_flash_attn,
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                N_views=N_views_tgt
            )


        if use_prope:

            self.ca = CrossAttentionPropeOptimized(
                embed_dim=embed_dim,
                kv_dim=kv_dim,
                num_heads=num_heads,
                proj_drop=drop,
                cos_attn=cos_attn,
                N_views_src=N_views_src
            )

        else:
            self.ca = CrossAttention(embed_dim=embed_dim, kv_dim=kv_dim, num_heads=num_heads, proj_drop=drop, cos_attn=cos_attn)

        self.using_swiglu = swiglu
        self.ffn = (FFNSwiGLU if swiglu else FFN)(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio / 256) * 256, drop=drop, fused_mlp=fused_mlp)

        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-6)
        self.ca_norm = norm_layer(embed_dim, elementwise_affine=True)

        self.shared_aln = shared_aln
        if self.shared_aln: # always True
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)

        if cross_attn_layer_scale >= 0:
            self.ca_gamma = nn.Parameter(cross_attn_layer_scale * torch.ones(embed_dim), requires_grad=True)
        else:
            self.ca_gamma = 1

        self.checkpointing_sa_only = checkpointing_sa_only
        self.use_prope = use_prope

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self,
                x,
                cond_BD,
                ca_kv, #kv_compact, cu_seqlens_k, max_seqlen_k, kv_compact: [B*N_tokens_reference_view, D]
                attn_bias_or_two_vector,
                attn_fn=None,
                scale_schedule=None,
                rope2d_freqs_grid=None,
                scale_ind=None,
                poses=None,
                intrs=None,
                poses_src=None,
                intrs_src=None,
                input_size=None):    # todo: minGPT and vqgan also uses pre-norm, just like this, while MaskGiT uses post-norm

        with torch.amp.autocast("cuda",enabled=False):    # disable half precision
            if self.shared_aln: # always True;                   (1, 1, 6, C)  + (B, 1, 6, C)
                gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
            else:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)

        if self.use_prope:

            assert poses is not None
            assert intrs is not None

            if self.fused_norm_func is None:
                x_sa = self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1)
                if self.checkpointing_sa_only and self.training:
                    x_sa = checkpoint(self.sa,
                                      x_sa,
                                      attn_bias_or_two_vector,
                                      attn_fn,
                                      scale_schedule,
                                      rope2d_freqs_grid,
                                      scale_ind=scale_ind,
                                      use_reentrant=False,
                                      poses=poses,
                                      intrs=intrs,
                                      input_size=input_size)
                else:
                    x_sa = self.sa(x_sa,
                                   attn_bias_or_two_vector,
                                   attn_fn,
                                   scale_schedule,
                                   rope2d_freqs_grid,
                                   scale_ind=scale_ind,
                                   poses=poses,
                                   intrs=intrs,
                                   input_size=input_size)
                x = x + self.drop_path(x_sa.mul_(gamma1))

                x_ca = self.ca(self.ca_norm(x),
                                ca_kv,
                                poses=poses,
                                poses_src=poses_src,
                                intrs=intrs,
                                intrs_src=intrs_src,
                                scale_schedule=scale_schedule,
                                scale_ind=scale_ind,
                                rope2d_freqs_grid=rope2d_freqs_grid).float().mul_(self.ca_gamma)

                x = x + self.drop_path(x_ca)


                x = x + self.drop_path(self.ffn( self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
            else:
                x_sa = self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale1, shift=shift1)
                if self.checkpointing_sa_only and self.training:
                    x_sa = checkpoint(self.sa,
                                      x_sa,
                                      attn_bias_or_two_vector,
                                      attn_fn,
                                      scale_schedule,
                                      rope2d_freqs_grid,
                                      scale_ind=scale_ind,
                                      use_reentrant=False,
                                      poses=poses,
                                      intrs=intrs,
                                      input_size=input_size)
                else:
                    x_sa = self.sa(x_sa,
                                   attn_bias_or_two_vector,
                                   attn_fn,
                                   scale_schedule,
                                   rope2d_freqs_grid,
                                   scale_ind=scale_ind,
                                   poses=poses,
                                   intrs=intrs,
                                   input_size=input_size)
                x = x + self.drop_path(x_sa.mul_(gamma1))
                x_ca = self.ca(self.ca_norm(x),
                                ca_kv,
                                poses=poses,
                                poses_src=poses_src,
                                intrs=intrs,
                                intrs_src=intrs_src,
                                scale_schedule=scale_schedule,
                                scale_ind=scale_ind,
                                rope2d_freqs_grid=rope2d_freqs_grid).float().mul_(self.ca_gamma)
                x = x + self.drop_path(x_ca)
                x = x + self.drop_path(self.ffn(self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale2, shift=shift2)).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP

        else:

            if self.fused_norm_func is None:
                x_sa = self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1)
                if self.checkpointing_sa_only and self.training:
                    x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
                else:
                    x_sa = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, scale_ind=scale_ind)
                x = x + self.drop_path(x_sa.mul_(gamma1))
                x = x + self.ca(self.ca_norm(x), ca_kv).float().mul_(self.ca_gamma)
                x = x + self.drop_path(self.ffn( self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
            else:
                x_sa = self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale1, shift=shift1)
                if self.checkpointing_sa_only and self.training:
                    x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
                else:
                    x_sa = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, scale_ind=scale_ind)
                x = x + self.drop_path(x_sa.mul_(gamma1))
                x = x + self.ca(self.ca_norm(x), ca_kv).float().mul_(self.ca_gamma)
                x = x + self.drop_path(self.ffn(self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale2, shift=shift2)).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
        return x

    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, fused_norm={self.fused_norm_func is not None}, ca_gamma={"<learnable>" if isinstance(self.ca_gamma, nn.Parameter) else self.ca_gamma}'


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, act: bool, norm_layer: partial, fused_norm_func=None):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-6)
        lin = nn.Linear(D, 2*C)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)

    def forward(self, x_BLC: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        if self.fused_norm_func is None:
            return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
        else:
            return self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x_BLC, scale=scale, shift=shift)
