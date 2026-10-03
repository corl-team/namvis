"""
Wrap torch's flex attention and handle mess info or potentially refactor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    flex_attention_available = True
except ImportError:
    print(f"[Warning] flex attention need pytorch 2.5.0+ but your version is {torch.__version__}")
    flex_attention_available = False


_compiled_flex_attention = None


# PyTorch 2.5.1 can select a 128-token FlexAttention tile whose shared-memory
# requirement exceeds the 99 KiB limit of SM80 GPUs (for example, head_dim=128
# requested 131074 bytes on an A100).  Keep the sparse mask block size at 128,
# but compute each sparse block using smaller Triton tiles. FP32 ProPE Q/K/V
# require 32-token forward and 16-token backward tiles to remain below that
# limit. The backward names are the PyTorch 2.5 kernel-option names (without
# the newer fwd_/bwd_ prefixes).
SWITTI_FLEX_KERNEL_OPTIONS = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "BLOCK_M1": 16,
    "BLOCK_N1": 16,
    "BLOCK_M2": 16,
    "BLOCK_N2": 16,
}


def get_compiled_flex_attention():
    """Return one lazily compiled FlexAttention callable per process."""
    global _compiled_flex_attention
    if not flex_attention_available:
        raise RuntimeError(
            "FlexAttention requires PyTorch 2.5.1 or newer; "
            f"the current PyTorch version is {torch.__version__}."
        )
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(
            flex_attention,
            backend="inductor",
            # PyTorch 2.5.1 cannot reliably lower FlexAttention when the
            # sequence and BlockMask dimensions remain symbolic. Specialize
            # once for each runtime target-view/sequence-length combination.
            dynamic=False,
        )
    return _compiled_flex_attention

# import os
# os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"



def _causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def _length_to_offsets(lengths, device):
    """Converts a list of lengths to a list of offsets.

    Args:
        lengths: A list of lengths.

    """
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets

def _generate_var_mask_mod(offsets):
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        offsets: This tensor should be of shape(num_documents + 1)
            this should contain the cumulative counts of document tokens.
            e.g. if you have 3 documents of length 2, 4, 3 then
            offsets = [0, 2, 6, 9]

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.
    """

    def _offsets_to_doc_ids_tensor(offsets):
        device = offsets.device
        counts = offsets[1:] - offsets[:-1]
        return torch.repeat_interleave(
            torch.arange(len(counts), device=device, dtype=torch.int32), counts
        )

    document_id = _offsets_to_doc_ids_tensor(offsets)

    def var_mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        causal_mask = _causal_mask(b, h, q_idx, kv_idx)
        return same_doc | causal_mask

    return var_mask_mod


def build_switti_scale_ids(block_scales, n_views, sequence_length, device):
    """Build the scale ID for every token in a scale-major SWITTI sequence."""
    if n_views < 1:
        raise ValueError(f"n_views must be positive, got {n_views}")
    if not block_scales:
        raise ValueError("block_scales must contain at least one scale")

    lengths = [
        int(n_views) * int(t) * int(h) * int(w)
        for t, h, w in block_scales
    ]
    expected_length = sum(lengths)
    if expected_length != int(sequence_length):
        raise ValueError(
            "SWITTI FlexAttention requires an unpadded scale-major sequence: "
            f"expected {expected_length} tokens from block_scales and "
            f"n_views={n_views}, got {sequence_length}."
        )
    return torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.int32, device=device),
        torch.tensor(lengths, dtype=torch.int64, device=device),
    )


def _generate_switti_mask_mod(scale_ids):
    """Allow attention only between tokens belonging to the same scale."""
    sequence_length = scale_ids.numel()
    last_valid_index = sequence_length - 1

    def switti_mask_mod(b, h, q_idx, kv_idx):
        # FlexAttention evaluates mask_mod over complete 128-token tiles, so
        # q_idx/kv_idx can point just beyond a non-aligned sequence tail. Clamp
        # before indexing and then explicitly hide every padded position.
        q_is_valid = q_idx < sequence_length
        kv_is_valid = kv_idx < sequence_length
        q_safe = torch.clamp(q_idx, max=last_valid_index)
        kv_safe = torch.clamp(kv_idx, max=last_valid_index)
        return (
            q_is_valid
            & kv_is_valid
            & (scale_ids[q_safe] == scale_ids[kv_safe])
        )

    return switti_mask_mod


def create_switti_block_mask(block_scales, n_views, sequence_length, device):
    """Create an exact same-scale BlockMask broadcast across batches and heads."""
    if not flex_attention_available:
        raise RuntimeError(
            "switti_attn_backend='flex' requires PyTorch FlexAttention "
            f"(torch>=2.5.1); current version is {torch.__version__}."
        )
    scale_ids = build_switti_scale_ids(
        block_scales,
        n_views,
        sequence_length,
        device,
    )
    return create_block_mask(
        _generate_switti_mask_mod(scale_ids),
        B=None,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=device,
        _compile=True,
    )

def _generate_var_infer_mask_with_kv_cache(lengths):
    kv_len = sum(lengths)
    def var_mask_mod(b, h, q_idx, kv_idx):
        return kv_idx < kv_len

    return var_mask_mod

class FlexAttn(nn.Module):
    def __init__(
            self, 
            block_scales:list, 
            mask_type:str, 
            B, H, L:int, 
            N_views:int,
            auto_padding=False
    ):
        """
        :param block_scales: accept VAR's block sizes like [(1,1), (2,2), (3,3)]
        :param mask_type: var/causal
        :param B: batch size
        :param H: heads num
        :param L: sequence length
        """
        super().__init__()
        if not flex_attention_available:
            raise NotImplementedError((f"[Error] flex attention need pytorch 2.5.0+ but your version is {torch.__version__}"))

        self.support_mask_type = ["var", "causal", "var_infer_mask_with_kv_cache"]
        self.auto_padding = auto_padding

        # import torch._dynamo
        # torch._dynamo.reset()

        compile_options = {"max_autotune": True}
        self.flex_attention = torch.compile(
            flex_attention,
            backend="inductor",
            options=compile_options,
        )

        self.block_scales = block_scales
        self.N_views=N_views
        self.lengths = [ self.N_views*x * y * z for x,y,z in block_scales]

        self.offsets = _length_to_offsets(self.lengths, device='cuda')

        print(f"[FlexAttn] self.offsets: {self.offsets}")

        # if L paded to align 128, block need to cover padding area
        if self.offsets[-1] < L:
            self.offsets = torch.cat((self.offsets, torch.tensor([L], device='cuda')), dim=0)

        if mask_type == "var": #during training
            self.mask_mod = _generate_var_mask_mod(self.offsets)
            self.block_mask = create_block_mask(self.mask_mod, B = B, H = H, Q_LEN = L, KV_LEN = L, device = 'cuda', _compile = True)
        elif mask_type == "causal":
            self.mask_mod = _causal_mask
            self.block_mask = create_block_mask(self.mask_mod, B = B, H = H, Q_LEN = L, KV_LEN = L, device = 'cuda', _compile = True)
        elif mask_type == 'var_infer_mask_with_kv_cache':
            self.mask_mod = _generate_var_infer_mask_with_kv_cache(self.lengths)
            self.block_mask = create_block_mask(self.mask_mod, B = B, H = H, Q_LEN = L, KV_LEN = L, device = 'cuda', _compile = True)
        else:
            raise NotImplementedError(f"{mask_type} not supportted in FlexAttn, support type:{self.support_mask_type}")


    def forward(self, q, k, v, scale = None):
        if self.auto_padding:
            q_pad_len = (128 - q.shape[-2] % 128) % 128
            kv_pad_len = (128 - k.shape[-2] % 128) % 128
            q_pad = F.pad(q, (0, 0, 0, q_pad_len))
            k_pad = F.pad(k, (0, 0, 0, kv_pad_len))
            v_pad = F.pad(v, (0, 0, 0, kv_pad_len))
            oup = self.flex_attention(q_pad.to(v_pad.dtype), k_pad.to(v.dtype), v_pad, block_mask = self.block_mask, scale = scale)
            if q_pad_len > 0:
                oup = oup[:,:,:-q_pad_len]
        else:
            oup = self.flex_attention(q.to(v.dtype), k.to(v.dtype), v, block_mask = self.block_mask, scale = scale)
        return oup

    def extra_repr(self) -> str:
        tail = ''
        return f'block size:{self.block_scales} {tail}'
