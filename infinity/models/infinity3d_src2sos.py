"""
Definition of Infinity transformer model.
"""

import math
import random
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from timm.models import register_model
import numpy as np

import infinity.utils.dist as dist
from infinity.utils.dist import for_visualize
from infinity.models.basic import flash_attn_func, flash_fused_op_installed, AdaLNBeforeHead, CrossAttnBlock, CrossAttention, FastRMSNorm
from infinity.models.flex_attn import FlexAttn
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

try:
    from infinity.models.fused_op import fused_ada_layer_norm, fused_ada_rms_norm
except:
    fused_ada_layer_norm, fused_ada_rms_norm = None, None


class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class TextAttentivePool(nn.Module):
    def __init__(self, Ct5: int, D: int):
        super().__init__()
        # 2048, 2048
        self.Ct5, self.D = Ct5, D
        if D > 4096:
            self.head_dim = 64 
        else:
            self.head_dim = 128

        self.num_heads = Ct5 // self.head_dim
        self.ca = CrossAttention(for_attn_pool=True, embed_dim=self.D, kv_dim=Ct5, num_heads=self.num_heads)
    def forward(self, ca_kv):
        return self.ca(None, ca_kv).squeeze(1)

class ImageAttentivePool(nn.Module):
    def __init__(self, D_vae: int, D_out: int):
        """
        Attentive pooling for VAE tokens.
        Q: [1, D]
        K,V: [h*w, D]
        Args:
            D_vae (int): The dimension of VAE tokens (e.g., 32).
            D_out (int): The target output dimension (e.g., 2048).
        """
        super().__init__()
        self.D_vae, self.D_out = D_vae, D_out
        
        # Use the same head_dim logic as the original, or pick a standard one
        self.head_dim = 128
        
        # Calculate num_heads based on the *output* dimension D_out
        # This is the standard way (Q-dimension)
        assert D_out % self.head_dim == 0, "Output dimension D_out must be divisible by head_dim"
        self.num_heads = D_out // self.head_dim

        # This is the new cross-attention layer
        # It will have a learnable query (Q) of dim D_out (4096)
        # and accept keys/values (K/V) of dim D_vae (32)
        self.ca = CrossAttention(
            for_attn_pool=True,     # This is key!
            embed_dim=self.D_out,   # Q dim (4096)
            kv_dim=self.D_vae,    # K/V dim (32)
            num_heads=self.num_heads
        )
        
    def forward(self, 
                vae_tokens_BLD: torch.Tensor, #vae features: [B, h*w, 32] h=H//8, w=W//8
                ):
        """
        Takes a batch of VAE token sequences and returns a single pooled vector per sequence.
        Args:
            vae_tokens_BLD (torch.Tensor): Shape [B, L, D_vae], e.g., [B, 520, 32]
        
        Returns:
            torch.Tensor: Shape [B, D_out], e.g., [B, 4096]
        """
        # print(f"vae_tokens_BLD.shape: {vae_tokens_BLD.shape}")
        B, L, _ = vae_tokens_BLD.shape
        
        
        # [B, L, D_vae] -> [B*L, D_vae]
        kv_compact = vae_tokens_BLD.reshape(B * L, self.D_vae).contiguous()
        
        # Create sequence length info
        cu_seqlens_k = torch.arange(0, (B+1)*L, step=L, dtype=torch.int32, device=vae_tokens_BLD.device)
        max_seqlen_k = L
        
        ca_kv_tuple = (kv_compact, cu_seqlens_k, max_seqlen_k)
        
        # The CrossAttention module (with for_attn_pool=True)
        # will use its internal learnable query to attend to this `ca_kv_tuple`.
        # self.ca(None, ...) signals it to use its learnable query.
        return self.ca(None, ca_kv_tuple).squeeze(1)

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).reshape(-1, 1, 6, C)   # B16C


class MultipleLayers(nn.Module):
    def __init__(self, ls, num_blocks_in_a_chunk, index):
        super().__init__()
        self.module = nn.ModuleList()
        for i in range(index, index+num_blocks_in_a_chunk):
            self.module.append(ls[i])

    def forward(self,
                x, 
                cond_BD, 
                ca_kv, 
                attn_bias_or_two_vector,
                poses=None,
                intrs=None,
                poses_src=None,
                intrs_src=None,
                input_size=None,
                attn_fn=None, 
                scale_schedule=None, 
                checkpointing_full_block=False, 
                rope2d_freqs_grid=None):
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(m, 
                                                      h,
                                                      cond_BD,
                                                      ca_kv,
                                                      attn_bias_or_two_vector,
                                                      attn_fn,
                                                      scale_schedule,
                                                      rope2d_freqs_grid,
                                                      poses=poses,
                                                      intrs=intrs,
                                                      poses_src=poses_src,
                                                      intrs_src=intrs_src,
                                                      input_size=input_size,
                                                      use_reentrant=False)
            else:
                h = m(h, 
                      cond_BD,
                      ca_kv,
                      attn_bias_or_two_vector,
                      attn_fn,
                      scale_schedule,
                      rope2d_freqs_grid,
                      poses=poses,
                      intrs=intrs,
                      poses_src=poses_src,
                      intrs_src=intrs_src,
                      input_size=input_size,)
        return h

class Infinity3DSrc2Sos(nn.Module):
    def __init__(
        self, vae_local,
        text_channels=0, text_maxlen=0,     # text-cond generation
        selecting_idx=None,                 # class-cond generation
        embed_dim=1024, depth=16, num_heads=16, mlp_ratio=4.,   # model's architecture
        drop_rate=0., drop_path_rate=0.,    # drop out and drop path
        norm_eps=1e-6, rms_norm=False,      # norm layer
        shared_aln=False, head_aln=True,    # adaptive norm
        cond_drop_rate=0.1,                 # for classifier-free guidance
        rand_uncond=False,
        cross_attn_layer_scale=-1., nm0=False, tau=1, cos_attn=True, swiglu=False,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        head_depth=1,
        top_p=0.0, top_k=0.0,
        customized_flash_attn=False, fused_mlp=False, fused_norm=False,
        block_chunks=1,
        checkpointing=None,
        pad_to_multiplier=0,
        use_flex_attn=False,
        batch_size=2,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        rope2d_each_sa_layer=0,
        rope2d_normalized_by_hw=0,
        pn=None,
        train_h_div_w_list=None,
        video_frames=1,
        always_training_scales=20,
        apply_spatial_patchify = 0,
        inference_mode=False,
        N_views_tgt=1,
        N_views_src=1,
        use_prope=False,
    ):
        # set hyperparameters
        self.C = embed_dim
        self.inference_mode = inference_mode
        self.apply_spatial_patchify = apply_spatial_patchify
        if self.apply_spatial_patchify:
            self.d_vae = vae_local.embed_dim * 4
        else:
            self.d_vae = vae_local.embed_dim
        self.use_bit_label = use_bit_label
        self.codebook_dim = self.d_vae

        # self.V = 32*2 = 64, i.e. 1s and 0s
        self.V = (self.codebook_dim * 2) if self.use_bit_label else vae_local.vocab_size
        self.bit_mask = vae_local.quantizer.lfq.mask if self.use_bit_label else None
        self.Ct5 = text_channels
        self.depth = depth
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.mlp_ratio = mlp_ratio
        self.cond_drop_rate = cond_drop_rate
        self.norm_eps = norm_eps
        self.prog_si = -1
        self.pn = pn
        self.train_h_div_w_list = train_h_div_w_list if train_h_div_w_list else h_div_w_templates
        self.video_frames = video_frames
        self.always_training_scales = always_training_scales

        self.N_views_tgt = N_views_tgt
        self.N_views_src = N_views_src

        assert add_lvl_embeding_only_first_block in [0,1]
        self.add_lvl_embeding_only_first_block = add_lvl_embeding_only_first_block
        assert rope2d_each_sa_layer in [0,1]
        self.rope2d_each_sa_layer = rope2d_each_sa_layer
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        head_up_method = ''
        word_patch_size = 1 if head_up_method in {'', 'no'} else 2
        if word_patch_size > 1:
            assert all(raw_pn % word_patch_size == 0 for raw_pn in raw_scale_schedule), f'raw_scale_schedule={raw_scale_schedule}, not compatible with word_patch_size={word_patch_size}'
        
        self.checkpointing = checkpointing
        self.pad_to_multiplier = max(1, pad_to_multiplier)
        
        customized_kernel_installed = any('Infinity' in arg_name for arg_name in flash_attn_func.__code__.co_varnames)
        self.customized_flash_attn = customized_flash_attn and customized_kernel_installed
        if customized_flash_attn and not customized_kernel_installed:
            import inspect, warnings
            file_path = inspect.getsourcefile(flash_attn_func)
            line_number = inspect.getsourcelines(flash_attn_func)[1]
            info = (
                f'>>>>>> Customized FlashAttention2 is not installed or compiled, but specified in args by --flash=1. Set customized_flash_attn = False. <<<<<<\n'
                f'>>>>>> `flash_attn_func` is in [line {line_number}] [file {file_path}] <<<<<<\n'
                f'>>>>>> {flash_attn_func.__code__.co_varnames=} <<<<<<\n'
            )
            warnings.warn(info, ImportWarning)
            print(info, flush=True)
        
        self.raw_scale_schedule = raw_scale_schedule    # 'raw' means before any patchifying
        self.first_l = 1
        # solve top-p top-k sampling hyperparameters
        self.top_p, self.top_k = max(min(top_p, 1), 0), (round(top_k * self.V) if 0 < top_k < 1 else round(top_k))
        if self.top_p < 1e-5: self.top_p = 0
        if self.top_k >= self.V or self.top_k <= 0: self.top_k = 0
        
        t = torch.zeros(dist.get_world_size(), device=dist.get_device())
        t[dist.get_rank()] = float(flash_fused_op_installed)
        dist.barrier()
        dist.allreduce(t)
        assert round(t.sum().item()) in {0, dist.get_world_size()}, f'flash_fused_op_installed: {t}'
        
        super().__init__()
        self.rng = torch.Generator(device=dist.get_device())
        self.maybe_record_function = nullcontext
        self.text_maxlen = text_maxlen
        self.t2i = text_channels != 0
        
        self.use_prope = use_prope
        # [inp & position embedding]
        init_std = math.sqrt(1 / self.C / 3)
        self.norm0_cond = nn.Identity()
        if self.t2i:
            self.selecting_idx = None
            self.num_classes = 0
            self.D = self.C

            #AT THE MOMENT CFG WORKS FOR square images only
            assert len(self.train_h_div_w_list) == 1
            h_div_w = self.train_h_div_w_list[0]
            h_div_w_template = h_div_w_templates[np.argmin(np.abs(float(h_div_w) - h_div_w_templates))]
            full_scale_schedule = dynamic_resolution_h_w[h_div_w_template][self.pn]['scales']

            token_schedule = torch.tensor([h*w for (_,h,w) in full_scale_schedule])
            n_tokens_total_perview = torch.sum(token_schedule[1:])
            # n_tokens_total = self.N_views_tgt * n_tokens_total_perview

            rng = torch.Generator(device='cpu')
            rng.manual_seed(0)

            # cfg_uncond = torch.empty(self.text_maxlen, self.Ct5)
            # rng = torch.Generator(device='cpu')
            # rng.manual_seed(0)
            # torch.nn.init.trunc_normal_(cfg_uncond, std=1.2, generator=rng)
            # cfg_uncond /= self.Ct5 ** 0.5

            # [512, 2048]
            # if rand_uncond:
            #     self.register_buffer('cfg_uncond', cfg_uncond)
            # else:
            #     self.cfg_uncond = nn.Parameter(cfg_uncond)

            # print("cfg_uncond (init) shape, dtype, device, ndim, is_param:", 
            # getattr(self, 'cfg_uncond').shape, getattr(self, 'cfg_uncond').dtype,
            # getattr(self, 'cfg_uncond').device if hasattr(getattr(self,'cfg_uncond'), 'device') else 'no-device',
            # getattr(self, 'cfg_uncond').ndim,
            # isinstance(getattr(self,'cfg_uncond'), nn.Parameter))
            
            self.text_norm = FastRMSNorm(self.Ct5, elementwise_affine=True, eps=norm_eps)
            self.text_proj_for_sos = TextAttentivePool(self.Ct5, self.D)
            self.text_proj_for_ca = nn.Sequential(
                nn.Linear(self.Ct5, self.D),
                nn.GELU(approximate='tanh'),
                nn.Linear(self.D, self.D),
            )

            self.D_pool = 32

            # self.D_cond = 512
            self.D_cond = 32
            self.D_pose = 0 if self.use_prope else 6
            self.img_norm = FastRMSNorm(self.D_cond, elementwise_affine=True, eps=norm_eps)
            
            self.img_proj_for_sos = ImageAttentivePool(self.D_pool+self.D_pose, self.D)
            self.img_proj_for_ca = nn.Sequential(
                nn.Linear(self.D_cond+self.D_pose, self.D),
                nn.GELU(approximate='tanh'),
                nn.Linear(self.D, self.D),
            )

            N_tokens_image = 16
            cfg_uncond = torch.empty(N_tokens_image, N_tokens_image, self.D_cond)
            torch.nn.init.trunc_normal_(cfg_uncond, std=1.2, generator=rng)
            cfg_uncond /= self.D_cond ** 0.5
            if rand_uncond:
                self.register_buffer('cfg_uncond', cfg_uncond)
            else:
                self.cfg_uncond = nn.Parameter(cfg_uncond)


        else:   # class-label cond
            if selecting_idx is None:
                num_classes = 1000
                print(f'======= WARNING: selecting_idx not specified, set to 1/{num_classes} @ {dist.get_device()} =======')
                selecting_idx = torch.full((1, num_classes), fill_value=1/num_classes, dtype=torch.float32, device=dist.get_device())
            self.selecting_idx = selecting_idx
            self.num_classes = selecting_idx.shape[-1]
            self.D = self.C
            self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
            nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # if self.rope2d_each_sa_layer:
        #     rope2d_freqs_grid = precompute_rope2d_freqs_grid(dim=self.C//self.num_heads, 
        #                                                      dynamic_resolution_h_w=dynamic_resolution_h_w, 
        #                                                      pad_to_multiplier=self.pad_to_multiplier, 
        #                                                      rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
        #                                                      N_views=self.N_views_tgt)
        #     self.rope2d_freqs_grid = rope2d_freqs_grid
        # else:
        #     raise ValueError(f'self.rope2d_each_sa_layer={self.rope2d_each_sa_layer} not implemented')

        self.rope2d_freqs_grid = {}
        
        self.lvl_embed = nn.Embedding(15, self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # [input layers] input norm && input embedding
        norm_layer = partial(FastRMSNorm if rms_norm else nn.LayerNorm, eps=norm_eps)
        self.norm0_ve = norm_layer(self.d_vae) if nm0 else nn.Identity()

        if not self.use_prope:
            self.word_embed = nn.Linear(self.d_vae+6, self.C)
        else:
            self.word_embed = nn.Linear(self.d_vae, self.C)
        # self.word_embed = nn.Linear(self.d_vae, self.C)

        # [shared adaptive layernorm mapping network]
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        # fused norm
        if fused_norm:
            fused_norm_func = fused_ada_rms_norm if rms_norm else fused_ada_layer_norm
            if fused_norm_func is not None: # pre-compile
                B = 2
                x = torch.randn(B, 1, self.C).requires_grad_(True)
                scale = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                shift = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                # fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale, shift=shift).mean().backward()
                del B, x, scale, shift
        else:
            fused_norm_func = None
        
        # [backbone and head]
        self.use_flex_attn = use_flex_attn
        self.attn_fn_compile_dict = {}
        self.batch_size = batch_size
        if self.use_flex_attn:
            self.attn_fn_compile_dict = self.compile_flex_attn()

        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # dpr means drop path rate (linearly increasing)
        self.unregistered_blocks = []
        for block_idx in range(depth):
            block = CrossAttnBlock(
                embed_dim=self.C, kv_dim=self.D, cross_attn_layer_scale=cross_attn_layer_scale, cond_dim=self.D, act=True, shared_aln=shared_aln, norm_layer=norm_layer,
                num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[block_idx], tau=tau, cos_attn=cos_attn,
                swiglu=swiglu, customized_flash_attn=self.customized_flash_attn, fused_mlp=fused_mlp, fused_norm_func=fused_norm_func,
                checkpointing_sa_only=self.checkpointing == 'self-attn',
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                N_views_tgt=self.N_views_tgt,
                N_views_src=self.N_views_src,
                use_prope=use_prope,
            )
            self.unregistered_blocks.append(block)
        
        # [head]
        V = self.V
        if head_aln:
            self.head_nm = AdaLNBeforeHead(self.C, self.D, act=True, norm_layer=norm_layer, fused_norm_func=fused_norm_func)
            self.head = nn.Linear(self.C, V) if head_depth == 1 else nn.Sequential(nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        else:
            # self.head_nm = MultiInpIdentity()
            # self.head = nn.Sequential(norm_layer(self.C), nn.Linear(self.C, V)) if head_depth == 1 else nn.Sequential(norm_layer(self.C), nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))

            self.head_nm = MultiInpIdentity()
            self.head = nn.Linear(self.C, V) if head_depth==1 else nn.Sequential(nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))

        # block_chunks = depth
        self.num_block_chunks = block_chunks or 1
        # self.num_blocks_in_a_chunk = 32 // 8
        self.num_blocks_in_a_chunk = depth // block_chunks

        assert self.num_blocks_in_a_chunk * block_chunks == depth
        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i*self.num_blocks_in_a_chunk))
        print(
            f'\n[constructor]  ==== customized_flash_attn={self.customized_flash_attn} (using_flash={sum(b.sa.using_flash for b in self.unregistered_blocks)}/{self.depth}), fused_mlp={fused_mlp} (fused_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.unregistered_blocks)}/{self.depth}) ==== \n'
            f'    [Infinity config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}, swiglu={swiglu} num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}\n'
            f'    [drop ratios] drop_rate={drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )


    def _inference_block_groups(self):
        """Yield attention layers, unwrapping FSDP chunks during evaluation.

        FSDP callers must materialize parameters with summon_full_params before
        running inference directly through the underlying attention layers.
        """
        if self.num_block_chunks == 1:
            yield self.unregistered_blocks
        else:
            for chunk in self.block_chunks:
                if isinstance(chunk, FSDP):
                    chunk = chunk.module
                yield chunk.module

    def compile_flex_attn(self):
        attn_fn_compile_dict = {}
        for h_div_w in self.train_h_div_w_list:
            h_div_w_template = h_div_w_templates[np.argmin(np.abs(float(h_div_w) - h_div_w_templates))]
            full_scale_schedule = dynamic_resolution_h_w[h_div_w_template][self.pn]['scales']
            print(f"[compile_flex_attn] ")
            print(f"full_scale_schedule: {full_scale_schedule}")
            if self.inference_mode:
                # apply_flex_attn_scales: [1, n_scales]
                apply_flex_attn_scales = list(range(1, 1+len(full_scale_schedule)))
                mask_type = "infinity_infer_mask_with_kv_cache"
                auto_padding = True
            else:
                mask_type = 'var'
                auto_padding = False
                apply_flex_attn_scales = [min(self.always_training_scales, len(full_scale_schedule))]

            print(f"self.always_training_scales: {self.always_training_scales}")
            print(f"apply_flex_attn_scales: {apply_flex_attn_scales}")
            
            # go over all scales
            for scales_num in apply_flex_attn_scales:
                print(f'====== apply flex attn hdivw: {h_div_w} scales: {scales_num} ======')
                scale_schedule = full_scale_schedule[:scales_num]
                scale_schedule = [ (min(t, self.video_frames//4+1), h, w) for (t,h, w) in scale_schedule]
                print(f"[min] scale_schedule: {scale_schedule}")
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( self.N_views * pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                print(f"self.N_views: {self.N_views}")
                print(f"SEQ_L: {SEQ_L}")
                print(f"aligned_L: {aligned_L}")
                print(f"patchs_nums_tuple: {patchs_nums_tuple}")
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L,
                                        N_views = self.N_views,
                                        auto_padding=auto_padding)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn

            if self.video_frames > 1: # append image attn_fn when self.video_frames > 1 (namely videos)
                scale_schedule = [ (1, h, w) for (t,h, w) in scale_schedule]
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn
        return attn_fn_compile_dict
        
    def get_logits(self, h: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        """
        :param h: hidden_state, shaped (B or batch_size, L or seq_len, C or hidden_dim)
        :param cond_BD: shaped (B or batch_size, D or cond_dim)
        :param tau: temperature
        :return: logits, shaped (B or batch_size, V or vocabulary_size)
        """
        with torch.amp.autocast('cuda', enabled=False):
            return self.head(self.head_nm(h.float(), cond_BD.float()))

    def add_lvl_embeding(self, feature, scale_ind, scale_schedule, need_to_pad=0, N_views=1):
        bs, seq_len, c = feature.shape

        # get total number of tokens for current scale
        patch_t, patch_h, patch_w = scale_schedule[scale_ind]
        t_mul_h_mul_w = N_views * patch_t * patch_h * patch_w

        # print(f"scale_ind: {scale_ind}")
        lvl_embed0 = self.lvl_embed(scale_ind*torch.ones((bs, t_mul_h_mul_w),dtype=torch.int).to(feature.device))

        # extract embedding from nn.Embedding and add it to features
        assert t_mul_h_mul_w + need_to_pad == seq_len
        feature[:, :t_mul_h_mul_w] += self.lvl_embed(scale_ind*torch.ones((bs, t_mul_h_mul_w),dtype=torch.int).to(feature.device))
        return feature
    
    def add_lvl_embeding_for_x_BLC(self, 
                                   x_BLC, 
                                   scale_schedule, 
                                   need_to_pad=0, 
                                   N_views=1):
        ptr = 0
        x_BLC_list = []
        for scale_ind, patch_t_h_w in enumerate(scale_schedule):
            # get total number of tokens for scale
            scale_seq_len = N_views * np.array(patch_t_h_w).prod()
            # x_BLC_this_scale: [B, scale_seq_len, D] | get tokens of current scale 
            x_BLC_this_scale = x_BLC[:,ptr:ptr+scale_seq_len] # shape: [bs, patch_h*patch_w, c]
            ptr += scale_seq_len
            # add scale embedding 
            x_BLC_this_scale = self.add_lvl_embeding(x_BLC_this_scale, scale_ind, scale_schedule, N_views=N_views)
            # print(f"[after lvl] x_BLC_this_scale.shape: {x_BLC_this_scale.shape}")
            x_BLC_list.append(x_BLC_this_scale)

        assert x_BLC.shape[1] == (ptr + need_to_pad), f'{x_BLC.shape[1]} != {ptr} + {need_to_pad}'
        # at the end add all the rest(padding)
        x_BLC_list.append(x_BLC[:,ptr:])
        x_BLC = torch.cat(x_BLC_list, dim=1)
        return x_BLC

    def forward(self, 
                label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], #text features, (kv_compact, lens, cu_seqlens_k, Ltext)
                vae_features_src, #features of src [B, N_views_src, 32(512), 16, 16]
                x_BLC_wo_prefix: torch.Tensor, #GT tokens for prefix: [B*N_views_tgt, N_tokens(520), 32]
                scale_schedule: List[Tuple[int]],
                images_src=None,
                cfg_infer=False,
                N_views=1,
                rays=None, #plucker rays for target [B, N_views, N_tokens, 6]
                poses=None, #poses for target [B, N]
                intrs=None, #intrs for target
                rays_src=None,
                poses_src=None,
                intrs_src=None,
                input_size=[1,1],
                **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # returns logits_BLV
        """
        label_B_or_BLT: label_B or (kv_compact, cu_seqlens_k, max_seqlen_k) text features
        :return: logits BLV, V is vocab_size
        """
        self.rope2d_freqs_grid.clear()
        if cfg_infer:
            return self.autoregressive_infer_cfg(label_B_or_BLT=label_B_or_BLT, 
                                                 scale_schedule=scale_schedule,
                                                 **kwargs)
        
        x_BLC_wo_prefix = x_BLC_wo_prefix.float()       # input should be float32
        
        N_views_src = poses_src.shape[1]
        token_schedule = [scale_i[0]*scale_i[1]*scale_i[2] for scale_i in scale_schedule]

        # get real batch size and reshape input
        B = x_BLC_wo_prefix.shape[0] // N_views
        x_BLC_wo_prefix = x_BLC_wo_prefix.reshape(B, N_views, *x_BLC_wo_prefix.shape[1:])

        # [B*N_views_src, 512, 16, 16] -> [B*N_views_src, 16, 16, 512]
        vae_features_src = vae_features_src.permute(0,2,3,1)

        # [1. get input sequence x_BLC]
        with torch.amp.autocast('cuda', enabled=False):


            # self.cfg_uncond [16*16,512]
            for i in range(vae_features_src.shape[0]):
                if random.random() < self.cond_drop_rate:
                    vae_features_src[i] = self.cfg_uncond
                
            #NOTE trick for ddp, to make all nn.Parameter participate in graph
            must_on_graph = self.cfg_uncond[0, 0, 0] * 0
            vae_features_src[0,0,0,0] += must_on_graph

            vae_features_src = vae_features_src.reshape(B,N_views_src,*vae_features_src.shape[1:])
            vae_height, vae_width = vae_features_src.shape[2], vae_features_src.shape[3]

            vae_features_src = self.img_norm(vae_features_src)

            vae_features = []
            if not self.use_prope:
                # add src rays
                start = 0
                for idx, n_tokens_i in enumerate(token_schedule):
                    end = start + n_tokens_i
                    if idx != len(token_schedule)-1:
                        start=end
                        continue

                    for v in range(N_views_src):
                        # print(f"rays_src[:,v,start:end].shape: {rays_src[:,v,start:end].shape}")
                        # [B,H,W,32] (+) [B,H,W,6]
                        vae_features_src_posed_i = torch.cat((vae_features_src[:,v],
                                                            rays_src[:,v,start:end].reshape(B,vae_height,vae_width,6)), 
                                                            dim=-1).reshape(B,vae_height*vae_width,-1)                
                        vae_features.append(vae_features_src_posed_i)
                    start=end

                vae_features_src = torch.cat(vae_features, dim=1)
            
            # print(f"[1]vae_features_src.shape: {vae_features_src.shape}")
            # [B*N_views_src, 16, 16, 512] -> [B,N_views_src*16*16,512]
            vae_features_src = vae_features_src.reshape(B,-1,self.D_cond+self.D_pose)

            # sos: [B, 1, 2048]

            # print(f"[text] sos.shape: {sos.shape}")
            # print(f"[forward] kv_compact.shape: {kv_compact.shape}")

            #TODO(change to unpacked dense) create packed version, similar to text
            #kv_embed: [B, H*W, 2048]
            kv_embed = self.img_proj_for_ca(vae_features_src)
            #kv_embed: [B*H*W, 2048]
            kv_compact = kv_embed.reshape(-1, self.D).contiguous()
            L_src = vae_features_src.shape[1]
            cu_seqlens_k = torch.arange(0, (B + 1) * L_src, step=L_src, dtype=torch.int32, device=kv_compact.device)
            max_seqlen_k = L_src
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
            
            sos = cond_BD = self.img_proj_for_sos(vae_features_src)

            cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()  # gss: gamma, scale, shift; cond_BD_or_gss should be float32


            sos = sos.unsqueeze(1).expand(B, 1, -1) + self.pos_start.expand(B, 1, -1)            
            token_schedule = [scale_i[0]*scale_i[1]*scale_i[2] for scale_i in scale_schedule]

            x_BLC = []
            # add 'sos' for every target image
            for v in range(N_views):
                x_BLC.append(sos)

            start = 0
            for idx, n_tokens_i in enumerate(token_schedule):
                # first idx not used, instead 'sos' is used
                if idx==0:
                    continue
                end = start + n_tokens_i
                for v in range(N_views):

                    # cat together prefix and raymaps
                    # x_BLC_posed: [B, N_tokens, 32+6]
                    if not self.use_prope:
                        x_BLC_posed = torch.cat((self.norm0_ve(x_BLC_wo_prefix[:,v, start:end]),
                                                rays[:,v,start:end]), 
                                                dim=-1)
                    else:
                        x_BLC_posed = self.norm0_ve(x_BLC_wo_prefix[:,v, start:end])
                    # 32+6 -> 2048                        
                    x_BLC_posed = self.word_embed(x_BLC_posed)
                    x_BLC.append(x_BLC_posed)

                start = end

            # [B, N_views*521, 2048]
            x_BLC = torch.cat(x_BLC, dim=1) 

            # [1.1. pad the seqlen dim] turned off for now, cause flex_attn=False
            # causal mask, tokens of same scale see all tokens from other views
            l_end = x_BLC.shape[1] // N_views

            need_to_pad = (l_end + self.pad_to_multiplier - 1) // self.pad_to_multiplier * self.pad_to_multiplier - l_end # 0
            if self.customized_flash_attn:
                Infinity_visible_kvlen = self.Infinity_visible_kvlen[:l_end]
                Infinity_invisible_qlen = self.Infinity_invisible_qlen[:l_end]
                attn_bias_or_two_vector = (Infinity_visible_kvlen, Infinity_invisible_qlen)
                # todo: solve need_to_pad here
            elif self.use_flex_attn:
                if need_to_pad:
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                assert x_BLC.shape[-1] % 128 == 0, 'x_BLC.shape[-1] % 128 != 0'
                attn_bias_or_two_vector = None
            else:
                d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2],), i) for i, pn in enumerate(scale_schedule)]).view(1, l_end, 1)
                l_end_nviews = l_end * N_views
                d = d.repeat_interleave(N_views).view(1, l_end_nviews, 1)
                
                dT = d.transpose(1, 2)    # dT: 11L
                attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, l_end_nviews, l_end_nviews)
                attn_bias = attn_bias_for_masking[:, :, :l_end_nviews, :l_end_nviews].contiguous()   # attn_bias: 11LL
                if need_to_pad:
                    attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                    attn_bias[0, 0, l_end_nviews:, 0] = 0
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                attn_bias_or_two_vector = attn_bias.type_as(x_BLC).to(x_BLC.device)
        #default: self.use_flex_attn=False
        if self.use_flex_attn:
            attn_fn = self.attn_fn_compile_dict[tuple(scale_schedule)]
        else:
            attn_fn = None

        # [2. block loop]
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training
        # by default != 1
        if self.num_block_chunks == 1:
            for i, b in enumerate(self.blocks):
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad, N_views)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad, N_views)
                if checkpointing_full_block:
                    x_BLC = torch.utils.checkpoint.checkpoint(b, x_BLC, cond_BD_or_gss, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, self.rope2d_freqs_grid, use_reentrant=False)
                else:
                    x_BLC = b(x=x_BLC, 
                              cond_BD=cond_BD_or_gss,
                              ca_kv=ca_kv,
                              attn_bias_or_two_vector=attn_bias_or_two_vector,
                              poses=poses,
                              intrs=intrs,
                              poses_src=poses_src,
                              intrs_src=intrs_src,
                              input_size=input_size,
                              attn_fn=attn_fn,
                              scale_schedule=scale_schedule,
                              rope2d_freqs_grid=self.rope2d_freqs_grid)
        else:
            for i, chunk in enumerate(self.block_chunks): # this path
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad, N_views)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad, N_views)

                # [B, N_views*n_tokens_per_view, 2048]
                x_BLC = chunk(x=x_BLC, 
                              cond_BD=cond_BD_or_gss, 
                              ca_kv=ca_kv, #reference vae feats
                              attn_bias_or_two_vector=attn_bias_or_two_vector,
                              poses=poses,
                              intrs=intrs,
                              poses_src=poses_src,
                              intrs_src=intrs_src,
                              input_size=input_size,
                              attn_fn=attn_fn,
                              scale_schedule=scale_schedule, 
                              checkpointing_full_block=checkpointing_full_block, 
                              rope2d_freqs_grid=self.rope2d_freqs_grid)


        # [3. unpad the seqlen dim, and then get logits]
        self.rope2d_freqs_grid.clear()
        # logits.shape: torch.Size([1, 1042, 64])
        logits = self.get_logits(x_BLC[:, :l_end_nviews], cond_BD)
        return logits    # return logits BLV, V is vocab_size

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae_features_src, # [B, N_views_src, 32, H//16, W//16] or [B, N_views_src, 512, H//16, H//16]
        # x_BLC_wo_prefix_tgt,
        vae=None,
        scale_schedule=None,
        images_src=None,
        label_B_or_BLT=None,
        rays=None, # tensor. rays[i]: [B,N_views_tgt, N_tokens_scale, 6]
        poses=None,
        intrs=None,
        rays_src=None,
        poses_src=None,
        intrs_src=None,
        N_views=1,
        input_size=None,
        B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
        g_seed=None, cfg_list=[], tau_list=[], cfg_sc=3, top_k=0, top_p=0.0,
        returns_vemb=0, ratio_Bl1=None, gumbel=0, norm_cfg=False,
        cfg_exp_k: float=0.0, cfg_insertion_layer=[-5],
        vae_type=0, softmax_merge_topk=-1, ret_img=False,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        inference_mode=False,
        save_img_path=None,
        sampling_per_bits=1,
        **kwargs,
    ):   # returns List[idx_Bl]
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        # assert len(cfg_list) >= len(scale_schedule)
        # assert len(tau_list) >= len(scale_schedule)

        # scale_schedule is used by infinity, vae_scale_schedule is used by vae if there exists a spatial patchify, 
        # we need to convert scale_schedule to vae_scale_schedule by multiply 2 to h and w
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule

        bs = B
        N_views_src = poses_src.shape[1]
        token_schedule = [scale_i[0]*scale_i[1]*scale_i[2] for scale_i in scale_schedule]

        # [B,N_views_src,D,h,w] -> # [B,N_views_src,h,w,D]
        vae_features_src = vae_features_src.permute(0,1,3,4,2)

        # self.cfg_uncond [16*16,512]
        if any(np.array(cfg_list) != 1):
            bs = 2*B
            vae_features_src_un = vae_features_src.clone()
            
            for i in range(vae_features_src.shape[0]):
                for j in range(vae_features_src.shape[1]):
                    vae_features_src_un[i,j] = self.cfg_uncond

            vae_features_src = torch.cat((vae_features_src, vae_features_src_un), dim=0)
            
            poses_src_un = poses_src.clone()
            intrs_src_un = intrs_src.clone()
            poses_src = torch.cat((poses_src, poses_src_un), dim=0)
            intrs_src = torch.cat((intrs_src, intrs_src_un), dim=0)

            poses_un = poses.clone()
            intrs_un = intrs.clone()
            poses = torch.cat((poses, poses_un), dim=0)
            intrs = torch.cat((intrs, intrs_un), dim=0)

            if not self.use_prope:
                rays_un = rays.clone()
                rays = torch.cat([rays, rays_un], dim=0)

                rays_src_un = rays_src.clone()
                rays_src = torch.cat([rays_src, rays_src_un], dim=0)

        vae_features_src = self.img_norm(vae_features_src)
        vae_height, vae_width = vae_features_src.shape[2], vae_features_src.shape[3]

        vae_features = []
        if not self.use_prope:
            # add src rays
            start = 0
            for idx, n_tokens_i in enumerate(token_schedule):
                end = start + n_tokens_i
                if idx != len(token_schedule)-1:
                    start=end
                    continue

                for v in range(N_views_src):
                    # rays_src.shape: torch.Size([2, 2, 521, 6])
                    # vae_features_src.shape: torch.Size([2, 2, 16, 16, 32])

                    # [B,H,W,32] (+) [B,H,W,6]
                    vae_features_src_posed_i = torch.cat((vae_features_src[:,v],
                                                        rays_src[:,v,start:end].reshape(bs,vae_height,vae_width,6)), 
                                                        dim=-1).reshape(bs,vae_height*vae_width,-1)

                    # print(f"vae_features_src_posed_i.shape: {vae_features_src_posed_i.shape}")
            
                    vae_features.append(vae_features_src_posed_i)
                start=end

            vae_features_src = torch.cat(vae_features, dim=1)


        vae_features_src = vae_features_src.reshape(bs,-1,self.D_cond+self.D_pose)

        # [B,N_views_src,D,H//16,W//16] -> [B,N_views_src,H//16,W//16,D] -> [B, N_views_src*H//16*W//16, D]
        
        # kv_embed.shape: torch.Size([1, 512, 2048]): [B, N_views_src*N_tokens_src, D]
        kv_embed = self.img_proj_for_ca(vae_features_src)

        # kv_embed.shape: [B*N_views_src*N_tokens_src, D]
        kv_compact = kv_embed.reshape(-1, self.D).contiguous()
        # print(f"[cfg] kv_compact.shape: {kv_compact.shape}")
        L_src = vae_features_src.shape[1]
        cu_seqlens_k = torch.arange(0, (bs+1)*L_src, step=L_src, dtype=torch.int32, device=kv_compact.device)
        max_seqlen_k = L_src
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k

        sos = cond_BD = self.img_proj_for_sos(vae_features_src)

        with torch.amp.autocast('cuda', enabled=False):
            cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()

        last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)


        # create 'sos' per generated view
        last_stage_n_views = []
        for i in range(N_views):
            last_stage_n_views.append(last_stage)
        # last_stage: [1, N_views(2), 2048]
        last_stage = torch.cat(last_stage_n_views, dim=1)

        accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
        idx_Bl_list, idx_Bld_list = [], []

        block_groups = tuple(self._inference_block_groups())
        for blocks in block_groups:
            for block in blocks:
                (block.sa if isinstance(block, CrossAttnBlock) else block.attn).kv_caching(True)
        
        abs_cfg_insertion_layers = []
        add_cfg_on_logits, add_cfg_on_probs = False, False
        leng = len(self.unregistered_blocks)
        for item in cfg_insertion_layer:
            if item == 0:
                add_cfg_on_logits = True
            elif item == 1:
                add_cfg_on_probs = True
            elif item < 0: # determine to add cfg at item-th layer's output
                assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={self.num_block_chunks}'
                abs_cfg_insertion_layers.append(leng+item)
            else:
                raise ValueError(f'cfg_insertion_layer: {item} is not valid')

        num_stages_minus_1 = len(scale_schedule)-1
        summed_codes = 0
        ray_start = 0
        codes_list = []
        # Clear cache before starting generation
        self.rope2d_freqs_grid.clear()
        for si, pn in enumerate(scale_schedule):   # si: i-th segment
            cfg = cfg_list[si]
            if si >= trunk_scale:
                break
            cur_L += np.array(pn).prod()

            need_to_pad = 0
            attn_fn = None
            if self.use_flex_attn:
                # need_to_pad = (self.pad_to_multiplier - cur_L % self.pad_to_multiplier) % self.pad_to_multiplier
                # if need_to_pad:
                #     last_stage = F.pad(last_stage, (0, 0, 0, need_to_pad))
                attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule[:(si+1)]), None)

            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            layer_idx = 0
            for block_idx, blocks in enumerate(block_groups):
                # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
                if self.add_lvl_embeding_only_first_block and block_idx == 0:
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad, N_views=N_views)
                if not self.add_lvl_embeding_only_first_block: 
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad, N_views=N_views)
                
                for m in blocks:
                    last_stage = m(x=last_stage, 
                                   cond_BD=cond_BD_or_gss, 
                                   ca_kv=ca_kv, 
                                   attn_bias_or_two_vector=None, 
                                   attn_fn=attn_fn, 
                                   scale_schedule=scale_schedule,
                                   rope2d_freqs_grid=self.rope2d_freqs_grid, 
                                   scale_ind=si,
                                   poses=poses,
                                   intrs=intrs,
                                   poses_src=poses_src,
                                   intrs_src=intrs_src,
                                   input_size=input_size)

                    layer_idx += 1

            if (cfg != 1) and add_cfg_on_logits:
                logits_BlV = self.get_logits(last_stage, cond_BD).mul(1/tau_list[si])
                logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
            else:
                logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])
            
            # print(F"logits_BlV.shape: {logits_BlV.shape}")
            # logits_BlV = self.get_logits(last_stage, cond_BD)

            # combine batches and views(sampling of bits is parallel anyway)
            # [B*N_views, N_tokens, 2048]
            logits_BlV = logits_BlV.reshape(B*N_views, pn[0]*pn[1]*pn[2],*logits_BlV.shape[2:])

            # idx_Bld: [B*N_views, N_tokens_scale, 32] 
            if self.use_bit_label:
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
            else:
                idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
            
            # by default vae_type!=0
            if vae_type != 0:
                assert returns_vemb
                #default: gt_leak=-1
                if si < gt_leak: 
                    idx_Bld = gt_ls_Bl[si]
                else:
                    assert pn[0] == 1
                    idx_Bld = idx_Bld.reshape(B*N_views, pn[1], pn[2], -1) # shape: [B, h, w, d] or [B, h, w, 4d]
                    
                    #default: self.apply_spatial_patchify=False
                    if self.apply_spatial_patchify: # unpatchify operation
                        idx_Bld = idx_Bld.permute(0,3,1,2) # [B, 4d, h, w]
                        idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, d, 2h, 2w]
                        idx_Bld = idx_Bld.permute(0,2,3,1) # [B, 2h, 2w, d]
                    idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]
                idx_Bld_list.append(idx_Bld)
                codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                if si != num_stages_minus_1:
                    summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                    last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                    # [B, 32, 1, n_tokens_height, n_tokens_width]
                    last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
                    if self.apply_spatial_patchify: # patchify operation(default False)
                        last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, 4d, h, w]
                    last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
                    last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
                else:
                    summed_codes += codes

                codes_list.append(summed_codes.clone())
            else:
                if si < gt_leak:
                    idx_Bl = gt_ls_Bl[si]
                h_BChw = self.quant_only_used_in_inference[0].embedding(idx_Bl).float()   # BlC

                # h_BChw = h_BChw.float().transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1])
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1], scale_schedule[si][2])
                ret.append(h_BChw if returns_vemb != 0 else idx_Bl)
                idx_Bl_list.append(idx_Bl)
                if si != num_stages_minus_1:
                    accu_BChw, last_stage = self.quant_only_used_in_inference[0].one_step_fuse(si, num_stages_minus_1+1, accu_BChw, h_BChw, scale_schedule)
            
            # print(f"last_stage.shape: {last_stage.shape}")
            # exit()
            if si != num_stages_minus_1:

                last_stage = last_stage.reshape(B,N_views,*last_stage.shape[1:])
                last_stage = self.norm0_ve(last_stage)
                last_stage = last_stage.repeat(bs//B, 1, 1, 1)

                token_count_i = torch.prod(torch.tensor(scale_schedule[si+1]))
                ray_end = ray_start + token_count_i
                last_stage_posed = []
                for v in range(N_views):
                    if not self.use_prope:
                        last_stage_posed_v = torch.cat(( last_stage[:,v], rays[:,v,ray_start:ray_end]), dim=-1)                
                        last_stage_posed_v = self.word_embed(last_stage_posed_v)
                        
                    else:
                        last_stage_posed_v = self.word_embed(last_stage[:,v])
                        # last_stage_posed.append(self.word_embed(last_stage))
                
                    last_stage_posed.append(last_stage_posed_v)
                ray_start = ray_end
                last_stage = torch.cat(last_stage_posed, dim=1).reshape(bs,N_views*token_count_i,-1)


        for blocks in block_groups:
            for block in blocks:
                (block.sa if isinstance(block, CrossAttnBlock) else block.attn).kv_caching(False)

        self.rope2d_freqs_grid.clear()

        if not ret_img:
            return ret, idx_Bl_list, []
        
        for i in range(len(codes_list)):
            codes_list[i] = vae.decode(codes_list[i].squeeze(-3))
            codes_list[i] = (codes_list[i] + 1) / 2
            codes_list[i] = codes_list[i].permute(0, 2, 3, 1).mul_(255).to(torch.uint8) #.flip(dims=(3,))

        if vae_type != 0:
            img = vae.decode(summed_codes.squeeze(-3)) # [B*N_views, 32, 16, 16]
        else:
            img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule, same_shape=True, last_one=True)

        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
        return ret, idx_Bl_list, img, codes_list


    @for_visualize
    def vis_key_params(self, ep):
        return
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=False, assign=False):
        # for k in state_dict:
        #     if 'cfg_uncond' in k:
        #         old, new = state_dict[k], self.cfg_uncond.data
        #         min_tlen = min(old.shape[0], new.shape[0])
        #         if min_tlen == old.shape[0]:
        #             state_dict[k] = torch.cat((old.to(device=new.device, dtype=new.dtype), new[min_tlen:]))
        #         else:
        #             state_dict[k] = old[:min_tlen]
        
        if 'cfg_uncond' in state_dict:
            # Check if shapes match
            if state_dict['cfg_uncond'].shape != self.cfg_uncond.shape:
                print(f"Shape mismatch for cfg_uncond: checkpoint {state_dict['cfg_uncond'].shape} "
                      f"vs model {self.cfg_uncond.shape}. Dropping from checkpoint.")
                # Delete it so strict=False ignores it (instead of crashing on mismatch)
                del state_dict['cfg_uncond']

        for buf_name in ('lvl_1L', 'attn_bias_for_masking', 'Infinity_visible_kvlen', 'Infinity_invisible_qlen'):
            state_dict.pop(buf_name, None)
            if hasattr(self, buf_name):
                state_dict[buf_name] = getattr(self, buf_name)
        
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
    
    def special_init(
        self,
        aln_init: float,
        aln_gamma_init: float,
        scale_head: float,
        scale_proj: int,
    ):
        # init head's norm
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(aln_init)    # there's no gamma for head
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        # init head's proj
        if scale_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(scale_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(scale_head)
                self.head[-1].bias.data.zero_()
        
        depth = len(self.unregistered_blocks)
        for block_idx, sab in enumerate(self.unregistered_blocks):
            sab: CrossAttnBlock
            # init proj
            scale = 1 / math.sqrt(2*depth if scale_proj == 1 else 2*(1 + block_idx))
            if scale_proj == 1:
                sab.sa.proj.weight.data.mul_(scale)
                sab.ca.proj.weight.data.mul_(scale)
                sab.ffn.fc2.weight.data.mul_(scale)
            # if sab.using_swiglu:
            #     nn.init.ones_(sab.ffn.fcg.bias)
            #     nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            
            # init ada_lin
            if hasattr(sab, 'ada_lin'):
                lin = sab.ada_lin[-1]
                lin.weight.data[:2*self.C].mul_(aln_gamma_init)     # init gamma
                lin.weight.data[2*self.C:].mul_(aln_init)           # init scale and shift
                if hasattr(lin, 'bias') and lin.bias is not None:
                    lin.bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, :2, :].mul_(aln_gamma_init)  # init gamma
                sab.ada_gss.data[:, :, 2:, :].mul_(aln_init)        # init scale and shift
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate}'
    
    def get_layer_id_and_scale_exp(self, para_name: str):
        raise NotImplementedError


def sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    # print(f"top_k: {top_k} | top_p: {top_p}")
    B, l, V = logits_BlV.shape # [1,,2]
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

TIMM_KEYS = {'img_size', 'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay', 'global_pool'}

@register_model
def infinity3d_src2sos_1b(depth=16, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, **kwargs): 
    return Infinity3DSrc2Sos(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
