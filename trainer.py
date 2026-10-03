import time
from pprint import pformat
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import torch.distributed as tdist

import infinity.utils.dist as dist
from infinity.models import Infinity3DSrc2Sos
from infinity.models.ema import update_ema
from infinity.models.bitwise_self_correction import BitwiseSelfCorrection, sample_schedule_severity
from infinity.utils import arg_util, misc
from infinity.utils.amp_opt import AmpOptimizer
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from infinity.utils.rays import plucker_rays_paired

# from prope.torch import _invert_SE3

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
fulloptstate_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)


def _split_flat_bits_to_scale_tensors(
    bits: torch.Tensor,
    gt_ms_idx_Bl: List[torch.Tensor],
    vae_scale_schedule: List[Tuple[int, int, int]],
    training_scales: int,
    n_views: int,
) -> List[torch.Tensor]:
    B, seq_len, code_dim = bits.shape
    scale_tensors = []
    ptr = 0
    for scale_idx, idx_Bl in enumerate(gt_ms_idx_Bl[:training_scales]):
        if scale_idx >= len(vae_scale_schedule):
            break
        pt, ph, pw = vae_scale_schedule[scale_idx]
        _, n_tokens_i, gt_code_dim = idx_Bl.shape
        if n_tokens_i != pt * ph * pw:
            raise ValueError(
                f"VAE token count mismatch at scale {scale_idx}: "
                f"got {n_tokens_i}, expected {pt * ph * pw} from {vae_scale_schedule[scale_idx]}"
            )
        if code_dim != gt_code_dim:
            raise ValueError(f"VAE code dim mismatch: got {code_dim}, expected {gt_code_dim}")

        seg_len = n_views * n_tokens_i
        if ptr + seg_len > seq_len:
            break
        scale_bits = bits[:, ptr:ptr + seg_len]
        scale_bits = scale_bits.reshape(B * n_views, pt, ph, pw, code_dim).contiguous()
        scale_tensors.append(scale_bits)
        ptr += seg_len
    return scale_tensors


def _reshape_ms_indices_for_vae_decode(
    ms_idx_Bl: List[torch.Tensor],
    vae_scale_schedule: List[Tuple[int, int, int]],
    training_scales: int,
) -> List[torch.Tensor]:
    """Reshape flattened per-scale VAE bits to the decoder's spatial layout."""
    decoded_indices = []
    for scale_idx, idx_Bl in enumerate(ms_idx_Bl[:training_scales]):
        if scale_idx >= len(vae_scale_schedule):
            break
        pt, ph, pw = vae_scale_schedule[scale_idx]
        batch_views, n_tokens, code_dim = idx_Bl.shape
        expected_tokens = pt * ph * pw
        if n_tokens != expected_tokens:
            raise ValueError(
                f"VAE token count mismatch at scale {scale_idx}: got {n_tokens}, "
                f"expected {expected_tokens} from {vae_scale_schedule[scale_idx]}"
            )
        decoded_indices.append(
            idx_Bl.reshape(batch_views, pt, ph, pw, code_dim).contiguous()
        )
    if not decoded_indices:
        raise ValueError("No VAE scales available for target decode.")
    return decoded_indices


def soft_decode_bit_logits(
    vae,
    logits: torch.Tensor,
    gt_ms_idx_Bl: List[torch.Tensor],
    vae_scale_schedule: List[Tuple[int, int, int]],
    training_scales: int,
    n_views: int,
    decode_mode: str = "hard",
    clamp_output: bool = False,
) -> torch.Tensor:
    """Decode bit logits through a frozen multiscale VAE with STE gradients."""
    B, seq_len, _ = logits.shape

    logits_bits = logits.reshape(B, seq_len, -1, 2)
    probs = logits_bits.softmax(dim=-1)[..., 1]

    if decode_mode == "soft":
        # Current behavior: continuous p(bit=1)
        bits = probs

    elif decode_mode == "hard":
        # Hard argmax in forward, soft STE gradient in backward.
        hard_bits = logits_bits.argmax(dim=-1).to(probs.dtype)
        bits = probs + (hard_bits - probs).detach()

    else:
        raise ValueError(
            f"Unknown decode_mode={decode_mode!r}. Expected 'soft' or 'hard'."
        )

    scale_bits = _split_flat_bits_to_scale_tensors(
        bits,
        gt_ms_idx_Bl,
        vae_scale_schedule,
        training_scales,
        n_views,
    )

    if len(scale_bits) == 0:
        raise ValueError("No VAE scales available for decode.")

    target_size = vae_scale_schedule[len(scale_bits) - 1]
    summed_codes = 0

    for bits_i in scale_bits:
        codes = vae.quantizer.lfq.indices_to_codes(
            bits_i,
            label_type="bit_label",
        )

        summed_codes = summed_codes + F.interpolate(
            codes,
            size=target_size,
            mode=vae.quantizer.z_interplote_up,
        )

    assert summed_codes.shape[-3] == 1
    z = summed_codes.squeeze(-3)

    chunk_size = 4
    decoded_chunks = []

    for i in range(0, z.size(0), chunk_size):
        z_chunk = z[i:i+chunk_size]

        if z_chunk.requires_grad:
            out_chunk = checkpoint.checkpoint(
                vae.decoder,
                z_chunk,
                use_reentrant=False,
            )
        else:
            out_chunk = vae.decoder(z_chunk)

        decoded_chunks.append(out_chunk)

    decoded = torch.cat(decoded_chunks, dim=0)
    if clamp_output:
        decoded = torch.clamp(decoded, min=-1, max=1)

    return decoded


def soft_decoded_rgb_recon_loss(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    valid_mask: torch.Tensor,
    n_views: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean per-pixel RGB MSE against frozen VAE reconstructions.

    Both image tensors are flattened as [B * N_views, 3, H, W]. Scene-level
    validity is applied after averaging all target views of a scene.
    """
    if pred_rgb.shape != target_rgb.shape:
        raise ValueError(
            f"RGB reconstruction shape mismatch: predicted {tuple(pred_rgb.shape)}, "
            f"target {tuple(target_rgb.shape)}"
        )
    if pred_rgb.ndim != 4 or pred_rgb.shape[1] != 3:
        raise ValueError(f"Expected RGB tensors [B*N,3,H,W], got {tuple(pred_rgb.shape)}")
    if pred_rgb.shape[0] % n_views != 0:
        raise ValueError(
            f"Decoded RGB batch {pred_rgb.shape[0]} is not divisible by {n_views} views"
        )

    B = pred_rgb.shape[0] // n_views
    squared_error = (pred_rgb.float() - target_rgb.to(device=pred_rgb.device, dtype=torch.float32)).square()
    per_view = squared_error.mean(dim=(1, 2, 3))
    per_sample = per_view.reshape(B, n_views).mean(dim=1)
    valid_float = valid_mask.to(device=pred_rgb.device, dtype=pred_rgb.dtype)
    if valid_float.numel() != B:
        raise ValueError(f"Expected {B} validity entries, got {valid_float.numel()}")
    loss_sum = (per_sample * valid_float).sum()
    valid_count = valid_float.sum()
    loss = loss_sum / valid_count.clamp_min(1.0)
    stats = torch.stack([loss_sum.detach(), valid_count.detach()])
    return loss, stats


class InfinityTrainer(object):
    def __init__(
        self, is_visualizer: bool, device, raw_scale_schedule: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local, gpt_wo_ddp: Infinity3DSrc2Sos, gpt: DDP, ema_ratio: float, max_it: int,
        gpt_opt: AmpOptimizer, label_smooth: float, z_loss_ratio: float, eq_loss: int, xen: bool,
        dbg_unused=False,zero=0, vae_type=True, reweight_loss_by_scale=False,
        gpt_wo_ddp_ema=None, gpt_ema=None, use_fsdp_model_ema=False, other_args=None,
    ):
        super(InfinityTrainer, self).__init__()
        self.dbg_unused = dbg_unused
        
        self.zero = zero
        self.vae_type = vae_type
        
        self.gpt: Union[DDP, FSDP, nn.Module]
        self.gpt, self.vae_local, self.quantize_local = gpt, vae_local, vae_local.quantize

        self.gpt_opt: AmpOptimizer = gpt_opt
        self.gpt_wo_ddp: Union[Infinity3DSrc2Sos, torch._dynamo.eval_frame.OptimizedModule] = gpt_wo_ddp  # after torch.compile
        self.gpt_wo_ddp_ema = gpt_wo_ddp_ema
        self.gpt_ema = gpt_ema
        self.bitwise_self_correction = BitwiseSelfCorrection(self.vae_local, other_args)

        rgb_schedule = self.bitwise_self_correction.noise_apply_strength_schedule
        if rgb_schedule is not None:
            print(
                "RGB corruption maxima by context scale: "
                f"{list(rgb_schedule)}; expected means: {[value / 2 for value in rgb_schedule]}"
            )

        self.use_fsdp_model_ema = use_fsdp_model_ema
        self.batch_size, self.seq_len = 0, 0
        self.seq_len_each = []
        self.reweight_loss_by_scale = reweight_loss_by_scale
        print(f'self.reweight_loss_by_scale: {self.reweight_loss_by_scale}')
        
        self.using_ema = ema_ratio != 0 and self.zero == 0
        self.ema_ratio = abs(ema_ratio)
        self.ema_cpu = ema_ratio < 0
        self.is_visualizer = is_visualizer
        
        gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        # del gpt_uncompiled.rng
        gpt_uncompiled.rng = torch.Generator(device=device)
        del gpt_uncompiled
        
        self.cached_state_not_ema = None
        self.last_train_loss = None
        if self.using_ema:
            self.pi_para_copy_for_parallel_ema = []
            all_tot = tot = 0
            for pi, para in enumerate(self.gpt_opt.paras):          # only learnable parameters need ema update
                if pi % dist.get_world_size() == dist.get_rank():   # model-parallel-style split
                    p_ema = para.data.cpu() if self.ema_cpu else para.data.clone()
                    self.pi_para_copy_for_parallel_ema.append((pi, p_ema))
                    tot += p_ema.numel()
                all_tot += para.numel()
            t = torch.zeros(dist.get_world_size())
            t[dist.get_rank()] = float(tot)
            dist.allreduce(t)
            t = [round(x) for x in t.tolist()]
            print(f'[ema tot #para] min={min(t)/1e6:.2f}, max={max(t)/1e6:.2f}, sum={sum(t)/1e6:.2f}, error={sum(t)-all_tot}')
            # lvl_1L, attn_bias_for_masking, zero_k_bias are never changed
            # check we only have these buffers so that we can skip buffer copy in ema update (only perform param update)
            assert all(any(s in name for s in ('lvl_1L', 'attn_bias_for_masking', 'zero_k_bias')) for name, _ in self.gpt_wo_ddp.named_buffers())
        else:
            self.pi_para_copy_for_parallel_ema = None
        
        self.label_smooth = label_smooth
        self.z_loss_ratio = z_loss_ratio
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')
        self.eq_loss = eq_loss
        
        if self.eq_loss:
            self.loss_eq_weight = torch.empty(1, self.raw_L, device=device)
            cur = 0
            for raw_pn in raw_scale_schedule:
                l = raw_pn*raw_pn
                self.loss_eq_weight[0, cur:cur+l] = 1./((raw_pn*raw_pn) if self.eq_loss == 2 else raw_pn)
                cur += l
            self.loss_eq_weight /= self.loss_eq_weight.sum()
        else:
            self.loss_eq_weight = 1.
        
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        self.generator = np.random.default_rng(0)

    def train_step(
        self, 
        ep: int, 
        it: int, 
        g_it: int, 
        stepping: bool, 
        clip_decay_ratio: float, 
        metric_lg: misc.MetricLogger, 
        logging_params: bool,
        text_cond_tuple: Union[ITen, FTen],
        images_src, # [B,N_views_src,3,height,width]
        images_tgt, # [B,N_views_tgt,3,height,width]
        poses_src,
        poses_tgt,
        intrs_src,
        intrs_tgt,
        args: arg_util.Args,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        

        B, N_views_src, _, H, W = images_src.shape
        B, N_views_tgt, _, H, W = images_tgt.shape
        T = 1
        V = self.vae_local.vocab_size
        device = images_src.device
        # Corrupted RGB samples are filtered before collation.
        valid_mask = torch.ones(B, device=device, dtype=torch.float32)

        images_src = images_src.reshape(B*N_views_src, 3, H, W)
        images_tgt = images_tgt.reshape(B*N_views_tgt, 3, H, W)
        
        h_div_w = H / W
        h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
        scale_schedule = [ (min(t, T//4+1), h, w) for (t,h, w) in scale_schedule]
        
        # [forward]
        with self.gpt_opt.amp_ctx:
            with torch.amp.autocast('cuda', enabled=False):
                with torch.no_grad():
                    if args.apply_spatial_patchify:
                        vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
                    else:
                        vae_scale_schedule = scale_schedule
                    # raw_features_source, _, _ = self.vae_local.encode_for_raw_features(inp_B3HW, scale_schedule=vae_scale_schedule)
                    # raw_features_src, _, _ = self.vae_local.encode_for_raw_features(images_src, scale_schedule=vae_scale_schedule)
                    raw_features_src, hs_src, hs_mid_src = self.vae_local.encode_for_raw_features(images_src, scale_schedule=scale_schedule)

                    raw_features_tgt, _, _ = self.vae_local.encode_for_raw_features(images_tgt, scale_schedule=vae_scale_schedule)

            
            # create rays for tgt pose conditioning
            rays = []
            with torch.no_grad():            
                # intrs_per_scale = intrs.unsqueeze(0).repeat(len(vae_scale_schedule),1,1,1)

                for si, (pt, ph, pw) in enumerate(vae_scale_schedule):
                    if si==0:
                        continue

                    rays_i = plucker_rays_paired(poses_src[:,:1], poses_tgt, intrs_tgt.clone(), target_size=[ph,pw], real_size=[ph,pw]) 
                    # real_size=[H,W])
                    rays.append(rays_i.reshape(B,N_views_tgt, ph*pw,6))
            rays = torch.cat(rays, 2)

            rays_src = []
            with torch.no_grad():            
                # intrs_per_scale = intrs.unsqueeze(0).repeat(len(vae_scale_schedule),1,1,1)

                for si, (pt, ph, pw) in enumerate(vae_scale_schedule):
                    # if si==0:
                    #     continue

                    rays_i = plucker_rays_paired(poses_src[:,:1], poses_src, intrs_src.clone(), target_size=[ph,pw], real_size=[ph,pw]) 
                    # real_size=[H,W])
                    rays_src.append(rays_i.reshape(B,N_views_src, ph*pw,6))
                    # print(f"rays_src[-1].shape: {rays_src[-1].shape}")

            rays_src = torch.cat(rays_src, 2)

            # gt_ms_idx_Bl : [B*N_views_tgt, n_token_i, 32]
            scheduled_corruption_severity = None
            if self.bitwise_self_correction.noise_apply_strength_schedule is not None:
                scheduled_corruption_severity = sample_schedule_severity(
                    len(vae_scale_schedule) - 1
                )

            x_BLC_wo_prefix, gt_ms_idx_Bl = self.bitwise_self_correction.flip_requant(vae_scale_schedule, 
                                                                                      images_tgt, 
                                                                                      raw_features_tgt, 
                                                                                      device,
                                                                                      corruption_severity=scheduled_corruption_severity)


            multiscale_feat_src, _ = self.bitwise_self_correction.flip_requant_nonoise(vae_scale_schedule, 
                                                                                      images_src, 
                                                                                      raw_features_src, 
                                                                                      device)

            # truncate scales
            training_scales = min(args.always_training_scales, len(scale_schedule), len(vae_scale_schedule))
            training_seq_len = np.array(scale_schedule)[:training_scales].prod(axis=1).sum()

            x_BLC_wo_prefix = x_BLC_wo_prefix[:, :(training_seq_len-np.array(scale_schedule[0]).prod()), :]


            multiscale_feat_src = multiscale_feat_src.reshape(B,N_views_src,*multiscale_feat_src.shape[1:])

            images_src = images_src.reshape(B,N_views_src, 3, H, W)

            self.gpt_wo_ddp.forward
            logits_BLV = self.gpt(text_cond_tuple,
                                raw_features_src,
                                x_BLC_wo_prefix,
                                scale_schedule=scale_schedule[:training_scales],
                                images_src=images_src,
                                multiscale_feat_src=multiscale_feat_src,
                                rays=rays,
                                poses=poses_tgt,
                                intrs=intrs_tgt,
                                rays_src=rays_src,
                                poses_src=poses_src,
                                intrs_src=intrs_src,
                                N_views=N_views_tgt,
                                input_size=[H,W]) # [bs, 1*1+...+64*64, vocab_size or log2(vocab_size)*2]

            self.batch_size, self.seq_len = logits_BLV.shape[:2]

            self.seq_len_each = [idx_Bl.shape[1] for idx_Bl in gt_ms_idx_Bl]
            
            gt_BL = []
            for i in range(len(gt_ms_idx_Bl)):
                _, n_tokens_i,_ = gt_ms_idx_Bl[i].shape
                # save GT in scale-first order: [s0view0, s0view1, s1view0, s1view1, ...]
                gt_BL.append(gt_ms_idx_Bl[i].reshape(B, N_views_tgt*n_tokens_i, 32))

            gt_BL = torch.cat(gt_BL, dim=1).type(torch.long)

            if args.use_bit_label:
                tmp_bs, tmp_seq_len, tmp_channel = logits_BLV.shape
                loss = self.train_loss(logits_BLV.reshape(tmp_bs, tmp_seq_len, -1, 2).permute(0,3,1,2), gt_BL)
                if args.bitloss_type == 'mean':
                    loss = loss.mean(dim=-1)
                elif args.bitloss_type == 'sum':
                    loss = loss.sum(dim=-1)
                else:
                    raise NotImplementedError(f'{args.bitloss_type=}')
            else:
                loss = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1)


            if self.reweight_loss_by_scale:
                lw = []
                last_scale_area = np.sqrt(np.array(scale_schedule[-1]).prod())
                for (pt, ph, pw) in scale_schedule[:training_scales]:
                    this_scale_area = np.sqrt(pt * ph * pw)
                    lw.extend([last_scale_area / this_scale_area for _ in range(pt * ph * pw * N_views_tgt)])
                    # print(f"lw[-1].shape: {lw[-1].shape}")
                #lw: [1, 521]
                lw = torch.tensor(lw, device=loss.device)[None, ...]
                lw = lw / lw.sum()
            else:
                lw = 1. / self.seq_len
            loss = loss.mul(lw) #.sum(dim=-1).mean()

            rgb_recon_loss = None
            rgb_recon_stats = None
            rgb_recon_enabled = float(getattr(args, "rgb_recon_loss_weight", 0.0)) > 0.0
            if rgb_recon_enabled:
                if not args.use_bit_label:
                    raise ValueError("--rgb_recon_loss_weight requires --use_bit_label=1.")
                pred_rgb_soft = soft_decode_bit_logits(
                    vae=self.vae_local,
                    logits=logits_BLV,
                    gt_ms_idx_Bl=gt_ms_idx_Bl,
                    vae_scale_schedule=vae_scale_schedule,
                    training_scales=training_scales,
                    n_views=N_views_tgt,
                    decode_mode=args.rgb_recon_decode_mode,
                    clamp_output=True,
                )
                with torch.no_grad():
                    target_rgb_indices = _reshape_ms_indices_for_vae_decode(
                        gt_ms_idx_Bl,
                        vae_scale_schedule,
                        training_scales,
                    )
                    _, target_rgb = self.vae_local.decode_from_indices(
                        target_rgb_indices,
                        vae_scale_schedule[:training_scales],
                        label_type="bit_label",
                    )
                rgb_recon_loss, rgb_recon_stats = soft_decoded_rgb_recon_loss(
                    pred_rgb=pred_rgb_soft,
                    target_rgb=target_rgb,
                    valid_mask=valid_mask,
                    n_views=N_views_tgt,
                )

            # loss = loss.sum(dim=-1).mean()
            loss_per_sample = loss.sum(dim=-1) # Shape: [B]
            valid_mask = valid_mask.to(loss_per_sample.device)

            # RGB samples reaching the trainer have already passed filtering.
            num_skips = (valid_mask == 0).sum().item()
            
            loss = (loss_per_sample * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            if rgb_recon_loss is not None:
                loss = loss + float(args.rgb_recon_loss_weight) * rgb_recon_loss

        
        # [backward]
        self.last_train_loss = loss.detach()
        grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(
            ep=ep,
            it=it,
            g_it=g_it,
            stepping=stepping,
            logging_params=logging_params,
            loss=loss,
            clip_decay_ratio=clip_decay_ratio,
            stable=args.stable,
        )
        
        # update ema
        if args.use_fsdp_model_ema:
            update_ema(self.gpt_ema, self.gpt)

        # [zero_grad]
        if stepping:
            if self.using_ema: self.ema_update(g_it)
            if self.dbg_unused:
                ls = []
                for n, p in self.gpt_wo_ddp.named_parameters():
                    if p.grad is None:
                        ls.append(n)
                if len(ls):
                    raise AttributeError(f'unused param: {ls}')
        
            self.gpt_opt.optimizer.zero_grad(set_to_none=True)
        
        # [metric logging]
        if metric_lg.log_every_iter or it == 0 or it in metric_lg.log_iters:
            B, seq_len = logits_BLV.shape[:2]
            #NOTE by default true!
            if args.use_bit_label:
                res_loss = self.train_loss(logits_BLV.reshape(B, seq_len, -1, 2).permute(0,3,1,2), gt_BL).mean(dim=-1).mean(0)
                bitwise_acc = (logits_BLV.reshape(B, seq_len, -1, 2).argmax(dim=-1) == gt_BL).float() # shape: [bs, seq_len, codebook_dim]
            else:
                res_loss = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1).mean(0)
                pred_BL = logits_BLV.argmax(dim=-1)
                mask = self.vae_local.quantizer.lfq.mask
                pred_bits = ((pred_BL[..., None].int() & mask) != 0)
                gt_bits = ((gt_BL[..., None].int() & mask) != 0)
                bitwise_acc = (pred_bits == gt_bits).float() # shape: [bs, seq_len, codebook_dim]
            res_bit_acc = bitwise_acc.mean(-1).mean(0)

            res_token_acc = (bitwise_acc.sum(-1) == self.vae_local.codebook_dim).float().mean(0)

            
            loss_token_mean, acc_bit_mean, acc_token_mean = res_loss.mean().item(), res_bit_acc.mean().item() * 100., res_token_acc.mean().item() * 100.

            rgb_recon_mean = None
            rgb_recon_weighted = None
            if rgb_recon_enabled:
                if rgb_recon_stats is not None:
                    rgb_recon_log_stats = rgb_recon_stats.detach().clone().to(dtype=torch.float32)
                else:
                    rgb_recon_log_stats = torch.zeros(2, device=loss.device, dtype=torch.float32)
                tdist.all_reduce(rgb_recon_log_stats, op=tdist.ReduceOp.SUM)
                if rgb_recon_log_stats[1].item() > 0:
                    rgb_recon_mean = (rgb_recon_log_stats[0] / rgb_recon_log_stats[1].clamp_min(1.0)).item()
                    rgb_recon_weighted = float(args.rgb_recon_loss_weight) * rgb_recon_mean
            

                # res_token_acc = (bitwise_acc.sum(-1) == self.vae_local.codebook_dim).float().mean(0)
            
            
            ptr = 0
            L_list, acc_bit_list, acc_token_list = [], [], []
            for scale_ind in range(min(training_scales, len(scale_schedule))):
                start, end = ptr, ptr + np.array(scale_schedule[scale_ind]).prod()
                L_list.append(res_loss[start:end].mean().item())
                acc_bit_list.append(res_bit_acc[start:end].mean().item() * 100.)
                acc_token_list.append(res_token_acc[start:end].mean().item() * 100.)
                ptr = end
            
            metrics = torch.tensor(L_list + acc_bit_list + acc_token_list +[grad_norm_t.item(), loss_token_mean, acc_bit_mean, acc_token_mean], device=loss.device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            metrics = metrics.cpu().data.numpy() / dist.get_world_size()
            leng = len(L_list)
            L_list, acc_bit_list, acc_token_list, grad_norm_t, loss_token_mean, acc_bit_mean, acc_token_mean = metrics[:leng], \
                metrics[leng:2*leng], metrics[2*leng:3*leng], metrics[-4], metrics[-3], metrics[-2], metrics[-1]
            Lmean = loss_token_mean
            Ltail = L_list[-1]
            acc_mean = acc_bit_mean if args.use_bit_label else acc_token_mean
            acc_tail = acc_bit_list[-1] if args.use_bit_label else acc_token_list[-1]
            rgb_metrics = {}
            if rgb_recon_mean is not None:
                rgb_metrics.update(RGBRecon=rgb_recon_mean, RGBReconW=rgb_recon_weighted)
            metric_lg.update(
                Lm=Lmean,
                Lt=Ltail,
                Accm=acc_mean,
                Acct=acc_tail,
                tnm=grad_norm_t,
                **rgb_metrics,
            )    # todo: Accm, Acct
            
            # wandb_log_dict = {"Overall/L_mean": Lmean, 'Overall/Acc_bit_mean': acc_bit_mean, 'Overall/Acc_token_mean': acc_token_mean, 'Overall/grad_norm_t': grad_norm_t}
            # for si, (loss_si, acc_bit_si, acc_token_si) in enumerate(zip(L_list, acc_bit_list, acc_token_list)):
            #     wandb_log_dict[f'Detail/L_s{si+1:02d}'] = loss_si
            #     wandb_log_dict[f'Detail/Acc_bit_s{si+1:02d}'] = acc_bit_si
            #     wandb_log_dict[f'Detail/Acc_token_s{si+1:02d}'] = acc_token_si
            # wandb_utils.log(wandb_log_dict, step=g_it)
        
        return grad_norm_t, scale_log2_t, num_skips
    
    def __repr__(self):
        return (
            f'\n'
            f'[VGPTTr.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[VGPTTr.structure]: {super(InfinityTrainer, self).__repr__().replace(InfinityTrainer.__name__, "")}'
        )
    
    def ema_load(self):
        self.cached_state_not_ema = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            self.gpt_opt.paras[pi].data.copy_(p_ema)
        for pi, para in enumerate(self.gpt_opt.paras):
            dist.broadcast(para, src_rank=pi % dist.get_world_size())
    
    def ema_recover(self):
        self.gpt_wo_ddp.load_state_dict(self.cached_state_not_ema)
        del self.cached_state_not_ema
        self.cached_state_not_ema = None
    
    # p_ema = p_ema*0.9 + p*0.1 <==> p_ema.lerp_(p, 0.1)
    # p_ema.mul_(self.ema_ratio).add_(p.mul(self.ema_ratio_1))
    # @profile(precision=4, stream=open('ema_update.log', 'w+'))
    def ema_update(self, g_it): # todo: 将来再用离线ema
        # if self.using_ema and (g_it + 1) in self.ema_upd_it:
        stt = time.time()
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            p = self.gpt_opt.paras[pi]
            p_ema.data.mul_(self.ema_ratio).add_(p.data.to(p_ema.device), alpha=1-self.ema_ratio)
        # ii = self.ema_upd_it.index(g_it + 1)
        ii = g_it
        if ii < 3:
            print(f'[ema upd {self.ema_ratio}, cpu={self.ema_cpu}, @ g_it={g_it}] cost: {time.time()-stt:.2f}s')
    
    def get_config(self):
        return {
            'dynamic_resolution_h_w': dynamic_resolution_h_w,
            'label_smooth': self.label_smooth, 'eq_loss': self.eq_loss,
            'ema_ratio':    self.ema_ratio,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        m = self.vae_local
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        state = {'config': self.get_config(), 'vae_local': m.state_dict()}
        
        if self.zero:   # TODO: fixme
            state['gpt_fsdp'] = None
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                state['gpt_fsdp'] = self.gpt.state_dict()
                if self.use_fsdp_model_ema:
                    state['gpt_ema_fsdp'] = self.gpt_ema.state_dict()
                state['gpt_fsdp_opt'] = FSDP.optim_state_dict(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=self.gpt_opt.optimizer.state_dict())
            if self.gpt_opt.scaler is not None:
                state['gpt_opt_scaler'] = self.gpt_opt.scaler.state_dict()
        
        else:
            if self.using_ema:  # TODO: fixme
                self.ema_load()
                state['gpt_ema_for_vis'] = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
                self.ema_recover()
            
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        if self.zero:
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                self.gpt.load_state_dict(state['gpt_fsdp'])
                if self.use_fsdp_model_ema:
                    self.gpt_ema.load_state_dict(state['gpt_ema_fsdp'])
                one_group_opt_state = state['gpt_fsdp_opt']
                """
                AdamW state['gpt_fsdp_opt']:
                {
                    'state': { <para_name>: {'exp_avg': <unsharded_tensor>, 'exp_avg_sq': <unsharded_tensor>, 'step': <int>} },
                    'param_groups': [
                        {
                            'wd_sc': 1.0, 'lr_sc': 1.0, 'lr': xxx, 'betas': (0.9, 0.97), 'eps': 1e-08, 'weight_decay': 0.02,
                            'amsgrad': False, 'foreach': None, 'maximize': False, 'capturable': False, 'differentiable': False, 'fused': True,
                            'params': [<para_name> x m]
                        } x n
                    ]
                }
                one_group_opt_state['param_groups'] = self.gpt_opt.optimizer.state_dict()['param_groups']
                """
                optim_state_dict = FSDP.optim_state_dict_to_load(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=one_group_opt_state)
                self.gpt_opt.optimizer.load_state_dict(optim_state_dict)

            if self.gpt_opt.scaler is not None:
                try: self.gpt_opt.scaler.load_state_dict(state['gpt_opt_scaler'])
                except Exception as e: print(f'[fp16 load_state_dict err] {e}')
        else:
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                if skip_vae and 'vae' in k: continue
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    ret = m.load_state_dict(state[k], strict=strict)
                    if ret is not None:
                        missing, unexpected = ret
                        print(f'[VGPTTr.load_state_dict] {k} missing:  {missing}')
                        print(f'[VGPTTr.load_state_dict] {k} unexpected:  {unexpected}')
            
            if self.using_ema:
                if 'gpt_ema_for_vis' in state:
                    for pi, para in self.pi_para_copy_for_parallel_ema:
                        para.copy_(state['gpt_ema_for_vis'][self.gpt_opt.names[pi]])
                    print(f'[VGPTTr.load_state_dict] gpt_ema_for_vis: load succeed')
                else:
                    print(f'[VGPTTr.load_state_dict] gpt_ema_for_vis: key NOT FOUND in state!!')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VGPT.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)
