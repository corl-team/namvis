import gc
import json
import math
import os
import random
import sys
import time
import traceback
from collections import deque
from contextlib import nullcontext
from functools import partial
from typing import List
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from torch.profiler import record_function
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
import torch.distributed as tdist
import torchvision

import infinity.utils.dist as dist
from infinity.utils.save_and_load import CKPTSaver, auto_resume
from infinity.utils import arg_util, misc, wandb_utils

from infinity.utils.rays import plucker_rays_paired
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from inference.calc_metric import calc_2D_metrics
from PIL import Image

import trackio
import glob


import webdataset as wds

from infinity.dataset.webdataset_utils import process_multiview_rgb, collate_multiview_rgb


def build_everything_from_args(args: arg_util.Args, saver):
    if not os.path.isfile(args.vae_ckpt):
        raise FileNotFoundError(
            f"Visual tokenizer checkpoint not found: {args.vae_ckpt!r}. "
            "Download infinity_vae_d32reg.pth as described in README.md "
            "and set --vae_ckpt to its path."
        )

    # set seed
    args.set_initial_seed(benchmark=True)
    if args.seed is not None and not args.rand: # check the randomness
        misc.check_randomness(args)

    # build data
    iters_train, ld_train, ld_eval = build_dataloaders(args)


    # train_h_div_w_list = list(ld_train.dataset.h_div_w_template2generator.keys())
    # print(f"{train_h_div_w_list=}")
    
    train_h_div_w_list = ['1.000']
    args.train_h_div_w_list = train_h_div_w_list

    # load VAE
    print(f'Load vae form {args.vae_ckpt}')
    vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu', weights_only=True)

    # Build the RGB VAE and image-conditioned autoregressive model.
    text_tokenizer, text_encoder, vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim = build_model_optimizer(args, vae_ckpt)
    

    # IMPORTANT: import heavy package `InfinityTrainer` after the Dataloader object creation/iteration to avoid OOM
    from trainer import InfinityTrainer
    # build trainer
    trainer = InfinityTrainer(
        is_visualizer=dist.is_visualizer(), device=args.device, raw_scale_schedule=args.scale_schedule, resos=args.resos,
        vae_local=vae_local, gpt_wo_ddp=gpt_wo_ddp, gpt=gpt_ddp, ema_ratio=args.tema, max_it=iters_train * args.ep,
        gpt_opt=gpt_optim, label_smooth=args.ls, z_loss_ratio=args.lz, eq_loss=args.eq, xen=args.xen,
        dbg_unused=args.dbg, zero=args.zero, vae_type=args.vae_type,
        reweight_loss_by_scale=args.reweight_loss_by_scale, gpt_wo_ddp_ema=gpt_wo_ddp_ema, 
        gpt_ema=gpt_ddp_ema, use_fsdp_model_ema=args.use_fsdp_model_ema, other_args=args,

        
    )
    
    # auto resume from broken experiment
    auto_resume_info, start_ep, start_it, acc_str, eval_milestone, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    args.dump_log()
    if start_ep == args.ep:
        args.dump_log()
        print(f'[vgpt] AR finished ({acc_str}), skipping ...\n\n')
        return None
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
    
    start_it = start_it % iters_train
    print(f"{start_it=}, {iters_train=}")
    
    del vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim
    dist.barrier()
    return (
        text_tokenizer, text_encoder, trainer,
        start_ep, start_it, acc_str, eval_milestone, iters_train, ld_train, ld_eval
    )
def build_model_optimizer(args, vae_rgb_ckpt):
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from infinity.models.infinity3d_src2sos import MultipleLayers
    from infinity.models.init_param import init_weights
    from infinity.utils.amp_opt import AmpOptimizer
    from infinity.utils.lr_control import filter_params
    from infinity.utils.load import build_vae_gpt
    
    # disable builtin initialization for speed
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

    vae_local, gpt_wo_ddp, gpt_wo_ddp_ema = build_vae_gpt(
        args, vae_rgb_ckpt=vae_rgb_ckpt, skip_gpt=False,
        device=args.model_init_device,
    )
    del vae_rgb_ckpt

    # Lock the RGB VAE in eval mode and freeze its gradients.
    vae_local.eval()
    vae_local.requires_grad_(False)
    
    
    if args.tini < 0:
        args.tini = math.sqrt(1 / gpt_wo_ddp.C / 3)
    init_weights(gpt_wo_ddp, other_std=args.tini)
    gpt_wo_ddp.special_init(aln_init=args.aln, aln_gamma_init=args.alng, scale_head=args.hd0, scale_proj=args.diva)

    if args.rush_resume:
        print(f"{args.rush_resume=}")
        # Weights-only state dict: the released checkpoint or a *-statedict.pth export.
        state_dict = torch.load(args.rush_resume, map_location='cpu', weights_only=True)
        ema_state_dict = state_dict

        def drop_unfit_weights(state_dict):
            # 1. Remap the RGB word embedding weights
            if hasattr(gpt_wo_ddp, 'word_embed_rgb'):
                if 'word_embed.weight' in state_dict:
                    print("Remapping 'word_embed' to 'word_embed_rgb'")
                    state_dict['word_embed_rgb.weight'] = state_dict.pop('word_embed.weight')
                if 'word_embed.bias' in state_dict:
                    state_dict['word_embed_rgb.bias'] = state_dict.pop('word_embed.bias')
                if 'norm0_ve.weight' in state_dict:
                    state_dict['norm0_ve_rgb.weight'] = state_dict.pop('norm0_ve.weight')
                if 'norm0_ve.bias' in state_dict:
                    state_dict['norm0_ve_rgb.bias'] = state_dict.pop('norm0_ve.bias')

                if 'word_embed_rgb.weight' in state_dict and (state_dict['word_embed_rgb.weight'].shape[1] != gpt_wo_ddp.word_embed_rgb.in_features):
                    print("Shape mismatch for word_embed_rgb, dropping...")
                    del state_dict['word_embed_rgb.weight']
            else:
                # We are using the normal Src2Sos model
                if 'word_embed.weight' in state_dict and (state_dict['word_embed.weight'].shape[1] != gpt_wo_ddp.word_embed.in_features):
                    del state_dict['word_embed.weight']

            # if 'word_embed.weight' in state_dict and (state_dict['word_embed.weight'].shape[1] != gpt_wo_ddp.word_embed.in_features):
            #     del state_dict['word_embed.weight']
            if 'head.weight' in state_dict and (state_dict['head.weight'].shape[0] != gpt_wo_ddp.head.out_features):
                del state_dict['head.weight']
            if 'head.bias' in state_dict and (state_dict['head.bias'].shape[0] != gpt_wo_ddp.head.bias.shape[0]):
                del state_dict['head.bias']
            if state_dict['text_proj_for_sos.ca.mat_kv.weight'].shape != gpt_wo_ddp.text_proj_for_sos.ca.mat_kv.weight.shape:
                del state_dict['cfg_uncond']
                for key in list(state_dict.keys()):
                    if 'text' in key:
                        del state_dict[key]

            if '1b' in args.model.lower():
                # 1. Dynamically figure out the chunk size and max layer
                max_chunk = -1
                max_module = -1
                for key in state_dict.keys():
                    if key.startswith('block_chunks.'):
                        parts = key.split('.')
                        c_idx = int(parts[1])
                        m_idx = int(parts[3]) # block_chunks.0.module.1 -> index 3 is the module number
                        max_chunk = max(max_chunk, c_idx)
                        max_module = max(max_module, m_idx)
                
                chunk_size = max_module + 1
                max_layer = (max_chunk * chunk_size) + max_module
    
                # 2. Only apply the 2B -> 1B halving if it's a deep checkpoint
                if max_layer > 15:
                    print(f"Detected 1B model target ({args.model}) and deep checkpoint (max layer {max_layer}). Applying alternating layer drop for chunked weights...")
                    
                    for key in list(state_dict.keys()):
                        if key.startswith('block_chunks.'):
                            parts = key.split('.')
                            c_idx = int(parts[1])
                            m_idx = int(parts[3])
                            
                            # Calculate the absolute layer index (0 to 31)
                            abs_layer_idx = (c_idx * chunk_size) + m_idx
                            
                            if abs_layer_idx % 2 == 0:
                                # Keep even layers, but remap them to their new 1B positions
                                new_abs_idx = abs_layer_idx // 2
                                new_c_idx = new_abs_idx // chunk_size
                                new_m_idx = new_abs_idx % chunk_size
                                
                                parts[1] = str(new_c_idx)
                                parts[3] = str(new_m_idx)
                                new_key = '.'.join(parts)
                                
                                # Move the weights to the new target key
                                state_dict[new_key] = state_dict.pop(key)
                            else:
                                # Drop the odd layers entirely
                                del state_dict[key]
                else:
                    print(f"Detected 1B model target ({args.model}), but checkpoint max layer is {max_layer}. Skipping alternating layer drop.")
            return state_dict
        
        gpt_wo_ddp.load_state_dict(drop_unfit_weights(state_dict), strict=False)
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema.load_state_dict(drop_unfit_weights(ema_state_dict), strict=False)

    if args.rwe:
        gpt_wo_ddp.word_embed.weight.requires_grad = False
        torch.nn.init.trunc_normal_(gpt_wo_ddp.word_embed.weight.data, std=1.5 * math.sqrt(1 / gpt_wo_ddp.C / 3))
        if hasattr(gpt_wo_ddp.word_embed, 'bias'):
            gpt_wo_ddp.word_embed.bias.requires_grad = False
            gpt_wo_ddp.word_embed.bias.data.zero_()
    ndim_dict = {name: para.ndim for name, para in gpt_wo_ddp.named_parameters() if para.requires_grad}
    
    print(f'[PT] GPT model = {gpt_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}'
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('VAE', vae_local), ('VAE.quant', vae_local.quantize)
    )]))
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('GPT', gpt_wo_ddp),
    )]) + '\n\n')
    

    gpt_uncompiled = gpt_wo_ddp
    gpt_wo_ddp = args.compile_model(gpt_wo_ddp, args.tfast)

    gpt_ddp_ema = None
    if args.zero:
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        from torch.distributed.device_mesh import init_device_mesh

        # use mix prec: https://github.com/pytorch/pytorch/issues/76607
        if gpt_wo_ddp.num_block_chunks == 1:  # no chunks
            auto_wrap_policy = ModuleWrapPolicy([type(gpt_wo_ddp.unregistered_blocks[0]), ])
        else:
            auto_wrap_policy = ModuleWrapPolicy([MultipleLayers, ])
        
        if args.enable_hybrid_shard:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD if args.zero == 3 else ShardingStrategy._HYBRID_SHARD_ZERO2
            world_size = dist.get_world_size()
            assert world_size % args.inner_shard_degree == 0
            assert args.inner_shard_degree > 1 and args.inner_shard_degree < world_size
            device_mesh = init_device_mesh('cuda', (world_size // args.inner_shard_degree, args.inner_shard_degree))
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD if args.zero == 3 else ShardingStrategy.SHARD_GRAD_OP
            device_mesh = None
        print(f'{">" * 45 + " " * 5} FSDP INIT with {args.zero=} {sharding_strategy=} {auto_wrap_policy=} {" " * 5 + "<" * 45}', flush=True)
        
        gpt_ddp: FSDP = FSDP(
            gpt_wo_ddp, 
            device_id=dist.get_local_rank(),
            sharding_strategy=sharding_strategy, 
            mixed_precision=None,
            auto_wrap_policy=auto_wrap_policy, 
            use_orig_params=True, 
            sync_module_states=True, 
            limit_all_gathers=True,
            device_mesh=device_mesh,
        ).to(args.device)
        
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema = gpt_wo_ddp_ema.to(args.device)
            gpt_ddp_ema: FSDP = FSDP(
                gpt_wo_ddp_ema, 
                device_id=dist.get_local_rank(),
                sharding_strategy=sharding_strategy, 
                mixed_precision=None,
                auto_wrap_policy=auto_wrap_policy, 
                use_orig_params=args.fsdp_orig, 
                sync_module_states=True, 
                limit_all_gathers=True,
            )
    else:
        ddp_class = DDP if dist.initialized() else misc.NullDDP
        gpt_ddp: DDP = ddp_class(gpt_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=args.dbg, broadcast_buffers=False)
    torch.cuda.synchronize()

    # =============== build optimizer ===============
    nowd_keys = set()
    if args.nowd >= 1:
        nowd_keys |= {
            'cls_token', 'start_token', 'task_token', 'cfg_uncond',
            'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
            'gamma', 'beta',
            'ada_gss', 'moe_bias',
            'scale_mul',
            'text_proj_for_sos.ca.mat_q',
        }
    if args.nowd >= 2:
        nowd_keys |= {'class_emb', 'embedding'}
    names, paras, para_groups = filter_params(gpt_ddp if args.zero else gpt_wo_ddp, ndim_dict, nowd_keys=nowd_keys)
    del ndim_dict


    for group in para_groups:
        group['is_backbone'] = True
        group['lr'] = args.tlr

    if '_' in args.ada:
        beta0, beta1 = map(float, args.ada.split('_'))
    else:
        beta0, beta1 = float(args.ada), -1
    
    opt_clz = {
        'sgd':   partial(torch.optim.SGD, momentum=beta0, nesterov=True),
        'adam':  partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.afuse),
    }[args.opt]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    if args.oeps: opt_kw['eps'] = args.oeps
    print(f'[vgpt] optim={opt_clz}, opt_kw={opt_kw}\n')
    gpt_optim = AmpOptimizer('gpt', args.fp16, opt_clz(params=para_groups, **opt_kw), gpt_ddp if args.zero else gpt_wo_ddp, args.r_accu, args.tclip, args.zero)
    del names, paras, para_groups
    
    if args.online_t5:
        print(f'Loading T5 from {args.t5_path}...')
        text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(args.t5_path, revision=None, legacy=True)
        text_tokenizer.model_max_length = args.tlen
        text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(args.t5_path, torch_dtype=torch.float16)
        text_encoder.to(args.device)
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        [p.requires_grad_(False) for p in text_encoder.parameters()]
    else:
        text_tokenizer = text_encoder = None
    
    return text_tokenizer, text_encoder, vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim


def load_eval_config(json_path):
    from infinity.dataset.eval_config import load_eval_config as load_config
    return load_config(json_path)


def flatten_tasks(iterator):
    """Takes a list of individual task dictionaries and streams them one by one."""
    for item in iterator:
        if isinstance(item, list):
            for task in item:
                yield task
        else:
            yield item

def build_dataloaders(args):
    if args.workers < 1:
        raise ValueError('--workers must be at least 1 for WebDataset training')

    print(f"args.data_load_reso: {args.data_load_reso}")
    print(f"args.multiview: {args.multiview}")

    if args.eval_backend not in ('webdataset', 'rendered'):
        raise ValueError("eval_backend must be 'webdataset' or 'rendered'")
    if args.eval_backend == 'rendered':
        from inference.rendered_eval import rendered_eval_config
        rendered_eval_config(args)  # Validate paths and indices before loading model weights.

    if args.overfit_run:
        args.workers = 1

    raw_paths = args.data_path.split(',') if isinstance(args.data_path, str) else args.data_path
    if isinstance(raw_paths, str):
        raw_paths = [raw_paths]

    all_urls = []
    for p in raw_paths:
        p = p.strip()
        if "*" in p:
            all_urls.extend(glob.glob(p))
        elif os.path.isdir(p):
            all_urls.extend(glob.glob(os.path.join(p, "*.tar")))
        else:
            # Handles single .tar files OR WebDataset brace expansions like "shard-{00..99}.tar"
            all_urls.append(p)

    if isinstance(all_urls, list):
        all_urls.sort()

    if args.max_urls != 0:
        all_urls = all_urls[:args.max_urls]
    print(f"len(all_urls): {len(all_urls)}")
    print(f"args.eval_json: {args.eval_json}")

    eval_dict = load_eval_config(args.eval_json) if args.eval_json else {}

    process_fn_train = partial(process_multiview_rgb, args=args, is_eval=False)
    process_fn_eval = partial(process_multiview_rgb, args=args, is_eval=True, eval_dict=eval_dict)

    collate_fn_train = partial(collate_multiview_rgb, args=args, is_eval=False)
    collate_fn_eval = partial(collate_multiview_rgb, args=args, is_eval=True)

    world_size = tdist.get_world_size() if tdist.is_initialized() else 1        
    scenes_per_gpu = math.ceil(args.train_scenes / world_size)
    batches_per_gpu = math.ceil(scenes_per_gpu / args.batch_size)
    
    safe_workers = max(1, args.workers)
    batches_per_worker = math.ceil(batches_per_gpu / safe_workers)

    print(f"args.overfit_run: {args.overfit_run}")
    print(f"batches_per_worker: {batches_per_worker}")
    print(f"args.data_path: {args.data_path}")
    print(f"args.train_scenes: {args.train_scenes}")
    print(f"eval_dict: {eval_dict}")

    def no_split(urls):
        return urls

    if args.overfit_run:
        args.prefetch_factor = None
        # Define a dummy splitter that just hands the files to everyone
        
        
        batches_per_worker = math.ceil(args.train_scenes / args.batch_size)
        # train loader
        wds_train = wds.WebDataset(all_urls, 
                                    resampled=False, 
                                    empty_check=False,
                                    nodesplitter=no_split, # <--- Tells WDS: "I know it's multi-GPU, don't split!"
                                    workersplitter=no_split)

        dataset_train = (
            wds_train
            # 2. Grab only the first N valid training scenes
            .slice(args.train_scenes) 
            # 3. Loop them infinitely
            .repeat()                 
            .map(process_fn_train)
            .select(lambda x: x is not None)
            .batched(args.batch_size, partial=False, collation_fn=collate_fn_train)
            .with_epoch(batches_per_worker) 
            .with_length(batches_per_worker * args.workers)
        )

    else:

        wds_train = wds.WebDataset(
            all_urls, 
            resampled=True,        # Safest way to invoke internal wds.ResampledShards
            empty_check=False,
            nodesplitter=wds.split_by_node,     # <--- ADD THIS
            workersplitter=wds.split_by_worker
        )

        # 2. Process and enforce batch quotas for DDP syncing
        dataset_train = (
            wds_train
            .shuffle(1000)
            # 1. Bouncer: Drop eval scenes to prevent data leakage
            .select(lambda sample: sample["__key__"] not in eval_dict) 
            .map(process_fn_train)
            .select(lambda x: x is not None)
            .batched(args.batch_size, partial=False, collation_fn=collate_fn_train)
            .with_epoch(batches_per_worker) 
            .with_length(batches_per_worker * args.workers)
        )

    dataset_eval = None
    if args.eval_backend == 'webdataset':
        # 4. BUILD EVAL LOADER (Only Keeps Eval Scenes)
        dataset_eval = (
            wds.WebDataset(args.eval_path, 
                            resampled=False, 
                            empty_check=False,
                            nodesplitter=no_split,
                            workersplitter=no_split)
            # .split_by_node()
            # .split_by_worker()
            .select(lambda sample: sample["__key__"] in eval_dict) # <--- KEEPS ONLY EVAL SCENES
            .map(process_fn_eval)
            .select(lambda x: x is not None)
            .compose(flatten_tasks)
            .batched(1, partial=True, collation_fn=collate_fn_eval)
        )
            
    type_train_set = type(dataset_train).__name__
    vbs = round(args.batch_size * 1.5)
    print(f"{args.batch_size=}, {vbs=}", flush=True)
    # ld_val = math.ceil(50000 / vbs)
    # ld_train = DataLoader(dataset=dataset_train, num_workers=args.workers, pin_memory=True, generator=args.get_different_generator_for_each_rank(), batch_size=None, prefetch_factor=args.prefetch_factor)
    
    ld_train = DataLoader(dataset=dataset_train,
                          num_workers=args.workers,
                          batch_size=None,
                          pin_memory=True,
                          persistent_workers=True if args.workers > 0 else False,
                          prefetch_factor=args.prefetch_factor)

    ld_eval = (
        DataLoader(dataset_eval, num_workers=1, batch_size=None, pin_memory=True)
        if dataset_eval is not None else None
    )


    iters_train = len(ld_train)
    # print(f'len(dataloader): {len(ld_train)}, len(dataset): {len(dataset_train)}, total_samples: {dataset_train.total_samples()}')
    print(f'len(dataloader): {len(ld_train)}, len(dataset): {len(dataset_train)}')
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, type(train_set)={type_train_set}')
    return iters_train, ld_train, ld_eval

def eval_model(ld_eval, trainer, gpt, vae, bitwise_self_correction, text_tokenizer, text_encoder, vae_type, device, pn="0.06M", args=None):
    if ld_eval is None:
        return None, None, None
        
    generated_images_list = []

    target_images_list = []
    source_images_list = []

    for batch in ld_eval:
        
        images_src, captions_src, poses_src, intrs_src, images_tgt, captions_tgt, poses_tgt, intrs_tgt = batch
        
        # Move everything to device
        images_src = images_src.to(device)
        poses_src = poses_src.to(device)
        intrs_src = intrs_src.to(device)
        images_tgt = images_tgt.to(device)
        poses_tgt = poses_tgt.to(device)
        intrs_tgt = intrs_tgt.to(device)

        # Prepare Target & Source lists for metrics and grid saving
        for b in range(images_tgt.shape[0]):
            tgt_image_i = (images_tgt[b] + 1) / 2
            tgt_image_i = (255 * tgt_image_i).to(torch.uint8)
            target_images_list.append(tgt_image_i)

            src_image_i = (images_src[b] + 1) / 2
            src_image_i = (255 * src_image_i).to(torch.uint8)
            source_images_list.append(src_image_i)


        _, _, H, W = images_src[0].shape
        h_div_w = 1.0
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][pn]['scales']
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

        # Loop over the items in the batch
        B = images_src.shape[0]
        for pair_idx in range(B):
            # Extract single item (acting like a batch size of 1 for the inference logic)
            im_src = images_src[pair_idx]
            im_tgt = images_tgt[pair_idx]
            pos_src = poses_src[pair_idx]
            pos_tgt = poses_tgt[pair_idx]
            int_src = intrs_src[pair_idx]
            int_tgt = intrs_tgt[pair_idx]

            N_views_tgt = im_tgt.shape[0]

            raw_features_src, _, _ = vae.encode_for_raw_features(im_src, scale_schedule=scale_schedule)
            multiscale_feat_src, _ = bitwise_self_correction.flip_requant_nonoise(scale_schedule, im_src, raw_features_src, device)

            rays = []
            for i in range(1, len(scale_schedule)):
                _, h_i, w_i = scale_schedule[i]
                rays_i = plucker_rays_paired(pos_src[:1][None], pos_tgt[None], int_tgt[None].clone(), target_size=[h_i,w_i], real_size=[h_i, w_i])
                rays.append(rays_i.reshape(1, N_views_tgt, h_i*w_i, 6).to(device))
            rays = torch.cat(rays, 2)

            rays_src = []
            for i in range(len(scale_schedule)):
                _, h_i, w_i = scale_schedule[i]
                rays_i = plucker_rays_paired(pos_src[:1][None], pos_src[None], int_src[None].clone(), target_size=[h_i,w_i], real_size=[h_i, w_i])
                rays_src.append(rays_i.reshape(1, im_src.shape[0], h_i*w_i, 6).to(device))
            rays_src = torch.cat(rays_src, 2)

            cfg_list = [1.0] * len(scale_schedule)

            gpt.eval()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
                
                _, _, img_list, _ = gpt.autoregressive_infer_cfg(
                    vae_features_src=raw_features_src[None],
                    images_src=im_src[None],
                    multiscale_feat_src=multiscale_feat_src[None],
                    vae=vae,
                    scale_schedule=scale_schedule,
                    rays=rays,
                    poses=pos_tgt[None], 
                    intrs=int_tgt[None], 
                    rays_src=rays_src,
                    poses_src=pos_src[None],
                    intrs_src=int_src[None],
                    N_views=N_views_tgt,
                    input_size=[H,W],
                    g_seed=1,
                    B=1, 
                    cfg_list=cfg_list, 
                    tau_list=[0.5]*len(scale_schedule), 
                    cfg_insertion_layer=[0],
                    top_k=900, top_p=0.97, gumbel=0, gt_leak=-1, gt_ls_Bl=None, 
                    sampling_per_bits=1, cfg_exp_k=0.0, vae_type=vae_type, ret_img=True, 
                    trunk_scale=1000, inference_mode=True,
                    returns_vemb=1,
                )


                img_list = img_list.flip(dims=(3,))
                generated_images_list.append(img_list)

    return (
        generated_images_list,
        target_images_list,
        source_images_list,
    )


def save_multiview_comparison_grid(src_tensor, tgt_tensor, gen_tensor, save_path, padding=2):
    """
    Saves a stacked image grid for multiview generation comparison.
    Row 1: Source/Reference views
    Row 2: Target/Ground Truth views
    Row 3: Generated views
    
    Args:
        src_tensor (torch.Tensor): Reference images of shape (M, C, H, W).
        tgt_tensor (torch.Tensor): Target images of shape (N, C, H, W).
        gen_tensor (torch.Tensor): Generated images of shape (N, C, H, W).
        save_path (str): Full file path where the image will be saved.
        padding (int): Padding between images in the grid.
    """
    M = src_tensor.size(0)
    N = tgt_tensor.size(0)
    C, H, W = gen_tensor.shape[1:]
    
    max_cols = max(M, N)
    
    # Helper to pad shorter rows with black images
    def pad_to_max(tensor, desired_length):
        curr_len = tensor.size(0)
        if curr_len < desired_length:
            blanks = torch.zeros(
                (desired_length - curr_len, C, H, W), 
                device=tensor.device, 
                dtype=tensor.dtype
            )
            return torch.cat([tensor, blanks], dim=0)
        return tensor

    # Pad all inputs so they have the same number of images (max_cols)
    src_padded = pad_to_max(src_tensor, max_cols)
    tgt_padded = pad_to_max(tgt_tensor, max_cols)
    gen_padded = pad_to_max(gen_tensor, max_cols)
    
    # Stack vertically: total shape becomes (3 * max_cols, C, H, W)
    combined_imgs = torch.cat([src_padded, tgt_padded, gen_padded], dim=0)
    
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    grid = torchvision.utils.make_grid(combined_imgs, nrow=max_cols, padding=padding)
    
    # 2. Use your exact original PIL saving logic!
    Image.fromarray(grid.permute(1, 2, 0).detach().cpu().numpy()).save(save_path)

    # Save the grid (nrow=max_cols forces the clean 3-row layout)
    # torchvision.utils.save_image(combined_imgs, save_path, nrow=max_cols, padding=padding)


def run_training_evaluation(
    *,
    args: arg_util.Args,
    trainer,
    ld_eval,
    text_tokenizer,
    text_encoder,
    ep: int,
    iters_train: int,
    global_iteration: int,
    evaluation_tag: str,
    save_visuals: bool,
) -> dict:
    """Run the existing evaluation suite synchronously and report it immediately."""
    rank = dist.get_rank()
    evaluation_error = [None]
    evaluation_metrics = {}
    model_was_training = trainer.gpt.training

    with torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params(trainer.gpt):
        if rank == 0:
            python_rng_state = random.getstate()
            numpy_rng_state = np.random.get_state()
            torch_rng_state = torch.get_rng_state()
            cuda_rng_state = (
                torch.cuda.get_rng_state(torch.cuda.current_device())
                if torch.cuda.is_available()
                else None
            )
            model_rng = getattr(trainer.gpt_wo_ddp, "rng", None)
            model_rng_state = model_rng.get_state() if model_rng is not None else None
            try:
                trainer.gpt.eval()
                if args.eval_backend == 'rendered':
                    from inference.rendered_eval import evaluate_rendered_objaverse
                    print(
                        f"\n[evaluation][{evaluation_tag}][global_iteration={global_iteration}] "
                        f"Running rendered Objaverse evaluation (CFG={args.eval_cfg}, seed={args.eval_seed})",
                        flush=True,
                    )
                    evaluation_metrics = evaluate_rendered_objaverse(
                        args, trainer.gpt, trainer.vae_local,
                        evaluation_tag, global_iteration, save_visuals,
                    )
                else:
                    print(
                        f"\n[evaluation][{evaluation_tag}][global_iteration={global_iteration}] "
                        "Running RGB evaluation (CFG=1.0)",
                        flush=True,
                    )
                    lpips_values = []
                    psnr_values = []
                    ssim_values = []
                    trainer.gpt.eval()

                    with torch.no_grad():
                        (
                            generated_images_list,
                            target_images_list,
                            source_images_list,
                        ) = eval_model(
                            ld_eval,
                            trainer,
                            trainer.gpt,
                            trainer.vae_local,
                            trainer.bitwise_self_correction,
                            text_tokenizer,
                            text_encoder,
                            args.vae_type,
                            device=args.device,
                            pn=args.pn,
                            args=args,
                        )

                        if generated_images_list is not None:
                            for generated_images, target_images in zip(
                                generated_images_list,
                                target_images_list,
                            ):
                                for image_index in range(generated_images.shape[0]):
                                    _, lpips_i, ssim_i, psnr_i = calc_2D_metrics(
                                        generated_images[image_index].cpu().numpy(),
                                        target_images[image_index].permute(1, 2, 0).cpu().numpy(),
                                    )
                                    lpips_values.append(lpips_i)
                                    psnr_values.append(psnr_i)
                                    ssim_values.append(ssim_i)


                    evaluation_metrics.update(
                        lpips_eval=float(np.mean(lpips_values)) if lpips_values else float("nan"),
                        ssim_eval=float(np.mean(ssim_values)) if ssim_values else float("nan"),
                        psnr_eval=float(np.mean(psnr_values)) if psnr_values else float("nan"),
                    )

                    if save_visuals and generated_images_list is not None:
                        os.makedirs(args.out_dir, exist_ok=True)
                        for i, generated_images in enumerate(generated_images_list):
                            save_multiview_comparison_grid(
                                source_images_list[i],
                                target_images_list[i],
                                generated_images.permute(0, 3, 1, 2),
                                os.path.join(args.out_dir, f"[eval]img_{i}_{evaluation_tag}.png"),
                            )


                print(
                    f"[evaluation][{evaluation_tag}] "
                    f"lpips_eval={evaluation_metrics['lpips_eval']:.4f} | "
                    f"ssim_eval={evaluation_metrics['ssim_eval']:.4f} | "
                    f"psnr_eval={evaluation_metrics['psnr_eval']:.4f}",
                    flush=True,
                )

                evaluation_log = {
                    "evaluation/global_iteration": global_iteration,
                    **evaluation_metrics,
                }
                if args.logger_name == "wandb":
                    wandb_utils.wandb.log(evaluation_log, step=global_iteration)
                elif args.logger_name == "trackio":
                    trackio.log(evaluation_log, step=global_iteration)
            except Exception:
                evaluation_error[0] = traceback.format_exc()
            finally:
                random.setstate(python_rng_state)
                np.random.set_state(numpy_rng_state)
                torch.set_rng_state(torch_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state, torch.cuda.current_device())
                if model_rng is not None and model_rng_state is not None:
                    model_rng.set_state(model_rng_state)

        tdist.broadcast_object_list(evaluation_error, src=0)
        if evaluation_error[0] is not None:
            trainer.gpt.train(model_was_training)
            raise RuntimeError(
                "Integrated training evaluation failed on rank 0:\n"
                f"{evaluation_error[0]}"
            )
        tdist.barrier()

    trainer.gpt.train(model_was_training)
    return evaluation_metrics


def main_train(args: arg_util.Args):
    if args.eval_freq <= 0:
        raise ValueError("--eval_freq must be greater than 0")
    if args.eval_iters_freq < 0:
        raise ValueError("--eval_iters_freq must be >= 0")
    if args.eval_iters_freq > 0 and args.eval_iters_freq % args.ac != 0:
        raise ValueError(
            "--eval_iters_freq must be divisible by --ac so evaluation runs "
            "immediately after a completed optimizer update"
        )
    saver = CKPTSaver(dist.is_master(), eval_milestone=None)


    ret = build_everything_from_args(args, saver)

    if ret is None:
        return
    (
        text_tokenizer, text_encoder, trainer,
        start_ep, start_it, acc_str, eval_milestone,
        iters_train, ld_train, ld_eval
    ) = ret
    gc.collect(), torch.cuda.empty_cache()
    

    start_time, min_L_mean, min_L_tail, max_acc_mean, max_acc_tail = time.time(), 999., 999., -1., -1.
    seg5 = np.linspace(1, args.ep, 5+1, dtype=int).tolist()
    logging_params_milestone: List[int] = np.linspace(1, args.ep, 10+1, dtype=int).tolist()
    vis_milestone_ep = set(seg5[:]) | set(x for x in (2, 4, 8, 16) if x <= args.ep)
    for x in [6, 12, 3, 24, 18, 48, 72, 96]:
        if len(vis_milestone_ep) < 10 and x <= args.ep:
            vis_milestone_ep.add(x)
    
    PARA_EMB, PARA_ALN, PARA_OT = 0, 0, 0
    for n, p in trainer.gpt_wo_ddp.named_parameters():
        if not p.requires_grad: continue
        if any(k in n for k in ('class_emb', 'pos_1LC', 'lvl_embed')):
            PARA_EMB += p.numel()
        elif any(k in n for k in ('ada_lin',)):
            PARA_ALN += p.numel()
        else:
            PARA_OT += p.numel()
    PARA_ALL = PARA_EMB + PARA_ALN + PARA_OT
    
    trainer.gpt_opt.log_param(ep=-1)
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    ep_lg = max(1, args.ep // 10) if args.ep <= 100 else max(1, args.ep // 20)
    
    # ============================================= epoch loop begins =============================================
    L_mean, L_tail = -1, -1
    epochs_loss_nan = 0
    # build logger
    if dist.is_master():
        if args.logger_name == "wandb":
            wandb_utils.wandb.init(project=args.project_name, name=args.exp_name, config={})
        elif args.logger_name == "trackio":
            trackio.init(project=args.project_name, name=args.run_name, config={"epochs": args.ep})
        else:
            print("No logger chosen")
        
    
    if dist.is_master() and ld_eval is not None: # Only run on rank 0
        print("\n--- RUNNING DATALOADER SANITY CHECK ---")
        try:
            # 1. Pull exactly one batch
            debug_batch = next(iter(ld_eval))

            images_src, captions_src, poses_src, intrs_src, images_tgt, captions_tgt, poses_tgt, intrs_tgt = debug_batch

            # 2. Print out the shapes and types
            print(f"images_src shape: {images_src.shape} | dtype: {images_src.dtype}")
            print(f"poses_src shape:  {poses_src.shape}")
            print(f"intrs_src shape:  {intrs_src.shape}")
            print(f"images_tgt shape: {images_tgt.shape}")
            print(f"captions_src:     {captions_src}")
            print(f"captions_tgt:     {captions_tgt}")
            
            # 3. Save the images to disk to check visually
            os.makedirs(args.out_dir, exist_ok=True)
            
            # Un-normalize from [-1, 1] to [0, 1] for saving
            vis_src = (images_src.view(-1, 3, images_src.shape[-2], images_src.shape[-1]) + 1) / 2
            vis_tgt = (images_tgt.view(-1, 3, images_tgt.shape[-2], images_tgt.shape[-1]) + 1) / 2
            
            torchvision.utils.save_image(vis_src, os.path.join(args.out_dir, "debug_eval_src.png"), nrow=images_src.shape[1])
            torchvision.utils.save_image(vis_tgt, os.path.join(args.out_dir, "debug_eval_tgt.png"), nrow=images_tgt.shape[1])
            
            print(f"Sanity check complete! Check {args.out_dir} for debug_eval_src.png and debug_eval_tgt.png")
            print("---------------------------------------\n")
            
        except Exception as e:
            print(f"Dataloader failed during sanity check: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    for ep in range(start_ep, args.ep):
        if ep % ep_lg == 0 or ep == start_ep:
            print(f'[PT info]  from ep{start_ep} it{start_it}, acc_str: {acc_str}, diffs: {args.diffs},    =======>  bed: {args.bed}  <=======\n')
        # set epoch for dataloader
        # if args.use_streaming_dataset:
        #     ld_train.dataset.set_epoch(ep)

        evaluation_start_it = start_it if ep == start_ep else 0
        if ep % args.eval_freq == 0:
            run_training_evaluation(
                args=args,
                trainer=trainer,
                ld_eval=ld_eval,
                text_tokenizer=text_tokenizer,
                text_encoder=text_encoder,
                ep=ep,
                iters_train=iters_train,
                global_iteration=ep * iters_train + evaluation_start_it,
                evaluation_tag=(
                    f"ep{ep:04d}-it{evaluation_start_it:06d}-"
                    f"g{ep * iters_train + evaluation_start_it:09d}"
                ),
                save_visuals=True,
            )

        # [train one epoch]
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep=ep,
            is_first_ep=ep == start_ep,
            start_it=start_it if ep == start_ep else 0,
            me=None,
            saver=saver,
            args=args,
            ld_or_itrt=iter(ld_train),
            iters_train=iters_train,
            text_tokenizer=text_tokenizer, text_encoder=text_encoder,
            trainer=trainer,
            logging_params_milestone=logging_params_milestone,
            eval_callback=partial(
                run_training_evaluation,
                args=args,
                trainer=trainer,
                ld_eval=ld_eval,
                text_tokenizer=text_tokenizer,
                text_encoder=text_encoder,
                iters_train=iters_train,
                save_visuals=False,
            ),
        )
        
        # [update the best loss or acc]
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        min_L_mean, max_acc_mean, max_acc_tail = min(min_L_mean, L_mean), max(max_acc_mean, acc_mean), max(max_acc_tail, acc_tail)
        if L_tail != -1:
            min_L_tail = min(min_L_tail, L_tail)


        if dist.get_rank() == 0:
            training_log = {
                "L_mean": L_mean,
                "L_tail": L_tail,
                "acc_mean": acc_mean,
                "acc_tail": acc_tail,
                "grad_norm": grad_norm,
            }
            epoch_end_global_iteration = (ep + 1) * iters_train
            if args.logger_name == "wandb":
                wandb_utils.wandb.log(training_log, step=epoch_end_global_iteration)
            elif args.logger_name == "trackio":
                trackio.log(training_log, step=epoch_end_global_iteration)


        # [check nan]
        epochs_loss_nan += int(not math.isfinite(L_mean))
        if (args.fp16 == 1 and epochs_loss_nan >= 2) or (args.fp16 != 1 and epochs_loss_nan >= 1):
            print(f'[rk{dist.get_rank():02d}] L_mean is {L_mean}, stopping training!', flush=True, force=True)
            sys.exit(666)
        
        # [logging]
        args.cur_phase = 'AR'
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        args.last_Lnll, args.last_Ld, args.acc_all, args.acc_real, args.acc_fake, args.last_wei_g = min_L_mean, min_L_tail, None, (None if max_acc_mean < 0 else max_acc_mean), (None if max_acc_tail < 0 else max_acc_tail), grad_norm
        if math.isfinite(args.last_wei_g) and args.last_wei_g > 4:
            args.grad_boom = 'boom'
        
        save_training_stats = (
            (ep + 1) < 10
            or (ep + 1) % max(1, args.ep // 25) == 0
            or (ep + 1) == args.ep
        )
        if dist.is_master() and save_training_stats:
            law_stats = {
                'last_Lm': L_mean, 'best_Lm': min_L_mean, 'last_Am': acc_mean, 'best_Am': max_acc_mean,
                'last_Lt': L_tail, 'best_Lt': min_L_tail, 'last_At': acc_tail, 'best_At': max_acc_tail,
                'pe': PARA_EMB, 'paln': PARA_ALN, 'pot': PARA_OT, 'pall': PARA_ALL,
            }
            stat_file = os.path.join(args.bed, 'law.stat')
            if os.path.exists(stat_file):
                with open(stat_file, 'r', encoding='utf-8') as law_fp: tag_to_epv = json.load(law_fp)
            else:
                tag_to_epv = {tag: {} for tag in law_stats.keys()}
            for tag, v in law_stats.items():
                tag_to_epv[tag][ep + 1] = v
            with open(stat_file, 'w', encoding='utf-8') as law_fp: json.dump(tag_to_epv, law_fp, indent=2)
            
            # ============= LEGACY =============
            with open(os.path.join(args.bed, 'law'), 'w') as law_fp:
                json.dump(law_stats, law_fp, indent=2)
        print(f'  [*] [ep{ep}]  Lmean: {min_L_mean:.3f} ({L_mean:.3f}), Ltail {min_L_tail:.3f} ({L_tail:.3f}),  Acc m-t: {max_acc_mean:.2f} {max_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        args.dump_log()
    # ============================================= epoch loop ends =============================================
    
    if dist.get_rank() == 0:
        if args.logger_name == "trackio":
            trackio.finish()

    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total Time: {total_time},   Lm: {min_L_mean:.3f} ({L_mean}),   Lt: {min_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    del stats, iters_train, ld_train #, visualizer
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    return


g_speed_ls = deque(maxlen=128)
def train_one_ep(
    ep: int, is_first_ep: bool, start_it: int, me: misc.MetricLogger,
    saver: CKPTSaver, args: arg_util.Args, ld_or_itrt, iters_train: int, 
    text_tokenizer: T5TokenizerFast, text_encoder: T5EncoderModel, trainer, logging_params_milestone,
    eval_callback=None,
):
    # IMPORTANT: import heavy packages after the Dataloader object creation/iteration to avoid OOM
    from trainer import InfinityTrainer
    from infinity.utils.lr_control import lr_wd_annealing
    trainer: InfinityTrainer
    
    step_cnt = 0
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    last_touch = time.time()
    g_it, max_it = ep * iters_train, args.ep * iters_train

    doing_profiling = args.prof and ep == 0 and (args.profall or dist.is_master())
    maybe_record_function = record_function if doing_profiling else nullcontext
    trainer.gpt_wo_ddp.maybe_record_function = maybe_record_function

    last_t_perf = time.time()
    speed_ls: deque = g_speed_ls
    FREQ = max(min(args.prof_freq, iters_train//2-1),1)

    me = misc.MetricLogger()
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ['tlr']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['tnm']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    me.add_meter('skips', misc.SmoothedValue(window_size=iters_train, fmt='{global_avg:.0f}')) # Shows total skips per epoch
    # ============================================= iteration loop begins =============================================
    for it, data in me.log_every(start_it, iters_train, ld_or_itrt, args.log_freq, args.log_every_iter, header):
        g_it = ep * iters_train + it


        if (it+1) % FREQ == 0:
            speed_ls.append((time.time() - last_t_perf) / FREQ)
            last_t_perf = time.time()

        
        with maybe_record_function('before_train'):
            # [get data]

            images_src, captions_src, poses_src, intrs_src, images_tgt, captions_tgt, poses_tgt, intrs_tgt = data


            images_src = images_src.to(args.device)
            images_tgt = images_tgt.to(args.device)

            poses_src = poses_src.to(args.device)
            poses_tgt = poses_tgt.to(args.device)

            intrs_src = intrs_src.to(args.device)
            intrs_tgt = intrs_tgt.to(args.device)
                

            text_cond_tuple = None

            # [logging]
            args.cur_it = f'{it+1}/{iters_train}'
            args.last_wei_g = me.meters['tnm'].median
            if dist.is_local_master() and (it >= start_it + 10) and (time.time() - last_touch > 90):
                _, args.remain_time, args.finish_time = me.iter_time.time_preds(max_it - g_it + (args.ep - ep) * 15)      # +15: other cost
                args.dump_log()
                last_touch = time.time()

            # [schedule learning rate]
            wp_it = args.wp * iters_train
            min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.gpt_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)

            # --- CONDITIONAL FREEZE & RAMP-UP LOGIC ---
            if args.freeze_steps > 0:
                freeze_steps = args.freeze_steps
                ramp_steps = args.ramp_steps

                for group in trainer.gpt_opt.optimizer.param_groups:
                    if group.get('is_backbone', False):
                        if g_it < freeze_steps:
                            group['lr'] = 0.0  # Phase 1: Frozen
                        elif g_it < (freeze_steps + ramp_steps):
                            progress = (g_it - freeze_steps) / ramp_steps
                            # Phase 2: Ramp-up from 1% to 100%
                            start_multiplier = 0.01
                            end_multiplier = 1.0
                            current_multiplier = start_multiplier + progress * (end_multiplier - start_multiplier)
                            group['lr'] = group['lr'] * current_multiplier


            # [get scheduled hyperparameters]
            progress = g_it / (max_it - 1)
            clip_decay_ratio = (0.3 ** (20 * progress) + 0.2) if args.cdec else 1

            stepping = (g_it + 1) % args.ac == 0
            step_cnt += int(stepping)

        with maybe_record_function('in_training'):
            grad_norm_t, scale_log2_t, num_skips = trainer.train_step(
                ep=ep, it=it, g_it=g_it, stepping=stepping, clip_decay_ratio=clip_decay_ratio,
                metric_lg=me,
                logging_params=stepping and step_cnt == 1 and (ep < 4 or ep in logging_params_milestone),
                text_cond_tuple=text_cond_tuple,
                images_src=images_src,
                images_tgt=images_tgt,
                poses_src=poses_src,
                poses_tgt=poses_tgt,
                intrs_src=intrs_src,
                intrs_tgt=intrs_tgt,
                args=args,
            )

        completed_iteration = g_it + 1
        # Defer an interval that falls inside gradient accumulation until the
        # next optimizer update, so checkpoints never lose pending gradients.
        checkpoint_due = (
            stepping
            and completed_iteration // args.save_model_iters_freq
            > (completed_iteration - args.ac) // args.save_model_iters_freq
        )
        if checkpoint_due:
            epoch_finished = it + 1 == iters_train
            saver.sav(
                args=args, g_it=completed_iteration,
                next_ep=ep + 1 if epoch_finished else ep,
                next_it=0 if epoch_finished else it + 1,
                trainer=trainer,
            )

        with maybe_record_function('after_train'):
            # me.update(tlr=max_tlr)
            me.update(tlr=max_tlr, skips=num_skips)

        evaluation_due = (
            eval_callback is not None
            and args.eval_iters_freq > 0
            and stepping
            and (g_it + 1) % args.eval_iters_freq == 0
        )
        next_epoch_will_evaluate_same_weights = (
            it + 1 == iters_train
            and ep + 1 < args.ep
            and (ep + 1) % args.eval_freq == 0
        )
        if evaluation_due and not next_epoch_will_evaluate_same_weights:
            completed_iteration = g_it + 1
            eval_callback(
                ep=ep,
                global_iteration=completed_iteration,
                evaluation_tag=(
                    f"ep{ep:04d}-it{it + 1:06d}-g{completed_iteration:09d}"
                ),
            )
    # ============================================= iteration loop ends =============================================
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


def main():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    
    print("Available GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("GPU Name:", torch.cuda.get_device_name(0))

    main_train(args)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    args.cur_phase = 'OK'
    print(f'final args:\n\n{str(args)}')
    args.dump_log()
    if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
        sys.stdout.close(), sys.stderr.close()
    dist.barrier()


if __name__ == '__main__':
    try:
        main()
    except Exception as _e:
        time.sleep(dist.get_rank() * 1 + random.random() * 0.5)
        try:
            # noinspection PyArgumentList
            print(f'[rk{dist.get_rank():2d}] {type(_e).__name__}', flush=True, force=True)
        except:
            try: print(f'[rk{dist.get_rank():2d}] {type(_e).__name__}', flush=True)
            except: pass
        if dist.is_master():
            print(f'[err]:\n{_e}')
            traceback.print_exc()
        raise _e
    finally:
        dist.finalize()
        if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
            sys.stdout.close(), sys.stderr.close()
