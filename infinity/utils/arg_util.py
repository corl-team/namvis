import glob
import json
import math
import os
import random
import shutil
import sys
import time
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import torch
from tap import Tap

import infinity.utils.dist as dist


class Args(Tap):
    local_out_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_output')  # directory for save checkpoints
    data_path: str = ''                 # dataset
    bed: str = ''                       # bed directory for copy checkpoints apart from local_out_path
    bed_auto_backup: bool = True        # copy local_out_path files to bed during autosave
    vae_ckpt: str = ''                  # VAE ckpt
    exp_name: str = ''                  # experiment name
    ds: str = 'oi'                      # only used in GPT training::load_viz_data & FID benchmark
    model: str = ''                     # for VAE training, 'b' or any other for GPT training
    short_cap_prob: float = 0.2         # prob for training with short captions
    project_name: str = 'namvis'        # wandb/trackio project name
    tf32: bool = True                   # whether to use TensorFloat32
    auto_resume: bool = True            # whether to automatically resume from the last checkpoint found in args.bed
    rush_resume: str = ''               # pretrained infinity checkpoint
    nowd: int = 1                       # whether to disable weight decay on sparse params (like class token)
    enable_hybrid_shard: bool = False   # whether to use hybrid FSDP
    inner_shard_degree: int = 1         # inner degree for FSDP
    zero: int = 0                       # ds zero
    buck: str = 'chunk'                 # =0 for using module-wise
    fsdp_orig: bool = True
    enable_checkpointing: str = None    # checkpointing strategy: full-block, self-attn
    pad_to_multiplier: int = 1          # >1 for padding the seq len to a multiplier of this
    log_every_iter: bool = False
    checkpoint_type: str = 'torch'      # checkpoint_type: torch, onmistore
    seed: int = None                    # 3407
    rand: bool = True                   # actual seed = seed + (dist.get_rank()*512 if rand else 0)
    device: str = 'cpu'
    chunk_nodes: int = None
    is_master_node: bool = None
    # dir
    log_txt_path: str = ''
    t5_path: str = ''                   # local T5 checkpoint; used only with --online_t5
    online_t5: bool = False             # encode text prompts with T5 (unused by image conditioning)
    # GPT
    sdpa_mem: bool = True               # whether to use with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)
    tfast: int = 0                      # compile GPT
    model_alias: str = 'b'              # [automatically set; don't specify this]
    rms: bool = False
    aln: float = 1e-3                   # multiplier of ada_lin.w's initialization
    alng: float = -1                    # multiplier of ada_lin.w[gamma channels]'s initialization, -1: the same as aln
    saln: bool = False                  # whether to use a shared adaln layer
    haln: bool = True                   # whether to use a specific adaln layer in head layer
    nm0: bool = False                   # norm before word proj linear
    tau: float = 1                      # tau of self attention in GPT
    cos: bool = True                    # cosine attn as in swin v2
    swi: bool = False                   # whether to use FFNSwiGLU, instead of vanilla FFN
    dp: float = -1
    drop: float = 0.0                   # GPT's dropout (VAE's is --vd)
    hd: int = 0
    ca_gamma: float = -1                # >=0 for using layer-scale for cross attention
    diva: int = 1                       # rescale_attn_fc_weights
    hd0: float = 0.02                   # head.w *= hd0
    dec: int = 1                        # dec depth
    cum: int = 3                        # cumulating fea map as GPT TF input, 0: not cum; 1: cum @ next hw, 2: cum @ final hw
    rwe: bool = False                   # random word emb
    tp: float = 0.0                     # top-p
    tk: float = 0.0                     # top-k
    tini: float = 0.02                  # init parameters
    cfg: float = 0.1                    # >0: classifier-free guidance, drop cond with prob cfg
    rand_uncond = False                 # whether to use random, unlearnable uncond embeding
    ema: float = 0.9999                 # VAE's ema ratio, not VAR's. 0.9977844 == 0.5 ** (32 / (10 * 1000)) from gans, 0.9999 from SD
    tema: float = 0                     # 0.9999 in DiffiT, DiT
    fp16: int = 0                       # 1: fp16, 2: bf16, >2: fp16's max scaling multiplier todo: 记得让quantize相关的feature都强制fp32！另外residueal最好也是fp32（根据flash-attention）nn.Conv2d有一个参数是use_float16？
    fuse: bool = False                  # whether to use fused mlp
    fused_norm: bool = False            # whether to use fused norm
    flash: bool = False                 # whether to use customized flash-attn kernel
    xen: bool = False                   # whether to use xentropy
    use_flex_attn: bool = False         # whether to use flex_attn to speedup training
    use_ar: bool = True                 # True: VAR cross-scale SA; False: SWITTI same-scale-only SA
    switti_attn_backend: str = 'sdpa'   # sdpa, scale-wise SDPA, packed FlashAttention varlen, or FlexAttention
    stable: bool = False
    gblr: float = 1e-4
    dblr: float = None                  # =gblr if is None
    tblr: float = 6e-4
    glr: float = None
    dlr: float = None
    tlr: float = None                   # vqgan: 4e-5
    gwd: float = 0.005
    dwd: float = 0.0005
    twd: float = 0.005                  # vqgan: 0.01
    gwde: float = 0
    dwde: float = 0
    twde: float = 0
    ls: float = 0.0                     # label smooth
    lz: float = 0.0                     # z loss from PaLM = 1e-4   todo
    eq: int = 0                         # equalized loss
    ep: int = 100
    wp: float = 0
    wp0: float = 0.005
    wpe: float = 0.3                    # 0.001, final cosine lr = wpe * peak lr
    sche: str = ''                      # cos, exp, lin
    log_freq: int = 50                  # log frequency in the stdout
    gclip: float = 6.                   # <=0 for not grad clip VAE
    dclip: float = 6.                   # <=0 for not grad clip discriminator
    tclip: float = 2.                   # <=0 for not grad clip GPT; >100 for per-param clip (%%= 100 automatically)
    cdec: bool = False                  # decay the grad clip thresholds of GPT and GPT's word embed
    opt: str = 'adamw'                  # lion: https://cloud.tencent.com/developer/article/2336657?areaId=106001 lr=5e-5（比Adam学习率低四倍）和wd=0.8（比Adam高八倍）；比如在小的 batch_size 时，Lion 的表现不如 AdamW
    ada: str = ''                       # adam's beta0 and beta1 for VAE or GPT, '0_0.99' from style-swin and magvit, '0.5_0.9' from VQGAN
    dada: str = ''                      # adam's beta0 and beta1 for discriminator
    oeps: float = 0                     # adam's eps, pixart uses 1e-10
    afuse: bool = True                  # fused adam
    # data
    pn: str = ''                        # pixel nums, choose from 0.06M, 0.25M, 1M
    scale_schedule: tuple = None        # [automatically set; don't specify this] = tuple(map(int, args.pn.replace('-', '_').split('_')))
    patch_size: int = None              # [automatically set; don't specify this] = 2 ** (len(args.scale_schedule) - 1)
    resos: tuple = None                 # [automatically set; don't specify this]
    data_load_reso: int = None          # [automatically set; don't specify this]
    workers: int = 1                    # DataLoader workers; must be at least 1
    lbs: int = 0                        # local batch size; if lbs != 0, bs will be ignored, and will be reset as round(args.lbs / args.ac) * dist.get_world_size()
    bs: int = 0                         # global batch size; if lbs != 0, bs will be ignored
    batch_size: int = 0                 # [automatically set; don't specify this] batch size per GPU = round(args.bs / args.ac / dist.get_world_size())
    glb_batch_size: int = 0             # [automatically set; don't specify this] global batch size = args.batch_size * dist.get_world_size()
    ac: int = 1                         # gradient accumulation
    r_accu: float = 1.0                 # [automatically set; don't specify this] = 1 / args.ac
    norm_eps: float = 1e-6              # norm eps for infinity
    tlen: int = 512                     # truncate text embedding to this length
    Ct5: int = 2048                     # feature dimension of text encoder
    use_bit_label: int = 1              # pred bitwise labels or index-wise labels
    bitloss_type: str = 'mean'          # mean or sum
    dynamic_resolution_across_gpus: int = 1 # allow dynamic resolution across gpus
    enable_dynamic_length_prompt: int = 0 # enable dynamic length prompt during training
    use_streaming_dataset: int = 0      # use streaming dataset
    iterable_data_buffersize: int = 90000 # streaming dataset buffer size
    save_model_iters_freq: int = 1000   # save model iter freq
    noise_apply_layers: int = -1        # Bitwise Self-Correction: apply noise to layers, -1 means not apply noise
    noise_apply_strength: float = -1    # Bitwise Self-Correction: apply noise strength, -1 means not apply noise
    noise_apply_strength_schedule: str = '' # Optional comma-separated per-context-scale maximum strengths; overrides scalar noise settings
    noise_apply_requant: int = 1        # Bitwise Self-Correction: requant after apply noise
    rope2d_each_sa_layer: int = 0       # apply rope2d to each self-attention layer
    rope2d_normalized_by_hw: int = 1    # apply normalized rope2d
    use_fsdp_model_ema: int = 0         # use fsdp model ema
    add_lvl_embeding_only_first_block: int = 1 # apply lvl pe embedding only first block or each block
    reweight_loss_by_scale: int = 0     # reweight loss by scale
    always_training_scales: int = 100   # trunc training scales
    vae_type: int = 1                   # here 16/32/64 is bsq vae of different quant bits
    fake_vae_input: bool = False        # fake vae input for debug
    model_init_device: str = 'cuda'     # model_init_device
    prefetch_factor: int = 2            # prefetch_factor for dataset
    apply_spatial_patchify: int = 0     # apply apply_spatial_patchify or not
    task_type: str = 't2i'              # take type to t2i or t2v

    # arguments related to multiview:
    multiview: bool = False
    # Optional RGB reconstruction loss through the frozen VAE.
    rgb_recon_loss_weight: float = 0.0
    rgb_recon_decode_mode: str = 'hard'

    out_dir: str = 'outputs'

    use_real_intrinsics: bool = False  # camera intrinsics from the data; False = the placeholder the released checkpoint used
    use_prope: bool = False
    use_rayrope: bool = False
    
    img2img: bool = False
    src2sos: bool = False

    # arguments related to tracking:
    run_name: str = "default_run_name"
    logger_name: str = "" # at the moment trackio/wandb

    # for multi-input/multi-output:
    N_views_src: int = 0
    N_views_tgt: int = 0
    # Optional comma-separated source-view counts sampled once per training
    # batch, e.g. "1,2,3". Requires N_views_src=0 and N_views_tgt>0.
    N_views_src_choices: str = ''

    # optionally specify total number of views
    N_views_total: int = 0

    # whether sos token is created from image or text

    # whether to use model with convnext encoder(instead of VAE encoder)
    convnext: bool = False

    # whether to use multiscale features
    multiscale_context: bool = False

    # whether to only have self attn
    self_only: bool = False

    # whether to run test with src-ref identity poses and same images
    identity_test: bool = False

    eval_path: str = ""  # TAR for webdataset; Objaverse scene-folder root for rendered
    eval_json: str = ""
    eval_backend: str = 'webdataset'  # 'rendered' uses the infer_batched.sh protocol
    # Rendered evaluation settings follow inference/infer_batched.sh.
    eval_max_scenes: int = 100
    eval_cfg: float = 1.0
    eval_tau: float = 0.5
    eval_seed: int = 0
    train_scenes: int = 1
    overfit_run: bool = False
    eval_freq: int = 1
    eval_iters_freq: int = 0  # Run evaluation every N training iterations; <=0 disables mid-epoch evaluation
    max_urls: int = 0
    # number of freeze iterations(weights that were loaded will be frozen for that amount of steps)
    freeze_steps: int  = 0
    ramp_steps: int = 0


    ############################  Attention! The following arguments and configurations are set automatically, you can skip reading the following part ###############################
    ############################  Attention! The following arguments and configurations are set automatically, you can skip reading the following part ###############################
    ############################  Attention! The following arguments and configurations are set automatically, you can skip reading the following part ###############################


    # would be automatically set in runtime
    # branch: str = subprocess.check_output(f'git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD', shell=True).decode('utf-8').strip() or '[unknown]' # [automatically set; don't specify this]
    commit_id: str = '' # subprocess.check_output(f'git rev-parse HEAD', shell=True).decode('utf-8').strip() or '[unknown]'  # [automatically set; don't specify this]
    commit_msg: str = ''# (subprocess.check_output(f'git log -1', shell=True).decode('utf-8').strip().splitlines() or ['[unknown]'])[-1].strip()    # [automatically set; don't specify this]
    # cmd: str = ' '.join(a.replace('--exp_name=', '').replace('--exp_name ', '') for a in sys.argv[7:])  # [automatically set; don't specify this]
    tag: str = 'UK'                     # [automatically set; don't specify this]
    acc_all: float = None               # [automatically set; don't specify this]
    acc_real: float = None              # [automatically set; don't specify this]
    acc_fake: float = None              # [automatically set; don't specify this]
    last_Lnll: float = None             # [automatically set; don't specify this]
    last_L1: float = None               # [automatically set; don't specify this]
    last_Ld: float = None               # [automatically set; don't specify this]
    last_wei_g: float = None            # [automatically set; don't specify this]
    grad_boom: str = None               # [automatically set; don't specify this]
    diff: float = None                  # [automatically set; don't specify this]
    diffs: str = ''                     # [automatically set; don't specify this]
    diffs_ema: str = None               # [automatically set; don't specify this]
    ca_performance: str = ''            # [automatically set; don't specify this]
    cur_phase: str = ''                 # [automatically set; don't specify this]
    cur_it: str = ''                    # [automatically set; don't specify this]
    cur_ep: str = ''                    # [automatically set; don't specify this]
    remain_time: str = ''               # [automatically set; don't specify this]
    finish_time: str = ''               # [automatically set; don't specify this]
    iter_speed: float = None            # [automatically set; don't specify this]
    img_per_day: float = None           # [automatically set; don't specify this]
    max_nvidia_smi: float = 0           # [automatically set; don't specify this]
    max_memory_allocated: float = None  # [automatically set; don't specify this]
    max_memory_reserved: float = None   # [automatically set; don't specify this]
    num_alloc_retries: int = None       # [automatically set; don't specify this]
    MFU: float = None                   # [automatically set; don't specify this]
    HFU: float = None                   # [automatically set; don't specify this]
    # ==================================================================================================================
    # ======================== ignore these parts below since they are only for debug use ==============================
    # ==================================================================================================================
    dbg_modified: bool = False
    dbg: bool = False                   # enable autograd anomaly detection
    ks: bool = False
    nodata: bool = False    # if True, will set nova=True as well
    nodata_tlen: int = 320
    nova: bool = False      # no val, no FID
    prof: int = 0           # profile
    prof_freq: int = 50     # profile
    tos_profiler_file_prefix: str = 'vgpt_default/'
    profall: int = 0
    @property
    def is_vae_visualization_only(self) -> bool:
        return self.v_seed > 0
    v_seed: int = 0     # v_seed != 0 means the visualization-only mode
    @property
    def is_gpt_visualization_only(self) -> bool:
        return self.g_seed > 0
    g_seed: int = 0     # g_seed != 0 means the visualization-only mode
    # ==================================================================================================================
    # ======================== ignore these parts above since they are only for debug use ==============================
    # ==================================================================================================================
    
    @property
    def gpt_training(self):
        return len(self.model) > 0

    def set_initial_seed(self, benchmark: bool):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = benchmark
        if self.seed is None:
            torch.backends.cudnn.deterministic = False
        else:
            seed = self.seed + (dist.get_rank()*512 if self.rand else 0)
            torch.backends.cudnn.deterministic = True
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
    
    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:   # for random augmentation
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.seed + dist.get_rank()*512)
        return g

    def compile_model(self, m, fast):
        if fast == 0:
            return m
        return torch.compile(m, mode={
            1: 'reduce-overhead',
            2: 'max-autotune',
            3: 'default',
        }[fast]) if hasattr(torch, 'compile') else m
    
    def dump_log(self):
        if not dist.is_local_master():
            return
        nd = {'is_master': dist.is_visualizer()}
        for k, v in {
            # 'name': self.exp_name, 'tag': self.tag, 'cmd': self.cmd, 'commit': self.commit_id, 'branch': self.branch,
            'Lnll': self.last_Lnll, 'L1': self.last_L1,
            'Ld': self.last_Ld,
            'acc': self.acc_all, 'acc_r': self.acc_real, 'acc_f': self.acc_fake,
            'weiG': self.last_wei_g if (self.last_wei_g is None or math.isfinite(self.last_wei_g)) else -23333,
            'grad': self.grad_boom,
            
            'cur': self.cur_phase, 'cur_ep': self.cur_ep, 'cur_it': self.cur_it,
            'rema': self.remain_time, 'fini': self.finish_time, 'last_upd': time.strftime("%Y-%m-%d %H:%M", time.localtime()),
            'bsep': f'{self.glb_batch_size}/{self.ep}',
            'G_lrwd': f'{self.glr:.1e}'.replace('.0', '').replace('-0', '-').replace('+0', '+') + f'/{self.gwd:g}',
            'D_lrwd': f'{self.dlr:.1e}'.replace('.0', '').replace('-0', '-').replace('+0', '+') + f'/{self.dwd:g}',
            'T_lrwd': f'{self.tlr:.1e}'.replace('.0', '').replace('-0', '-').replace('+0', '+') + f'/{self.twd:g}',
            'diff': self.diff, 'diffs': self.diffs, 'diffs_ema': self.diffs_ema if self.diffs_ema else None,
            'opt': self.opt,
            'is_master_node': self.is_master_node,
        }.items():
            if hasattr(v, 'item'):v = v.item()
            if v is None or (isinstance(v, str) and len(v) == 0): continue
            nd[k] = v
        
        with open(self.log_txt_path, 'w') as fp:
            json.dump(nd, fp, indent=2)
    
    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        # self.as_dict() would contain methods, but we only need variables
        for k in self.class_variables.keys():
            if k != 'device':     # not serializable
                d[k] = getattr(self, k)
        return d
    
    def load_state_dict(self, d: Union[OrderedDict, dict, str]):
        if isinstance(d, str):  # for compatibility with old version
            d: dict = eval('\n'.join([l for l in d.splitlines() if '<bound' not in l and 'device(' not in l]))
        for k in d.keys():
            if k in {'is_large_model', 'gpt_training'}:
                continue
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f'k={k}, v={d[k]}')
                raise e
    
    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')
    
    def __str__(self):
        s = []
        for k in self.class_variables.keys():
            if k != 'device':     # not serializable
                s.append(f'  {k:20s}: {getattr(self, k)}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}\n'


def init_dist_and_get_args():
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
            del sys.argv[i]
            break
    args = Args(explicit_bool=True).parse_args(known_only=True)
    if args.workers < 1:
        raise ValueError('--workers must be at least 1 for WebDataset training')
    if args.ac < 1:
        raise ValueError('--ac must be at least 1')
    if args.save_model_iters_freq < 1:
        raise ValueError('--save_model_iters_freq must be greater than 0')
    args.chunk_nodes = int(os.environ.get('CK', '') or '0')
    
    if len(args.extra_args) > 0 and dist.is_master():
        print(f'======================================================================================')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================\n{args.extra_args}')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================')
        print(f'======================================================================================\n\n')
    
    args.set_tf32(args.tf32)
    if args.dbg:
        torch.autograd.set_detect_anomaly(True)
    
    try: os.makedirs(args.bed, exist_ok=True)
    except: pass
    try: os.makedirs(args.local_out_path, exist_ok=True)
    except: pass
    
    day3 = 60*24*3
    dist.init_distributed_mode(local_out_path=args.local_out_path, fork=False, timeout_minutes=day3 if int(os.environ.get('LONG_DBG', '0') or '0') > 0 else 30)
    
    args.tlen = max(args.tlen, args.nodata_tlen)
    if args.zero and args.tema != 0:
        args.tema = 0
        print(f'======================================================================================')
        print(f'======================== WARNING: args.tema:=0, due to zero={args.zero} ========================')
        print(f'======================================================================================\n\n')
    
    if args.nodata:
        args.nova = True
    
    if not args.tos_profiler_file_prefix.endswith('/'): args.tos_profiler_file_prefix += '/'
    
    if args.alng < 0:
        args.alng = args.aln
    
    args.device = dist.get_device()
    args.r_accu = 1 / args.ac   # gradient accumulation
    args.data_load_reso = None
    args.rand |= args.seed is None
    args.sche = args.sche or ('lin0' if args.gpt_training else 'cos')
    if args.wp == 0:
        args.wp = args.ep * 1/100
    
    di = {
        'b': 'bilinear', 'c': 'bicubic', 'n': 'nearest', 'a': 'area', 'aa': 'area+area',
        'at': 'auto', 'auto': 'auto',
        'v': 'vae',
        'x': 'pix', 'xg': 'pix_glu', 'gx': 'pix_glu', 'g': 'pix_glu'
    }
    
    args.ada = args.ada or ('0.9_0.96' if args.gpt_training else '0.5_0.9')
    args.dada = args.dada or args.ada
    args.opt = args.opt.lower().strip()
    
    if args.lbs:
        bs_per_gpu = args.lbs / args.ac
    else:
        bs_per_gpu = args.bs / args.ac / dist.get_world_size()
    bs_per_gpu = round(bs_per_gpu)
    args.batch_size = bs_per_gpu
    args.bs = args.glb_batch_size = args.batch_size * dist.get_world_size()
    args.workers = min(args.workers, bs_per_gpu)
    args.dblr = args.dblr or args.gblr
    args.glr = args.ac * args.gblr * args.glb_batch_size / 4096
    args.dlr = args.ac * args.dblr * args.glb_batch_size / 4096
    args.tlr = args.ac * args.tblr * args.glb_batch_size / 4096
    args.gwde = args.gwde or args.gwd
    args.dwde = args.dwde or args.dwd
    args.twde = args.twde or args.twd
    
    if args.dbg_modified:
        torch.autograd.set_detect_anomaly(True)
    
    # gpt args
    if args.gpt_training:
        assert args.vae_ckpt, 'VAE ckpt must be specified when training GPT'
        if not (args.multiview and args.src2sos and args.task_type == 't2i') or any(
            (args.img2img, args.use_rayrope,
             args.self_only, args.multiscale_context, args.convnext)
        ):
            raise ValueError('This release only supports multiview src2sos t2i training')
        args.model = f'infinity3d_src2sos_{args.model}'


    print(f"args.model: {args.model}")
    
    args.log_txt_path = os.path.join(args.local_out_path, 'log.txt')
    
    args.enable_checkpointing = None if args.enable_checkpointing in [False, 0, "0"] else args.enable_checkpointing
    args.enable_checkpointing = "full-block" if args.enable_checkpointing in [True, 1, "1"] else args.enable_checkpointing
    assert args.enable_checkpointing in [None, "full-block", "full-attn", "self-attn"], \
        f"only support no-checkpointing or full-block/full-attn checkpointing, but got {args.enable_checkpointing}."
    
    if len(args.exp_name) == 0:
        args.exp_name = os.path.basename(args.bed) or 'test_exp'
    
    if '-' in args.exp_name:
        args.tag, args.exp_name = args.exp_name.split('-', maxsplit=1)
    else:
        args.tag = 'UK'
    
    if dist.is_master():
        for marker in glob.glob(os.path.join(args.bed, 'ready-node*')) + glob.glob(os.path.join(args.local_out_path, 'ready-node*')):
            if os.path.isdir(marker):
                shutil.rmtree(marker, ignore_errors=True)
            else:
                os.remove(marker)
    
    if args.sdpa_mem:
        from torch.backends.cuda import enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(True)
        enable_math_sdp(False)
    
    return args
