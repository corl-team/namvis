import torch
import os.path as osp
import os
import cv2
import json

from infinity.dataset.webdataset_utils import transform_wintr
from infinity.models.infinity3d_src2sos import Infinity3DSrc2Sos

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
from PIL import Image as PImage

def add_common_arguments(parser):
    parser.add_argument('--cfg', type=str, default='1')
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--pn', type=str, choices=['0.06M', '0.25M', '1M'], default='0.06M')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--vae_type', type=int, default=32)
    parser.add_argument('--vae_path', type=str, default='weights/infinity_vae_d32reg.pth')
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=1, choices=[0,1])
    parser.add_argument('--use_bit_label', type=int, default=1, choices=[0,1])
    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1,2,4,8,16])
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0,1])
    parser.add_argument('--h_div_w_template', type=float, default=1.000)
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0,1])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bf16', type=int, default=1, choices=[0,1])
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0,1])
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--use_prope', type=int, default=1, choices=[0,1])
    parser.add_argument('--use_real_intrinsics', type=int, default=0, choices=[0,1], help='Use camera intrinsics from transforms.json (1) or the placeholder the released checkpoint was trained with (0)')
    parser.add_argument('--cos', type=int, default=0, choices=[0,1], help='Use cosine attention (1) or scaled dot-product attention (0)')

def depth_to_global_pointmap(depth_array: np.ndarray, K: np.ndarray, c2w: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    H, W = depth_array.shape[:2]
    if depth_array.ndim > 2:
        depth_array = depth_array[:, :, 0]
    true_depth = depth_min + (depth_array * (depth_max - depth_min))
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X_cam = ((u - cx) / fx) * true_depth
    Y_cam = ((v - cy) / fy) * true_depth
    Z_cam = true_depth
    pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=-1)
    pts_cam_flat = pts_cam.reshape(-1, 3)
    pts_cam_hom = np.ones((pts_cam_flat.shape[0], 4), dtype=np.float32)
    pts_cam_hom[:, :3] = pts_cam_flat
    pts_global_hom = (c2w @ pts_cam_hom.T).T
    pts_global = pts_global_hom[:, :3]
    return pts_global.reshape(H, W, 3).astype(np.float32)
    
def load_scene(scene_path, tgt_h=256, tgt_w=256, requested_indices=None, use_depth_norm=False, use_camera_distance_norm=False, return_normalization=False, use_real_intrinsics=False):
    json_path = osp.join(scene_path, "transforms.json")
    if not osp.exists(json_path):
        raise FileNotFoundError(f"Missing transforms.json in {scene_path}")
        
    with open(json_path, 'r') as f:
        meta = json.load(f)
        
    frames = meta.get('frames', [])
    if not frames:
        raise ValueError(f"No frames found in {json_path}")
    
    if requested_indices is None:
        requested_indices = list(range(len(frames)))
    if len(requested_indices) == 0:
        raise ValueError(f"No requested views for {scene_path}")
    if use_depth_norm and use_camera_distance_norm:
        raise ValueError("Choose either depth or camera-distance normalization")

    # Training anchors poses on the first selected source view. Preserve the
    # legacy reference for other callers unless this mode is requested.
    frame0 = frames[requested_indices[0] if use_camera_distance_norm else 0]
    c2w_0_raw = np.array(frame0['transform_matrix'], dtype=np.float32)
    blender2cv = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float32)
    
    c2w_0 = c2w_0_raw @ blender2cv
    w2c_0 = np.linalg.inv(c2w_0)
    alpha = None

    if use_camera_distance_norm:
        # Match training: normalize selected cameras relative to the first source.
        max_distance = max(
            float(np.linalg.norm((w2c_0 @ (np.array(frames[idx]['transform_matrix'], dtype=np.float32) @ blender2cv))[:3, 3]))
            for idx in requested_indices
        )
        alpha = 1.0 / max_distance if max_distance > 1e-5 else 1.0

    if use_depth_norm:
        stem0 = osp.splitext(frame0['file_path'])[0]
        depth0_path = osp.join(scene_path, f"depth_{stem0}.exr")
        if osp.exists(depth0_path):
            depth0_map = cv2.imread(depth0_path, cv2.IMREAD_UNCHANGED)
            if depth0_map is not None:
                img0_path = osp.join(scene_path, frame0['file_path'])
                with PImage.open(img0_path) as im0:
                    W0, H0 = im0.size
                
                cam_angle_x = frame0.get('camera_angle_x', meta.get('camera_angle_x', 0.6981317007977318))
                fl_x0 = W0 / (2.0 * np.tan(cam_angle_x / 2.0))
                
                if 'K' in frame0:
                    K0 = np.array(frame0['K'], dtype=np.float32)
                else:
                    K0 = np.array([
                        [fl_x0, 0.0, W0 / 2.0],
                        [0.0, fl_x0, H0 / 2.0],
                        [0.0, 0.0, 1.0]
                    ], dtype=np.float32)
                    
                d_min_0 = frame0.get("depth", {}).get("min", 1.0)
                d_max_0 = frame0.get("depth", {}).get("max", 3.0)
                pm_global_0 = depth_to_global_pointmap(depth0_map, K0, c2w_0, d_min_0, d_max_0)
                pts_global = pm_global_0.reshape(-1, 3)
                pts_hom = np.ones((len(pts_global), 4), dtype=np.float32)
                pts_hom[:, :3] = pts_global
                pts_local_hom = (w2c_0 @ pts_hom.T).T
                pts_local_0 = pts_local_hom[:, :3]
                z_vals = pts_local_0[:, 2]
                valid_z = z_vals[(z_vals > 1e-5) & np.isfinite(z_vals)]
                if len(valid_z) > 0:
                    mean_depth_0 = valid_z.mean()
                    alpha = 1.0 / float(mean_depth_0)

    if alpha is None:
        dist = np.linalg.norm(c2w_0[:3, 3])
        alpha = 1.0 / dist if dist > 1e-5 else 1.0

    images, poses, intrs = [], [], []
    for idx in requested_indices:
        frame = frames[idx]
        img_path = osp.join(scene_path, frame['file_path'])
        
        if not osp.exists(img_path):
            img_path = osp.join(scene_path, f"{idx:04d}.webp")

        img_i: PImage.Image = PImage.open(img_path)
        img_i = img_i.convert("RGBA")
        background = PImage.new("RGB", img_i.size, (255, 255, 255))
        background.paste(img_i, mask=img_i.split()[3])
        img_i = background

        W, H = img_i.size
        cam_angle_x = frame.get('camera_angle_x', meta.get('camera_angle_x', 0.6981317007977318))
        fl_x = W / (2.0 * np.tan(cam_angle_x / 2.0))
        
        if 'K' in frame:
            intr_i = np.array(frame['K'], dtype=np.float32)
        else:
            intr_i = np.array([
                [fl_x, 0.0, W / 2.0],
                [0.0, fl_x, H / 2.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
        intr_i = torch.from_numpy(intr_i)

        orig_c2w = np.array(frame['transform_matrix'], dtype=np.float32)
        orig_c2w = orig_c2w @ blender2cv
        
        rel_c2w = w2c_0 @ orig_c2w
        rel_c2w[:3, 3] *= alpha
        pose_i = torch.from_numpy(rel_c2w)
        
        img_B3HW, new_K = transform_wintr(img_i, intr_i, tgt_h, tgt_w, use_real_intrinsics)

        images.append(img_B3HW)
        poses.append(pose_i)
        intrs.append(new_K)

    result = torch.stack(images), torch.stack(poses), torch.stack(intrs)
    return (*result, float(alpha)) if return_normalization else result

def load_visual_tokenizer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.vae_type in [14,16,18,20,24,32,64]:
        from infinity.models.bsq_vae.vae import vae_model
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type
        codebook_size = 2**codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult=[1, 2, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult=[1, 2, 4, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4, 4]
        vae = vae_model(args.vae_path, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                        encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(device)
    else:
        raise ValueError(f'vae_type={args.vae_type} not supported')
    return vae

def load_infinity(
    rope2d_each_sa_layer, rope2d_normalized_by_hw, pn,
    use_bit_label, add_lvl_embeding_only_first_block, model_path='', scale_schedule=None,
    vae=None, device='cuda', model_kwargs=None, text_channels=2048, apply_spatial_patchify=0,
    use_flex_attn=False, bf16=False, cos_attn=True,
):
    print(f'[Loading Infinity3DSrc2Sos]')
    text_maxlen = 512
    with torch.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
        infinity_test: Infinity3DSrc2Sos = Infinity3DSrc2Sos(
            vae_local=vae, text_channels=text_channels, text_maxlen=text_maxlen,
            shared_aln=True, raw_scale_schedule=scale_schedule, checkpointing='full-block',
            customized_flash_attn=False, fused_norm=True, pad_to_multiplier=0,
            use_flex_attn=use_flex_attn, add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
            use_bit_label=use_bit_label, rope2d_each_sa_layer=rope2d_each_sa_layer,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw, pn=pn,
            apply_spatial_patchify=apply_spatial_patchify, inference_mode=False,
            train_h_div_w_list=[1.0], cos_attn=cos_attn, **model_kwargs,
        ).to(device=device)
        print(f'[you selected Infinity with {model_kwargs=}] model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={bf16}')

        if bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()

        infinity_test.eval()
        infinity_test.requires_grad_(False)
        infinity_test.cuda()
        torch.cuda.empty_cache()

        print(f'[Load Infinity weights]')
        # Expects a weights-only state dict: the released checkpoint or a
        # training *-statedict.pth export.
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        if not cos_attn:
            cosine_scale_keys = [key for key in state_dict if key.endswith('.scale_mul_1H11')]
            for key in cosine_scale_keys:
                del state_dict[key]
            if cosine_scale_keys:
                print(f'[Load Infinity weights] Ignored {len(cosine_scale_keys)} cosine-attention scale tensors for --cos=0')
        print(infinity_test.load_state_dict(state_dict))
        infinity_test.rng = torch.Generator(device=device)
        return infinity_test

def load_transformer(vae, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depth = {'1b': 16}[args.model]

    kwargs_model = dict(depth=depth, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, 
                        mlp_ratio=4, block_chunks=8, use_prope=args.use_prope, 
                        N_views_src=args.N_views_src, N_views_tgt=args.N_views_tgt)

    infinity = load_infinity(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer, rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        pn=args.pn,
        use_bit_label=args.use_bit_label, add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block,
        model_path=args.model_path, scale_schedule=None, vae=vae, device=device,
        model_kwargs=kwargs_model, text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify, use_flex_attn=args.use_flex_attn,
        bf16=args.bf16, cos_attn=bool(args.cos),
    )
    return infinity

def gen_one_img(
    images_src, test_poses_src, test_poses_tgt, test_intrs_src, test_intrs_tgt,
    infinity_test, vae, cfg_list=[], tau_list=[], scale_schedule=None,
    top_k=900, top_p=0.97, cfg_exp_k=0.0, cfg_insertion_layer=0, vae_type=0,
    gumbel=0, softmax_merge_topk=-1, gt_leak=-1, gt_ls_Bl=None, g_seed=None,
    sampling_per_bits=1, report_stats=True,
):

    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)

    H, W = images_src.shape[-2], images_src.shape[-1]
    raw_features_src, _, _ = vae.encode_for_raw_features(images_src, scale_schedule=scale_schedule)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        if report_stats:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_event.record()

        _, _, img_list, _ = infinity_test.autoregressive_infer_cfg(
            vae_features_src=raw_features_src[None], vae=vae, scale_schedule=scale_schedule,
            poses=test_poses_tgt, intrs=test_intrs_tgt, poses_src=test_poses_src,
            intrs_src=test_intrs_src, N_views=test_poses_tgt.shape[1],
            input_size=(H,W), g_seed=g_seed, B=1, force_gt_Bhw=None,
            cfg_list=cfg_list, tau_list=tau_list, cfg_insertion_layer=cfg_insertion_layer,
            top_k=top_k, top_p=top_p, returns_vemb=1, ratio_Bl1=None, gumbel=gumbel,
            norm_cfg=False, cfg_exp_k=cfg_exp_k, vae_type=vae_type, ret_img=True,
            trunk_scale=1000, gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl,
            inference_mode=True, sampling_per_bits=sampling_per_bits,
        )

        if report_stats:
            end_event.record()
            torch.cuda.synchronize()
            cost_s = start_event.elapsed_time(end_event) / 1000.0
            peak_alloc = torch.cuda.max_memory_allocated()
            print(f"time_s: {cost_s:.4f}, peak_alloc_MB: {peak_alloc / 1024**2:.1f}")

    return img_list.flip(dims=(3,))
