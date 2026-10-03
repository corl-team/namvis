import argparse
import os
import os.path as osp
import torch
import numpy as np
from PIL import Image
import torchvision
import json

from inference import infer_single
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

def save_comparison_grid(src_tensor, gen_tensor, gt_tensor, save_paths, padding=2):
    """
    Saves a stacked image grid.
    Row 0: Input (Source)
    Row 1: Generated
    Row 2: Ground Truth (Target)
    """
    if isinstance(save_paths, str):
        save_paths = [save_paths]

    src = (src_tensor.float() + 1.0) / 2.0
    gt = (gt_tensor.float() + 1.0) / 2.0
    
    gen = gen_tensor.float().permute(0, 3, 1, 2)
    if gen.max() > 2.0:
        gen = gen / 255.0

    max_cols = max(src.size(0), gen.size(0), gt.size(0))
    _, C, H, W = gt.shape
    
    def pad_to_max(tensor):
        curr_len = tensor.size(0)
        if curr_len < max_cols:
            blanks = torch.ones((max_cols - curr_len, C, H, W), device=tensor.device)
            return torch.cat([tensor, blanks], dim=0)
        return tensor

    src_padded = pad_to_max(src)
    gen_padded = pad_to_max(gen)
    gt_padded = pad_to_max(gt)
    
    combined_imgs = torch.cat([src_padded, gen_padded, gt_padded], dim=0)
    
    grid = torchvision.utils.make_grid(combined_imgs, nrow=max_cols, padding=padding)
    grid_np = (grid.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    grid_img = Image.fromarray(grid_np)

    for path in save_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        grid_img.save(path)


def main():
    parser = argparse.ArgumentParser()
    infer_single.add_common_arguments(parser)
    
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--N_views_src', type=int, default=2)
    parser.add_argument('--N_views_tgt', type=int, default=2)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--max_scenes', type=int, default=100)
    parser.add_argument('--model', type=str, default='1b', choices=['1b'])
    
    parser.add_argument('--view_step', type=int, default=1)
    parser.add_argument('--src_offset', type=int, default=0)
    parser.add_argument('--tgt_offset', type=int, default=0)
    parser.add_argument('--src_indices', type=int, nargs='+', default=None)
    parser.add_argument('--tgt_indices', type=int, nargs='+', default=None)
    
    parser.add_argument('--pad', action='store_true')
    parser.add_argument('--grid_out_dir', type=str, default=None)
    parser.add_argument('--use_depth_norm', action='store_true', help="Use first-frame depth for camera normalization")
    parser.add_argument('--use_camera_distance_norm', action='store_true', help="Match training: center on first source and normalize by maximum selected-camera distance")
    parser.add_argument('--export_consistency_data', action='store_true', help="Export GT depth pointmaps and cameras for cross-view RGB evaluation")
    
    args = parser.parse_args()
    if args.use_depth_norm and args.use_camera_distance_norm:
        parser.error("--use_depth_norm and --use_camera_distance_norm cannot be combined")
    if args.use_camera_distance_norm:
        for label, indices, count in (
            ("source", args.src_indices, args.N_views_src),
            ("target", args.tgt_indices, args.N_views_tgt),
        ):
            if indices is not None and len(indices) < count:
                parser.error(f"Requested {count} {label} views but only {len(indices)} {label} indices were supplied")

    user_req_src = args.N_views_src
    user_req_tgt = args.N_views_tgt

    args.cfg = list(map(float, str(args.cfg).split(','))) if isinstance(args.cfg, str) else [args.cfg]
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("==================================================")
    print("Loading models... (This will only happen once!)")
    print("==================================================")
    
    vae = infer_single.load_visual_tokenizer(args)
    infinity = infer_single.load_transformer(vae, args)

    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    scene_names = sorted(os.listdir(args.data_path))
    scene_paths = [os.path.join(args.data_path, scene_name) for scene_name in scene_names][:args.max_scenes]

    print(f"\nFound {len(scene_paths)} scenes to process in {args.data_path}")

    for i, scene_path in enumerate(scene_paths):
        scene_name = scene_names[i]
        scene_out_dir = osp.join(args.out_dir, scene_name)
        os.makedirs(scene_out_dir, exist_ok=True)

        print(f"\n---> Processing scene {i+1}/{len(scene_paths)}: {scene_name}")

        try:
            # 1. READ JSON FIRST
            json_path = osp.join(scene_path, "transforms.json")
            if not osp.exists(json_path):
                raise FileNotFoundError("No transforms.json found.")
            with open(json_path, 'r') as f:
                meta = json.load(f)
            num_images = len(meta.get('frames', []))

            # 2. SELECT INDICES
            def select_indices(offset, count, step, explicit_indices):
                if explicit_indices is not None:
                    truncated_indices = explicit_indices[:count]
                    valid_idx = [idx for idx in truncated_indices if idx < num_images]
                    return np.array(valid_idx, dtype=int)
                selected = []
                curr_idx = offset
                while len(selected) < count and curr_idx < num_images:
                    selected.append(curr_idx)
                    curr_idx += step
                return np.array(selected, dtype=int)

            src_ids = select_indices(args.src_offset, user_req_src, args.view_step, args.src_indices)
            tgt_ids = select_indices(args.tgt_offset, user_req_tgt, args.view_step, args.tgt_indices)

            if len(src_ids) == 0 or len(tgt_ids) == 0:
                print(f"Skipping {scene_name}: Could not find enough valid source/target views.")
                continue
            if args.use_camera_distance_norm and (
                len(src_ids) != user_req_src or len(tgt_ids) != user_req_tgt
            ):
                print(
                    f"Skipping {scene_name}: requested {user_req_src} source and "
                    f"{user_req_tgt} target views, found {len(src_ids)} and {len(tgt_ids)}."
                )
                continue

            print(f"src_ids: {src_ids}")
            print(f"tgt_ids: {tgt_ids}")

            # 3. COMBINE AND LOAD (Passes the exact indices and handles normalization)
            combined_indices = src_ids.tolist() + tgt_ids.tolist()
            
            loaded_scene = infer_single.load_scene(
                scene_path,
                requested_indices=combined_indices,
                use_depth_norm=args.use_depth_norm,
                use_camera_distance_norm=args.use_camera_distance_norm,
                return_normalization=args.export_consistency_data,
                use_real_intrinsics=bool(args.use_real_intrinsics),
            )
            images, poses, intrs = loaded_scene[:3]
            normalization_scale = loaded_scene[3] if args.export_consistency_data else None

            # 4. SPLIT BACK INTO SRC AND TGT
            num_src = len(src_ids)
            images_src = images[:num_src].to(device)
            poses_src = poses[:num_src].to(device)
            intrs_src = intrs[:num_src].to(device)

            images_tgt = images[num_src:].to(device)
            poses_tgt = poses[num_src:].to(device)
            intrs_tgt = intrs[num_src:].to(device)

        except (OSError, ValueError, KeyError, IndexError) as e:
            print(f"Skipping {scene_name}: {e}")
            continue

        # ==========================================
        # 5. TARGET-ONLY PADDING 
        # ==========================================
        total_views = images_src.shape[0] + images_tgt.shape[0]
        required_views = 5
        original_tgt_count = images_tgt.shape[0]
        
        pad_triggered = (total_views < required_views) and args.pad

        if pad_triggered:
            needed = required_views - total_views
            print(f"Padding sequence: Adding {needed} dummy TARGET views to reach 5 total.")

            dummy_images = images_tgt[-1:].repeat(needed, 1, 1, 1)
            images_tgt = torch.cat([images_tgt, dummy_images], dim=0).contiguous()

            dummy_poses = poses_tgt[-1:].repeat(needed, 1, 1)
            # Add microscopic noise so cameras don't overlap exactly
            dummy_poses[:, :3, 3] += torch.randn_like(dummy_poses[:, :3, 3]) * 0.001
            poses_tgt = torch.cat([poses_tgt, dummy_poses], dim=0).contiguous()

            dummy_intrs = intrs_tgt[-1:].repeat(needed, 1, 1)
            intrs_tgt = torch.cat([intrs_tgt, dummy_intrs], dim=0).contiguous()
            
            args.N_views_tgt = images_tgt.shape[0]
            args.N_views_src = images_src.shape[0]

        elif total_views < required_views:
            print(f"Total views ({total_views}) < 5. Proceeding without padding (--pad not used).")

        # ==========================================
        # 6. INFERENCE
        # ==========================================
        with torch.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                generated_image = infer_single.gen_one_img(
                    images_src,
                    poses_src[None],
                    poses_tgt[None],
                    intrs_src[None],
                    intrs_tgt[None],
                    infinity,
                    vae,
                    g_seed=args.seed,
                    gt_leak=-1,
                    gt_ls_Bl=None,
                    cfg_list=args.cfg,
                    tau_list=args.tau,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[args.cfg_insertion_layer],
                    vae_type=args.vae_type,
                    sampling_per_bits=args.sampling_per_bits,
                )

                # ==========================================
                # 7. CLEANUP & SAVE
                # ==========================================
                if pad_triggered:
                    generated_image = generated_image[:original_tgt_count]
                    images_tgt = images_tgt[:original_tgt_count]
                
                args.N_views_src = user_req_src
                args.N_views_tgt = user_req_tgt

                src_dir = osp.join(scene_out_dir, "input")
                gt_dir = osp.join(scene_out_dir, "gt")
                gen_dir = osp.join(scene_out_dir, "gen")
                
                os.makedirs(src_dir, exist_ok=True)
                os.makedirs(gt_dir, exist_ok=True)
                os.makedirs(gen_dir, exist_ok=True)

                for j in range(images_src.shape[0]):
                    src_image_i = (images_src[j] + 1) / 2
                    src_image_i = (255 * src_image_i).to(torch.uint8)
                    save_path = osp.join(src_dir, f"{src_ids[j]}.png")
                    Image.fromarray(src_image_i.permute(1, 2, 0).detach().cpu().numpy()).save(save_path)

                for j in range(images_tgt.shape[0]):
                    gt_image_i = (images_tgt[j] + 1) / 2
                    gt_image_i = (255 * gt_image_i).to(torch.uint8)
                    save_path = osp.join(gt_dir, f"{tgt_ids[j]}.png")
                    Image.fromarray(gt_image_i.permute(1, 2, 0).detach().cpu().numpy()).save(save_path)

                for j in range(generated_image.shape[0]):
                    save_path = osp.join(gen_dir, f"{tgt_ids[j]}.png")
                    Image.fromarray(generated_image[j].detach().cpu().numpy()).save(save_path)

                if args.export_consistency_data:
                    from inference.consistency_export import export_consistency_data
                    export_consistency_data(
                        scene_dir=scene_path,
                        output_dir=scene_out_dir,
                        frames=meta["frames"],
                        target_ids=tgt_ids.tolist(),
                        target_poses=poses_tgt[:original_tgt_count].detach().cpu().numpy(),
                        normalization_scale=normalization_scale,
                        image_height=images_tgt.shape[-2],
                        image_width=images_tgt.shape[-1],
                        default_camera_angle_x=meta.get("camera_angle_x", 0.6981317007977318),
                    )

                grid_save_paths = [osp.join(scene_out_dir, "grid_comparison.png")]
                if args.grid_out_dir is not None:
                    grid_save_paths.append(osp.join(args.grid_out_dir, f"{scene_name}_grid.png"))
                
                save_comparison_grid(images_src, generated_image, images_tgt, grid_save_paths)

if __name__ == '__main__':
    main()
