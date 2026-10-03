"""Evaluate the live training model using the Objaverse batch-inference protocol."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from inference import infer_single
from inference.infer_ext import save_comparison_grid
from infinity.dataset.eval_config import load_eval_config
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w


def rendered_eval_config(args):
    root = Path(args.eval_path)
    if not args.eval_path or not root.is_dir():
        raise ValueError(f"Rendered Objaverse evaluation folder does not exist: {args.eval_path!r}")
    if not args.use_prope:
        raise ValueError('Rendered batch evaluation requires use_prope=True')
    if args.eval_max_scenes <= 0:
        raise ValueError("eval_max_scenes must be positive")

    if not math.isfinite(args.eval_cfg):
        raise ValueError("eval_cfg must be finite")
    if not math.isfinite(args.eval_tau) or args.eval_tau <= 0:
        raise ValueError("eval_tau must be finite and positive")

    if not args.eval_json:
        raise ValueError('Rendered evaluation requires eval_json with scene names and view indices')
    selections = load_eval_config(args.eval_json)
    scenes = [root / name for name in selections][:args.eval_max_scenes]
    pairs = next(iter(selections.values()))
    for scene in scenes:
        if not (scene / 'transforms.json').is_file():
            raise ValueError(f'{args.eval_json}: selected scene has no transforms.json: {scene}')
        with (scene / 'transforms.json').open() as source:
            metadata = json.load(source)
        frame_count = len(metadata.get('frames', []))
        for pair_index, pair in enumerate(pairs):
            for role in ('src', 'tgt'):
                for index in pair[role]:
                    if index >= frame_count:
                        raise ValueError(
                            f'{args.eval_json}: scene {scene.name}, pair {pair_index}, '
                            f'{role} index {index} is out of range for {frame_count} frames'
                        )
    return scenes, pairs


def _save_images(images, view_ids, folder, normalized):
    folder.mkdir(parents=True, exist_ok=True)
    if normalized:
        images = ((images + 1) / 2 * 255).to(torch.uint8).permute(0, 2, 3, 1)
    for image, view_id in zip(images, view_ids):
        Image.fromarray(image.detach().cpu().numpy()).save(folder / f'{view_id}.png')


@torch.no_grad()
def evaluate_rendered_objaverse(args, model, vae, evaluation_tag, global_iteration, save_visuals):
    # Reuse the standalone metrics, including its scikit-image SSIM definition.
    from inference.calc_metric import calc_2D_metrics

    scenes, pairs = rendered_eval_config(args)
    schedule = dynamic_resolution_h_w[1.0][args.pn]['scales']
    schedule = [(1, height, width) for _, height, width in schedule]
    height, width = dynamic_resolution_h_w[1.0][args.pn]['pixel']
    output_root = Path(args.out_dir) / 'evaluation' / evaluation_tag
    metric_names = ('mse', 'lpips', 'ssim', 'psnr')
    aggregate = {name: [] for name in metric_names}
    metrics = {}

    for pair_index, pair in enumerate(pairs):
        src_ids, tgt_ids = pair['src'], pair['tgt']
        n_src, n_tgt = len(src_ids), len(tgt_ids)
        combination = f'pair{pair_index:03d}_src{n_src}_tgt{n_tgt}'
        values = {name: [] for name in metric_names}
        scene_count = 0
        prefix = f'eval/objaverse/{combination}'
        for scene_path in scenes:
            try:
                # Match infer_batched.sh: legacy normalization/intrinsics,
                # JSON-selected views, and no target-view padding.
                images, poses, intrs = infer_single.load_scene(
                    str(scene_path), tgt_h=height, tgt_w=width,
                    requested_indices=src_ids + tgt_ids,
                    use_real_intrinsics=args.use_real_intrinsics,
                )
            except (OSError, ValueError, KeyError, IndexError) as exc:
                print(f'[evaluation] Skipping {scene_path.name}: {exc}', flush=True)
                continue

            source_count = len(src_ids)
            images_src = images[:source_count].to(args.device)
            images_tgt = images[source_count:].to(args.device)
            poses = poses.to(args.device)
            intrs = intrs.to(args.device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=images_src.is_cuda):
                generated = infer_single.gen_one_img(
                    images_src, poses[:source_count][None], poses[source_count:][None],
                    intrs[:source_count][None], intrs[source_count:][None], model, vae,
                    cfg_list=args.eval_cfg, tau_list=args.eval_tau,
                    scale_schedule=schedule, cfg_insertion_layer=[0],
                    vae_type=args.vae_type, sampling_per_bits=1,
                    g_seed=args.eval_seed, gt_leak=-1, gt_ls_Bl=None,
                    report_stats=False,
                )
            targets = ((images_tgt + 1) / 2 * 255).to(torch.uint8).permute(0, 2, 3, 1)
            if generated.shape != targets.shape:
                raise RuntimeError(f'Generated/target shape mismatch for {scene_path.name}: {generated.shape} vs {targets.shape}')
            for prediction, target in zip(generated, targets):
                result = calc_2D_metrics(prediction.cpu().numpy(), target.cpu().numpy())
                for name, value in zip(metric_names, result):
                    values[name].append(float(value))
            scene_count += 1

            if save_visuals:
                destination = output_root / combination / 'objaverse' / scene_path.name
                _save_images(images_src, src_ids, destination / 'input', normalized=True)
                _save_images(images_tgt, tgt_ids, destination / 'gt', normalized=True)
                _save_images(generated, tgt_ids, destination / 'gen', normalized=False)
                save_comparison_grid(images_src, generated, images_tgt, str(destination / 'grid_comparison.png'))

        if not scene_count:
            raise RuntimeError(f'No valid Objaverse scenes for {combination} in {args.eval_path}')
        for name in metric_names:
            metrics[f'{prefix}/{name}'] = float(np.mean(values[name]))
            aggregate[name].extend(values[name])
        metrics[f'{prefix}/num_scenes'] = scene_count
        metrics[f'{prefix}/num_images'] = len(values['psnr'])
        print(
            f'[evaluation][{evaluation_tag}][objaverse][{combination}] '
            f"scenes={scene_count} | LPIPS={metrics[f'{prefix}/lpips']:.4f} | "
            f"SSIM={metrics[f'{prefix}/ssim']:.4f} | PSNR={metrics[f'{prefix}/psnr']:.4f}",
            flush=True,
        )

    # Keep the existing aggregate dashboard keys as well as per-combination scores.
    for name in ('lpips', 'ssim', 'psnr'):
        metrics[f'{name}_eval'] = float(np.mean(aggregate[name]))
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        'global_iteration': global_iteration,
        'settings': {
            'dataset': 'objaverse', 'data_path': args.eval_path,
            'scene_names': [scene.name for scene in scenes],
            'eval_json': args.eval_json, 'view_pairs': pairs,
            'cfg': args.eval_cfg, 'tau': args.eval_tau, 'seed': args.eval_seed,
            'pn': args.pn, 'vae_type': args.vae_type, 'padding': False,
        },
        'metrics': metrics,
    }
    with (output_root / 'metrics.json').open('w') as destination:
        json.dump(report, destination, indent=2)
    return metrics
