import os
import cv2
import torch
import numpy as np
import argparse
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import warnings

import lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim_func

# Loaded once at import; reused for every image.
loss_fn_alex = lpips_lib.LPIPS(net='alex')

def LPIPS(pred, gt):
    """LPIPS (AlexNet) for inputs normalized to [-1, 1]."""
    return loss_fn_alex(pred, gt)

def calculate_ssim(pred_np, gt_np, channel_axis=2):
    return ssim_func(pred_np, gt_np, channel_axis=channel_axis, data_range=255)

def calc_2D_metrics(pred_np, gt_np):
    # pred_np: [H, W, 3], [0, 255], np.uint8
    pred_image = torch.from_numpy(pred_np).unsqueeze(0).permute(0, 3, 1, 2)
    gt_image = torch.from_numpy(gt_np).unsqueeze(0).permute(0, 3, 1, 2)
    
    # [0-255] -> [-1, 1]
    pred_image = pred_image.float() / 127.5 - 1
    gt_image = gt_image.float() / 127.5 - 1
    
    loss = F.mse_loss(pred_image[0], gt_image[0].cpu()).item()
    
    lpips_val = LPIPS(pred_image[0].cpu(), gt_image[0].cpu()).item() 
    
    # SSIM
    ssim_val = calculate_ssim(pred_np, gt_np, channel_axis=2)
    
    # PSNR
    psnr_val = cv2.PSNR(gt_np, pred_np)

    return loss, lpips_val, ssim_val, psnr_val


def resolve_prediction_dir(scene_path):
    """Return the prediction directory, preferring legacy ``gen/`` outputs."""
    for subdir in ("gen", "images"):
        candidate = os.path.join(scene_path, subdir)
        if os.path.isdir(candidate):
            return candidate
    return None


def load_yolo_bbox(annotation_path, image_width, image_height):
    """Load one YOLO box and return clipped ``(x1, y1, x2, y2)`` pixels."""
    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(f"Missing YOLO annotation: {annotation_path}")

    with open(annotation_path, "r") as annotation_file:
        rows = [line.strip() for line in annotation_file if line.strip()]

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one YOLO row in {annotation_path}, found {len(rows)}"
        )

    fields = rows[0].split()
    if len(fields) != 5:
        raise ValueError(
            f"Expected 5 YOLO fields in {annotation_path}, found {len(fields)}"
        )

    try:
        class_id, x_center, y_center, box_width, box_height = map(float, fields)
    except ValueError as exc:
        raise ValueError(f"Non-numeric YOLO value in {annotation_path}") from exc

    values = np.array(
        [class_id, x_center, y_center, box_width, box_height], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite YOLO value in {annotation_path}")
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"YOLO box must have positive size in {annotation_path}")

    x1 = int(np.floor((x_center - box_width / 2.0) * image_width))
    y1 = int(np.floor((y_center - box_height / 2.0) * image_height))
    x2 = int(np.ceil((x_center + box_width / 2.0) * image_width))
    y2 = int(np.ceil((y_center + box_height / 2.0) * image_height))

    x1 = max(0, min(image_width, x1))
    y1 = max(0, min(image_height, y1))
    x2 = max(0, min(image_width, x2))
    y2 = max(0, min(image_height, y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"YOLO box is empty after clipping in {annotation_path}: "
            f"({x1}, {y1}, {x2}, {y2})"
        )

    return x1, y1, x2, y2


def resolve_yolo_annotation(yolo_root, scene, image_filename):
    """Resolve a view's annotation within its source scene directory."""
    view_stem = os.path.splitext(image_filename)[0]
    try:
        view_id = int(view_stem)
    except ValueError as exc:
        raise ValueError(
            f"Expected a numeric view filename, got {image_filename!r} "
            f"in scene {scene!r}"
        ) from exc

    if view_id < 0:
        raise ValueError(
            f"Expected a non-negative view ID, got {view_id} in scene {scene!r}"
        )

    scene_annotation_root = os.path.join(yolo_root, scene)
    annotation_filename = f"{view_id:04d}.txt"
    preferred_path = os.path.join(
        scene_annotation_root,
        "annotation",
        annotation_filename,
    )
    if os.path.isfile(preferred_path):
        return preferred_path

    matches = sorted(
        glob(
            os.path.join(scene_annotation_root, "**", annotation_filename),
            recursive=True,
        )
    )
    matches = [path for path in matches if os.path.isfile(path)]

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple YOLO annotations for view {view_id} in scene "
            f"{scene!r}: {', '.join(matches)}"
        )

    raise FileNotFoundError(
        f"Missing YOLO annotation {annotation_filename} in source scene: "
        f"{scene_annotation_root}"
    )


def main(args):
    root_dir = args.root
    yolo_annotations_root = getattr(args, 'yolo_annotations_root', None)
    
    # Find all scene folders
    scene_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    scene_folders.sort()
    
    print(f"Found {len(scene_folders)} scenes in {root_dir}")

    excluded_scenes = set(args.exclude_scenes)
    if excluded_scenes:
        available_scenes = set(scene_folders)
        matched_exclusions = sorted(excluded_scenes & available_scenes)
        missing_exclusions = sorted(excluded_scenes - available_scenes)

        if matched_exclusions:
            print(
                f"Excluding {len(matched_exclusions)} scene(s): "
                f"{', '.join(matched_exclusions)}"
            )

        if missing_exclusions:
            warnings.warn(
                "Requested excluded scene(s) not found: "
                f"{', '.join(missing_exclusions)}"
            )

        scene_folders = [
            scene for scene in scene_folders if scene not in excluded_scenes
        ]
        print(f"Evaluating {len(scene_folders)} scene(s) after exclusions.")
    
    # Store aggregate metrics
    dataset_metrics = defaultdict(list)
    
    # Iterate over each scene
    for scene in scene_folders:
        print(f"\nProcessing Scene: {scene}")
        scene_path = os.path.join(root_dir, scene)
        gen_path = resolve_prediction_dir(scene_path)
        gt_path = os.path.join(scene_path, 'gt')
        
        if gen_path is None:
            print(
                f"Skipping {scene}: neither 'gen' nor 'images' prediction "
                "folder exists."
            )
            continue

        if not os.path.isdir(gt_path):
            print(f"Skipping {scene}: 'gt' folder missing.")
            continue

        print(
            "Using prediction directory: "
            f"{os.path.basename(gen_path)}/"
        )
            
        # Get list of images in GT (assuming GEN matches GT filenames)
        # Using sorted glob to match 0.png, 1.png, etc.
        gt_files = sorted(glob(os.path.join(gt_path, "*.png")))
        
        if len(gt_files) == 0:
            print(f"No .png files found in {gt_path}")
            continue

        scene_metrics = defaultdict(list)
        
        # Iterate over images in the scene
        for gt_file in tqdm(gt_files, desc=f"Scene {scene}"):
            filename = os.path.basename(gt_file)
            gen_file = os.path.join(gen_path, filename)
            
            if not os.path.exists(gen_file):
                warnings.warn(f"Missing generated file: {gen_file}")
                continue
                
            # Load Images (OpenCV loads as BGR, [H, W, 3], uint8)
            gt_np = cv2.imread(gt_file)
            pred_np = cv2.imread(gen_file)
            
            # Sanity Check
            if gt_np is None or pred_np is None:
                warnings.warn(f"Could not read image: {filename}")
                continue
            
            if gt_np.shape != pred_np.shape:
                # Resize pred to match GT if necessary, or skip
                pred_np = cv2.resize(pred_np, (gt_np.shape[1], gt_np.shape[0]))

            # OpenCV loads BGR; metrics expect RGB [H, W, 3] uint8.
            gt_np = cv2.cvtColor(gt_np, cv2.COLOR_BGR2RGB)
            pred_np = cv2.cvtColor(pred_np, cv2.COLOR_BGR2RGB)

            # Calculate Metrics
            loss, lpips_val, ssim_val, psnr_val = calc_2D_metrics(pred_np, gt_np)
            
            # Store
            scene_metrics['mse'].append(loss)
            scene_metrics['lpips'].append(lpips_val)
            scene_metrics['ssim'].append(ssim_val)
            scene_metrics['psnr'].append(psnr_val)

            if yolo_annotations_root is not None:
                annotation_path = resolve_yolo_annotation(
                    yolo_annotations_root,
                    scene,
                    filename,
                )
                x1, y1, x2, y2 = load_yolo_bbox(
                    annotation_path,
                    image_width=gt_np.shape[1],
                    image_height=gt_np.shape[0],
                )
                bbox_pred_np = pred_np[y1:y2, x1:x2]
                bbox_gt_np = gt_np[y1:y2, x1:x2]
                bbox_loss, bbox_lpips, bbox_ssim, bbox_psnr = calc_2D_metrics(
                    bbox_pred_np,
                    bbox_gt_np,
                )
                scene_metrics['bbox_mse'].append(bbox_loss)
                scene_metrics['bbox_lpips'].append(bbox_lpips)
                scene_metrics['bbox_ssim'].append(bbox_ssim)
                scene_metrics['bbox_psnr'].append(bbox_psnr)

        # Calculate Scene Averages
        print(f"--- Results for {scene} ---")
        for key, vals in scene_metrics.items():
            avg = np.mean(vals)
            print(f"{key.upper()}: {avg:.4f}")
            dataset_metrics[key].extend(vals) # Add to global dataset list

    # Final Dataset Report
    print("\n" + "="*30)
    print("FINAL DATASET AVERAGES")
    print("="*30)
    if len(dataset_metrics['psnr']) > 0:
        for key, vals in dataset_metrics.items():
            print(f"Avg {key.upper()}: {np.mean(vals):.4f}")
    else:
        print("No metrics calculated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Path to the main folder containing scene folders')
    parser.add_argument(
        '--exclude-scenes',
        nargs='+',
        default=[],
        metavar='SCENE_ID',
        help='Exact scene directory name(s) to exclude from metric calculation',
    )
    parser.add_argument(
        '--yolo-annotations-root',
        type=str,
        default=None,
        help=(
            'Optional dataset root containing one annotation directory per '
            'scene. The conventional <scene>/annotation/<view_id:04d>.txt '
            'path is preferred; otherwise a unique matching TXT file is '
            'located within the scene. When omitted, only the original '
            'full-image metrics are calculated.'
        ),
    )
    args = parser.parse_args()
    main(args)
