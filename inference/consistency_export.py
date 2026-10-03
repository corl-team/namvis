"""Export raw GT depth correspondences for cross-view RGB evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _resize_crop_depth_and_alpha(
    depth: np.ndarray, alpha: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, float, float, int, int]:
    original_height, original_width = depth.shape
    if original_width / original_height <= width / height:
        resized_width = width
        resized_height = int(width / (original_width / original_height))
    else:
        resized_height = height
        resized_width = int((original_width / original_height) * height)
    crop_x = (resized_width - width) // 2
    crop_y = (resized_height - height) // 2
    if (resized_width, resized_height) != (original_width, original_height):
        size = (resized_width, resized_height)
        depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
        alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_NEAREST)
    crop = np.s_[crop_y : crop_y + height, crop_x : crop_x + width]
    return (
        depth[crop], alpha[crop],
        resized_width / original_width, resized_height / original_height,
        crop_x, crop_y,
    )


def export_consistency_data(
    scene_dir: str | Path,
    output_dir: str | Path,
    frames: list[dict],
    target_ids: list[int],
    target_poses: np.ndarray,
    normalization_scale: float,
    image_height: int,
    image_width: int,
    default_camera_angle_x: float = 0.6981317007977318,
) -> None:
    """Write GT model-space pointmaps and camera metadata for generated targets."""
    scene_dir = Path(scene_dir)
    output_dir = Path(output_dir)
    if len(target_ids) != len(target_poses):
        raise ValueError("Target IDs and poses must have the same length")
    pointmap_dir = output_dir / "gt_pointmaps"
    pointmap_dir.mkdir(parents=True, exist_ok=True)
    exported_frames = []

    for target_id, pose in zip(target_ids, target_poses):
        frame = frames[target_id]
        image_path = scene_dir / frame["file_path"]
        with Image.open(image_path) as image:
            original_width, original_height = image.size
            alpha = np.asarray(image.convert("RGBA").getchannel("A"), dtype=np.float32) / 255.0

        depth_path = scene_dir / f"depth_{Path(frame['file_path']).stem}.exr"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Missing or unreadable GT depth: {depth_path}")
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.shape != (original_height, original_width):
            raise ValueError(f"GT depth/image size mismatch for view {target_id}")
        depth, alpha, scale_x, scale_y, crop_x, crop_y = _resize_crop_depth_and_alpha(
            depth, alpha, image_width, image_height
        )

        if "K" in frame:
            intrinsics = np.asarray(frame["K"], dtype=np.float32).copy()
        else:
            angle = float(frame.get("camera_angle_x", default_camera_angle_x))
            focal = original_width / (2.0 * np.tan(angle / 2.0))
            intrinsics = np.array(
                [[focal, 0.0, original_width / 2.0],
                 [0.0, focal, original_height / 2.0],
                 [0.0, 0.0, 1.0]], dtype=np.float32
            )
        intrinsics[0, :] *= scale_x
        intrinsics[1, :] *= scale_y
        intrinsics[0, 2] -= crop_x
        intrinsics[1, 2] -= crop_y

        bounds = frame.get("depth", {})
        metric_depth = float(bounds.get("min", 1.0)) + depth * (
            float(bounds.get("max", 3.0)) - float(bounds.get("min", 1.0))
        )
        valid = (
            (alpha > 0.05) & np.isfinite(depth) & (depth >= 0.0) & (depth < 0.999)
            & np.isfinite(metric_depth) & (metric_depth > 1e-6)
        )
        y, x = np.indices((image_height, image_width), dtype=np.float32)
        xyz_camera = np.stack(
            ((x - intrinsics[0, 2]) / intrinsics[0, 0] * metric_depth,
             (y - intrinsics[1, 2]) / intrinsics[1, 1] * metric_depth,
             metric_depth),
            axis=-1,
        )
        xyz_camera[~valid] = 0.0
        xyz_model = (xyz_camera * normalization_scale) @ pose[:3, :3].T + pose[:3, 3]
        xyz_model[~valid] = 0.0
        pointmap = np.concatenate((xyz_model, valid[..., None].astype(np.float32)), axis=-1)
        np.save(pointmap_dir / f"pointmap_{target_id}.npy", np.moveaxis(pointmap, -1, 0).astype(np.float32))

        normalized_intrinsics = intrinsics.copy()
        normalized_intrinsics[0, :] /= image_width
        normalized_intrinsics[1, :] /= image_height
        exported_frames.append({
            "role": "tgt",
            "view_id": str(target_id),
            "file_path": f"gen/{target_id}.png",
            "gt_pointmap_path": f"gt_pointmaps/pointmap_{target_id}.npy",
            "transform_matrix_normalized": pose.tolist(),
            "intrinsics": normalized_intrinsics.tolist(),
        })

    metadata = {"scene_key": scene_dir.name, "format": "rgb_consistency_scene_v1", "frames": exported_frames}
    (output_dir / "transforms.json").write_text(json.dumps(metadata, indent=2) + "\n")
