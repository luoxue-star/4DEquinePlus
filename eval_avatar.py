import os
import math
import torch
import torch.nn.functional as F
import json
import numpy as np
import argparse
import shutil
from pathlib import Path
import imageio
import pickle
from PIL import Image
from typing import Dict, List, Optional
from tqdm import tqdm
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

from amr.utils import recursive_to
from amr.configs import get_config
from amr.datasets.avatar_dataset import AvatarInferDataset
from amr.models.hrm import HRM
from amr.models.hrm_mv import MVHRM
from amr.utils.camera_utils import *
from amr.utils.renderer import cam_crop_to_full
from amr.utils.evaluate_metric import Evaluator

import warnings
warnings.filterwarnings("ignore")


def as_axis_angle_sequence(value: torch.Tensor, num_frames: int, device: torch.device) -> torch.Tensor:
    value = value.to(device)
    if value.shape[-2:] == (3, 3):
        value = matrix_to_axis_angle(value.reshape(num_frames, -1, 3, 3))
    elif value.dim() == 2 and value.shape[-1] != 3 and value.shape[-1] % 6 == 0:
        value = matrix_to_axis_angle(rotation_6d_to_matrix(value.reshape(num_frames, -1, 6)))
    elif value.shape[-1] == 6:
        value = matrix_to_axis_angle(rotation_6d_to_matrix(value.reshape(num_frames, -1, 6)))
    elif value.shape[-1] == 3:
        value = value.reshape(num_frames, -1, 3)
    elif value.dim() == 2 and value.shape[-1] % 3 == 0:
        value = value.reshape(num_frames, -1, 3)
    else:
        raise ValueError(f"Cannot convert pose parameter with shape {tuple(value.shape)} to axis-angle")
    return value.unsqueeze(0)


def select_source_indices(num_frames: int, requested_count: int) -> List[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if requested_count <= 0:
        raise ValueError("requested_count must be positive")

    source_count = min(int(requested_count), int(num_frames))
    if source_count == num_frames:
        return list(range(num_frames))

    return np.linspace(0, num_frames - 1, num=source_count, dtype=int).tolist()


def build_source_item(source_items: List[Dict]) -> Dict:
    if len(source_items) == 0:
        raise ValueError("source_items must contain at least one item")

    first_item = source_items[0]
    item = {
        "source_rgbs": torch.stack([source_item["source_rgbs"] for source_item in source_items], dim=0).unsqueeze(0),
        "source_masks": torch.stack([source_item["source_masks"] for source_item in source_items], dim=0).unsqueeze(0),
        "source_valid_mask": torch.ones(1, len(source_items), dtype=torch.bool),
        "image_ori_height": torch.as_tensor(first_item["image_ori_height"]).reshape(1),
        "image_ori_width": torch.as_tensor(first_item["image_ori_width"]).reshape(1),
    }

    if "source_intrs" in first_item:
        item["source_intrs"] = torch.stack([source_item["source_intrs"] for source_item in source_items], dim=0).unsqueeze(0)

    return item


def to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def load_animation_params(sequence: str, postrefine_dir: Path) -> Dict[str, torch.Tensor]:
    animation_params_path = postrefine_dir / "refined_results.pt"
    if animation_params_path.is_file():
        return torch.load(animation_params_path, map_location='cpu', weights_only=True)

    npz_candidates = [postrefine_dir / f"{sequence}.npz", *sorted(postrefine_dir.glob("*.npz"))]
    for npz_path in npz_candidates:
        if not npz_path.is_file():
            continue
        npz_data = np.load(npz_path)
        return {
            "refined_global_orient": torch.from_numpy(npz_data["global_orient"]).float(),
            "refined_pose": torch.from_numpy(npz_data["pose"]).float(),
            "refined_betas": torch.from_numpy(npz_data["shape"]).float(),
            "refined_trans": torch.from_numpy(npz_data["transl"]).float(),
            "refined_tail_scale": torch.from_numpy(npz_data["tail_scale"]).float(),
        }

    raise FileNotFoundError(f"No refined_results.pt or {sequence}.npz found under {postrefine_dir}")


def load_track_masks(postrefine_dir: Path, num_frames: int) -> List[np.ndarray]:
    track_mask_path = postrefine_dir.parent / "mask_list.pkl"
    if track_mask_path.is_file():
        return pickle.load(open(track_mask_path, "rb"))[:num_frames]

    for mask_dir in (postrefine_dir.parent / "masks", postrefine_dir / "masks"):
        if not mask_dir.is_dir():
            continue
        mask_files = sorted(
            path for path in mask_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if len(mask_files) == 0:
            continue
        return [np.array(Image.open(mask_file).convert("L")) for mask_file in mask_files[:num_frames]]

    raise FileNotFoundError(f"No mask_list.pkl or masks directory found for {postrefine_dir}")


def compute_mask_iou(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_threshold: float = 0.5,
    gt_threshold: float = 0.5,
) -> torch.Tensor:
    pred = pred_mask.detach() > pred_threshold
    gt = gt_mask.detach() > gt_threshold

    pred = pred.reshape(pred.shape[0], -1)
    gt = gt.reshape(gt.shape[0], -1)
    intersection = (pred & gt).sum(dim=1).float()
    union = (pred | gt).sum(dim=1).float()
    frame_iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return frame_iou.mean()


class HRMEvaluator:
    def __init__(self, model: HRM, chunk_size: Optional[int] = None):
        self.model = model
        self.evaluator = Evaluator(model.renderer.varen_model.layer)
        self.chunk_size = chunk_size if chunk_size is None or chunk_size > 0 else None

    def forward_gs(self, item: Dict, source_smal_params: Dict):
        query_points, varen_data = self.model.renderer.get_query_points(source_smal_params, item['source_rgbs'].device)
        source_rgbs = item['source_rgbs']
        if isinstance(self.model, MVHRM):
            latent_points, image_feats = self.model.forward_latent_points(
                query_points,
                source_rgbs,
                valid_mask=item.get('source_valid_mask', None),
                source_masks=item.get('source_masks', None),
            )
        else:
            if source_rgbs.dim() == 5:
                source_rgbs = source_rgbs[:, 0]
            latent_points, image_feats = self.model.forward_latent_points(query_points, source_rgbs)

        preview_image = item['source_rgbs'][:, 0] if item['source_rgbs'].dim() == 5 else item['source_rgbs']
        gs_attr_list, query_points, varen_data = self.model.renderer.forward_gs(
            gs_hidden_features=latent_points,
            query_points=query_points,
            varen_data=varen_data,
            additional_features={"image_feats": image_feats, "image": preview_image},
        )
        return gs_attr_list, query_points, varen_data

    def animate_gs(self, 
                   item: Dict, 
                   animate_smal_params: Dict, 
                   img_info: Dict):
        """
        Args:
            item: Dict: input image and mask
            animate_smal_params: Dict: animate smal params
            img_info: Dict: image height, image width, focal length
        Return: 
            output: Dict: rendered results
        """
        # Acquire Canonical Space Gaussian Attributes
        gs_attr_list, query_points, varen_data = self.forward_gs(item, animate_smal_params)
        Nv = varen_data['pose'].shape[1]
        device = query_points.device
        c2w = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(Nv, 1, 1).unsqueeze(0)
        intr = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(Nv, 1, 1).unsqueeze(0)
        height = img_info['height']
        width = img_info['width']
        focal_length = img_info['focal_length']
        intr[:, :, 0, 0] = float(focal_length)
        intr[:, :, 1, 1] = float(focal_length)
        intr[:, :, 0, 2] = float(width) / 2.0
        intr[:, :, 1, 2] = float(height) / 2.0

        # Animate Gaussian
        output = self._forward_animate_with_chunking(
            gs_attr_list=gs_attr_list,
            query_points=query_points,
            varen_data=varen_data,
            c2w=c2w,
            intr=intr,
            height=height,
            width=width,
            background_color=torch.ones(size=(1, Nv, 3), device=device),
            debug=False,
        )
        output['comp_rgb'] = output['comp_rgb'][0].permute(0, 2, 3, 1)
        return output

    def _forward_animate_with_chunking(
        self,
        gs_attr_list,
        query_points,
        varen_data,
        c2w,
        intr,
        height,
        width,
        background_color,
        debug: bool = False,
    ):
        total_views = varen_data['pose'].shape[1]
        chunk_size = self.chunk_size or total_views

        if chunk_size >= total_views:
            return self.model.renderer.forward_animate_gs(
                gs_attr_list,
                query_points,
                varen_data,
                c2w,
                intr,
                height,
                width,
                background_color,
                debug=debug,
            )

        outputs = []
        for start in range(0, total_views, chunk_size):
            end = min(total_views, start + chunk_size)
            chunk_varen = self._slice_varen_data(varen_data, start, end, total_views)
            outputs.append(
                self.model.renderer.forward_animate_gs(
                    gs_attr_list,
                    query_points,
                    chunk_varen,
                    c2w[:, start:end],
                    intr[:, start:end],
                    height,
                    width,
                    background_color[:, start:end] if background_color is not None else None,
                    debug=debug,
                )
            )

        merged_output = {}
        for key in outputs[0]:
            value = outputs[0][key]
            if isinstance(value, torch.Tensor):
                merged_output[key] = torch.cat([out[key] for out in outputs], dim=1)
            elif key == '3dgs':
                merged_batches = []
                num_batches = len(value)
                for batch_idx in range(num_batches):
                    merged_batch = []
                    for out in outputs:
                        merged_batch.extend(out[key][batch_idx])
                    merged_batches.append(merged_batch)
                merged_output[key] = merged_batches
            else:
                merged_output[key] = value

        return merged_output

    def _slice_varen_data(self, varen_data: Dict, start: int, end: int, total_views: int):
        static_keys = {"betas", "tail_scale", "joint_offset", "face_offset", "transform_mat_neutral_pose"}
        chunk = {}
        chunk_views = end - start
        for key, value in varen_data.items():
            if not torch.is_tensor(value) or value.dim() < 2 or key in static_keys:
                chunk[key] = value
            elif value.shape[1] == total_views:
                chunk[key] = value[:, start:end]
            elif value.shape[1] == 1 and chunk_views > 1:
                repeat_shape = [1] * value.dim()
                repeat_shape[1] = chunk_views
                chunk[key] = value.repeat(*repeat_shape)
            else:
                chunk[key] = value
        return chunk

    def render_side_view(
        self,
        render_output: Dict,
        img_info: Dict,
        offset_multiplier: float = 1.1,
        varen_translation: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        gaussian_batches = render_output.get('3dgs', None)
        if gaussian_batches is None or len(gaussian_batches) == 0:
            return None

        if isinstance(gaussian_batches[0], list):
            gs_models: List = gaussian_batches[0]
        else:
            gs_models = gaussian_batches

        if len(gs_models) == 0:
            return None

        height = int(img_info['height'])
        width = int(img_info['width'])
        focal_length = float(img_info['focal_length'])

        device = gs_models[0].xyz.device

        centers: List[torch.Tensor] = []
        extents: List[torch.Tensor] = []
        with torch.no_grad():
            for gs_model in gs_models:
                xyz = gs_model.xyz
                centers.append(xyz.mean(dim=0))
                extent = xyz.max(dim=0).values - xyz.min(dim=0).values
                extents.append(torch.max(extent))

            max_extent = torch.stack(extents).max()
            base_distance = max(max_extent.item() * offset_multiplier, 1.0)

            trans_values: Optional[torch.Tensor] = None
            if varen_translation is not None:
                trans_values = varen_translation
                if trans_values.dim() == 3:
                    trans_values = trans_values[0]
                trans_values = trans_values.to(device)

            side_direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
            up_reference = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)

            c2w_list: List[torch.Tensor] = []
            for idx, center in enumerate(centers):
                frame_distance = base_distance
                if trans_values is not None and idx < trans_values.shape[0]:
                    depth = float(trans_values[idx, 2].abs().item())
                    if depth > 0:
                        frame_distance = max(frame_distance, depth * offset_multiplier)

                camera_position = center + side_direction * frame_distance

                forward = F.normalize(center - camera_position, dim=0, eps=1e-6)
                up = up_reference
                if torch.abs(torch.dot(forward, up)) > 0.99:
                    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

                right = torch.cross(forward, up)
                right = F.normalize(right, dim=0, eps=1e-6)
                up = torch.cross(right, forward)
                up = F.normalize(up, dim=0, eps=1e-6)

                rotation = torch.stack([right, up, forward], dim=1)
                c2w = torch.eye(4, dtype=torch.float32, device=device)
                c2w[:3, :3] = rotation
                c2w[:3, 3] = camera_position
                c2w_list.append(c2w)

            c2ws = torch.stack(c2w_list, dim=0)
            intr = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(c2ws.shape[0], 1, 1)
            intr[:, 0, 0] = focal_length
            intr[:, 1, 1] = focal_length
            intr[:, 0, 2] = float(width) / 2.0
            intr[:, 1, 2] = float(height) / 2.0

            background = torch.ones((c2ws.shape[0], 3), dtype=torch.float32, device=device)

            side_out = self.model.renderer.forward_single_batch(
                gs_models,
                c2ws,
                intr,
                height,
                width,
                background,
            )

        return side_out['comp_rgb']

    def render_free_view(
        self,
        render_output: Dict,
        img_info: Dict,
        num_views: int = 18,
        offset_multiplier: float = 1.1,
        varen_translation: Optional[torch.Tensor] = None,
        start_angle_degrees: float = 0.0,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Render a sequence of evenly spaced azimuth views around the subject.

        Args:
            render_output: Output dictionary returned by the renderer that includes `3dgs`.
            img_info: Dictionary containing `height`, `width`, and `focal_length`.
            num_views: Number of free views to render (e.g. 18 -> 20° steps, 36 -> 10° steps).
            offset_multiplier: Safety factor applied to the bounding radius for the orbit.
            varen_translation: Optional per-frame translation tensor for dynamic distance adjustment.
            start_angle_degrees: Starting azimuth angle in degrees.

        Returns:
            Dictionary containing stacked renderings keyed by render components (e.g. `comp_rgb`)
            with shape `[num_frames, num_views, ...]`, alongside auxiliary camera metadata.
        """
        gaussian_batches = render_output.get('3dgs', None)
        if gaussian_batches is None or len(gaussian_batches) == 0:
            return None
        if num_views <= 0:
            return None

        if isinstance(gaussian_batches[0], list):
            gs_models: List = gaussian_batches[0]
        else:
            gs_models = gaussian_batches

        if len(gs_models) == 0:
            return None

        height = int(img_info['height'])
        width = int(img_info['width'])
        focal_length = float(img_info['focal_length'])

        device = gs_models[0].xyz.device

        centers: List[torch.Tensor] = []
        extents: List[torch.Tensor] = []
        view_outputs: Dict[str, List[torch.Tensor]] = {}
        c2ws_per_view: List[torch.Tensor] = []
        intr_per_view: List[torch.Tensor] = []

        with torch.no_grad():
            for gs_model in gs_models:
                xyz = gs_model.xyz
                centers.append(xyz.mean(dim=0))
                extent = xyz.max(dim=0).values - xyz.min(dim=0).values
                extents.append(torch.max(extent))

            max_extent = torch.stack(extents).max()
            base_distance = max(max_extent.item() * offset_multiplier, 1.0)

            trans_values: Optional[torch.Tensor] = None
            if varen_translation is not None:
                trans_values = varen_translation
                if trans_values.dim() == 3:
                    trans_values = trans_values[0]
                trans_values = trans_values.to(device)

            up_reference = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
            intr_template = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(len(gs_models), 1, 1)
            intr_template[:, 0, 0] = focal_length
            intr_template[:, 1, 1] = focal_length
            intr_template[:, 0, 2] = float(width) / 2.0
            intr_template[:, 1, 2] = float(height) / 2.0
            background = torch.ones((len(gs_models), 3), dtype=torch.float32, device=device)

            angle_step_rad = 2.0 * math.pi / float(num_views)
            start_angle_rad = math.radians(start_angle_degrees)
            angles = torch.arange(num_views, device=device, dtype=torch.float32) * angle_step_rad + start_angle_rad
            directions = torch.stack(
                [torch.sin(angles), torch.zeros_like(angles), torch.cos(angles)],
                dim=1,
            )

            for view_idx in range(num_views):
                direction = directions[view_idx]
                c2w_list: List[torch.Tensor] = []
                for idx, center in enumerate(centers):
                    frame_distance = base_distance
                    if trans_values is not None and idx < trans_values.shape[0]:
                        depth = float(trans_values[idx, 2].abs().item())
                        if depth > 0:
                            frame_distance = max(frame_distance, depth * offset_multiplier)

                    camera_position = center + direction * frame_distance

                    forward = F.normalize(center - camera_position, dim=0, eps=1e-6)
                    up = up_reference
                    if torch.abs(torch.dot(forward, up)) > 0.99:
                        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

                    right = torch.cross(forward, up)
                    if torch.norm(right) < 1e-6:
                        right = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
                    right = F.normalize(right, dim=0, eps=1e-6)
                    up = torch.cross(right, forward)
                    up = F.normalize(up, dim=0, eps=1e-6)

                    rotation = torch.stack([right, up, forward], dim=1)
                    c2w = torch.eye(4, dtype=torch.float32, device=device)
                    c2w[:3, :3] = rotation
                    c2w[:3, 3] = camera_position
                    c2w_list.append(c2w)

                c2ws = torch.stack(c2w_list, dim=0)
                intr = intr_template.clone()

                view_out = self.model.renderer.forward_single_batch(
                    gs_models,
                    c2ws,
                    intr,
                    height,
                    width,
                    background,
                )

                for key, value in view_out.items():
                    if isinstance(value, torch.Tensor):
                        view_outputs.setdefault(key, []).append(value)

                c2ws_per_view.append(c2ws)
                intr_per_view.append(intr)

        if not view_outputs:
            return None

        stacked_outputs: Dict[str, torch.Tensor] = {}
        for key, tensors in view_outputs.items():
            stacked_outputs[key] = torch.stack(tensors, dim=1)

        c2ws_tensor = torch.stack(c2ws_per_view, dim=0).permute(1, 0, 2, 3).contiguous()
        intr_tensor = torch.stack(intr_per_view, dim=0).permute(1, 0, 2, 3).contiguous()
        view_angles_deg = (
            start_angle_degrees
            + torch.arange(num_views, device=device, dtype=torch.float32) * (360.0 / float(num_views))
        ).detach().cpu()

        stacked_outputs['c2ws'] = c2ws_tensor
        stacked_outputs['intrinsics'] = intr_tensor
        stacked_outputs['view_angles_deg'] = view_angles_deg
        return stacked_outputs

    def eval_sequence(self, sequence: str, image_dir: Path, postrefine_dir: Path):
        # --- Load Animation Params ---
        animation_params = load_animation_params(sequence, postrefine_dir)
        for k, v in animation_params.items():
            if "global_orient" in k:
                N = v.shape[0]
                break

        def process_params(params: Dict) -> Dict:
            smal_params = {}
            for k, v in params.items():
                if "global_orient" in k:
                    smal_params['global_orient'] = as_axis_angle_sequence(v, N, torch.device(args.device))
                elif "pose" in k:
                    smal_params['pose'] = as_axis_angle_sequence(v, N, torch.device(args.device))
                elif "betas" in k:
                    betas = v.to(args.device)
                    smal_params['betas'] = betas.unsqueeze(0) if betas.dim() == 1 else betas.mean(dim=0, keepdim=True)
                elif "tail_scale" in k:
                    tail_scale = v.to(args.device)
                    smal_params['tail_scale'] = (
                        tail_scale.unsqueeze(0) if tail_scale.dim() == 1 else tail_scale.mean(dim=0, keepdim=True)
                    )
                elif "cam_t" in k or "transl" in k or "trans" in k:
                    trans = v.to(args.device)
                    if trans.dim() == 1:
                        trans = trans.reshape(1, 1, -1)
                    elif trans.dim() == 2:
                        trans = trans.unsqueeze(0)
                    if trans.shape[-1] != 3:
                        raise ValueError(f"Expected translation parameters with shape [N, 3], got {tuple(trans.shape)}")
                    smal_params['trans'] = trans[:, :N]
                elif "cam" in k and "cam_t" not in k:
                    cam = v.to(args.device)
                    if cam.dim() == 1:
                        cam = cam.reshape(1, -1)
                    if cam.shape[-1] == 1:
                        zeros = torch.zeros(*cam.shape[:-1], 2, device=cam.device, dtype=cam.dtype)
                        cam = torch.cat([cam, zeros], dim=-1)
                    if cam.shape[-1] != 3:
                        raise ValueError(f"Expected camera parameters with shape [N, 3], got {tuple(cam.shape)}")
                    smal_params['cam'] = cam[:N]

            for k, v in smal_params.items():
                if k not in ("betas", "tail_scale", "cam", "trans"):
                    smal_params[k] = v[:, :N]
            return smal_params
        
        animation_params = process_params(animation_params)
        # --- Load Animation Params ---

        # --- Load Original Images & Masks ---
        out_dir = os.path.join(args.out_folder, sequence)
        os.makedirs(out_dir, exist_ok=True)
        original_image_files = sorted(os.listdir(image_dir))
        img_path = image_dir / original_image_files[0]
        shutil.copy(img_path, os.path.join(out_dir, str(0).zfill(6) + ".jpg"))
        # Obtain Frame and Segmetation Mask
        frames = []
        for i in range(N):
            frame = np.array(Image.open(image_dir / original_image_files[i]))
            frames.append(frame)
        track_masks = load_track_masks(postrefine_dir, N)  # [N, H, W]
        # --- Load Original Images & Masks ---

        # --- Forward GS ---
        dataset = AvatarInferDataset(image=frames, mask=track_masks, intrs=[None] * N, render_image_res=448)
        source_indices = select_source_indices(len(dataset), args.num_input_images)
        source_items = [dataset[i] for i in source_indices]
        item = build_source_item(source_items)
        item = recursive_to(item, args.device)

        video_path = os.path.join(out_dir, "video.mp4")
        writer = imageio.get_writer(video_path, fps=10, macro_block_size=1)
        psnr_sum, ssim_sum, lpips_sum, iou_sum, metric_frames = 0.0, 0.0, 0.0, 0.0, 0

        try:
            with torch.no_grad():
                # Animate Gaussian
                # Compute translation for Original images
                scaled_focal_length = 5000 / 256 * max(item['image_ori_height'].item(), item['image_ori_width'].item())
                if 'cam' in animation_params:
                    box_center, box_size, img_size = [], [], []  # [N, 2], [N], [N, 2]
                    for i in range(N):
                        y, x = np.where(track_masks[i] > 0)
                        x1, y1, x2, y2 = x.min(), y.min(), x.max(), y.max()
                        box_center.append([(x1 + x2) // 2, (y1 + y2) // 2])
                        box_size.append(max(x2 - x1, y2 - y1))
                        img_size.append([item['image_ori_width'].item(), item['image_ori_height'].item()])
                    box_center = torch.tensor(box_center)
                    box_size = torch.tensor(box_size)
                    img_size = torch.tensor(img_size)

                    pred_cam_t_full = cam_crop_to_full(animation_params['cam'].cpu()[:N],
                                                    box_center,
                                                    box_size,
                                                    img_size,
                                                    scaled_focal_length)
                    animation_params['trans'] = pred_cam_t_full.unsqueeze(0).to(args.device)
                elif 'trans' in animation_params:
                    animation_params['trans'] = animation_params['trans'][:, :N].to(args.device)
                else:
                    raise ValueError("Animation params must contain either weak-perspective cam or full translation trans.")
                img_info = {
                    "height": item['image_ori_height'].item(),
                    "width": item['image_ori_width'].item(),
                    "focal_length": scaled_focal_length,
                }

                gs_attr_list, query_points, varen_data = self.forward_gs(item, animation_params)
                Nv = N
                device = query_points.device
                c2w = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(Nv, 1, 1).unsqueeze(0)
                intr = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(Nv, 1, 1).unsqueeze(0)
                intr[:, :, 0, 0] = float(img_info["focal_length"])
                intr[:, :, 1, 1] = float(img_info["focal_length"])
                intr[:, :, 0, 2] = float(img_info["width"]) / 2.0
                intr[:, :, 1, 2] = float(img_info["height"]) / 2.0
                background_color = torch.ones(size=(1, Nv, 3), device=device)
                chunk_size = self.chunk_size or Nv

                for start in range(0, Nv, chunk_size):
                    end = min(Nv, start + chunk_size)
                    chunk_varen = self._slice_varen_data(varen_data, start, end, Nv)
                    render_output = self.model.renderer.forward_animate_gs(
                        gs_attr_list,
                        query_points,
                        chunk_varen,
                        c2w[:, start:end],
                        intr[:, start:end],
                        img_info["height"],
                        img_info["width"],
                        background_color[:, start:end],
                        debug=False,
                    )
                    pred_rgb = render_output['comp_rgb'][0].permute(0, 2, 3, 1).detach()
                    pred_mask = render_output['comp_mask'].detach()[0, :, 0]

                    side_rgb = None
                    if args.render_side_view:
                        side_rgb = self.render_side_view(
                            render_output,
                            img_info,
                            varen_translation=animation_params.get('trans', None)[:, start:end],
                        )
                        side_rgb = side_rgb.detach() if side_rgb is not None else None
                    else:
                        render_output.pop('3dgs', None)

                    gt_rgb = []
                    gt_masks = []
                    for frame_idx in range(start, end):
                        mask = (track_masks[frame_idx] > 128).astype(np.float32)
                        rgb = frames[frame_idx] / 255.0 * mask[:, :, None] + 1.0 * (1 - mask[:, :, None])
                        gt_rgb.append(rgb)
                        gt_masks.append(mask)
                    gt_rgb = torch.from_numpy(np.array(gt_rgb)).to(pred_rgb.device, dtype=pred_rgb.dtype)
                    gt_masks = torch.from_numpy(np.array(gt_masks)).to(pred_mask.device, dtype=pred_mask.dtype)
                    assert pred_rgb.shape == gt_rgb.shape
                    assert pred_mask.shape == gt_masks.shape

                    metric_start = max(start, N // 2)
                    if metric_start < end:
                        metric_offset = metric_start - start
                        metric_count = end - metric_start
                        pred_rgb_metric = pred_rgb[metric_offset:].permute(0, 3, 1, 2).float()
                        gt_rgb_metric = gt_rgb[metric_offset:].permute(0, 3, 1, 2).float()
                        psnr, ssim, lpips = self.evaluator._compute_image_metrics(pred_rgb_metric, gt_rgb_metric)
                        iou = compute_mask_iou(pred_mask[metric_offset:], gt_masks[metric_offset:])
                        psnr_sum += float(psnr) * metric_count
                        ssim_sum += float(ssim) * metric_count
                        lpips_sum += float(lpips) * metric_count
                        iou_sum += float(iou) * metric_count
                        metric_frames += metric_count
                        del pred_rgb_metric, gt_rgb_metric, psnr, ssim, lpips, iou

                    pred_rgb_cpu = pred_rgb.cpu()
                    gt_rgb_cpu = gt_rgb.cpu()
                    side_rgb_cpu = side_rgb.cpu() if side_rgb is not None else None
                    for chunk_idx in range(end - start):
                        column_tensors = [pred_rgb_cpu[chunk_idx]]
                        if side_rgb_cpu is not None:
                            column_tensors.append(side_rgb_cpu[chunk_idx])
                        column_tensors.append(gt_rgb_cpu[chunk_idx])
                        frame_tensor = torch.cat(column_tensors, dim=1)
                        frame = (frame_tensor * 255.0).clamp(0, 255).to(torch.uint8).numpy()
                        writer.append_data(frame)

                    del render_output, pred_rgb, pred_mask, gt_rgb, gt_masks
                    del pred_rgb_cpu, gt_rgb_cpu, side_rgb, side_rgb_cpu
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                del gs_attr_list, query_points, varen_data, c2w, intr, background_color
            # --- Forward GS ---
        finally:
            writer.close()

        if metric_frames > 0:
            psnr = psnr_sum / metric_frames
            ssim = ssim_sum / metric_frames
            lpips = lpips_sum / metric_frames
            iou = iou_sum / metric_frames
        else:
            psnr, ssim, lpips, iou = 0.0, 0.0, 0.0, 0.0
        return dict(PSNR=psnr, SSIM=ssim, LPIPS=lpips, IOU=iou, Num_Frames=metric_frames)


def main(args):
    # Load Horse Reconstruction Model
    model_cfg = str(Path(args.checkpoint).parent.parent / '.hydra/config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    use_multiview = model_cfg.MODEL.BACKBONE.get("USE_MULTIVIEW", False)
    if args.num_input_images > 1 and not use_multiview:
        print("Warning: checkpoint is not a multi-view model; only the first input image will be used.")
    ModelClass = MVHRM if use_multiview else HRM
    model = ModelClass.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        cfg=model_cfg,
        strict=False,
        weights_only=False,
    )
    model = model.to(args.device)
    model = model.eval()

    # Create Evaluator
    evaluator = HRMEvaluator(model, chunk_size=args.chunk_size)
    image_root = Path(args.image_dir)
    postrefine_root = Path(args.postrefine_dir)
    results = {}
    sequences = sorted(path.name for path in image_root.iterdir() if path.is_dir())
    for sequence in tqdm(sequences, desc="Evaluating sequences", total=len(sequences)):
        image_dir = first_existing(image_root / sequence / "images", image_root / sequence)
        postrefine_dir = first_existing(postrefine_root / sequence / "postrefine", postrefine_root / sequence)
        try:
            results[sequence] = evaluator.eval_sequence(sequence, image_dir, postrefine_dir)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Compute Average Metrics
    metrics_dict = {}
    psnr, ssim, lpips, iou, num_frames = 0.0, 0.0, 0.0, 0.0, 0
    for seq, metrics in results.items():
        metrics_serializable = to_serializable(metrics)
        frames = int(metrics_serializable.get('Num_Frames', 0))
        psnr += float(metrics_serializable['PSNR']) * frames
        ssim += float(metrics_serializable['SSIM']) * frames
        lpips += float(metrics_serializable['LPIPS']) * frames
        iou += float(metrics_serializable['IOU']) * frames
        num_frames += frames
        metrics_dict[seq] = metrics_serializable

    if num_frames > 0:
        psnr /= num_frames
        ssim /= num_frames
        lpips /= num_frames
        iou /= num_frames
    print(f"Average PSNR: {psnr:.4f}, Average SSIM: {ssim:.4f}, Average LPIPS: {lpips:.4f}, Average IOU: {iou:.4f}, Num Frames: {num_frames}")

    with open(os.path.join(args.out_folder, "metrics.json"), 'w') as f:
        json.dump(metrics_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Horse Avatar Reconstruction Evaluation Code')
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument("--out_folder", type=str, default="outputs/", help="Path to the output folder")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the image folder")
    parser.add_argument("--postrefine_dir", type=str, required=True, help="Path to the postrefine folder")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Max frames to render per chunk. Defaults to the full video length.",
    )
    parser.add_argument(
        "--num_input_images",
        type=int,
        default=1,
        help="Number of source images to use as input. Clamped to the number of available images.",
    )
    parser.add_argument(
        "--render_side_view",
        action="store_true",
        help="Also render side-view video columns. Disabled by default to reduce evaluation memory.",
    )
    args = parser.parse_args()
    main(args)
