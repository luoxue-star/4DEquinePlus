#!/usr/bin/env python3
import argparse
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypeVar

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
DEFAULT_STRIP_SUFFIXES = ("_left180",)

T = TypeVar("T")


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0


def masks_for_refine(
    animer_outputs: Dict[str, torch.Tensor],
    raw_masks: Sequence[np.ndarray],
    image_size: int,
) -> np.ndarray:
    if "mask" in animer_outputs:
        mask_tensor = animer_outputs["mask"]
        if torch.is_tensor(mask_tensor):
            return mask_tensor.detach().cpu().numpy()
        return np.asarray(mask_tensor)

    resized_masks = []
    for mask in raw_masks:
        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        mask_img = Image.fromarray(mask_arr.astype(np.uint8))
        mask_img = mask_img.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
        resized_masks.append(np.asarray(mask_img))
    return np.stack(resized_masks, axis=0)


def refined_cam_to_full_frame_transl(
    refined_cam: torch.Tensor,
    animer_outputs: Dict[str, torch.Tensor],
    full_image_size: Sequence[int],
    focal_length: float,
    model_image_size: int,
) -> np.ndarray:
    required = ("box_center", "bbox_size")
    missing = [key for key in required if key not in animer_outputs]
    if missing:
        raise KeyError(
            f"Cannot export full-frame camera translation; animer_outputs is missing {missing}. "
            "Re-run stage1 or keep intermediate animer_outputs.pt."
        )

    cam = refined_cam.detach().cpu().float()
    num_frames = int(cam.shape[0])
    box_center = animer_outputs["box_center"][:num_frames].detach().cpu().float()
    bbox_size = animer_outputs["bbox_size"][:num_frames].detach().cpu().float().reshape(-1)

    width, height = float(full_image_size[0]), float(full_image_size[1])
    img_size = torch.tensor([[width, height]], dtype=cam.dtype).repeat(num_frames, 1)
    scaled_focal = float(focal_length) / float(model_image_size) * max(width, height)

    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy = box_center[:, 0], box_center[:, 1]
    half_w, half_h = img_w / 2.0, img_h / 2.0
    bs = bbox_size * cam[:, 0] + 1e-9
    tz = 2.0 * scaled_focal / bs
    tx = (2.0 * (cx - half_w) / bs) + cam[:, 1]
    ty = (2.0 * (cy - half_h) / bs) + cam[:, 2]
    return torch.stack([tx, ty, tz], dim=-1).numpy().astype(np.float32)


def export_varen_npz(
    refined_path: Path,
    output_path: Path,
    *,
    animer_outputs: Optional[Dict[str, torch.Tensor]] = None,
    full_image_size: Optional[Sequence[int]] = None,
    focal_length: float = 5000.0,
    model_image_size: int = 256,
) -> None:
    from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix

    refined = torch.load(refined_path, map_location="cpu", weights_only=True)
    pose_6d = refined["refined_pose"]
    global_orient_6d = refined["refined_global_orient"]
    num_frames = pose_6d.shape[0]

    pose = matrix_to_axis_angle(rotation_6d_to_matrix(pose_6d.reshape(num_frames, -1, 6))).numpy()
    global_orient = matrix_to_axis_angle(
        rotation_6d_to_matrix(global_orient_6d.reshape(num_frames, -1, 6))
    ).reshape(num_frames, -1, 3)[:, 0].numpy()
    shape = refined["refined_betas"].mean(dim=0).numpy().astype(np.float32)
    if animer_outputs is not None and full_image_size is not None:
        transl = refined_cam_to_full_frame_transl(
            refined["refined_cam"],
            animer_outputs=animer_outputs,
            full_image_size=full_image_size,
            focal_length=focal_length,
            model_image_size=model_image_size,
        )
    else:
        transl = refined["refined_cam_t"].numpy().astype(np.float32)
    tail_scale = refined["refined_tail_scale"].mean(dim=0).numpy().astype(np.float32)

    np.savez(
        output_path,
        pose=pose.astype(np.float32),
        global_orient=global_orient.astype(np.float32),
        shape=shape,
        transl=transl,
        tail_scale=tail_scale,
    )


def cleanup_intermediate(output_dir: Path) -> None:
    for name in ("animer_outputs.pt", "vitpose_results.pkl", "mask_list.pkl"):
        path = output_dir / name
        if path.exists():
            path.unlink()
    shutil.rmtree(output_dir / "postrefine", ignore_errors=True)


def build_cached_sam3_pipeline(args):
    from transformers import Sam3Model, Sam3Processor, Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

    from amr.models.pre_post_optimization import PostProcessPipeline
    from amr.utils.track_bboxes import detect_bboxes_using_sam3, load_data, track_bboxes_using_sam3

    class CachedSam3PostProcessPipeline(PostProcessPipeline):
        def __init__(self, *pipeline_args, **pipeline_kwargs):
            super().__init__(*pipeline_args, **pipeline_kwargs)
            self._sam3_model = None
            self._sam3_processor = None
            self._sam3_tracker_model = None
            self._sam3_tracker_processor = None

        def _ensure_sam3_detector(self):
            if self._sam3_model is None or self._sam3_processor is None:
                self._sam3_model = Sam3Model.from_pretrained("data/sam3").to(self.device)
                self._sam3_processor = Sam3Processor.from_pretrained("data/sam3")
            return self._sam3_model, self._sam3_processor

        def _ensure_sam3_tracker(self):
            if self._sam3_tracker_model is None or self._sam3_tracker_processor is None:
                tracker = Sam3TrackerVideoModel.from_pretrained("data/sam3")
                device_str = str(self.device)
                if device_str.startswith("cuda"):
                    tracker = tracker.to(self.device, dtype=torch.bfloat16)
                else:
                    tracker = tracker.to(self.device)
                self._sam3_tracker_model = tracker
                self._sam3_tracker_processor = Sam3TrackerVideoProcessor.from_pretrained("data/sam3")
            return self._sam3_tracker_model, self._sam3_tracker_processor

        def _detect_init_bbox(self, video_path: str):
            detector, processor = self._ensure_sam3_detector()
            boxes, scores, frames = detect_bboxes_using_sam3(
                video_path,
                prompt=self.prompt,
                device=self.device,
                model=detector,
                processor=processor,
            )
            if boxes is None or len(boxes) == 0:
                raise ValueError(f"No detections found by SAM3 on the first frame for prompt: {self.prompt}")

            scores = np.asarray(scores).reshape(-1)
            boxes = np.asarray(boxes)
            if boxes.ndim == 1:
                boxes = boxes.reshape(1, -1)

            valid_indices = np.where(scores >= self.detection_score_thresh)[0]
            if valid_indices.size == 0:
                valid_indices = np.arange(scores.shape[0])

            best_idx = int(valid_indices[np.argmax(scores[valid_indices])])
            return boxes[best_idx].astype(np.float32), frames

        def _track_with_sam3(self, video_path: str, init_bbox=None):
            if init_bbox is None:
                init_bbox, frames_np = self._detect_init_bbox(video_path)
            else:
                frames_np = load_data(video_path)

            if frames_np is None or len(frames_np) == 0:
                raise ValueError("No frames loaded from the given video path.")

            tracker, processor = self._ensure_sam3_tracker()
            raw_valid_idx, raw_bboxes, raw_masks = track_bboxes_using_sam3(
                np.asarray(init_bbox, dtype=np.float32).reshape(1, 4),
                frames_np,
                device=self.device,
                tracker=tracker,
                processor=processor,
            )

            target_id = 0
            valid_idx_list = raw_valid_idx.get(target_id, [])
            bbox_list = raw_bboxes.get(target_id, [])
            mask_entries = raw_masks.get(target_id, [])

            num_frames = frames_np.shape[0]
            height, width = frames_np.shape[1:3]
            all_bboxes = np.zeros((num_frames, 4), dtype=np.float32)
            full_masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(num_frames)]

            for fid, bbox in zip(valid_idx_list, bbox_list):
                all_bboxes[int(fid)] = np.asarray(bbox, dtype=np.float32)
            for fid, mask in zip(valid_idx_list, mask_entries):
                full_masks[int(fid)] = np.asarray(mask, dtype=np.uint8)

            return (
                np.array(valid_idx_list, dtype=np.int32),
                all_bboxes,
                np.stack(full_masks, axis=0),
            )

    return CachedSam3PostProcessPipeline(
        device=args.device,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch process videos into the VarenTexPP/VarenWan sequence layout. "
            "The SAM3 detector and tracker checkpoints are loaded once and reused "
            "for all videos processed by this script."
        )
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="outputs/wan_horse",
        help="Video file or directory containing videos.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/VarenWanHorse",
        help="Destination root. Each video becomes one sample subdirectory.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="AniMerVAREN checkpoint.")
    parser.add_argument("--cfg", type=str, default=None, help="AniMerVAREN config yaml.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", type=str, default="horse", help="SAM3 text prompt for detection.")
    parser.add_argument("--name_prefix", type=str, default="horse", help="Prefix for output sample folders.")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument(
        "--strip_suffix",
        action="append",
        default=list(DEFAULT_STRIP_SUFFIXES),
        help="Suffix to strip from video stems before adding --name_prefix. Can be passed multiple times.",
    )
    parser.add_argument("--recursive", action="store_true", help="Collect videos recursively under --input_root.")
    parser.add_argument("--stage1", action="store_true", help="Run detection, pose, and AniMerVAREN.")
    parser.add_argument("--stage2", action="store_true", help="Run refinement from saved stage-1 outputs.")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples whose final .npz already exists.",
    )
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="Keep animer_outputs.pt, vitpose_results.pkl, and mask_list.pkl.",
    )
    parser.add_argument(
        "--save_refine_overlap",
        action="store_true",
        help="Save a refined mesh overlap video at 16 FPS under debug/refine_overlap.mp4.",
    )
    return parser.parse_args()


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def collect_video_paths(input_root: Path, recursive: bool = False) -> List[Path]:
    if is_video_file(input_root):
        return [input_root]
    if not input_root.is_dir():
        raise ValueError(f"Input path is neither a video nor a directory: {input_root}")

    iterator = input_root.rglob("*") if recursive else input_root.iterdir()
    video_paths = sorted(path for path in iterator if is_video_file(path))
    if not video_paths:
        raise ValueError(f"No videos found in: {input_root}")
    return video_paths


def strip_known_suffixes(stem: str, strip_suffixes: Sequence[str]) -> str:
    for suffix in strip_suffixes:
        if suffix and stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def output_sample_name(video_path: Path, prefix: str = "horse", strip_suffixes: Sequence[str] = DEFAULT_STRIP_SUFFIXES) -> str:
    stem = strip_known_suffixes(video_path.stem, strip_suffixes)
    return f"{prefix}_{stem}" if prefix else stem


def shard_sequence(items: Sequence[T], rank: int, world_size: int) -> List[T]:
    if world_size <= 1:
        return list(items)
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid distributed rank {rank} for world_size {world_size}")
    return list(items)[rank::world_size]


def configure_distributed_from_env(
    args,
    env: Optional[Dict[str, str]] = None,
    set_cuda_device: bool = True,
) -> DistributedContext:
    env = os.environ if env is None else env
    world_size = int(env.get("WORLD_SIZE", "1"))
    rank = int(env.get("RANK", "0"))
    local_rank = int(env.get("LOCAL_RANK", "0"))
    context = DistributedContext(
        enabled=world_size > 1,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
    if not context.enabled:
        return context

    device = str(getattr(args, "device", "cuda"))
    if device == "cuda" or device.startswith("cuda:"):
        args.device = f"cuda:{local_rank}"
        if set_cuda_device and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return context


def remap_vitpose_results(
    vitpose_results: Dict[int, Dict[str, np.ndarray]],
    frame_indices: Sequence[int],
) -> Dict[int, Dict[str, np.ndarray]]:
    remapped = {}
    for new_idx, frame_idx in enumerate(frame_indices):
        if int(frame_idx) in vitpose_results:
            remapped[new_idx] = vitpose_results[int(frame_idx)]
    return remapped


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    return (mask_arr > 0).astype(np.uint8) * 255


def save_dataset_frames_and_masks(
    frames_rgb: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    output_dir: Path,
) -> None:
    if len(frames_rgb) != len(masks):
        raise ValueError(f"Frame/mask count mismatch: {len(frames_rgb)} frames vs {len(masks)} masks")

    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame in enumerate(frames_rgb):
        frame_arr = np.asarray(frame)
        if frame_arr.ndim != 3 or frame_arr.shape[2] < 3:
            raise ValueError(f"Expected RGB frame at index {idx}, got shape {frame_arr.shape}")
        Image.fromarray(frame_arr[..., :3].astype(np.uint8), mode="RGB").save(
            image_dir / f"{idx:04d}.jpg",
            quality=95,
        )

    for idx, mask in enumerate(masks):
        Image.fromarray(normalize_mask(mask), mode="L").save(mask_dir / f"{idx:04d}.png")


def load_video_frames_rgb(video_path: Path, frame_indices: Sequence[int]) -> List[np.ndarray]:
    import cv2

    wanted = {int(idx) for idx in frame_indices}
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx in wanted:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            frame_idx += 1
    finally:
        cap.release()

    if len(frames) != len(frame_indices):
        raise ValueError(f"Loaded {len(frames)} frames but expected {len(frame_indices)} from {video_path}")
    return frames


def masks_for_frame_indices(mask_list: Sequence[np.ndarray], frame_indices: Sequence[int]) -> List[np.ndarray]:
    masks = list(mask_list)
    if not masks:
        raise ValueError("Cannot export dataset masks because stage1 produced no masks.")
    if max(frame_indices, default=-1) < len(masks):
        return [np.asarray(masks[int(frame_idx)], dtype=np.uint8) for frame_idx in frame_indices]
    if len(masks) == len(frame_indices):
        return [np.asarray(mask, dtype=np.uint8) for mask in masks]
    raise ValueError(f"Cannot align {len(masks)} masks to frame indices ending at {max(frame_indices)}")


def infer_output_frame_indices(valid_idx: np.ndarray, animer_outputs: Dict[str, Any]) -> List[int]:
    if "pred_cam" not in animer_outputs:
        raise ValueError("AniMerVAREN outputs are missing pred_cam.")
    pred_len = int(animer_outputs["pred_cam"].shape[0])
    valid_idx_list = [int(idx) for idx in np.asarray(valid_idx).reshape(-1).tolist()]
    if pred_len != len(valid_idx_list):
        raise ValueError(
            f"Cannot align AniMerVAREN outputs ({pred_len} frames) with valid_idx ({len(valid_idx_list)} frames)."
        )
    return valid_idx_list


def save_stage1_outputs(
    output_dir: Path,
    animer_outputs: Dict[str, Any],
    vitpose_results: Dict[int, Dict[str, np.ndarray]],
    masks: Sequence[np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(animer_outputs, output_dir / "animer_outputs.pt")
    with open(output_dir / "vitpose_results.pkl", "wb") as f:
        pickle.dump(vitpose_results, f)
    with open(output_dir / "mask_list.pkl", "wb") as f:
        pickle.dump(list(masks), f)


def is_segmentation_failure(exc: Exception) -> bool:
    message = str(exc)
    return "No detections found by SAM3" in message or "No valid detection" in message


def run_stage1(args, video_path: Path, output_dir: Path, pipeline: Any) -> bool:
    valid_idx, detected_bboxes, mask_list = pipeline.track_bounding_boxes(str(video_path))
    if len(valid_idx) == 0:
        print(f"Skipping {video_path.name}: no valid detection frames")
        return False

    vitpose_results = pipeline.infer_vitposepp(str(video_path), valid_idx, detected_bboxes, return_pose_image=False)
    animer_outputs = pipeline.infer_animer_varen(str(video_path), valid_idx, detected_bboxes, mask_list)
    if animer_outputs is None:
        print(f"Skipping {video_path.name}: AniMerVAREN produced no outputs")
        return False

    frame_indices = infer_output_frame_indices(valid_idx, animer_outputs)
    remapped_vitpose_results = remap_vitpose_results(vitpose_results, frame_indices)
    masks = masks_for_frame_indices(mask_list, frame_indices)
    frames = load_video_frames_rgb(video_path, frame_indices)

    save_stage1_outputs(output_dir, animer_outputs, remapped_vitpose_results, masks)
    save_dataset_frames_and_masks(frames, masks, output_dir)
    return True


def load_stage1_for_refine(output_dir: Path, image_size: int):
    from amr.datasets.utils import trans_point2d

    animer_outputs = torch.load(output_dir / "animer_outputs.pt", map_location="cpu", weights_only=True)
    with open(output_dir / "vitpose_results.pkl", "rb") as f:
        vitpose_results = pickle.load(f)
    with open(output_dir / "mask_list.pkl", "rb") as f:
        masks = pickle.load(f)

    vitpose_results = {
        k: trans_point2d(v["keypoints"][0], animer_outputs["trans"][k])[None]
        for k, v in vitpose_results.items()
        if "keypoints" in v and len(v["keypoints"]) > 0
    }
    refine_masks = masks_for_refine(animer_outputs, masks, image_size=image_size)
    valid_idx = np.arange(animer_outputs["pred_cam"].shape[0], dtype=np.int32)
    return animer_outputs, vitpose_results, refine_masks, valid_idx


@torch.no_grad()
def save_refined_overlap_video(
    refiner: Any,
    animer_outputs: Dict[str, torch.Tensor],
    output_path: Path,
    fps: int = 16,
) -> Path:
    import imageio.v2 as imageio
    from amr.utils.renderer import Renderer
    from pytorch3d.transforms import rotation_6d_to_matrix

    img_seq = animer_outputs.get("img")
    if not isinstance(img_seq, torch.Tensor):
        raise ValueError("Cannot save refined overlap video because animer_outputs['img'] is missing.")

    total_frames = min(int(refiner.param_global_orient.shape[0]), int(img_seq.shape[0]))
    if total_frames <= 0:
        raise ValueError("Cannot save refined overlap video because there are no frames.")

    frame_ids = list(range(total_frames))
    frame_index = torch.as_tensor(frame_ids, device=refiner.device, dtype=torch.long)
    num_frames = len(frame_ids)

    focal_default = float(getattr(refiner.cfg.EXTRA, "FOCAL_LENGTH", 5000))
    selected_cam = refiner.param_cam.index_select(0, frame_index)
    focal_px = torch.tensor(
        [focal_default, focal_default],
        device=refiner.device,
        dtype=selected_cam.dtype,
    ).unsqueeze(0).repeat(num_frames, 1)

    smal_output = refiner.varen(
        global_orient=rotation_6d_to_matrix(
            refiner.param_global_orient.index_select(0, frame_index).view(num_frames, -1, 6)
        ),
        pose=rotation_6d_to_matrix(refiner.param_pose.index_select(0, frame_index).view(num_frames, -1, 6)),
        betas=refiner.param_betas.mean(dim=0, keepdim=True).expand(num_frames, -1),
        tail_scale=refiner.param_tail_scale.mean(dim=0, keepdim=True).expand(num_frames, -1),
        pose2rot=False,
    )
    pred_cam_t = torch.stack(
        [
            selected_cam[:, 1],
            selected_cam[:, 2],
            2 * focal_px[:, 0] / (refiner.cfg.MODEL.IMAGE_SIZE * selected_cam[:, 0] + 1e-9),
        ],
        dim=-1,
    )

    renderer = Renderer(refiner.cfg, faces=refiner.faces)
    video_frames = []
    for local_idx, frame_id in enumerate(tqdm(frame_ids, desc="Saving refined overlap video", dynamic_ncols=True)):
        overlay_rgb = renderer(
            smal_output.vertices[local_idx].detach().cpu().numpy(),
            pred_cam_t[local_idx].detach().cpu().numpy(),
            img_seq[int(frame_id)],
            mesh_base_color=(0.65098039, 0.74117647, 0.85882353),
            scene_bg_color=(1, 1, 1),
        )
        video_frames.append((overlay_rgb * 255).astype(np.uint8))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), video_frames, fps=fps, macro_block_size=1)
    return output_path


def run_stage2(args, output_dir: Path) -> Path:
    from amr.configs import get_config
    from amr.models.post_optimization import PostProcessRefiner

    if args.cfg is None:
        raise ValueError("--cfg is required for --stage2")

    cfg = get_config(args.cfg)
    animer_outputs, vitpose_results, masks, valid_idx = load_stage1_for_refine(output_dir, image_size=args.image_size)
    refiner = PostProcessRefiner(cfg, animer_outputs, device=args.device)
    derived_opt = refiner.optimize_parameters(
        keypoints_results=vitpose_results,
        mask_list=masks,
        valid_idx=valid_idx,
        image_size=args.image_size,
        num_iters=[75, 100],
        lr=[5e-3, 5e-3],
        chunk_size=args.chunk_size,
        w_silhouette=[100.0, 10000.0],
        w_keypoints=[10000.0, 100.0],
        w_temporal=[0.0, 0.0],
        w_reg=[1000.0, 300.0],
        w_pose_prior=[1000.0, 300.0],
        params_per_stage=[
            ["pose", "betas", "cam", "tail_scale", "global_orient"],
            ["tail_scale", "betas"],
        ],
        output_dir=str(output_dir),
        render_every=None,
        use_adaptive_lr=True,
        lr_scheduler_patience=20,
        lr_scheduler_factor=0.5,
    )

    postrefine_dir = output_dir / "postrefine"
    refiner.save_refined_results(
        output_dir=str(postrefine_dir),
        derived=derived_opt,
        image_size=args.image_size,
        chunk_size=args.chunk_size,
    )
    refined_path = postrefine_dir / "refined_results.pt"
    first_image_path = output_dir / "images" / "0000.jpg"
    if not first_image_path.exists():
        raise FileNotFoundError(f"Cannot export full-frame camera because image is missing: {first_image_path}")
    with Image.open(first_image_path) as image:
        full_image_size = image.size  # (width, height)
    export_varen_npz(
        refined_path,
        output_dir / f"{output_dir.name}.npz",
        animer_outputs=animer_outputs,
        full_image_size=full_image_size,
        focal_length=float(getattr(cfg.EXTRA, "FOCAL_LENGTH", 5000)),
        model_image_size=int(getattr(cfg.MODEL, "IMAGE_SIZE", args.image_size)),
    )
    if args.save_refine_overlap:
        save_refined_overlap_video(
            refiner=refiner,
            animer_outputs=animer_outputs,
            output_path=output_dir / "debug" / "refine_overlap.mp4",
        )
    return refined_path


def remove_sample_output(output_dir: Path) -> None:
    shutil.rmtree(output_dir, ignore_errors=True)


def main(args) -> None:
    if not args.stage1 and not args.stage2:
        args.stage1 = True
        args.stage2 = True
    if args.stage1 and args.checkpoint is None:
        raise ValueError("--checkpoint is required for --stage1")

    dist = configure_distributed_from_env(args)

    video_paths = collect_video_paths(Path(args.input_root), recursive=args.recursive)
    video_paths = video_paths[args.start_idx : args.end_idx]
    total_videos = len(video_paths)
    video_paths = shard_sequence(video_paths, rank=dist.rank, world_size=dist.world_size)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if dist.enabled:
        print(
            f"[rank {dist.rank}/{dist.world_size} local_rank={dist.local_rank}] "
            f"device={args.device}, assigned {len(video_paths)}/{total_videos} videos"
        )

    pipeline = build_cached_sam3_pipeline(args) if args.stage1 else None

    for video_path in tqdm(video_paths, desc="Processing videos", dynamic_ncols=True):
        sample_name = output_sample_name(video_path, prefix=args.name_prefix, strip_suffixes=args.strip_suffix)
        output_dir = output_root / sample_name
        npz_path = output_dir / f"{sample_name}.npz"
        if args.skip_existing and npz_path.exists():
            print(f"Skipping existing sample: {sample_name}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            if args.stage1:
                if not run_stage1(args, video_path, output_dir, pipeline):
                    remove_sample_output(output_dir)
                    continue
            if args.stage2:
                run_stage2(args, output_dir)
            if not args.keep_intermediate and args.stage2:
                cleanup_intermediate(output_dir)
        except ValueError as exc:
            if not is_segmentation_failure(exc):
                raise
            print(f"Skipping {video_path.name}: {exc}")
            remove_sample_output(output_dir)


if __name__ == "__main__":
    main(parse_args())
