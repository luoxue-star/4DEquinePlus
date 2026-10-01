import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import json
import cv2
import traceback
import glob
import io
import tarfile
from collections import OrderedDict

from PIL import Image
from typing import Optional, Union, Tuple, List

from yacs.config import CfgNode
from omegaconf import DictConfig


class MiniVarenAvatarDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: CfgNode, root_dir: str, is_train: bool = True):
        """
        Dataset for VAREN-based avatar data. A directory represents one sequence and must contain:
            - c2w.pt: [Nv, 4, 4] camera-to-world matrices
            - mvp_matrix.pt: [Nv, 4, 4] intrinsic matrices
            - params.npz: VAREN parameters including keys: 'shape', 'global_orient', 'pose', 'transl'
            - images and optional masks (either co-located or in a 'masks' subfolder)
        """
        super().__init__()
        self.sample_side_views = cfg.DATASETS.get("SAMPLE_SIDE_VIEWS", 4)
        self.render_image_res_low = cfg.DATASETS.get("RENDER_IMAGE_RES_LOW", 512)
        self.render_image_res_high = cfg.DATASETS.get("RENDER_IMAGE_RES_HIGH", 512)
        self.render_region_size = cfg.DATASETS.get("RENDER_REGION_SIZE", 512)
        self.source_image_res = cfg.DATASETS.get("SOURCE_IMAGE_RES", 512)

        self.crop_range_ratio_hw = cfg.DATASETS.get("CROP_RANGE_RATIO_HW", [1.0, 1.0])
        self.aspect_standard = cfg.DATASETS.get("ASPECT_STANDARD", 1.0)

        assert self.render_image_res_low == self.render_image_res_high
        self.render_image_res = self.render_image_res_low
        self.enlarge_ratio = cfg.DATASETS.get("ENLARGE_RATIO", [0.8, 1.2])
        self.multiply = cfg.DATASETS.get("MULTIPLY", 16)
        self.source_crop_prob = float(cfg.DATASETS.get("SOURCE_CROP_PROB", 0.0))
        crop_ratio_min, crop_ratio_max = cfg.DATASETS.get("SOURCE_CROP_RATIO_RANGE", [0.5, 0.85])
        crop_ratio_min = max(0.5, float(crop_ratio_min))
        crop_ratio_max = max(crop_ratio_min, float(crop_ratio_max))
        self.source_crop_ratio_range = [crop_ratio_min, crop_ratio_max]
        self.source_mask_prob = float(cfg.DATASETS.get("SOURCE_MASK_PROB", cfg.DATASETS.get("MASK_PROB", 0.0)))
        self.source_mask_pattern = cfg.DATASETS.get("SOURCE_MASK_PATTERN", cfg.get("MASK_PATTERN", None))
        self.source_mask_coverage_range = cfg.DATASETS.get("SOURCE_MASK_COVERAGE_RANGE", [0.4, 1.0])
        self.source_mask_webdataset = None
        self.source_mask_iterator = None
        self.source_mask_worker_key = None
        self.source_mask_webdataset_initialized = False
        self.cache_max_sequences = int(cfg.DATASETS.get("DATA_CACHE_MAX_SEQUENCES", 4096))
        self.cache_max_tensors = int(cfg.DATASETS.get("DATA_CACHE_MAX_TENSORS", 1024))
        self.cache_max_images = int(cfg.DATASETS.get("DATA_CACHE_MAX_IMAGES", 128))
        self.cache_max_shards = int(cfg.DATASETS.get("DATA_CACHE_MAX_SHARDS", 16))
        self._sequence_info_cache = OrderedDict()
        self._camera_cache = OrderedDict()
        self._params_cache = OrderedDict()
        self._image_cache = OrderedDict()
        self._tar_cache = OrderedDict()
        self._shard_file_cache = OrderedDict()
        # set data deterministic
        self.is_val = not is_train

        self.root_dirs = root_dir
        self.shard_index_path = None
        self.shard_index_dir = None
        self.shard_entries = None
        self.shard_entries_by_seq = None
        if self.root_dirs and str(self.root_dirs).endswith(".jsonl"):
            if not os.path.isfile(self.root_dirs):
                raise FileNotFoundError(
                    f"Shard index not found: {self.root_dirs}. Run `python temp.py --overwrite` to build shards first."
                )
            self._load_shard_index(self.root_dirs)
        else:
            self.sequence = os.listdir(self.root_dirs)

    @staticmethod
    def _cache_get(cache, key):
        if key not in cache:
            return None
        cache.move_to_end(key)
        return cache[key]

    @staticmethod
    def _cache_put(cache, key, value, max_items):
        if max_items <= 0:
            return value
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_items:
            cache.popitem(last=False)
        return value

    def _source_mask_current_worker_key(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = None if worker_info is None else worker_info.id
        return os.getpid(), worker_id

    def _load_shard_index(self, index_path):
        self.shard_index_path = os.path.abspath(index_path)
        self.shard_index_dir = os.path.dirname(self.shard_index_path)
        self.root_dirs = self.shard_index_dir
        entries = []
        with open(self.shard_index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                shard_path = entry["shard"]
                if not os.path.isabs(shard_path):
                    shard_path = os.path.join(self.shard_index_dir, shard_path)
                entry["shard_path"] = shard_path
                entries.append(entry)
        self.shard_entries = entries
        self.shard_entries_by_seq = {entry["seq_name"]: entry for entry in entries}
        self.sequence = [entry["seq_name"] for entry in entries]

    @property
    def use_shards(self):
        return self.shard_entries is not None

    def _get_shard_tar(self, shard_path):
        cached = self._cache_get(self._tar_cache, shard_path)
        if cached is not None:
            return cached
        tar = tarfile.open(shard_path, "r")
        self._tar_cache[shard_path] = tar
        self._tar_cache.move_to_end(shard_path)
        while len(self._tar_cache) > self.cache_max_shards:
            _, old_tar = self._tar_cache.popitem(last=False)
            old_tar.close()
        return tar

    def _read_shard_member(self, entry, member):
        member_offsets = entry.get("member_offsets", None)
        if member_offsets is not None and member in member_offsets:
            cached_file = self._cache_get(self._shard_file_cache, entry["shard_path"])
            if cached_file is None:
                cached_file = open(entry["shard_path"], "rb")
                self._shard_file_cache[entry["shard_path"]] = cached_file
                self._shard_file_cache.move_to_end(entry["shard_path"])
                while len(self._shard_file_cache) > self.cache_max_shards:
                    _, old_file = self._shard_file_cache.popitem(last=False)
                    old_file.close()
            offset, size = member_offsets[member]
            cached_file.seek(offset)
            return cached_file.read(size)

        tar = self._get_shard_tar(entry["shard_path"])
        extracted = tar.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"Missing shard member {member} in {entry['shard_path']}")
        return extracted.read()

    def _load_shard_image(self, entry, member, mode):
        key = ("shard_image", entry["shard_path"], member, mode)
        cached = self._cache_get(self._image_cache, key)
        if cached is not None:
            return cached
        data = self._read_shard_member(entry, member)
        image = np.array(Image.open(io.BytesIO(data)).convert(mode))
        return self._cache_put(self._image_cache, key, image, self.cache_max_images)

    def _load_shard_concat_views(self, entry):
        key = ("shard_concat", entry["shard_path"], entry["seq_name"])
        cached = self._cache_get(self._image_cache, key)
        if cached is not None:
            return cached
        members = entry["members"]
        image = self._load_shard_image(entry, members["image"], "RGB")
        mask = self._load_shard_image(entry, members["mask"], "L")
        images = tuple(image[:, i * image.shape[0]:(i + 1) * image.shape[0]] for i in range(image.shape[1] // image.shape[0]))
        mask_images = tuple(mask[:, i * image.shape[0]:(i + 1) * image.shape[0]] for i in range(mask.shape[1] // image.shape[0]))
        assert len(images) == len(mask_images), f"Number of images and masks mismatch: {len(images)}, {len(mask_images)}"
        return self._cache_put(self._image_cache, key, (images, mask_images), self.cache_max_images)

    def _load_shard_frame_pair(self, entry, image_member, mask_member):
        image = self._load_shard_image(entry, image_member, "RGB")
        mask = self._load_shard_image(entry, mask_member, "L")
        return image, mask

    def _init_source_mask_webdataset(self):
        self.source_mask_webdataset_initialized = True
        self.source_mask_worker_key = self._source_mask_current_worker_key()
        if self.source_mask_prob <= 0.0 or not self.source_mask_pattern:
            return
        try:
            import webdataset as wds
            webdataset_kwargs = dict(
                shardshuffle=10,
                nodesplitter=wds.split_by_node,
                workersplitter=getattr(wds, "split_by_worker", None),
                empty_check=False,
            )
            self.source_mask_webdataset = (
                wds.WebDataset(
                    self.source_mask_pattern,
                    **webdataset_kwargs,
                )
                .map(self._decode_source_mask_sample)
                .shuffle(100)
            )
            self.source_mask_iterator = iter(self.source_mask_webdataset)
        except Exception as ex:
            print(f"Error loading avatar source masks: {ex}")
            self.source_mask_webdataset = None
            self.source_mask_iterator = None

    def _ensure_source_mask_webdataset(self):
        worker_key = self._source_mask_current_worker_key()
        if self.source_mask_worker_key != worker_key:
            self.source_mask_webdataset = None
            self.source_mask_iterator = None
            self.source_mask_webdataset_initialized = False
        if not self.source_mask_webdataset_initialized:
            self._init_source_mask_webdataset()
        return self.source_mask_iterator is not None

    def _decode_source_mask_sample(self, sample):
        try:
            for key in ("png", "jpg", "jpeg"):
                if key in sample:
                    return np.array(Image.open(io.BytesIO(sample[key])).convert("RGBA"))
        except Exception as ex:
            print(f"Error decoding avatar source mask sample: {ex}")
        return None

    def _get_random_occlusion_mask(self):
        if not self._ensure_source_mask_webdataset():
            return None
        try:
            return next(self.source_mask_iterator)
        except StopIteration:
            if self.source_mask_webdataset is None:
                return None
            self.source_mask_iterator = iter(self.source_mask_webdataset)
            try:
                return next(self.source_mask_iterator)
            except StopIteration:
                return None
        except Exception as ex:
            print(f"Error reading avatar source mask sample: {ex}")
            return None

    def _get_sequence_info(self, uid):
        if self.use_shards:
            seq_name = os.path.basename(uid)
            entry = self.shard_entries_by_seq[seq_name]
            return {
                "use_concat_image": bool(entry["use_concat_image"]),
                "num_views": int(entry["num_views"]),
                "shard_entry": entry,
            }

        cached = self._cache_get(self._sequence_info_cache, uid)
        if cached is not None:
            return cached

        image_jpg = os.path.join(uid, "image.jpg")
        image_png = os.path.join(uid, "image.png")
        concat_image = image_jpg if os.path.exists(image_jpg) else image_png
        concat_mask = os.path.join(uid, "mask.png")

        try:
            if os.path.exists(concat_image) and os.path.exists(concat_mask):
                with Image.open(concat_image) as image:
                    width, height = image.size
                with Image.open(concat_mask) as mask:
                    mask_width, mask_height = mask.size
                assert height == mask_height and width == mask_width
                assert height > 0 and width % height == 0
                info = {
                    "use_concat_image": True,
                    "num_views": width // height,
                    "image_file": concat_image,
                    "mask_file": concat_mask,
                }
                return self._cache_put(self._sequence_info_cache, uid, info, self.cache_max_sequences)
        except Exception:
            pass

        image_files = sorted(glob.glob(os.path.join(uid, "images", "*.jpg")))
        mask_files = sorted(glob.glob(os.path.join(uid, "masks", "*.png")))
        assert len(image_files) == len(mask_files), f"Number of images and masks mismatch: {len(image_files)}, {len(mask_files)}"
        info = {
            "use_concat_image": False,
            "num_views": len(image_files),
            "image_files": image_files,
            "mask_files": mask_files,
        }
        return self._cache_put(self._sequence_info_cache, uid, info, self.cache_max_sequences)

    def _load_concat_views(self, image_file, mask_file):
        key = ("concat", image_file, mask_file)
        cached = self._cache_get(self._image_cache, key)
        if cached is not None:
            return cached

        image = np.array(Image.open(image_file).convert('RGB'))
        mask = np.array(Image.open(mask_file).convert('L'))
        images = tuple(image[:, i * image.shape[0]:(i + 1) * image.shape[0]] for i in range(image.shape[1] // image.shape[0]))
        mask_images = tuple(mask[:, i * image.shape[0]:(i + 1) * image.shape[0]] for i in range(mask.shape[1] // image.shape[0]))
        assert len(images) == len(mask_images), f"Number of images and masks mismatch: {len(images)}, {len(mask_images)}"
        return self._cache_put(self._image_cache, key, (images, mask_images), self.cache_max_images)

    def _load_frame_pair(self, image_file, mask_file):
        key = ("frame", image_file, mask_file)
        cached = self._cache_get(self._image_cache, key)
        if cached is not None:
            return cached

        image = np.array(Image.open(image_file).convert('RGB'))
        mask = np.array(Image.open(mask_file).convert('L'))
        return self._cache_put(self._image_cache, key, (image, mask), self.cache_max_images)

    def _load_camera_tensors(self, uid):
        if self.use_shards:
            seq_name = os.path.basename(uid)
            entry = self.shard_entries_by_seq[seq_name]
            members = entry["members"]
            if "c2w" not in members or "mvp_matrix" not in members:
                return None
            key = ("shard_camera", entry["shard_path"], entry["seq_name"])
            cached = self._cache_get(self._camera_cache, key)
            if cached is not None:
                return cached
            tensors = (
                torch.load(io.BytesIO(self._read_shard_member(entry, members["c2w"])), weights_only=True, map_location="cpu"),
                torch.load(io.BytesIO(self._read_shard_member(entry, members["mvp_matrix"])), weights_only=True, map_location="cpu"),
            )
            return self._cache_put(self._camera_cache, key, tensors, self.cache_max_tensors)

        c2w_path = os.path.join(uid, 'c2w.pt')
        mvp_matrix_path = os.path.join(uid, 'mvp_matrix.pt')
        if not (os.path.exists(c2w_path) and os.path.exists(mvp_matrix_path)):
            return None

        key = (c2w_path, mvp_matrix_path)
        cached = self._cache_get(self._camera_cache, key)
        if cached is not None:
            return cached

        tensors = (
            torch.load(c2w_path, weights_only=True, map_location="cpu"),
            torch.load(mvp_matrix_path, weights_only=True, map_location="cpu"),
        )
        return self._cache_put(self._camera_cache, key, tensors, self.cache_max_tensors)

    def _parse_varen_params_npz(self, params_npz, num_views):
        varen = {}
        # betas
        if 'shape' in params_npz:
            betas = np.array(params_npz['shape'])
            betas = betas.reshape(-1)
            varen['betas'] = torch.from_numpy(betas).float()
        # global_orient
        if 'global_orient' in params_npz:
            go = np.array(params_npz['global_orient'])
            if go.ndim == 1:
                go = go.reshape(1, 3)
            if go.shape[0] == 1 and num_views > 1:
                go = np.repeat(go[None, ...], num_views, axis=0)  # [Nv, 1, 3] or [Nv, 3]
            if go.ndim == 2:
                go = go[:, None, :]  # [Nv, 1, 3]
            varen['global_orient'] = torch.from_numpy(go).float()
        # pose
        if 'pose' in params_npz:
            pose = np.array(params_npz['pose'])
            if pose.ndim == 1:
                pose = pose.reshape(-1, 3)
            if pose.ndim == 2 and pose.shape[-1] == 3:  # [J, 3]
                pose = np.repeat(pose[None, ...], num_views, axis=0)  # [Nv, J, 3]
            varen['pose'] = torch.from_numpy(pose).float()
        # trans
        if 'transl' in params_npz:
            trans = np.array(params_npz['transl'])
            if trans.ndim == 1:
                trans = np.repeat(trans[None, ...], num_views, axis=0)  # [Nv, 3]
            varen['trans'] = torch.from_numpy(trans).float()
        if 'tail_scale' in params_npz:
            tail_scale = np.array(params_npz['tail_scale'])
            tail_scale = tail_scale.reshape(-1)
            varen['tail_scale'] = torch.from_numpy(tail_scale).float()
        return varen

    def _load_varen_params_cached(self, params_path, num_views):
        if self.use_shards:
            seq_name = os.path.basename(os.path.dirname(params_path))
            entry = self.shard_entries_by_seq[seq_name]
            key = ("shard_params", entry["shard_path"], entry["seq_name"], int(num_views))
            cached = self._cache_get(self._params_cache, key)
            if cached is not None:
                return cached
            data = self._read_shard_member(entry, entry["members"]["params"])
            with np.load(io.BytesIO(data), allow_pickle=True) as params_npz:
                params = self._parse_varen_params_npz(params_npz, num_views=num_views)
            return self._cache_put(self._params_cache, key, params, self.cache_max_tensors)

        key = (params_path, int(num_views))
        cached = self._cache_get(self._params_cache, key)
        if cached is not None:
            return cached

        params = self._load_varen_params(params_path, num_views=num_views)
        return self._cache_put(self._params_cache, key, params, self.cache_max_tensors)

    @staticmethod
    def _binary_mask_from_image(mask_img):
        if mask_img is None:
            return None
        if isinstance(mask_img, torch.Tensor):
            mask_img = mask_img.detach().cpu().numpy()
        mask_img = np.asarray(mask_img)
        if mask_img.ndim == 2:
            alpha = mask_img
        elif mask_img.ndim == 3 and mask_img.shape[2] == 4:
            alpha = mask_img[:, :, 3]
        elif mask_img.ndim == 3:
            alpha = np.mean(mask_img[:, :, :3], axis=2)
        else:
            return None
        alpha = alpha.astype(np.float32)
        if alpha.max() > 1.0:
            alpha = alpha / 255.0
        binary = (alpha > 0.5).astype(np.float32)
        if not np.any(binary > 0.0):
            return None
        return binary

    @staticmethod
    def _mask_bbox(mask):
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1

    def _apply_source_crop_augmentation(self, source_rgbs, source_masks):
        was_single = source_rgbs.ndim == 3
        source_rgbs_in = source_rgbs.unsqueeze(0) if was_single else source_rgbs
        source_masks_in = source_masks.unsqueeze(0) if source_masks.ndim == 3 else source_masks

        cropped_rgbs = source_rgbs_in.clone()
        cropped_masks = source_masks_in.clone()
        _, _, height, width = cropped_rgbs.shape

        for src_idx in range(cropped_rgbs.shape[0]):
            src_mask = source_masks_in[src_idx, 0].detach().cpu().numpy()
            src_bbox = self._mask_bbox(src_mask)
            if src_bbox is None:
                continue

            src_x, src_y, src_w, src_h = src_bbox
            ratio_min, ratio_max = self.source_crop_ratio_range
            crop_ratio = random.uniform(float(ratio_min), float(ratio_max))
            output_aspect = width / max(height, 1)
            bbox_aspect = src_w / max(src_h, 1)
            if bbox_aspect >= output_aspect:
                max_crop_h = src_h
                max_crop_w = max(1, int(round(max_crop_h * output_aspect)))
            else:
                max_crop_w = src_w
                max_crop_h = max(1, int(round(max_crop_w / output_aspect)))
            max_crop_w = min(src_w, max_crop_w)
            max_crop_h = min(src_h, max_crop_h)
            crop_w = max(1, min(max_crop_w, int(round(max_crop_w * crop_ratio))))
            crop_h = max(1, int(round(crop_w / output_aspect)))
            if crop_h > max_crop_h:
                crop_h = max_crop_h
                crop_w = max(1, int(round(crop_h * output_aspect)))
            crop_w = min(max_crop_w, crop_w)
            crop_h = min(max_crop_h, crop_h)

            rel_x = random.randint(0, max(0, src_w - crop_w))
            rel_y = random.randint(0, max(0, src_h - crop_h))
            crop_x0 = max(0, min(src_x + rel_x, width - crop_w))
            crop_y0 = max(0, min(src_y + rel_y, height - crop_h))
            crop_x1 = min(crop_x0 + crop_w, width)
            crop_y1 = min(crop_y0 + crop_h, height)
            if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
                continue

            rgb_patch = source_rgbs_in[src_idx:src_idx + 1, :, crop_y0:crop_y1, crop_x0:crop_x1]
            mask_patch = source_masks_in[src_idx:src_idx + 1, :, crop_y0:crop_y1, crop_x0:crop_x1]
            cropped_rgbs[src_idx:src_idx + 1] = F.interpolate(
                rgb_patch,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            cropped_masks[src_idx:src_idx + 1] = F.interpolate(
                mask_patch,
                size=(height, width),
                mode="nearest",
            )

        if was_single:
            return cropped_rgbs.squeeze(0), cropped_masks.squeeze(0)
        return cropped_rgbs, cropped_masks

    def _apply_source_mask_occlusion(self, source_rgbs, source_masks=None, source_bg_colors=None, force=False):
        if self.source_mask_prob <= 0.0 and not force:
            return source_rgbs.clone()

        was_single = source_rgbs.ndim == 3
        source_rgbs_in = source_rgbs.unsqueeze(0) if was_single else source_rgbs
        source_masks_in = None
        if source_masks is not None:
            source_masks_in = source_masks.unsqueeze(0) if source_masks.ndim == 3 else source_masks
        source_bg_colors_in = source_bg_colors
        if source_bg_colors_in is not None:
            source_bg_colors_in = source_bg_colors_in.unsqueeze(0) if source_bg_colors_in.ndim == 1 else source_bg_colors_in

        occluded = source_rgbs_in.clone()
        _, _, height, width = occluded.shape

        for src_idx in range(occluded.shape[0]):
            if not force and random.random() > self.source_mask_prob:
                continue

            occ_mask = self._binary_mask_from_image(self._get_random_occlusion_mask())
            occ_bbox = self._mask_bbox(occ_mask) if occ_mask is not None else None
            if occ_bbox is None:
                continue

            obj_x, obj_y, obj_w, obj_h = occ_bbox
            obj_region = occ_mask[obj_y:obj_y + obj_h, obj_x:obj_x + obj_w]

            if source_masks_in is not None:
                src_mask = source_masks_in[src_idx, 0].detach().cpu().numpy()
                src_bbox = self._mask_bbox(src_mask)
            else:
                src_mask = None
                src_bbox = (0, 0, width, height)
            if src_bbox is None:
                continue

            src_x, src_y, src_w, src_h = src_bbox
            coverage_min, coverage_max = self.source_mask_coverage_range
            coverage = random.uniform(float(coverage_min), float(coverage_max))
            target_w = max(1, int(src_w * coverage))
            target_h = max(1, int(src_h * coverage))

            obj_aspect = obj_w / max(obj_h, 1)
            target_aspect = target_w / max(target_h, 1)
            if obj_aspect > target_aspect:
                final_w = target_w
                final_h = max(1, int(target_w / obj_aspect))
            else:
                final_h = target_h
                final_w = max(1, int(target_h * obj_aspect))

            resized = cv2.resize(obj_region, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
            resized = (resized > 0.5).astype(np.float32)

            rel_x = random.randint(0, max(0, src_w - final_w))
            rel_y = random.randint(0, max(0, src_h - final_h))
            abs_x = max(0, min(src_x + rel_x, width - final_w))
            abs_y = max(0, min(src_y + rel_y, height - final_h))
            end_x = min(abs_x + final_w, width)
            end_y = min(abs_y + final_h, height)
            actual_w = end_x - abs_x
            actual_h = end_y - abs_y
            if actual_w <= 0 or actual_h <= 0:
                continue

            region_mask = resized[:actual_h, :actual_w]
            if src_mask is not None:
                region_mask = region_mask * (src_mask[abs_y:end_y, abs_x:end_x] > 0.5).astype(np.float32)

            region_mask_t = torch.from_numpy(region_mask).to(
                device=occluded.device,
                dtype=occluded.dtype,
            ).unsqueeze(0)
            if source_bg_colors_in is None:
                bg_color = torch.zeros(occluded.shape[1], device=occluded.device, dtype=occluded.dtype)
            else:
                bg_color = source_bg_colors_in[src_idx].to(device=occluded.device, dtype=occluded.dtype)
            bg_color = bg_color.view(-1, 1, 1)
            occluded[src_idx, :, abs_y:end_y, abs_x:end_x] = (
                occluded[src_idx, :, abs_y:end_y, abs_x:end_x] * (1.0 - region_mask_t)
                + bg_color * region_mask_t
            )

        return occluded.squeeze(0) if was_single else occluded

    def _apply_source_augmentations(self, source_rgbs, source_masks=None, source_bg_colors=None):
        crop_prob = max(0.0, float(self.source_crop_prob))
        mask_prob = max(0.0, float(self.source_mask_prob))
        total_prob = crop_prob + mask_prob
        if total_prob <= 0.0:
            return source_rgbs.clone(), None if source_masks is None else source_masks.clone()

        source_rgbs_aug = source_rgbs.clone()
        source_masks_aug = None if source_masks is None else source_masks.clone()
        num_sources = source_rgbs_aug.shape[0] if source_rgbs_aug.ndim == 4 else 1

        for src_idx in range(num_sources):
            action_value = random.random()
            src_slice = slice(src_idx, src_idx + 1)
            if action_value < crop_prob and source_masks_aug is not None:
                cropped_rgb, cropped_mask = self._apply_source_crop_augmentation(
                    source_rgbs_aug[src_slice],
                    source_masks_aug[src_slice],
                )
                source_rgbs_aug[src_slice] = cropped_rgb
                source_masks_aug[src_slice] = cropped_mask
            elif action_value < total_prob:
                src_bg = None if source_bg_colors is None else source_bg_colors[src_slice]
                source_rgbs_aug[src_slice] = self._apply_source_mask_occlusion(
                    source_rgbs_aug[src_slice],
                    None if source_masks_aug is None else source_masks_aug[src_slice],
                    src_bg,
                    force=True,
                )

        return source_rgbs_aug, source_masks_aug

    def _load_varen_params(self, params_path, num_views):
        with np.load(params_path, allow_pickle=True) as params_npz:
            return self._parse_varen_params_npz(params_npz, num_views=num_views)

    @staticmethod
    def compute_intrinsics_from_projection(P, width: int, height: int) -> Tuple[float, float, float, float]:
        """
        Given an OpenGL-style projection matrix P (with sign-flipped Y as in nvdiffrast):
        P[0,0] = 1/(tan(fovy/2)*aspect), P[1,1] = -1/tan(fovy/2)
        Derive fx, fy for a pinhole camera used by pyrender.
        We assume principal point at (width/2, height/2).
        Returns fx, fy, cx, cy.
        """
        if isinstance(P, torch.Tensor):
            P = P.detach().cpu().numpy()
        fx = width * 0.5 * abs(P[0, 0])
        fy = height * 0.5 * abs(P[1, 1])
        cx = width * 0.5
        cy = height * 0.5
        return float(fx), float(fy), float(cx), float(cy)

    def scale_intrs(self, intrs, ratio_x, ratio_y):
        if len(intrs.shape) >= 3:
            intrs[:, 0] = intrs[:, 0] * ratio_x
            intrs[:, 1] = intrs[:, 1] * ratio_y
        else:
            intrs[0] = intrs[0] * ratio_x
            intrs[1] = intrs[1] * ratio_y  
        return intrs

    def center_crop_according_to_mask(self, img, mask, aspect_standard, enlarge_ratio):
        """ 
            img: [H, W, 3]
            mask: [H, W]
        """ 
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            raise Exception("empty mask")

        x_min = np.min(xs)
        x_max = np.max(xs)
        y_min = np.min(ys)
        y_max = np.max(ys)
        
        center_x, center_y = img.shape[1]//2, img.shape[0]//2
        
        half_w = max(abs(center_x - x_min), abs(center_x -  x_max))
        half_h = max(abs(center_y - y_min), abs(center_y -  y_max))
        aspect = half_h / half_w

        if aspect >= aspect_standard:                
            half_w = round(half_h / aspect_standard)
        else:
            half_h = round(half_w * aspect_standard)

        if abs(enlarge_ratio[0] - 1) > 0.01 or abs(enlarge_ratio[1] - 1) >  0.01:
            enlarge_ratio_min, enlarge_ratio_max = enlarge_ratio
            enlarge_ratio_max_real = min(center_y / half_h, center_x / half_w)
            enlarge_ratio_max = min(enlarge_ratio_max_real, enlarge_ratio_max)
            enlarge_ratio_min = min(enlarge_ratio_max_real, enlarge_ratio_min)
            enlarge_ratio_cur = np.random.rand() * (enlarge_ratio_max - enlarge_ratio_min) + enlarge_ratio_min
            half_h, half_w = round(enlarge_ratio_cur * half_h), round(enlarge_ratio_cur * half_w)
            
        assert half_h <= center_y
        assert half_w <= center_x
        assert abs(half_h / half_w - aspect_standard) < 0.03
        
        offset_x = center_x - half_w
        offset_y = center_y - half_h
        
        new_img = img[offset_y: offset_y + 2*half_h, offset_x: offset_x + 2*half_w]
        new_mask = mask[offset_y: offset_y + 2*half_h, offset_x: offset_x + 2*half_w]
        
        return  new_img, new_mask, offset_x, offset_y      

    def calc_new_tgt_size_by_aspect(self, cur_hw, aspect_standard, tgt_size, multiply):
        assert abs(cur_hw[0] / cur_hw[1] - aspect_standard) < 0.03
        tgt_size = tgt_size * aspect_standard, tgt_size
        tgt_size = int(tgt_size[0] / multiply) * multiply, int(tgt_size[1] / multiply) * multiply
        ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
        return tgt_size, ratio_y, ratio_x

    def load_rgb_image_with_aug_bg(self, rgb, mask, bg_color, max_tgt_size, aspect_standard, enlarge_ratio,
                                   render_tgt_size, multiply, intr):
        interpolation = cv2.INTER_AREA
        assert rgb.shape[0] == mask.shape[0] and rgb.shape[1] == mask.shape[1], f"RGB and mask shape mismatch: {rgb.shape}, {mask.shape}"
        # if rgb.shape[0] != self.render_image_res and rgb.shape[0] == rgb.shape[1]:
        #     rgb = cv2.resize(rgb, (self.render_image_res, self.render_image_res), interpolation=interpolation)
        #     mask = cv2.resize(mask, (self.render_image_res, self.render_image_res), interpolation=interpolation)
        
        rgb = rgb / 255.0
        mask = (mask > 127).astype(np.float32)
        rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])
    
        try:
            rgb, mask, offset_x, offset_y = self.center_crop_according_to_mask(rgb, mask, aspect_standard, enlarge_ratio)
        except Exception as ex:
            print(ex)

        try:
            intr[0, 2] -= offset_x
            intr[1, 2] -= offset_y
        except Exception as ex:
            print(ex)
            print(intr.shape)

        # resize to render_tgt_size for training
        tgt_hw_size, ratio_y, ratio_x = self.calc_new_tgt_size_by_aspect(cur_hw=rgb.shape[:2], 
                                                                         aspect_standard=aspect_standard,
                                                                         tgt_size=render_tgt_size, multiply=multiply)
        rgb = cv2.resize(rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=interpolation)
        mask = cv2.resize(mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=interpolation)
        intr = self.scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
        
        assert abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5, f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5, f"{intr[1, 2] * 2}, {rgb.shape[0]}"
        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2
        
        rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
        mask = torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)

        return rgb, mask, intr

    def uniform_sample_in_chunk(self, sample_num, sample_data):
        chunks = np.array_split(sample_data, sample_num)
        select_list = []
        for chunk in chunks:
            select_list.append(np.random.choice(chunk))
        return select_list

    def uniform_sample_in_chunk_det(self, sample_num, sample_data):
        chunks = np.array_split(sample_data, sample_num)
        select_list = []
        for chunk in chunks:
            select_list.append(chunk[len(chunk)//2])
        return select_list
    
    def inner_get_item(self, idx):
        sample_total_views = self.sample_side_views + 1

        seq_name = self.sequence[idx]
        uid = os.path.join(self.root_dirs, seq_name)
        sequence_info = self._get_sequence_info(uid)
        Nv = sequence_info["num_views"]

        # Sample view indices
        if Nv >= sample_total_views:
            if not self.is_val:
                if np.random.rand() < 0.7 and Nv > sample_total_views:
                    frame_id_list = self.uniform_sample_in_chunk(sample_total_views, np.arange(Nv))
                else:
                    replace = Nv < sample_total_views
                    frame_id_list = np.random.choice(Nv, size=sample_total_views, replace=replace)
            else:
                if Nv > sample_total_views:
                    frame_id_list = self.uniform_sample_in_chunk_det(sample_total_views, np.arange(Nv))
                else:
                    frame_id_list = np.random.choice(Nv, size=sample_total_views, replace=True)
        else:
            if not self.is_val:
                replace = Nv < sample_total_views
                frame_id_list = np.random.choice(Nv, size=sample_total_views, replace=replace)
            else:
                if Nv > 1:
                    frame_id_list = np.linspace(0, Nv - 1, num=sample_total_views, endpoint=True)
                    frame_id_list = [round(e) for e in frame_id_list]
                else:
                    frame_id_list = [0 for _ in range(sample_total_views)]
        frame_id_list = list(map(int, frame_id_list))
        assert self.sample_side_views + 1 == len(frame_id_list)

        shard_entry = sequence_info.get("shard_entry")
        if sequence_info["use_concat_image"]:
            if shard_entry is not None:
                images_all, mask_images_all = self._load_shard_concat_views(shard_entry)
            else:
                images_all, mask_images_all = self._load_concat_views(sequence_info["image_file"], sequence_info["mask_file"])
            images = [images_all[i] for i in frame_id_list]
            mask_images = [mask_images_all[i] for i in frame_id_list]
        else:
            if shard_entry is not None:
                image_members = shard_entry["members"]["images"]
                mask_members = shard_entry["members"]["masks"]
                image_pairs = [self._load_shard_frame_pair(shard_entry, image_members[i], mask_members[i]) for i in frame_id_list]
            else:
                image_file = sequence_info["image_files"]
                mask_file = sequence_info["mask_files"]
                image_pairs = [self._load_frame_pair(image_file[i], mask_file[i]) for i in frame_id_list]
            images = [pair[0] for pair in image_pairs]
            mask_images = [pair[1] for pair in image_pairs]

        # Load camera extrinsics and intrinsics
        camera_tensors = self._load_camera_tensors(uid)
        if camera_tensors is not None:
            c2ws_full, mvp_matrix_full = camera_tensors
            c2ws_all = c2ws_full[frame_id_list]
            mvp_matrix_all = mvp_matrix_full[frame_id_list]
            proj_all = torch.bmm(mvp_matrix_all, c2ws_all)
            c2ws_all = c2ws_all.float()
            proj_all = proj_all.float()
            assert c2ws_all.shape[0] == proj_all.shape[0], "c2w and mvp must have same Nv"

            intrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(sample_total_views, 1, 1)
            for i, proj in enumerate(proj_all):
                fx, fy, cx, cy = self.compute_intrinsics_from_projection(proj, images[i].shape[1], images[i].shape[0])
                intrinsics[i, 0, 0] = fx
                intrinsics[i, 1, 1] = fy
                intrinsics[i, 0, 2] = cx
                intrinsics[i, 1, 2] = cy
        else:
            c2ws_all = torch.eye(4, dtype=torch.float32)
            c2ws_all[1, 1] = c2ws_all[2, 2] = -1.0
            c2ws_all = c2ws_all.unsqueeze(0).repeat(sample_total_views, 1, 1)
            intrinsics = torch.eye(4, dtype=torch.float32)
            intrinsics[0, 2] = images[0].shape[1] // 2
            intrinsics[1, 2] = images[0].shape[0] // 2
            # TODO: hardcode for now, need to change to dynamic
            intrinsics[0, 0] = intrinsics[1, 1] = 5000. / 256. * max(images[0].shape[0], images[0].shape[1])
            intrinsics = intrinsics.unsqueeze(0).repeat(sample_total_views, 1, 1)

        # Load VAREN parameters
        params_path = os.path.join(uid, f'{seq_name}.npz')
        if not self.use_shards:
            assert os.path.exists(params_path), f"Missing {seq_name}.npz in {uid}"
        varen_params_all = self._load_varen_params_cached(params_path, num_views=Nv)

        # Build source/render batches
        c2ws, intrs, rgbs, bg_colors, masks = [], [], [], [], []
        S = torch.tensor([[1., 0., 0., 0.],
                         [0.,-1., 0., 0.],
                         [0., 0.,-1., 0.],
                         [0., 0., 0., 1.]], dtype=torch.float32)
        for frame_id in range(len(frame_id_list)):
            c2w = c2ws_all[frame_id] @ S  # Convert OpenGL to Gaussian
            ori_intrinsic = intrinsics[frame_id]

            bg_color = random.choice([0.0, 0.5, 1.0])
            rgb, mask, intrinsic = self.load_rgb_image_with_aug_bg(
                images[frame_id],
                mask_images[frame_id],
                bg_color=bg_color,
                max_tgt_size=None,
                aspect_standard=self.aspect_standard,
                enlarge_ratio=self.enlarge_ratio,
                render_tgt_size=self.render_image_res,
                multiply=self.multiply,
                intr=ori_intrinsic.clone(),
            )

            c2ws.append(c2w)
            intrs.append(intrinsic)
            rgbs.append(rgb)
            bg_colors.append(bg_color)
            masks.append(mask)

        c2ws = torch.stack(c2ws, dim=0)
        intrs = torch.stack(intrs, dim=0)
        rgbs = torch.cat(rgbs, dim=0)
        masks = torch.cat(masks, dim=0)
        bg_colors = torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)

        # Subset varen params according to selected views
        varen_params = {}
        if 'betas' in varen_params_all:
            varen_params['betas'] = varen_params_all['betas']
        if 'global_orient' in varen_params_all:
            varen_params['global_orient'] = varen_params_all['global_orient'][frame_id_list]
        if 'pose' in varen_params_all:
            varen_params['pose'] = varen_params_all['pose'][frame_id_list]
        if 'trans' in varen_params_all:
            varen_params['trans'] = varen_params_all['trans'][frame_id_list]
        if 'tail_scale' in varen_params_all:
            varen_params['tail_scale'] = varen_params_all['tail_scale']

        # Reference/source images (pick one view)
        prob_refidx = np.ones(self.sample_side_views + 1)
        if not self.is_val:
            prob_refidx[0] = 0.5
        else:
            prob_refidx[0] = 1.0
        prob_refidx[1:] = (1 - prob_refidx[0]) / len(prob_refidx[1:])
        ref_idx = int(np.random.choice(self.sample_side_views + 1, p=prob_refidx))

        source_c2ws = c2ws[ref_idx: ref_idx + 1]
        source_intrs = intrs[ref_idx: ref_idx + 1]
        source_rgbs = rgbs[ref_idx: ref_idx + 1]
        source_masks = masks[ref_idx: ref_idx + 1]
        source_bg_colors = bg_colors[ref_idx: ref_idx + 1]
        if not self.is_val:
            source_rgbs, source_masks = self._apply_source_augmentations(source_rgbs, source_masks, source_bg_colors)

        source_varen_params = {}
        if 'betas' in varen_params:
            source_varen_params['betas'] = varen_params['betas']
        if 'global_orient' in varen_params:
            source_varen_params['global_orient'] = varen_params['global_orient'][ref_idx: ref_idx + 1]
        if 'pose' in varen_params:
            source_varen_params['pose'] = varen_params['pose'][ref_idx: ref_idx + 1]
        if 'trans' in varen_params:
            source_varen_params['trans'] = varen_params['trans'][ref_idx: ref_idx + 1]
        if 'tail_scale' in varen_params:
            source_varen_params['tail_scale'] = varen_params['tail_scale']

        render_image = rgbs
        render_mask = masks
        tgt_size = render_image.shape[2:4]
        assert abs(intrs[0, 0, 2] * 2 - render_image.shape[3]) <= 2.5, f"{intrs[0, 0, 2] * 2}, {render_image.shape}"
        assert abs(intrs[0, 1, 2] * 2 - render_image.shape[2]) <= 2.5, f"{intrs[0, 1, 2] * 2}, {render_image.shape}"

        ret = {
            'uid': uid,
            'source_c2ws': source_c2ws,
            'source_intrs': source_intrs,
            'source_rgbs': source_rgbs.clamp(0, 1),
            'render_image': render_image.clamp(0, 1),
            'render_mask': render_mask.clamp(0, 1),
            'c2ws': c2ws,
            'intrs': intrs,
            'render_full_resolutions': torch.tensor([tgt_size], dtype=torch.float32).repeat(self.sample_side_views + 1, 1),
            'render_bg_colors': bg_colors,
            'pytorch3d_transpose_R': torch.tensor([0.0]),
        }

        ret["smal_params"] = varen_params
        ret["source_smal_params"] = source_varen_params

        return ret
    
    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        if len(self.sequence) == 0:
            raise IndexError("MiniVarenAvatarDataset has no samples to load.")

        # Bounded retry to avoid unbounded recursion when samples are bad.
        max_trials = len(self.sequence)
        cur_idx = idx % len(self.sequence)
        for _ in range(max_trials):
            try:
                return self.inner_get_item(cur_idx)
            except Exception:
                traceback.print_exc()
                print(f"[DEBUG-DATASET] Error when loading {self.sequence[cur_idx]}")
                cur_idx = (cur_idx + 1) % len(self.sequence)

        # If all attempts fail, surface a clear error.
        raise RuntimeError("MiniVarenAvatarDataset failed to load any sample; all trials errored.")


class MergedMiniVarenAvatarDataset(torch.utils.data.Dataset):
    """
    Mix multiple MiniVarenAvatarDataset instances using per-dataset sampling weights.
    The weight in cfg.DATASETS.<name>.WEIGHT controls sampling probability during training.
    """

    @classmethod
    def _dataset_cls(cls):
        """Return the per-root dataset class. Override in subclasses."""
        return MiniVarenAvatarDataset

    def __init__(self, cfg: CfgNode, is_train: bool = True):
        super().__init__()
        self.cfg = cfg
        self.is_val = not is_train
        # deterministic base seed so DDP / DistributedSampler can rely on __getitem__(idx)
        self.base_seed = int(cfg.get("SEED", 42)) if hasattr(cfg, "get") else 42
        # track epoch to make sampling differ across epochs while still reproducible
        self.epoch = 0
        self._epoch_perm = None
        self._epoch_perm_version = -1

        dataset_entries = []
        weights_provided = []
        for name, node in cfg.DATASETS.items():
            if isinstance(node, (CfgNode, DictConfig, dict)):
                root = node.get("SHARD_INDEX", node.get("ROOT", None))
                is_val_entry = bool(node.get("IS_VAL", False))
                weight = node.get("WEIGHT", None)
            elif hasattr(node, "get"):
                try:
                    root = node.get("SHARD_INDEX", node.get("ROOT", None))
                    is_val_entry = bool(node.get("IS_VAL", False))
                    weight = node.get("WEIGHT", None)
                except Exception:
                    continue
            else:
                continue

            if root is None or is_val_entry != self.is_val:
                continue

            weights_provided.append(weight is not None)
            dataset_entries.append((name, root, weight))

        if len(dataset_entries) == 0:
            stage = "val" if self.is_val else "train"
            raise ValueError(
                f"MergedMiniVarenAvatarDataset requires at least one dataset with ROOT and IS_VAL={self.is_val} in cfg.DATASETS for {stage} stage."
            )

        self.datasets: List[MiniVarenAvatarDataset] = []
        self.dataset_names: List[str] = []
        probs = []
        lengths = []
        _ds_cls = self._dataset_cls()
        for name, root_dir, weight in dataset_entries:
            ds = _ds_cls(cfg, root_dir=root_dir, is_train=is_train)
            if len(ds) == 0:
                continue
            self.datasets.append(ds)
            self.dataset_names.append(name)
            # If weight is missing, fall back to dataset length to preserve coverage.
            if weight is None:
                probs.append(float(len(ds)))
            else:
                probs.append(max(float(weight), 0.0))
            lengths.append(len(ds))

        if len(self.datasets) == 0:
            raise ValueError("MergedMiniVarenAvatarDataset found only empty datasets.")

        prob_sum = float(sum(probs))
        if prob_sum <= 0:
            self.sample_probs = np.ones(len(probs), dtype=np.float32) / len(probs)
        else:
            self.sample_probs = np.array(probs, dtype=np.float32) / prob_sum

        self.lengths = lengths
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.total_len = int(sum(self.lengths))
        # If all weights were missing, prefer length-based deterministic sampling to ensure full coverage each epoch.
        self.length_based_sampling = len(weights_provided) > 0 and all(not w for w in weights_provided)

    def set_epoch(self, epoch: int):
        """Allow external samplers (e.g., DistributedSampler) to inject epoch for per-epoch shuffling."""
        self.epoch = int(epoch)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if len(self.datasets) == 0:
            raise IndexError("MergedMiniVarenAvatarDataset has no datasets to sample from.")

        # Make sampling deterministic w.r.t. idx for DDP compatibility
        if self.length_based_sampling and not self.is_val:
            # Shuffle the flattened index space once per epoch so ordering changes every epoch
            if self._epoch_perm_version != self.epoch or self._epoch_perm is None:
                rng = np.random.default_rng(self.base_seed + self.epoch)
                self._epoch_perm = rng.permutation(self.total_len)
                self._epoch_perm_version = self.epoch
            wrapped_idx = idx % self.total_len
            mapped_idx = int(self._epoch_perm[wrapped_idx])
            ds_idx = int(np.searchsorted(self.cumulative_lengths, mapped_idx, side="right"))
            prev_end = 0 if ds_idx == 0 else self.cumulative_lengths[ds_idx - 1]
            sample_idx = int((mapped_idx - prev_end) % self.lengths[ds_idx])
        elif self.is_val:
            wrapped_idx = idx % self.total_len
            ds_idx = int(np.searchsorted(self.cumulative_lengths, wrapped_idx, side="right"))
            prev_end = 0 if ds_idx == 0 else self.cumulative_lengths[ds_idx - 1]
            sample_idx = int((wrapped_idx - prev_end) % self.lengths[ds_idx])
        else:
            # Probability sampling: change randomness each epoch so sequence differs across epochs
            rng = np.random.default_rng(self.base_seed + self.epoch * 1000003 + idx)
            ds_idx = int(rng.choice(len(self.datasets), p=self.sample_probs))
            sample_idx = int(rng.integers(0, self.lengths[ds_idx]))

        sample = self.datasets[ds_idx][sample_idx]
        sample["dataset_name"] = self.dataset_names[ds_idx]
        sample["dataset_idx"] = ds_idx
        return sample


class Bbox:
    def __init__(self, box, mode="whwh"):

        assert len(box) == 4
        assert mode in ["whwh", "xywh"]
        self.box = box
        self.mode = mode

    def to_xywh(self):

        if self.mode == "whwh":

            l, t, r, b = self.box

            center_x = (l + r) / 2
            center_y = (t + b) / 2
            width = r - l
            height = b - t
            return Bbox([center_x, center_y, width, height], mode="xywh")
        else:
            return self

    def to_whwh(self):

        if self.mode == "whwh":
            return self
        else:

            cx, cy, w, h = self.box
            l = cx - w // 2
            t = cy - h // 2
            r = cx + w - (w // 2)
            b = cy + h - (h // 2)

            return Bbox([l, t, r, b], mode="whwh")

    def area(self):

        box = self.to_xywh()
        _, __, w, h = box.box

        return w * h

    def get_box(self):
        return list(map(int, self.box))

    def scale(self, scale, width, height):
        new_box = self.to_xywh()
        cx, cy, w, h = new_box.get_box()
        w = w * scale
        h = h * scale

        l = cx - w // 2
        t = cy - h // 2
        r = cx + w - (w // 2)
        b = cy + h - (h // 2)

        l = int(max(l, 0))
        t = int(max(t, 0))
        r = int(min(r, width))
        b = int(min(b, height))

        return Bbox([l, t, r, b], mode="whwh")

    def __repr__(self):
        box = self.to_whwh()
        l, t, r, b = box.box

        return f"BBox(left={l}, top={t}, right={r}, bottom={b})"


class AvatarInferDataset(torch.utils.data.Dataset):
    def __init__(self, image: List[np.ndarray], mask: List[np.ndarray], intrs: Optional[np.ndarray] = None, render_image_res: int = 512):
        super().__init__()
        self.image = image
        self.mask = mask
        assert len(self.image) == len(self.mask), "Image and mask must have the same length"
        self.intrs = intrs
        self.render_image_res = render_image_res

    def __getitem__(self, idx):
        image = self.image[idx]
        image_height, image_width = image.shape[:2]
        mask = self.mask[idx]
        intr = self.intrs[idx] if self.intrs is not None else None
        image, mask, intr = self.infer_preprocess_image(image, mask, intr, render_tgt_size=self.render_image_res, max_tgt_size=self.render_image_res)
        if intr is None:
            item = {
                "source_rgbs": image,
                "source_masks": mask,
                "image_ori_height": image_height,
                "image_ori_width": image_width,
            }
        else:
            item = {
                "source_rgbs": image,
                "source_masks": mask,
                "source_intrs": intr,
                "image_ori_height": image_height,
                "image_ori_width": image_width,
            }
        return item

    def __len__(self):
        return len(self.image)

    def get_bbox(self, mask):
        height, width = mask.shape
        # Normalize mask to [0, 1], robust to boolean or 0-255 inputs
        pha = mask.astype(np.float32)
        if pha.max() > 1.0:
            pha = pha / 255.0
        pha[pha < 0.5] = 0.0
        pha[pha >= 0.5] = 1.0

        # obtain bbox
        _h, _w = np.where(pha == 1)

        whwh = [
            _w.min().item(),
            _h.min().item(),
            _w.max().item(),
            _h.max().item(),
        ]

        box = Bbox(whwh)
        # scale box to 1.05
        scale_box = box.scale(1.1, width=width, height=height)
        return scale_box

    def resize_image_keepaspect_np(self, img, max_tgt_size):
        """
        similar to ImageOps.contain(img_pil, (img_size, img_size)) # keep the same aspect ratio
        """
        h, w = img.shape[:2]
        ratio = max_tgt_size / max(h, w)
        new_h, new_w = round(h * ratio), round(w * ratio)
        return cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)

    def center_crop_according_to_mask(self, img, mask, aspect_standard, enlarge_ratio):
        """
        img: [H, W, 3]
        mask: [H, W]
        """
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            raise Exception("empty mask")

        x_min = np.min(xs)
        x_max = np.max(xs)
        y_min = np.min(ys)
        y_max = np.max(ys)

        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

        half_w = max(abs(center_x - x_min), abs(center_x - x_max))
        half_h = max(abs(center_y - y_min), abs(center_y - y_max))
        half_w_raw = half_w
        half_h_raw = half_h
        aspect = half_h / half_w

        if aspect >= aspect_standard:
            half_w = round(half_h / aspect_standard)
        else:
            half_h = round(half_w * aspect_standard)

        # not exceed original image
        if half_h > center_y:
            half_w = round(half_h_raw / aspect_standard)
            half_h = half_h_raw
        if half_w > center_x:
            half_h = round(half_w_raw * aspect_standard)
            half_w = half_w_raw

        if abs(enlarge_ratio[0] - 1) > 0.01 or abs(enlarge_ratio[1] - 1) > 0.01:
            enlarge_ratio_min, enlarge_ratio_max = enlarge_ratio
            enlarge_ratio_max_real = min(center_y / half_h, center_x / half_w)
            enlarge_ratio_max = min(enlarge_ratio_max_real, enlarge_ratio_max)
            enlarge_ratio_min = min(enlarge_ratio_max_real, enlarge_ratio_min)
            enlarge_ratio_cur = (
                np.random.rand() * (enlarge_ratio_max - enlarge_ratio_min)
                + enlarge_ratio_min
            )
            half_h, half_w = round(enlarge_ratio_cur * half_h), round(
                enlarge_ratio_cur * half_w
            )

        assert half_h <= center_y
        assert half_w <= center_x
        assert abs(half_h / half_w - aspect_standard) < 0.03

        offset_x = center_x - half_w
        offset_y = center_y - half_h

        new_img = img[offset_y : offset_y + 2 * half_h, offset_x : offset_x + 2 * half_w]
        new_mask = mask[offset_y : offset_y + 2 * half_h, offset_x : offset_x + 2 * half_w]

        return new_img, new_mask, offset_x, offset_y

    def calc_new_tgt_size_by_aspect(self, cur_hw, aspect_standard, tgt_size, multiply):
        assert abs(cur_hw[0] / cur_hw[1] - aspect_standard) < 0.03
        tgt_size = tgt_size * aspect_standard, tgt_size
        tgt_size = (
            int(tgt_size[0] / multiply) * multiply,
            int(tgt_size[1] / multiply) * multiply,
        )
        ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
        return tgt_size, ratio_y, ratio_x

    def scale_intrs(self, intrs, ratio_x, ratio_y):
        if len(intrs.shape) >= 3:
            intrs[:, 0] = intrs[:, 0] * ratio_x
            intrs[:, 1] = intrs[:, 1] * ratio_y
        else:
            intrs[0] = intrs[0] * ratio_x
            intrs[1] = intrs[1] * ratio_y  
        return intrs

    def infer_preprocess_image(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        intr: Optional[np.ndarray] = None,
        bg_color: float = 1.0,
        max_tgt_size: int = 512,
        aspect_standard: float = 1.0,
        enlarge_ratio: List[float] = [1.0, 1.0],
        render_tgt_size: int = 512,
        multiply: int = 16,
    ):
        rgb_raw = rgb.copy()

        bbox = self.get_bbox(mask)
        bbox_list = bbox.get_box()

        rgb = rgb[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]
        mask = mask[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]

        h, w, _ = rgb.shape
        cur_ratio = h / w
        scale_ratio = cur_ratio / aspect_standard


        target_w = int(min(w * scale_ratio, h))
        if target_w - w >0:
            offset_w = (target_w - w) // 2

            rgb = np.pad(
                rgb,
                ((0, 0), (offset_w, offset_w), (0, 0)),
                mode="constant",
                constant_values=255,
            )

            mask = np.pad(
                mask,
                ((0, 0), (offset_w, offset_w)),
                mode="constant",
                constant_values=0,
            )
        else:
            target_h = w * aspect_standard
            offset_h = int(target_h - h)

            rgb = np.pad(
                rgb,
                ((offset_h, 0), (0, 0), (0, 0)),
                mode="constant",
                constant_values=255,
            )

            mask = np.pad(
                mask,
                ((offset_h, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        rgb = rgb / 255.0  # normalize to [0, 1]
        mask = mask / 255.0

        mask = (mask > 0.5).astype(np.float32)
        rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

        # resize to specific size require by preprocessor of smplx-estimator.
        rgb = self.resize_image_keepaspect_np(rgb, max_tgt_size)
        mask = self.resize_image_keepaspect_np(mask, max_tgt_size)

        # crop image to enlarge human area.
        rgb, mask, offset_x, offset_y = self.center_crop_according_to_mask(
            rgb, mask, aspect_standard, enlarge_ratio
        )
        if intr is not None:
            intr[0, 2] -= offset_x
            intr[1, 2] -= offset_y

        # resize to render_tgt_size for training

        tgt_hw_size, ratio_y, ratio_x = self.calc_new_tgt_size_by_aspect(
            cur_hw=rgb.shape[:2],
            aspect_standard=aspect_standard,
            tgt_size=render_tgt_size,
            multiply=multiply,
        )

        rgb = cv2.resize(
            rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
        )
        mask = cv2.resize(
            mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
        )

        if intr is not None:

            # ******************** Merge *********************** #
            intr = self.scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
            assert (
                abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5
            ), f"{intr[0, 2] * 2}, {rgb.shape[1]}"
            assert (
                abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5
            ), f"{intr[1, 2] * 2}, {rgb.shape[0]}"

            # ******************** Merge *********************** #
            intr[0, 2] = rgb.shape[1] // 2
            intr[1, 2] = rgb.shape[0] // 2

        rgb = torch.from_numpy(rgb).float().permute(2, 0, 1)  # [3, H, W]
        mask = (
            torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1)
        )  # [1, H, W]
        return rgb, mask, intr


def gen_valid_id_json():
    root_dir = "./train_data/vfhq_vhap/export"
    save_path = "./train_data/vfhq_vhap/label/valid_id_list.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    valid_id_list = []
    for file in os.listdir(root_dir):
        if not file.startswith("."):
            valid_id_list.append(file)
    print(len(valid_id_list), valid_id_list[:2])
    with open(save_path, "w") as fp:
        json.dump(valid_id_list, fp)


def gen_valid_id_json():
    root_dir = "./train_data/vfhq_vhap/export"
    mask_root_dir = "./train_data/vfhq_vhap/mask"
    save_path = "./train_data/vfhq_vhap/label/valid_id_list.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    valid_id_list = []
    for file in os.listdir(root_dir):
        if not file.startswith(".") and ".txt" not in file:
            valid_id_list.append(file)
    print("raw:", len(valid_id_list), valid_id_list[:2])

    mask_valid_id_list = []
    for file in os.listdir(mask_root_dir):
        if not file.startswith(".") and ".txt" not in file:
            mask_valid_id_list.append(file)
    print("mask:", len(mask_valid_id_list), mask_valid_id_list[:2])

    valid_id_list = list(set(valid_id_list).intersection(set(mask_valid_id_list)))
    print("intesection:", len(mask_valid_id_list), mask_valid_id_list[:2])

    with open(save_path, "w") as fp:
        json.dump(valid_id_list, fp)
        
    save_train_path = "./train_data/vfhq_vhap/label/valid_id_train_list.json"
    save_val_path = "./train_data/vfhq_vhap/label/valid_id_val_list.json"
    valid_id_list = sorted(valid_id_list)
    idxs = np.linspace(0, len(valid_id_list)-1, num=20, endpoint=True).astype(np.int64)
    valid_id_train_list = []
    valid_id_val_list = []
    for i in range(len(valid_id_list)):
        if i in idxs:
            valid_id_val_list.append(valid_id_list[i])
        else:
            valid_id_train_list.append(valid_id_list[i])

    print(len(valid_id_train_list), len(valid_id_val_list), valid_id_val_list)
    with open(save_train_path, "w") as fp:
        json.dump(valid_id_train_list, fp)
        
    with open(save_val_path, "w") as fp:
        json.dump(valid_id_val_list, fp)


class MultiSourceMiniVarenAvatarDataset(MiniVarenAvatarDataset):
    """
    Extension of MiniVarenAvatarDataset that supports a variable number of source
    images as input.

    Each call to inner_get_item randomly draws the number of source views from
    cfg.DATASETS.NUM_SOURCE_VIEWS_CHOICES (default [1, 2, 4, 8, 16]).  If the
    sequence has fewer available views than the sampled count, the count is
    clamped to the actual number of available views.

    To allow DataLoader batching despite the variable source count, source tensors
    are always padded to cfg.DATASETS.MAX_SOURCE_VIEWS (default max of choices)
    slots with zeros.  A boolean mask ``source_valid_mask`` of shape
    [MAX_SOURCE_VIEWS] indicates which slots hold real data.

    Returned dict vs. parent:
        source_c2ws        : [MAX_SOURCE_VIEWS, 4, 4]    (zero-padded)
        source_intrs       : [MAX_SOURCE_VIEWS, 4, 4]    (zero-padded)
        source_rgbs        : [MAX_SOURCE_VIEWS, C, H, W] (zero-padded)
        source_valid_mask  : [MAX_SOURCE_VIEWS]  bool, True for real slots
        source_smal_params : per-view fields padded to MAX_SOURCE_VIEWS
    """

    def __init__(self, cfg: CfgNode, root_dir: str, is_train: bool = True):
        super().__init__(cfg, root_dir=root_dir, is_train=is_train)
        self.num_source_views_choices = list(
            cfg.DATASETS.get("NUM_SOURCE_VIEWS_CHOICES", [1, 2, 4, 8, 16])
        )
        assert all(n >= 1 for n in self.num_source_views_choices), \
            "All NUM_SOURCE_VIEWS_CHOICES must be >= 1"
        self.max_source_views = int(
            cfg.DATASETS.get("MAX_SOURCE_VIEWS", max(self.num_source_views_choices))
        )

    def inner_get_item(self, idx):
        # ------------------------------------------------------------------ #
        # Sampling strategy:
        #   1. Randomly pick num_src from NUM_SOURCE_VIEWS_CHOICES.
        #   2. If Nv < num_src, clamp to the largest valid choice <= Nv.
        #   3. If Nv >= num_src + render_views: sample that many distinct
        #      frames and split into non-overlapping source / render groups.
        #      Otherwise allow overlap between source and render.
        # ------------------------------------------------------------------ #
        seq_name = self.sequence[idx]
        uid = os.path.join(self.root_dirs, seq_name)

        # ---- Determine available views (Nv) ----
        sequence_info = self._get_sequence_info(uid)
        Nv = sequence_info["num_views"]

        # ---- Step 1: Pick number of source views, clamp to available ----
        num_src = int(np.random.choice(self.num_source_views_choices))
        if num_src > Nv:
            valid_choices = sorted([c for c in self.num_source_views_choices if c <= Nv])
            num_src = valid_choices[-1] if valid_choices else 1

        render_views = self.sample_side_views

        # ---- Step 2: Total distinct frames to sample & overlap strategy ----
        if Nv >= num_src + render_views:
            sample_total = num_src + render_views
            allow_overlap = False
        else:
            sample_total = min(Nv, num_src + render_views)
            allow_overlap = True

        # ---- Step 3: Sample sample_total frame indices from [0, Nv) ----
        # sample_total = min(Nv, ...) guarantees sample_total <= Nv, so replace=False is always valid.
        if not self.is_val:
            if np.random.rand() < 0.7 and Nv > sample_total:
                frame_id_list = self.uniform_sample_in_chunk(sample_total, np.arange(Nv))
            else:
                frame_id_list = np.random.choice(Nv, size=sample_total, replace=False).tolist()
        else:
            if Nv > sample_total:
                frame_id_list = self.uniform_sample_in_chunk_det(sample_total, np.arange(Nv))
            else:
                frame_id_list = list(range(Nv))
        frame_id_list = list(map(int, frame_id_list))

        # ---- Step 4: Split local indices into source vs. render ----
        if not allow_overlap:
            # Enough views: randomly assign num_src as source, rest as render.
            perm = np.random.permutation(sample_total).tolist()
            src_local = sorted(perm[:num_src])
            render_local = sorted(perm[num_src:])
        else:
            # Not enough to fully separate: allow overlap.
            num_src = min(num_src, sample_total)
            src_local = sorted(
                np.random.choice(sample_total, size=num_src, replace=False).tolist()
            )
            # Render: randomly pick render_views indices (with replacement when insufficient).
            if sample_total >= render_views:
                render_local = sorted(np.random.choice(sample_total, size=render_views, replace=False).tolist())
            else:
                render_local = list(range(sample_total))
                extra = np.random.choice(sample_total, size=render_views - sample_total, replace=True).tolist()
                render_local.extend(extra)

        # ---- Step 5: Load cameras, images, intrinsics for sampled frames ----
        shard_entry = sequence_info.get("shard_entry")
        if sequence_info["use_concat_image"]:
            if shard_entry is not None:
                images_all, mask_images_all = self._load_shard_concat_views(shard_entry)
            else:
                images_all, mask_images_all = self._load_concat_views(sequence_info["image_file"], sequence_info["mask_file"])
            images = [images_all[i] for i in frame_id_list]
            mask_images = [mask_images_all[i] for i in frame_id_list]
        else:
            if shard_entry is not None:
                image_members = shard_entry["members"]["images"]
                mask_members = shard_entry["members"]["masks"]
                image_pairs = [self._load_shard_frame_pair(shard_entry, image_members[i], mask_members[i]) for i in frame_id_list]
            else:
                image_file_list = sequence_info["image_files"]
                mask_file_list = sequence_info["mask_files"]
                image_pairs = [self._load_frame_pair(image_file_list[i], mask_file_list[i]) for i in frame_id_list]
            images = [pair[0] for pair in image_pairs]
            mask_images = [pair[1] for pair in image_pairs]

        camera_tensors = self._load_camera_tensors(uid)
        if camera_tensors is not None:
            c2ws_full, mvp_matrix_full = camera_tensors
            c2ws_raw = c2ws_full[frame_id_list]
            mvp_matrix_raw = mvp_matrix_full[frame_id_list]
            proj_all = torch.bmm(mvp_matrix_raw, c2ws_raw)
            c2ws_raw = c2ws_raw.float()
            proj_all = proj_all.float()
            assert c2ws_raw.shape[0] == proj_all.shape[0]
            intrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(sample_total, 1, 1)
            for i, proj in enumerate(proj_all):
                fx, fy, cx, cy = self.compute_intrinsics_from_projection(proj, images[i].shape[1], images[i].shape[0])
                intrinsics[i, 0, 0] = fx
                intrinsics[i, 1, 1] = fy
                intrinsics[i, 0, 2] = cx
                intrinsics[i, 1, 2] = cy
        else:
            c2ws_raw = torch.eye(4, dtype=torch.float32)
            c2ws_raw[1, 1] = c2ws_raw[2, 2] = -1.0
            c2ws_raw = c2ws_raw.unsqueeze(0).repeat(sample_total, 1, 1)
            intrinsics = torch.eye(4, dtype=torch.float32)
            intrinsics[0, 2] = images[0].shape[1] // 2
            intrinsics[1, 2] = images[0].shape[0] // 2
            intrinsics[0, 0] = intrinsics[1, 1] = 5000. / 256. * max(images[0].shape[0], images[0].shape[1])
            intrinsics = intrinsics.unsqueeze(0).repeat(sample_total, 1, 1)

        # Load VAREN parameters
        params_path = os.path.join(uid, f'{seq_name}.npz')
        if not self.use_shards:
            assert os.path.exists(params_path), f"Missing {seq_name}.npz in {uid}"
        varen_params_all = self._load_varen_params_cached(params_path, num_views=Nv)

        # ---- Step 6: Process all sample_total frames ----
        c2ws, intrs, rgbs, bg_colors, masks = [], [], [], [], []
        S_mat = torch.tensor([[1., 0., 0., 0.],
                              [0.,-1., 0., 0.],
                              [0., 0.,-1., 0.],
                              [0., 0., 0., 1.]], dtype=torch.float32)
        for i in range(sample_total):
            c2w = c2ws_raw[i] @ S_mat
            bg_color = random.choice([0.0, 0.5, 1.0])
            rgb, mask_t, intrinsic = self.load_rgb_image_with_aug_bg(
                images[i],
                mask_images[i],
                bg_color=bg_color,
                max_tgt_size=None,
                aspect_standard=self.aspect_standard,
                enlarge_ratio=self.enlarge_ratio,
                render_tgt_size=self.render_image_res,
                multiply=self.multiply,
                intr=intrinsics[i].clone(),
            )
            c2ws.append(c2w)
            intrs.append(intrinsic)
            rgbs.append(rgb)
            bg_colors.append(bg_color)
            masks.append(mask_t)

        c2ws = torch.stack(c2ws, dim=0)     # [sample_total, 4, 4]
        intrs = torch.stack(intrs, dim=0)    # [sample_total, 4, 4]
        rgbs = torch.cat(rgbs, dim=0)        # [sample_total, C, H, W]
        masks = torch.cat(masks, dim=0)      # [sample_total, 1, H, W]
        bg_colors_t = torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)

        # VAREN params indexed by frame_id_list (local order)
        varen_params = {}
        if 'betas' in varen_params_all:
            varen_params['betas'] = varen_params_all['betas']
        if 'global_orient' in varen_params_all:
            varen_params['global_orient'] = varen_params_all['global_orient'][frame_id_list]
        if 'pose' in varen_params_all:
            varen_params['pose'] = varen_params_all['pose'][frame_id_list]
        if 'trans' in varen_params_all:
            varen_params['trans'] = varen_params_all['trans'][frame_id_list]
        if 'tail_scale' in varen_params_all:
            varen_params['tail_scale'] = varen_params_all['tail_scale']

        # ---- Step 7: Build render tensors (using render_local) ----
        render_image = rgbs[render_local]          # [render_views, C, H, W]
        render_mask = masks[render_local]           # [render_views, 1, H, W]
        render_c2ws = c2ws[render_local]            # [render_views, 4, 4]
        render_intrs = intrs[render_local]          # [render_views, 4, 4]
        render_bg = bg_colors_t[render_local]       # [render_views, 3]

        render_varen_params = {}
        if 'betas' in varen_params:
            render_varen_params['betas'] = varen_params['betas']
        if 'global_orient' in varen_params:
            render_varen_params['global_orient'] = varen_params['global_orient'][render_local]
        if 'pose' in varen_params:
            render_varen_params['pose'] = varen_params['pose'][render_local]
        if 'trans' in varen_params:
            render_varen_params['trans'] = varen_params['trans'][render_local]
        if 'tail_scale' in varen_params:
            render_varen_params['tail_scale'] = varen_params['tail_scale']

        # ---- Step 8: Build source tensors, padded to max_source_views ----
        source_rgbs_valid  = rgbs[src_local]    # [num_src, C, H, W]
        source_masks_valid = masks[src_local]   # [num_src, 1, H, W]
        source_bg_valid = bg_colors_t[src_local] # [num_src, 3]
        if not self.is_val:
            source_rgbs_valid, source_masks_valid = self._apply_source_augmentations(
                source_rgbs_valid,
                source_masks_valid,
                source_bg_valid,
            )
        source_c2ws_valid  = c2ws[src_local]    # [num_src, 4, 4]
        source_intrs_valid = intrs[src_local]   # [num_src, 4, 4]

        max_src = self.max_source_views
        C_img, H_img, W_img = source_rgbs_valid.shape[1:]

        source_rgbs = torch.zeros(max_src, C_img, H_img, W_img, dtype=source_rgbs_valid.dtype)
        source_rgbs[:num_src] = source_rgbs_valid
        source_masks = torch.zeros(max_src, 1, H_img, W_img, dtype=source_masks_valid.dtype)
        source_masks[:num_src] = source_masks_valid
        source_c2ws = torch.zeros(max_src, 4, 4, dtype=torch.float32)
        source_c2ws[:num_src] = source_c2ws_valid
        source_intrs = torch.zeros(max_src, 4, 4, dtype=torch.float32)
        source_intrs[:num_src] = source_intrs_valid
        source_valid_mask = torch.zeros(max_src, dtype=torch.bool)
        source_valid_mask[:num_src] = True

        source_varen_params = {}
        if 'betas' in varen_params:
            source_varen_params['betas'] = varen_params['betas']
        if 'global_orient' in varen_params:
            go_v = varen_params['global_orient'][src_local]
            go_pad = torch.zeros(max_src, *go_v.shape[1:], dtype=go_v.dtype)
            go_pad[:num_src] = go_v
            source_varen_params['global_orient'] = go_pad
        if 'pose' in varen_params:
            pose_v = varen_params['pose'][src_local]
            pose_pad = torch.zeros(max_src, *pose_v.shape[1:], dtype=pose_v.dtype)
            pose_pad[:num_src] = pose_v
            source_varen_params['pose'] = pose_pad
        if 'trans' in varen_params:
            trans_v = varen_params['trans'][src_local]
            trans_pad = torch.zeros(max_src, *trans_v.shape[1:], dtype=trans_v.dtype)
            trans_pad[:num_src] = trans_v
            source_varen_params['trans'] = trans_pad
        if 'tail_scale' in varen_params:
            source_varen_params['tail_scale'] = varen_params['tail_scale']

        # ---- Assemble return dict ----
        tgt_size = render_image.shape[2:4]
        assert abs(render_intrs[0, 0, 2] * 2 - render_image.shape[3]) <= 2.5
        assert abs(render_intrs[0, 1, 2] * 2 - render_image.shape[2]) <= 2.5

        ret = {
            'uid': uid,
            'source_c2ws': source_c2ws,
            'source_intrs': source_intrs,
            'source_rgbs': source_rgbs.clamp(0, 1),
            'source_masks': source_masks.clamp(0, 1),
            'source_valid_mask': source_valid_mask,
            'render_image': render_image.clamp(0, 1),
            'render_mask': render_mask.clamp(0, 1),
            'c2ws': render_c2ws,
            'intrs': render_intrs,
            'render_full_resolutions': torch.tensor([tgt_size], dtype=torch.float32).repeat(render_views, 1),
            'render_bg_colors': render_bg,
            'pytorch3d_transpose_R': torch.tensor([0.0]),
        }
        ret["smal_params"] = render_varen_params
        ret["source_smal_params"] = source_varen_params
        return ret


class MergedMultiSourceMiniVarenAvatarDataset(MergedMiniVarenAvatarDataset):
    """MergedMiniVarenAvatarDataset that uses MultiSourceMiniVarenAvatarDataset
    as the per-root dataset class."""

    @classmethod
    def _dataset_cls(cls):
        return MultiSourceMiniVarenAvatarDataset


def debug_minivaren_avatar_dataset():
    """
    Debug MiniVarenAvatarDataset by loading one sample and saving images/masks.

    - ROOT: /data2/lvjin/cvpr26/data/VarenAvatar/
    - SAMPLE_SIDE_VIEWS: 4
    - Other arguments use dataset defaults
    """
    import os
    import traceback

    # Build minimal config
    cfg = CfgNode(new_allowed=True)
    cfg.DATASETS = CfgNode(new_allowed=True)
    cfg.DATASETS.VarenAvatar = CfgNode(new_allowed=True)
    cfg.DATASETS.VarenAvatar.ROOT = "/data2/lvjin/4DEquine/Animal4DEval/"
    # MiniVarenAvatarDataset reads shared settings from cfg.DATASETS directly.
    cfg.DATASETS.SAMPLE_SIDE_VIEWS = 4
    cfg.DATASETS.RENDER_IMAGE_RES_LOW = 448
    cfg.DATASETS.RENDER_IMAGE_RES_HIGH = 448
    cfg.DATASETS.RENDER_REGION_SIZE = 448
    cfg.DATASETS.SOURCE_IMAGE_RES = 448

    dataset = MiniVarenAvatarDataset(cfg, root_dir=cfg.DATASETS.VarenAvatar.ROOT, is_train=False)

    print(f"[MiniVarenAvatarDataset] length: {len(dataset)}")
    save_root = "./debug_vis/mini_varen_avatar_dataset"
    os.makedirs(save_root, exist_ok=True)

    try:
        data = dataset[0]
    except Exception as e:
        traceback.print_exc()
        print(f"[DEBUG] Failed to load first sample: {e}")
        try:
            seq0 = dataset.sequence[0] if hasattr(dataset, "sequence") and len(dataset.sequence) > 0 else None
            print(f"First sequence: {seq0}")
        except Exception:
            pass
        return

    # Print keys and shapes
    print("[MiniVarenAvatarDataset] Sample 0 keys and shapes:")
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            try:
                print(f"  - {k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
            except Exception:
                print(f"  - {k}: tensor")
        else:
            print(f"  - {k}: type={type(v)}")

    def save_tensor_as_image(t: torch.Tensor, path: str):
        img = t.detach().cpu().clamp(0, 1)
        if img.dim() == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0).numpy()
        else:
            img = img.numpy()
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        img = (img * 255.0).astype(np.uint8)
        # OpenCV expects BGR
        img = img[..., ::-1]
        cv2.imwrite(path, img)

    # Save source images
    if isinstance(data.get("source_rgbs"), torch.Tensor):
        src = data["source_rgbs"]
        for i in range(min(4, src.shape[0])):
            save_tensor_as_image(src[i], os.path.join(save_root, f"source_{i}.jpg"))

    # Save render images and masks
    if isinstance(data.get("render_image"), torch.Tensor):
        rgbs = data["render_image"]
        for i in range(min(4, rgbs.shape[0])):
            save_tensor_as_image(rgbs[i], os.path.join(save_root, f"render_{i}.jpg"))

    if isinstance(data.get("render_mask"), torch.Tensor):
        mks = data["render_mask"]
        for i in range(min(4, mks.shape[0])):
            m = mks[i]
            if m.shape[0] == 1:
                m = m.repeat(3, 1, 1)
            save_tensor_as_image(m, os.path.join(save_root, f"mask_{i}.jpg"))

    try:
        import sys
        ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from amr.models.rendering.gs_varen_renderer import GS3DRenderer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Minimal renderer config
        cfg_render = CfgNode(new_allowed=True)
        cfg_render.RENDERER = CfgNode(new_allowed=True)
        cfg_render.RENDERER.VAREN_MODEL_PATH = 'data/varen/'
        cfg_render.RENDERER.SUBDIVIDE_NUM = 1
        cfg_render.RENDERER.FEAT_DIM = 64
        cfg_render.RENDERER.QUERY_DIM = 64
        cfg_render.RENDERER.USE_RGB = True
        cfg_render.RENDERER.SH_DEGREE = 3
        cfg_render.RENDERER.XYZ_OFFSET_MAX_STEP = 1.0
        cfg_render.RENDERER.MLP_NETWORK_CONFIG = CfgNode(new_allowed=True)
        cfg_render.RENDERER.MLP_NETWORK_CONFIG.activation = 'silu'
        cfg_render.RENDERER.MLP_NETWORK_CONFIG.n_hidden_layers = 2
        cfg_render.RENDERER.MLP_NETWORK_CONFIG.n_neurons = 512
        cfg_render.RENDERER.SHAPE_PARAM_DIM = 39
        cfg_render.RENDERER.CLIP_SCALING = [100, 1000, 1000, 3000]
        cfg_render.RENDERER.DECODER_MLP = False
        cfg_render.RENDERER.SKIP_DECODER = True
        cfg_render.RENDERER.FIX_OPACITY = False
        cfg_render.RENDERER.FIX_ROTATION = False
        cfg_render.RENDERER.DECODE_WITH_EXTRA_INFO = None
        cfg_render.RENDERER.GRADIENT_CHECKPOINTING = False
        cfg_render.RENDERER.APPLY_POSE_BLENDSHAPE = False

        renderer = GS3DRenderer(cfg_render).to(device)

        # Build varen_data from dataset output
        def to_batched_tensor(t, is_betas=False):
            t = t.to(dtype=torch.float32, device=device)
            if is_betas:
                if t.dim() == 1:
                    t = t.unsqueeze(0)  # [D] -> [1, D]
                return t
            if t.dim() == 2 and t.shape[-1] == 3:
                # [Nv, 3] -> [1, Nv, 1, 3]
                t = t.unsqueeze(0).unsqueeze(2)
            elif t.dim() == 3 and t.shape[-1] == 3:
                # [Nv, J, 3] or [Nv, 1, 3] -> [1, Nv, J/1, 3]
                t = t.unsqueeze(0)
            return t

        varen_data = {}
        params = data['smal_params']
        if 'betas' in params:
            varen_data['betas'] = to_batched_tensor(params['betas'], is_betas=True)
        if 'global_orient' in params:
            varen_data['global_orient'] = to_batched_tensor(params['global_orient'])
        if 'pose' in params:
            varen_data['pose'] = to_batched_tensor(params['pose'])
        if 'trans' in params:
            varen_data['trans'] = to_batched_tensor(params['trans'])
        if 'tail_scale' in params:
            varen_data['tail_scale'] = to_batched_tensor(params['tail_scale'], is_betas=True)

        # Query canonical points
        positions, varen_data = renderer.get_query_points(varen_data, device)

        B = positions.shape[0]
        N = positions.shape[1]
        query_dim = cfg_render.RENDERER.QUERY_DIM
        gs_hidden_features = torch.zeros((B, N, query_dim), dtype=torch.float32, device=device)

        # Prepare cameras and intrinsics
        c2w = data['c2ws'].to(device=device, dtype=torch.float32).unsqueeze(0)
        intr = data['intrs'].to(device=device, dtype=torch.float32).unsqueeze(0)
        H = int(data['render_image'].shape[-2])
        W = int(data['render_image'].shape[-1])
        bg = data.get('render_bg_colors', torch.ones(c2w.shape[1], 3, dtype=torch.float32)).to(device=device)
        bg = bg.unsqueeze(0)

        out = renderer.forward(
            gs_hidden_features=gs_hidden_features,
            query_points=positions,
            varen_data=varen_data,
            c2w=c2w,
            intrinsic=intr,
            height=H,
            width=W,
            additional_features=None,
            background_color=bg,
            debug=False,
            df_data=None,
        )

        # Save GS render outputs
        gs_save = os.path.join(save_root, "gs_render")
        os.makedirs(gs_save, exist_ok=True)
        comp_rgb = out.get('comp_rgb')  # [B, Nv, 3, H, W]
        if isinstance(comp_rgb, torch.Tensor):
            comp_rgb_np = (comp_rgb[0].permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
            for i in range(min(4, comp_rgb_np.shape[0])):
                # OpenCV expects BGR
                cv2.imwrite(os.path.join(gs_save, f"gs_render_{i}.jpg"), comp_rgb_np[i][..., ::-1])
        comp_mask = out.get('comp_mask')  # [B, Nv, 1, H, W]
        if isinstance(comp_mask, torch.Tensor):
            mask_np = (comp_mask[0, :, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)
            for i in range(min(4, mask_np.shape[0])):
                cv2.imwrite(os.path.join(gs_save, f"gs_mask_{i}.jpg"), mask_np[i])
        print(f"[MiniVarenAvatarDataset] GS render saved to {gs_save}")
    except Exception as e:
        traceback.print_exc()
        print(f"[DEBUG] GS render failed: {e}")

    print(f"[MiniVarenAvatarDataset] Saved debug outputs to {save_root}")


def debug_minivaren_avatar_dataset_2():
    """
    For loop all samples in the dataset
    - ROOT: /data/lvjin/cvpr26/data/VarenTex/
    - SAMPLE_SIDE_VIEWS: 4
    - Other arguments use dataset defaults
    """
    # Build minimal config
    cfg = CfgNode(new_allowed=True)
    cfg.DATASETS = CfgNode(new_allowed=True)
    cfg.DATASETS.VarenAvatar = CfgNode(new_allowed=True)
    cfg.DATASETS.VarenAvatar.ROOT = "/data/lvjin/cvpr26/data/VarenTex/"
    cfg.DATASETS.SAMPLE_SIDE_VIEWS = 4

    dataset = MiniVarenAvatarDataset(cfg, root_dir=cfg.DATASETS.VarenAvatar.ROOT, is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, 
                                            num_workers=8, pin_memory=True, prefetch_factor=8)

    from tqdm import tqdm
    for item in tqdm(dataloader, total=len(dataloader)):
        continue


def debug_merged_minivaren_avatar_dataset():
    """
    Iterate over a merged multi-root MiniVarenAvatarDataset for quick sanity checks.
    Uses the hrm24g config-style fields: three roots with equal weights.
    """
    cfg = CfgNode(new_allowed=True)
    cfg.DATASETS = CfgNode(new_allowed=True)
    # Multiple datasets with weights to test merged sampling
    for name, root in [
        ("VarenAvatar", "/data2/lvjin/cvpr26/data/VarenTex"),
        ("VarenWan", "/data2/lvjin/4DEquine/VarenWan"),
        ("Animal4D", "/data2/lvjin/4DEquine/Animal4D"),
    ]:
        cfg.DATASETS[name] = CfgNode(new_allowed=True)
        cfg.DATASETS[name].ROOT = root
        cfg.DATASETS[name].WEIGHT = 1.0
    cfg.DATASETS.SAMPLE_SIDE_VIEWS = 4
    cfg.DATASETS.RENDER_IMAGE_RES_LOW = 448
    cfg.DATASETS.RENDER_IMAGE_RES_HIGH = 448
    cfg.DATASETS.RENDER_REGION_SIZE = 448
    cfg.DATASETS.SOURCE_IMAGE_RES = 448
    cfg.SEED = 42

    dataset = MergedMiniVarenAvatarDataset(cfg, is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False,
                                             num_workers=8, pin_memory=True, prefetch_factor=2,
                                             drop_last=True)

    from tqdm import tqdm
    for item in tqdm(dataloader, total=len(dataset) // 8, dynamic_ncols=True):
        # Just iterate a few batches to verify shape/key consistency
        assert "dataset_name" in item, "Merged dataset should include dataset_name key"
        assert "source_rgbs" in item, "Expected source_rgbs in sample"


def debug_multi_source_dataset():
    """
    Debug MultiSourceMiniVarenAvatarDataset:
      - Verify source / render views are non-overlapping when Nv is sufficient.
      - Verify shape correctness and padding.
      - Save source & render images for visual inspection.

    Dataset paths from hrm_mv_A100.yaml.
    """
    from tqdm import tqdm

    cfg = CfgNode(new_allowed=True)
    cfg.DATASETS = CfgNode(new_allowed=True)
    for name, root, weight, is_val in [
        ("VarenAvatar", "/backup/lvjin/tpami_4DEquinePP/data/VarenTex", 0.5, False),
        ("VarenWan", "/backup/lvjin/tpami_4DEquinePP/data/VarenWanv1", 1.0, False),
        ("Animal4D", "/backup/lvjin/tpami_4DEquinePP/data/Animal4D", 0.1, False),
        ("Animal4DEval", "/backup/lvjin/tpami_4DEquinePP/data/Animal4DEval/", None, True),
    ]:
        cfg.DATASETS[name] = CfgNode(new_allowed=True)
        cfg.DATASETS[name].ROOT = root
        cfg.DATASETS[name].IS_VAL = is_val
        if weight is not None:
            cfg.DATASETS[name].WEIGHT = weight
    cfg.DATASETS.SAMPLE_SIDE_VIEWS = 4
    cfg.DATASETS.RENDER_IMAGE_RES_LOW = 448
    cfg.DATASETS.RENDER_IMAGE_RES_HIGH = 448
    cfg.DATASETS.RENDER_REGION_SIZE = 448
    cfg.DATASETS.SOURCE_IMAGE_RES = 448
    cfg.DATASETS.NUM_SOURCE_VIEWS_CHOICES = [1, 2, 4, 8, 16]
    cfg.DATASETS.MAX_SOURCE_VIEWS = 16
    cfg.SEED = 42

    render_views = cfg.DATASETS.SAMPLE_SIDE_VIEWS
    max_src = cfg.DATASETS.MAX_SOURCE_VIEWS

    save_root = "./debug_vis/multi_source_dataset"
    os.makedirs(save_root, exist_ok=True)

    def save_tensor_as_image(t, path):
        img = t.detach().cpu().clamp(0, 1)
        if img.dim() == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0).numpy()
        else:
            img = img.numpy()
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        img = (img * 255.0).astype(np.uint8)[..., ::-1]
        cv2.imwrite(path, img)

    # ---- Test single dataset ----
    print("=" * 60)
    print("Testing MultiSourceMiniVarenAvatarDataset (single root)")
    print("=" * 60)
    ds = MultiSourceMiniVarenAvatarDataset(
        cfg, root_dir="/backup/lvjin/tpami_4DEquinePP/data/VarenTex/", is_train=False,
    )
    print(f"  Dataset length: {len(ds)}")

    for sample_idx in range(min(5, len(ds))):
        data = ds[sample_idx]
        num_valid = int(data["source_valid_mask"].sum().item())
        print(f"\n  [Sample {sample_idx}] uid={data['uid']}")
        print(f"    source_rgbs      : {tuple(data['source_rgbs'].shape)}")
        print(f"    source_valid_mask: {data['source_valid_mask'].tolist()}  (num_valid={num_valid})")
        print(f"    render_image     : {tuple(data['render_image'].shape)}")
        print(f"    c2ws (render)    : {tuple(data['c2ws'].shape)}")
        print(f"    intrs (render)   : {tuple(data['intrs'].shape)}")

        # Shape checks
        assert data["source_rgbs"].shape[0] == max_src, \
            f"source_rgbs dim-0 should be {max_src}, got {data['source_rgbs'].shape[0]}"
        assert data["source_valid_mask"].shape[0] == max_src
        assert data["render_image"].shape[0] == render_views, \
            f"render_image dim-0 should be {render_views}, got {data['render_image'].shape[0]}"
        assert data["c2ws"].shape[0] == render_views
        assert data["intrs"].shape[0] == render_views
        assert data["render_bg_colors"].shape[0] == render_views

        # Verify padding slots are zero
        if num_valid < max_src:
            assert (data["source_rgbs"][num_valid:] == 0).all(), \
                "Padded source slots should be zero"

        # Save images for visual inspection (first sample only)
        if sample_idx == 0:
            for i in range(num_valid):
                save_tensor_as_image(data["source_rgbs"][i],
                                     os.path.join(save_root, f"src_{i}.jpg"))
            for i in range(render_views):
                save_tensor_as_image(data["render_image"][i],
                                     os.path.join(save_root, f"render_{i}.jpg"))
            print(f"    Saved images to {save_root}")

    # ---- Test merged dataset with dataloader ----
    print("\n" + "=" * 60)
    print("Testing MergedMultiSourceMiniVarenAvatarDataset (train)")
    print("=" * 60)
    merged = MergedMultiSourceMiniVarenAvatarDataset(cfg, is_train=True)
    print(f"  Merged dataset length: {len(merged)}")
    loader = torch.utils.data.DataLoader(
        merged, batch_size=4, shuffle=True, num_workers=4,
        pin_memory=True, prefetch_factor=2, drop_last=True,
    )

    num_batches = 0
    num_overlap = 0
    num_no_overlap = 0
    for batch in tqdm(loader, total=min(50, len(loader)), desc="  Iterating batches"):
        B = batch["source_rgbs"].shape[0]
        assert batch["source_rgbs"].shape == (B, max_src, 3, 448, 448), \
            f"Unexpected source_rgbs shape: {batch['source_rgbs'].shape}"
        assert batch["render_image"].shape[0] == B
        assert batch["render_image"].shape[1] == render_views

        # Check source-render overlap by comparing pixel content
        for b in range(B):
            n_valid = int(batch["source_valid_mask"][b].sum().item())
            src_pixels = batch["source_rgbs"][b, :n_valid].flatten(1)   # [n_valid, C*H*W]
            rnd_pixels = batch["render_image"][b].flatten(1)            # [render_views, C*H*W]
            # Check if any source image is pixel-identical to any render image
            has_overlap = False
            for s in range(n_valid):
                for r in range(render_views):
                    if torch.allclose(src_pixels[s], rnd_pixels[r], atol=1e-5):
                        has_overlap = True
                        break
                if has_overlap:
                    break
            if has_overlap:
                num_overlap += 1
            else:
                num_no_overlap += 1

        num_batches += 1
        if num_batches >= 50:
            break

    total_samples = num_overlap + num_no_overlap
    print(f"\n  Checked {total_samples} samples across {num_batches} batches:")
    print(f"    No overlap (src != render): {num_no_overlap} ({100.*num_no_overlap/max(total_samples,1):.1f}%)")
    print(f"    Has overlap (src in render): {num_overlap} ({100.*num_overlap/max(total_samples,1):.1f}%)")
    print("  (Overlap is expected only when Nv < num_src + render_views)")
    print("\nAll checks passed!")


if __name__ == "__main__":
    # debug_minivaren_avatar_dataset()
    # debug_minivaren_avatar_dataset_2()
    # debug_merged_minivaren_avatar_dataset()
    debug_multi_source_dataset()
