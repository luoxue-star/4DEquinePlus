from typing import Dict, Optional
from torch.utils.data import DistributedSampler, WeightedRandomSampler
import torch
import pytorch_lightning as pl
from yacs.config import CfgNode
from omegaconf import DictConfig
import torch.distributed as dist
from amr.datasets.avatar_dataset import (
    MiniVarenAvatarDataset,
    MergedMiniVarenAvatarDataset,
    MultiSourceMiniVarenAvatarDataset,
    MergedMultiSourceMiniVarenAvatarDataset,
)
from .varen_dataset import *
from amr.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def _worker_kwargs(num_workers, prefetch_factor=None):
    kwargs = {"num_workers": num_workers}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


class VARENDataModule(pl.LightningDataModule):

    def __init__(self, cfg: CfgNode) -> None:
        """
        Initialize LightningDataModule for AMR training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
        """
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load datasets necessary for training
        Args:
            stage:
        """
        if self.train_dataset is None:
            self.train_dataset = VARENTrainMixedDataset(self.cfg, is_train=True)
        if self.val_dataset is None:
            self.val_dataset = VarenEvalTemporalDataset(self.cfg, dataset_name="APT36K", train=False)

    def train_dataloader(self) -> Dict:
        """
        Setup training data loader.
        Returns:
            Dict: Dictionary containing image and mocap data dataloaders
        """
        try:
            train_dataloader = self.train_dataset.get_loader()
        except:
            train_dataloader = torch.utils.data.DataLoader(self.train_dataset, self.cfg.TRAIN.BATCH_SIZE.get("IMAGE", 16), drop_last=True,
                                                        pin_memory=True,
                                                        shuffle=True,
                                                        **_worker_kwargs(self.cfg.GENERAL.NUM_WORKERS, self.cfg.GENERAL.PREFETCH_FACTOR),
                                                        )
        return {'img': train_dataloader}

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Setup val data loader.
        Returns:
            torch.utils.data.DataLoader: Validation dataloader
        """
        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, 1, drop_last=True,
                                                     pin_memory=True,
                                                     **_worker_kwargs(self.cfg.GENERAL.NUM_WORKERS))
        return val_dataloader


class AvatarDataModule(pl.LightningDataModule):
    def __init__(self, cfg: CfgNode) -> None:
        """
        Initialize LightningDataModule for AMR training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
        """
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @staticmethod
    def _parse_dataset_node(node):
        """
        Normalize dataset entry coming from either yacs CfgNode, OmegaConf DictConfig
        or plain dict. Returns (root, is_val, weight) or None if unsupported.
        """
        if isinstance(node, (CfgNode, DictConfig, dict)):
            root = node.get("SHARD_INDEX", node.get("ROOT", None))
            is_val = bool(node.get("IS_VAL", False))
            weight = node.get("WEIGHT", None)
            return root, is_val, weight
        if hasattr(node, "get"):
            try:
                root = node.get("SHARD_INDEX", node.get("ROOT", None))
                is_val = bool(node.get("IS_VAL", False))
                weight = node.get("WEIGHT", None)
                return root, is_val, weight
            except Exception:
                return None
        return None

    def _select_avatar_dataset_entries(self, is_train: bool):
        target_is_val = not is_train
        dataset_entries = []
        for node in self.cfg.DATASETS.values():
            parsed = self._parse_dataset_node(node)
            if parsed is None:
                continue
            root, is_val_entry, _ = parsed
            if root is None or is_val_entry != target_is_val:
                continue
            dataset_entries.append(node)
        if len(dataset_entries) == 0:
            stage = "validation" if target_is_val else "training"
            raise ValueError(
                f"No dataset with ROOT and IS_VAL={target_is_val} found in cfg.DATASETS for AvatarDataModule {stage} stage."
            )
        return dataset_entries

    def _build_avatar_dataset(self, is_train: bool):
        dataset_entries = self._select_avatar_dataset_entries(is_train=is_train)
        use_multi_source = bool(self.cfg.DATASETS.get("NUM_SOURCE_VIEWS_CHOICES", None))
        if len(dataset_entries) == 1:
            root_dir, _, _ = self._parse_dataset_node(dataset_entries[0])
            if use_multi_source:
                return MultiSourceMiniVarenAvatarDataset(self.cfg, root_dir=root_dir, is_train=is_train)
            return MiniVarenAvatarDataset(self.cfg, root_dir=root_dir, is_train=is_train)
        if use_multi_source:
            return MergedMultiSourceMiniVarenAvatarDataset(self.cfg, is_train=is_train)
        return MergedMiniVarenAvatarDataset(self.cfg, is_train=is_train)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load datasets necessary for training
        Args:
            stage:
        """
        if self.train_dataset is None:
            self.train_dataset = self._build_avatar_dataset(is_train=True)
        if self.val_dataset is None:
            self.val_dataset = self._build_avatar_dataset(is_train=False)

    def train_dataloader(self) -> Dict:
        """
        Setup training data loader.
        Returns:
            Dict: Dictionary containing image and mocap data dataloaders
        """
        try:
            train_dataloader = self.train_dataset.get_loader()
        except:
            train_dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                                           self.cfg.TRAIN.BATCH_SIZE, 
                                                           drop_last=True,
                                                           pin_memory=True,
                                                           shuffle=True,
                                                           **_worker_kwargs(self.cfg.GENERAL.NUM_WORKERS, self.cfg.GENERAL.PREFETCH_FACTOR))
        return train_dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Setup val data loader.
        Returns:
            torch.utils.data.DataLoader: Validation dataloader
        """
        val_dataloader = torch.utils.data.DataLoader(self.val_dataset, 
                                                     self.cfg.TRAIN.BATCH_SIZE, 
                                                     drop_last=True,
                                                     num_workers=0, 
                                                     pin_memory=False)
        return val_dataloader
