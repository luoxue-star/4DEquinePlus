import os
from typing import Any, Dict, List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import TrainerFn


def fit_checkpoint_kwargs(cfg: Any) -> Dict[str, Any]:
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path is None:
        ckpt_path = "last"

    return {
        "ckpt_path": ckpt_path,
        "weights_only": False,
    }


class LatestCheckpointOnValidationEnd(Callback):
    """Always refresh last.ckpt after validation, independent of best-checkpoint ranking."""

    def __init__(self, dirpath: str, filename: str = "last.ckpt") -> None:
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self._last_global_step_saved = None

    @property
    def filepath(self) -> str:
        return os.path.join(self.dirpath, self.filename)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if (
            trainer.sanity_checking
            or trainer.state.fn != TrainerFn.FITTING
            or self._last_global_step_saved == trainer.global_step
        ):
            return

        os.makedirs(self.dirpath, exist_ok=True)
        trainer.save_checkpoint(self.filepath, weights_only=False)
        self._last_global_step_saved = trainer.global_step


def build_checkpoint_callbacks(cfg: Any) -> List[Callback]:
    checkpoint_dir = os.path.join(cfg.paths.output_dir, "checkpoints")
    best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        every_n_epochs=cfg.GENERAL.CHECKPOINT_EPOCHS,
        save_last=False,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
        monitor="val/loss",
        mode="min",
    )
    latest_checkpoint_callback = LatestCheckpointOnValidationEnd(
        dirpath=checkpoint_dir,
    )
    return [best_checkpoint_callback, latest_checkpoint_callback]
