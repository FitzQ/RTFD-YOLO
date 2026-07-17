from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseSegModel
from ultralytics.utils import DEFAULT_CFG, RANK


class PoseSegTrainer(yolo.detect.DetectionTrainer):
    """Trainer for a shared-backbone person pose and instance-segmentation model."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        overrides = overrides or {}
        overrides["task"] = "poseg"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights: str | Path | None = None, verbose: bool = True) -> PoseSegModel:
        if isinstance(cfg, dict) and cfg.get("head", [[None, None, None]])[-1][2] != "PoseSeg26":
            raise ValueError("poseg train requires a yolo26*-poseg.yaml model or a poseg checkpoint")
        model = PoseSegModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]
        self.model.kpt_names = self.data.get("kpt_names", {})

    def get_validator(self):
        self.loss_names = (
            "box_loss",
            "pose_loss",
            "kobj_loss",
            "cls_loss",
            "dfl_loss",
            "rle_loss",
            "seg_loss",
            "sem_loss",
        )
        return yolo.poseg.PoseSegValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        data = super().get_dataset()
        for key in ("kpt_shape", "pose_labels", "seg_labels"):
            if key not in data:
                raise KeyError(f"PoseSeg dataset requires `{key}` in {self.args.data}")
        return data
