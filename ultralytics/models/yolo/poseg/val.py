from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import OKS_SIGMA, PoseSegMetrics, kpt_iou, mask_iou


class PoseSegValidator(DetectionValidator):
    """Validator reporting box, pose, and mask AP from joint predictions."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "poseg"
        self.metrics = PoseSegMetrics()
        self.process = None
        self.kpt_shape = None
        self.sigma = None

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].float()
        batch["keypoints"] = batch["keypoints"].float()
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        self.sigma = OKS_SIGMA if self.kpt_shape == [17, 3] else np.ones(self.kpt_shape[0]) / self.kpt_shape[0]
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask

    def get_desc(self) -> str:
        return ("%22s" + "%11s" * 14) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds) -> list[dict[str, torch.Tensor]]:
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        processed = super().postprocess(preds[0][0])
        nm = proto.shape[1]
        imgsz = [4 * x for x in proto.shape[2:]]
        for i, pred in enumerate(processed):
            extra = pred.pop("extra")
            coefficients = extra[:, :nm]
            pred["keypoints"] = extra[:, nm:].view(-1, *self.kpt_shape)
            pred["masks"] = (
                self.process(proto[i], coefficients, pred["bboxes"], shape=imgsz)
                if len(coefficients)
                else torch.zeros((0, *proto.shape[2:]), dtype=torch.uint8, device=pred["bboxes"].device)
            )
        return processed

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        prepared = super()._prepare_batch(si, batch)
        idx = batch["batch_idx"] == si
        kpts = batch["keypoints"][idx].clone()
        h, w = prepared["imgsz"]
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        prepared["keypoints"] = kpts
        nl = len(prepared["cls"])
        if self.args.overlap_mask:
            masks = batch["masks"][si]
            index = torch.arange(1, nl + 1, device=masks.device).view(nl, 1, 1)
            masks = (masks == index).float()
        else:
            masks = batch["masks"][idx]
        if nl:
            mask_size = [s if self.process is ops.process_mask_native else s // 4 for s in prepared["imgsz"]]
            if list(masks.shape[1:]) != mask_size:
                masks = F.interpolate(masks[None], mask_size, mode="bilinear", align_corners=False)[0].gt_(0.5)
        prepared["masks"] = masks
        return prepared

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        result = super()._process_batch(preds, batch)
        if not len(batch["cls"]) or not len(preds["cls"]):
            empty = np.zeros((len(preds["cls"]), self.niou), dtype=bool)
            result.update(tp_p=empty, tp_m=empty.copy())
            return result
        area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
        pose_iou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
        seg_iou = mask_iou(batch["masks"].flatten(1), preds["masks"].flatten(1).float())
        result["tp_p"] = self.match_predictions(preds["cls"], batch["cls"], pose_iou).cpu().numpy()
        result["tp_m"] = self.match_predictions(preds["cls"], batch["cls"], seg_iou).cpu().numpy()
        return result

    def gather_stats(self) -> None:
        super().gather_stats()
        self._gather_image_metrics(self.metrics.pose)
        self._gather_image_metrics(self.metrics.seg)

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
        for pred in preds:
            pred["masks"] = torch.as_tensor(pred["masks"][: self.args.max_det], dtype=torch.uint8).cpu()
        super().plot_predictions(batch, preds, ni, max_det=self.args.max_det)

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], 1),
            masks=torch.as_tensor(predn["masks"], dtype=torch.uint8),
            keypoints=predn["keypoints"],
        ).save_txt(file, save_conf=save_conf)
