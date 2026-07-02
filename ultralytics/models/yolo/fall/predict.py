# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections import defaultdict, deque
from functools import partial

import torch

from ultralytics.models.yolo.pose.predict import PosePredictor
from ultralytics.trackers import register_tracker
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT

from .model import (
    FALL_CENTER_Y_INDEX,
    FALL_FEATURE_DIM,
    FALL_KEYPOINT_DIM,
    load_fall_head,
    mean_keypoint_confidence,
    normalize_keypoints,
    pad_or_trim,
)


class FallPredictor(PosePredictor):
    """Fall prediction on top of YOLO pose predictions."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "fall"
        if getattr(self.args, "tracker", None) == DEFAULT_CFG_DICT.get("tracker"):
            self.args.tracker = getattr(self.args, "fall_tracker", None) or "fall_botsort.yaml"
        self.histories = defaultdict(lambda: deque(maxlen=int(getattr(self.args, "fall_window", 60))))
        self.fall_probs: list[float | None] = []
        self.fall_head = None
        self.fall_window = int(getattr(self.args, "fall_window", 60))
        self.fall_threshold = float(getattr(self.args, "fall_threshold", 0.5))
        self.fall_min_conf = float(getattr(self.args, "fall_min_conf", 0.2))
        self.fall_input_dim = int(getattr(self.args, "fall_feature_dim", FALL_FEATURE_DIM))
        register_tracker(self, persist=True)
        self.add_callback("on_predict_postprocess_end", partial(_on_fall_postprocess_end))

    def setup_model(self, model, verbose: bool = True):
        super().setup_model(model, verbose)
        self.fall_head, config = load_fall_head(
            getattr(self.args, "fall_weights", None), self.device, input_dim=FALL_FEATURE_DIM, window=self.fall_window
        )
        self.fall_window = int(config.get("window") or self.fall_window)
        self.fall_stride = int(config.get("stride", getattr(self.args, "fall_stride", 15)))
        self.histories = defaultdict(lambda: deque(maxlen=self.fall_window))
        self.fall_input_dim = int(config.get("input_dim", self.fall_input_dim))
        if self.fall_head is not None and self.fall_input_dim not in {FALL_KEYPOINT_DIM, FALL_FEATURE_DIM}:
            raise ValueError(
                f"Fall head input_dim={self.fall_input_dim} is unsupported. "
                f"Train a keypoint-based fall head with input_dim={FALL_FEATURE_DIM}."
            )

    def _track_ids(self, result, count: int) -> list[int]:
        if result.boxes is not None and result.boxes.is_track and result.boxes.id is not None:
            return [int(x) for x in result.boxes.id.cpu().tolist()]
        return list(range(count))

    def _update_fall_probs(self, result) -> None:
        self.fall_probs = []
        if self.fall_head is None:
            self.fall_probs = [None] * len(result)
            return
        if result.keypoints is None or len(result.keypoints) == 0:
            self.fall_probs = [None] * len(result)
            return
        ids = self._track_ids(result, len(result.keypoints))
        previous_center_y = []
        for track_id in ids:
            history = self.histories[track_id]
            previous_center_y.append(float(history[-1][FALL_CENTER_Y_INDEX]) if history and self.fall_input_dim > FALL_CENTER_Y_INDEX else None)
        boxes_xywhn = result.boxes.xywhn.detach().to(self.device) if result.boxes is not None else None
        feats = normalize_keypoints(
            result.keypoints.data.detach().to(self.device),
            result.orig_shape,
            boxes_xywhn=boxes_xywhn,
            previous_center_y=previous_center_y,
            feature_dim=self.fall_input_dim,
        )
        windows = []
        valid_indexes = []
        for det_index, (track_id, feat) in enumerate(zip(ids, feats)):
            history = self.histories[track_id]
            history.append(feat.detach())
            window = pad_or_trim(torch.stack(tuple(history)), self.fall_window)
            if mean_keypoint_confidence(window) < self.fall_min_conf:
                continue
            valid_indexes.append(det_index)
            windows.append(window)
        if not windows:
            self.fall_probs = [None] * len(result)
            result.fall_probs = self.fall_probs
            self._set_box_extra_labels(result, self.fall_probs)
            return
        with torch.no_grad():
            probs = self.fall_head(torch.stack(windows).to(self.device)).sigmoid().detach().cpu().tolist()
        self.fall_probs = [None] * feats.shape[0]
        for index, prob in zip(valid_indexes, probs):
            self.fall_probs[index] = prob
        result.fall_probs = self.fall_probs
        self._set_box_extra_labels(result, self.fall_probs)

    def _set_box_extra_labels(self, result, probs) -> None:
        result.box_extra_labels = [
            f"{'FALL' if prob >= self.fall_threshold else 'fall'} {prob:.2f}" if prob is not None else ""
            for prob in probs
        ]

def _on_fall_postprocess_end(predictor: FallPredictor) -> None:
    """Compute fall probabilities after tracker callbacks have finalized Results."""
    for result in predictor.results:
        predictor._update_fall_probs(result)
