# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections import defaultdict, deque
from functools import partial

import torch

from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.trackers import register_tracker
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT

from .model import SEGFALL_FEATURE_DIM, load_segfall_head, mean_segment_confidence, normalize_segments, pad_or_trim, segment_state


class SegFallPredictor(SegmentationPredictor):
    """Segmentation-based fall prediction on top of YOLO segmentation predictions."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segfall"
        if getattr(self.args, "tracker", None) == DEFAULT_CFG_DICT.get("tracker"):
            self.args.tracker = getattr(self.args, "segfall_tracker", None) or "segfall_botsort.yaml"
        self.histories = defaultdict(lambda: deque(maxlen=int(getattr(self.args, "segfall_window", 60))))
        self.states: dict[int, dict[str, float]] = {}
        self.segfall_probs: list[float | None] = []
        self.segfall_head = None
        self.segfall_window = int(getattr(self.args, "segfall_window", 60))
        self.segfall_threshold = float(getattr(self.args, "segfall_threshold", 0.5))
        self.segfall_min_conf = float(getattr(self.args, "segfall_min_conf", 0.2))
        self.segfall_input_dim = int(getattr(self.args, "segfall_feature_dim", SEGFALL_FEATURE_DIM))
        register_tracker(self, persist=True)
        self.add_callback("on_predict_postprocess_end", partial(_on_segfall_postprocess_end))

    def setup_model(self, model, verbose: bool = True):
        super().setup_model(model, verbose)
        self.segfall_head, config = load_segfall_head(
            getattr(self.args, "segfall_weights", None), self.device, input_dim=SEGFALL_FEATURE_DIM, window=self.segfall_window
        )
        self.segfall_window = int(config.get("window") or self.segfall_window)
        self.segfall_stride = int(config.get("stride", getattr(self.args, "segfall_stride", 15)))
        self.histories = defaultdict(lambda: deque(maxlen=self.segfall_window))
        self.states = {}
        self.segfall_input_dim = int(config.get("input_dim", self.segfall_input_dim))

    def _track_ids(self, result, count: int) -> list[int]:
        if result.boxes is not None and result.boxes.is_track and result.boxes.id is not None:
            return [int(x) for x in result.boxes.id.cpu().tolist()]
        return list(range(count))

    def _update_segfall_probs(self, result) -> None:
        self.segfall_probs = []
        if self.segfall_head is None:
            self.segfall_probs = [None] * len(result)
            return
        if result.boxes is None or len(result.boxes) == 0 or result.masks is None:
            self.segfall_probs = [None] * len(result)
            return

        count = len(result.boxes)
        ids = self._track_ids(result, count)
        boxes_xywhn = result.boxes.xywhn.detach().to(self.device)
        confs = result.boxes.conf.detach().to(self.device) if result.boxes.conf is not None else None
        previous = [self.states.get(track_id) for track_id in ids]
        feats = normalize_segments(
            boxes_xywhn,
            result.masks.xy,
            result.orig_shape,
            conf=confs,
            previous_state=previous,
            feature_dim=self.segfall_input_dim,
        )

        windows = []
        valid_indexes = []
        for det_index, (track_id, feat) in enumerate(zip(ids, feats)):
            history = self.histories[track_id]
            history.append(feat.detach())
            self.states[track_id] = segment_state(feat.detach().cpu())
            window = pad_or_trim(torch.stack(tuple(history)), self.segfall_window)
            if mean_segment_confidence(window) < self.segfall_min_conf:
                continue
            valid_indexes.append(det_index)
            windows.append(window)
        if not windows:
            self.segfall_probs = [None] * count
            result.segfall_probs = self.segfall_probs
            self._set_box_extra_labels(result, self.segfall_probs)
            return
        with torch.no_grad():
            probs = self.segfall_head(torch.stack(windows).to(self.device)).sigmoid().detach().cpu().tolist()
        self.segfall_probs = [None] * count
        for index, prob in zip(valid_indexes, probs):
            self.segfall_probs[index] = prob
        result.segfall_probs = self.segfall_probs
        self._set_box_extra_labels(result, self.segfall_probs)

    def _set_box_extra_labels(self, result, probs) -> None:
        result.box_extra_labels = [
            f"{'FALL' if prob >= self.segfall_threshold else 'fall'} {prob:.2f}" if prob is not None else ""
            for prob in probs
        ]


def _on_segfall_postprocess_end(predictor: SegFallPredictor) -> None:
    """Compute fall probabilities after tracker callbacks have finalized Results."""
    for result in predictor.results:
        predictor._update_segfall_probs(result)
