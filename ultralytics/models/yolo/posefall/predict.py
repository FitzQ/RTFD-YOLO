# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections import defaultdict, deque
from functools import partial
import torch

from ultralytics.models.yolo.pose.predict import PosePredictor
from ultralytics.models.yolo.fall_utils import (
    append_resampled_feature,
    fall_tracker_arg,
    predictor_fps,
    safe_video_source,
)
from ultralytics.trackers import register_tracker
from ultralytics.utils import DEFAULT_CFG

from .model import (
    POSEFALL_FEATURE_DIM,
    mean_keypoint_confidence,
    pad_or_trim,
    pose_result_features,
)


class PoseFallPredictor(PosePredictor):
    """Pose-based fall prediction on top of YOLO pose predictions."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "posefall"
        self.args.tracker = fall_tracker_arg(self.args)
        self.histories = defaultdict(lambda: deque(maxlen=int(getattr(self.args, "fall_window", 60))))
        self.posefall_probs: list[float | None] = []
        self.posefall_head = None
        self.fall_window = int(getattr(self.args, "fall_window", 60))
        self.fall_threshold = float(getattr(self.args, "fall_threshold", 0.5))
        self.fall_min_conf = float(getattr(self.args, "fall_min_conf", 0.2))
        self.posefall_input_dim = int(getattr(self.args, "posefall_feature_dim", POSEFALL_FEATURE_DIM))
        self.fall_target_fps = float(getattr(self.args, "fall_target_fps", 30.0))
        self.fall_max_gap_seconds = float(getattr(self.args, "fall_max_gap_seconds", 0.5))
        self._fall_frame = 0
        self._fall_last_tick: dict[int, int] = {}
        register_tracker(self, persist=True)
        self.add_callback("on_predict_postprocess_end", partial(_on_posefall_postprocess_end))

    def setup_model(self, model, verbose: bool = True):
        super().setup_model(model, verbose)
        pose_model = self.model.model
        self.posefall_head = getattr(pose_model, "posefall_head", None)
        if self.posefall_head is None:
            raise ValueError("posefall predict requires complete PoseFall weights containing `posefall_head`")
        self.posefall_head = self.posefall_head.float().to(self.device).eval()
        config = getattr(pose_model, "posefall_config", {})
        self.fall_window = int(config.get("window") or self.fall_window)
        self.fall_stride = int(config.get("stride", getattr(self.args, "fall_stride", 15)))
        self.fall_target_fps = float(config.get("target_fps", self.fall_target_fps))
        self.fall_max_gap_seconds = float(config.get("max_gap_seconds", self.fall_max_gap_seconds))
        self.histories = defaultdict(lambda: deque(maxlen=self.fall_window))
        self.posefall_input_dim = int(config.get("input_dim", self.posefall_input_dim))
        if self.posefall_input_dim != POSEFALL_FEATURE_DIM:
            raise ValueError(f"PoseFall input_dim={self.posefall_input_dim}, expected {POSEFALL_FEATURE_DIM}")

    def setup_source(self, source, stride: int | None = None):
        self._fall_frame = 0
        self._fall_last_tick.clear()
        self.histories.clear()
        return super().setup_source(safe_video_source(source, "posefall"), stride)

    def _track_ids(self, result, count: int) -> list[int]:
        if result.boxes is not None and result.boxes.is_track and result.boxes.id is not None:
            return [int(x) for x in result.boxes.id.cpu().tolist()]
        return list(range(count))

    def _update_posefall_probs(self, result) -> None:
        self.posefall_probs = []
        source_frame = self._fall_frame * max(int(getattr(self.args, "vid_stride", 1) or 1), 1)
        tick = round(source_frame * self.fall_target_fps / predictor_fps(self))
        self._fall_frame += 1
        if self.posefall_head is None:
            self.posefall_probs = [None] * len(result)
            return
        if result.keypoints is None or len(result.keypoints) == 0:
            self.posefall_probs = [None] * len(result)
            return
        ids = self._track_ids(result, len(result.keypoints))
        feats = pose_result_features(result, device=self.device)
        windows = []
        valid_indexes = []
        for det_index, (track_id, feat) in enumerate(zip(ids, feats)):
            history = self.histories[track_id]
            self._fall_last_tick[track_id] = append_resampled_feature(
                history,
                feat.detach(),
                tick,
                self._fall_last_tick.get(track_id),
                max(1, round(self.fall_max_gap_seconds * self.fall_target_fps)),
            )
            window = pad_or_trim(torch.stack(tuple(history)), self.fall_window)
            if mean_keypoint_confidence(window) < self.fall_min_conf:
                continue
            valid_indexes.append(det_index)
            windows.append(window)
        if not windows:
            self.posefall_probs = [None] * len(result)
            result.posefall_probs = self.posefall_probs
            self._set_box_extra_labels(result, self.posefall_probs)
            return
        with torch.no_grad():
            probs = self.posefall_head(torch.stack(windows).to(self.device)).sigmoid().detach().cpu().tolist()
        self.posefall_probs = [None] * feats.shape[0]
        for index, prob in zip(valid_indexes, probs):
            self.posefall_probs[index] = prob
        result.posefall_probs = self.posefall_probs
        self._set_box_extra_labels(result, self.posefall_probs)

    def _set_box_extra_labels(self, result, probs) -> None:
        result.box_extra_labels = [
            f"{'FALL' if prob >= self.fall_threshold else 'fall'} {prob:.2f}" if prob is not None else ""
            for prob in probs
        ]

def _on_posefall_postprocess_end(predictor: PoseFallPredictor) -> None:
    """Compute fall probabilities after tracker callbacks have finalized Results."""
    for result in predictor.results:
        predictor._update_posefall_probs(result)
