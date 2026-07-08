# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections import defaultdict, deque
from functools import partial
import hashlib
import subprocess
from pathlib import Path

import torch

from ultralytics.models.yolo.pose.predict import PosePredictor
from ultralytics.trackers import register_tracker
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT

from .model import (
    POSEFALL_CENTER_Y_INDEX,
    POSEFALL_FEATURE_DIM,
    POSEFALL_KEYPOINT_DIM,
    load_posefall_head,
    mean_keypoint_confidence,
    normalize_keypoints,
    pad_or_trim,
)


def _safe_stem(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in value)


def _opencv_safe_video(video: Path) -> Path:
    """Transcode legacy AVI files to cached MP4 before OpenCV/YOLO reads them."""
    if video.suffix.lower() != ".avi":
        return video
    cache_dir = Path("runs/posefall/video_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(video.resolve()).encode()).hexdigest()[:12]
    cached = cache_dir / f"{digest}_{_safe_stem(video.stem)}.mp4"
    if cached.exists() and cached.stat().st_size > 0:
        return cached
    tmp = cached.with_suffix(".mp4.part")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "mp4",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"ffmpeg is required to predict legacy AVI videos such as {video}") from e
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to transcode AVI video for posefall prediction: {video}") from e
    tmp.rename(cached)
    return cached


def _safe_source(source):
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_file():
            return str(_opencv_safe_video(path))
    if isinstance(source, (list, tuple)):
        return [_safe_source(item) for item in source]
    return source


class PoseFallPredictor(PosePredictor):
    """Pose-based fall prediction on top of YOLO pose predictions."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "posefall"
        if getattr(self.args, "tracker", None) == DEFAULT_CFG_DICT.get("tracker"):
            self.args.tracker = getattr(self.args, "posefall_tracker", None) or "posefall_botsort.yaml"
        self.histories = defaultdict(lambda: deque(maxlen=int(getattr(self.args, "posefall_window", 60))))
        self.posefall_probs: list[float | None] = []
        self.posefall_head = None
        self.posefall_window = int(getattr(self.args, "posefall_window", 60))
        self.posefall_threshold = float(getattr(self.args, "posefall_threshold", 0.5))
        self.posefall_min_conf = float(getattr(self.args, "posefall_min_conf", 0.2))
        self.posefall_input_dim = int(getattr(self.args, "posefall_feature_dim", POSEFALL_FEATURE_DIM))
        register_tracker(self, persist=True)
        self.add_callback("on_predict_postprocess_end", partial(_on_posefall_postprocess_end))

    def setup_model(self, model, verbose: bool = True):
        super().setup_model(model, verbose)
        self.posefall_head, config = load_posefall_head(
            getattr(self.args, "posefall_weights", None), self.device, input_dim=POSEFALL_FEATURE_DIM, window=self.posefall_window
        )
        self.posefall_window = int(config.get("window") or self.posefall_window)
        self.posefall_stride = int(config.get("stride", getattr(self.args, "posefall_stride", 15)))
        self.histories = defaultdict(lambda: deque(maxlen=self.posefall_window))
        self.posefall_input_dim = int(config.get("input_dim", self.posefall_input_dim))
        if self.posefall_head is not None and self.posefall_input_dim not in {POSEFALL_KEYPOINT_DIM, POSEFALL_FEATURE_DIM}:
            raise ValueError(
                f"PoseFall head input_dim={self.posefall_input_dim} is unsupported. "
                f"Train a keypoint-based posefall head with input_dim={POSEFALL_FEATURE_DIM}."
            )

    def setup_source(self, source, stride: int | None = None):
        return super().setup_source(_safe_source(source), stride)

    def _track_ids(self, result, count: int) -> list[int]:
        if result.boxes is not None and result.boxes.is_track and result.boxes.id is not None:
            return [int(x) for x in result.boxes.id.cpu().tolist()]
        return list(range(count))

    def _update_posefall_probs(self, result) -> None:
        self.posefall_probs = []
        if self.posefall_head is None:
            self.posefall_probs = [None] * len(result)
            return
        if result.keypoints is None or len(result.keypoints) == 0:
            self.posefall_probs = [None] * len(result)
            return
        ids = self._track_ids(result, len(result.keypoints))
        previous_center_y = []
        for track_id in ids:
            history = self.histories[track_id]
            previous_center_y.append(float(history[-1][POSEFALL_CENTER_Y_INDEX]) if history and self.posefall_input_dim > POSEFALL_CENTER_Y_INDEX else None)
        boxes_xywhn = result.boxes.xywhn.detach().to(self.device) if result.boxes is not None else None
        feats = normalize_keypoints(
            result.keypoints.data.detach().to(self.device),
            result.orig_shape,
            boxes_xywhn=boxes_xywhn,
            previous_center_y=previous_center_y,
            feature_dim=self.posefall_input_dim,
        )
        windows = []
        valid_indexes = []
        for det_index, (track_id, feat) in enumerate(zip(ids, feats)):
            history = self.histories[track_id]
            history.append(feat.detach())
            window = pad_or_trim(torch.stack(tuple(history)), self.posefall_window)
            if mean_keypoint_confidence(window) < self.posefall_min_conf:
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
            f"{'FALL' if prob >= self.posefall_threshold else 'fall'} {prob:.2f}" if prob is not None else ""
            for prob in probs
        ]

def _on_posefall_postprocess_end(predictor: PoseFallPredictor) -> None:
    """Compute fall probabilities after tracker callbacks have finalized Results."""
    for result in predictor.results:
        predictor._update_posefall_probs(result)
