# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ultralytics.models.yolo.fall_engine import FallValidator
from ultralytics.models.yolo.fall_utils import (
    fall_device,
    fall_tracker_arg,
    fall_yolo_model,
    feature_cache_path,
    opencv_safe_video,
    resample_track,
    resolve_feature_cache_key,
    video_classification_metrics,
    video_fps,
)
from ultralytics.utils import LOGGER, TQDM

from .model import SEGFALL_FEATURE_DIM, mean_segment_confidence, segment_result_features
from .train import _extract_one_video, _load_track_sequences, segfall_feature_settings

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
NEGATIVE_TOKENS = {"no_fall", "nofall", "nonfall", "notfall", "not_fall", "adl", "normal", "negative", "0"}
POSITIVE_TOKENS = {"fall", "falls", "fallen", "positive", "1"}


@dataclass(frozen=True)
class VideoItem:
    path: Path
    label: int


def _norm_token(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _label_from_parts(path: Path, root: Path) -> int | None:
    parts = [_norm_token(part) for part in path.relative_to(root).parts[:-1]]
    for part in reversed(parts):
        compact = part.replace("_", "")
        if part in NEGATIVE_TOKENS or compact in NEGATIVE_TOKENS:
            return 0
        if part in POSITIVE_TOKENS or compact in POSITIVE_TOKENS:
            return 1
        if "fall" in part and not any(prefix in part for prefix in ("no", "non", "not")):
            return 1
    name = _norm_token(path.stem)
    compact = name.replace("_", "")
    if any(token in name or token in compact for token in NEGATIVE_TOKENS):
        return 0
    if any(token in name or token in compact for token in POSITIVE_TOKENS):
        return 1
    return None


def _video_items(root: str | Path, limit: int = 0) -> list[VideoItem]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Fall val data path does not exist: {root}")
    items = []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS):
        label = _label_from_parts(path, root)
        if label is not None:
            items.append(VideoItem(path, label))
    if not items:
        raise ValueError(
            f"No labeled videos found under {root}. Expected class folders like Fall/No_Fall, fall/nonfall, or adl/fall."
        )
    return _limit_video_items(items, limit)


def _limit_video_items(items: list[VideoItem], limit: int = 0) -> list[VideoItem]:
    if limit <= 0:
        return items
    by_label: dict[int, list[VideoItem]] = {}
    for item in items:
        by_label.setdefault(item.label, []).append(item)
    if len(by_label) <= 1:
        return items[:limit]
    per_class = max(1, limit // len(by_label))
    return [item for label in sorted(by_label) for item in by_label[label][:per_class]]


def _videos_in_dir(video_dir: str | Path) -> list[Path]:
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Expected fall val video directory not found: {video_dir}")
    return sorted(p for p in video_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def _video_items_from_dirs(fall_dir: str | Path, nofall_dir: str | Path, limit: int = 0) -> list[VideoItem]:
    fall_videos = _videos_in_dir(fall_dir)
    nofall_videos = _videos_in_dir(nofall_dir)
    if not fall_videos:
        raise ValueError(f"No fall validation videos found under {fall_dir}")
    if not nofall_videos:
        raise ValueError(f"No non-fall validation videos found under {nofall_dir}")
    items = [VideoItem(path, 0) for path in nofall_videos] + [VideoItem(path, 1) for path in fall_videos]
    return _limit_video_items(items, limit)


def _video_items_from_args(args, limit: int = 0) -> list[VideoItem]:
    fall_data = getattr(args, "fall_val_fall_data", None)
    nofall_data = getattr(args, "fall_val_nofall_data", None)
    if fall_data and nofall_data:
        return _video_items_from_dirs(fall_data, nofall_data, limit=limit)
    data = getattr(args, "data", None)
    if not data:
        raise ValueError(
            "segfall val requires fall_val_fall_data=/path/to/Fall "
            "fall_val_nofall_data=/path/to/No_Fall or data=/path/to/labeled/video_dataset"
        )
    return _video_items(data, limit=limit)


def _track_ids(result, count: int) -> list[int | None]:
    if result.boxes is not None and result.boxes.is_track and result.boxes.id is not None:
        return [int(x) for x in result.boxes.id.cpu().tolist()]
    return list(range(count))


class SegFallValidator(FallValidator):
    """End-to-end segmentation-based video-level fall validator.

    This validator starts from raw videos, runs YOLO segmentation tracking, converts the main tracked person to segfall
    features, and evaluates the trained segfall head as a binary video classifier.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None):
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)

    def __call__(self, trainer=None, model=None):
        if trainer is not None:
            return self.validate_cached(trainer)
        device = getattr(self.args, "device", None)
        feature_dim = int(getattr(self.args, "segfall_feature_dim", SEGFALL_FEATURE_DIM))
        min_track_len = int(getattr(self.args, "fall_min_track_len", 30))
        min_conf = float(getattr(self.args, "fall_min_conf", 0.2))
        threshold = float(getattr(self.args, "fall_threshold", 0.5))
        limit = int(getattr(self.args, "fall_limit", 0))
        items = _video_items_from_args(self.args, limit=limit)

        seg_model = fall_yolo_model(model, getattr(self.args, "model", "segfall26n.pt"), task="segment")
        torch_device = fall_device(device)
        segfall_head = getattr(seg_model.model, "segfall_head", None)
        if segfall_head is None:
            raise ValueError("segfall val requires complete SegFall weights containing `segfall_head`")
        segfall_head = segfall_head.float().to(torch_device).eval()
        config = getattr(seg_model.model, "segfall_config", {})
        expected_dim = int(config.get("input_dim", feature_dim))
        self.fall_target_fps = float(config.get("target_fps", getattr(self.args, "fall_target_fps", 30.0)))
        self.fall_max_gap_seconds = float(config.get("max_gap_seconds", getattr(self.args, "fall_max_gap_seconds", 0.5)))
        settings = segfall_feature_settings(self.args)
        feature_key = resolve_feature_cache_key(
            self.args, seg_model, "segfall", "segfall_feature_key", settings
        )
        self.args.segfall_feature_key = feature_key
        feature_dir = Path(getattr(self.args, "segfall_feature_dir", None) or "runs/segfall/features") / feature_key
        end2end = bool(getattr(self.args, "fall_val_end2end", True))
        if end2end:
            LOGGER.info("Running SegFall end-to-end validation from decoded videos")
        else:
            LOGGER.info(f"Using shared SegFall feature cache: {feature_dir}")

        rows = []
        processed_frames = 0
        processing_seconds = 0.0
        start = time.time()
        self.run_callbacks("on_val_start")
        description = "SegFall end-to-end val" if end2end else "SegFall cached val"
        for i, item in TQDM(enumerate(items, 1), total=len(items), desc=description, unit="video"):
            self.run_callbacks("on_val_batch_start")
            if end2end:
                if torch_device.type == "cuda":
                    torch.cuda.synchronize(torch_device)
                video_start = time.perf_counter()
                prob, reason, frames = self._predict_video_end2end(
                    seg_model,
                    segfall_head,
                    item.path,
                    torch_device,
                    feature_dim=expected_dim,
                    min_track_len=min_track_len,
                    min_conf=min_conf,
                )
                if torch_device.type == "cuda":
                    torch.cuda.synchronize(torch_device)
                processing_seconds += time.perf_counter() - video_start
                processed_frames += frames
            else:
                prob, reason = self._predict_video_cached(
                    seg_model,
                    segfall_head,
                    item,
                    feature_dir,
                    torch_device,
                    feature_dim=expected_dim,
                    min_track_len=min_track_len,
                    min_conf=min_conf,
                )
            pred = int(prob >= threshold) if prob is not None else None
            rows.append({"path": str(item.path), "label": item.label, "pred": pred, "prob": prob, "reason": reason})
            LOGGER.debug(
                f"segfall val {i}/{len(items)} label={item.label} pred={pred} "
                f"prob={prob if prob is not None else 'None'} reason={reason} {item.path}"
            )
            self.run_callbacks("on_val_batch_end")

        self.metrics = self._compute_metrics(rows, threshold=threshold, elapsed=time.time() - start)
        self.metrics["end2end"] = end2end
        if end2end:
            self.metrics["processed_frames"] = processed_frames
            self.metrics["processing_seconds"] = processing_seconds
            self.metrics["ms_per_frame"] = 1000.0 * processing_seconds / max(processed_frames, 1)
            self.metrics["processing_fps"] = processed_frames / max(processing_seconds, 1e-12)
        self._save_rows(rows)
        self._save_metrics()
        LOGGER.info(
            "segfall val: "
            f"accuracy={self.metrics['accuracy']:.4f} precision={self.metrics['precision']:.4f} "
            f"recall={self.metrics['recall']:.4f} f1={self.metrics['f1']:.4f} "
            f"p90={self.metrics['p90']:.4f} p95={self.metrics['p95']:.4f} competition_map={self.metrics['competition_map']:.2f} "
            f"tp={self.metrics['tp']} fp={self.metrics['fp']} tn={self.metrics['tn']} fn={self.metrics['fn']} "
            f"coverage={self.metrics['coverage']:.4f} ({self.metrics['valid_videos']}/{self.metrics['videos']})"
        )
        if end2end:
            LOGGER.info(
                f"segfall end-to-end speed: frames={processed_frames} time={processing_seconds:.3f}s "
                f"ms/frame={self.metrics['ms_per_frame']:.3f} FPS={self.metrics['processing_fps']:.2f}"
            )
        self.run_callbacks("on_val_end")
        return self.metrics

    def _predict_video_end2end(
        self,
        seg_model,
        segfall_head,
        video: Path,
        device,
        feature_dim: int,
        min_track_len: int,
        min_conf: float,
    ):
        source_video = opencv_safe_video(video, "segfall")
        tracks: dict[int, list[torch.Tensor]] = {}
        track_frame_indices: dict[int, list[int]] = {}
        frames_seen = segment_frames = 0
        frame_step = max(int(getattr(self.args, "vid_stride", 1) or 1), 1)
        for processed_frame_idx, result in enumerate(
            seg_model.track(
                source=str(source_video),
                stream=True,
                persist=False,
                device=getattr(self.args, "device", None),
                imgsz=int(getattr(self.args, "imgsz", 640)),
                vid_stride=int(getattr(self.args, "vid_stride", 1)),
                conf=getattr(self.args, "conf", None),
                iou=getattr(self.args, "iou", None),
                tracker=fall_tracker_arg(self.args),
                save=False,
                verbose=False,
            )
        ):
            frame_idx = processed_frame_idx * frame_step
            frames_seen += 1
            if result.boxes is None or len(result.boxes) == 0 or result.masks is None:
                continue
            segment_frames += 1
            feats = segment_result_features(result, device="cpu")
            if feats.shape[1] != feature_dim:
                return None, f"feature_dim={feats.shape[1]} expected={feature_dim}", frames_seen
            for track_id, feat in zip(_track_ids(result, len(feats)), feats):
                if track_id is not None:
                    tracks.setdefault(track_id, []).append(feat)
                    track_frame_indices.setdefault(track_id, []).append(frame_idx)

        candidates = []
        source_fps = video_fps(source_video)
        for track_id, frames in tracks.items():
            segments = resample_track(
                frames,
                track_frame_indices[track_id],
                source_fps,
                self.fall_target_fps,
                self.fall_max_gap_seconds,
            )
            for sequence, _ in segments:
                confidence = mean_segment_confidence(sequence)
                if len(sequence) >= min_track_len and confidence >= min_conf:
                    candidates.append((len(sequence), confidence, sequence))
        if not candidates:
            reason = "decode_failed_or_empty_video" if not frames_seen else f"no_valid_track segment_frames={segment_frames}"
            return None, reason, frames_seen
        candidates.sort(key=lambda value: (value[0], value[1]), reverse=True)
        sequence = candidates[0][2]
        with torch.no_grad():
            probability = float(segfall_head(sequence.to(device)).sigmoid().item())
        window, stride = int(segfall_head.window), int(segfall_head.stride)
        clips = 1 if len(sequence) <= window else ((len(sequence) - window) // stride + 1 + int((len(sequence) - window) % stride != 0))
        return probability, f"track_len={len(sequence)} clips={clips} candidates={len(candidates)}", frames_seen

    def _predict_video_cached(
        self,
        seg_model,
        segfall_head,
        item: VideoItem,
        feature_dir: Path,
        device,
        feature_dim: int,
        min_track_len: int,
        min_conf: float,
    ):
        path = feature_cache_path(feature_dir, "Fall" if item.label else "No_Fall", item.path)
        cache_hit = path.is_file() and not bool(getattr(self.args, "fall_overwrite_features", False))
        _extract_one_video(
            seg_model,
            (item.path, item.label),
            feature_dir,
            getattr(self.args, "device", None),
            int(getattr(self.args, "imgsz", 640)),
            int(getattr(self.args, "vid_stride", 1)),
            getattr(self.args, "conf", None),
            getattr(self.args, "iou", None),
            fall_tracker_arg(self.args),
            bool(getattr(self.args, "fall_overwrite_features", False)),
            min_track_len,
            min_conf,
        )
        if not path.is_file():
            return None, "no_valid_feature_cache"
        try:
            sequences = _load_track_sequences(
                [path], min_track_len, min_conf, feature_dim, self.fall_target_fps, self.fall_max_gap_seconds
            )
        except (ValueError, RuntimeError, OSError) as error:
            return None, f"invalid_feature_cache: {error}"
        seq = sequences[0].sequence
        with torch.no_grad():
            prob = float(segfall_head(seq.to(device)).sigmoid().item())
        window = int(getattr(segfall_head, "window", getattr(self.args, "fall_window", 60)))
        stride = int(getattr(segfall_head, "stride", getattr(self.args, "fall_stride", 15)))
        clips = 1 if seq.shape[0] <= window else ((seq.shape[0] - window) // stride + 1 + int((seq.shape[0] - window) % stride != 0))
        return prob, f"cache={'hit' if cache_hit else 'created'} track_len={seq.shape[0]} clips={clips}"

    @staticmethod
    def _compute_metrics(rows: list[dict], threshold: float, elapsed: float) -> dict:
        return video_classification_metrics(rows, threshold, elapsed, SegFallValidator._competition_metrics)

    @staticmethod
    def _competition_metrics(targets: list[int], probs: list[float]) -> dict[str, float]:
        def precision_at_recall(min_recall: float) -> float:
            positives = sum(1 for y in targets if y == 1)
            if positives <= 0:
                return 0.0
            tp = fp = 0
            best = 0.0
            for prob, target in sorted(zip(probs, targets), key=lambda item: item[0], reverse=True):
                if target == 1:
                    tp += 1
                else:
                    fp += 1
                recall = tp / positives
                if recall >= min_recall:
                    best = max(best, tp / max(tp + fp, 1))
            return best

        p90 = precision_at_recall(0.90)
        p95 = precision_at_recall(0.95)
        return {"p90": p90, "p95": p95, "competition_map": (p90 + p95) * 50.0}

    def _save_rows(self, rows: list[dict]) -> None:
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "predictions.csv"
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "pred", "prob", "reason"])
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info(f"segfall val predictions saved to {out}")

    def _save_metrics(self) -> None:
        out = Path(self.save_dir) / "metrics.json"
        with out.open("w") as file:
            json.dump(self.metrics, file, indent=2)
        LOGGER.info(f"segfall val metrics saved to {out}")
