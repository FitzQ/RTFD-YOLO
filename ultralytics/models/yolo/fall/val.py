# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import csv
import hashlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER

from .model import FALL_CENTER_Y_INDEX, FALL_FEATURE_DIM, load_fall_head, mean_keypoint_confidence, normalize_keypoints

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
    fall_data = getattr(args, "val_fall_data", None)
    nofall_data = getattr(args, "val_nofall_data", None)
    if fall_data and nofall_data:
        return _video_items_from_dirs(fall_data, nofall_data, limit=limit)
    data = getattr(args, "data", None)
    if not data:
        raise ValueError("fall val requires val_fall_data=/path/to/Fall val_nofall_data=/path/to/No_Fall or data=/path/to/labeled/video_dataset")
    return _video_items(data, limit=limit)


def _track_ids(result, count: int) -> list[int | None]:
    if result.boxes is not None and result.boxes.is_track and result.boxes.id is not None:
        return [int(x) for x in result.boxes.id.cpu().tolist()]
    return list(range(count))


def _fall_tracker_arg(args) -> str:
    tracker = getattr(args, "tracker", None)
    if tracker and tracker != DEFAULT_CFG_DICT.get("tracker"):
        return tracker
    return getattr(args, "fall_tracker", None) or tracker or "fall_botsort.yaml"


def _safe_stem(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in value)


def _opencv_safe_video(video: Path) -> Path:
    """Transcode legacy AVI files to cached MP4 before OpenCV/YOLO reads them.

    Some Le2i AVI files contain raw BGR video plus malformed MP3 audio; OpenCV may segfault before Python can catch it.
    """
    if video.suffix.lower() != ".avi":
        return video
    cache_dir = Path("runs/fall/video_cache")
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
        raise RuntimeError(f"ffmpeg is required to validate legacy AVI videos such as {video}") from e
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to transcode AVI video for validation: {video}") from e
    tmp.rename(cached)
    return cached


class FallValidator:
    """End-to-end video-level fall validator.

    This validator starts from raw videos, runs YOLO pose tracking, converts the main tracked person to fall features,
    and evaluates the trained fall head as a binary video classifier.
    """

    def __init__(self, args=None, _callbacks: dict | None = None):
        self.args = get_cfg(DEFAULT_CFG, args or {})
        self.callbacks = _callbacks
        self.metrics = {}

    def __call__(self, model=None):
        from ultralytics import YOLO

        device = getattr(self.args, "device", None)
        feature_dim = int(getattr(self.args, "fall_feature_dim", FALL_FEATURE_DIM))
        min_track_len = int(getattr(self.args, "fall_min_track_len", 30))
        min_conf = float(getattr(self.args, "fall_min_conf", 0.2))
        threshold = float(getattr(self.args, "fall_threshold", 0.5))
        limit = int(getattr(self.args, "fall_limit", 0))
        items = _video_items_from_args(self.args, limit=limit)

        pose_model = YOLO(getattr(self.args, "model", "yolo26n-pose.pt"), task="pose")
        torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        fall_head, config = load_fall_head(getattr(self.args, "fall_weights", None), torch_device)
        expected_dim = int(config.get("input_dim", feature_dim))

        rows = []
        start = time.time()
        for i, item in enumerate(items, 1):
            prob, reason = self._predict_video(
                pose_model,
                fall_head,
                item.path,
                torch_device,
                feature_dim=expected_dim,
                min_track_len=min_track_len,
                min_conf=min_conf,
            )
            pred = int(prob >= threshold) if prob is not None else 0
            rows.append({"path": str(item.path), "label": item.label, "pred": pred, "prob": prob, "reason": reason})
            LOGGER.info(
                f"fall val {i}/{len(items)} label={item.label} pred={pred} "
                f"prob={prob if prob is not None else 'None'} reason={reason} {item.path}"
            )

        self.metrics = self._compute_metrics(rows, threshold=threshold, elapsed=time.time() - start)
        self._save_rows(rows)
        LOGGER.info(
            "fall val: "
            f"accuracy={self.metrics['accuracy']:.4f} precision={self.metrics['precision']:.4f} "
            f"recall={self.metrics['recall']:.4f} f1={self.metrics['f1']:.4f} "
            f"tp={self.metrics['tp']} fp={self.metrics['fp']} tn={self.metrics['tn']} fn={self.metrics['fn']}"
        )
        return self.metrics

    def _predict_video(
        self,
        pose_model,
        fall_head,
        video: Path,
        device,
        feature_dim: int,
        min_track_len: int,
        min_conf: float,
    ):
        source_video = _opencv_safe_video(video)
        tracks: dict[int, list[torch.Tensor]] = {}
        previous_center_y: dict[int, float] = {}
        frames_seen = 0
        keypoint_frames = 0
        for result in pose_model.track(
            source=str(source_video),
            stream=True,
            persist=True,
            device=getattr(self.args, "device", None),
            imgsz=int(getattr(self.args, "imgsz", 640)),
            vid_stride=int(getattr(self.args, "vid_stride", 1)),
            conf=getattr(self.args, "conf", None),
            iou=getattr(self.args, "iou", None),
            tracker=_fall_tracker_arg(self.args),
            save=False,
            verbose=False,
        ):
            frames_seen += 1
            if result.keypoints is None or len(result.keypoints) == 0:
                continue
            keypoint_frames += 1
            ids = _track_ids(result, len(result.keypoints))
            boxes_xywhn = result.boxes.xywhn.detach().cpu() if result.boxes is not None else None
            prev = [previous_center_y.get(track_id) if track_id is not None else None for track_id in ids]
            feats = normalize_keypoints(
                result.keypoints.data.detach().cpu(),
                result.orig_shape,
                boxes_xywhn=boxes_xywhn,
                previous_center_y=prev,
                feature_dim=feature_dim,
            )
            for track_id, feat in zip(ids, feats):
                if track_id is None:
                    continue
                tracks.setdefault(track_id, []).append(feat)
                if feat.shape[0] > FALL_CENTER_Y_INDEX:
                    previous_center_y[track_id] = float(feat[FALL_CENTER_Y_INDEX])

        candidates = []
        for frames in tracks.values():
            if len(frames) < min_track_len:
                continue
            seq = torch.stack(frames)
            conf = mean_keypoint_confidence(seq)
            if conf >= min_conf:
                candidates.append((seq.shape[0], conf, seq))
        if not candidates:
            if frames_seen == 0:
                return None, "decode_failed_or_empty_video"
            if keypoint_frames == 0:
                return None, f"no_keypoints frames={frames_seen}"
            track_lengths = sorted((len(frames) for frames in tracks.values()), reverse=True)
            longest = track_lengths[0] if track_lengths else 0
            return None, f"no_valid_track frames={frames_seen} keypoint_frames={keypoint_frames} tracks={len(tracks)} longest={longest}"
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        seq = candidates[0][2]
        with torch.no_grad():
            prob = float(fall_head(seq.to(device)).sigmoid().item())
        window = int(getattr(fall_head, "window", getattr(self.args, "fall_window", 60)))
        stride = int(getattr(fall_head, "stride", getattr(self.args, "fall_stride", 15)))
        clips = 1 if seq.shape[0] <= window else ((seq.shape[0] - window) // stride + 1 + int((seq.shape[0] - window) % stride != 0))
        return prob, f"track_len={seq.shape[0]} clips={clips} candidates={len(candidates)}"

    @staticmethod
    def _compute_metrics(rows: list[dict], threshold: float, elapsed: float) -> dict:
        tp = sum(row["label"] == 1 and row["pred"] == 1 for row in rows)
        tn = sum(row["label"] == 0 and row["pred"] == 0 for row in rows)
        fp = sum(row["label"] == 0 and row["pred"] == 1 for row in rows)
        fn = sum(row["label"] == 1 and row["pred"] == 0 for row in rows)
        total = max(len(rows), 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return {
            "accuracy": (tp + tn) / total,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "videos": len(rows),
            "threshold": threshold,
            "seconds": elapsed,
        }

    def _save_rows(self, rows: list[dict]) -> None:
        save_dir = Path(getattr(self.args, "save_dir", None) or "runs/fall/val")
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "predictions.csv"
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "pred", "prob", "reason"])
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info(f"fall val predictions saved to {out}")
