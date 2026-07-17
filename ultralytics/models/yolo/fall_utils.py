from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch

from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.torch_utils import select_device


@contextmanager
def _exclusive_file_lock(path: Path):
    """Serialize cache creation across independent training and validation processes."""
    import fcntl

    with path.open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def opencv_safe_video(video: str | Path, task: str) -> Path:
    """Transcode legacy AVI input to a cached MP4 to avoid native OpenCV decoder crashes."""
    video = Path(video)
    if video.suffix.lower() != ".avi":
        return video
    cache_dir = Path("runs") / task / "video_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(video.resolve()).encode()).hexdigest()[:12]
    safe_stem = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in video.stem)
    cached = cache_dir / f"{digest}_{safe_stem}.mp4"
    if cached.exists() and cached.stat().st_size > 0:
        return cached
    with _exclusive_file_lock(cached.with_suffix(".mp4.lock")):
        if cached.exists() and cached.stat().st_size > 0:
            return cached
        descriptor, temporary_name = tempfile.mkstemp(prefix=f"{cached.name}.", suffix=".part", dir=cache_dir)
        os.close(descriptor)
        temporary = Path(temporary_name)
        command = [
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
            str(temporary),
        ]
        try:
            subprocess.run(command, check=True)
            os.replace(temporary, cached)
        except FileNotFoundError as error:
            raise RuntimeError(f"ffmpeg is required to read legacy AVI video: {video}") from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Failed to transcode legacy AVI video: {video}") from error
        finally:
            temporary.unlink(missing_ok=True)
    return cached


def fall_device(device=None) -> torch.device:
    """Resolve a physical CLI device to the process-local device used by Ultralytics."""
    return select_device(device, verbose=False)


def fall_tracker_arg(args) -> str:
    """Resolve the shared Fall tracker while honoring an explicit generic tracker override."""
    tracker = getattr(args, "tracker", None)
    if tracker and tracker != DEFAULT_CFG_DICT.get("tracker"):
        return tracker
    return getattr(args, "fall_tracker", None) or tracker or "fall_botsort.yaml"


def fall_yolo_model(model, fallback: str | Path, task: str):
    """Return a YOLO wrapper whether BaseValidator supplied weights or an already loaded module."""
    from ultralytics import YOLO

    if hasattr(model, "track") and hasattr(model, "model"):
        return model
    wrapper = YOLO(model if isinstance(model, (str, Path)) else fallback, task=task)
    if isinstance(model, torch.nn.Module):
        wrapper.model = model
        wrapper.task = task
        wrapper.overrides["task"] = task
    return wrapper


def safe_video_source(source, task: str):
    """Apply AVI safety conversion to file or list sources while preserving cameras and streams."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_file():
            return str(opencv_safe_video(path, task))
    if isinstance(source, (list, tuple)):
        return [safe_video_source(item, task) for item in source]
    return source


def aggregate_clip_logits(clip_logits: torch.Tensor) -> torch.Tensor:
    """Apply the video rule: one positive clip makes the full video positive."""
    if clip_logits.numel() == 0:
        raise ValueError("Cannot aggregate an empty clip-logit tensor")
    return clip_logits.amax(dim=0)


def valid_prediction_rows(rows: list[dict]) -> list[dict]:
    """Return rows backed by an actual model probability."""
    return [row for row in rows if row.get("prob") is not None]


def video_classification_metrics(rows: list[dict], threshold: float, elapsed: float, competition_fn) -> dict:
    """Compute video metrics without silently treating failed videos as negative predictions."""
    valid_rows = valid_prediction_rows(rows)
    tp = sum(row["label"] == 1 and row["pred"] == 1 for row in valid_rows)
    tn = sum(row["label"] == 0 and row["pred"] == 0 for row in valid_rows)
    fp = sum(row["label"] == 0 and row["pred"] == 1 for row in valid_rows)
    fn = sum(row["label"] == 1 and row["pred"] == 0 for row in valid_rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    valid_count = len(valid_rows)
    competition = competition_fn(
        [int(row["label"]) for row in valid_rows],
        [float(row["prob"]) for row in valid_rows],
    )
    return {
        "accuracy": (tp + tn) / max(valid_count, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        **competition,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "videos": len(rows),
        "valid_videos": valid_count,
        "invalid_videos": len(rows) - valid_count,
        "coverage": valid_count / max(len(rows), 1),
        "threshold": threshold,
        "seconds": elapsed,
    }


def cached_feature_paths(root: Path, sources: list[tuple[Path, int]], validator) -> list[Path]:
    """Return valid cache files for exactly the configured sources, without scanning unrelated datasets."""
    paths = []
    for video, label in sources:
        path = feature_cache_path(root, "Fall" if label else "No_Fall", video)
        if not path.is_file():
            continue
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            continue
        if validator(payload):
            paths.append(path)
    return paths


def video_fps(path: str | Path, fallback: float = 30.0) -> float:
    """Read a video's FPS without keeping the decoder open."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
    cap.release()
    return fps if math.isfinite(fps) and fps > 0 else fallback


def resample_track(
    features: list[torch.Tensor],
    frame_indices: list[int],
    source_fps: float,
    target_fps: float,
    max_gap_seconds: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Resample sparse tracked observations to a stable timebase and split long detection gaps."""
    if not features:
        return []
    source_fps = max(float(source_fps), 1e-6)
    target_fps = max(float(target_fps), 1e-6)
    max_gap_frames = max(1, round(max_gap_seconds * source_fps))
    segments: list[tuple[torch.Tensor, torch.Tensor]] = []
    start = 0
    for end in range(1, len(frame_indices) + 1):
        split = end == len(frame_indices) or frame_indices[end] - frame_indices[end - 1] > max_gap_frames
        if not split:
            continue
        segment_features = torch.stack(features[start:end])
        source_indices = torch.tensor(frame_indices[start:end], dtype=torch.float64)
        first_tick = math.ceil(float(source_indices[0]) * target_fps / source_fps)
        last_tick = math.floor(float(source_indices[-1]) * target_fps / source_fps)
        target_ticks = torch.arange(first_tick, max(first_tick, last_tick) + 1, dtype=torch.long)
        source_positions = target_ticks.to(torch.float64) * source_fps / target_fps
        selected = torch.searchsorted(source_indices, source_positions, right=True).sub_(1).clamp_(0, len(source_indices) - 1)
        segments.append((segment_features[selected], source_positions.round().long()))
        start = end
    return segments


def append_resampled_feature(history, feature: torch.Tensor, tick: int, last_tick: int | None, max_gap_ticks: int) -> int:
    """Append one real-time feature while preserving target-time ticks with repeat-last filling."""
    if last_tick is None or tick - last_tick > max_gap_ticks:
        history.clear()
        history.append(feature)
        return tick
    if tick <= last_tick:
        if history:
            history[-1] = feature
        else:
            history.append(feature)
        return last_tick
    for _ in range(tick - last_tick - 1):
        history.append(history[-1].clone())
    history.append(feature)
    return tick


def predictor_fps(predictor, fallback: float = 30.0) -> float:
    """Return scalar source FPS from an Ultralytics predictor dataset."""
    fps = getattr(getattr(predictor, "dataset", None), "fps", fallback)
    if isinstance(fps, (list, tuple)):
        fps = fps[0] if fps else fallback
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = fallback
    return fps if math.isfinite(fps) and fps > 0 else fallback


def feature_cache_key(model_path: str | Path, task: str, settings: dict) -> str:
    """Build a cache identity from model contents and all extraction-affecting settings."""
    path = Path(str(model_path))
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    else:
        digest.update(str(model_path).encode())
    digest.update(json.dumps(settings, sort_keys=True, default=str).encode())
    stem = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in path.stem)
    return f"{stem}-{task}-{digest.hexdigest()[:12]}"


def resolve_feature_cache_key(args, model, task: str, arg_name: str, settings: dict) -> str:
    """Resolve the shared front-end cache key from CLI, complete weights, or the original front-end weights."""
    explicit = getattr(args, arg_name, None)
    if explicit:
        return str(explicit)

    objects = [model]
    inner_model = getattr(model, "model", None)
    if inner_model is not None:
        objects.append(inner_model)
    for obj in objects:
        config = getattr(obj, f"{task}_config", {}) or {}
        if config.get("feature_key"):
            return str(config["feature_key"])

    stored_args = []
    for obj in objects:
        for value in (getattr(obj, "overrides", None), getattr(obj, "args", None)):
            if value:
                stored_args.append(value if isinstance(value, dict) else vars(value))
        checkpoint = getattr(obj, "ckpt", None)
        if isinstance(checkpoint, dict) and checkpoint.get("train_args"):
            stored_args.append(checkpoint["train_args"])
    for model_args in stored_args:
        if model_args.get(arg_name):
            return str(model_args[arg_name])
    for model_args in stored_args:
        source_model = model_args.get("model")
        if source_model and task not in Path(str(source_model)).stem and Path(str(source_model)).is_file():
            return feature_cache_key(source_model, task, settings)
    return feature_cache_key(getattr(args, "model", task), task, settings)


def feature_cache_path(root: Path, class_name: str, video: Path) -> Path:
    """Return a collision-resistant path for recursively discovered videos."""
    suffix = hashlib.sha1(str(video.resolve()).encode()).hexdigest()[:10]
    return root / class_name / f"{video.stem}-{suffix}.pt"


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Write a torch payload atomically so interrupted extraction cannot leave a valid-looking partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
