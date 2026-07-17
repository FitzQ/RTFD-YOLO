# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import random
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ultralytics.models.yolo.fall_utils import (
    atomic_torch_save,
    cached_feature_paths,
    fall_tracker_arg,
    feature_cache_key,
    feature_cache_path,
    opencv_safe_video,
    resample_track,
    video_fps,
)
from ultralytics.models.yolo.fall_engine import FallTrainer, collate_fall_sequences, single_device_args
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.torch_utils import unwrap_model

from .model import (
    POSEFALL_FEATURE_DIM,
    POSEFALL_FEATURE_VERSION,
    PoseFallTransformer,
    mean_keypoint_confidence,
    pose_result_features,
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


@dataclass(frozen=True)
class TrackSequence:
    label: int
    sequence: torch.Tensor
    frame_indices: torch.Tensor
    source_video: str
    track_id: int


def _videos_in_dir(video_dir: str | Path) -> list[Path]:
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Expected video directory not found: {video_dir}")
    return sorted(p for p in video_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def _video_items(root: str | Path) -> list[tuple[Path, int]]:
    root = Path(root)
    return _video_items_from_dirs(root / "Fall", root / "No_Fall")


def _video_items_from_dirs(fall_dir: str | Path, nofall_dir: str | Path) -> list[tuple[Path, int]]:
    fall_videos = _videos_in_dir(fall_dir)
    nofall_videos = _videos_in_dir(nofall_dir)
    if not fall_videos:
        raise ValueError(f"No fall videos found under {fall_dir}")
    if not nofall_videos:
        raise ValueError(f"No non-fall videos found under {nofall_dir}")
    return [(p, 0) for p in nofall_videos] + [(p, 1) for p in fall_videos]


def _video_items_from_args(args) -> list[tuple[Path, int]]:
    fall_data = getattr(args, "fall_train_fall_data", None)
    nofall_data = getattr(args, "fall_train_nofall_data", None)
    if fall_data and nofall_data:
        return _video_items_from_dirs(fall_data, nofall_data)
    dataset_root = Path(getattr(args, "data", None) or "datasets")
    return _video_items(dataset_root)


def _valid_feature_payload(payload: dict, feature_dim: int) -> bool:
    return (
        int(payload.get("feature_dim", 0)) == feature_dim
        and int(payload.get("feature_version", 0)) == POSEFALL_FEATURE_VERSION
        and isinstance(payload.get("sequence"), torch.Tensor)
        and isinstance(payload.get("frame_indices"), torch.Tensor)
        and float(payload.get("source_fps", 0)) > 0
        and bool(payload.get("source_video"))
        and "selected_track_id" in payload
    )


def _limit_items(items: list[tuple[Path, int]], limit: int) -> list[tuple[Path, int]]:
    if limit <= 0:
        return items
    by_label: dict[int, list[tuple[Path, int]]] = {}
    for item in items:
        by_label.setdefault(item[1], []).append(item)
    if len(by_label) <= 1:
        return items[:limit]
    per_class = max(1, limit // len(by_label))
    return [item for label in sorted(by_label) for item in by_label[label][:per_class]]


def _split_feature_items(items: list[Path], seed: int) -> tuple[list[Path], list[Path]]:
    by_class: dict[str, list[Path]] = {}
    for item in items:
        by_class.setdefault(item.parent.name, []).append(item)
    train, val = [], []
    for class_name in sorted(by_class):
        class_train, class_val = _split_labeled_items(by_class[class_name], seed)
        train.extend(class_train)
        val.extend(class_val)
    return train, val


def _split_labeled_items(items, seed: int):
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    cut = max(1, int(len(items) * 0.8))
    return items[:cut], items[cut:] or items[:cut]


def _path_arg(value, default: str) -> Path:
    return Path(value or default)


def _feature_model_key(model_path: str | Path | None) -> str:
    """Return a stable cache folder name for the pose model used to extract posefall features."""
    stem = Path(str(model_path or "yolo26n-pose.pt")).stem
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in stem)


def posefall_feature_settings(args, feature_dim: int) -> dict:
    """Return settings that change cached PoseFall front-end tensors."""
    return {
        "version": POSEFALL_FEATURE_VERSION,
        "feature_dim": feature_dim,
        "imgsz": int(args.imgsz),
        "vid_stride": int(args.vid_stride),
        "conf": args.conf,
        "iou": args.iou,
        "tracker": fall_tracker_arg(args),
        "min_conf": float(args.fall_min_conf),
    }


def _track_ids(result, count: int) -> list[int | None]:
    if result.boxes is not None and result.boxes.is_track and result.boxes.id is not None:
        return [int(x) for x in result.boxes.id.cpu().tolist()]
    return [None] * count


def _extract_tracking_features(args, posefall_feature_dir: Path) -> None:
    model_path = getattr(args, "model", "yolo26n-pose.pt")
    device = getattr(args, "device", None)
    imgsz = int(getattr(args, "imgsz", 640))
    vid_stride = int(getattr(args, "vid_stride", 1))
    conf = getattr(args, "conf", None)
    iou = getattr(args, "iou", None)
    tracker = fall_tracker_arg(args)
    limit = int(getattr(args, "fall_limit", 0))
    overwrite = bool(getattr(args, "fall_overwrite_features", False))
    min_track_len = int(getattr(args, "fall_min_track_len", 30))
    min_conf = float(getattr(args, "fall_min_conf", 0.2))
    feature_dim = int(getattr(args, "posefall_feature_dim", POSEFALL_FEATURE_DIM))
    workers = max(1, int(getattr(args, "fall_extract_workers", 20)))

    LOGGER.info(
        f"Extracting tracked keypoint features to {posefall_feature_dir} "
        f"from fall_train_fall_data={getattr(args, 'fall_train_fall_data', None)} "
        f"fall_train_nofall_data={getattr(args, 'fall_train_nofall_data', None)} with workers={workers}..."
    )
    items = _limit_items(_video_items_from_args(args), limit)
    desc = f"Extracting posefall features ({_feature_model_key(model_path)})"
    if workers == 1:
        for _ in TQDM(
            _extract_video_chunk(
                items, posefall_feature_dir, model_path, device, imgsz, vid_stride, conf, iou, tracker, overwrite, min_track_len, min_conf, feature_dim
            ),
            total=len(items),
            desc=desc,
            unit="video",
        ):
            pass
        return

    chunk_size = max(1, math.ceil(len(items) / workers))
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        futures = [
            executor.submit(
                _extract_video_chunk_count,
                chunk,
                posefall_feature_dir,
                model_path,
                device,
                imgsz,
                vid_stride,
                conf,
                iou,
                tracker,
                overwrite,
                min_track_len,
                min_conf,
                feature_dim,
            )
            for chunk in chunks
        ]
        with TQDM(total=len(items), desc=desc, unit="video") as pbar:
            for future in as_completed(futures):
                pbar.update(future.result())


def _extract_video_chunk_count(
    items: list[tuple[Path, int]],
    posefall_feature_dir: Path,
    model_path: str,
    device,
    imgsz: int,
    vid_stride: int,
    conf,
    iou,
    tracker: str,
    overwrite: bool,
    min_track_len: int,
    min_conf: float,
    feature_dim: int,
) -> int:
    return sum(
        1
        for _ in _extract_video_chunk(
            items, posefall_feature_dir, model_path, device, imgsz, vid_stride, conf, iou, tracker, overwrite, min_track_len, min_conf, feature_dim
        )
    )


def _extract_video_chunk(
    items: list[tuple[Path, int]],
    posefall_feature_dir: Path,
    model_path: str,
    device,
    imgsz: int,
    vid_stride: int,
    conf,
    iou,
    tracker: str,
    overwrite: bool,
    min_track_len: int,
    min_conf: float,
    feature_dim: int,
):
    from ultralytics import YOLO

    pose_model = YOLO(model_path, task="pose")
    for item in items:
        _extract_one_video(
            pose_model, item, posefall_feature_dir, device, imgsz, vid_stride, conf, iou, tracker, overwrite, min_track_len, min_conf, feature_dim
        )
        yield item


def _extract_one_video(
    pose_model,
    item: tuple[Path, int],
    posefall_feature_dir: Path,
    device,
    imgsz: int,
    vid_stride: int,
    conf,
    iou,
    tracker: str,
    overwrite: bool,
    min_track_len: int,
    min_conf: float,
    feature_dim: int,
) -> None:
    video, label = item
    class_name = "Fall" if label else "No_Fall"
    out_path = feature_cache_path(posefall_feature_dir, class_name, video)
    if out_path.exists() and not overwrite:
        try:
            payload = torch.load(out_path, map_location="cpu")
            if _valid_feature_payload(payload, feature_dim):
                LOGGER.debug(f"Skipping existing features: {out_path}")
                return
        except Exception:
            pass
        LOGGER.debug(f"Refreshing incompatible feature cache: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tracks: dict[int, list[torch.Tensor]] = {}
    track_frame_indices: dict[int, list[int]] = {}
    LOGGER.debug(f"Extracting tracked keypoints from {video}...")
    source_video = opencv_safe_video(video, "posefall")
    frame_step = max(int(vid_stride or 1), 1)
    for processed_frame_idx, result in enumerate(pose_model.track(
        source=str(source_video),
        stream=True,
        persist=False,
        device=device,
        imgsz=imgsz,
        vid_stride=vid_stride,
        conf=conf,
        iou=iou,
        tracker=tracker,
        save=False,
        verbose=False,
    )):
        frame_idx = processed_frame_idx * frame_step
        if result.keypoints is None or len(result.keypoints) == 0:
            continue
        ids = _track_ids(result, len(result.keypoints))
        feats = pose_result_features(result, device="cpu")
        for track_id, feat in zip(ids, feats):
            if track_id is None:
                continue
            tracks.setdefault(track_id, []).append(feat)
            track_frame_indices.setdefault(track_id, []).append(frame_idx)
    candidates = []
    for track_id, frames in tracks.items():
        seq = torch.stack(frames)
        mean_conf = mean_keypoint_confidence(seq)
        if mean_conf >= min_conf:
            frame_indices = torch.tensor(track_frame_indices[track_id], dtype=torch.long)
            duration = int(frame_indices[-1] - frame_indices[0] + 1)
            candidates.append((duration, mean_conf, track_id, seq, frame_indices))
    if not candidates:
        LOGGER.debug(
            f"Skipping {video}: no tracked keypoint sequence reached fall_min_track_len={min_track_len} "
            f"and fall_min_conf={min_conf}."
        )
        return
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected_len, selected_conf, selected_track_id, sequence, frame_indices = candidates[0]
    atomic_torch_save(
        {
            "sequence": sequence,
            "frame_indices": frame_indices,
            "source_fps": video_fps(source_video),
            "source_video": str(video),
            "selected_track_id": int(selected_track_id),
            "label": label,
            "feature_dim": feature_dim,
            "feature_version": POSEFALL_FEATURE_VERSION,
        },
        out_path,
    )
    LOGGER.debug(
        f"Saved {out_path} selected_track_span={selected_len} observations={len(sequence)} "
        f"selected_track_id={selected_track_id} candidates={len(candidates)}"
    )


def _load_track_sequences(
    items: list[Path],
    min_track_len: int = 30,
    min_conf: float = 0.2,
    feature_dim: int = POSEFALL_FEATURE_DIM,
    target_fps: float = 30.0,
    max_gap_seconds: float = 0.5,
) -> list[TrackSequence]:
    sequences: list[TrackSequence] = []
    for path in items:
        payload = torch.load(path, map_location="cpu")
        if not _valid_feature_payload(payload, feature_dim):
            LOGGER.warning(
                f"Skipping incompatible posefall feature cache while loading: {path} "
                f"feature_dim={payload.get('feature_dim')} version={payload.get('feature_version')} "
                f"expected_dim={feature_dim} expected_version={POSEFALL_FEATURE_VERSION}"
            )
            continue
        label = int(payload["label"])
        sparse_seq = payload["sequence"].float()
        sparse_indices = payload["frame_indices"].long()
        segments = resample_track(
            list(sparse_seq.unbind()), sparse_indices.tolist(), payload["source_fps"], target_fps, max_gap_seconds
        )
        if not segments:
            continue
        seq, frame_indices = max(segments, key=lambda item: len(item[0]))
        if seq.shape[1] == feature_dim and seq.shape[0] == frame_indices.shape[0] and seq.shape[0] >= min_track_len and mean_keypoint_confidence(seq) >= min_conf:
            sequences.append(
                TrackSequence(
                    label=label,
                    sequence=seq,
                    frame_indices=frame_indices,
                    source_video=str(payload["source_video"]),
                    track_id=int(payload["selected_track_id"]),
                )
            )
    if not sequences:
        raise ValueError("No tracked keypoint sequences were found in cached feature files.")
    return sequences


class PoseFallTrackDataset(Dataset):
    """One training sample per tracked person sequence."""

    def __init__(self, sequences: list[TrackSequence]):
        self.sequences = sequences
        if not self.sequences:
            raise ValueError("No usable training samples were built from tracked keypoint sequences.")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.sequences[index]
        return item.sequence, torch.tensor(float(item.label), dtype=torch.float32)


class PoseFallTrainer(FallTrainer):
    """Native Ultralytics trainer for the frozen Pose front end and PoseFall temporal head."""

    task_name = "posefall"
    head_attr = "posefall_head"
    config_attr = "posefall_config"
    frontend_type = PoseModel

    @property
    def validator_class(self):
        from .val import PoseFallValidator

        return PoseFallValidator

    def build_fall_datasets(self):
        feature_dim = int(getattr(self.args, "posefall_feature_dim", POSEFALL_FEATURE_DIM))
        min_track_len = int(getattr(self.args, "fall_min_track_len", 30))
        min_conf = float(getattr(self.args, "fall_min_conf", 0.2))
        target_fps = float(getattr(self.args, "fall_target_fps", 30.0))
        max_gap_seconds = float(getattr(self.args, "fall_max_gap_seconds", 0.5))
        settings = posefall_feature_settings(self.args, feature_dim)
        root = _path_arg(self.args.posefall_feature_dir, "runs/posefall/features")
        key = self.args.posefall_feature_key or feature_cache_key(self.args.model, "posefall", settings)
        self.args.posefall_feature_key = key
        feature_dir = root / key
        expected = _limit_items(_video_items_from_args(self.args), int(self.args.fall_limit))
        cached_items = cached_feature_paths(
            feature_dir, expected, lambda payload: _valid_feature_payload(payload, feature_dim)
        )
        complete = len(cached_items) == len(expected)
        if self.args.fall_overwrite_features or not complete:
            _extract_tracking_features(single_device_args(self.args), feature_dir)
            cached_items = cached_feature_paths(
                feature_dir, expected, lambda payload: _valid_feature_payload(payload, feature_dim)
            )
        train_items, val_items = _split_feature_items(cached_items, int(self.args.seed))
        train = PoseFallTrackDataset(
            _load_track_sequences(train_items, min_track_len, min_conf, feature_dim, target_fps, max_gap_seconds)
        )
        val = PoseFallTrackDataset(
            _load_track_sequences(val_items, min_track_len, min_conf, feature_dim, target_fps, max_gap_seconds)
        )
        train.collate_fn = val.collate_fn = collate_fall_sequences
        return train, val

    def build_fall_head(self):
        return PoseFallTransformer(
            input_dim=int(self.args.posefall_feature_dim),
            window=int(self.args.fall_window),
            stride=int(self.args.fall_stride),
        )

    def complete_config(self):
        head = unwrap_model(self.model).fall_head
        return {
            "window": head.window,
            "input_dim": head.input_dim,
            "d_model": head.d_model,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "stride": head.stride,
            "feature_version": POSEFALL_FEATURE_VERSION,
            "feature_key": self.args.posefall_feature_key,
            "feature_type": "box_keypoints",
            "pooling": "learned_attention",
            "clip_aggregation": "max",
            "positive_topk": int(self.args.fall_positive_topk),
            "positive_clip_weight": float(self.args.fall_positive_clip_weight),
            "negative_clip_weight": float(self.args.fall_negative_clip_weight),
            "temporal_smooth_weight": float(self.args.fall_temporal_smooth_weight),
            "target_fps": float(self.args.fall_target_fps),
            "max_gap_seconds": float(self.args.fall_max_gap_seconds),
        }
