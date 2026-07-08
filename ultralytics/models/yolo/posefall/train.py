# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import random
import math
import hashlib
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, TQDM

from .model import POSEFALL_CENTER_Y_INDEX, POSEFALL_FEATURE_DIM, PoseFallTransformer, load_posefall_head, mean_keypoint_confidence, normalize_keypoints

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


@dataclass(frozen=True)
class TrackSequence:
    label: int
    sequence: torch.Tensor
    frame_indices: torch.Tensor
    source_video: str
    track_id: int


@dataclass(frozen=True)
class SequenceSample:
    sequence_index: int
    start: int
    end: int


@dataclass(frozen=True)
class PoseFallClipCandidate:
    source_video: str
    frame_indices: tuple[int, ...]
    out_path: Path


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
    fall_data = getattr(args, "posefall_train_fall_data", None)
    nofall_data = getattr(args, "posefall_train_nofall_data", None)
    if fall_data and nofall_data:
        return _video_items_from_dirs(fall_data, nofall_data)
    dataset_root = Path(getattr(args, "data", None) or "datasets")
    return _video_items(dataset_root)


def _valid_feature_payload(payload: dict, feature_dim: int) -> bool:
    return (
        int(payload.get("feature_dim", 0)) == feature_dim
        and int(payload.get("feature_version", 0)) == 3
        and isinstance(payload.get("sequence"), torch.Tensor)
        and isinstance(payload.get("frame_indices"), torch.Tensor)
        and bool(payload.get("source_video"))
        and "selected_track_id" in payload
    )


def _feature_items(root: str | Path, feature_dim: int = POSEFALL_FEATURE_DIM) -> list[Path]:
    root = Path(root)
    items = []
    for class_name in ("No_Fall", "Fall"):
        class_dir = root / class_name
        if class_dir.exists():
            for path in sorted(class_dir.glob("*.pt")):
                try:
                    payload = torch.load(path, map_location="cpu")
                except Exception:
                    LOGGER.warning(f"Skipping unreadable posefall feature cache: {path}")
                    continue
                if _valid_feature_payload(payload, feature_dim):
                    items.append(path)
                else:
                    LOGGER.warning(
                        f"Skipping incompatible posefall feature cache: {path} "
                        f"feature_dim={payload.get('feature_dim')} version={payload.get('feature_version')} expected_dim={feature_dim} expected_version=3"
                    )
    if not items:
        raise ValueError(f"No feature .pt files found under {root}/{{Fall,No_Fall}}")
    return items


def _has_feature_items(root: str | Path, feature_dim: int = POSEFALL_FEATURE_DIM) -> bool:
    root = Path(root)
    counts = {"No_Fall": 0, "Fall": 0}
    for class_name in counts:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for path in class_dir.glob("*.pt"):
            try:
                payload = torch.load(path, map_location="cpu")
            except Exception:
                continue
            if _valid_feature_payload(payload, feature_dim):
                counts[class_name] += 1
    return all(count > 0 for count in counts.values())


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


def _torch_device(device_arg) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if device_arg in {None, ""}:
        return torch.device("cuda:0")
    device = str(device_arg)
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _path_arg(value, default: str) -> Path:
    return Path(value or default)


def _posefall_tracker_arg(args) -> str:
    tracker = getattr(args, "tracker", None)
    if tracker and tracker != DEFAULT_CFG_DICT.get("tracker"):
        return tracker
    return getattr(args, "posefall_tracker", None) or tracker or "posefall_botsort.yaml"


def _posefall_lr_arg(args) -> float:
    lr0 = float(getattr(args, "lr0", DEFAULT_CFG_DICT.get("lr0", 0.01)))
    if lr0 != float(DEFAULT_CFG_DICT.get("lr0", 0.01)):
        return lr0
    return float(getattr(args, "posefall_lr0", 5e-4))


def _feature_model_key(model_path: str | Path | None) -> str:
    """Return a stable cache folder name for the pose model used to extract posefall features."""
    stem = Path(str(model_path or "yolo26n-pose.pt")).stem
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in stem)


def _model_posefall_feature_dir(feature_root: str | Path, model_path: str | Path | None) -> Path:
    """Feature cache root scoped by pose model, e.g. runs/posefall/features/yolo26n-pose."""
    return Path(feature_root) / _feature_model_key(model_path)


def _track_ids(result, count: int) -> list[int | None]:
    if result.boxes is not None and result.boxes.is_track and result.boxes.id is not None:
        return [int(x) for x in result.boxes.id.cpu().tolist()]
    return [None] * count


def _safe_stem(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in value)


def _opencv_safe_video(video: Path) -> Path:
    """Transcode legacy AVI files to cached MP4 before OpenCV/YOLO reads them.

    Some Le2i AVI files contain raw BGR video plus malformed audio; OpenCV may abort in native code
    before Python can catch the error. Training uses this same safe path as validation.
    """
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
        raise RuntimeError(f"ffmpeg is required to extract legacy AVI videos such as {video}") from e
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to transcode AVI video for posefall feature extraction: {video}") from e
    tmp.rename(cached)
    return cached


def _extract_tracking_features(args, posefall_feature_dir: Path) -> None:
    model_path = getattr(args, "model", "yolo26n-pose.pt")
    device = getattr(args, "device", None)
    imgsz = int(getattr(args, "imgsz", 640))
    vid_stride = int(getattr(args, "vid_stride", 1))
    conf = getattr(args, "conf", None)
    iou = getattr(args, "iou", None)
    tracker = _posefall_tracker_arg(args)
    limit = int(getattr(args, "posefall_limit", 0))
    overwrite = bool(getattr(args, "posefall_overwrite_features", False))
    min_track_len = int(getattr(args, "posefall_min_track_len", 30))
    min_conf = float(getattr(args, "posefall_min_conf", 0.2))
    feature_dim = int(getattr(args, "posefall_feature_dim", POSEFALL_FEATURE_DIM))
    workers = max(1, int(getattr(args, "posefall_extract_workers", 20)))

    LOGGER.info(
        f"Extracting tracked keypoint features to {posefall_feature_dir} "
        f"from posefall_train_fall_data={getattr(args, 'posefall_train_fall_data', None)} "
        f"posefall_train_nofall_data={getattr(args, 'posefall_train_nofall_data', None)} with workers={workers}..."
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
    out_path = posefall_feature_dir / class_name / f"{video.stem}.pt"
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
    previous_center_y: dict[int, float] = {}
    last_frame_idx: dict[int, int] = {}
    last_feat: dict[int, torch.Tensor] = {}
    LOGGER.debug(f"Extracting tracked keypoints from {video}...")
    source_video = _opencv_safe_video(video)
    frame_step = max(int(vid_stride or 1), 1)
    for processed_frame_idx, result in enumerate(pose_model.track(
        source=str(source_video),
        stream=True,
        persist=True,
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
        boxes_xywhn = result.boxes.xywhn.detach().cpu() if result.boxes is not None else None
        prev = [previous_center_y.get(track_id) if track_id is not None else None for track_id in ids]
        feats = normalize_keypoints(
            result.keypoints.data.detach().cpu(), result.orig_shape, boxes_xywhn=boxes_xywhn, previous_center_y=prev, feature_dim=feature_dim
        )
        for track_id, feat in zip(ids, feats):
            if track_id is None:
                continue
            if track_id in last_frame_idx and track_id in last_feat:
                for missing_idx in range(last_frame_idx[track_id] + 1, frame_idx):
                    tracks.setdefault(track_id, []).append(last_feat[track_id].clone())
                    track_frame_indices.setdefault(track_id, []).append(missing_idx)
            tracks.setdefault(track_id, []).append(feat)
            track_frame_indices.setdefault(track_id, []).append(frame_idx)
            last_frame_idx[track_id] = frame_idx
            last_feat[track_id] = feat.detach().clone()
            if feat.shape[0] > POSEFALL_CENTER_Y_INDEX:
                previous_center_y[track_id] = float(feat[POSEFALL_CENTER_Y_INDEX])
    candidates = []
    for track_id, frames in tracks.items():
        if len(frames) < min_track_len:
            continue
        seq = torch.stack(frames)
        mean_conf = mean_keypoint_confidence(seq)
        if mean_conf >= min_conf:
            frame_indices = torch.tensor(track_frame_indices[track_id], dtype=torch.long)
            candidates.append((seq.shape[0], mean_conf, track_id, seq, frame_indices))
    if not candidates:
        LOGGER.debug(
            f"Skipping {video}: no tracked keypoint sequence reached posefall_min_track_len={min_track_len} "
            f"and posefall_min_conf={min_conf}."
        )
        return
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected_len, selected_conf, selected_track_id, sequence, frame_indices = candidates[0]
    torch.save(
        {
            "sequence": sequence,
            "frame_indices": frame_indices,
            "source_video": str(video),
            "selected_track_id": int(selected_track_id),
            "label": label,
            "feature_dim": feature_dim,
            "feature_version": 3,
        },
        out_path,
    )
    LOGGER.debug(f"Saved {out_path} selected_track_len={selected_len} selected_track_id={selected_track_id} candidates={len(candidates)}")


def _load_track_sequences(
    items: list[Path], min_track_len: int = 30, min_conf: float = 0.2, feature_dim: int = POSEFALL_FEATURE_DIM
) -> list[TrackSequence]:
    sequences: list[TrackSequence] = []
    for path in items:
        payload = torch.load(path, map_location="cpu")
        if not _valid_feature_payload(payload, feature_dim):
            LOGGER.warning(
                f"Skipping incompatible posefall feature cache while loading: {path} "
                f"feature_dim={payload.get('feature_dim')} version={payload.get('feature_version')} expected_dim={feature_dim} expected_version=3"
            )
            continue
        label = int(payload["label"])
        seq = payload["sequence"].float()
        frame_indices = payload["frame_indices"].long()
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


def collate_track_sequences(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Keep variable-length sequences as a list; the model performs sliding-window pooling."""
    sequences, labels = zip(*batch)
    return list(sequences), torch.stack(labels)


class PoseFallTrainer:
    """Train the posefall temporal head from YOLO pose postprocess keypoints and track ids."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        overrides = dict(overrides or {})
        self.session = overrides.pop("session", None)
        self.args = get_cfg(cfg, overrides)
        self.callbacks = _callbacks

    def __call__(self, *args, **kwargs):
        return self.train(*args, **kwargs)

    def train(self):
        feature_root = _path_arg(getattr(self.args, "posefall_feature_dir", None), "runs/posefall/features")
        model_path = getattr(self.args, "model", "yolo26n-pose.pt")
        posefall_feature_dir = _model_posefall_feature_dir(feature_root, model_path)
        batch = int(getattr(self.args, "batch", 64))
        epochs = int(getattr(self.args, "epochs", 100))
        patience = int(getattr(self.args, "patience", 100))
        lr = _posefall_lr_arg(self.args)
        seed = int(getattr(self.args, "seed", 0))
        device = _torch_device(getattr(self.args, "device", None))
        out = _path_arg(getattr(self.args, "posefall_out", None), "")
        window = int(getattr(self.args, "posefall_window", getattr(self.args, "window", 60)))
        stride = int(getattr(self.args, "posefall_stride", 15))
        min_track_len = int(getattr(self.args, "posefall_min_track_len", 30))
        min_conf = float(getattr(self.args, "posefall_min_conf", 0.2))
        feature_dim = int(getattr(self.args, "posefall_feature_dim", POSEFALL_FEATURE_DIM))
        summary_tail = int(getattr(self.args, "posefall_summary_tail", 60))
        overwrite = bool(getattr(self.args, "posefall_overwrite_features", False))
        grad_clip = float(getattr(self.args, "posefall_grad_clip", 1.0))
        posefall_weights = getattr(self.args, "posefall_weights", None)
        save_train_clips = bool(getattr(self.args, "posefall_save_train_clips", False))
        train_clip_dir = _path_arg(getattr(self.args, "posefall_train_clip_dir", None), "")
        train_clip_threshold = float(getattr(self.args, "posefall_train_clip_threshold", 0.5))
        train_clip_max = int(getattr(self.args, "posefall_train_clip_max", 200))

        torch.manual_seed(seed)
        if posefall_weights:
            model, config = load_posefall_head(posefall_weights, device, input_dim=feature_dim, window=window)
            if model is None:
                raise ValueError(f"posefall_weights={posefall_weights!r} did not load a posefall head checkpoint.")
            feature_dim = int(config.get("input_dim", feature_dim))
            window = int(config.get("window") or window)
            stride = int(config.get("stride", stride))
            summary_tail = int(config.get("summary_tail", summary_tail))
            model.train()
            LOGGER.info(f"Resuming posefall head training from posefall_weights={posefall_weights}")
        else:
            model = PoseFallTransformer(input_dim=feature_dim, window=window, summary_tail=summary_tail, stride=stride).to(device)

        LOGGER.info(f"Using posefall feature cache: {posefall_feature_dir}")
        if overwrite or not _has_feature_items(posefall_feature_dir, feature_dim=feature_dim):
            _extract_tracking_features(self.args, posefall_feature_dir)
        train_items, val_items = _split_feature_items(_feature_items(posefall_feature_dir, feature_dim=feature_dim), seed)
        train_sequences = _load_track_sequences(train_items, min_track_len=min_track_len, min_conf=min_conf, feature_dim=feature_dim)
        val_sequences = _load_track_sequences(val_items, min_track_len=min_track_len, min_conf=min_conf, feature_dim=feature_dim)
        train_set = PoseFallTrackDataset(train_sequences)
        val_set = PoseFallTrackDataset(val_sequences)
        train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_track_sequences)
        val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_track_sequences)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        positives = sum(item.label for item in train_sequences)
        negatives = len(train_sequences) - positives
        # pos_weight = torch.tensor([negatives / max(positives, 1)], dtype=torch.float32, device=device)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion = nn.BCEWithLogitsLoss()
        LOGGER.info(
            f"posefall train samples={len(train_sequences)} positives={positives} negatives={negatives} "
            f"val_samples={len(val_sequences)} lr={lr:g}"
        )
        best_loss = float("inf")
        best_epoch = 0
        best_metrics = None
        epochs_without_improvement = 0
        out.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = correct = total = 0
            train_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
            train_probs: list[float] = []
            train_targets: list[int] = []
            for x, y in train_loader:
                x, y = [seq.to(device) for seq in x], y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite posefall training loss at epoch={epoch}: {float(loss.detach().cpu())}")
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                total_loss += loss.item() * y.numel()
                probs = logits.sigmoid()
                preds = probs >= 0.5
                correct += (preds == y.bool()).sum().item()
                self._update_binary_counts(train_counts, preds, y)
                train_probs.extend(float(v) for v in probs.detach().cpu())
                train_targets.extend(int(v) for v in y.detach().cpu())
                total += y.numel()
            train_metrics = self._binary_metrics(train_counts, total_loss / max(total, 1), correct / max(total, 1))
            train_metrics.update(self._competition_metrics(train_targets, train_probs))
            val_metrics = self._evaluate(model, val_loader, criterion, device)
            LOGGER.info(
                f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f} "
                f"train_p={train_metrics['precision']:.3f} train_r={train_metrics['recall']:.3f} train_f1={train_metrics['f1']:.3f} "
                f"train_p90={train_metrics['p90']:.3f} train_p95={train_metrics['p95']:.3f} train_map={train_metrics['competition_map']:.2f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} "
                f"val_p={val_metrics['precision']:.3f} val_r={val_metrics['recall']:.3f} val_f1={val_metrics['f1']:.3f} "
                f"val_p90={val_metrics['p90']:.3f} val_p95={val_metrics['p95']:.3f} val_map={val_metrics['competition_map']:.2f}"
            )
            val_loss = val_metrics["loss"]
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_metrics = val_metrics
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": {
                            "window": window,
                            "input_dim": feature_dim,
                            "num_classes": 1,
                            "d_model": 256,
                            "nhead": 4,
                            "num_layers": 3,
                            "dim_feedforward": 256,
                            "dropout": 0.1,
                            "posefall_min_conf": min_conf,
                            "feature_version": 3,
                            "pooling": "temporal_summary",
                            "summary_tail": summary_tail,
                            "stride": stride,
                        },
                        "labels": ["normal", "fall"],
                    },
                    out,
                )
            else:
                epochs_without_improvement += 1
                if patience > 0 and epochs_without_improvement >= patience:
                    LOGGER.info(
                        f"early stopping at epoch={epoch:03d}: no val_loss improvement for {patience} epochs "
                        f"(best_epoch={best_epoch:03d}, best_val_loss={best_loss:.4f})"
                    )
                    break
        if save_train_clips:
            saved = self._save_train_predicted_clips(
                model, train_sequences, train_clip_dir, train_clip_threshold, train_clip_max, window, stride, device
            )
            LOGGER.info(f"saved {saved} train predicted posefall clips to {train_clip_dir}")
        LOGGER.info(
            f"best checkpoint saved to {out}\n"
            f" best_epoch={best_epoch:03d}, best_val_loss={best_loss:.4f}, best_val_acc={best_metrics['acc']:.3f}, best_val_f1={best_metrics['f1']:.3f},"
            f" best_val_p={best_metrics['precision']:.3f}, best_val_r={best_metrics['recall']:.3f},"
            f" best_val_p90={best_metrics['p90']:.3f}, best_val_p95={best_metrics['p95']:.3f}, best_val_map={best_metrics['competition_map']:.2f}"
        )
        self.best = out
        return {"best": out, "best_loss": best_loss, "best_acc": best_metrics["acc"]}

    @staticmethod
    def _evaluate(model, loader, criterion, device):
        model.eval()
        total_loss = correct = total = 0
        counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        probs_all: list[float] = []
        targets_all: list[int] = []
        with torch.no_grad():
            for x, y in loader:
                x, y = [seq.to(device) for seq in x], y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                total_loss += loss.item() * y.numel()
                probs = logits.sigmoid()
                preds = probs >= 0.5
                correct += (preds == y.bool()).sum().item()
                PoseFallTrainer._update_binary_counts(counts, preds, y)
                probs_all.extend(float(v) for v in probs.detach().cpu())
                targets_all.extend(int(v) for v in y.detach().cpu())
                total += y.numel()
        metrics = PoseFallTrainer._binary_metrics(counts, total_loss / max(total, 1), correct / max(total, 1))
        metrics.update(PoseFallTrainer._competition_metrics(targets_all, probs_all))
        return metrics

    @staticmethod
    def _update_binary_counts(counts: dict[str, int], preds: torch.Tensor, y: torch.Tensor) -> None:
        target = y.bool()
        counts["tp"] += (preds & target).sum().item()
        counts["fp"] += (preds & ~target).sum().item()
        counts["tn"] += (~preds & ~target).sum().item()
        counts["fn"] += (~preds & target).sum().item()

    @staticmethod
    def _binary_metrics(counts: dict[str, int], loss: float, acc: float) -> dict[str, float]:
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return {"loss": loss, "acc": acc, "precision": precision, "recall": recall, "f1": f1, **counts}

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

    @staticmethod
    def _clip_starts(seq_len: int, window: int, stride: int) -> list[int]:
        if seq_len <= window:
            return [0]
        starts = list(range(0, seq_len - window + 1, stride))
        if (seq_len - window) % stride != 0:
            starts.append(seq_len - window)
        return starts

    @staticmethod
    def _window_frame_indices(frame_indices: torch.Tensor, start: int, window: int) -> list[int]:
        values = frame_indices.tolist()
        clip = values[start : start + window]
        if not clip:
            return []
        if len(clip) < window:
            clip.extend([clip[-1]] * (window - len(clip)))
        return [int(x) for x in clip]

    @staticmethod
    def _safe_stem(value: str) -> str:
        return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in value)

    @staticmethod
    def _write_video_clip_from_capture(cap, fps: float, source_video: str, frame_indices: tuple[int, ...], out_path: Path) -> bool:
        import cv2

        if not frame_indices:
            return False
        writer = None
        current_frame_idx = None
        last_frame = None
        try:
            for frame_idx in frame_indices:
                frame_idx = max(int(frame_idx), 0)
                if current_frame_idx == frame_idx and last_frame is not None:
                    frame = last_frame
                else:
                    if current_frame_idx is None or frame_idx < current_frame_idx or frame_idx > current_frame_idx + 1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame = cap.read()
                    if not ok:
                        LOGGER.warning(f"Unable to read frame={frame_idx} from {source_video}")
                        return False
                    current_frame_idx = frame_idx
                    last_frame = frame
                if writer is None:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    if not writer.isOpened():
                        LOGGER.warning(f"Unable to open posefall clip writer: {out_path}")
                        out_path.unlink(missing_ok=True)
                        return False
                writer.write(frame)
        finally:
            if writer is not None:
                writer.release()
        return True

    @staticmethod
    def _write_grouped_video_clips(candidates: list[PoseFallClipCandidate]) -> int:
        import cv2

        grouped: dict[str, list[PoseFallClipCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.source_video, []).append(candidate)

        saved = 0
        for source_video, group in TQDM(grouped.items(), desc="Saving train posefall clips"):
            cap = cv2.VideoCapture(source_video)
            if not cap.isOpened():
                LOGGER.warning(f"Unable to open source video for posefall clip export: {source_video}")
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            try:
                group.sort(key=lambda item: item.frame_indices[0] if item.frame_indices else 0)
                for candidate in group:
                    if PoseFallTrainer._write_video_clip_from_capture(cap, fps, source_video, candidate.frame_indices, candidate.out_path):
                        saved += 1
                    else:
                        candidate.out_path.unlink(missing_ok=True)
            finally:
                cap.release()
        return saved

    @staticmethod
    def _save_train_predicted_clips(
        model: PoseFallTransformer,
        sequences: list[TrackSequence],
        out_dir: Path,
        threshold: float,
        max_clips: int,
        window: int,
        stride: int,
        device: torch.device,
    ) -> int:
        model.eval()
        candidates: list[PoseFallClipCandidate] = []
        with torch.no_grad():
            for seq_index, item in enumerate(sequences):
                seq = item.sequence.to(device)
                clips = model._sliding_clips(seq)
                probs = model._clip_logits(clips).sigmoid().detach().cpu().tolist()
                starts = PoseFallTrainer._clip_starts(item.sequence.shape[0], window, stride)
                for clip_index, prob in sorted(enumerate(probs), key=lambda x: x[1], reverse=True):
                    if prob < threshold:
                        continue
                    if max_clips > 0 and len(candidates) >= max_clips:
                        return PoseFallTrainer._write_grouped_video_clips(candidates)
                    start = starts[clip_index] if clip_index < len(starts) else 0
                    frames = PoseFallTrainer._window_frame_indices(item.frame_indices, start, window)
                    if not frames:
                        continue
                    source_stem = PoseFallTrainer._safe_stem(Path(item.source_video).stem)
                    out_path = (
                        out_dir
                        / f"label{item.label}_prob{prob:.3f}_{source_stem}_track{item.track_id}_seq{seq_index}_clip{clip_index}_len{len(frames)}_frames{frames[0]}-{frames[-1]}.mp4"
                    )
                    candidates.append(PoseFallClipCandidate(item.source_video, tuple(frames), out_path))
        return PoseFallTrainer._write_grouped_video_clips(candidates)
