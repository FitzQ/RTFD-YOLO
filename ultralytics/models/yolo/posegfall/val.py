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
    valid_prediction_rows,
    video_classification_metrics,
    video_fps,
)
from ultralytics.utils import LOGGER, TQDM

from .model import POSEGFALL_FEATURE_DIM, mean_poseg_confidence, poseg_result_features
from .train import _extract_one_video, _load_track_sequences, posegfall_feature_settings

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
            "posegfall val requires fall_val_fall_data=/path/to/Fall "
            "fall_val_nofall_data=/path/to/No_Fall or data=/path/to/labeled/video_dataset"
        )
    return _video_items(data, limit=limit)


def _track_ids(result, count: int) -> list[int | None]:
    if result.boxes is not None and result.boxes.is_track and result.boxes.id is not None:
        return [int(x) for x in result.boxes.id.cpu().tolist()]
    return list(range(count))


class PoseSegFallValidator(FallValidator):
    """End-to-end pose-based video-level fall validator.

    This validator starts from raw videos, runs YOLO pose tracking, converts the main tracked person to posegfall
    features, and evaluates the trained posegfall head as a binary video classifier.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None):
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)

    def __call__(self, trainer=None, model=None):
        if trainer is not None:
            return self.validate_cached(trainer)
        device = getattr(self.args, "device", None)
        feature_dim = int(getattr(self.args, "posegfall_feature_dim", POSEGFALL_FEATURE_DIM))
        min_track_len = int(getattr(self.args, "fall_min_track_len", 30))
        min_conf = float(getattr(self.args, "fall_min_conf", 0.2))
        threshold = float(getattr(self.args, "fall_threshold", 0.5))
        limit = int(getattr(self.args, "fall_limit", 0))
        items = _video_items_from_args(self.args, limit=limit)

        pose_model = fall_yolo_model(model, getattr(self.args, "model", "yolo26n-posegfall.pt"), task="poseg")
        torch_device = fall_device(device)
        posegfall_head = getattr(pose_model.model, "posegfall_head", None)
        if posegfall_head is None:
            raise ValueError("posegfall val requires complete PoseSegFall weights containing `posegfall_head`")
        posegfall_head = posegfall_head.float().to(torch_device).eval()
        config = getattr(pose_model.model, "posegfall_config", {})
        expected_dim = int(config.get("input_dim", feature_dim))
        self.fall_target_fps = float(config.get("target_fps", getattr(self.args, "fall_target_fps", 30.0)))
        self.fall_max_gap_seconds = float(config.get("max_gap_seconds", getattr(self.args, "fall_max_gap_seconds", 0.5)))
        settings = posegfall_feature_settings(self.args, expected_dim)
        feature_key = resolve_feature_cache_key(
            self.args, pose_model, "posegfall", "posegfall_feature_key", settings
        )
        self.args.posegfall_feature_key = feature_key
        feature_dir = Path(getattr(self.args, "posegfall_feature_dir", None) or "runs/posegfall/features") / feature_key
        end2end = bool(getattr(self.args, "fall_val_end2end", True))
        if end2end:
            LOGGER.info("Running PoseSegFall end-to-end validation from decoded videos")
        else:
            LOGGER.info(f"Using shared PoseSegFall feature cache: {feature_dir}")

        rows = []
        processed_frames = 0
        processing_seconds = 0.0
        start = time.time()
        self.run_callbacks("on_val_start")
        description = "PoseSegFall end-to-end val" if end2end else "PoseSegFall cached val"
        for i, item in TQDM(enumerate(items, 1), total=len(items), desc=description, unit="video"):
            self.run_callbacks("on_val_batch_start")
            if end2end:
                if torch_device.type == "cuda":
                    torch.cuda.synchronize(torch_device)
                video_start = time.perf_counter()
                prob, reason, frames = self._predict_video_end2end(
                    pose_model,
                    posegfall_head,
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
                    pose_model,
                    posegfall_head,
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
                f"posegfall val {i}/{len(items)} label={item.label} pred={pred} "
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
        self._save_metrics(rows)
        if bool(getattr(self.args, "plots", True)):
            self._save_competition_plots(rows)
        LOGGER.info(
            "posegfall val: "
            f"accuracy={self.metrics['accuracy']:.4f} precision={self.metrics['precision']:.4f} "
            f"recall={self.metrics['recall']:.4f} f1={self.metrics['f1']:.4f} "
            f"p90={self.metrics['p90']:.4f} p95={self.metrics['p95']:.4f} competition_map={self.metrics['competition_map']:.2f} "
            f"tp={self.metrics['tp']} fp={self.metrics['fp']} tn={self.metrics['tn']} fn={self.metrics['fn']} "
            f"coverage={self.metrics['coverage']:.4f} ({self.metrics['valid_videos']}/{self.metrics['videos']})"
        )
        if end2end:
            LOGGER.info(
                f"posegfall end-to-end speed: frames={processed_frames} time={processing_seconds:.3f}s "
                f"ms/frame={self.metrics['ms_per_frame']:.3f} FPS={self.metrics['processing_fps']:.2f}"
            )
        self.run_callbacks("on_val_end")
        return self.metrics

    def _predict_video_end2end(
        self,
        pose_model,
        posegfall_head,
        video: Path,
        device,
        feature_dim: int,
        min_track_len: int,
        min_conf: float,
    ):
        source_video = opencv_safe_video(video, "posegfall")
        tracks: dict[int, list[torch.Tensor]] = {}
        track_frame_indices: dict[int, list[int]] = {}
        frames_seen = poseg_frames = 0
        frame_step = max(int(getattr(self.args, "vid_stride", 1) or 1), 1)
        for processed_frame_idx, result in enumerate(
            pose_model.track(
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
            if result.keypoints is None or result.masks is None or len(result.keypoints) == 0:
                continue
            poseg_frames += 1
            feats = poseg_result_features(result, device="cpu")
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
                confidence = mean_poseg_confidence(sequence)
                if len(sequence) >= min_track_len and confidence >= min_conf:
                    candidates.append((len(sequence), confidence, sequence))
        if not candidates:
            reason = "decode_failed_or_empty_video" if not frames_seen else f"no_valid_track poseg_frames={poseg_frames}"
            return None, reason, frames_seen
        candidates.sort(key=lambda value: (value[0], value[1]), reverse=True)
        sequence = candidates[0][2]
        with torch.no_grad():
            probability = float(posegfall_head(sequence.to(device)).sigmoid().item())
        window, stride = int(posegfall_head.window), int(posegfall_head.stride)
        clips = 1 if len(sequence) <= window else ((len(sequence) - window) // stride + 1 + int((len(sequence) - window) % stride != 0))
        return probability, f"track_len={len(sequence)} clips={clips} candidates={len(candidates)}", frames_seen

    def _predict_video_cached(
        self,
        pose_model,
        posegfall_head,
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
            pose_model,
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
            feature_dim,
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
            prob = float(posegfall_head(seq.to(device)).sigmoid().item())
        window = int(getattr(posegfall_head, "window", getattr(self.args, "fall_window", 60)))
        stride = int(getattr(posegfall_head, "stride", getattr(self.args, "fall_stride", 15)))
        clips = 1 if seq.shape[0] <= window else ((seq.shape[0] - window) // stride + 1 + int((seq.shape[0] - window) % stride != 0))
        return prob, f"cache={'hit' if cache_hit else 'created'} track_len={seq.shape[0]} clips={clips}"

    @staticmethod
    def _compute_metrics(rows: list[dict], threshold: float, elapsed: float) -> dict:
        return video_classification_metrics(rows, threshold, elapsed, PoseSegFallValidator._competition_metrics)

    @staticmethod
    def _competition_metrics(targets: list[int], probs: list[float]) -> dict[str, float]:
        points = PoseSegFallValidator._curve_points(targets, probs)
        p90 = PoseSegFallValidator._operating_point(points, 0.90)
        p95 = PoseSegFallValidator._operating_point(points, 0.95)
        return {
            "p90": p90["precision"],
            "p95": p95["precision"],
            "p90_threshold": p90["threshold"],
            "p95_threshold": p95["threshold"],
            "p90_recall": p90["recall"],
            "p95_recall": p95["recall"],
            "competition_map": (p90["precision"] + p95["precision"]) * 50.0,
        }

    @staticmethod
    def _curve_points(targets: list[int], probs: list[float]) -> list[dict]:
        """Return confusion metrics at every unique score threshold, treating tied scores as one group."""
        positives = sum(targets)
        negatives = len(targets) - positives
        points = []
        tp = fp = 0
        pairs = sorted(zip(probs, targets), key=lambda item: item[0], reverse=True)
        i = 0
        while i < len(pairs):
            threshold = float(pairs[i][0])
            while i < len(pairs) and float(pairs[i][0]) == threshold:
                if pairs[i][1] == 1:
                    tp += 1
                else:
                    fp += 1
                i += 1
            precision = tp / max(tp + fp, 1)
            recall = tp / positives if positives else 0.0
            points.append(
                {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                    "tp": tp,
                    "fp": fp,
                    "tn": negatives - fp,
                    "fn": positives - tp,
                }
            )
        return points

    @staticmethod
    def _operating_point(points: list[dict], min_recall: float) -> dict:
        eligible = [point for point in points if point["recall"] >= min_recall]
        if not eligible:
            return {"threshold": 1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
        return max(eligible, key=lambda point: (point["precision"], point["threshold"]))

    @staticmethod
    def _metrics_at_threshold(targets: list[int], probs: list[float], threshold: float) -> dict:
        tp = sum(target == 1 and prob >= threshold for target, prob in zip(targets, probs))
        fp = sum(target == 0 and prob >= threshold for target, prob in zip(targets, probs))
        tn = sum(target == 0 and prob < threshold for target, prob in zip(targets, probs))
        fn = sum(target == 1 and prob < threshold for target, prob in zip(targets, probs))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    def _save_dir(self) -> Path:
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _save_rows(self, rows: list[dict]) -> None:
        out = self._save_dir() / "predictions.csv"
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "pred", "prob", "reason"])
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info(f"posegfall val predictions saved to {out}")

    def _save_metrics(self, rows: list[dict]) -> None:
        rows = valid_prediction_rows(rows)
        targets = [int(row["label"]) for row in rows]
        probs = [float(row["prob"]) for row in rows]
        points = self._curve_points(targets, probs)
        p90 = self._operating_point(points, 0.90)
        p95 = self._operating_point(points, 0.95)
        deployment = self._metrics_at_threshold(targets, probs, float(self.metrics["threshold"]))
        payload = {
            **self.metrics,
            "deployment_operating_point": deployment,
            "p90_operating_point": p90,
            "p95_operating_point": p95,
        }
        out = self._save_dir() / "metrics.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        curve_out = self._save_dir() / "competition_curve.csv"
        with curve_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn"])
            writer.writeheader()
            writer.writerows(points)
        LOGGER.info(f"posegfall val metrics saved to {out} and {curve_out}")

    def _save_competition_plots(self, rows: list[dict]) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_dir = self._save_dir()
        rows = valid_prediction_rows(rows)
        targets = [int(row["label"]) for row in rows]
        probs = [float(row["prob"]) for row in rows]
        points = self._curve_points(targets, probs)
        if not points:
            LOGGER.warning("posegfall val plots skipped: no predictions")
            return
        p90 = self._operating_point(points, 0.90)
        p95 = self._operating_point(points, 0.95)
        deployment = self._metrics_at_threshold(targets, probs, float(self.metrics["threshold"]))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            [0.0] + [p["recall"] for p in points],
            [1.0] + [p["precision"] for p in points],
            linewidth=2,
            label="PR curve",
        )
        for point, label, color in ((p90, "P90", "#f59e0b"), (p95, "P95", "#dc2626")):
            ax.scatter(
                point["recall"],
                point["precision"],
                s=70,
                color=color,
                zorder=3,
                label=f"{label}={point['precision']:.3f} @ t={point['threshold']:.3f}",
            )
        ax.scatter(
            deployment["recall"],
            deployment["precision"],
            marker="x",
            s=80,
            color="#2563eb",
            zorder=3,
            label=f"Deployment t={deployment['threshold']:.3f}",
        )
        ax.set(
            xlabel="Recall",
            ylabel="Precision",
            xlim=(0, 1.01),
            ylim=(0, 1.01),
            title=f"PoseSegFall PR Curve | Competition Score={self.metrics['competition_map']:.2f}",
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(save_dir / "pr_curve.png", dpi=200)
        plt.close(fig)

        thresholds = [i / 200 for i in range(201)]
        sweep = [self._metrics_at_threshold(targets, probs, threshold) for threshold in thresholds]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(thresholds, [p["precision"] for p in sweep], label="Precision", linewidth=2)
        ax.plot(thresholds, [p["recall"] for p in sweep], label="Recall", linewidth=2)
        ax.plot(thresholds, [p["f1"] for p in sweep], label="F1", linewidth=2)
        ax.axvline(
            deployment["threshold"],
            color="#2563eb",
            linestyle="--",
            label=f"Deployment {deployment['threshold']:.3f}",
        )
        ax.axvline(p90["threshold"], color="#f59e0b", linestyle=":", label=f"P90 threshold {p90['threshold']:.3f}")
        ax.axvline(p95["threshold"], color="#dc2626", linestyle=":", label=f"P95 threshold {p95['threshold']:.3f}")
        ax.set(
            xlabel="Video Probability Threshold",
            ylabel="Metric",
            xlim=(0, 1),
            ylim=(0, 1.01),
            title="PoseSegFall Metrics vs Threshold",
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(save_dir / "threshold_curves.png", dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        matrices = (
            (axes[0], deployment, f"Deployment t={deployment['threshold']:.3f}"),
            (axes[1], p95, f"Recall>=95% t={p95['threshold']:.3f}"),
        )
        for ax, point, title in matrices:
            matrix = [[point["tn"], point["fp"]], [point["fn"], point["tp"]]]
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(max(row) for row in matrix) or 1)
            for y, row in enumerate(matrix):
                row_total = sum(row)
                for x, value in enumerate(row):
                    percent = value / row_total * 100 if row_total else 0.0
                    ax.text(x, y, f"{value}\n{percent:.1f}%", ha="center", va="center", color="black")
            ax.set_xticks([0, 1], ["No Fall", "Fall"])
            ax.set_yticks([0, 1], ["No Fall", "Fall"])
            ax.set(xlabel="Predicted", ylabel="Actual", title=title)
        fig.suptitle("PoseSegFall Video-Level Confusion Matrices")
        fig.tight_layout()
        fig.savefig(save_dir / "confusion_matrices.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 6))
        fall_probs = [prob for target, prob in zip(targets, probs) if target == 1]
        nofall_probs = [prob for target, prob in zip(targets, probs) if target == 0]
        bins = [i / 25 for i in range(26)]
        if nofall_probs:
            ax.hist(nofall_probs, bins=bins, alpha=0.6, label=f"No Fall (n={len(nofall_probs)})", color="#2563eb")
        if fall_probs:
            ax.hist(fall_probs, bins=bins, alpha=0.6, label=f"Fall (n={len(fall_probs)})", color="#dc2626")
        ax.axvline(
            deployment["threshold"],
            color="#111827",
            linestyle="--",
            label=f"Deployment {deployment['threshold']:.3f}",
        )
        ax.axvline(p95["threshold"], color="#f59e0b", linestyle=":", label=f"P95 threshold {p95['threshold']:.3f}")
        ax.set(
            xlabel="Video Fall Probability",
            ylabel="Video Count",
            xlim=(0, 1),
            title="PoseSegFall Probability Distribution",
        )
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(save_dir / "probability_distribution.png", dpi=200)
        plt.close(fig)
        LOGGER.info(f"posegfall val plots saved to {save_dir}")
