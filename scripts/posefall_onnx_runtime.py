#!/usr/bin/env python3
"""Run an exported PoseFall ONNX package on a video, stream, or camera."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

FEATURE_DIM = 56
BOX_DIM = 5
KEYPOINT_COUNT = 17


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="PoseFall ONNX video inference runtime.")
    parser.add_argument("--source", required=True, help="Video/image path, URL, stream, or camera index such as 0.")
    parser.add_argument("--pose", default=str(base / "pose_yolo.onnx"), help="Exported YOLO pose ONNX model.")
    parser.add_argument("--head", default=str(base / "posefall_head.onnx"), help="Exported PoseFall head ONNX model.")
    parser.add_argument(
        "--meta", default=str(base / "posefall_export_meta.json"), help="PoseFall export metadata JSON."
    )
    parser.add_argument("--tracker", default=str(base / "fall_botsort.yaml"), help="Ultralytics tracker YAML.")
    parser.add_argument("--device", default="cpu", help="Ultralytics pose device, e.g. cpu, 0, or 0,1.")
    parser.add_argument(
        "--head-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="ONNX Runtime provider for the temporal head.",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Pose inference image size; defaults to export metadata.")
    parser.add_argument("--conf", type=float, default=0.1, help="Pose detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="Pose NMS IoU threshold.")
    parser.add_argument("--vid-stride", type=int, default=1, help="Process every Nth source frame.")
    parser.add_argument("--source-fps", type=float, default=0.0, help="Override source FPS; 0 reads it from the source.")
    parser.add_argument("--threshold", type=float, default=None, help="Fall threshold; defaults to export metadata.")
    parser.add_argument("--min-conf", type=float, default=None, help="Minimum mean keypoint confidence.")
    parser.add_argument("--target-fps", type=float, default=None, help="Temporal-head target FPS.")
    parser.add_argument("--max-gap-seconds", type=float, default=None, help="Reset a track after this observation gap.")
    parser.add_argument("--output", default="posefall_result.mp4", help="Annotated output video path.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True, help="Save annotated output.")
    parser.add_argument("--show", action="store_true", help="Display annotated frames; press q to stop.")
    parser.add_argument("--verbose", action="store_true", help="Enable Ultralytics per-frame logging.")
    return parser.parse_args()


def load_metadata(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PoseFall metadata not found: {path}")
    return json.loads(path.read_text())


def parse_source(value: str):
    return int(value) if value.isdigit() else value


@contextmanager
def safe_video_source(source):
    """Transcode legacy AVI files because some OpenCV builds crash while decoding them."""
    if not isinstance(source, str):
        yield source
        return
    path = Path(source)
    if not path.is_file() or path.suffix.lower() != ".avi":
        yield source
        return
    with tempfile.TemporaryDirectory(prefix="posefall_onnx_") as directory:
        converted = Path(directory) / f"{path.stem}.mp4"
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
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
            str(converted),
        ]
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as error:
            raise RuntimeError("ffmpeg is required to run PoseFall on legacy AVI videos") from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Unable to transcode AVI input: {path}") from error
        yield str(converted)


def make_head_session(path: str | Path, requested: str) -> ort.InferenceSession:
    available = ort.get_available_providers()
    use_cuda = requested == "cuda" or requested == "auto" and "CUDAExecutionProvider" in available
    if requested == "cuda" and "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable. Install onnxruntime-gpu or use --head-device cpu."
        )
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), providers=providers)


def predictor_fps(model: YOLO, fallback: float = 30.0) -> float:
    fps = getattr(getattr(getattr(model, "predictor", None), "dataset", None), "fps", fallback)
    if isinstance(fps, (list, tuple)):
        fps = fps[0] if fps else fallback
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = fallback
    return fps if math.isfinite(fps) and fps > 0 else fallback


def track_ids(result, count: int) -> list[int]:
    if result.boxes is not None and result.boxes.is_track and result.boxes.id is not None:
        return [int(value) for value in result.boxes.id.cpu().tolist()]
    return list(range(count))


def pose_features(result) -> np.ndarray:
    """Match pose_result_features(): xywhn + box confidence + normalized (x, y, confidence) keypoints."""
    if result.boxes is None or result.keypoints is None or not len(result.boxes):
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
    boxes = result.boxes.xywhn.detach().cpu().numpy().astype(np.float32, copy=False)
    confidence = result.boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)[:, None]
    keypoints = result.keypoints.data.detach().cpu().numpy().astype(np.float32, copy=True)
    height, width = result.orig_shape
    keypoints[..., 0] /= max(float(width), 1.0)
    keypoints[..., 1] /= max(float(height), 1.0)
    keypoints[..., 2] = np.clip(keypoints[..., 2], 0.0, 1.0)
    features = np.concatenate((boxes, confidence, keypoints.reshape(len(keypoints), -1)), axis=1)
    if features.shape[1] != FEATURE_DIM:
        raise ValueError(f"PoseFall feature_dim={features.shape[1]}, expected {FEATURE_DIM}")
    return features


def append_resampled_feature(
    history: deque[np.ndarray],
    feature: np.ndarray,
    tick: int,
    last_tick: int | None,
    max_gap_ticks: int,
) -> int:
    """Match real-time repeat-last filling used by posefall predict."""
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
        history.append(history[-1].copy())
    history.append(feature)
    return tick


def pad_or_trim(sequence: np.ndarray, window: int) -> np.ndarray:
    if len(sequence) >= window:
        return sequence[-window:]
    padding = np.repeat(sequence[:1], window - len(sequence), axis=0)
    return np.concatenate((padding, sequence), axis=0)


def mean_keypoint_confidence(sequence: np.ndarray) -> float:
    keypoints = sequence[..., BOX_DIM:].reshape(-1, KEYPOINT_COUNT, 3)
    return float(keypoints[..., 2].mean())


def draw_results(result, probabilities: list[float | None], threshold: float) -> np.ndarray:
    frame = result.plot(boxes=False, labels=False)
    if result.boxes is None:
        return frame
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()
    ids = track_ids(result, len(boxes))
    line_width = max(round((frame.shape[0] + frame.shape[1]) / 600), 2)
    font_scale = max(line_width / 3, 0.55)
    for index, (box, det_conf, track_id) in enumerate(zip(boxes, confidences, ids)):
        probability = probabilities[index] if index < len(probabilities) else None
        is_fall = probability is not None and probability >= threshold
        color = (0, 0, 255) if is_fall else (0, 170, 0)
        state = "FALL" if is_fall else "fall"
        probability_text = f" {state} {probability:.2f}" if probability is not None else ""
        label = f"id:{track_id} person {float(det_conf):.2f}{probability_text}"
        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width
        )
        text_x = min(max(x1, 0), max(frame.shape[1] - text_width - 4, 0))
        text_y = max(y1, text_height + baseline + 2)
        cv2.rectangle(
            frame,
            (text_x, text_y - text_height - baseline - 4),
            (text_x + text_width + 4, text_y + 2),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (text_x + 2, text_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            line_width,
            cv2.LINE_AA,
        )
    return frame


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.meta)
    window = int(metadata.get("window", 60))
    feature_dim = int(metadata.get("feature_dim", FEATURE_DIM))
    if feature_dim != FEATURE_DIM:
        raise ValueError(f"Runtime supports PoseFall feature_dim={FEATURE_DIM}, export requires {feature_dim}")
    threshold = float(args.threshold if args.threshold is not None else metadata.get("threshold", 0.5))
    min_conf = float(args.min_conf if args.min_conf is not None else metadata.get("min_conf", 0.2))
    target_fps = float(args.target_fps if args.target_fps is not None else metadata.get("target_fps", 30.0))
    max_gap_seconds = float(
        args.max_gap_seconds
        if args.max_gap_seconds is not None
        else metadata.get("max_gap_seconds", 0.5)
    )
    imgsz = int(args.imgsz if args.imgsz is not None else metadata.get("imgsz", 640))

    pose_path, head_path, tracker_path = Path(args.pose), Path(args.head), Path(args.tracker)
    for label, path in (("pose model", pose_path), ("fall head", head_path), ("tracker", tracker_path)):
        if not path.is_file():
            raise FileNotFoundError(f"PoseFall {label} not found: {path}")

    pose_model = YOLO(str(pose_path), task="pose")
    head_session = make_head_session(head_path, args.head_device)
    head_input = head_session.get_inputs()[0]
    head_shape = head_input.shape
    if len(head_shape) != 3 or head_shape[1:] != [window, feature_dim]:
        raise ValueError(
            f"Unexpected PoseFall head input shape {head_shape}; metadata requires [batch, {window}, {feature_dim}]"
        )

    histories: dict[int, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=window))
    last_ticks: dict[int, int] = {}
    processed_frame = 0
    max_probability: float | None = None
    video_fall = False
    writer = None
    output = Path(args.output)
    source = parse_source(args.source)

    print(
        f"PoseFall ONNX runtime: window={window} threshold={threshold:.3f} "
        f"target_fps={target_fps:g} head_provider={head_session.get_providers()[0]}"
    )
    try:
        with safe_video_source(source) as safe_source:
            results = pose_model.track(
                source=safe_source,
                stream=True,
                persist=True,
                task="pose",
                tracker=str(tracker_path),
                device=args.device,
                imgsz=imgsz,
                conf=args.conf,
                iou=args.iou,
                vid_stride=max(args.vid_stride, 1),
                save=False,
                verbose=args.verbose,
            )
            for result in results:
                source_fps = args.source_fps if args.source_fps > 0 else predictor_fps(pose_model)
                source_frame = processed_frame * max(args.vid_stride, 1)
                tick = round(source_frame * target_fps / source_fps)
                processed_frame += 1

                probabilities: list[float | None] = [None] * (
                    len(result.boxes) if result.boxes is not None else 0
                )
                if result.keypoints is not None and len(result.keypoints):
                    ids = track_ids(result, len(result.keypoints))
                    features = pose_features(result)
                    windows, indexes = [], []
                    for detection_index, (track_id, feature) in enumerate(zip(ids, features)):
                        history = histories[track_id]
                        last_ticks[track_id] = append_resampled_feature(
                            history,
                            feature,
                            tick,
                            last_ticks.get(track_id),
                            max(1, round(max_gap_seconds * target_fps)),
                        )
                        sequence = pad_or_trim(np.stack(history), window)
                        if mean_keypoint_confidence(sequence) < min_conf:
                            continue
                        indexes.append(detection_index)
                        windows.append(sequence)
                    if windows:
                        batch = np.stack(windows).astype(np.float32, copy=False)
                        output_values = head_session.run(None, {head_input.name: batch})[0].reshape(-1)
                        for index, probability in zip(indexes, output_values):
                            value = float(probability)
                            probabilities[index] = value
                            max_probability = value if max_probability is None else max(max_probability, value)
                            video_fall = video_fall or value >= threshold

                frame = draw_results(result, probabilities, threshold)
                if args.save:
                    if writer is None:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        height, width = frame.shape[:2]
                        output_fps = max(source_fps / max(args.vid_stride, 1), 1.0)
                        writer = cv2.VideoWriter(
                            str(output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height)
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Unable to open output video writer: {output}")
                    writer.write(frame)
                if args.show:
                    cv2.imshow("PoseFall ONNX", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    probability_text = "None" if max_probability is None else f"{max_probability:.6f}"
    print(
        f"PoseFall result: fall={video_fall} max_probability={probability_text} "
        f"processed_frames={processed_frame}"
    )
    if args.save:
        print(f"Saved annotated video: {output.resolve()}")


if __name__ == "__main__":
    main()
