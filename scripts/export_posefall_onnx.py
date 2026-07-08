#!/usr/bin/env python3
# Ultralytics posefall ONNX export helper.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo.posefall.model import POSEFALL_FEATURE_DIM, load_posefall_head


class PoseFallHeadExport(nn.Module):
    """Export wrapper that returns fall probability for fixed-window posefall features."""

    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export posefall pipeline models to ONNX.")
    parser.add_argument("--pose-model", default="yolo26n-pose.pt", help="YOLO pose .pt used for keypoint extraction.")
    parser.add_argument("--posefall-weights", default="posefall26n-pose.pt", help="Posefall temporal head checkpoint.")
    parser.add_argument("--out-dir", default="exports/posefall", help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO export image size.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--device", default="cpu", help="Device for export, e.g. cpu or 0.")
    parser.add_argument("--half", action="store_true", help="Export YOLO model with FP16 when supported.")
    parser.add_argument("--dynamic", action="store_true", help="Export YOLO model with dynamic image shapes.")
    parser.add_argument("--skip-yolo", action="store_true", help="Only export posefall head.")
    return parser.parse_args()


def export_yolo_pose(args: argparse.Namespace, out_dir: Path) -> Path | None:
    if args.skip_yolo:
        return None

    model = YOLO(args.pose_model)
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=args.dynamic,
        half=args.half,
        device=args.device,
    )
    exported = Path(exported)
    target = out_dir / "pose_yolo.onnx"
    if exported.resolve() != target.resolve():
        target.write_bytes(exported.read_bytes())
    return target


def export_posefall_head(args: argparse.Namespace, out_dir: Path) -> tuple[Path, dict]:
    device = torch.device("cpu")
    head, config = load_posefall_head(args.posefall_weights, device, input_dim=POSEFALL_FEATURE_DIM)
    if head is None:
        raise ValueError(f"Unable to load posefall head from {args.posefall_weights!r}")

    head.eval()
    input_dim = int(config.get("input_dim", POSEFALL_FEATURE_DIM))
    window = int(config.get("window", getattr(head, "window", 60)))
    dummy = torch.zeros(1, window, input_dim, dtype=torch.float32)

    out_path = out_dir / "posefall_head.onnx"
    torch.onnx.export(
        PoseFallHeadExport(head),
        dummy,
        out_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["fall_prob"],
        dynamic_axes={"features": {0: "batch"}, "fall_prob": {0: "batch"}},
    )
    return out_path, config


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pose_onnx = export_yolo_pose(args, out_dir)
    head_onnx, config = export_posefall_head(args, out_dir)

    meta = {
        "pipeline": "posefall",
        "pose_model": str(args.pose_model),
        "pose_onnx": str(pose_onnx) if pose_onnx else None,
        "posefall_weights": str(args.posefall_weights),
        "posefall_head_onnx": str(head_onnx),
        "feature_dim": int(config.get("input_dim", POSEFALL_FEATURE_DIM)),
        "window": int(config.get("window", 60)),
        "stride": int(config.get("stride", 15)),
        "output": "fall_prob sigmoid probability, threshold normally 0.5 unless tuned on validation set",
        "notes": [
            "This exports the two neural-network parts only.",
            "Video decode, YOLO postprocess, tracking, posefall feature normalization, missing-frame fill, sliding windows, and video-level aggregation remain in deployment code.",
        ],
    }
    meta_path = out_dir / "posefall_export_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    print(f"pose ONNX: {pose_onnx}")
    print(f"posefall head ONNX: {head_onnx}")
    print(f"metadata: {meta_path}")


if __name__ == "__main__":
    main()
