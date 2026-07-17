#!/usr/bin/env python3
# Ultralytics posefall ONNX export helper.

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import MethodType

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo.posefall.model import POSEFALL_FEATURE_DIM
from ultralytics.nn.tasks import load_checkpoint


class PoseFallHeadExport(nn.Module):
    """Export wrapper that returns fall probability for fixed-window posefall features."""

    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x))


def svp_pose_raw_postprocess(head: nn.Module, predictions: torch.Tensor) -> torch.Tensor:
    """Return one-to-one pose candidates without TopK, formatted for standard CPU NMS."""
    boxes, scores, keypoints = predictions.split([4, head.nc, head.nk], dim=-1)
    center = (boxes[..., :2] + boxes[..., 2:]) / 2
    size = boxes[..., 2:] - boxes[..., :2]
    return torch.cat((center, size, scores, keypoints), dim=-1).permute(0, 2, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export posefall pipeline models to ONNX.")
    parser.add_argument("--weights", default="posefall26n.pt", help="Complete PoseFall checkpoint.")
    parser.add_argument("--out-dir", default="exports/posefall", help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO export image size.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--device", default="cpu", help="Device for export, e.g. cpu or 0.")
    parser.add_argument("--half", action="store_true", help="Export YOLO model with FP16 when supported.")
    parser.add_argument("--dynamic", action="store_true", help="Export YOLO model with dynamic image shapes.")
    parser.add_argument("--skip-yolo", action="store_true", help="Only export posefall head.")
    parser.add_argument(
        "--svp",
        action="store_true",
        help="Export static Hi3516CV610-compatible ONNX models and generate ATC conversion assets.",
    )
    return parser.parse_args()


def export_yolo_pose(args: argparse.Namespace, out_dir: Path) -> Path | None:
    if args.skip_yolo:
        return None

    model = YOLO(args.weights, task="pose")
    model.task = "pose"
    model.model.task = "pose"
    model.overrides["task"] = "pose"
    if isinstance(model.model.args, dict):
        model.model.args = {**model.model.args, "task": "pose", "data": None}
    if args.svp:
        pose_head = model.model.model[-1]
        pose_head.postprocess = MethodType(svp_pose_raw_postprocess, pose_head)
    export_args = dict(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=args.dynamic,
        half=args.half,
        device=args.device,
    )
    exported = model.export(**export_args)
    exported = Path(exported)
    target = out_dir / "pose_yolo.onnx"
    if exported.resolve() != target.resolve():
        shutil.move(str(exported), target)
    if args.svp:
        import onnx

        onnx_model = onnx.load(target)
        metadata = {item.key: item.value for item in onnx_model.metadata_props}
        metadata["end2end"] = "False"
        del onnx_model.metadata_props[:]
        for key, value in metadata.items():
            item = onnx_model.metadata_props.add()
            item.key, item.value = key, value
        onnx.save(onnx_model, target)
    return target


def export_posefall_head(args: argparse.Namespace, out_dir: Path) -> tuple[Path, dict]:
    device = torch.device("cpu")
    model, _ = load_checkpoint(args.weights, device=device, fuse=False)
    head = getattr(model, "posefall_head", None)
    if head is None:
        raise ValueError(f"Complete PoseFall weights {args.weights!r} do not contain `posefall_head`")
    config = dict(getattr(model, "posefall_config", {}))
    model_args = getattr(model, "args", {})

    def model_arg(name: str, default):
        return model_args.get(name, default) if isinstance(model_args, dict) else getattr(model_args, name, default)

    config.setdefault("threshold", float(model_arg("fall_threshold", 0.5)))
    config.setdefault("min_conf", float(model_arg("fall_min_conf", 0.2)))
    config.setdefault("target_fps", float(model_arg("fall_target_fps", 30.0)))
    config.setdefault("max_gap_seconds", float(model_arg("fall_max_gap_seconds", 0.5)))

    head.eval()
    input_dim = int(config.get("input_dim", POSEFALL_FEATURE_DIM))
    window = int(config.get("window", getattr(head, "window", 60)))
    dummy = torch.zeros(1, window, input_dim, dtype=torch.float32)

    out_path = out_dir / "posefall_head.onnx"
    fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        dynamic_axes = None if args.svp else {"features": {0: "batch"}, "fall_prob": {0: "batch"}}
        torch.onnx.export(
            PoseFallHeadExport(head),
            dummy,
            out_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["features"],
            output_names=["fall_prob"],
            dynamic_axes=dynamic_axes,
        )
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
    return out_path, config


def export_svp_assets(args: argparse.Namespace, out_dir: Path, config: dict) -> Path:
    """Generate Hi3516CV610 ATC conversion inputs and documentation."""
    svp_dir = out_dir / "svp"
    calibration_dir = svp_dir / "calibration"
    model_dir = svp_dir / "model"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    pose_values = " ".join("0" for _ in range(3 * args.imgsz * args.imgsz))
    (calibration_dir / "pose_input.txt").write_text(pose_values + "\n")

    window = int(config.get("window", 60))
    input_dim = int(config.get("input_dim", POSEFALL_FEATURE_DIM))
    head_values = " ".join("0" for _ in range(window * input_dim))
    (calibration_dir / "head_input.txt").write_text(head_values + "\n")

    convert_path = svp_dir / "convert_om.sh"
    convert_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -eo pipefail\n\n"
        'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'PACKAGE_ROOT="$(cd "${ROOT}/.." && pwd)"\n'
        'CANN_SETENV="${CANN_SETENV:-/usr/local/Ascend/ascend-toolkit/svp_latest/x86_64-linux/script/setenv.sh}"\n'
        'if [[ ! -f "${CANN_SETENV}" ]]; then\n'
        '  echo "CANN setenv.sh not found: ${CANN_SETENV}" >&2\n'
        '  echo "Set CANN_SETENV to the installed SVP CANN setenv.sh path." >&2\n'
        "  exit 1\n"
        "fi\n"
        "set +u\n"
        'source "${CANN_SETENV}"\n'
        "set -u\n"
        'cd "${ROOT}"\n\n'
        "atc \\\n"
        "  --framework=5 \\\n"
        '  --model="${PACKAGE_ROOT}/pose_yolo.onnx" \\\n'
        '  --output="${ROOT}/model/pose_yolo" \\\n'
        f'  --input_shape="images:1,3,{args.imgsz},{args.imgsz}" \\\n'
        '  --input_type="images:FP32" \\\n'
        '  --image_list="images:${ROOT}/calibration/pose_input.txt" \\\n'
        "  --soc_version=Hi3516CV610 \\\n"
        "  --save_original_model=true\n\n"
        "atc \\\n"
        "  --framework=5 \\\n"
        '  --model="${PACKAGE_ROOT}/posefall_head.onnx" \\\n'
        '  --output="${ROOT}/model/posefall_head" \\\n'
        f'  --input_shape="features:1,{window},{input_dim}" \\\n'
        '  --input_type="features:FP32" \\\n'
        '  --image_list="features:${ROOT}/calibration/head_input.txt" \\\n'
        "  --soc_version=Hi3516CV610 \\\n"
        "  --save_original_model=true\n\n"
        'echo "SVP OM models saved under ${ROOT}/model"\n'
    )
    convert_path.chmod(0o755)

    model_io = {
        "soc_version": "Hi3516CV610",
        "pose": {
            "onnx": "../pose_yolo.onnx",
            "om": "model/pose_yolo_original.om",
            "input": {"name": "images", "dtype": "float32", "shape": [1, 3, args.imgsz, args.imgsz]},
            "output": {
                "layout": "decoded_pose_candidates",
                "shape": [1, POSEFALL_FEATURE_DIM, sum((args.imgsz // stride) ** 2 for stride in (8, 16, 32))],
            },
            "postprocess": "Run confidence filtering/NMS and tracking on the ARM CPU.",
        },
        "posefall_head": {
            "onnx": "../posefall_head.onnx",
            "om": "model/posefall_head_original.om",
            "input": {"name": "features", "dtype": "float32", "shape": [1, window, input_dim]},
            "output": {"name": "fall_prob", "dtype": "float32", "shape": [1]},
        },
    }
    (svp_dir / "model_io.json").write_text(json.dumps(model_io, ensure_ascii=False, indent=2) + "\n")
    (svp_dir / "README.md").write_text(
        "# PoseFall on Hi3516CV610 SVP NPU\n\n"
        "This directory converts the two static ONNX networks to SVP offline `.om` models.\n\n"
        "## Convert\n\n"
        "```bash\n"
        "CANN_SETENV=/path/to/ascend-toolkit/svp_latest/x86_64-linux/script/setenv.sh ./convert_om.sh\n"
        "```\n\n"
        "The bundled zero-valued calibration inputs are only conversion smoke-test data. Replace them with "
        "representative normalized pose images and real 60x56 PoseFall windows before final accuracy validation.\n\n"
        "## Board pipeline\n\n"
        "1. Decode camera/video frames and resize/letterbox to the exported image size.\n"
        "2. Run `pose_yolo_original.om` through SVP ACL.\n"
        "3. Perform confidence filtering, NMS, keypoint parsing, and person tracking on the ARM CPU.\n"
        "4. Build the exact 56-value feature vector and maintain the fixed temporal window.\n"
        "5. Run `posefall_head_original.om` through SVP ACL and apply the configured threshold.\n\n"
        "`posefall_onnx_runtime.py` is the PC reference implementation. It is not the board SVP ACL application.\n"
    )
    return svp_dir


def main() -> None:
    args = parse_args()
    if args.svp:
        args.dynamic = False
        args.half = False
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pose_onnx = export_yolo_pose(args, out_dir)
    head_onnx, config = export_posefall_head(args, out_dir)
    runtime_path = out_dir / "posefall_onnx_runtime.py"
    tracker_path = out_dir / "fall_botsort.yaml"
    requirements_path = out_dir / "requirements.txt"
    shutil.copy2(ROOT / "scripts" / "posefall_onnx_runtime.py", runtime_path)
    shutil.copy2(ROOT / "ultralytics" / "cfg" / "trackers" / "fall_botsort.yaml", tracker_path)
    requirements_path.write_text(
        "ultralytics>=8.4.64\n"
        "onnxruntime>=1.17.0\n"
        "opencv-python>=4.8.0\n"
        "numpy>=1.23.0\n"
    )
    svp_dir = export_svp_assets(args, out_dir, config) if args.svp else None

    meta = {
        "pipeline": "posefall",
        "target": "Hi3516CV610_SVP" if args.svp else "onnxruntime",
        "weights": str(args.weights),
        "pose_onnx": pose_onnx.name if pose_onnx else None,
        "posefall_head_onnx": head_onnx.name,
        "feature_dim": int(config.get("input_dim", POSEFALL_FEATURE_DIM)),
        "window": int(config.get("window", 60)),
        "stride": int(config.get("stride", 15)),
        "threshold": float(config.get("threshold", 0.5)),
        "min_conf": float(config.get("min_conf", 0.2)),
        "target_fps": float(config.get("target_fps", 30.0)),
        "max_gap_seconds": float(config.get("max_gap_seconds", 0.5)),
        "imgsz": args.imgsz,
        "pose_end2end": not args.svp,
        "output": "fall_prob sigmoid probability, threshold normally 0.5 unless tuned on validation set",
        "notes": [
            "posefall_onnx_runtime.py connects video decode, pose tracking, feature tensorization, missing-frame fill, and temporal-head inference.",
            "The runtime reports a full video as fall when any evaluated window reaches the configured threshold.",
            "ffmpeg is required only for legacy AVI input.",
        ],
    }
    meta_path = out_dir / "posefall_export_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    print(f"pose ONNX: {pose_onnx}")
    print(f"posefall head ONNX: {head_onnx}")
    print(f"metadata: {meta_path}")
    print(f"runtime: {runtime_path}")
    if svp_dir:
        print(f"SVP assets: {svp_dir}")


if __name__ == "__main__":
    main()
