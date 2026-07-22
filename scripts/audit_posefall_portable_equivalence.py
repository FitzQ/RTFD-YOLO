#!/usr/bin/env python3
"""Recheck the source Transformer and its PicoVision-compatible rewrite."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("posefall_picovision_export", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(module, payload, device: torch.device) -> dict[str, object]:
    model = module.PicoPoseFallTransformer(module.PortablePicoNN, payload["pos_embed"])
    module.load_original_weights(model, payload["state_dict"])
    model.eval().to(device)
    expected = payload["reference_probabilities"].cpu()
    actual = []
    with torch.no_grad():
        for sample in payload["reference_features"].to(device):
            actual.append(model(sample.unsqueeze(0)).cpu())
    actual = torch.cat(actual)
    difference = (actual - expected).abs()
    return {
        "device": str(device),
        "samples": int(actual.numel()),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "class_agreement": float(
            ((actual >= 0.5) == (expected >= 0.5)).float().mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-script", type=Path, required=True)
    parser.add_argument("--intermediate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    module = load_module(args.export_script)
    payload = torch.load(args.intermediate, map_location="cpu", weights_only=False)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    report = {
        "schema_version": 1,
        "purpose": "Numerical audit of the source Transformer and PicoVision-compatible rewrite before ATC quantization.",
        "intermediate": str(args.intermediate),
        "input_shape": [1, 60, 56],
        "threshold": 0.5,
        "results": [compare(module, payload, device) for device in devices],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
