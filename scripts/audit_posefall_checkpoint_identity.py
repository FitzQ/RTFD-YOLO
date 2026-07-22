#!/usr/bin/env python3
"""Audit which tensors changed between two PoseFall checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


HEAD_PREFIX = "posefall_head."


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint["model"]
    return {
        key: value.detach().cpu().float().contiguous()
        for key, value in model.state_dict().items()
    }


def tensor_digest(state: dict[str, torch.Tensor], *, head: bool) -> dict[str, object]:
    digest = hashlib.sha256()
    tensors = parameters = 0
    for key in sorted(state):
        if key.startswith(HEAD_PREFIX) != head:
            continue
        value = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
        tensors += 1
        parameters += value.numel()
    return {"sha256": digest.hexdigest(), "tensors": tensors, "parameters": parameters}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    states = {"old": load_state(args.old), "new": load_state(args.new)}
    keys = sorted(set(states["old"]) | set(states["new"]))
    changed: list[str] = []
    max_abs = {"pose_frontend": 0.0, "transformer_head": 0.0}
    for key in keys:
        old = states["old"].get(key)
        new = states["new"].get(key)
        section = "transformer_head" if key.startswith(HEAD_PREFIX) else "pose_frontend"
        if old is None or new is None or old.shape != new.shape:
            changed.append(key)
            max_abs[section] = None
            continue
        difference = float((old - new).abs().max()) if old.numel() else 0.0
        if max_abs[section] is not None:
            max_abs[section] = max(max_abs[section], difference)
        if not torch.equal(old, new):
            changed.append(key)

    report = {
        "schema_version": 1,
        "normalization": "ema if present else model; tensors converted to contiguous float32",
        "digest_algorithm": "SHA-256 over sorted key, dtype, shape, and raw tensor bytes",
        "checkpoints": {
            "old": {"path": str(args.old), "sha256": file_sha256(args.old)},
            "new": {"path": str(args.new), "sha256": file_sha256(args.new)},
        },
        "pose_frontend": {
            "old": tensor_digest(states["old"], head=False),
            "new": tensor_digest(states["new"], head=False),
            "max_abs_difference": max_abs["pose_frontend"],
            "bitwise_identical": max_abs["pose_frontend"] == 0.0,
        },
        "transformer_head": {
            "old": tensor_digest(states["old"], head=True),
            "new": tensor_digest(states["new"], head=True),
            "max_abs_difference": max_abs["transformer_head"],
            "bitwise_identical": max_abs["transformer_head"] == 0.0,
        },
        "changed_tensor_count": len(changed),
        "changed_pose_frontend_tensor_count": sum(
            not key.startswith(HEAD_PREFIX) for key in changed
        ),
        "changed_transformer_head_tensor_count": sum(
            key.startswith(HEAD_PREFIX) for key in changed
        ),
        "changed_tensor_names": changed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
