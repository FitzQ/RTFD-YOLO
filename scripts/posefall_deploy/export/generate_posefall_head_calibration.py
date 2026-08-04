#!/usr/bin/env python3
"""Build representative 60x56 PoseFall calibration windows from feature caches."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch


WINDOW = 60
FEATURE_DIM = 56


def window_from_sequence(sequence: torch.Tensor, rng: random.Random) -> np.ndarray:
    sequence = sequence.detach().cpu().float()
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIM or sequence.shape[0] == 0:
        raise ValueError(f"unexpected sequence shape: {tuple(sequence.shape)}")
    length = sequence.shape[0]
    if length >= WINDOW:
        start = rng.randrange(length - WINDOW + 1)
        sequence = sequence[start : start + WINDOW]
    else:
        # Match PoseFallDataset.pad_or_trim(): prepend the first observation.
        padding = sequence[:1].repeat(WINDOW - length, 1)
        sequence = torch.cat((padding, sequence), dim=0)
    return sequence.numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sample-bin", type=Path)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--seed", type=int, default=610)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    selected: list[tuple[Path, int]] = []
    for class_name, label in (("No_Fall", 0), ("Fall", 1)):
        files = sorted((args.cache_root / class_name).glob("*.pt"))
        if len(files) < args.samples_per_class:
            raise RuntimeError(f"only {len(files)} files in {class_name}")
        selected.extend((path, label) for path in rng.sample(files, args.samples_per_class))
    rng.shuffle(selected)

    windows, labels, sources = [], [], []
    for path, label in selected:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        windows.append(window_from_sequence(payload["sequence"], rng))
        labels.append(label)
        sources.append(str(path))
    data = np.stack(windows).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii", newline="\n") as stream:
        for window in data:
            stream.write(" ".join(format(float(value), ".9g") for value in window.ravel()))
            stream.write("\n")
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.reference, features=data, labels=np.asarray(labels), sources=sources)
    if args.sample_bin is not None:
        args.sample_bin.parent.mkdir(parents=True, exist_ok=True)
        data[0].tofile(args.sample_bin)
    manifest = args.reference.with_suffix(".json")
    manifest.write_text(json.dumps({
        "shape": list(data.shape), "labels": labels, "sources": sources,
        "min": float(data.min()), "max": float(data.max()),
        "mean": float(data.mean()), "std": float(data.std()),
    }, indent=2), encoding="utf-8")
    print(f"wrote {data.shape} to {args.output}")
    print(f"range={data.min():.6g}..{data.max():.6g}, mean={data.mean():.6g}, std={data.std():.6g}")


if __name__ == "__main__":
    main()
