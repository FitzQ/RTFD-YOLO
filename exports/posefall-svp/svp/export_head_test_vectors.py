#!/usr/bin/env python3
"""Export representative high-confidence fall and normal head inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.reference)
    features = data["features"].astype(np.float32)
    labels = data["labels"]
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    probabilities = np.asarray([
        float(session.run(None, {"features": sample[None]})[0].reshape(-1)[0])
        for sample in features
    ])
    fall_candidates = np.flatnonzero(labels == 1)
    normal_candidates = np.flatnonzero(labels == 0)
    fall_index = int(fall_candidates[np.argmax(probabilities[fall_candidates])])
    normal_index = int(normal_candidates[np.argmin(probabilities[normal_candidates])])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features[fall_index].tofile(args.output_dir / "head_fall.bin")
    features[normal_index].tofile(args.output_dir / "head_normal.bin")
    result = {
        "fall_index": fall_index,
        "fall_probability": float(probabilities[fall_index]),
        "normal_index": normal_index,
        "normal_probability": float(probabilities[normal_index]),
    }
    (args.output_dir / "head_test_vectors.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
