#!/usr/bin/env python3
"""Replay a board trace through Ultralytics BoT-SORT and compare IDs/boxes."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, help="CSV emitted by posefall_predict --trace")
    parser.add_argument("--box-atol", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames: dict[int, list[dict[str, str]]] = defaultdict(list)
    with args.trace.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frames[int(row["frame"])].append(row)

    config = SimpleNamespace(
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
        gmc_method="none",
        proximity_thresh=0.5,
        appearance_thresh=0.8,
        with_reid=False,
        model="auto",
    )
    tracker = BOTSORT(config)
    checked = 0
    failures: list[str] = []
    for frame in range(min(frames), max(frames) + 1):
        rows = frames.get(frame, [])
        detection_rows = [row for row in rows if row.get("det_conf")]
        data = [
            [
                float(row["det_x1"]),
                float(row["det_y1"]),
                float(row["det_x2"]),
                float(row["det_y2"]),
                float(row["det_conf"]),
                0.0,
            ]
            for row in detection_rows
        ]
        tensor = torch.tensor(data, dtype=torch.float32).reshape(-1, 6)
        output = tracker.update(Boxes(tensor, (640, 640)))
        expected_by_detection = {int(item[-1]): item for item in output}

        for detection_index, row in enumerate(detection_rows):
            board_id = int(row["track_id"]) if row.get("track_id") else None
            expected = expected_by_detection.get(detection_index)
            expected_id = int(expected[4]) if expected is not None else None
            if board_id != expected_id:
                failures.append(
                    f"frame {frame} detection {detection_index}: "
                    f"board ID={board_id}, Python ID={expected_id}"
                )
                continue
            if board_id is None:
                continue
            board_box = torch.tensor(
                [
                    float(row["track_x1"]),
                    float(row["track_y1"]),
                    float(row["track_x2"]),
                    float(row["track_y2"]),
                ]
            )
            error = float(torch.max(torch.abs(board_box - torch.as_tensor(expected[:4]))))
            if error > args.box_atol:
                failures.append(
                    f"frame {frame} ID {board_id}: box max_abs={error:.6g} "
                    f"> {args.box_atol:.6g}"
                )
            checked += 1

    if failures:
        print("\n".join(failures))
        print(f"FAIL: {len(failures)} mismatches, {checked} tracked rows checked")
        return 1
    print(f"PASS: IDs agree and boxes are within {args.box_atol:g} px ({checked} tracked rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
