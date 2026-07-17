#!/usr/bin/env python3
"""Build an image-level intersection of COCO person pose and segmentation labels."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


SPLITS = ("train2017", "val2017")


def replace_symlink(link: Path, target: Path) -> None:
    """Create a relative directory symlink, retaining a correct existing link."""
    relative_target = Path(os.path.relpath(target, link.parent))
    if link.is_symlink() and Path(os.readlink(link)) == relative_target:
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Refusing to replace existing path: {link}")
    link.symlink_to(relative_target, target_is_directory=True)


def labeled_stems(label_dir: Path) -> set[str]:
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    return {path.stem for path in label_dir.glob("*.txt") if path.stat().st_size > 0}


def build_dataset(coco: Path, pose: Path, seg: Path, output: Path) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    replace_symlink(output / "images", coco / "images")
    replace_symlink(output / "labels", pose / "labels")
    replace_symlink(output / "labels_pose", pose / "labels")
    replace_symlink(output / "labels_seg", seg / "labels")

    counts = {}
    for split in SPLITS:
        pose_stems = labeled_stems(pose / "labels" / split)
        seg_stems = labeled_stems(seg / "labels" / split)
        common = sorted(pose_stems & seg_stems)
        missing_images = [stem for stem in common if not (coco / "images" / split / f"{stem}.jpg").is_file()]
        if missing_images:
            sample = ", ".join(missing_images[:5])
            raise FileNotFoundError(f"{len(missing_images)} intersected {split} images are missing, e.g. {sample}")
        if not common:
            raise RuntimeError(f"No common pose/seg labels found for {split}")
        lines = [f"./images/{split}/{stem}.jpg\n" for stem in common]
        (output / f"{split}.txt").write_text("".join(lines), encoding="utf-8")
        counts[split] = len(common)
        print(
            f"{split}: pose={len(pose_stems)} seg={len(seg_stems)} "
            f"intersection={len(common)} pose_only={len(pose_stems - seg_stems)} seg_only={len(seg_stems - pose_stems)}"
        )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, default=Path("datasets/coco"), help="COCO image dataset root")
    parser.add_argument("--pose", type=Path, default=Path("datasets/coco-pose"), help="COCO pose label root")
    parser.add_argument("--seg", type=Path, default=Path("datasets/coco-person-seg"), help="Person-seg label root")
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/coco-person-pose-seg"), help="Intersection dataset root"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(args.coco.resolve(), args.pose.resolve(), args.seg.resolve(), args.output.resolve())
