#!/usr/bin/env python3
"""Distill the PoseFall Transformer into an SVP-friendly fixed-window MLP."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


WINDOW = 60
DIM = 56


class DeployablePoseFallHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(WINDOW * DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class SequenceDataset(Dataset):
    def __init__(self, entries: list[tuple[torch.Tensor, float]], training: bool, seed: int):
        self.entries = entries
        self.training = training
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        sequence, label = self.entries[index]
        length = len(sequence)
        if length >= WINDOW:
            if self.training:
                start = self.rng.randrange(length - WINDOW + 1)
            else:
                start = max(0, (length - WINDOW) // 2)
            window = sequence[start : start + WINDOW]
        else:
            window = torch.cat((sequence[:1].repeat(WINDOW - length, 1), sequence), 0)
        return window, torch.tensor(label, dtype=torch.float32)


def split_key(source: str) -> int:
    return int(hashlib.sha1(source.encode()).hexdigest()[:8], 16) % 10


def load_entries(root: Path):
    train, val = [], []
    counts = {"train": [0, 0], "val": [0, 0]}
    for class_name, label in (("No_Fall", 0), ("Fall", 1)):
        for path in sorted((root / class_name).glob("*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            sequence = payload["sequence"].float()
            if sequence.ndim != 2 or sequence.shape[1] != DIM or not len(sequence):
                continue
            source = str(payload.get("source_video", path.name))
            bucket = val if split_key(source) < 2 else train
            name = "val" if bucket is val else "train"
            bucket.append((sequence, float(label)))
            counts[name][label] += 1
    return train, val, counts


@torch.no_grad()
def evaluate(student, teacher, loader, device):
    student.eval()
    teacher.eval()
    all_student, all_teacher, all_labels = [], [], []
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        all_student.append(student(features).sigmoid().cpu())
        all_teacher.append(teacher(features).sigmoid().cpu())
        all_labels.append(labels)
    sp = torch.cat(all_student)
    tp = torch.cat(all_teacher)
    labels = torch.cat(all_labels)
    pred = sp >= 0.5
    truth = labels >= 0.5
    tp_n = int((pred & truth).sum())
    fp_n = int((pred & ~truth).sum())
    fn_n = int((~pred & truth).sum())
    precision = tp_n / max(tp_n + fp_n, 1)
    recall = tp_n / max(tp_n + fn_n, 1)
    return {
        "samples": len(sp),
        "label_accuracy": float((pred == truth).float().mean()),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "teacher_mae": float((sp - tp).abs().mean()),
        "teacher_class_agreement": float(((sp >= 0.5) == (tp >= 0.5)).float().mean()),
        "student_probability_mean": float(sp.mean()),
        "teacher_probability_mean": float(tp.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=610)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_entries, val_entries, counts = load_entries(args.cache_root)
    print("split", counts, flush=True)
    train_loader = DataLoader(SequenceDataset(train_entries, True, args.seed), batch_size=args.batch,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    val_loader = DataLoader(SequenceDataset(val_entries, False, args.seed), batch_size=args.batch,
                            shuffle=False, num_workers=4, pin_memory=True)

    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    teacher = checkpoint["model"].posefall_head.float().eval().to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = DeployablePoseFallHead().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-5)
    bce = nn.BCEWithLogitsLoss()

    best = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        student.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                soft_targets = teacher(features).sigmoid()
            logits = student(features)
            # Teacher supervision preserves clip timing; labels retain a weak
            # video-level anchor for outliers where the teacher is uncertain.
            loss = 0.85 * bce(logits, soft_targets) + 0.15 * bce(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(features)
        scheduler.step()
        metrics = evaluate(student, teacher, val_loader, device)
        metrics.update(epoch=epoch + 1, train_loss=total_loss / len(train_entries))
        print(json.dumps(metrics), flush=True)
        score = metrics["teacher_mae"]
        if best is None or score < best[0]:
            best = (score, metrics)
            torch.save({"model": student.state_dict(), "metrics": metrics}, args.output_dir / "best.pt")

    saved = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    student = DeployablePoseFallHead().eval()
    student.load_state_dict(saved["model"])

    class WithSigmoid(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, features):
            return self.model(features).sigmoid()

    onnx_path = args.output_dir / "posefall_head_mlp.onnx"
    torch.onnx.export(WithSigmoid(student), torch.zeros(1, WINDOW, DIM), onnx_path,
                      input_names=["features"], output_names=["fall_prob"], opset_version=13,
                      do_constant_folding=True)
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "split": counts, "best": best[1], "window": WINDOW, "feature_dim": DIM,
        "architecture": "Flatten-Linear(3360,128)-ReLU-Linear(128,32)-ReLU-Linear(32,1)-Sigmoid",
    }, indent=2), encoding="utf-8")
    print(f"exported {onnx_path}")


if __name__ == "__main__":
    main()
