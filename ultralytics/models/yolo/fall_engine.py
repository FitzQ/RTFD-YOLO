# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy, deepcopy

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from ultralytics.data import build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.validator import BaseValidator
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, RANK, TQDM
from ultralytics.utils.torch_utils import autocast, unwrap_model

from .fall_utils import atomic_torch_save


def collate_fall_sequences(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict:
    """Pack variable-length videos without padding; lengths restore sequence boundaries in the model."""
    sequences, labels = zip(*batch)
    return {
        "img": torch.cat(sequences),
        "lengths": torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long),
        "cls": torch.stack(labels).float(),
    }


def single_device_args(args):
    """Use one accelerator for cache extraction before native DDP temporal-head training starts."""
    result = copy(args)
    device = result.device
    if isinstance(device, (list, tuple)):
        result.device = device[0] if device else None
    elif isinstance(device, str) and "," in device:
        result.device = device.split(",", 1)[0]
    return result


class FallTrainingModel(nn.Module):
    """Lightweight train-only adapter around a temporal Fall head."""

    def __init__(
        self,
        head: nn.Module,
        negative_clip_weight: float,
        positive_clip_weight: float,
        positive_topk: int,
        temporal_smooth_weight: float,
        task: str,
    ):
        super().__init__()
        self.fall_head = head
        self.negative_clip_weight = float(negative_clip_weight)
        self.positive_clip_weight = float(positive_clip_weight)
        self.positive_topk = max(int(positive_topk), 1)
        self.temporal_smooth_weight = float(temporal_smooth_weight)
        self.task = task
        self.names = {0: "no_fall", 1: "fall"}
        self.stride = torch.tensor([32.0])
        self.yaml = {"task": task}
        self.args = {"task": task}
        self.criterion = None

    @staticmethod
    def _sequences(batch: dict) -> list[torch.Tensor]:
        return list(torch.split(batch["img"], batch["lengths"].detach().cpu().tolist()))

    def forward(self, batch, *args, **kwargs):
        if isinstance(batch, dict):
            return self.loss(batch)
        raise TypeError("FallTrainingModel inference requires a packed training batch")

    def predict_logits(self, batch: dict) -> torch.Tensor:
        return self.fall_head(self._sequences(batch))

    def loss(self, batch: dict, preds=None) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = self._sequences(batch)
        targets = batch["cls"].view(-1)
        video_logits, clip_logits = self.fall_head.forward_with_clips(sequences)
        video_loss = F.binary_cross_entropy_with_logits(video_logits, targets)
        positive_losses = []
        negative_losses = [
            F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
            for logits, target in zip(clip_logits, targets)
            if target.detach().item() < 0.5
        ]
        smooth_losses = []
        for logits, target in zip(clip_logits, targets):
            if target.detach().item() >= 0.5:
                selected = logits.topk(min(self.positive_topk, logits.numel())).values
                positive_losses.append(F.binary_cross_entropy_with_logits(selected, torch.ones_like(selected)))
            if logits.numel() > 1:
                probabilities = logits.sigmoid()
                smooth_losses.append(F.smooth_l1_loss(probabilities[1:], probabilities[:-1]))
        positive_clip_loss = torch.stack(positive_losses).mean() if positive_losses else video_loss.new_zeros(())
        negative_clip_loss = torch.stack(negative_losses).mean() if negative_losses else video_loss.new_zeros(())
        smooth_loss = torch.stack(smooth_losses).mean() if smooth_losses else video_loss.new_zeros(())
        loss = (
            video_loss
            + self.positive_clip_weight * positive_clip_loss
            + self.negative_clip_weight * negative_clip_loss
            + self.temporal_smooth_weight * smooth_loss
        )
        return loss, torch.stack(
            (video_loss.detach(), positive_clip_loss.detach(), negative_clip_loss.detach(), smooth_loss.detach())
        )


class FallMetrics:
    """Metric interface expected by BaseTrainer and integrations."""

    keys = [
        "metrics/accuracy",
        "metrics/precision",
        "metrics/recall",
        "metrics/f1",
        "metrics/p90",
        "metrics/p95",
        "metrics/competition_map",
    ]

    def __init__(self):
        self.results_dict = {key: 0.0 for key in self.keys}


def fall_metrics(targets: list[int], probs: list[float], threshold: float) -> dict[str, float]:
    preds = [prob >= threshold for prob in probs]
    tp = sum(target == 1 and pred for target, pred in zip(targets, preds))
    fp = sum(target == 0 and pred for target, pred in zip(targets, preds))
    tn = sum(target == 0 and not pred for target, pred in zip(targets, preds))
    fn = sum(target == 1 and not pred for target, pred in zip(targets, preds))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    def precision_at_recall(required: float) -> float:
        positives = sum(targets)
        running_tp = running_fp = 0
        best = 0.0
        ranked = sorted(zip(probs, targets), key=lambda item: item[0], reverse=True)
        index = 0
        while index < len(ranked):
            score = ranked[index][0]
            while index < len(ranked) and ranked[index][0] == score:
                target = ranked[index][1]
                running_tp += target == 1
                running_fp += target == 0
                index += 1
            if positives and running_tp / positives >= required:
                best = max(best, running_tp / max(running_tp + running_fp, 1))
        return best

    p90, p95 = precision_at_recall(0.90), precision_at_recall(0.95)
    return {
        "metrics/accuracy": (tp + tn) / max(len(targets), 1),
        "metrics/precision": precision,
        "metrics/recall": recall,
        "metrics/f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "metrics/p90": p90,
        "metrics/p95": p95,
        "metrics/competition_map": (p90 + p95) * 50.0,
    }


class FallValidator(BaseValidator):
    """BaseValidator implementation for cached full-video sequences during training."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.metrics = FallMetrics()

    def validate_cached(self, trainer) -> dict[str, float] | None:
        self.training = True
        self.device = trainer.device
        self.data = trainer.data
        self.args.half = self.device.type != "cpu" and trainer.amp
        model = trainer.ema.ema or trainer.model
        model = unwrap_model(model).float().eval()
        self.loss = torch.zeros_like(trainer.loss_items, device=self.device)
        self.targets, self.probs = [], []
        self.run_callbacks("on_val_start")
        bar = TQDM(trainer.test_loader, desc=self.get_desc(), total=len(trainer.test_loader))
        for batch_i, batch in enumerate(bar):
            self.batch_i = batch_i
            self.run_callbacks("on_val_batch_start")
            batch = self.preprocess(batch)
            with autocast(self.args.half, device=self.device.type):
                logits = model.predict_logits(batch)
                _, loss_items = model.loss(batch)
            self.loss += loss_items
            self.probs.extend(logits.sigmoid().float().cpu().tolist())
            self.targets.extend(batch["cls"].int().cpu().tolist())
            self.run_callbacks("on_val_batch_end")
        self.gather_stats()
        loss = self.loss.clone().detach()
        if trainer.world_size > 1:
            dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
        if RANK > 0:
            return None
        stats = self.get_stats()
        self.print_results()
        self.run_callbacks("on_val_end")
        stats.update(trainer.label_loss_items(loss.cpu() / max(len(trainer.test_loader), 1), prefix="val"))
        self.metrics.results_dict = stats
        return {key: round(float(value), 5) for key, value in stats.items()}

    def preprocess(self, batch: dict) -> dict:
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda").float()
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def get_stats(self) -> dict[str, float]:
        threshold = float(self.args.fall_threshold)
        return fall_metrics(self.targets, self.probs, threshold)

    def gather_stats(self) -> None:
        if RANK == 0:
            world = dist.get_world_size()
            gathered_probs, gathered_targets = [None] * world, [None] * world
            dist.gather_object(self.probs, gathered_probs, dst=0)
            dist.gather_object(self.targets, gathered_targets, dst=0)
            self.probs = [value for values in gathered_probs for value in values]
            self.targets = [value for values in gathered_targets for value in values]
        elif RANK > 0:
            dist.gather_object(self.probs, None, dst=0)
            dist.gather_object(self.targets, None, dst=0)

    def get_desc(self) -> str:
        return f"{'Class':>12}{'Videos':>10}{'Acc':>10}{'P':>10}{'R':>10}{'F1':>10}{'P90':>10}{'P95':>10}{'Score':>10}"

    def print_results(self) -> None:
        m = self.get_stats()
        LOGGER.info(
            f"{'all':>12}{len(self.targets):>10}{m['metrics/accuracy']:>10.3f}{m['metrics/precision']:>10.3f}"
            f"{m['metrics/recall']:>10.3f}{m['metrics/f1']:>10.3f}{m['metrics/p90']:>10.3f}"
            f"{m['metrics/p95']:>10.3f}{m['metrics/competition_map']:>10.2f}"
        )


class FallTrainer(BaseTrainer):
    """Common native Ultralytics trainer for PoseFall, SegFall, and PoseSegFall."""

    task_name = "posefall"
    head_attr = "posefall_head"
    config_attr = "posefall_config"
    frontend_type = nn.Module

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})
        overrides["task"] = self.task_name
        if overrides.get("data") is None:
            overrides["data"] = {"task": self.task_name, "format": "fall_video_directories"}
        if "lr0" not in overrides:
            overrides["lr0"] = float(overrides.get("fall_lr0", DEFAULT_CFG_DICT["fall_lr0"]))
        if "optimizer" not in overrides:
            overrides["optimizer"] = overrides.get("fall_optimizer", DEFAULT_CFG_DICT["fall_optimizer"])
        if "warmup_bias_lr" not in overrides:
            overrides["warmup_bias_lr"] = overrides.get(
                "fall_warmup_bias_lr", DEFAULT_CFG_DICT["fall_warmup_bias_lr"]
            )
        if overrides.get("compile"):
            LOGGER.warning("Fall full-video batches do not support compile yet; setting compile=False")
            overrides["compile"] = False
        super().__init__(cfg, overrides, _callbacks)
        self.loss_names = ("video_loss", "pos_clip_loss", "neg_clip_loss", "smooth_loss")
        self.frontend_template = None

    def get_dataset(self):
        train_set, val_set = self.build_fall_datasets()
        return {"train": train_set, "val": val_set, "names": {0: "no_fall", 1: "fall"}, "nc": 2, "channels": 1}

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        return build_dataloader(
            dataset_path,
            batch=batch_size,
            workers=self.args.workers,
            shuffle=mode == "train",
            rank=rank,
            pin_memory=True,
        )

    def setup_model(self):
        if isinstance(self.model, FallTrainingModel):
            return None
        frontend, ckpt = load_checkpoint(self.model, device="cpu", fuse=False)
        if not isinstance(frontend, self.frontend_type):
            raise TypeError(f"{self.task_name} requires {self.frontend_type.__name__} weights, got {type(frontend).__name__}")
        self.frontend_template = frontend.cpu().float()
        existing_head = getattr(frontend, self.head_attr, None)
        head = deepcopy(existing_head).float() if existing_head is not None else self.build_fall_head()
        for parameter in head.parameters():
            parameter.requires_grad = True
        self.model = FallTrainingModel(
            head,
            negative_clip_weight=float(self.args.fall_negative_clip_weight),
            positive_clip_weight=float(self.args.fall_positive_clip_weight),
            positive_topk=int(self.args.fall_positive_topk),
            temporal_smooth_weight=float(self.args.fall_temporal_smooth_weight),
            task=self.task_name,
        )
        if ckpt is not None and self.resume:
            ckpt = dict(ckpt)
            ckpt["ema"] = deepcopy(self.model).float()
        return ckpt

    def set_model_attributes(self):
        self.model.names = self.data["names"]
        self.model.args = vars(self.args)

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda").float()
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def get_validator(self):
        validator = self.validator_class(
            dataloader=self.test_loader,
            save_dir=self.save_dir,
            args=vars(self.args),
            _callbacks=self.callbacks,
        )
        return validator

    def validate(self):
        metrics = self.validator.validate_cached(self)
        if metrics is None:
            return None, None
        fitness = -(
            float(metrics["val/video_loss"])
            + float(self.args.fall_positive_clip_weight) * float(metrics["val/pos_clip_loss"])
            + float(self.args.fall_negative_clip_weight) * float(metrics["val/neg_clip_loss"])
            + float(self.args.fall_temporal_smooth_weight) * float(metrics["val/smooth_loss"])
        )
        if self.best_fitness is None or fitness > self.best_fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{name}" for name in self.loss_names]
        if loss_items is None:
            return keys
        values = loss_items if loss_items.ndim else loss_items.unsqueeze(0)
        return dict(zip(keys, values.tolist()))

    def progress_string(self):
        display_names = [name.removesuffix("_loss") for name in self.loss_names]
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *display_names,
            "Videos",
            "Features",
        )

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        max_norm = float(self.args.fall_grad_clip)
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def auto_batch(self, max_num_obj=0, dataset_size=0):
        """Probe temporal-head batch size using the longest cached full-video sequences."""
        dataset = self.data["train"]
        if self.device.type == "cpu":
            return min(16, len(dataset))
        fraction = float(self.batch_size) if 0 < float(self.batch_size) < 1 else 0.6
        ordered = sorted(range(len(dataset)), key=lambda index: len(dataset.sequences[index].sequence), reverse=True)
        best = 1
        for candidate in (1, 2, 4, 8, 16, 32, 64, 128):
            if candidate > len(dataset):
                break
            indexes = [ordered[index % len(ordered)] for index in range(candidate)]
            batch = collate_fall_sequences([dataset[index] for index in indexes])
            batch = self.preprocess_batch(batch)
            try:
                self.model.zero_grad(set_to_none=True)
                with autocast(self.amp, device=self.device.type):
                    loss, _ = self.model(batch)
                loss.backward()
                used = torch.cuda.max_memory_reserved(self.device)
                total = torch.cuda.get_device_properties(self.device).total_memory
                if used / total > fraction:
                    break
                best = candidate
            except torch.cuda.OutOfMemoryError:
                break
            finally:
                self.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device)
        LOGGER.info(f"AutoBatch: using batch={best} for full-video {self.task_name} sequences")
        return best

    def save_model(self):
        saved = super().save_model()
        if not saved:
            return False
        paths = [self.last]
        if self.best_fitness == self.fitness:
            paths.append(self.best)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            paths.append(self.wdir / f"epoch{self.epoch}.pt")
        for path in paths:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            adapter = checkpoint["ema"].float()
            checkpoint["ema"] = self.build_complete_model(adapter.fall_head).half()
            atomic_torch_save(checkpoint, path)
        return True

    def train(self):
        result = super().train()
        if self.validator is not None:
            metrics = self.validator.metrics
            return metrics.results_dict if hasattr(metrics, "results_dict") else metrics
        return result

    def build_complete_model(self, head: nn.Module):
        complete = deepcopy(self.frontend_template).cpu().float()
        setattr(complete, self.head_attr, deepcopy(head).cpu().float())
        setattr(complete, self.config_attr, self.complete_config())
        complete.task = self.task_name
        complete.args = {**dict(getattr(complete, "args", {})), **vars(self.args), "task": self.task_name}
        return complete

    def build_fall_datasets(self):
        raise NotImplementedError

    def build_fall_head(self):
        raise NotImplementedError

    def complete_config(self):
        raise NotImplementedError
