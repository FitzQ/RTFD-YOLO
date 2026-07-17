from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.models.yolo.fall_utils import aggregate_clip_logits

POSEGFALL_MASK_SIZE = 16
POSEGFALL_BOX_DIM = 5
POSEGFALL_KEYPOINT_DIM = 51
POSEGFALL_FEATURE_DIM = POSEGFALL_BOX_DIM + POSEGFALL_KEYPOINT_DIM + POSEGFALL_MASK_SIZE**2
POSEGFALL_FEATURE_VERSION = 2


def sinusoidal_positional_encoding(window: int, d_model: int) -> torch.Tensor:
    position = torch.arange(window, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
    encoding = torch.zeros(window, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


class PoseSegFallTransformer(nn.Module):
    """Learn fall representations directly from box, keypoint, and spatial mask tensors over time."""

    def __init__(
        self,
        input_dim: int = POSEGFALL_FEATURE_DIM,
        num_classes: int = 1,
        window: int = 60,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        stride: int = 15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.window = window
        self.stride = stride
        self.d_model = d_model
        self.input_proj1 = nn.Linear(56, d_model // 2)
        self.input_proj2 = nn.Linear(256, d_model // 2)
        self.input_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        # self.input_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model), nn.GELU())
        self.register_buffer("pos_embed", sinusoidal_positional_encoding(window, d_model), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_score = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        return self.forward_with_clips(x)[0]

    def forward_with_clips(self, x: torch.Tensor | list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if isinstance(x, (list, tuple)):
            outputs = [self._sequence_outputs(seq) for seq in x]
            return torch.stack([item[0] for item in outputs]), [item[1] for item in outputs]
        if x.ndim == 2:
            video, clips = self._sequence_outputs(x)
            return video.unsqueeze(0), [clips]
        if x.ndim == 3 and x.shape[1] == self.window:
            logits = self._clip_logits(x)
            return logits, [item.reshape(1) for item in logits.unbind()]
        if x.ndim == 3:
            outputs = [self._sequence_outputs(seq) for seq in x]
            return torch.stack([item[0] for item in outputs]), [item[1] for item in outputs]
        raise ValueError(f"Expected [T,D], [B,T,D], or list of [T,D], got {tuple(x.shape)}")

    def _sequence_outputs(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clips = self._sliding_clips(sequence)
        logits = self._clip_logits(clips)
        return aggregate_clip_logits(logits), logits

    def _sliding_clips(self, sequence: torch.Tensor) -> torch.Tensor:
        if len(sequence) <= self.window:
            return pad_or_trim(sequence, self.window).unsqueeze(0)
        clips = [sequence[start : start + self.window] for start in range(0, len(sequence) - self.window + 1, self.stride)]
        if (len(sequence) - self.window) % self.stride:
            clips.append(sequence[-self.window :])
        return torch.stack(clips)

    def _clip_logits(self, clips: torch.Tensor) -> torch.Tensor:
        x1=self.input_proj1(clips[..., :56])
        x2=self.input_proj2(clips[..., 56:])
        positional = self.pos_embed[:, : clips.shape[1]].to(device=clips.device, dtype=clips.dtype)
        encoded = self.encoder(self.input_proj(torch.cat([x1, x2], dim=-1)) + positional)
        # encoded = self.encoder(self.input_proj(clips) + positional)
        weights = self.pool_score(encoded).softmax(1)
        return self.head((encoded * weights).sum(1)).squeeze(-1)


def pad_or_trim(sequence: torch.Tensor, window: int) -> torch.Tensor:
    if len(sequence) >= window:
        return sequence[-window:]
    if not len(sequence):
        return sequence.new_zeros((window, sequence.shape[-1]))
    return torch.cat((sequence[:1].repeat(window - len(sequence), 1), sequence), 0)


def poseg_result_features(result, device: torch.device | str | None = None) -> torch.Tensor:
    """Create fixed tensors from PoseSeg outputs without hand-designed geometric statistics."""
    if result.boxes is None or result.keypoints is None or result.masks is None or not len(result.boxes):
        return torch.zeros((0, POSEGFALL_FEATURE_DIM), device=device or "cpu")
    target_device = torch.device(device) if device is not None else result.boxes.data.device
    boxes = result.boxes.xywhn.to(target_device)
    confidence = result.boxes.conf[:, None].to(target_device)
    height, width = result.orig_shape
    keypoints = result.keypoints.data.to(target_device).float().clone()
    keypoints[..., 0] /= max(float(width), 1.0)
    keypoints[..., 1] /= max(float(height), 1.0)
    keypoints[..., 2].clamp_(0, 1)
    masks = result.masks.data.to(target_device).float()
    mask_h, mask_w = masks.shape[-2:]
    mask_tokens = []
    for mask, box in zip(masks, boxes):
        cx, cy, bw, bh = box
        x1 = max(0, min(mask_w - 1, int((cx - bw / 2) * mask_w)))
        y1 = max(0, min(mask_h - 1, int((cy - bh / 2) * mask_h)))
        x2 = max(x1 + 1, min(mask_w, int((cx + bw / 2) * mask_w)))
        y2 = max(y1 + 1, min(mask_h, int((cy + bh / 2) * mask_h)))
        crop = mask[y1:y2, x1:x2][None, None]
        mask_tokens.append(F.interpolate(crop, (POSEGFALL_MASK_SIZE, POSEGFALL_MASK_SIZE), mode="bilinear", align_corners=False).flatten())
    return torch.cat((boxes, confidence, keypoints.flatten(1), torch.stack(mask_tokens)), 1)


def mean_poseg_confidence(sequence: torch.Tensor) -> float:
    if not sequence.numel():
        return 0.0
    keypoints = sequence[:, POSEGFALL_BOX_DIM : POSEGFALL_BOX_DIM + POSEGFALL_KEYPOINT_DIM].reshape(-1, 17, 3)
    return float(keypoints[..., 2].mean())
