# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
from torch import nn

from ultralytics.models.yolo.fall_utils import aggregate_clip_logits

POSEFALL_BOX_DIM = 5
POSEFALL_KEYPOINT_DIM = 51
POSEFALL_FEATURE_DIM = POSEFALL_BOX_DIM + POSEFALL_KEYPOINT_DIM
POSEFALL_FEATURE_VERSION = 5


def sinusoidal_positional_encoding(window: int, d_model: int) -> torch.Tensor:
    """Build sinusoidal positional encodings shaped [1, window, d_model]."""
    position = torch.arange(window, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
    encoding = torch.zeros(window, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


class PoseFallTransformer(nn.Module):
    """Transformer head for pose-based video-level fall classification over sliding clips."""

    def __init__(
        self,
        input_dim: int = POSEFALL_FEATURE_DIM,
        num_classes: int = 1,
        window: int = 60,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        stride: int = 15,
    ):
        super().__init__()
        self.window = window
        self.input_dim = input_dim
        self.d_model = d_model
        self.stride = stride
        self.input_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model), nn.GELU())
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
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Return one logit per complete video sequence."""
        return self.forward_with_clips(x)[0]

    def forward_with_clips(self, x: torch.Tensor | list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return video-level logits and clip-level logits used by MIL supervision."""
        if isinstance(x, (list, tuple)):
            outputs = [self._sequence_outputs(sequence) for sequence in x]
            return torch.stack([item[0] for item in outputs]), [item[1] for item in outputs]
        if x.ndim == 2:
            video_logit, clip_logits = self._sequence_outputs(x)
            return video_logit.unsqueeze(0), [clip_logits]
        if x.ndim == 3 and x.shape[1] == self.window:
            logits = self._clip_logits(x)
            return logits, [item.reshape(1) for item in logits.unbind()]
        if x.ndim == 3:
            outputs = [self._sequence_outputs(sequence) for sequence in x]
            return torch.stack([item[0] for item in outputs]), [item[1] for item in outputs]
        raise ValueError(f"Expected [T,D], [B,T,D], or list of [T,D], got {tuple(x.shape)}")

    def _sequence_outputs(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clip_logits = self._clip_logits(self._sliding_clips(sequence))
        return aggregate_clip_logits(clip_logits), clip_logits

    def _sliding_clips(self, sequence: torch.Tensor) -> torch.Tensor:
        if len(sequence) <= self.window:
            return pad_or_trim(sequence, self.window).unsqueeze(0)
        clips = [sequence[start : start + self.window] for start in range(0, len(sequence) - self.window + 1, self.stride)]
        if (len(sequence) - self.window) % self.stride:
            clips.append(sequence[-self.window :])
        return torch.stack(clips)

    def _clip_logits(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.shape[1] > self.pos_embed.shape[1]:
            positional = sinusoidal_positional_encoding(clips.shape[1], self.d_model).to(clips.device, clips.dtype)
        else:
            positional = self.pos_embed[:, : clips.shape[1]].to(clips.device, clips.dtype)
        encoded = self.encoder(self.input_proj(clips) + positional)
        weights = self.pool_score(encoded).squeeze(-1).softmax(1)
        summary = (encoded * weights.unsqueeze(-1)).sum(1)
        return self.head(summary).squeeze(-1)


def pad_or_trim(sequence: torch.Tensor, window: int) -> torch.Tensor:
    """Pad short sequences at the beginning by repeating their first observation."""
    if len(sequence) >= window:
        return sequence[-window:]
    padding = sequence[:1].repeat(window - len(sequence), 1)
    return torch.cat((padding, sequence))


def pose_result_features(result, device: torch.device | str | None = None) -> torch.Tensor:
    """Tensorize raw box/confidence and globally normalized 17-point pose output."""
    if result.boxes is None or result.keypoints is None or not len(result.boxes):
        return torch.zeros((0, POSEFALL_FEATURE_DIM), device=device or "cpu")
    target_device = torch.device(device) if device is not None else result.boxes.data.device
    boxes = result.boxes.xywhn.to(target_device).float()
    confidence = result.boxes.conf[:, None].to(target_device).float()
    height, width = result.orig_shape
    keypoints = result.keypoints.data.to(target_device).float().clone()
    keypoints[..., 0] /= max(float(width), 1.0)
    keypoints[..., 1] /= max(float(height), 1.0)
    keypoints[..., 2].clamp_(0, 1)
    return torch.cat((boxes, confidence, keypoints.flatten(1)), 1)


def mean_keypoint_confidence(sequence: torch.Tensor) -> float:
    """Return mean confidence from flattened [x, y, confidence] keypoints."""
    if sequence.numel() == 0:
        return 0.0
    keypoints = sequence[..., POSEFALL_BOX_DIM:].reshape(-1, 17, 3)
    return float(keypoints[..., 2].mean().item())
