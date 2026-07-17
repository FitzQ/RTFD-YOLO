# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
from torch import nn

from ultralytics.models.yolo.fall_utils import aggregate_clip_logits

SEGFALL_MASK_SIZE = 16
SEGFALL_BOX_DIM = 5
SEGFALL_FEATURE_DIM = SEGFALL_BOX_DIM + SEGFALL_MASK_SIZE**2
SEGFALL_FEATURE_VERSION = 4
SEGFALL_FEATURE_TYPE = "raw_box_conf_mask_grid"


def sinusoidal_positional_encoding(window: int, d_model: int) -> torch.Tensor:
    """Build sinusoidal positional encodings shaped [1, window, d_model]."""
    position = torch.arange(window, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe = torch.zeros(window, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class SegFallTransformer(nn.Module):
    """Transformer head for segmentation-based video-level fall classification with internal sliding-window pooling."""

    def __init__(
        self,
        input_dim: int = SEGFALL_FEATURE_DIM,
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
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Classify full sequences using fixed-length sliding windows."""
        return self.forward_with_clips(x)[0]

    def forward_with_clips(self, x: torch.Tensor | list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return video logits and the corresponding clip logits."""
        if isinstance(x, (list, tuple)):
            outputs = [self._sequence_outputs(seq) for seq in x]
            return torch.stack([item[0] for item in outputs]), [item[1] for item in outputs]
        if x.ndim == 2:
            video_logit, clip_logits = self._sequence_outputs(x)
            return video_logit.unsqueeze(0), [clip_logits]
        if x.ndim == 3 and x.shape[1] == self.window:
            logits = self._clip_logits(x)
            return logits, [item.reshape(1) for item in logits.unbind()]
        if x.ndim == 3:
            outputs = [self._sequence_outputs(seq) for seq in x]
            return torch.stack([item[0] for item in outputs]), [item[1] for item in outputs]
        raise ValueError(f"Expected x as [T,D], [B,T,D], or list of [T,D] tensors, got shape={tuple(x.shape)}")

    def _sequence_outputs(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clips = self._sliding_clips(seq)
        logits = self._clip_logits(clips)
        return aggregate_clip_logits(logits), logits

    def _sliding_clips(self, seq: torch.Tensor) -> torch.Tensor:
        """Build [num_clips, window, D] clips from one full sequence."""
        if seq.shape[0] <= self.window:
            return pad_or_trim(seq, self.window).unsqueeze(0)
        clips = [seq[start : start + self.window] for start in range(0, seq.shape[0] - self.window + 1, self.stride)]
        if not clips or (seq.shape[0] - self.window) % self.stride != 0:
            clips.append(seq[-self.window :])
        return torch.stack(clips)

    def _clip_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Classify fixed-length clips shaped [B, window, D]."""
        if x.shape[1] > self.pos_embed.shape[1]:
            pos_embed = sinusoidal_positional_encoding(x.shape[1], self.d_model).to(device=x.device, dtype=x.dtype)
        else:
            pos_embed = self.pos_embed[:, : x.shape[1]].to(dtype=x.dtype, device=x.device)
        x = self.input_proj(x)
        x = self.encoder(x + pos_embed)
        weights = self.pool_score(x).softmax(1)
        return self.head((x * weights).sum(1)).squeeze(-1)


def pad_or_trim(seq: torch.Tensor, window: int) -> torch.Tensor:
    """Return a fixed-length sequence, padding the beginning with the first frame when needed."""
    if seq.shape[0] >= window:
        return seq[-window:]
    pad = seq[:1].repeat(window - seq.shape[0], 1)
    return torch.cat([pad, seq], dim=0)


def _polygon_mask_grid(poly, box: torch.Tensor, orig_shape, size: int = SEGFALL_MASK_SIZE) -> torch.Tensor:
    """Rasterize an instance polygon into a fixed ROI grid without geometric feature engineering."""
    device, dtype = box.device, box.dtype
    points = torch.as_tensor(poly, device=device, dtype=dtype) if poly is not None else box.new_empty((0, 2))
    if points.ndim != 2 or points.shape[0] < 3:
        return box.new_zeros(size * size)
    height, width = (float(orig_shape[0]), float(orig_shape[1]))
    points = torch.stack((points[:, 0] / max(width, 1.0), points[:, 1] / max(height, 1.0)), 1)
    cx, cy, bw, bh = box[:4]
    axis = (torch.arange(size, device=device, dtype=dtype) + 0.5) / size
    ys, xs = torch.meshgrid(cy - bh / 2 + axis * bh, cx - bw / 2 + axis * bw, indexing="ij")
    xs, ys = xs.flatten(), ys.flatten()
    x1, y1 = points[:, 0:1], points[:, 1:2]
    x2, y2 = torch.roll(x1, 1, 0), torch.roll(y1, 1, 0)
    intersects = ((y1 > ys) != (y2 > ys)) & (xs < (x2 - x1) * (ys - y1) / (y2 - y1 + 1e-7) + x1)
    return (intersects.sum(0) % 2).to(dtype)


def raw_segment_features(
    boxes_xywhn: torch.Tensor,
    mask_polygons,
    orig_shape: tuple[int, int] | torch.Tensor,
    conf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tensorize raw segmentation detections as box/confidence plus a spatial mask grid."""
    if boxes_xywhn is None or boxes_xywhn.numel() == 0:
        device = boxes_xywhn.device if isinstance(boxes_xywhn, torch.Tensor) else torch.device("cpu")
        return torch.zeros((0, SEGFALL_FEATURE_DIM), device=device)
    boxes = boxes_xywhn.float()
    confidence = conf.to(device=boxes.device, dtype=boxes.dtype).clamp(0, 1) if conf is not None else boxes.new_zeros(len(boxes))
    features = []
    for i, box in enumerate(boxes):
        poly = mask_polygons[i] if mask_polygons is not None and i < len(mask_polygons) else None
        features.append(torch.cat((box[:4].clamp(0, 1), confidence[i : i + 1], _polygon_mask_grid(poly, box, orig_shape))))
    return torch.stack(features) if features else boxes.new_zeros((0, SEGFALL_FEATURE_DIM))


def segment_result_features(result, device: torch.device | str | None = None) -> torch.Tensor:
    """Create raw fixed-size tensors from a segmentation Results object."""
    if result.boxes is None or result.masks is None or not len(result.boxes):
        return torch.zeros((0, SEGFALL_FEATURE_DIM), device=device or "cpu")
    target_device = torch.device(device) if device is not None else result.boxes.data.device
    boxes = result.boxes.xywhn.to(target_device)
    confidence = result.boxes.conf.to(target_device) if result.boxes.conf is not None else None
    return raw_segment_features(boxes, result.masks.xy, result.orig_shape, confidence)


def mean_segment_confidence(seq: torch.Tensor) -> float:
    """Return a simple validity score for segmentation sequences."""
    if seq.numel() == 0:
        return 0.0
    return float(seq[..., 4].mean().item())
