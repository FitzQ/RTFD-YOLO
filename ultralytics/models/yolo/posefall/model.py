# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
from torch import nn

POSEFALL_KEYPOINT_DIM = 51
POSEFALL_FEATURE_DIM = 56
POSEFALL_CENTER_Y_INDEX = POSEFALL_KEYPOINT_DIM


def sinusoidal_positional_encoding(window: int, d_model: int) -> torch.Tensor:
    """Build sinusoidal positional encodings shaped [1, window, d_model]."""
    position = torch.arange(window, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe = torch.zeros(window, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class PoseFallTransformer(nn.Module):
    """Transformer head for pose-based video-level fall classification with internal sliding-window pooling."""

    def __init__(
        self,
        input_dim: int = 51,
        num_classes: int = 1,
        window: int = 60,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        summary_tail: int = 60,
        stride: int = 15,
    ):
        super().__init__()
        self.window = window
        self.input_dim = input_dim
        self.d_model = d_model
        self.summary_tail = summary_tail
        self.stride = stride
        self.input_proj = nn.Linear(input_dim, d_model)
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
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes))
        # self.head = nn.Sequential(nn.Linear(d_model * 4, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Classify full sequences by max-pooling logits over fixed-length sliding windows."""
        if isinstance(x, (list, tuple)):
            return torch.stack([self._sequence_logit(seq) for seq in x])
        if x.ndim == 2:
            return self._sequence_logit(x).unsqueeze(0)
        if x.ndim == 3 and x.shape[1] == self.window:
            return self._clip_logits(x)
        if x.ndim == 3:
            return torch.stack([self._sequence_logit(seq) for seq in x])
        raise ValueError(f"Expected x as [T,D], [B,T,D], or list of [T,D] tensors, got shape={tuple(x.shape)}")

    def _sequence_logit(self, seq: torch.Tensor) -> torch.Tensor:
        """Return one video-level logit as max clip logit over sliding windows."""
        clips = self._sliding_clips(seq)
        # return self._clip_logits(clips).amax(dim=0)
        return self._clip_logits(clips).topk(3 if clips.shape[0] > 3 else clips.shape[0]).values.mean()

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
        x = self.input_proj(x) * (self.d_model**0.5)
        x = self.encoder(x + pos_embed)
        # mean_pool = x.mean(dim=1)
        # max_pool = x.amax(dim=1)
        # tail = x[:, -self.summary_tail :]
        # tail_mean = tail.mean(dim=1)
        # tail_max = tail.amax(dim=1)
        # summary = torch.cat((mean_pool, max_pool, tail_mean, tail_max), dim=1)
        # return self.head(summary).squeeze(-1)
        # Use attention-based pooling instead of mean/max pooling
        pool_weights = self.pool_score(x).squeeze(-1)
        pool_weights = torch.softmax(pool_weights, dim=1)
        summary = torch.sum(x * pool_weights.unsqueeze(-1), dim=1)
        return self.head(summary).squeeze(-1)


def pad_or_trim(seq: torch.Tensor, window: int) -> torch.Tensor:
    """Return a fixed-length sequence, padding the end with the last frame when needed."""
    if seq.shape[0] >= window:
        return seq[-window:]
    pad = seq[:1].repeat(window - seq.shape[0], 1)
    return torch.cat([pad, seq], dim=0)


def _shape_hw(orig_shape: tuple[int, int] | torch.Tensor) -> tuple[float, float]:
    if isinstance(orig_shape, torch.Tensor):
        height, width = orig_shape.tolist()[:2]
    else:
        height, width = orig_shape[:2]
    return max(float(height), 1.0), max(float(width), 1.0)


def _box_from_keypoints(y_norm: torch.Tensor, x_norm: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    """Build normalized [cx, cy, w, h] from visible keypoints when detector boxes are unavailable."""
    valid = conf > 0.05
    if valid.any():
        xs = x_norm[valid]
        ys = y_norm[valid]
    else:
        xs = x_norm
        ys = y_norm
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1).clamp_min(1e-6), (y2 - y1).clamp_min(1e-6)))


def normalize_keypoints(
    kpts: torch.Tensor,
    orig_shape: tuple[int, int] | torch.Tensor,
    boxes_xywhn: torch.Tensor | None = None,
    previous_center_y: torch.Tensor | list[float | None] | None = None,
    feature_dim: int = POSEFALL_FEATURE_DIM,
) -> torch.Tensor:
    """Convert [N, 17, 3] keypoints to body-relative fall features.

    The first 51 values are [y_body_rel, x_body_rel, confidence] for 17 keypoints.
    The extended 56-dim format appends [center_y, center_vy, bbox_w, bbox_h, bbox_aspect].
    """
    out = kpts.clone().float()
    height, width = _shape_hw(orig_shape)
    x_norm = out[..., 0] / width
    y_norm = out[..., 1] / height
    conf = out[..., 2].clamp(0, 1)
    if boxes_xywhn is not None:
        boxes_xywhn = boxes_xywhn.to(device=out.device, dtype=out.dtype)

    features = []
    for i in range(out.shape[0]):
        if boxes_xywhn is not None and i < boxes_xywhn.shape[0]:
            box = boxes_xywhn[i, :4].clamp(min=0)
        else:
            box = _box_from_keypoints(y_norm[i], x_norm[i], conf[i])
        cx, cy, bw, bh = box
        bw = bw.clamp_min(1e-6)
        bh = bh.clamp_min(1e-6)
        scale = torch.maximum(bw, bh).clamp_min(1e-6)
        rel = torch.stack(((y_norm[i] - cy) / scale, (x_norm[i] - cx) / scale, conf[i]), dim=-1).flatten()
        if feature_dim == POSEFALL_KEYPOINT_DIM:
            features.append(rel)
            continue

        prev_cy = None
        if previous_center_y is not None:
            prev_cy = previous_center_y[i] if isinstance(previous_center_y, list) else previous_center_y[i]
        center_vy = cy.new_tensor(0.0) if prev_cy is None else cy - cy.new_tensor(float(prev_cy))
        extras = torch.stack((cy.clamp(0, 1), center_vy.clamp(-1, 1), bw.clamp(0, 1), bh.clamp(0, 1), (bw / bh).clamp(0, 10)))
        features.append(torch.cat((rel, extras)))
    return torch.stack(features) if features else out.new_zeros((0, feature_dim))


def mean_keypoint_confidence(seq: torch.Tensor) -> float:
    """Return mean keypoint confidence from flattened [y, x, conf] sequences."""
    if seq.numel() == 0:
        return 0.0
    return float(seq[..., :POSEFALL_KEYPOINT_DIM].reshape(-1, 17, 3)[..., 2].mean().item())


def load_posefall_head(weights: str | None, device: torch.device | str, input_dim: int = POSEFALL_FEATURE_DIM, window: int = 60):
    """Load a posefall head checkpoint, or return None when no weights are provided."""
    if not weights:
        return None, {"window": window, "input_dim": input_dim}
    ckpt = torch.load(weights, map_location=device)
    config = ckpt.get("config", {})
    model = PoseFallTransformer(
        input_dim=int(config.get("input_dim", input_dim)),
        num_classes=int(config.get("num_classes", 1)),
        window=int(config.get("window") or window),
        d_model=int(config.get("d_model", 256)),
        nhead=int(config.get("nhead", 4)),
        num_layers=int(config.get("num_layers", 3)),
        dim_feedforward=int(config.get("dim_feedforward", 256)),
        dropout=float(config.get("dropout", 0.1)),
        summary_tail=int(config.get("summary_tail", 60)),
        stride=int(config.get("stride", 15)),
    )
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        raise RuntimeError(
            f"PoseFall head weights at {weights!r} are incompatible with the current posefall feature head. "
            "Retrain with `yolo posefall train ...` to generate a new checkpoint."
        ) from e
    model.to(device).eval()
    return model, config
