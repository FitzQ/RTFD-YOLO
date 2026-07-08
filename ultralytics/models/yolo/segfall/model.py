# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
from torch import nn

SEGFALL_FEATURE_DIM = 24
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
        self.head = nn.Sequential(nn.Linear(d_model * 4, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

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
        return self._clip_logits(clips).amax(dim=0)

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
        mean_pool = x.mean(dim=1)
        max_pool = x.amax(dim=1)
        tail = x[:, -self.summary_tail :]
        tail_mean = tail.mean(dim=1)
        tail_max = tail.amax(dim=1)
        summary = torch.cat((mean_pool, max_pool, tail_mean, tail_max), dim=1)
        return self.head(summary).squeeze(-1)


def pad_or_trim(seq: torch.Tensor, window: int) -> torch.Tensor:
    """Return a fixed-length sequence, padding the beginning with the first frame when needed."""
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


def _previous(previous: list[dict[str, float] | None] | None, index: int, key: str, default: float) -> float:
    if previous is None or index >= len(previous) or previous[index] is None:
        return default
    return float(previous[index].get(key, default))


def _mask_stats_from_polygon(poly, height: float, width: float, fallback_box: torch.Tensor, device, dtype) -> tuple[torch.Tensor, ...]:
    if poly is None or len(poly) == 0:
        cx, cy, bw, bh = fallback_box
        x1, y1 = cx - bw * 0.5, cy - bh * 0.5
        x2, y2 = cx + bw * 0.5, cy + bh * 0.5
        area = bw * bh
        conf = torch.tensor(0.0, device=device, dtype=dtype)
        return cx, cy, bw, bh, area, x1, y1, x2, y2, conf

    pts = torch.as_tensor(poly, device=device, dtype=dtype)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return _mask_stats_from_polygon(None, height, width, fallback_box, device, dtype)
    xs = (pts[:, 0] / width).clamp(0, 1)
    ys = (pts[:, 1] / height).clamp(0, 1)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    mw = (x2 - x1).clamp_min(1e-6)
    mh = (y2 - y1).clamp_min(1e-6)
    signed = 0.5 * torch.abs(torch.dot(xs, torch.roll(ys, -1)) - torch.dot(ys, torch.roll(xs, -1)))
    area = signed.clamp(0, 1)
    cx = xs.mean().clamp(0, 1)
    cy = ys.mean().clamp(0, 1)
    conf = torch.tensor(1.0, device=device, dtype=dtype)
    return cx, cy, mw, mh, area, x1, y1, x2, y2, conf


def normalize_segments(
    boxes_xywhn: torch.Tensor,
    mask_polygons,
    orig_shape: tuple[int, int] | torch.Tensor,
    conf: torch.Tensor | None = None,
    previous_state: list[dict[str, float] | None] | None = None,
    feature_dim: int = SEGFALL_FEATURE_DIM,
) -> torch.Tensor:
    """Convert tracked segmentation instances to 24-dim bbox/mask motion features."""
    if boxes_xywhn is None or boxes_xywhn.numel() == 0:
        device = boxes_xywhn.device if isinstance(boxes_xywhn, torch.Tensor) else torch.device("cpu")
        return torch.zeros((0, feature_dim), device=device)

    boxes = boxes_xywhn.float()
    height, width = _shape_hw(orig_shape)
    if conf is not None:
        conf = conf.to(device=boxes.device, dtype=boxes.dtype).clamp(0, 1)
    features = []
    for i, box in enumerate(boxes):
        bx, by, bw, bh = box[:4].clamp(0, 1)
        bw = bw.clamp_min(1e-6)
        bh = bh.clamp_min(1e-6)
        bbox_area = (bw * bh).clamp(0, 1)
        bbox_aspect = (bw / bh).clamp(0, 10)
        poly = mask_polygons[i] if mask_polygons is not None and i < len(mask_polygons) else None
        mcx, mcy, mw, mh, mask_area, left_x, top_y, right_x, bottom_y, _mask_conf = _mask_stats_from_polygon(
            poly, height, width, torch.stack((bx, by, bw, bh)), boxes.device, boxes.dtype
        )
        mask_aspect = (mw / mh).clamp(0, 10)
        fill_ratio = (mask_area / bbox_area.clamp_min(1e-6)).clamp(0, 1)
        prev_bx = _previous(previous_state, i, "bbox_cx", float(bx))
        prev_by = _previous(previous_state, i, "bbox_cy", float(by))
        prev_mcx = _previous(previous_state, i, "mask_cx", float(mcx))
        prev_mcy = _previous(previous_state, i, "mask_cy", float(mcy))
        prev_bottom = _previous(previous_state, i, "bottom_y", float(bottom_y))
        prev_area = _previous(previous_state, i, "mask_area", float(mask_area))
        prev_aspect = _previous(previous_state, i, "mask_aspect", float(mask_aspect))

        feat = torch.stack(
            (
                bx,
                by,
                (bx - bx.new_tensor(prev_bx)).clamp(-1, 1),
                (by - by.new_tensor(prev_by)).clamp(-1, 1),
                bw,
                bh,
                bbox_area,
                bbox_aspect,
                mcx,
                mcy,
                (mcx - mcx.new_tensor(prev_mcx)).clamp(-1, 1),
                (mcy - mcy.new_tensor(prev_mcy)).clamp(-1, 1),
                mw.clamp(0, 1),
                mh.clamp(0, 1),
                mask_area.clamp(0, 1),
                mask_aspect,
                fill_ratio,
                top_y.clamp(0, 1),
                bottom_y.clamp(0, 1),
                left_x.clamp(0, 1),
                right_x.clamp(0, 1),
                (bottom_y - bottom_y.new_tensor(prev_bottom)).clamp(-1, 1),
                (mask_area - mask_area.new_tensor(prev_area)).clamp(-1, 1),
                (mask_aspect - mask_aspect.new_tensor(prev_aspect)).clamp(-10, 10),
            )
        )
        if feature_dim > SEGFALL_FEATURE_DIM:
            feat = torch.cat((feat, feat.new_zeros(feature_dim - SEGFALL_FEATURE_DIM)))
        elif feature_dim < SEGFALL_FEATURE_DIM:
            feat = feat[:feature_dim]
        features.append(feat)
    return torch.stack(features) if features else boxes.new_zeros((0, feature_dim))


def segment_state(feat: torch.Tensor) -> dict[str, float]:
    """Return the values needed to compute the next frame's velocity features."""
    return {
        "bbox_cx": float(feat[0]),
        "bbox_cy": float(feat[1]),
        "mask_cx": float(feat[8]),
        "mask_cy": float(feat[9]),
        "mask_area": float(feat[14]),
        "mask_aspect": float(feat[15]),
        "bottom_y": float(feat[18]),
    }


def mean_segment_confidence(seq: torch.Tensor) -> float:
    """Return a simple validity score for segmentation sequences."""
    if seq.numel() == 0:
        return 0.0
    return float((seq[..., 4] > 0).float().mean().item())


def load_segfall_head(weights: str | None, device: torch.device | str, input_dim: int = SEGFALL_FEATURE_DIM, window: int = 60):
    """Load a segfall head checkpoint, or return None when no weights are provided."""
    if not weights:
        return None, {"window": window, "input_dim": input_dim}
    ckpt = torch.load(weights, map_location=device)
    config = ckpt.get("config", {})
    model = SegFallTransformer(
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
            f"SegFall head weights at {weights!r} are incompatible with the current segfall feature head. "
            "Retrain with `yolo segfall train ...` to generate a new checkpoint."
        ) from e
    model.to(device).eval()
    return model, config
