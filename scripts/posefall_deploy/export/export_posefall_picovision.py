#!/usr/bin/env python3
"""Rewrite the trained PoseFall Transformer into PicoVision's CHW/Conv form."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


WINDOW = 60
INPUT_DIM = 56
MODEL_DIM = 256
HEADS = 4
LAYERS = 3


class _PortableLayerNormFunction(torch.autograd.Function):
    """PicoVision LayerNorm ONNX node with an exact PyTorch reference forward."""

    @staticmethod
    def forward(ctx, x, weight, bias, axis, epsilon):
        del ctx
        dim = axis if axis >= 0 else x.ndim + axis
        mean = x.mean(dim=dim, keepdim=True)
        variance = (x - mean).square().mean(dim=dim, keepdim=True)
        shape = [1] * x.ndim
        shape[dim] = weight.numel()
        return (x - mean) * torch.rsqrt(variance + epsilon) * weight.reshape(shape) + bias.reshape(shape)

    @staticmethod
    def symbolic(g, x, weight, bias, axis, epsilon):
        return g.op(
            "custom_domain::LayerNorm",
            x,
            weight,
            bias,
            dim_i=int(axis),
            epsilon_f=float(epsilon),
        ).setType(x.type())


class _PortableGeluFunction(torch.autograd.Function):
    """PicoVision Gelu ONNX node with the original model's exact GELU forward."""

    @staticmethod
    def forward(ctx, x, approximate):
        del ctx, approximate
        return F.gelu(x, approximate="none")

    @staticmethod
    def symbolic(g, x, approximate):
        return g.op("custom_domain::Gelu", x, approximate_s=str(approximate)).setType(x.type())


class PortableLayerNorm(nn.Module):
    def __init__(self, normalized_shape, axis=-3, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.axis = axis
        self.eps = eps

    def forward(self, x):
        return _PortableLayerNormFunction.apply(x, self.weight, self.bias, self.axis, self.eps)


class PortableGELU(nn.Module):
    def __init__(self, approximate="better_approximation"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        return _PortableGeluFunction.apply(x, self.approximate)


class PortableMultiheadConvAttention(nn.Module):
    """Source-equivalent implementation of PicoVision MultiheadConvAttention."""

    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.q = nn.Conv2d(embed_dim, embed_dim, 1)
        self.k = nn.Conv2d(embed_dim, embed_dim, 1)
        self.v = nn.Conv2d(embed_dim, embed_dim, 1)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.scale = (embed_dim // n_heads) ** -0.5
        self.n_heads = n_heads

    def forward(self, q, k, v, mask=None):
        q, k, v = self.q(q), self.k(k), self.v(v)
        q = q.reshape(q.shape[0], self.n_heads, -1, q.shape[-1])
        k = k.reshape(k.shape[0], self.n_heads, -1, k.shape[-1])
        v = v.reshape(v.shape[0], self.n_heads, -1, v.shape[-1])
        scores = (k.transpose(-1, -2) @ q) * self.scale
        if mask is not None:
            scores = scores + mask.float() * -1000
        attention = scores.softmax(-2)
        output = (v @ attention).reshape(q.shape[0], -1, 1, q.shape[-1])
        return self.out_proj(output)


class PortablePicoNN:
    LayerNorm = PortableLayerNorm
    GELU = PortableGELU
    MultiheadConvAttention = PortableMultiheadConvAttention


def extract(args):
    import numpy as np

    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    head = checkpoint["model"].posefall_head.float().eval()
    reference = np.load(args.reference)
    features = torch.from_numpy(reference["features"].astype(np.float32))
    probabilities = []
    with torch.no_grad():
        for sample in features:
            probabilities.append(head(sample.unsqueeze(0)).sigmoid().cpu())
    payload = {
        "state_dict": head.state_dict(),
        "pos_embed": head.pos_embed.detach().cpu(),
        "reference_features": features,
        "reference_probabilities": torch.cat(probabilities),
    }
    args.intermediate.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.intermediate)
    print(f"extracted Transformer tensors to {args.intermediate}")


class PicoEncoderLayer(nn.Module):
    def __init__(self, pico_nn):
        super().__init__()
        self.attention = pico_nn.MultiheadConvAttention(MODEL_DIM, HEADS)
        self.norm1 = pico_nn.LayerNorm(MODEL_DIM, axis=-3)
        self.linear1 = nn.Conv2d(MODEL_DIM, MODEL_DIM, 1)
        self.activation = pico_nn.GELU("better_approximation")
        self.linear2 = nn.Conv2d(MODEL_DIM, MODEL_DIM, 1)
        self.norm2 = pico_nn.LayerNorm(MODEL_DIM, axis=-3)

    def forward(self, x):
        x = self.norm1(x + self.attention(x, x, x))
        x = self.norm2(x + self.linear2(self.activation(self.linear1(x))))
        return x


class PicoPoseFallTransformer(nn.Module):
    def __init__(self, pico_nn, pos_embed):
        super().__init__()
        self.input_norm = pico_nn.LayerNorm(INPUT_DIM, axis=-3)
        self.input_proj = nn.Conv2d(INPUT_DIM, MODEL_DIM, 1)
        self.input_activation = pico_nn.GELU("better_approximation")
        self.register_buffer("pos_embed", pos_embed.transpose(1, 2).unsqueeze(2).contiguous())
        self.layers = nn.ModuleList(PicoEncoderLayer(pico_nn) for _ in range(LAYERS))
        self.pool1 = nn.Conv2d(MODEL_DIM, MODEL_DIM // 2, 1)
        self.pool2 = nn.Conv2d(MODEL_DIM // 2, 1, 1)
        self.head1 = nn.Conv2d(MODEL_DIM, MODEL_DIM // 2, 1)
        self.head_activation = pico_nn.GELU("better_approximation")
        self.head2 = nn.Conv2d(MODEL_DIM // 2, 1, 1)

    def forward(self, features):
        x = features.transpose(1, 2).unsqueeze(2)
        x = self.input_activation(self.input_proj(self.input_norm(x))) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        scores = self.pool2(torch.tanh(self.pool1(x))).softmax(-1)
        summary = (x * scores).sum(-1, keepdim=True)
        return self.head2(self.head_activation(self.head1(summary))).flatten().sigmoid()


def copy_affine(target, weight, bias):
    with torch.no_grad():
        target.weight.copy_(weight)
        target.bias.copy_(bias)


def copy_conv(target, weight, bias):
    copy_affine(target, weight.unsqueeze(-1).unsqueeze(-1), bias)


def load_original_weights(model, state):
    copy_affine(model.input_norm, state["input_proj.0.weight"], state["input_proj.0.bias"])
    copy_conv(model.input_proj, state["input_proj.1.weight"], state["input_proj.1.bias"])
    for index, layer in enumerate(model.layers):
        prefix = f"encoder.layers.{index}."
        q_weight, k_weight, v_weight = state[prefix + "self_attn.in_proj_weight"].chunk(3, 0)
        q_bias, k_bias, v_bias = state[prefix + "self_attn.in_proj_bias"].chunk(3, 0)
        copy_conv(layer.attention.q, q_weight, q_bias)
        copy_conv(layer.attention.k, k_weight, k_bias)
        copy_conv(layer.attention.v, v_weight, v_bias)
        copy_conv(layer.attention.out_proj, state[prefix + "self_attn.out_proj.weight"],
                  state[prefix + "self_attn.out_proj.bias"])
        copy_conv(layer.linear1, state[prefix + "linear1.weight"], state[prefix + "linear1.bias"])
        copy_conv(layer.linear2, state[prefix + "linear2.weight"], state[prefix + "linear2.bias"])
        copy_affine(layer.norm1, state[prefix + "norm1.weight"], state[prefix + "norm1.bias"])
        copy_affine(layer.norm2, state[prefix + "norm2.weight"], state[prefix + "norm2.bias"])
    copy_conv(model.pool1, state["pool_score.0.weight"], state["pool_score.0.bias"])
    copy_conv(model.pool2, state["pool_score.2.weight"], state["pool_score.2.bias"])
    copy_conv(model.head1, state["head.0.weight"], state["head.0.bias"])
    copy_conv(model.head2, state["head.3.weight"], state["head.3.bias"])


def build(args):
    if args.portable:
        pico_nn = PortablePicoNN
    else:
        from picovision import nn as pico_nn

    payload = torch.load(args.intermediate, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PicoPoseFallTransformer(pico_nn, payload["pos_embed"])
    load_original_weights(model, payload["state_dict"])
    model.eval().to(device)
    features = payload["reference_features"].to(device)
    expected = payload["reference_probabilities"].cpu()
    actual = []
    with torch.no_grad():
        for sample in features:
            actual.append(model(sample.unsqueeze(0)).cpu())
    actual = torch.cat(actual)
    difference = (actual - expected).abs()
    print(f"reference max_abs={difference.max().item():.8f} mean_abs={difference.mean().item():.8f} "
          f"class_agreement={((actual >= .5) == (expected >= .5)).float().mean().item():.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    example = features[:1]
    torch.onnx.export(model, example, args.output, input_names=["features"], output_names=["fall_prob"],
                      opset_version=13, do_constant_folding=True, dynamo=False)
    print(f"exported {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--intermediate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--portable", action="store_true", help="Emit official PicoVision custom ONNX nodes without loading its binary extension.")
    args = parser.parse_args()
    if args.extract:
        if args.weights is None or args.reference is None:
            parser.error("--extract requires --weights and --reference")
        extract(args)
    else:
        if args.output is None:
            parser.error("build mode requires --output")
        build(args)


if __name__ == "__main__":
    main()
