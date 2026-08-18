"""Multi-scale deformable attention without mmcv.

A drop-in replacement for `mmcv.ops.MultiScaleDeformableAttention` that keeps
mmcv's parameter names, shapes and forward semantics so pretrained Track-On
checkpoints load and evaluate identically.

The inner sampling op is dispatched at runtime:

  1. the prebuilt fused CUDA kernel that `transformers` pulls from the HF Hub
     (`kernels-community/deformable-detr`) -- the same op mmcv compiles, but
     with no nvcc / CUDA_HOME / build step, and
  2. a pure-PyTorch `grid_sample` fallback otherwise.

The fused path is checked against the fallback numerically on first use, so a
signature or memory-layout mismatch downgrades to the fallback instead of
silently producing wrong features. Override with LOST_TOOLS_MSDA=fused|torch|auto.
"""

import math
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# pure-PyTorch reference implementation
# ---------------------------------------------------------------------------

def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations, attention_weights):
    """
    :args value: (B, num_value, num_heads, head_dim)
    :args value_spatial_shapes: (num_levels, 2) as (H, W)
    :args sampling_locations: (B, num_queries, num_heads, num_levels, num_points, 2)
    :args attention_weights: (B, num_queries, num_heads, num_levels, num_points)

    Returns (B, num_queries, num_heads * head_dim)
    """
    bs, _, num_heads, head_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    shapes = [(int(h), int(w)) for h, w in value_spatial_shapes]
    # grid_sample expects coordinates in [-1, 1]
    sampling_grids = 2 * sampling_locations - 1

    # Single-level fast path: skips the per-level split, the stack and one full
    # copy of the sampled features. Hit by every Extractor in the ViT adapter,
    # which is constructed with n_levels=1.
    if num_levels == 1:
        h, w = shapes[0]
        value_l = (value
                   .flatten(2)
                   .transpose(1, 2)
                   .reshape(bs * num_heads, head_dim, h, w))
        grid_l = sampling_grids[:, :, :, 0].transpose(1, 2).flatten(0, 1)
        sampled = F.grid_sample(value_l, grid_l, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
        weights = attention_weights.transpose(1, 2).reshape(
            bs * num_heads, 1, num_queries, num_points)
        output = ((sampled * weights)
                  .sum(-1)
                  .view(bs, num_heads * head_dim, num_queries))
        return output.transpose(1, 2).contiguous()

    value_list = value.split([h * w for h, w in shapes], dim=1)

    sampling_value_list = []
    for level, (h, w) in enumerate(shapes):
        # (B, H*W, heads, head_dim) -> (B*heads, head_dim, H, W)
        value_l = (value_list[level]
                   .flatten(2)
                   .transpose(1, 2)
                   .reshape(bs * num_heads, head_dim, h, w))
        # (B, queries, heads, points, 2) -> (B*heads, queries, points, 2)
        grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_list.append(
            F.grid_sample(value_l, grid_l, mode='bilinear',
                          padding_mode='zeros', align_corners=False))

    # (B, queries, heads, levels, points) -> (B*heads, 1, queries, levels*points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)

    output = ((torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
              .sum(-1)
              .view(bs, num_heads * head_dim, num_queries))

    return output.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# fused-kernel dispatch
# ---------------------------------------------------------------------------

_MODE = os.environ.get('LOST_TOOLS_MSDA', 'auto').lower()
_STATE = {'resolved': False, 'impl': None}


def _load_hub_kernel():
    """Return a callable wrapping the Hub fused kernel, or None."""
    try:
        from transformers.models.deformable_detr.modeling_deformable_detr import (
            MultiScaleDeformableAttention as _HFLayer,
        )
    except Exception:
        return None

    try:
        layer = _HFLayer()
    except Exception:
        return None

    # transformers only swaps in the Hub kernel once the layer is kernelized;
    # without the `kernels` package it stays on its own grid_sample path, which
    # is no faster than ours -- so treat that as "no fused impl available".
    try:
        from kernels import kernelize
    except Exception:
        return None

    try:
        layer = kernelize(layer, device='cuda')
    except Exception:
        return None

    def _call(value, spatial_shapes, level_start_index, sampling_locations,
              attention_weights, im2col_step):
        shapes_list = [(int(h), int(w)) for h, w in spatial_shapes]
        return layer(value, spatial_shapes, shapes_list, level_start_index,
                     sampling_locations, attention_weights, im2col_step)

    return _call


def _fused_matches_reference(impl, device):
    """One-off numerical check of the fused kernel against the reference."""
    torch.manual_seed(0)
    B, H, L, P, D = 1, 4, 3, 4, 16
    shapes = torch.tensor([[8, 8], [4, 4], [2, 2]], device=device)
    starts = torch.tensor([0, 64, 80], device=device)
    S = int((shapes[:, 0] * shapes[:, 1]).sum())
    Q = 7

    value = torch.randn(B, S, H, D, device=device)
    loc = torch.rand(B, Q, H, L, P, 2, device=device)
    attn = torch.rand(B, Q, H, L, P, device=device)
    attn = attn / attn.sum(-1, keepdim=True)

    try:
        got = impl(value, shapes, starts, loc, attn, 64)
    except Exception as exc:
        warnings.warn(f'[msda] fused kernel call failed ({exc}); '
                      f'using PyTorch fallback')
        return False

    want = multi_scale_deformable_attn_pytorch(value, shapes, loc, attn)
    if got.shape != want.shape:
        warnings.warn(f'[msda] fused kernel shape {tuple(got.shape)} != '
                      f'reference {tuple(want.shape)}; using PyTorch fallback')
        return False
    if not torch.allclose(got, want, atol=1e-4, rtol=1e-3):
        warnings.warn('[msda] fused kernel disagrees numerically with the '
                      'reference implementation; using PyTorch fallback')
        return False
    return True


def _get_impl(device):
    """Resolve the sampling implementation once, then cache it."""
    if _STATE['resolved']:
        return _STATE['impl']

    impl = None
    if _MODE != 'torch' and device.type == 'cuda':
        impl = _load_hub_kernel()
        if impl is not None and _MODE != 'fused':
            if not _fused_matches_reference(impl, device):
                impl = None

    if _MODE == 'fused' and impl is None:
        raise RuntimeError(
            'LOST_TOOLS_MSDA=fused but the Hub deformable-detr kernel could not '
            'be loaded. Install it with `pip install kernels`, or unset the '
            'variable to fall back to the PyTorch implementation.')

    if impl is not None:
        print('[msda] using fused Hub kernel (kernels-community/deformable-detr)')

    _STATE['impl'] = impl
    _STATE['resolved'] = True
    return impl


# ---------------------------------------------------------------------------
# module
# ---------------------------------------------------------------------------

class MultiScaleDeformableAttention(nn.Module):
    """API-compatible with mmcv.ops.MultiScaleDeformableAttention.

    Parameter names (`sampling_offsets`, `attention_weights`, `value_proj`,
    `output_proj`) match mmcv's so state_dicts are interchangeable.
    """

    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=4,
                 im2col_step=64, dropout=0.1, batch_first=False, norm_cfg=None,
                 init_cfg=None, value_proj_ratio=1.0):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f'embed_dims must be divisible by num_heads, '
                f'but got {embed_dims} and {num_heads}')

        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points)

        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Mirrors mmcv's grid-style initialisation of the sampling offsets."""
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, key=None, value=None, identity=None, query_pos=None,
                key_padding_mask=None, reference_points=None, spatial_shapes=None,
                level_start_index=None, **kwargs):
        # `key` is accepted for API parity; mmcv ignores it too.
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets
                / offset_normalizer[None, None, None, :, None, :])
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points
                * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                f'Last dim of reference_points must be 2 or 4, '
                f'but got {reference_points.shape[-1]}')

        impl = _get_impl(value.device)
        if impl is not None:
            if level_start_index is None:
                level_start_index = torch.cat([
                    spatial_shapes.new_zeros(1),
                    (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1],
                ])
            output = impl(value, spatial_shapes, level_start_index,
                          sampling_locations, attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        # mmcv folds the residual in here; Track-On then adds its own on top.
        return self.dropout(output) + identity
