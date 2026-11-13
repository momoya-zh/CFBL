# -*- coding: utf-8 -*-
"""模型层公共参数与张量校验工具。"""
from __future__ import annotations

import torch

SUPPORTED_SCALES: tuple[int, ...] = (2, 4)


def validate_scale(scale: int) -> int:
    """校验放大倍率是否为受支持的整数（2 或 4）。"""
    if isinstance(scale, bool) or not isinstance(scale, int):
        raise TypeError(
            f"放大倍率 scale 必须为整数，实际类型为 {type(scale).__name__}"
        )
    if scale not in SUPPORTED_SCALES:
        raise ValueError(
            f"不支持的放大倍率: {scale}，仅支持 {SUPPORTED_SCALES}"
        )
    return scale


def validate_image_tensor(x: torch.Tensor) -> None:
    """校验模型输入张量是否符合 [B, 3, H, W]、float32 约定。"""
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"输入必须为 torch.Tensor，实际类型为 {type(x).__name__}"
        )
    if x.dtype != torch.float32:
        raise TypeError(
            f"输入张量数据类型必须为 torch.float32，实际为 {x.dtype}"
        )
    if x.ndim != 4:
        raise ValueError(
            f"输入张量必须为四维 [B, 3, H, W]，实际维度为 {x.ndim}（shape={tuple(x.shape)}）"
        )
    batch, channels, height, width = x.shape
    if channels != 3:
        raise ValueError(
            f"输入张量通道数必须为 3（RGB），实际为 {channels}（shape={tuple(x.shape)}）"
        )
    if batch <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            f"输入张量 Batch 与空间尺寸必须为正整数，实际 shape={tuple(x.shape)}"
        )
