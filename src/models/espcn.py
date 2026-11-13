# -*- coding: utf-8 -*-
"""ESPCN（Efficient Sub-Pixel Convolutional Neural Network）模型定义。

直接在低分辨率空间提取特征，末层输出 ``3 * scale^2`` 通道后通过
``nn.PixelShuffle(scale)`` 重排为目标高分辨率 RGB 张量。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models._validation import validate_image_tensor, validate_scale


class ESPCN(nn.Module):
    """三通道 RGB ESPCN 网络。

    Args:
        scale: 放大倍率，仅支持 ``2`` 或 ``4``。
    """

    upsample_inside: bool = True

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale: int = validate_scale(scale)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3 * (self.scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(self.scale)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输入 LR 张量并输出放大 ``scale`` 倍的 RGB 张量。"""
        validate_image_tensor(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.pixel_shuffle(self.conv3(out))
        return out
