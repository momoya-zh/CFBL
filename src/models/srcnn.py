# -*- coding: utf-8 -*-
"""SRCNN（Super-Resolution Convolutional Neural Network）模型定义。

采用 RGB 三通道 9-5-5 结构，网络内部不执行空间上采样；
输入张量须为外部已完成 Pillow BICUBIC 预上采样的目标尺寸图像。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models._validation import validate_image_tensor, validate_scale


class SRCNN(nn.Module):
    """三通道 RGB SRCNN 网络。

    Args:
        scale: 目标放大倍率（``2`` 或 ``4``）。不改变卷积层拓扑，
            用于绑定并校验模型配置与推理预处理倍率，防止跨倍率权重误用。
    """

    upsample_inside: bool = False

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale: int = validate_scale(scale)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输出与输入空间尺寸相同的 RGB 重建张量。"""
        validate_image_tensor(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out
