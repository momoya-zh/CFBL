# -*- coding: utf-8 -*-
"""训练损失函数工厂模块。"""
from __future__ import annotations

import torch.nn as nn


def build_loss(name: str = "l1") -> nn.Module:
    """根据名称构建损失函数（阶段 5 冻结仅支持 L1）。"""
    if not isinstance(name, str):
        raise TypeError(f"损失名称必须为字符串，实际为 {type(name).__name__}")
    normalized = name.strip().lower()
    if normalized == "l1":
        return nn.L1Loss(reduction="mean")
    raise ValueError(f"不支持的损失函数: {name!r}，阶段 5 仅支持 'l1'")
