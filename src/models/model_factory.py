# -*- coding: utf-8 -*-
"""统一模型工厂模块。"""
from __future__ import annotations

import torch.nn as nn

from src.models._validation import SUPPORTED_SCALES, validate_scale
from src.models.espcn import ESPCN
from src.models.srcnn import SRCNN

SUPPORTED_MODELS: tuple[str, ...] = ("srcnn", "espcn")


def create_model(model_name: str, scale: int) -> nn.Module:
    """创建指定模型与倍率的独立网络实例。

    Args:
        model_name: 模型名称，支持 ``"srcnn"`` 或 ``"espcn"``（忽略大小写与首尾空白）。
        scale: 放大倍率，支持 ``2`` 或 ``4``。

    Returns:
        新初始化的 PyTorch 模型实例（每次调用返回独立参数存储）。

    Raises:
        TypeError: ``model_name`` 非字符串或 ``scale`` 非整数。
        ValueError: 不支持的模型名称或放大倍率。
    """
    if not isinstance(model_name, str):
        raise TypeError(
            f"模型名称 model_name 必须为字符串，实际类型为 {type(model_name).__name__}"
        )
    normalized_name = model_name.strip().lower()
    validated_scale = validate_scale(scale)

    if normalized_name == "srcnn":
        return SRCNN(scale=validated_scale)
    if normalized_name == "espcn":
        return ESPCN(scale=validated_scale)

    supported = ", ".join(SUPPORTED_MODELS)
    raise ValueError(
        f"不支持的模型名称: {model_name!r}，仅支持: {supported}"
    )


__all__ = [
    "SUPPORTED_MODELS",
    "SUPPORTED_SCALES",
    "create_model",
]
