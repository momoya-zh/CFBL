# -*- coding: utf-8 -*-
"""传统图像插值算法实现模块。

提供基于 OpenCV 的最近邻（Nearest Neighbor）、双线性（Bilinear）与
双三次（Bicubic）图像放大接口，作为后续深度学习超分辨率模型的传统基线，
并供最终 Gradio 推理界面复用。

通道约定：
    统一使用 RGB 通道顺序的 NumPy 数组（形状为 ``(H, W, 3)``）。
    输入图像不会被原地修改，返回图像保持相同的通道顺序与数据类型。
"""
from __future__ import annotations

import cv2
import numpy as np

SUPPORTED_SCALES: tuple[int, ...] = (2, 4)

INTERPOLATION_METHODS: dict[str, int] = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}

_SUPPORTED_DTYPES = (
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.int16),
    np.dtype(np.float32),
    np.dtype(np.float64),
)


def _validate_image(image: np.ndarray) -> None:
    """校验输入图像是否满足 RGB NumPy 数组规范。"""
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"输入图像必须为 numpy.ndarray，实际类型为 {type(image).__name__}"
        )
    if image.size == 0:
        raise ValueError("输入图像不能为空（size == 0）")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"输入图像必须为 RGB 三通道数组 (H, W, 3)，实际形状为 {image.shape}"
        )
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(
            f"输入图像宽高必须为正整数，实际形状为 {image.shape}"
        )
    if image.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"不支持的图像数据类型 {image.dtype}，支持的类型为: "
            + ", ".join(str(dt) for dt in _SUPPORTED_DTYPES)
        )


def _validate_scale(scale: int) -> int:
    """校验放大倍率是否合法（仅支持 2 或 4）。"""
    if isinstance(scale, bool) or not isinstance(scale, int):
        raise TypeError(
            f"放大倍率 scale 必须为整数，实际类型为 {type(scale).__name__}"
        )
    if scale not in SUPPORTED_SCALES:
        raise ValueError(
            f"不支持的放大倍率: {scale}，仅支持 {SUPPORTED_SCALES}"
        )
    return scale


def _validate_method(method: str) -> int:
    """校验插值算法名称并返回对应的 OpenCV 插值标志。"""
    if not isinstance(method, str):
        raise TypeError(
            f"插值算法 method 必须为字符串，实际类型为 {type(method).__name__}"
        )
    normalized = method.strip().lower()
    if normalized not in INTERPOLATION_METHODS:
        supported = ", ".join(INTERPOLATION_METHODS.keys())
        raise ValueError(
            f"不支持的插值算法: {method!r}，仅支持: {supported}"
        )
    return INTERPOLATION_METHODS[normalized]


def upscale_image(image: np.ndarray, scale: int, method: str) -> np.ndarray:
    """使用指定传统插值算法放大图像。

    Args:
        image: 输入图像，RGB 通道顺序的 NumPy 数组，形状为 ``(H, W, 3)``。
        scale: 放大倍率，支持 ``2`` 或 ``4``。
        method: 插值算法名称，支持 ``"nearest"``、``"bilinear"``、``"bicubic"``。

    Returns:
        放大后的 RGB 图像，形状为 ``(H * scale, W * scale, 3)``，
        数据类型与输入一致。不会修改调用者传入的原始图像。

    Raises:
        TypeError: 输入图像非 ``numpy.ndarray``、数据类型不支持，或参数类型错误。
        ValueError: 图像为空、非 RGB 三通道、倍率或算法名称不受支持。
    """
    _validate_image(image)
    validated_scale = _validate_scale(scale)
    interpolation_flag = _validate_method(method)

    height, width = image.shape[:2]
    target_width = width * validated_scale
    target_height = height * validated_scale

    # cv2.resize 按通道独立插值，不改变 RGB 通道顺序，也不修改输入数组
    upscaled = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=interpolation_flag,
    )
    return upscaled
