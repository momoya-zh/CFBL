# -*- coding: utf-8 -*-
"""基于 Pillow BICUBIC 的统一重采样模块。

根据阶段 4 冻结协议：
- 下采样与 SRCNN 预上采样统一使用 ``PIL.Image.Resampling.BICUBIC``
  （Catmull-Rom 核，a = -0.5）。
- 通道约定统一为 RGB 三通道 ``(H, W, 3)``，支持 ``uint8`` 与 ``float32``。
- 本模块供推理层（``src/inference/predictor.py``）与后续阶段 5 数据管线共用，
  确保训练预处理与推理预处理完全一致。
"""
from __future__ import annotations

import numpy as np
from PIL import Image

SUPPORTED_RESAMPLE_SCALES: tuple[int, ...] = (2, 4)


def bicubic_upsample_pillow(rgb: np.ndarray, scale: int) -> np.ndarray:
    """使用 Pillow BICUBIC 将 RGB 图像按整数倍率放大。

    Args:
        rgb: 输入 RGB 图像数组，形状为 ``(H, W, 3)``，支持 ``uint8`` 或 ``float32``。
        scale: 放大倍率，仅支持 ``2`` 或 ``4``。

    Returns:
        放大后的 RGB 图像数组，形状为 ``(H * scale, W * scale, 3)``，
        ``dtype`` 与输入保持一致。不会修改原始输入数组。

    Raises:
        TypeError: 输入非 ``numpy.ndarray``、``dtype`` 不支持或 ``scale`` 非整数。
        ValueError: 图像为空、非 ``(H, W, 3)`` 或 ``scale`` 不受支持。
    """
    if not isinstance(rgb, np.ndarray):
        raise TypeError(
            f"输入图像必须为 numpy.ndarray，实际类型为 {type(rgb).__name__}"
        )
    if rgb.size == 0:
        raise ValueError("输入图像不能为空（size == 0）")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"输入图像必须为 RGB 三通道数组 (H, W, 3)，实际形状为 {rgb.shape}"
        )
    if isinstance(scale, bool) or not isinstance(scale, int):
        raise TypeError(
            f"放大倍率 scale 必须为整数，实际类型为 {type(scale).__name__}"
        )
    if scale not in SUPPORTED_RESAMPLE_SCALES:
        raise ValueError(
            f"不支持的放大倍率: {scale}，仅支持 {SUPPORTED_RESAMPLE_SCALES}"
        )

    height, width = rgb.shape[:2]
    target_size = (width * scale, height * scale)

    if rgb.dtype == np.uint8:
        upsampled = Image.fromarray(rgb).resize(
            target_size, resample=Image.Resampling.BICUBIC
        )
        return np.asarray(upsampled, dtype=np.uint8)

    if rgb.dtype == np.float32:
        # 逐通道以 Pillow 'F' 模式执行精确 float32 双三次插值，避免 uint8 量化损失
        channels = [
            np.asarray(
                Image.fromarray(rgb[:, :, ch], mode="F").resize(
                    target_size, resample=Image.Resampling.BICUBIC
                ),
                dtype=np.float32,
            )
            for ch in range(3)
        ]
        return np.stack(channels, axis=2)

    raise TypeError(
        f"不支持的图像数据类型 {rgb.dtype}，仅支持 uint8 或 float32"
    )
