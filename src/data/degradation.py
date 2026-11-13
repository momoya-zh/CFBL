# -*- coding: utf-8 -*-
"""阶段 5 唯一冻结退化（LR 生成）实现模块。

冻结协议：
- 仅使用 ``PIL.Image.Resampling.BICUBIC``（Catmull-Rom，a = -0.5）。
- 仅在 ``uint8 [0, 255]`` RGB 三通道图像上执行，严禁调用 ``cv2.INTER_CUBIC`` 或 ``F.interpolate``。
- 尺寸取整严格采用向零取整除：``H_LR = H_HR // scale``、``W_LR = W_HR // scale``。
"""
from __future__ import annotations

import numpy as np
from PIL import Image

SUPPORTED_DEGRADATION_SCALES: tuple[int, ...] = (2, 4)


def _validate_hr_uint8(hr: np.ndarray, scale: int) -> tuple[int, int]:
    if not isinstance(hr, np.ndarray):
        raise TypeError(
            f"输入图像必须为 numpy.ndarray，实际类型为 {type(hr).__name__}"
        )
    if hr.dtype != np.uint8:
        raise TypeError(
            f"退化输入必须为 uint8 类型，实际为 {hr.dtype}"
        )
    if hr.size == 0 or hr.ndim != 3 or hr.shape[2] != 3:
        raise ValueError(
            f"输入图像必须为非空 RGB 三通道 (H, W, 3)，实际形状为 {hr.shape}"
        )
    if isinstance(scale, bool) or not isinstance(scale, int):
        raise TypeError(
            f"缩放倍率 scale 必须为整数，实际类型为 {type(scale).__name__}"
        )
    if scale not in SUPPORTED_DEGRADATION_SCALES:
        raise ValueError(
            f"不支持的缩放倍率: {scale}，仅支持 {SUPPORTED_DEGRADATION_SCALES}"
        )
    h, w = hr.shape[:2]
    if h < scale or w < scale:
        raise ValueError(
            f"输入图像尺寸 ({h}, {w}) 小于缩放倍率 {scale}，无法下采样"
        )
    return h, w


def downsample_bicubic(hr: np.ndarray, scale: int) -> np.ndarray:
    """使用冻结的 Pillow BICUBIC 协议对 uint8 RGB 图像执行下采样。

    Args:
        hr: 高分辨率 RGB 图像，``uint8``，形状 ``(H, W, 3)``。
        scale: 下采样倍率（``2`` 或 ``4``）。

    Returns:
        低分辨率 RGB 图像，``uint8``，形状 ``(H // scale, W // scale, 3)``。
    """
    h, w = _validate_hr_uint8(hr, scale)
    lr_h = h // scale
    lr_w = w // scale

    assert lr_h * scale <= h and lr_w * scale <= w
    assert (h - lr_h * scale) < scale and (w - lr_w * scale) < scale

    lr_pil = Image.fromarray(hr).resize(
        (lr_w, lr_h), resample=Image.Resampling.BICUBIC
    )
    return np.asarray(lr_pil, dtype=np.uint8)


def make_lr_from_patch(patch: np.ndarray, scale: int) -> np.ndarray:
    """从已对齐裁剪的 HR patch 生成训练用 LR patch。"""
    h, w = _validate_hr_uint8(patch, scale)
    if h % scale != 0 or w % scale != 0:
        raise ValueError(
            f"Patch 尺寸 ({h}, {w}) 必须能被倍率 {scale} 整除"
        )
    return downsample_bicubic(patch, scale)
