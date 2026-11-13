# -*- coding: utf-8 -*-
"""课本传统数字图像处理方法模块（空间域滤波、对比度增强、频域增强与形态学后处理）。"""

from __future__ import annotations

import cv2
import numpy as np


def _validate_uint8_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("输入图像必须为 numpy.ndarray")
    if image.dtype != np.uint8:
        raise ValueError(f"仅支持 uint8 图像，当前 dtype={image.dtype}")
    if image.ndim not in (2, 3):
        raise ValueError(f"仅支持 2D 灰度或 3D RGB 图像，当前 ndim={image.ndim}")
    if image.ndim == 3 and image.shape[2] not in (1, 3):
        raise ValueError(f"3D 图像通道数必须为 1 或 3，当前为 {image.shape[2]}")
    return image


def _validate_odd_ksize(ksize: int, name: str = "ksize") -> int:
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError(f"{name} 必须为 >= 3 的奇数，当前为 {ksize}")
    return int(ksize)


def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """空间域中值滤波：抑制脉冲与超声斑点噪声并较好保留边缘。"""
    img = _validate_uint8_image(image)
    k = _validate_odd_ksize(ksize)
    return cv2.medianBlur(img, k)


def gaussian_filter(
    image: np.ndarray,
    ksize: int = 3,
    sigma: float = 0.8,
) -> np.ndarray:
    """空间域高斯平滑滤波：抑制高斯噪声。"""
    img = _validate_uint8_image(image)
    k = _validate_odd_ksize(ksize)
    if sigma < 0:
        raise ValueError(f"sigma 不能为负数，当前为 {sigma}")
    return cv2.GaussianBlur(img, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """空间域全局直方图均衡化（RGB 图像仅对 YCrCb 亮度通道均衡化以保色相）。"""
    img = _validate_uint8_image(image)
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    if img.shape[2] == 1:
        return cv2.equalizeHist(img[:, :, 0])[:, :, None]

    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def clahe_enhancement(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """限制对比度自适应直方图均衡化（CLAHE），增强超声低对比病灶边界。"""
    img = _validate_uint8_image(image)
    if clip_limit <= 0:
        raise ValueError(f"clip_limit 必须 > 0，当前为 {clip_limit}")
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)

    if img.ndim == 2:
        return clahe.apply(img)
    if img.shape[2] == 1:
        return clahe.apply(img[:, :, 0])[:, :, None]

    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def laplacian_sharpen(image: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """空间域拉普拉斯锐化：I_sharp = clip(I - alpha * Laplacian(I), 0, 255)。"""
    img = _validate_uint8_image(image)
    if alpha < 0:
        raise ValueError(f"alpha 不能为负数，当前为 {alpha}")
    lap = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F, ksize=3)
    sharpened = img.astype(np.float32) - float(alpha) * lap
    return np.clip(np.rint(sharpened), 0, 255).astype(np.uint8)


def fft_highpass_filter(
    image: np.ndarray,
    cutoff_ratio: float = 0.15,
    gain: float = 0.4,
) -> np.ndarray:
    """频率域高斯高通高频增强：F'(u,v) = F(u,v) * (1 + gain * (1 - exp(-D^2 / (2*D0^2))))。"""
    img = _validate_uint8_image(image)
    if not (0.0 < cutoff_ratio <= 1.0):
        raise ValueError(f"cutoff_ratio 必须在 (0, 1] 内，当前为 {cutoff_ratio}")
    if gain < 0:
        raise ValueError(f"gain 不能为负数，当前为 {gain}")

    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    y = np.arange(h, dtype=np.float32)[:, None] - cy
    x = np.arange(w, dtype=np.float32)[None, :] - cx
    d2 = x * x + y * y
    d0 = max(min(h, w) * float(cutoff_ratio), 1.0)
    hp_mask = 1.0 - np.exp(-d2 / (2.0 * d0 * d0))
    transfer = 1.0 + float(gain) * hp_mask

    def _enhance_channel(ch: np.ndarray) -> np.ndarray:
        fshift = np.fft.fftshift(np.fft.fft2(ch.astype(np.float32)))
        filtered = np.fft.ifft2(np.fft.ifftshift(fshift * transfer))
        return np.clip(np.rint(np.real(filtered)), 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return _enhance_channel(img)
    channels = [_enhance_channel(img[:, :, c]) for c in range(img.shape[2])]
    return np.stack(channels, axis=2)


def morphological_refine_mask(
    mask: np.ndarray,
    open_ksize: int = 3,
    close_ksize: int = 5,
    keep_largest: bool = True,
) -> np.ndarray:
    """二值分割掩膜形态学后处理：开运算去孤立噪点 -> 闭运算填补空洞 -> 可选保留最大连通域。"""
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask 必须为 numpy.ndarray")
    if mask.ndim != 2:
        raise ValueError(f"mask 必须为 2D 数组，当前 ndim={mask.ndim}")

    binary = (mask > 0).astype(np.uint8) * 255

    if open_ksize >= 3:
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_validate_odd_ksize(open_ksize, "open_ksize"),) * 2
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    if close_ksize >= 3:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_validate_odd_ksize(close_ksize, "close_ksize"),) * 2
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    if keep_largest and np.any(binary > 0):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            # 跳过背景标签 0，挑选面积最大的前景连通域
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            binary = np.where(labels == largest_label, np.uint8(255), np.uint8(0))

    return (binary > 0).astype(np.uint8)
