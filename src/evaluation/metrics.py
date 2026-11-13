# -*- coding: utf-8 -*-
"""PSNR、SSIM 与二值图像分割评价指标实现。"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _validate_image_pair(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(pred, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("pred 与 target 必须为 numpy.ndarray")
    if pred.shape != target.shape:
        raise ValueError(f"pred 形状 {pred.shape} 与 target 形状 {target.shape} 不一致")
    if pred.ndim not in (2, 3):
        raise ValueError(f"仅支持 2D 灰度图或 3D 彩色图，当前 ndim={pred.ndim}")
    if data_range <= 0:
        raise ValueError(f"data_range 必须大于 0，当前为 {data_range}")

    pred_f = pred.astype(np.float64, copy=False)
    target_f = target.astype(np.float64, copy=False)
    return pred_f, target_f


def rgb_to_y_channel(image: np.ndarray, data_range: float = 255.0) -> np.ndarray:
    """按 ITU-R BT.601 标准将 RGB 图像转换为 Y 亮度通道（保持输入量纲）。"""
    if image.ndim == 2:
        return image.astype(np.float64, copy=False)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rgb_to_y_channel 需要 (H, W, 3) RGB 图像，当前为 {image.shape}")

    img_01 = image.astype(np.float64) / float(data_range)
    # BT.601 标准 Y 分量映射至 [16, 235] / 255.0 区间再还原到 data_range
    y_01 = (
        16.0
        + 65.481 * img_01[:, :, 0]
        + 128.553 * img_01[:, :, 1]
        + 24.966 * img_01[:, :, 2]
    ) / 255.0
    return y_01 * float(data_range)


def _crop_border_2d_or_3d(img: np.ndarray, crop_border: int) -> np.ndarray:
    if crop_border < 0:
        raise ValueError(f"crop_border 不能为负数，当前为 {crop_border}")
    if crop_border == 0:
        return img
    if img.shape[0] <= 2 * crop_border or img.shape[1] <= 2 * crop_border:
        raise ValueError(
            f"图像尺寸 {img.shape[:2]} 过小，无法裁剪 {crop_border} 像素边界"
        )
    return img[crop_border:-crop_border, crop_border:-crop_border]


def compute_psnr(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 255.0,
    crop_border: int = 0,
    y_channel: bool = False,
) -> float:
    """计算 PSNR（dB）。两图完全一致时返回 inf。"""
    pred_f, target_f = _validate_image_pair(pred, target, data_range)
    if y_channel and pred_f.ndim == 3:
        pred_f = rgb_to_y_channel(pred_f, data_range=data_range)
        target_f = rgb_to_y_channel(target_f, data_range=data_range)

    pred_f = _crop_border_2d_or_3d(pred_f, crop_border)
    target_f = _crop_border_2d_or_3d(target_f, crop_border)

    mse = float(np.mean((pred_f - target_f) ** 2))
    if mse == 0.0:
        return math.inf
    return float(peak_signal_noise_ratio(target_f, pred_f, data_range=data_range))


def compute_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 255.0,
    crop_border: int = 0,
    y_channel: bool = False,
) -> float:
    """计算结构相似性 SSIM（范围 [-1, 1]，完全一致为 1.0）。"""
    pred_f, target_f = _validate_image_pair(pred, target, data_range)
    if y_channel and pred_f.ndim == 3:
        pred_f = rgb_to_y_channel(pred_f, data_range=data_range)
        target_f = rgb_to_y_channel(target_f, data_range=data_range)

    pred_f = _crop_border_2d_or_3d(pred_f, crop_border)
    target_f = _crop_border_2d_or_3d(target_f, crop_border)

    channel_axis = -1 if pred_f.ndim == 3 else None
    return float(
        structural_similarity(
            target_f,
            pred_f,
            data_range=data_range,
            channel_axis=channel_axis,
        )
    )


def compute_segmentation_metrics(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """计算二值分割指标：IoU、Dice、Pixel Accuracy、Precision、Recall。"""
    if pred_mask.shape != target_mask.shape:
        raise ValueError(
            f"pred_mask 形状 {pred_mask.shape} 与 target_mask 形状 {target_mask.shape} 不一致"
        )
    pred_b = pred_mask.astype(bool)
    target_b = target_mask.astype(bool)

    tp = float(np.logical_and(pred_b, target_b).sum())
    fp = float(np.logical_and(pred_b, np.logical_not(target_b)).sum())
    fn = float(np.logical_and(np.logical_not(pred_b), target_b).sum())
    tn = float(np.logical_and(np.logical_not(pred_b), np.logical_not(target_b)).sum())

    total = tp + fp + fn + tn
    iou = (tp + eps) / (tp + fp + fn + eps) if (tp + fp + fn) > 0 else 1.0
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps) if (2.0 * tp + fp + fn) > 0 else 1.0
    accuracy = (tp + tn) / max(total, 1.0)
    precision = (tp + eps) / (tp + fp + eps) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = (tp + eps) / (tp + fn + eps) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
    }
