# -*- coding: utf-8 -*-
"""统一超分辨率推理接口模块。

严格按模型实例类型区分预处理分支（单一真源）：
- ``SRCNN``：使用 ``bicubic_upsample_pillow``（Pillow BICUBIC，a=-0.5）预上采样后送入网络。
- ``ESPCN``：直接将原始 LR 输入送入网络，由网络内部 PixelShuffle 完成空间放大。

模型 ``forward()`` 本身保留无界回归输出；推理层默认通过 ``clamp=True`` 将输出裁剪至 ``[0, 1]``。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models._validation import validate_image_tensor, validate_scale
from src.models.espcn import ESPCN
from src.models.srcnn import SRCNN
from src.processing.resample import bicubic_upsample_pillow


def _resolve_device(
    model: nn.Module, device: torch.device | str | None
) -> torch.device:
    """解析推理目标设备。"""
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _preupsample_tensor_pillow(lr_tensor: torch.Tensor, scale: int) -> torch.Tensor:
    """将 [B, 3, H, W] float32 张量逐样本经 Pillow BICUBIC 上采样至目标尺寸。"""
    lr_cpu = lr_tensor.detach().to(device="cpu", dtype=torch.float32)
    upsampled_batch: list[torch.Tensor] = []
    for idx in range(lr_cpu.shape[0]):
        # CHW -> HWC float32，调用与数据管线一致的 bicubic_upsample_pillow
        hwc = np.ascontiguousarray(lr_cpu[idx].permute(1, 2, 0).numpy())
        up_hwc = bicubic_upsample_pillow(hwc, scale)
        up_chw = torch.from_numpy(np.ascontiguousarray(up_hwc)).permute(2, 0, 1)
        upsampled_batch.append(up_chw)
    return torch.stack(upsampled_batch, dim=0)


def predict_tensor(
    model: nn.Module,
    lr_tensor: torch.Tensor,
    scale: int,
    device: torch.device | str | None = None,
    clamp: bool = True,
) -> torch.Tensor:
    """对低分辨率张量执行统一推理。

    Args:
        model: ``SRCNN`` 或 ``ESPCN`` 模型实例。
        lr_tensor: 低分辨率输入张量，形状为 ``[B, 3, H, W]``，``float32``，数值范围 ``[0, 1]``。
        scale: 期望放大倍率（``2`` 或 ``4``），用于与 ``model.scale`` 做一致性校验。
        device: 推理设备；为 ``None`` 时沿用模型当前所在设备。
        clamp: 是否将输出裁剪到 ``[0.0, 1.0]``，默认 ``True``。

    Returns:
        超分辨率输出张量，形状为 ``[B, 3, H * scale, W * scale]``，``float32``。

    Raises:
        TypeError: 模型或张量类型不合法。
        ValueError: ``scale`` 与 ``model.scale`` 不一致，或张量形状不合法。
    """
    if not isinstance(model, (SRCNN, ESPCN)):
        raise TypeError(
            f"不支持的模型类型: {type(model).__name__}，仅支持 SRCNN 或 ESPCN"
        )
    validated_scale = validate_scale(scale)
    if model.scale != validated_scale:
        raise ValueError(
            f"倍率不匹配: 模型配置 scale={model.scale}，但调用传入 scale={validated_scale}"
        )
    validate_image_tensor(lr_tensor)

    original_device = _resolve_device(model, None)
    target_device = _resolve_device(model, device)
    was_training = model.training

    try:
        model_to_run = model.to(target_device) if target_device != original_device else model
        model_to_run.eval()

        if isinstance(model_to_run, SRCNN):
            model_input = _preupsample_tensor_pillow(lr_tensor, validated_scale).to(
                target_device
            )
        else:
            model_input = lr_tensor.to(target_device)

        with torch.no_grad():
            output = model_to_run(model_input)
            if clamp:
                output = torch.clamp(output, 0.0, 1.0)
        return output
    finally:
        model.train(was_training)
        if target_device != original_device:
            model.to(original_device)


def predict_rgb_image(
    model: nn.Module,
    lr_rgb: np.ndarray,
    scale: int,
    device: torch.device | str | None = None,
    clamp: bool = True,
) -> np.ndarray:
    """对 uint8 RGB 图像执行超分辨率推理并返回 uint8 RGB 图像。"""
    if not isinstance(lr_rgb, np.ndarray) or lr_rgb.dtype != np.uint8:
        raise TypeError("lr_rgb 必须为 uint8 类型的 RGB numpy.ndarray")
    if lr_rgb.ndim != 3 or lr_rgb.shape[2] != 3 or lr_rgb.size == 0:
        raise ValueError(f"lr_rgb 必须为非空 (H, W, 3) 数组，实际形状为 {lr_rgb.shape}")
    if not isinstance(model, (SRCNN, ESPCN)):
        raise TypeError(
            f"不支持的模型类型: {type(model).__name__}，仅支持 SRCNN 或 ESPCN"
        )
    validated_scale = validate_scale(scale)
    if model.scale != validated_scale:
        raise ValueError(
            f"倍率不匹配: 模型配置 scale={model.scale}，但调用传入 scale={validated_scale}"
        )

    original_device = _resolve_device(model, None)
    target_device = _resolve_device(model, device)
    was_training = model.training

    try:
        model_to_run = model.to(target_device) if target_device != original_device else model
        model_to_run.eval()

        if isinstance(model_to_run, SRCNN):
            # 图像入口先在 uint8 域执行 Pillow BICUBIC 上采样再归一化
            preup_rgb = bicubic_upsample_pillow(lr_rgb, validated_scale)
            in_arr = np.ascontiguousarray(preup_rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
        else:
            in_arr = np.ascontiguousarray(lr_rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0

        in_tensor = torch.from_numpy(in_arr).unsqueeze(0).to(target_device)
        with torch.no_grad():
            out_tensor = model_to_run(in_tensor)
            if clamp:
                out_tensor = torch.clamp(out_tensor, 0.0, 1.0)

        out_hwc = out_tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        out_uint8 = np.clip(np.rint(out_hwc * 255.0), 0, 255).astype(np.uint8)
        return out_uint8
    finally:
        model.train(was_training)
        if target_device != original_device:
            model.to(original_device)
