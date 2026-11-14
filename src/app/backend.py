# -*- coding: utf-8 -*-
"""Gradio 演示系统后端推理与预处理调度模块。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.degradation import downsample_bicubic
from src.evaluation.metrics import (
    compute_psnr,
    compute_segmentation_metrics,
    compute_ssim,
)
from src.inference.predictor import predict_rgb_image
from src.models.model_factory import create_model
from src.processing.dip_enhancement import (
    clahe_enhancement,
    fft_highpass_filter,
    gaussian_filter,
    histogram_equalization,
    laplacian_sharpen,
    median_filter,
    morphological_refine_mask,
)
from src.processing.interpolation import upscale_image
from src.processing.resample import bicubic_upsample_pillow
from src.segmentation.unet import UNet

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SR_METHODS = [
    "Nearest (OpenCV)",
    "Bilinear (OpenCV)",
    "Bicubic (OpenCV)",
    "Bicubic (Pillow, 冻结基线)",
    "SRCNN (自训模型)",
    "ESPCN (自训模型)",
]

DIP_OPS = [
    "无增强",
    "空间域：中值滤波 (去斑点噪声)",
    "空间域：高斯滤波 (平滑去噪)",
    "空间域：直方图均衡化",
    "空间域：CLAHE 自适应对比度增强",
    "空间域：拉普拉斯锐化",
    "频率域：FFT 高斯高通增强",
]

SEG_PIPELINES = [
    "A: Baseline (Bicubic + U-Net)",
    "B: 课本 DIP 预处理 (Bicubic + 中值滤波 + CLAHE + U-Net)",
    "C1: 自训超分 SRCNN ×2 + U-Net",
    "C2: 自训超分 ESPCN ×2 + U-Net",
]


class ModelRegistry:
    """按需懒加载并缓存 SR 与分割模型权重。"""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._sr_cache: dict[tuple[str, int], torch.nn.Module] = {}
        self._unet_cache: dict[str, torch.nn.Module] = {}

    def get_sr_model(self, model_name: str, scale: int) -> torch.nn.Module:
        key = (model_name, int(scale))
        if key not in self._sr_cache:
            ckpt_path = PROJECT_ROOT / "checkpoints" / f"{model_name}_x{scale}" / "best.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"未找到 SR 权重: {ckpt_path}")
            model = create_model(model_name, scale).to(self.device)
            payload = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
            model.load_state_dict(state_dict)
            model.eval()
            self._sr_cache[key] = model
        return self._sr_cache[key]

    def get_unet_model(self, group_name: str) -> torch.nn.Module:
        if group_name not in self._unet_cache:
            ckpt_path = PROJECT_ROOT / "results" / "phase6" / f"{group_name}_best.pth"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"未找到分割权重: {ckpt_path}")
            model = UNet(in_channels=3, out_channels=1, base_channels=32).to(self.device)
            model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
            model.eval()
            self._unet_cache[group_name] = model
        return self._unet_cache[group_name]


REGISTRY = ModelRegistry()


def apply_dip_operator(image: np.ndarray, dip_choice: str) -> np.ndarray:
    """对 uint8 RGB 图像应用指定的课本 DIP 算子。"""
    if dip_choice == "无增强":
        return image
    if dip_choice.startswith("空间域：中值滤波"):
        return median_filter(image, ksize=3)
    if dip_choice.startswith("空间域：高斯滤波"):
        return gaussian_filter(image, ksize=3, sigma=0.8)
    if dip_choice.startswith("空间域：直方图均衡化"):
        return histogram_equalization(image)
    if dip_choice.startswith("空间域：CLAHE"):
        return clahe_enhancement(image, clip_limit=2.0)
    if dip_choice.startswith("空间域：拉普拉斯锐化"):
        return laplacian_sharpen(image, alpha=0.3)
    if dip_choice.startswith("频率域：FFT"):
        return fft_highpass_filter(image, cutoff_ratio=0.15, gain=0.4)
    raise ValueError(f"未知 DIP 算子: {dip_choice}")


def run_super_resolution(
    input_img: np.ndarray | None,
    method: str,
    scale: int,
    dip_choice: str,
    eval_degraded: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """执行 Tab 1 超分辨率重建与可选定量评价。"""
    if input_img is None:
        return None, None, "请先上传一张图像或点击下方样例。"

    rgb = np.asarray(input_img, dtype=np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    elif rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]

    scale = int(scale)
    gt_hr: np.ndarray | None = None
    if eval_degraded:
        h, w = rgb.shape[:2]
        h_mod, w_mod = h - (h % scale), w - (w % scale)
        if h_mod < scale or w_mod < scale:
            return None, None, "输入图像过小，无法执行降采样对照。"
        gt_hr = rgb[:h_mod, :w_mod]
        lr_rgb = downsample_bicubic(gt_hr, scale)
    else:
        lr_rgb = rgb

    t0 = time.perf_counter()
    bicubic_ref = bicubic_upsample_pillow(lr_rgb, scale)

    if method == "Nearest (OpenCV)":
        sr_out = upscale_image(lr_rgb, scale, "nearest")
    elif method == "Bilinear (OpenCV)":
        sr_out = upscale_image(lr_rgb, scale, "bilinear")
    elif method == "Bicubic (OpenCV)":
        sr_out = upscale_image(lr_rgb, scale, "bicubic")
    elif method == "Bicubic (Pillow, 冻结基线)":
        sr_out = bicubic_ref
    elif method == "SRCNN (自训模型)":
        model = REGISTRY.get_sr_model("srcnn", scale)
        sr_out = predict_rgb_image(model, lr_rgb, scale, device=REGISTRY.device)
    elif method == "ESPCN (自训模型)":
        model = REGISTRY.get_sr_model("espcn", scale)
        sr_out = predict_rgb_image(model, lr_rgb, scale, device=REGISTRY.device)
    else:
        raise ValueError(f"未知重建方法: {method}")

    sr_out = apply_dip_operator(sr_out, dip_choice)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    lines = [
        f"- **方法 / 倍率**：`{method}` ×{scale} | **后处理**：`{dip_choice}`",
        f"- **LR 输入尺寸**：`{lr_rgb.shape[1]}×{lr_rgb.shape[0]}` → **重建输出尺寸**：`{sr_out.shape[1]}×{sr_out.shape[0]}`",
        f"- **推理耗时**：`{elapsed_ms:.2f} ms`（设备：`{REGISTRY.device}`）",
    ]
    if gt_hr is not None:
        psnr_y = compute_psnr(sr_out, gt_hr, data_range=255.0, crop_border=scale, y_channel=True)
        ssim_y = compute_ssim(sr_out, gt_hr, data_range=255.0, crop_border=scale, y_channel=True)
        base_psnr = compute_psnr(bicubic_ref, gt_hr, data_range=255.0, crop_border=scale, y_channel=True)
        base_ssim = compute_ssim(bicubic_ref, gt_hr, data_range=255.0, crop_border=scale, y_channel=True)
        lines.append(
            f"- **定量指标（Y 通道，去边 {scale}px）**："
            f"当前方案 **PSNR = `{psnr_y:.4f} dB` / SSIM = `{ssim_y:.4f}`** "
            f"（Pillow Bicubic 基线：`{base_psnr:.4f} dB` / `{base_ssim:.4f}`）"
        )

    return sr_out, bicubic_ref, "\n".join(lines)


def _overlay_mask(image: np.ndarray, mask_bin: np.ndarray) -> np.ndarray:
    vis = image.copy().astype(np.float32)
    fg = mask_bin > 0
    # 半透明红色填充 + 绿色轮廓线
    vis[fg, 0] = 0.65 * vis[fg, 0] + 0.35 * 255.0
    vis[fg, 1] = 0.65 * vis[fg, 1]
    vis[fg, 2] = 0.65 * vis[fg, 2]
    out = np.clip(vis, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours((fg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 128), 2)
    return out


def run_ultrasound_segmentation(
    input_img: np.ndarray | None,
    gt_mask_img: np.ndarray | None,
    pipeline_choice: str,
    simulate_lr: bool,
    apply_morph: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
    """执行 Tab 2 超声病灶分割、中间预处理展示与形态学后处理。"""
    if input_img is None:
        return None, None, None, "请先上传一张乳腺超声图像或点击下方样例。"

    rgb = np.asarray(input_img, dtype=np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    elif rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]

    target_256 = np.asarray(
        Image.fromarray(rgb).resize((256, 256), Image.Resampling.BICUBIC), dtype=np.uint8
    )
    lr_128 = downsample_bicubic(target_256, 2) if simulate_lr else np.asarray(
        Image.fromarray(target_256).resize((128, 128), Image.Resampling.BICUBIC), dtype=np.uint8
    )

    t0 = time.perf_counter()
    if pipeline_choice.startswith("A:"):
        group_name = "A_baseline_bicubic"
        preproc = bicubic_upsample_pillow(lr_128, 2) if simulate_lr else target_256
    elif pipeline_choice.startswith("B:"):
        group_name = "B_dip_only"
        up = bicubic_upsample_pillow(lr_128, 2) if simulate_lr else target_256
        preproc = clahe_enhancement(median_filter(up, ksize=3), clip_limit=2.0)
    elif pipeline_choice.startswith("C1:"):
        group_name = "C1_srcnn_only"
        sr = REGISTRY.get_sr_model("srcnn", 2)
        preproc = predict_rgb_image(sr, lr_128, 2, device=REGISTRY.device) if simulate_lr else target_256
    elif pipeline_choice.startswith("C2:"):
        group_name = "C2_espcn_only"
        sr = REGISTRY.get_sr_model("espcn", 2)
        preproc = predict_rgb_image(sr, lr_128, 2, device=REGISTRY.device) if simulate_lr else target_256
    else:
        raise ValueError(f"未知分割流水线: {pipeline_choice}")

    unet = REGISTRY.get_unet_model(group_name)
    in_tensor = (
        torch.from_numpy(np.ascontiguousarray(preproc.transpose(2, 0, 1), dtype=np.float32) / 255.0)
        .unsqueeze(0)
        .to(REGISTRY.device)
    )
    with torch.no_grad():
        prob = torch.sigmoid(unet(in_tensor))[0, 0].cpu().numpy()

    pred_bin = (prob >= 0.5).astype(np.uint8)
    if apply_morph:
        pred_bin = morphological_refine_mask(pred_bin, open_ksize=3, close_ksize=5, keep_largest=True)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    overlay = _overlay_mask(preproc, pred_bin)
    mask_vis = (pred_bin * 255).astype(np.uint8)
    fg_pct = float(pred_bin.mean()) * 100.0

    lines = [
        f"- **实验流水线**：`{pipeline_choice}` | **形态学后处理**：`{'开启' if apply_morph else '关闭'}`",
        f"- **推理耗时**：`{elapsed_ms:.2f} ms` | **预测病灶面积占比**：`{fg_pct:.2f}%`",
    ]

    if gt_mask_img is not None:
        gt_arr = np.asarray(gt_mask_img, dtype=np.uint8)
        if gt_arr.ndim == 3:
            gt_arr = gt_arr[:, :, 0]
        gt_256 = np.asarray(
            Image.fromarray(gt_arr).resize((256, 256), Image.Resampling.NEAREST), dtype=np.uint8
        )
        gt_bin = (gt_256 > 0).astype(np.uint8)
        m = compute_segmentation_metrics(pred_bin, gt_bin)
        lines.append(
            f"- **实时分割指标（vs 上传真值掩膜）**："
            f"**IoU = `{m['iou']:.4f}`** | **Dice = `{m['dice']:.4f}`** | "
            f"Accuracy = `{m['accuracy']:.4f}` | Precision = `{m['precision']:.4f}` | Recall = `{m['recall']:.4f}`"
        )

    return preproc, overlay, mask_vis, "\n".join(lines)
