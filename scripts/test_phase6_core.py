# -*- coding: utf-8 -*-
"""阶段 6-A 核心模块单元测试：评价指标（PSNR/SSIM/分割指标）与课本 DIP 算子。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    compute_psnr,
    compute_segmentation_metrics,
    compute_ssim,
)
from src.processing.dip_enhancement import (  # noqa: E402
    clahe_enhancement,
    fft_highpass_filter,
    gaussian_filter,
    histogram_equalization,
    laplacian_sharpen,
    median_filter,
    morphological_refine_mask,
)


def main() -> int:
    rng = np.random.default_rng(42)
    img_rgb = rng.integers(20, 235, size=(64, 64, 3), dtype=np.uint8)
    img_gray = rng.integers(20, 235, size=(64, 64), dtype=np.uint8)

    # 1. PSNR / SSIM 精确性
    assert math.isinf(compute_psnr(img_rgb, img_rgb, data_range=255.0))
    assert abs(compute_ssim(img_rgb, img_rgb, data_range=255.0) - 1.0) < 1e-12

    noisy = np.clip(img_rgb.astype(np.int16) + 10, 0, 255).astype(np.uint8)
    psnr_rgb = compute_psnr(noisy, img_rgb, data_range=255.0, crop_border=2, y_channel=False)
    psnr_y = compute_psnr(noisy, img_rgb, data_range=255.0, crop_border=2, y_channel=True)
    ssim_y = compute_ssim(noisy, img_rgb, data_range=255.0, crop_border=2, y_channel=True)
    assert 20.0 < psnr_rgb < 35.0
    assert 20.0 < psnr_y < 40.0
    assert 0.5 < ssim_y < 1.0
    print(f"  [PASS] PSNR/SSIM: psnr_rgb={psnr_rgb:.2f} dB, psnr_y={psnr_y:.2f} dB, ssim_y={ssim_y:.4f}")

    # 2. 分割指标精确性
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[2:6, 2:6] = 1  # 16 px
    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[2:6, 4:8] = 1  # 16 px, overlap=8 px, union=24 px
    seg_m = compute_segmentation_metrics(pred, gt)
    assert abs(seg_m["iou"] - (8.0 / 24.0)) < 1e-6
    assert abs(seg_m["dice"] - (16.0 / 32.0)) < 1e-6
    assert abs(seg_m["precision"] - 0.5) < 1e-6
    assert abs(seg_m["recall"] - 0.5) < 1e-6
    print(f"  [PASS] 分割指标: IoU={seg_m['iou']:.4f}, Dice={seg_m['dice']:.4f}")

    # 3. 7 类课本 DIP 算子（RGB + 灰度 + 形态学）
    for fn, args in [
        (median_filter, (3,)),
        (gaussian_filter, (3, 0.8)),
        (histogram_equalization, ()),
        (clahe_enhancement, (2.0, (8, 8))),
        (laplacian_sharpen, (0.3,)),
        (fft_highpass_filter, (0.15, 0.4)),
    ]:
        out_rgb = fn(img_rgb, *args)
        out_gray = fn(img_gray, *args)
        assert out_rgb.shape == img_rgb.shape and out_rgb.dtype == np.uint8
        assert out_gray.shape == img_gray.shape and out_gray.dtype == np.uint8
    print("  [PASS] 6 类图像预处理/增强算子 (RGB & Gray) 形状与 dtype 校验通过")

    # 4. 形态学后处理：去除孤立噪点 + 填补内部小孔 + 仅保留最大连通域
    raw_mask = np.zeros((32, 32), dtype=np.uint8)
    raw_mask[6:22, 6:22] = 1
    raw_mask[13, 13] = 0  # 小孔洞
    raw_mask[1, 1] = 1  # 孤立噪点
    refined = morphological_refine_mask(raw_mask, open_ksize=3, close_ksize=5, keep_largest=True)
    assert refined.shape == raw_mask.shape and refined.dtype == np.uint8
    assert refined[1, 1] == 0, "孤立噪点应被开运算去除"
    assert refined[13, 13] == 1, "内部小孔洞应被闭运算填补"
    print("  [PASS] 形态学后处理 (开运算+闭运算+最大连通域) 校验通过")

    print("RESULT = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
