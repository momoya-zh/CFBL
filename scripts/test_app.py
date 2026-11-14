# -*- coding: utf-8 -*-
"""阶段 7 自动化冒烟测试：验证 Gradio UI 构建、Tab 1 超分推理与 Tab 2 超声分割推理全链路。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.backend import (  # noqa: E402
    DIP_OPS,
    SEG_PIPELINES,
    SR_METHODS,
    run_super_resolution,
    run_ultrasound_segmentation,
)
from src.app.gradio_app import build_demo  # noqa: E402


def main() -> int:
    demo = build_demo()
    assert demo is not None
    print("  [PASS] build_demo() 成功构建 Gradio Blocks")

    rng = np.random.default_rng(42)
    dummy_rgb = rng.integers(30, 220, size=(128, 128, 3), dtype=np.uint8)

    for method in SR_METHODS:
        out, base, info = run_super_resolution(
            dummy_rgb, method=method, scale=2, dip_choice=DIP_OPS[4], eval_degraded=True
        )
        assert out is not None and out.shape == (128, 128, 3) and out.dtype == np.uint8
        assert base is not None and base.shape == (128, 128, 3)
        assert "PSNR" in info
    print(f"  [PASS] Tab 1 全部 {len(SR_METHODS)} 种超分方法 + CLAHE + 定量指标校验通过")

    # Tab 2：使用真实 BUSI 测试样例验证全部 4 条流水线
    busi_img_p = sorted((PROJECT_ROOT / "datasets" / "BUSI_clean" / "test" / "images").glob("*.png"))[0]
    busi_msk_p = PROJECT_ROOT / "datasets" / "BUSI_clean" / "test" / "masks" / busi_img_p.name
    busi_img = np.asarray(Image.open(busi_img_p).convert("RGB"), dtype=np.uint8)
    busi_gt = np.asarray(Image.open(busi_msk_p).convert("L"), dtype=np.uint8)

    for pipe in SEG_PIPELINES:
        pre, overlay, mask_vis, info = run_ultrasound_segmentation(
            busi_img, busi_gt, pipeline_choice=pipe, simulate_lr=True, apply_morph=True
        )
        assert pre is not None and pre.shape == (256, 256, 3)
        assert overlay is not None and overlay.shape == (256, 256, 3)
        assert mask_vis is not None and mask_vis.shape == (256, 256)
        assert "IoU" in info and "Dice" in info
    print(f"  [PASS] Tab 2 全部 {len(SEG_PIPELINES)} 条超声分割流水线 + 形态学后处理 + 实时 IoU/Dice 校验通过")

    print("RESULT = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
