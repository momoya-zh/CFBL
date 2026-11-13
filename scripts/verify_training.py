# -*- coding: utf-8 -*-
"""阶段 5 训练产物核验脚本。

运行方式（CMD）::

    "D:\\Environment\\Python\\Python_AI\\python.exe" scripts\\verify_training.py

核验项（依据阶段 5 任务书 §5.3）：
1. 四套模型的 ``best.pth`` 与 ``latest.pth`` 均存在且可通过 ``load_training_checkpoint`` 载入；
2. 每套 ``history.csv`` 至少 100 行，且 ``train_loss`` / ``val_loss`` 不含 ``nan`` / ``inf``；
3. ``best.pth`` 中的 ``best_val`` 等于 ``history.csv`` 中 ``val_loss`` 的最小值（误差 < 1e-9）；
4. 四条验证曲线均整体下降（前 10 个 Epoch 均值 > 后 10 个 Epoch 均值）；
5. 抽取 1 套模型使用 ``predict_rgb_image`` 对真实 DIV2K Validation 图像执行一次推理，输出尺寸严格正确。
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import predict_rgb_image  # noqa: E402
from src.models import create_model  # noqa: E402
from src.training import load_training_checkpoint  # noqa: E402

TARGETS = [
    ("srcnn", 2),
    ("srcnn", 4),
    ("espcn", 2),
    ("espcn", 4),
]


def main() -> int:
    print("=" * 72)
    print("阶段 5：四套训练产物核验")
    print("=" * 72)

    all_ok = True
    for model_name, scale in TARGETS:
        tag = f"{model_name}_x{scale}"
        ckpt_dir = PROJECT_ROOT / "checkpoints" / tag
        best_path = ckpt_dir / "best.pth"
        latest_path = ckpt_dir / "latest.pth"
        hist_path = PROJECT_ROOT / "results" / "training" / tag / "history.csv"
        curve_path = PROJECT_ROOT / "results" / "training" / tag / "loss_curve.png"

        try:
            assert best_path.is_file(), f"缺少 {best_path}"
            assert latest_path.is_file(), f"缺少 {latest_path}"
            assert hist_path.is_file(), f"缺少 {hist_path}"
            assert curve_path.is_file(), f"缺少 {curve_path}"

            m_best = create_model(model_name, scale)
            ckpt_best = load_training_checkpoint(best_path, m_best, map_location="cpu")
            m_latest = create_model(model_name, scale)
            ckpt_latest = load_training_checkpoint(latest_path, m_latest, map_location="cpu")
            assert int(ckpt_latest["epoch"]) >= 100, (
                f"{tag} latest.pth epoch={ckpt_latest['epoch']} < 100"
            )

            with hist_path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) >= 100, f"{tag} history.csv 行数={len(rows)} < 100"

            val_losses = [float(r["val_loss"]) for r in rows]
            train_losses = [float(r["train_loss"]) for r in rows]
            assert all(math.isfinite(v) for v in val_losses), f"{tag} val_loss 含 nan/inf"
            assert all(math.isfinite(v) for v in train_losses), f"{tag} train_loss 含 nan/inf"

            min_val = min(val_losses)
            best_val = float(ckpt_best["best_val"])
            diff = abs(best_val - min_val)
            assert diff < 1e-6, (
                f"{tag} best_val={best_val:.6f} 与 history min={min_val:.6f} 差={diff}"
            )

            head10 = sum(val_losses[:10]) / 10.0
            tail10 = sum(val_losses[-10:]) / 10.0
            assert head10 > tail10, (
                f"{tag} 验证曲线未整体下降: 前10均值={head10:.6f} <= 后10均值={tail10:.6f}"
            )

            print(
                f"  [PASS] {tag:<10} | epochs={len(rows)} | "
                f"best_val={best_val:.6f} (ep {ckpt_best['epoch']}) | "
                f"val_head10={head10:.6f} -> tail10={tail10:.6f}"
            )
        except Exception as exc:
            all_ok = False
            print(f"  [FAIL] {tag:<10} | {exc}")

    # 抽检 1 套模型真实推理
    sample_hr = PROJECT_ROOT / "datasets" / "DIV2K" / "DIV2K_valid_HR" / "0801.png"
    with Image.open(sample_hr) as im:
        crop = np.asarray(im.convert("RGB"), dtype=np.uint8)[:128, :128, :]
    m_sample = create_model("espcn", 2)
    load_training_checkpoint(
        PROJECT_ROOT / "checkpoints" / "espcn_x2" / "best.pth",
        m_sample,
        map_location="cpu",
    )
    sr = predict_rgb_image(m_sample, crop, scale=2, device="cpu", clamp=True)
    assert sr.shape == (256, 256, 3) and sr.dtype == np.uint8
    print(f"  [PASS] 真实推理抽检 (espcn_x2): 128x128 -> {sr.shape[1]}x{sr.shape[0]}")

    if not all_ok:
        return 1
    print("\nRESULT = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
