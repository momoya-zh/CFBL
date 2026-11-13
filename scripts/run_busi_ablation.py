# -*- coding: utf-8 -*-
"""阶段 6-B：BUSI 乳腺超声分割受控消融对照实验（Baseline / +DIP / +SR / SR+DIP+形态学后处理）。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.degradation import downsample_bicubic  # noqa: E402
from src.inference.predictor import predict_rgb_image  # noqa: E402
from src.models.model_factory import create_model  # noqa: E402
from src.processing.dip_enhancement import (  # noqa: E402
    clahe_enhancement,
    median_filter,
    morphological_refine_mask,
)
from src.processing.resample import bicubic_upsample_pillow  # noqa: E402
from src.segmentation.trainer import SegmentationResult, train_and_evaluate_segmentation  # noqa: E402
from src.segmentation.unet import UNet  # noqa: E402


def _load_sr(model_name: str, scale: int, device: torch.device) -> torch.nn.Module:
    ckpt_path = PROJECT_ROOT / "checkpoints" / f"{model_name}_x{scale}" / "best.pth"
    model = create_model(model_name, scale).to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _save_visual_comparison(
    busi_clean_dir: Path,
    out_dir: Path,
    espcn_x2: torch.nn.Module,
    device: torch.device,
) -> None:
    test_imgs = sorted((busi_clean_dir / "test" / "images").glob("*.png"))[:4]
    if not test_imgs:
        return

    base_unet = UNet(3, 1, 32).to(device)
    base_unet.load_state_dict(
        torch.load(out_dir / "A_baseline_bicubic_best.pth", map_location=device, weights_only=True)
    )
    base_unet.eval()

    full_unet = UNet(3, 1, 32).to(device)
    full_unet.load_state_dict(
        torch.load(out_dir / "D_espcn_dip_morph_best.pth", map_location=device, weights_only=True)
    )
    full_unet.eval()

    fig, axes = plt.subplots(len(test_imgs), 5, figsize=(15, 3 * len(test_imgs)), dpi=150)
    if len(test_imgs) == 1:
        axes = axes[None, :]
    col_titles = ["LR Bicubic", "ESPCN x2 + DIP", "Ground Truth", "Group A (Baseline)", "Group D (SR+DIP+Morph)"]

    for r, img_path in enumerate(test_imgs):
        mask_path = busi_clean_dir / "test" / "masks" / img_path.name
        hr = np.asarray(Image.open(img_path).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC), dtype=np.uint8)
        gt = (np.asarray(Image.open(mask_path).convert("L").resize((256, 256), Image.Resampling.NEAREST)) > 127).astype(np.uint8)
        lr = downsample_bicubic(hr, 2)

        in_base = bicubic_upsample_pillow(lr, 2)
        in_full = clahe_enhancement(median_filter(predict_rgb_image(espcn_x2, lr, 2, device=device), 3), 2.0)

        with torch.no_grad():
            t_base = torch.from_numpy(in_base.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(device)
            p_base = (torch.sigmoid(base_unet(t_base))[0, 0].cpu().numpy() >= 0.5).astype(np.uint8)

            t_full = torch.from_numpy(in_full.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(device)
            p_full_raw = (torch.sigmoid(full_unet(t_full))[0, 0].cpu().numpy() >= 0.5).astype(np.uint8)
            p_full = morphological_refine_mask(p_full_raw, open_ksize=3, close_ksize=5, keep_largest=True)

        panels = [in_base, in_full, gt * 255, p_base * 255, p_full * 255]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            if c < 2:
                ax.imshow(panel)
            else:
                ax.imshow(panel, cmap="gray")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "busi_ablation_samples.png")
    plt.close(fig)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    busi_clean_dir = PROJECT_ROOT / "datasets" / "BUSI_clean"
    out_dir = PROJECT_ROOT / "results" / "phase6"
    out_dir.mkdir(parents=True, exist_ok=True)

    srcnn_x2 = _load_sr("srcnn", 2, device)
    espcn_x2 = _load_sr("espcn", 2, device)

    experiments = [
        ("A_baseline_bicubic", "baseline_bicubic", None, False),
        ("B_dip_only", "dip_only", None, False),
        ("C1_srcnn_only", "sr_only", srcnn_x2, False),
        ("C2_espcn_only", "sr_only", espcn_x2, False),
        ("D1_srcnn_dip_morph", "sr_plus_dip", srcnn_x2, True),
        ("D_espcn_dip_morph", "sr_plus_dip", espcn_x2, True),
    ]

    results: list[SegmentationResult] = []
    for name, mode, sr_m, morph in experiments:
        print(f"\n>>> 运行实验组: {name} (mode={mode}, morph={morph})")
        res = train_and_evaluate_segmentation(
            group_name=name,
            busi_clean_dir=busi_clean_dir,
            out_dir=out_dir,
            pipeline_mode=mode,
            sr_model=sr_m,
            sr_scale=2,
            apply_morph_post=morph,
            epochs=40,
            batch_size=16,
            lr=1e-3,
            seed=42,
        )
        results.append(res)
        print(
            f"    best_val_dice={res.best_val_dice:.4f} (ep {res.best_epoch}) | "
            f"TEST: mIoU={res.test_iou:.4f}, Dice={res.test_dice:.4f}, "
            f"Acc={res.test_accuracy:.4f}, P={res.test_precision:.4f}, R={res.test_recall:.4f} "
            f"({res.elapsed_sec:.1f}s)"
        )

    csv_path = out_dir / "busi_ablation_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "best_val_dice",
                "best_epoch",
                "test_iou",
                "test_dice",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "elapsed_sec",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "group": r.group_name,
                    "best_val_dice": f"{r.best_val_dice:.6f}",
                    "best_epoch": r.best_epoch,
                    "test_iou": f"{r.test_iou:.6f}",
                    "test_dice": f"{r.test_dice:.6f}",
                    "test_accuracy": f"{r.test_accuracy:.6f}",
                    "test_precision": f"{r.test_precision:.6f}",
                    "test_recall": f"{r.test_recall:.6f}",
                    "elapsed_sec": f"{r.elapsed_sec:.2f}",
                }
            )

    _save_visual_comparison(busi_clean_dir, out_dir, espcn_x2, device)
    print(f"\n已保存 BUSI 消融指标表与样例图: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
