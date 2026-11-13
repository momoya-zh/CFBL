# -*- coding: utf-8 -*-
"""阶段 6-A：超分辨率基准定量评估（DIV2K_valid_HR 及已就位的 Set5/Set14/BSD100/Urban100）。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.degradation import downsample_bicubic  # noqa: E402
from src.evaluation.metrics import compute_psnr, compute_ssim  # noqa: E402
from src.inference.predictor import predict_rgb_image  # noqa: E402
from src.models.model_factory import create_model  # noqa: E402
from src.processing.interpolation import upscale_image  # noqa: E402
from src.processing.resample import bicubic_upsample_pillow  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _collect_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def _load_trained_model(model_name: str, scale: int, device: torch.device) -> torch.nn.Module:
    ckpt_path = PROJECT_ROOT / "checkpoints" / f"{model_name}_x{scale}" / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"缺少训练权重: {ckpt_path}")
    model = create_model(model_name, scale).to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["DIV2K_valid", "Set5", "Set14", "BSD100", "Urban100"],
    )
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"评估设备: {device}")

    candidate_sets = {
        "DIV2K_valid": PROJECT_ROOT / "datasets" / "DIV2K" / "DIV2K_valid_HR",
        "Set5": PROJECT_ROOT / "datasets" / "Set5",
        "Set14": PROJECT_ROOT / "datasets" / "Set14",
        "BSD100": PROJECT_ROOT / "datasets" / "BSD100",
        "Urban100": PROJECT_ROOT / "datasets" / "Urban100",
    }

    datasets = {
        name: _collect_images(path)
        for name, path in candidate_sets.items()
        if name in args.datasets
    }
    datasets = {name: paths for name, paths in datasets.items() if len(paths) > 0}

    out_dir = PROJECT_ROOT / "results" / "phase6"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "sr_benchmark_metrics.csv"

    rows: list[dict[str, str | int | float]] = []

    for scale in (2, 4):
        srcnn = _load_trained_model("srcnn", scale, device)
        espcn = _load_trained_model("espcn", scale, device)

        for ds_name, img_paths in datasets.items():
            methods = [
                ("Nearest(cv2)", lambda lr, s=scale: upscale_image(lr, s, "nearest")),
                ("Bilinear(cv2)", lambda lr, s=scale: upscale_image(lr, s, "bilinear")),
                ("Bicubic(cv2)", lambda lr, s=scale: upscale_image(lr, s, "bicubic")),
                ("Bicubic(Pillow)", lambda lr, s=scale: bicubic_upsample_pillow(lr, s)),
                ("SRCNN", lambda lr, s=scale, m=srcnn: predict_rgb_image(m, lr, s, device=device)),
                ("ESPCN", lambda lr, s=scale, m=espcn: predict_rgb_image(m, lr, s, device=device)),
            ]

            accum: dict[str, list[tuple[float, float, float, float]]] = {
                m_name: [] for m_name, _ in methods
            }

            for img_path in img_paths:
                hr = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
                h, w = hr.shape[:2]
                h_mod = h - (h % scale)
                w_mod = w - (w % scale)
                hr_mod = hr[:h_mod, :w_mod]
                lr = downsample_bicubic(hr_mod, scale)

                for m_name, fn in methods:
                    sr = fn(lr)
                    psnr_y = compute_psnr(sr, hr_mod, data_range=255.0, crop_border=scale, y_channel=True)
                    ssim_y = compute_ssim(sr, hr_mod, data_range=255.0, crop_border=scale, y_channel=True)
                    psnr_rgb = compute_psnr(sr, hr_mod, data_range=255.0, crop_border=scale, y_channel=False)
                    ssim_rgb = compute_ssim(sr, hr_mod, data_range=255.0, crop_border=scale, y_channel=False)
                    accum[m_name].append((psnr_y, ssim_y, psnr_rgb, ssim_rgb))

            print(f"\n=== {ds_name} ({len(img_paths)} 张) | ×{scale} ===")
            print(f"{'Method':<18} {'PSNR-Y':>10} {'SSIM-Y':>10} {'PSNR-RGB':>10} {'SSIM-RGB':>10}")
            for m_name, _ in methods:
                vals = np.array(accum[m_name], dtype=np.float64)
                mean_py, mean_sy, mean_pr, mean_sr = vals.mean(axis=0)
                print(
                    f"{m_name:<18} {mean_py:>10.4f} {mean_sy:>10.4f} {mean_pr:>10.4f} {mean_sr:>10.4f}"
                )
                rows.append(
                    {
                        "dataset": ds_name,
                        "num_images": len(img_paths),
                        "scale": scale,
                        "method": m_name,
                        "psnr_y": f"{mean_py:.4f}",
                        "ssim_y": f"{mean_sy:.4f}",
                        "psnr_rgb": f"{mean_pr:.4f}",
                        "ssim_rgb": f"{mean_sr:.4f}",
                    }
                )

    mode = "a" if args.append and out_csv.exists() else "w"
    with out_csv.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "num_images",
                "scale",
                "method",
                "psnr_y",
                "ssim_y",
                "psnr_rgb",
                "ssim_rgb",
            ],
        )
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)

    print(f"\n已保存 SR 基准评估结果: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
