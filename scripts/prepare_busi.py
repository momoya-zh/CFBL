# -*- coding: utf-8 -*-
"""清洗 BUSI 数据集：合并同图多病灶掩膜，并按 70/15/15 严格划分 train/val/test（无泄露）。"""

from __future__ import annotations

import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "datasets" / "BUSI_raw"
CLEAN_DIR = PROJECT_ROOT / "datasets" / "BUSI_clean"


def main() -> int:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"未找到原始 BUSI 目录: {RAW_DIR}")

    # 仅使用包含病灶掩膜的 benign 与 malignant 类别（normal 掩膜全黑，不作为病灶目标，或可按需保留；
    # 在经典 BUSI 病灶分割基准中通常使用 benign + malignant 共 647 例有病灶样本以稳定计算 Dice/IoU）
    groups: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {"image": None, "masks": []}
    )

    for cls_name in ("benign", "malignant"):
        cls_dir = RAW_DIR / cls_name
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.glob("*.png")):
            stem = p.stem
            if "_mask" in stem:
                base_stem = re.sub(r"_mask.*$", "", stem)
                groups[(cls_name, base_stem)]["masks"].append(p)
            else:
                groups[(cls_name, stem)]["image"] = p

    valid_items: list[tuple[str, str, Path, list[Path]]] = []
    for (cls_name, base_stem), info in sorted(groups.items()):
        if info["image"] is not None and len(info["masks"]) > 0:
            valid_items.append((cls_name, base_stem, info["image"], info["masks"]))

    print(f"有效病灶样本数（benign + malignant）: {len(valid_items)}")
    if len(valid_items) == 0:
        raise RuntimeError("未找到有效样本")

    # 按类别分层随机划分 70% train / 15% val / 15% test（固定 seed=42）
    rng = random.Random(42)
    by_cls: dict[str, list[tuple[str, str, Path, list[Path]]]] = defaultdict(list)
    for item in valid_items:
        by_cls[item[0]].append(item)

    splits: dict[str, list[tuple[str, str, Path, list[Path]]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for cls_name, items in sorted(by_cls.items()):
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])

    manifest_rows: list[dict[str, object]] = []
    for split_name, items in splits.items():
        img_out_dir = CLEAN_DIR / split_name / "images"
        mask_out_dir = CLEAN_DIR / split_name / "masks"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        mask_out_dir.mkdir(parents=True, exist_ok=True)

        for cls_name, base_stem, img_path, mask_paths in items:
            safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", base_stem).strip("_") + ".png"
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            merged_mask = np.zeros((h, w), dtype=np.uint8)
            for mp in mask_paths:
                m_arr = np.asarray(Image.open(mp).convert("L"), dtype=np.uint8)
                if m_arr.shape != (h, w):
                    m_arr = np.asarray(
                        Image.open(mp).convert("L").resize((w, h), Image.Resampling.NEAREST),
                        dtype=np.uint8,
                    )
                merged_mask = np.maximum(merged_mask, (m_arr > 127).astype(np.uint8) * 255)

            img.save(img_out_dir / safe_name)
            Image.fromarray(merged_mask, mode="L").save(mask_out_dir / safe_name)

            manifest_rows.append(
                {
                    "split": split_name,
                    "class": cls_name,
                    "filename": safe_name,
                    "width": w,
                    "height": h,
                    "num_masks_merged": len(mask_paths),
                    "fg_ratio": f"{float((merged_mask > 0).mean()):.4f}",
                }
            )

    with (CLEAN_DIR / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "class", "filename", "width", "height", "num_masks_merged", "fg_ratio"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        f"划分完成: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
