# -*- coding: utf-8 -*-
"""从 HuggingFace datasets-server (kasumi222/busigt) 下载 647 例 BUSI 病灶图像与掩膜并严格划分 train/val/test。"""

from __future__ import annotations

import base64
import csv
import io
import json
import random
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_ROOT / "datasets" / "BUSI_clean"
DATASET_NAME = "kasumi222/busigt"


def main() -> int:
    rows_all: list[tuple[int, bytes, bytes]] = []
    offset = 0
    batch = 50
    total = 647

    import time

    print(f"正在从 HuggingFace datasets-server 获取 {DATASET_NAME}（共 {total} 例）...")
    while offset < total:
        url = (
            f"https://datasets-server.huggingface.co/rows?"
            f"dataset={DATASET_NAME}&config=default&split=train&offset={offset}&length={batch}"
        )
        data = None
        for attempt in range(5):
            try:
                data = json.loads(urllib.request.urlopen(url, timeout=60).read().decode("utf-8"))
                break
            except Exception as exc:
                if attempt == 4:
                    raise
                print(f"  请求重试 ({attempt + 1}/5): {exc}")
                time.sleep(3)
        assert data is not None
        batch_rows = data.get("rows", [])
        if not batch_rows:
            break
        for item in batch_rows:
            idx = int(item["row_idx"])
            r = item["row"]
            img_b = base64.b64decode(r["pixel_values"]["bytes"])
            msk_b = base64.b64decode(r["label"]["bytes"])
            rows_all.append((idx, img_b, msk_b))
        offset += len(batch_rows)
        print(f"  已获取: {len(rows_all)}/{total}")

    if len(rows_all) < 600:
        raise RuntimeError(f"获取样本过少: {len(rows_all)}")

    rng = random.Random(42)
    rng.shuffle(rows_all)
    n = len(rows_all)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))

    splits = {
        "train": rows_all[:n_train],
        "val": rows_all[n_train : n_train + n_val],
        "test": rows_all[n_train + n_val :],
    }

    manifest_rows: list[dict[str, object]] = []
    for split_name, items in splits.items():
        img_dir = CLEAN_DIR / split_name / "images"
        msk_dir = CLEAN_DIR / split_name / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

        for idx, img_b, msk_b in items:
            fname = f"busi_{idx:04d}.png"
            img = Image.open(io.BytesIO(img_b)).convert("RGB")
            msk = Image.open(io.BytesIO(msk_b)).convert("L")
            if msk.size != img.size:
                msk = msk.resize(img.size, Image.Resampling.NEAREST)
            msk_arr = (np.asarray(msk, dtype=np.uint8) > 0).astype(np.uint8) * 255

            img.save(img_dir / fname)
            Image.fromarray(msk_arr).save(msk_dir / fname)

            manifest_rows.append(
                {
                    "split": split_name,
                    "filename": fname,
                    "orig_idx": idx,
                    "width": img.size[0],
                    "height": img.size[1],
                    "fg_ratio": f"{float((msk_arr > 0).mean()):.4f}",
                }
            )

    with (CLEAN_DIR / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "filename", "orig_idx", "width", "height", "fg_ratio"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        f"BUSI 数据集就绪: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
