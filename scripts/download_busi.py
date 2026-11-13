# -*- coding: utf-8 -*-
"""从 HuggingFace 公开镜像下载 BUSI 乳腺超声图像数据集（Dataset_BUSI_with_GT）。"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "ztroll/breast-ultrasound-images-dataset-dataset-busi"
API_URL = f"https://huggingface.co/api/datasets/{REPO_ID}"
RAW_DIR = PROJECT_ROOT / "datasets" / "BUSI_raw"


def _download_one(filename: str) -> tuple[str, bool]:
    encoded = urllib.parse.quote(filename, safe="")
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{encoded}"
    lower = filename.lower()
    if lower.startswith("benign"):
        subdir = "benign"
    elif lower.startswith("malignant"):
        subdir = "malignant"
    elif lower.startswith("normal"):
        subdir = "normal"
    else:
        subdir = "other"

    dst = RAW_DIR / subdir / filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return filename, True

    data = urllib.request.urlopen(url, timeout=60).read()
    dst.write_bytes(data)
    return filename, False


def main() -> int:
    print(f"正在查询 {REPO_ID} 文件清单...")
    meta = json.loads(urllib.request.urlopen(API_URL, timeout=30).read().decode("utf-8"))
    png_files = [
        item["rfilename"]
        for item in meta.get("siblings", [])
        if item["rfilename"].lower().endswith(".png")
    ]
    print(f"共发现 {len(png_files)} 个 PNG 文件，开始并发下载到 {RAW_DIR} ...")

    done = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(_download_one, fn) for fn in png_files]
        for fut in as_completed(futures):
            _, _ = fut.result()
            done += 1
            if done % 200 == 0 or done == len(png_files):
                print(f"  进度: {done}/{len(png_files)}")

    print("BUSI 下载完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
