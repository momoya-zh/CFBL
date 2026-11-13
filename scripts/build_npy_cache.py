# -*- coding: utf-8 -*-
"""将训练集 3,450 张 2K PNG 预先解压为无压缩 .npy 缓存（供 mmap 切片读取）。

运行方式（CMD）::

    "D:\\Environment\\Python\\Python_AI\\python.exe" scripts\\build_npy_cache.py
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / "datasets" / "_cache_npy"

SOURCE_GROUPS = [
    ("DIV2K_train_HR", PROJECT_ROOT / "datasets" / "DIV2K" / "DIV2K_train_HR"),
    ("Flickr2K_HR", PROJECT_ROOT / "datasets" / "Flickr2K" / "HR"),
]


def _convert_one(task: tuple[Path, Path]) -> tuple[bool, str]:
    src_path, dst_path = task
    if dst_path.is_file():
        return False, dst_path.name
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(".tmp.npy")
    with Image.open(src_path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    np.save(tmp_path, arr, allow_pickle=False)
    tmp_path.replace(dst_path)
    return True, dst_path.name


def main() -> int:
    tasks: list[tuple[Path, Path]] = []
    for group_name, src_dir in SOURCE_GROUPS:
        out_dir = CACHE_ROOT / group_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(src_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                tasks.append((p, out_dir / f"{p.stem}.npy"))

    print(f"待检查/转换图像总数: {len(tasks)} -> {CACHE_ROOT}")
    t0 = time.perf_counter()
    converted = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_convert_one, t) for t in tasks]
        for idx, fut in enumerate(as_completed(futures), start=1):
            did_write, _ = fut.result()
            if did_write:
                converted += 1
            else:
                skipped += 1
            if idx % 500 == 0 or idx == len(tasks):
                elapsed = time.perf_counter() - t0
                print(
                    f"  进度: {idx}/{len(tasks)} | 新转换={converted} | 已跳过={skipped} | 耗时={elapsed:.1f}s"
                )

    print(f"完成！共耗时 {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
