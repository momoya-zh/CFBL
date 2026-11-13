# -*- coding: utf-8 -*-
"""阶段 2 数据集完整性核验（只读）。

运行方式::

    "D:\\Environment\\Python\\Python_AI\\python.exe" scripts\\verify_datasets.py

职责：只对 datasets/ 下的 HR 原始图像做检查与记录。
**不修改任何图像**，不生成 LR、不裁剪、不缩放、不重新编码、不删除重复文件、
不安装依赖、不改动环境配置。

检查项：文件数量、逐文件实际解码、格式 / 尺寸 / 通道统计、零字节文件、
重复文件名、SHA-256 内容重复、训练与验证划分隔离。
SHA-256 是**本地文件指纹**，不是官方校验值。

产出：
    datasets/dataset_inventory.csv
    logs/phase02_verification.log
"""
from __future__ import annotations

import csv
import hashlib
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
LOGS_DIR = ROOT / "logs"
INVENTORY_PATH = DATASETS_DIR / "dataset_inventory.csv"
LOG_PATH = LOGS_DIR / "phase02_verification.log"

# (数据集名, 目录, 预期数量, 用途)
TARGETS = [
    ("DIV2K_Train_HR", DATASETS_DIR / "DIV2K" / "DIV2K_train_HR", 800, "train"),
    ("DIV2K_Valid_HR", DATASETS_DIR / "DIV2K" / "DIV2K_valid_HR", 100, "valid"),
    ("Flickr2K_HR", DATASETS_DIR / "Flickr2K" / "HR", 2650, "train"),
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CHANNEL_OF_MODE = {"1": 1, "L": 1, "P": 1, "LA": 2, "RGB": 3,
                   "RGBA": 4, "CMYK": 4, "LAB": 3, "HSV": 3, "I": 1, "F": 1}

_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _lines.append(msg)


def section(title: str) -> None:
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


def sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def rel_to_datasets(path: Path) -> str:
    return path.relative_to(DATASETS_DIR).as_posix()


def main() -> int:
    from PIL import Image  # 已随 Pillow 11.3.0 安装，非新增依赖

    started = time.perf_counter()
    section("阶段 2 数据集完整性核验 / verify_datasets.py")
    log("解释器 = {}".format(sys.executable))
    log("Pillow = {}".format(Image.__version__))
    log("数据集根目录 = {}".format(DATASETS_DIR))
    log("说明：SHA-256 为本地文件指纹，不声称是官方校验值；本脚本不修改任何图像。")

    rows: list[dict] = []
    by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_name: dict[str, list[tuple[str, str]]] = defaultdict(list)
    unreadable: list[tuple[str, str, str]] = []
    zero_byte: list[tuple[str, str]] = []
    non_image: list[tuple[str, str]] = []
    per_dataset: dict[str, dict] = {}
    hash_by_dataset: dict[str, set[str]] = defaultdict(set)
    path_by_dataset: dict[str, set[str]] = defaultdict(set)

    for name, root, expected, role in TARGETS:
        section("数据集：{}  (用途={}, 预期={} 张)".format(name, role, expected))
        log("目录 = {}".format(root))
        if not root.is_dir():
            log("[FAIL] 目录不存在")
            per_dataset[name] = {"exists": False, "count": 0, "expected": expected}
            continue

        files = sorted(p for p in root.rglob("*")
                       if p.is_file() and p.name != ".gitkeep")
        images = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
        others = [p for p in files if p.suffix.lower() not in IMAGE_EXTS]
        for p in others:
            non_image.append((name, rel_to_datasets(p)))

        formats: Counter = Counter()
        modes: Counter = Counter()
        widths: list[int] = []
        heights: list[int] = []
        total_bytes = 0
        unreadable_here = 0

        for idx, path in enumerate(images, 1):
            rel = rel_to_datasets(path)
            size = path.stat().st_size
            total_bytes += size
            if size == 0:
                zero_byte.append((name, rel))
            digest = ""
            fmt = w = h = mode = ""
            try:
                digest = sha256_of(path)
                with Image.open(path) as im:
                    im.load()  # 真正解码像素，不只是读取头部
                    fmt = str(im.format)
                    w, h = int(im.size[0]), int(im.size[1])
                    mode = str(im.mode)
                formats[fmt] += 1
                modes["{} ({}ch)".format(mode, CHANNEL_OF_MODE.get(mode, "?"))] += 1
                widths.append(w)
                heights.append(h)
            except Exception as exc:
                unreadable_here += 1
                unreadable.append((name, rel, "{}: {}".format(type(exc).__name__, exc)))

            rows.append({
                "dataset": name,
                "role": role,
                "relative_path": rel,
                "file_size_bytes": size,
                "width": w,
                "height": h,
                "format": fmt,
                "mode": mode,
                "sha256": digest,
            })
            if digest:
                by_hash[digest].append((name, rel))
                hash_by_dataset[name].add(digest)
            by_name[path.name].append((name, rel))
            path_by_dataset[name].add(rel)
            if idx % 500 == 0:
                log("  已核验 {}/{} 张 ...".format(idx, len(images)))

        log("图像文件数       = {} （预期 {}）".format(len(images), expected))
        log("目录内非图像文件 = {}".format(len(others)))
        log("总字节数         = {} ({:.2f} GiB)".format(total_bytes, total_bytes / 2**30))
        log("解码失败数       = {}".format(unreadable_here))
        log("格式分布         = {}".format(dict(formats)))
        log("通道/模式分布    = {}".format(dict(modes)))
        if widths:
            log("宽度范围         = {} ~ {}".format(min(widths), max(widths)))
            log("高度范围         = {} ~ {}".format(min(heights), max(heights)))
        else:
            log("宽度/高度范围    = (无可解码图像)")
        ok_count = (len(images) == expected)
        log("数量判定         = {}".format("PASS" if ok_count else "FAIL"))
        per_dataset[name] = {
            "exists": True, "count": len(images), "expected": expected,
            "ok_count": ok_count, "unreadable": unreadable_here,
            "formats": dict(formats), "modes": dict(modes),
            "w_min": min(widths) if widths else None,
            "w_max": max(widths) if widths else None,
            "h_min": min(heights) if heights else None,
            "h_max": max(heights) if heights else None,
            "total_bytes": total_bytes,
        }

    section("全局：重复与零字节检查")
    dup_hash = {h: locs for h, locs in by_hash.items() if len(locs) > 1}
    dup_name = {n: locs for n, locs in by_name.items() if len(locs) > 1}
    log("重复文件名（basename 出现在多个路径）组数 = {}".format(len(dup_name)))
    for n, locs in list(sorted(dup_name.items()))[:20]:
        log("  NAME {}: {}".format(n, locs))
    if len(dup_name) > 20:
        log("  ... 其余 {} 组见日志省略（完整信息可由 CSV 还原）".format(len(dup_name) - 20))
    log("内容完全相同（SHA-256 相同）组数 = {}".format(len(dup_hash)))
    cross = 0
    for h, locs in list(sorted(dup_hash.items()))[:20]:
        names = {l[0] for l in locs}
        if len(names) > 1:
            cross += 1
        log("  HASH {}... : {}".format(h[:16], locs))
    log("  其中跨数据集重复组（前 20 组内）= {}".format(cross))
    log("零字节文件数 = {}".format(len(zero_byte)))
    for name, rel in zero_byte[:20]:
        log("  ZERO {}: {}".format(name, rel))
    log("非图像文件数（各 HR 目录内）= {}".format(len(non_image)))
    for name, rel in non_image[:20]:
        log("  NON_IMAGE {}: {}".format(name, rel))
    log("无法解码文件数 = {}".format(len(unreadable)))
    for name, rel, err in unreadable[:30]:
        log("  UNREADABLE {}: {} | {}".format(name, rel, err))
    log("提示：重复图像只报告，不删除（按任务书要求）。")

    section("全局：训练 / 验证划分隔离")
    train_name = "DIV2K_Train_HR"
    valid_name = "DIV2K_Valid_HR"
    flickr_name = "Flickr2K_HR"
    path_overlap = path_by_dataset[train_name] & path_by_dataset[valid_name]
    content_overlap = hash_by_dataset[train_name] & hash_by_dataset[valid_name]
    flickr_vs_valid = hash_by_dataset[flickr_name] & hash_by_dataset[valid_name]
    log("DIV2K Train 与 Valid 路径交集数 = {} （应为 0）".format(len(path_overlap)))
    log("DIV2K Train 与 Valid 内容(SHA-256)交集数 = {} （应为 0）".format(len(content_overlap)))
    log("Flickr2K 与 DIV2K Valid 内容交集数 = {} （应为 0；Flickr2K 全部归训练）".format(len(flickr_vs_valid)))
    log("Flickr2K 用途标记 = train（全部 {} 张）".format(len(path_by_dataset[flickr_name])))
    log("DIV2K Valid 用途标记 = valid（{} 张，未复制到训练目录）".format(len(path_by_dataset[valid_name])))
    split_ok = (not path_overlap) and (not content_overlap) and (not flickr_vs_valid)
    log("划分判定 = {}".format("PASS" if split_ok else "FAIL"))

    section("产出文件")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "role", "relative_path", "file_size_bytes",
              "width", "height", "format", "mode", "sha256"]
    with INVENTORY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log("逐文件清单 = {} （{} 行数据）".format(INVENTORY_PATH, len(rows)))

    section("汇总 / SUMMARY")
    failures = []
    for name, info in per_dataset.items():
        if not info.get("exists"):
            failures.append("{} 目录不存在".format(name))
        else:
            if not info.get("ok_count"):
                failures.append("{} 数量不符：{}/{}".format(name, info["count"], info["expected"]))
            if info.get("unreadable"):
                failures.append("{} 有 {} 张无法解码".format(name, info["unreadable"]))
    if zero_byte:
        failures.append("存在 {} 个零字节文件".format(len(zero_byte)))
    if unreadable:
        failures.append("存在 {} 个无法读取文件".format(len(unreadable)))
    if not split_ok:
        failures.append("训练 / 验证划分存在混用")
    log("耗时 = {:.1f} s".format(time.perf_counter() - started))
    if failures:
        for f in failures:
            log("  FAIL: " + f)
        log("RESULT = FAIL（{} 项）".format(len(failures)))
        result = 1
    else:
        log("RESULT = PASS（数量正确、全部可解码、无零字节、划分隔离）")
        result = 0
    log("注意：PASS 不代表图像内容语义正确，也不代表已生成 LR 或可开始训练。")
    log("训练速度与显存峰值的测量属于阶段 5，与本核验无关。")

    LOG_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    sys.exit(main())
