# -*- coding: utf-8 -*-
"""阶段 2：从 Flickr2K 双层归档中提取原始 HR 图像（只读归档，写出原始字节）。

运行方式::

    "D:\\Environment\\Python\\Python_AI\\python.exe" scripts\\extract_flickr2k.py

归档实际结构（2026-11-12 实测）：

    Flickr2K.tar.gz.00                      <- 外层：gzip 压缩的 tar
      Flickr2K/                            <- 目录
      Flickr2K/Flickr2K.tar                <- 内层：未压缩 tar（20.06 GiB）
        Flickr2K/Flickr2K_HR/*.png         <- 2650 张原始 HR（本阶段唯一提取对象）
        Flickr2K/Flickr2K_LR_unknown/X{2,3,4}/*.png   <- LR，共 7950 张，不提取
        Flickr2K/Flickr2K_LR_bicubic/X{2,3,4}/*.png   <- LR，共 7950 张，不提取

按任务书「本阶段只保存 HR 原始图像、不使用下载的 LR 图像」：
- 只提取 Flickr2K_HR 目录中的 PNG，**保留原始文件名（basename）**；
- LR 成员一律跳过，不落盘、不读取使用；
- 不缩放、不裁剪、不重新编码、不改名、不覆盖已有文件；
- 原始压缩包保留，不删除。

映射规则：Flickr2K/Flickr2K_HR/<basename> -> datasets/Flickr2K/HR/<basename>

产出：logs/phase02_flickr2k_extract.log
"""
from __future__ import annotations

import shutil
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "datasets" / "_archives" / "Flickr2K.tar.gz.00"
HR_DIR = ROOT / "datasets" / "Flickr2K" / "HR"
LOG_PATH = ROOT / "logs" / "phase02_flickr2k_extract.log"
EXPECTED_FILES = 2650
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
HR_DIR_NAME = "Flickr2K_HR"

_lines: list[str] = []
_failures: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _lines.append(msg)


def section(title: str) -> None:
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


def is_hr_image(name: str) -> bool:
    parts = name.replace("\\", "/").split("/")
    return (any(p == HR_DIR_NAME for p in parts)
            and Path(name).suffix.lower() in IMAGE_EXTS)


def group_of(name: str) -> str:
    parts = name.replace("\\", "/").split("/")
    if len(parts) >= 4:
        return parts[1] + "/" + parts[2]
    if len(parts) >= 3:
        return parts[1]
    return parts[0]


def main() -> int:
    started = time.perf_counter()
    section("阶段 2 Flickr2K HR 解压 / extract_flickr2k.py")
    log("解释器 = {}".format(sys.executable))
    log("归档 = {}".format(ARCHIVE))
    log("目标目录 = {}".format(HR_DIR))
    log("说明：双层归档，只提取 Flickr2K_HR 的 PNG；LR 不落盘；保留原始文件名与字节。")

    if not ARCHIVE.is_file():
        log("[FAIL] 归档不存在")
        _failures.append("归档不存在")
        LOG_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        return 1
    size = ARCHIVE.stat().st_size
    log("归档大小 = {} B ({:.2f} GiB)".format(size, size / 2**30))
    with ARCHIVE.open("rb") as f:
        head = f.read(8)
    log("归档头 8 字节 = {} （1f8b = gzip）".format(head.hex()))
    if not head.startswith(b"\x1f\x8b"):
        log("[FAIL] 不是 gzip 流")
        _failures.append("归档不是 gzip 流")

    HR_DIR.mkdir(parents=True, exist_ok=True)

    outer_members: list[tuple[str, int, str]] = []
    inner_total = 0
    inner_dirs = 0
    inner_hr = 0
    inner_skipped = 0
    inner_groups: Counter = Counter()
    suffixes: Counter = Counter()
    top_levels: set[str] = set()
    first_names: list[str] = []
    last_names: list[str] = []
    written = 0
    skipped_exists = 0
    conflicts: list[tuple[str, str]] = []
    total_written_bytes = 0
    mapping_examples: list[tuple[str, str]] = []
    inner_tar_name = ""

    try:
        log("开始流式遍历外层 gzip/tar，并直读内层 tar ...")
        with tarfile.open(ARCHIVE, "r:gz") as outer:
            for om in outer:
                outer_members.append((om.name, om.size, "dir" if om.isdir() else "file"))
                parts = om.name.replace("\\", "/").split("/")
                if parts:
                    top_levels.add(parts[0])
                if not om.isfile() or not om.name.lower().endswith(".tar"):
                    continue
                inner_tar_name = om.name
                log("检测到嵌套归档 = {} ({} B)".format(om.name, om.size))
                src = outer.extractfile(om)
                if src is None:
                    _failures.append("内层 tar 无法打开")
                    continue
                with tarfile.open(fileobj=src, mode="r|") as inner:
                    for im in inner:
                        name = im.name.replace("\\", "/")
                        inner_total += 1
                        if im.isdir():
                            inner_dirs += 1
                            continue
                        inner_groups[group_of(name)] += 1
                        suffixes[Path(name).suffix.lower()] += 1
                        if len(first_names) < 5:
                            first_names.append(name)
                        last_names.append(name)
                        if len(last_names) > 4:
                            last_names.pop(0)
                        if not is_hr_image(name):
                            inner_skipped += 1
                            continue
                        inner_hr += 1
                        basename = Path(name).name
                        dst = HR_DIR / basename
                        member_stream = inner.extractfile(im)
                        if member_stream is None:
                            conflicts.append((name, "extractfile returned None"))
                            continue
                        if dst.exists():
                            skipped_exists += 1
                            if len(conflicts) < 20:
                                conflicts.append((name, "target already exists: " + dst.name))
                            member_stream.close()
                            continue
                        with member_stream, dst.open("wb") as out:
                            shutil.copyfileobj(member_stream, out, length=1024 * 1024)
                        total_written_bytes += dst.stat().st_size
                        written += 1
                        if len(mapping_examples) < 8:
                            mapping_examples.append((name, "Flickr2K/HR/" + basename))
                        if written % 500 == 0:
                            log("  已写出 {}/{} ...".format(written, EXPECTED_FILES))
        log("外层 gzip/tar 流 = PASS（遍历至流末尾；CRC 亦由 verify_archives.py 独立校验）")
    except Exception as exc:
        log("[FAIL] 归档遍历失败：{}: {}".format(type(exc).__name__, exc))
        _failures.append("归档遍历失败：{}: {}".format(type(exc).__name__, exc))

    section("归档结构与目录映射")
    log("外层成员 = {}".format(outer_members))
    log("外层顶层条目 = {}".format(sorted(top_levels)[:10]))
    log("内层归档 = {}".format(inner_tar_name))
    log("内层成员总数 = {} （文件 {}, 目录 {}）".format(
        inner_total, inner_total - inner_dirs, inner_dirs))
    log("内层 HR 图像成员数 = {}".format(inner_hr))
    log("内层跳过成员数（LR 等，未落盘）= {}".format(inner_skipped))
    log("内层目录分组（前 15 组）:")
    for k, v in inner_groups.most_common(15):
        log("  {:42s} {}".format(k, v))
    log("前 5 个成员 = {}".format(first_names))
    log("后 4 个成员 = {}".format(last_names))
    log("扩展名分布 = {}".format(dict(suffixes)))
    log("映射规则：Flickr2K/Flickr2K_HR/<basename> -> datasets/Flickr2K/HR/<basename>")
    log("映射示例：")
    for a, b in mapping_examples:
        log("  {}  ->  {}".format(a, b))
    log("说明：归档内 6 组 LR 图像（unknown/bicubic 的 X2/X3/X4）按任务书不提取、不使用；")
    log("      后续阶段将从 HR 统一按 Bicubic 规则生成 ×2 / ×4 的 LR。")

    section("写出结果")
    log("实际写出图像数 = {} （预期 {}）".format(written, EXPECTED_FILES))
    log("写出总字节 = {} ({:.2f} GiB)".format(total_written_bytes, total_written_bytes / 2**30))
    log("因目标已存在而跳过 = {}".format(skipped_exists))
    log("冲突/异常记录数 = {}".format(len(conflicts)))
    for name, why in conflicts[:20]:
        log("  CONFLICT {}: {}".format(name, why))
    actual_files = len([p for p in HR_DIR.iterdir() if p.is_file()])
    log("HR 目录当前文件数 = {}".format(actual_files))

    section("汇总 / SUMMARY")
    if not inner_tar_name:
        _failures.append("未找到内层 tar")
    if inner_hr != EXPECTED_FILES:
        _failures.append("归档内 HR 图像数不符：{}/{}".format(inner_hr, EXPECTED_FILES))
    if written != EXPECTED_FILES:
        _failures.append("实际写出数不符：{}/{}".format(written, EXPECTED_FILES))
    if actual_files != EXPECTED_FILES:
        _failures.append("HR 目录最终文件数不符：{}/{}".format(actual_files, EXPECTED_FILES))
    if conflicts:
        _failures.append("存在 {} 条冲突/异常".format(len(conflicts)))
    log("耗时 = {:.1f} s".format(time.perf_counter() - started))
    if _failures:
        for f in _failures:
            log("  FAIL: " + f)
        log("RESULT = FAIL（{} 项）".format(len(_failures)))
        result = 1
    else:
        log("RESULT = PASS（双层归档流完整，2650 张 HR 原始图像已写出且文件名保留）")
        result = 0
    log("注意：本脚本不判断图像是否可解码，那一步由 verify_datasets.py 完成。")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    sys.exit(main())
