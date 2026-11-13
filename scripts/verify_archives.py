# -*- coding: utf-8 -*-
"""阶段 2 原始压缩包完整性与结构检查（只读，不解压、不修改归档）。

运行方式::

    "D:\\Environment\\Python\\Python_AI\\python.exe" scripts\\verify_archives.py

检查：归档文件大小、本地 SHA-256 指纹、ZIP 成员 CRC、gzip/tar 流完整性、
归档内部目录结构（用于说明「压缩包结构 → 最终目录」的对应关系）。

注意：SHA-256 是**本地文件指纹**，不是官方校验值。
官方未提供校验和时，不得把本地指纹声称成官方校验值。
产出：logs/phase02_archives.log
"""
from __future__ import annotations

import hashlib
import sys
import tarfile
import time
import zipfile
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES_DIR = ROOT / "datasets" / "_archives"
LOG_PATH = ROOT / "logs" / "phase02_archives.log"

EXPECTED_SIZES = {
    "DIV2K_train_HR.zip": 3530603713,
    "DIV2K_valid_HR.zip": 448993893,
    "Flickr2K.tar.gz.00": 21510576903,
}

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


def sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def top_levels(names: list[str]) -> list[str]:
    tops = set()
    for n in names:
        parts = n.replace("\\", "/").split("/")
        tops.add(parts[0] if parts else n)
    return sorted(tops)


def check_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        files = [i for i in infos if not i.is_dir()]
        log("归档类型       = ZIP")
        log("成员总数       = {} （文件 {}, 目录 {}）".format(
            len(infos), len(files), len(infos) - len(files)))
        log("未压缩总字节   = {}".format(sum(i.file_size for i in files)))
        log("顶层条目       = {}".format(top_levels(names)[:10]))
        log("前 8 个成员    = {}".format(names[:8]))
        log("后 4 个成员    = {}".format(names[-4:]))
        log("正在逐成员校验 CRC (testzip) ...")
        bad = z.testzip()
        if bad is None:
            log("CRC 校验       = PASS（全部成员 CRC 正确）")
        else:
            log("CRC 校验       = FAIL（首个损坏成员：{}）".format(bad))
            _failures.append("{} ZIP CRC 损坏：{}".format(path.name, bad))
        suffixes = Counter(Path(n).suffix.lower() for n in names if not n.endswith("/"))
        log("扩展名分布     = {}".format(dict(suffixes)))


def check_targz(path: Path) -> None:
    log("归档类型       = gzip 压缩的 tar 流（按内容识别，不依赖扩展名）")
    log("正在流式遍历整个归档（同时校验 gzip CRC / trailer）...")
    with tarfile.open(path, "r:gz") as t:
        members = t.getmembers()
    names = [m.name for m in members]
    files = [m for m in members if m.isfile()]
    dirs = [m for m in members if m.isdir()]
    log("gzip/tar 流     = PASS（可完整遍历到流末尾，CRC 校验通过）")
    log("成员总数       = {} （文件 {}, 目录 {}）".format(len(members), len(files), len(dirs)))
    log("未压缩总字节   = {}".format(sum(m.size for m in files)))
    log("顶层条目       = {}".format(top_levels(names)[:10]))
    log("前 8 个成员    = {}".format(names[:8]))
    log("后 4 个成员    = {}".format(names[-4:]))
    suffixes = Counter(Path(n).suffix.lower() for n in names if not n.endswith("/"))
    log("扩展名分布     = {}".format(dict(suffixes)))
    log("注：成员目录结构用于说明解压后应如何映射到 datasets/Flickr2K/HR/。")


def main() -> int:
    started = time.perf_counter()
    section("阶段 2 原始压缩包完整性与结构检查 / verify_archives.py")
    log("解释器 = {}".format(sys.executable))
    log("归档目录 = {}".format(ARCHIVES_DIR))
    log("说明：SHA-256 为本地文件指纹，不是官方校验值；本脚本不解压、不修改归档。")

    if not ARCHIVES_DIR.is_dir():
        log("[FAIL] 归档目录不存在")
        _failures.append("归档目录不存在")

    archives = sorted(p for p in ARCHIVES_DIR.glob("*") if p.is_file()) \
        if ARCHIVES_DIR.is_dir() else []
    for path in archives:
        section("归档：{}".format(path.name))
        size = path.stat().st_size
        expected = EXPECTED_SIZES.get(path.name)
        log("文件大小       = {} B ({:.2f} GiB)".format(size, size / 2**30))
        if expected is not None:
            log("来源声明大小   = {} B".format(expected))
            log("大小一致       = {}".format(size == expected))
            if size != expected:
                _failures.append("{} 大小不符：{} != {}".format(path.name, size, expected))
        log("计算本地 SHA-256 中 ...")
        log("SHA-256(本地指纹) = {}".format(sha256_of(path)))
        try:
            if path.suffix.lower() == ".zip" or zipfile.is_zipfile(path):
                check_zip(path)
            else:
                check_targz(path)
        except Exception as exc:
            log("完整性检查     = FAIL（{}: {}）".format(type(exc).__name__, exc))
            _failures.append("{} 完整性检查失败：{}: {}".format(path.name, type(exc).__name__, exc))

    section("官方校验和说明")
    log("DIV2K 官方页面与 Flickr2K 官方链接本次均未提供可核验的官方 SHA-256 / MD5 值。")
    log("本脚本计算的 SHA-256 全部记录为【本地文件指纹】，不得称为官方校验值。")

    section("汇总 / SUMMARY")
    log("耗时 = {:.1f} s".format(time.perf_counter() - started))
    if _failures:
        for f in _failures:
            log("  FAIL: " + f)
        log("RESULT = FAIL（{} 项）".format(len(_failures)))
        result = 1
    else:
        log("RESULT = PASS（归档大小符合来源声明、结构完整、CRC/流校验通过）")
        result = 0

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    sys.exit(main())
