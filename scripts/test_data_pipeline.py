# -*- coding: utf-8 -*-
"""阶段 5 数据管线测试脚本（D-1 ~ D-13）。

运行方式（CMD）::

    "D:\\Environment\\Python\\Python_AI\\python.exe" scripts\\test_data_pipeline.py
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.data.degradation as deg_mod  # noqa: E402
from src.data import (  # noqa: E402
    SeededEpochSampler,
    SRDataset,
    apply_dihedral,
    collect_image_paths,
    downsample_bicubic,
    make_lr_from_patch,
)

TRAIN_DIRS = [
    PROJECT_ROOT / "datasets" / "DIV2K" / "DIV2K_train_HR",
    PROJECT_ROOT / "datasets" / "Flickr2K" / "HR",
]
VAL_DIRS = [
    PROJECT_ROOT / "datasets" / "DIV2K" / "DIV2K_valid_HR",
]


@dataclass
class TestRecord:
    code: str
    name: str
    passed: bool
    detail: str = ""


def _assert_raises(exc_type: type[BaseException], fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:
        raise AssertionError(
            f"期望抛出 {exc_type.__name__}，实际抛出 {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"期望抛出 {exc_type.__name__}，但未抛出任何异常")


def run_data_pipeline_tests() -> list[TestRecord]:
    records: list[TestRecord] = []

    def check(code: str, name: str, fn) -> None:
        try:
            detail = fn() or "OK"
            records.append(TestRecord(code=code, name=name, passed=True, detail=detail))
        except Exception as exc:
            records.append(TestRecord(code=code, name=name, passed=False, detail=str(exc)))

    def test_d1() -> str:
        rng = np.random.default_rng(1)
        hr = rng.integers(0, 256, size=(128, 160, 3), dtype=np.uint8)
        for scale in (2, 4):
            lr = downsample_bicubic(hr, scale)
            ref = np.asarray(
                Image.fromarray(hr).resize(
                    (160 // scale, 128 // scale), resample=Image.Resampling.BICUBIC
                ),
                dtype=np.uint8,
            )
            diff = int(np.max(np.abs(lr.astype(np.int16) - ref.astype(np.int16))))
            assert diff == 0, f"scale={scale} maxdiff={diff} != 0"
        return "maxdiff = 0 (2x & 4x)"

    def test_d2() -> str:
        rng = np.random.default_rng(2)
        hr = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        for scale in (2, 4):
            lr = downsample_bicubic(hr, scale)
            cv_lr = cv2.resize(
                hr, (128 // scale, 128 // scale), interpolation=cv2.INTER_CUBIC
            )
            diff = int(np.max(np.abs(lr.astype(np.int16) - cv_lr.astype(np.int16))))
            assert diff > 0, f"scale={scale} 退化结果竟与 cv2.INTER_CUBIC 完全一致"
        return "maxdiff > 0（反例守卫通过）"

    def test_d3() -> str:
        paths = collect_image_paths(TRAIN_DIRS)[:50]
        assert len(paths) == 50
        for p in paths:
            with Image.open(p) as im:
                w, h = im.size
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            for scale in (2, 4):
                lr = downsample_bicubic(dummy, scale)
                lh, lw = lr.shape[:2]
                assert lh * scale <= h and lw * scale <= w
                assert 0 <= h - lh * scale < scale
                assert 0 <= w - lw * scale < scale
        return "50 张真实训练图 2x/4x 尺寸取整守卫全部通过"

    def test_d4() -> str:
        hr = np.zeros((1355, 2039, 3), dtype=np.uint8)
        for scale in (2, 4):
            lr = downsample_bicubic(hr, scale)
            assert lr.shape == (1355 // scale, 2039 // scale, 3)
        return "1355x2039 非整除尺寸 2x->677x1019, 4x->338x509 正确"

    def test_d5() -> str:
        img_path = PROJECT_ROOT / "datasets" / "DIV2K" / "DIV2K_valid_HR" / "0801.png"
        with Image.open(img_path) as im:
            hr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        y0, x0, ps = 128, 256, 128
        for scale in (2, 4):
            # 方式 A：先裁后降
            lr_crop_first = make_lr_from_patch(hr[y0 : y0 + ps, x0 : x0 + ps], scale)
            # 方式 B：整图先降再裁
            h_al = (hr.shape[0] // scale) * scale
            w_al = (hr.shape[1] // scale) * scale
            lr_full = downsample_bicubic(hr[:h_al, :w_al], scale)
            ly0, lx0, lps = y0 // scale, x0 // scale, ps // scale
            lr_down_first = lr_full[ly0 : ly0 + lps, lx0 : lx0 + lps]
            # 去掉边界 2 像素后应逐像素一致
            diff = int(
                np.max(
                    np.abs(
                        lr_crop_first[2:-2, 2:-2].astype(np.int16)
                        - lr_down_first[2:-2, 2:-2].astype(np.int16)
                    )
                )
            )
            assert diff == 0, f"scale={scale} 内部区域存在差异 maxdiff={diff}"
        return "内部区域（去边界 2px）maxdiff = 0"

    def test_d6() -> str:
        # 构造非对称测试图案，验证 8 种二面体变换均唯一且 4 次旋转可复原
        rng = np.random.default_rng(6)
        patch = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        variants = [apply_dihedral(patch, i).tobytes() for i in range(8)]
        assert len(set(variants)) == 8, f"8 种二面体变换未产生 8 种唯一结果: {len(set(variants))}"
        # 验证可逆性：不翻转的旋转做 4 次后还原；翻转后再翻转还原
        for i in range(8):
            t = apply_dihedral(patch, i)
            if i < 4:
                inv = apply_dihedral(t, (4 - i) % 4)
            else:
                inv = apply_dihedral(t, i)  # 反射操作自反（f(f(x)) == x）
            assert np.array_equal(inv, patch), f"op_id={i} 不可逆"
        return "8 种二面体变换唯一且全部可逆"

    def test_d7() -> str:
        rng = np.random.default_rng(7)
        hr_patch = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        for op_id in range(8):
            hr_aug = apply_dihedral(hr_patch, op_id)
            lr_aug = make_lr_from_patch(hr_aug, 4)
            # 校验均值颜色空间与角点对应关系保持一致
            assert lr_aug.shape == (32, 32, 3)
            # 左上角 4x4 块均值应接近 LR 左上角像素
            tl_hr_mean = hr_aug[:4, :4].mean()
            tl_lr_val = float(lr_aug[0, 0].mean())
            assert abs(tl_hr_mean - tl_lr_val) < 40.0
        return "先增强后降采样 HR/LR 空间对应关系一致"

    def test_d8() -> str:
        train_paths = collect_image_paths(TRAIN_DIRS)
        val_paths = collect_image_paths(VAL_DIRS)
        assert len(train_paths) == 3450 and len(val_paths) == 100
        overlap = {p.resolve() for p in train_paths} & {p.resolve() for p in val_paths}
        assert len(overlap) == 0, f"发现交集: {len(overlap)}"
        return "训练集 3450 与验证集 100 路径交集为空"

    def test_d9() -> str:
        train_paths = collect_image_paths(TRAIN_DIRS)[:10]
        ds1 = SRDataset(train_paths, scale=2, patch_size=128, base_seed=42, training=True, use_preupsample=True)
        ds2 = SRDataset(train_paths, scale=2, patch_size=128, base_seed=42, training=True, use_preupsample=True)
        ds1.set_epoch(3)
        ds2.set_epoch(3)
        a = ds1[5]
        b = ds2[5]
        for ta, tb in zip(a, b):
            assert torch.equal(ta, tb), "同 seed 同 epoch 同 index 样本不一致"
        return "同 seed 同 index 三元组逐位相等"

    def test_d10() -> str:
        sampler = SeededEpochSampler(data_source_len=100, base_seed=42)
        sampler.set_epoch(1)
        order1 = list(sampler)
        sampler.set_epoch(2)
        order2 = list(sampler)
        diff_pos = sum(1 for x, y in zip(order1, order2) if x != y)
        assert diff_pos >= 2, f"不同 epoch 顺序差异不足: {diff_pos}"
        return f"Epoch 1 与 Epoch 2 有 {diff_pos}/100 个位置不同"

    def test_d11() -> str:
        train_paths = collect_image_paths(TRAIN_DIRS)[:4]
        for scale in (2, 4):
            ds_espcn = SRDataset(train_paths, scale=scale, patch_size=128, training=True, use_preupsample=False)
            hr, lr = ds_espcn[0]
            assert hr.shape == (3, 128, 128) and lr.shape == (3, 128 // scale, 128 // scale)

            ds_srcnn = SRDataset(train_paths, scale=scale, patch_size=128, training=True, use_preupsample=True)
            hr2, lr2, lr_up = ds_srcnn[0]
            assert hr2.shape == (3, 128, 128) and lr2.shape == (3, 128 // scale, 128 // scale)
            assert lr_up.shape == (3, 128, 128)
        return "HR=128x128, LR=128/scale, LR_up=128x128 形状全部精确匹配"

    def test_d12() -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            small_path = Path(tmpdir) / "small_80x90.png"
            rng = np.random.default_rng(12)
            small_img = rng.integers(0, 256, size=(80, 90, 3), dtype=np.uint8)
            Image.fromarray(small_img).save(small_path)
            ds = SRDataset([small_path], scale=4, patch_size=128, training=True, use_preupsample=False)
            hr, lr = ds[0]
            assert hr.shape == (3, 128, 128) and lr.shape == (3, 32, 32)
            assert ds.padded_count == 1 and ds.padded_pixels_total > 0
        return "80x90 小图 reflect 补边后正确输出 128x128 / 32x32"

    def test_d13() -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_path = Path(tmpdir) / "tiny_30x30.png"
            tiny_img = np.zeros((30, 30, 3), dtype=np.uint8)
            Image.fromarray(tiny_img).save(tiny_path)
            ds = SRDataset([tiny_path], scale=4, patch_size=128, training=True, use_preupsample=False)
            _assert_raises(ValueError, ds.__getitem__, 0)
        return "30x30 过小图像单次 reflect 无法补足 128 时正确抛出 ValueError"

    check("D-1", "downsample_bicubic 与 PIL.Image.resize 相等", test_d1)
    check("D-2", "downsample_bicubic 与 cv2.INTER_CUBIC 不等", test_d2)
    check("D-3", "尺寸取整守卫（50张实测）", test_d3)
    check("D-4", "非整除尺寸 1355x2039 处理", test_d4)
    check("D-5", "先裁后降 vs 先降后裁内部一致性", test_d5)
    check("D-6", "二面体增强 8 种可达且可逆", test_d6)
    check("D-7", "增强后 HR/LR 空间对应关系", test_d7)
    check("D-8", "训练/验证路径无交集", test_d8)
    check("D-9", "同 seed 同 index 样本逐位相等", test_d9)
    check("D-10", "不同 epoch 采样顺序不同", test_d10)
    check("D-11", "HR/LR/LR_up Patch 形状精确匹配", test_d11)
    check("D-12", "短边 <128 小图补边不崩", test_d12)
    check("D-13", "过小图像补边不足抛出 ValueError", test_d13)
    return records


def run_data_mutation_probes() -> list[tuple[str, bool, str]]:
    """运行内存变异探针（不修改磁盘源码），验证守卫测试能正确变红。"""
    results: list[tuple[str, bool, str]] = []

    # 变异 T-8：临时将 downsample_bicubic 替换为 cv2.INTER_CUBIC
    orig_fn = deg_mod.downsample_bicubic
    try:
        def _mutated_cv2(hr: np.ndarray, scale: int) -> np.ndarray:
            h, w = hr.shape[:2]
            return cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)

        deg_mod.downsample_bicubic = _mutated_cv2  # type: ignore[assignment]
        globals()["downsample_bicubic"] = _mutated_cv2
        recs = run_data_pipeline_tests()
        failed_codes = [r.code for r in recs if not r.passed]
        caught = ("D-1" in failed_codes) and ("D-2" in failed_codes)
        results.append(("T-8(退化改cv2)", caught, f"触发 FAIL 用例: {failed_codes}"))
    finally:
        deg_mod.downsample_bicubic = orig_fn
        globals()["downsample_bicubic"] = orig_fn

    # 变异 T-10：临时将验证集目录设为与训练集重叠
    orig_val = list(VAL_DIRS)
    try:
        VAL_DIRS[:] = [PROJECT_ROOT / "datasets" / "DIV2K" / "DIV2K_train_HR"]
        recs = run_data_pipeline_tests()
        failed_codes = [r.code for r in recs if not r.passed]
        caught = "D-8" in failed_codes
        results.append(("T-10(训练/验证对调)", caught, f"触发 FAIL 用例: {failed_codes}"))
    finally:
        VAL_DIRS[:] = orig_val

    return results


def main() -> int:
    print("=" * 72)
    print("阶段 5：数据管线测试（D-1 ~ D-13）")
    print("=" * 72)

    records = run_data_pipeline_tests()
    passed = sum(1 for r in records if r.passed)
    failed = len(records) - passed
    for r in records:
        tag = "PASS" if r.passed else "FAIL"
        print(f"  [{tag}] {r.code} {r.name}: {r.detail}")
    print(f"\n汇总: 通过 {passed} / {len(records)}，失败 {failed}")

    if failed > 0:
        return 1

    print("\n[变异守卫探针（预期改坏后必须触发 FAIL）]：")
    for m_name, caught, detail in run_data_mutation_probes():
        tag = "CAUGHT" if caught else "MISSED"
        print(f"  [{tag}] {m_name} -> {detail}")
        if not caught:
            return 1

    print("\nRESULT = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
