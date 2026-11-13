# -*- coding: utf-8 -*-
"""阶段 3 传统图像插值算法测试与演示脚本。

运行方式（使用项目指定 Python 解释器）::

    "D:\\Environment\\Python\\Python_AI\\python.exe" scripts\\test_interpolation.py

测试内容：
1. 功能测试：三种算法调用、2x/4x 尺寸、RGB 通道、输入保护、异常校验、算法输出差异。
2. 边界测试：正方形、非正方形（横/竖）、极小尺寸（1x1、4x4）、float32 数据类型。
3. 实际图像测试：只读读取 DIV2K Validation 中的少量图像，在内存中生成临时演示用 LR 输入，
   分别使用 nearest / bilinear / bicubic 进行 2x 与 4x 放大，记录纯插值耗时，
   并保存各算法输出与带标注的局部细节对比图至 ``results/interpolation/``。

注意：
- 本脚本不修改 ``datasets/`` 中的任何原始图像。
- 临时生成的演示用 LR 输入仅保存在 ``results/interpolation/inputs/``，不混入训练数据。
- 耗时仅为单次运行观测值，不作为严谨性能基准；本阶段不计算正式 PSNR/SSIM 成绩。
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.processing.interpolation import (  # noqa: E402
    INTERPOLATION_METHODS,
    SUPPORTED_SCALES,
    upscale_image,
)

DIV2K_VALID_DIR = PROJECT_ROOT / "datasets" / "DIV2K" / "DIV2K_valid_HR"
RESULTS_ROOT = PROJECT_ROOT / "results" / "interpolation"
DEMO_IMAGES = ["0801.png", "0802.png", "0803.png"]
TIMING_WARMUP = 1
TIMING_REPEATS = 5


@dataclass
class TestRecord:
    """单条单元测试结果。"""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class TimingRecord:
    """单次实际图像插值耗时统计记录（含预热后多次重复）。"""

    image_name: str
    method: str
    scale: int
    input_size: tuple[int, int]  # (W, H)
    output_size: tuple[int, int]  # (W, H)
    mean_ms: float
    min_ms: float
    max_ms: float


def _assert_raises(exc_type: type[BaseException], fn, *args, **kwargs) -> None:
    """断言函数调用会抛出指定类型异常。"""
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:  # pragma: no cover - 仅在断言失败时触发
        raise AssertionError(
            f"期望抛出 {exc_type.__name__}，实际抛出 {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"期望抛出 {exc_type.__name__}，但未抛出任何异常")


def run_unit_tests() -> list[TestRecord]:
    """执行功能测试与边界测试。"""
    records: list[TestRecord] = []

    def check(name: str, fn) -> None:
        try:
            detail = fn() or "OK"
            records.append(TestRecord(name=name, passed=True, detail=detail))
        except Exception as exc:
            records.append(TestRecord(name=name, passed=False, detail=str(exc)))

    # ---- 1. 功能测试 ----
    def test_three_methods_and_distinct_outputs() -> str:
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
        outputs = {}
        for method in ("nearest", "bilinear", "bicubic"):
            out = upscale_image(img, 2, method)
            assert out.shape == (64, 96, 3), f"{method} shape={out.shape}"
            assert out.dtype == np.uint8
            outputs[method] = out
        assert not np.array_equal(outputs["nearest"], outputs["bilinear"]), (
            "nearest 与 bilinear 输出完全一致，可能未区分插值方式"
        )
        assert not np.array_equal(outputs["bilinear"], outputs["bicubic"]), (
            "bilinear 与 bicubic 输出完全一致，可能未区分插值方式"
        )
        assert not np.array_equal(outputs["nearest"], outputs["bicubic"]), (
            "nearest 与 bicubic 输出完全一致，可能未区分插值方式"
        )
        return "三种算法均可调用且输出存在预期差异"

    def test_scale_2x_and_4x_exact_size() -> str:
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        for method in INTERPOLATION_METHODS:
            out2 = upscale_image(img, 2, method)
            assert out2.shape == (480, 640, 3), f"{method} 2x: {out2.shape}"
            out4 = upscale_image(img, 4, method)
            assert out4.shape == (960, 1280, 3), f"{method} 4x: {out4.shape}"
        return "320x240 -> 640x480 (2x) & 1280x960 (4x) 严格匹配"

    def test_rgb_channel_preservation() -> str:
        # 构造左红、中绿、右蓝三色条纹图，验证 RGB 通道顺序未被交换
        img = np.zeros((16, 24, 3), dtype=np.uint8)
        img[:, :8, 0] = 255   # R
        img[:, 8:16, 1] = 255  # G
        img[:, 16:, 2] = 255   # B
        for scale in SUPPORTED_SCALES:
            for method in INTERPOLATION_METHODS:
                out = upscale_image(img, scale, method)
                assert out.ndim == 3 and out.shape[2] == 3
                # 检查色块中心像素仍保持纯 R / G / B
                cy = out.shape[0] // 2
                assert tuple(out[cy, 2 * scale]) == (255, 0, 0)
                assert tuple(out[cy, 12 * scale]) == (0, 255, 0)
                assert tuple(out[cy, 20 * scale]) == (0, 0, 255)
        return "RGB 三通道数量及顺序在 2x/4x 下均保持一致"

    def test_input_immutability() -> str:
        rng = np.random.default_rng(7)
        img = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
        snapshot = img.copy()
        for scale in SUPPORTED_SCALES:
            for method in INTERPOLATION_METHODS:
                out = upscale_image(img, scale, method)
                assert out is not img
                assert np.array_equal(img, snapshot), f"{method} x{scale} 修改了原图"
        return "原始输入数组在全部算法与倍率下均未被修改"

    def test_invalid_method() -> str:
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        for bad_method in ("lanczos", "area", "", "   "):
            _assert_raises(ValueError, upscale_image, img, 2, bad_method)
        for bad_type in (None, 123, ["nearest"]):
            _assert_raises(TypeError, upscale_image, img, 2, bad_type)
        # 验证大小写与前后空白规范化
        out = upscale_image(img, 2, "  Bicubic ")
        assert out.shape == (32, 32, 3)
        return "非法算法名正确抛出 ValueError/TypeError"

    def test_invalid_scale() -> str:
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        for bad_scale in (0, 1, 3, -2, 8):
            _assert_raises(ValueError, upscale_image, img, bad_scale, "nearest")
        for bad_type in (2.0, True, False, "2", None):
            _assert_raises(TypeError, upscale_image, img, bad_type, "nearest")
        return "非法倍率（含 bool/float/非2非4整数）正确报错"

    def test_empty_and_invalid_image() -> str:
        _assert_raises(ValueError, upscale_image, np.empty((0, 0, 3), dtype=np.uint8), 2, "nearest")
        _assert_raises(ValueError, upscale_image, np.zeros((0, 16, 3), dtype=np.uint8), 2, "bilinear")
        _assert_raises(ValueError, upscale_image, np.zeros((16, 16), dtype=np.uint8), 2, "bicubic")
        _assert_raises(ValueError, upscale_image, np.zeros((16, 16, 1), dtype=np.uint8), 2, "nearest")
        _assert_raises(ValueError, upscale_image, np.zeros((16, 16, 4), dtype=np.uint8), 2, "nearest")
        _assert_raises(TypeError, upscale_image, None, 2, "nearest")
        _assert_raises(TypeError, upscale_image, [[[1, 2, 3]]], 2, "nearest")
        _assert_raises(TypeError, upscale_image, np.zeros((8, 8, 3), dtype=bool), 2, "nearest")
        return "空图像、非 3 通道图像及非 ndarray/不支持 dtype 均正确报错"

    # ---- 2. 边界测试 ----
    def test_square_image() -> str:
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        for scale in SUPPORTED_SCALES:
            for method in INTERPOLATION_METHODS:
                out = upscale_image(img, scale, method)
                assert out.shape == (64 * scale, 64 * scale, 3)
        return "正方形 64x64 图像 2x/4x 尺寸正确"

    def test_non_square_images() -> str:
        for h, w in ((37, 83), (120, 45)):
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for scale in SUPPORTED_SCALES:
                for method in INTERPOLATION_METHODS:
                    out = upscale_image(img, scale, method)
                    assert out.shape == (h * scale, w * scale, 3)
        return "非正方形横图 (83x37) 与竖图 (45x120) 2x/4x 尺寸正确"

    def test_tiny_images() -> str:
        for h, w in ((1, 1), (4, 4)):
            img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
            for scale in SUPPORTED_SCALES:
                for method in INTERPOLATION_METHODS:
                    out = upscale_image(img, scale, method)
                    assert out.shape == (h * scale, w * scale, 3)
        return "极小尺寸 1x1 与 4x4 图像 2x/4x 尺寸正确"

    def test_float32_rgb_image() -> str:
        img = np.linspace(0.0, 1.0, 20 * 30 * 3, dtype=np.float32).reshape(20, 30, 3)
        for scale in SUPPORTED_SCALES:
            for method in INTERPOLATION_METHODS:
                out = upscale_image(img, scale, method)
                assert out.shape == (20 * scale, 30 * scale, 3)
                assert out.dtype == np.float32
        return "float32 RGB 图像在 2x/4x 下保持 dtype 与尺寸正确"

    def test_semantic_oracle_nearest_bilinear_bicubic() -> str:
        rng = np.random.default_rng(123)
        img = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
        for scale in SUPPORTED_SCALES:
            # 1) Nearest 必须逐像素等于 np.repeat（maxdiff == 0）
            out_nearest = upscale_image(img, scale, "nearest")
            expected_nearest = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
            assert np.array_equal(out_nearest, expected_nearest), (
                f"nearest x{scale} 与 np.repeat 不一致"
            )

            # 2) Bilinear 与 PIL BILINEAR 独立实现最大绝对差 <= 1
            out_bilinear = upscale_image(img, scale, "bilinear")
            pil_bilinear = np.asarray(
                Image.fromarray(img).resize(
                    (img.shape[1] * scale, img.shape[0] * scale),
                    resample=Image.Resampling.BILINEAR,
                ),
                dtype=np.uint8,
            )
            max_diff = int(
                np.max(np.abs(out_bilinear.astype(np.int16) - pil_bilinear.astype(np.int16)))
            )
            assert max_diff <= 1, f"bilinear x{scale} 与 PIL BILINEAR 最大差={max_diff} > 1"

        # 3) 强阶跃边缘上，Bicubic 存在负权重核引起的过冲/下冲（振铃），Bilinear 严格有界
        step_img = np.full((16, 16, 3), 50, dtype=np.uint8)
        step_img[:, 8:, :] = 200
        step_bilinear = upscale_image(step_img, 4, "bilinear")
        step_bicubic = upscale_image(step_img, 4, "bicubic")
        assert int(step_bilinear.min()) >= 50 and int(step_bilinear.max()) <= 200, (
            f"bilinear 在阶跃边缘出现越界: [{step_bilinear.min()}, {step_bilinear.max()}]"
        )
        assert int(step_bicubic.min()) < 50 and int(step_bicubic.max()) > 200, (
            f"bicubic 在阶跃边缘未出现预期过冲/下冲（可能误映射到 bilinear）: "
            f"[{step_bicubic.min()}, {step_bicubic.max()}]"
        )
        return "独立语义校验通过（nearest==np.repeat, bilinear~PIL<=1, bicubic阶跃过冲）"

    check("功能-三种算法调用与输出差异", test_three_methods_and_distinct_outputs)
    check("功能-三种算法语义正确性校验", test_semantic_oracle_nearest_bilinear_bicubic)
    check("功能-2x与4x输出尺寸核验", test_scale_2x_and_4x_exact_size)
    check("功能-RGB通道数与顺序保持", test_rgb_channel_preservation)
    check("功能-原始输入图像不被修改", test_input_immutability)
    check("功能-非法算法名称异常处理", test_invalid_method)
    check("功能-非法放大倍率异常处理", test_invalid_scale)
    check("功能-空图像与非法输入异常处理", test_empty_and_invalid_image)
    check("边界-正方形图像", test_square_image)
    check("边界-非正方形横竖图像", test_non_square_images)
    check("边界-极小尺寸图像(1x1/4x4)", test_tiny_images)
    check("边界-float32常规RGB图像", test_float32_rgb_image)
    return records


def _load_rgb_image(path: Path) -> np.ndarray:
    """只读加载图像为 RGB uint8 NumPy 数组。"""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _save_rgb_image(rgb: np.ndarray, path: Path) -> None:
    """保存 RGB uint8 NumPy 数组为 PNG。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, format="PNG")


def _make_demo_lr(hr_rgb: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """从 HR 图像在内存中裁剪可整除区域并下采样生成临时演示用 LR 输入。

    下采样方式：``PIL.Image.Resampling.BICUBIC``（仅用于阶段 3 演示，不写入 datasets/）。
    """
    h, w = hr_rgb.shape[:2]
    h_aligned = (h // scale) * scale
    w_aligned = (w // scale) * scale
    hr_cropped = hr_rgb[:h_aligned, :w_aligned, :]
    lr_w = w_aligned // scale
    lr_h = h_aligned // scale
    lr_pil = Image.fromarray(hr_cropped).resize(
        (lr_w, lr_h), resample=Image.Resampling.BICUBIC
    )
    return np.asarray(lr_pil, dtype=np.uint8), hr_cropped


def _find_high_detail_crop(hr_rgb: np.ndarray, crop_size: int) -> tuple[int, int]:
    """选取纹理梯度较高的局部区域左上角坐标，便于对比插值细节差异。"""
    h, w = hr_rgb.shape[:2]
    crop_h = min(crop_size, h)
    crop_w = min(crop_size, w)
    gray = cv2.cvtColor(hr_rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = np.abs(grad_x) + np.abs(grad_y)

    # 网格采样候选窗口，选平均梯度能量最高的窗口
    best_score = -1.0
    best_yx = ((h - crop_h) // 2, (w - crop_w) // 2)
    step_y = max(1, (h - crop_h) // 8)
    step_x = max(1, (w - crop_w) // 8)
    for y in range(0, max(1, h - crop_h + 1), step_y):
        for x in range(0, max(1, w - crop_w + 1), step_x):
            score = float(energy[y : y + crop_h, x : x + crop_w].mean())
            if score > best_score:
                best_score = score
                best_yx = (y, x)
    return best_yx


def _add_banner(panel_rgb: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    """在图像上方添加深色标题栏以便清晰区分算法与倍率。"""
    banner_h = 56
    h, w = panel_rgb.shape[:2]
    canvas = np.full((h + banner_h, w, 3), fill_value=24, dtype=np.uint8)
    canvas[banner_h:, :, :] = panel_rgb
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.putText(
        bgr,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr,
        subtitle,
        (12, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (190, 210, 230),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _build_comparison_grid(
    image_stem: str,
    scale: int,
    lr_rgb: np.ndarray,
    outputs: dict[str, np.ndarray],
    hr_ref: np.ndarray,
) -> np.ndarray:
    """生成 2x2 局部细节对比图（含 Nearest / Bilinear / Bicubic 与 HR 参考）。"""
    h, w = hr_ref.shape[:2]
    crop_size = min(256, h, w)
    y0, x0 = _find_high_detail_crop(hr_ref, crop_size)

    panel_size = 420
    panels = []
    specs = [
        ("nearest", f"{image_stem} | Nearest {scale}x", outputs["nearest"]),
        ("bilinear", f"{image_stem} | Bilinear {scale}x", outputs["bilinear"]),
        ("bicubic", f"{image_stem} | Bicubic {scale}x", outputs["bicubic"]),
        (
            "hr_ref",
            f"{image_stem} | HR Reference",
            hr_ref,
        ),
    ]
    lr_h, lr_w = lr_rgb.shape[:2]
    out_h, out_w = outputs["bicubic"].shape[:2]

    for key, title, img in specs:
        patch = img[y0 : y0 + crop_size, x0 : x0 + crop_size, :]
        # 将裁剪块按最近邻等比放大到固定面板尺寸，保留像素级锯齿/平滑差异便于观察
        patch_vis = cv2.resize(
            patch, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST
        )
        if key == "hr_ref":
            subtitle = f"Crop ({x0},{y0}) {crop_size}x{crop_size} | Full: {w}x{h}"
        else:
            subtitle = (
                f"In: {lr_w}x{lr_h} -> Out: {out_w}x{out_h} | "
                f"Crop {crop_size}x{crop_size}"
            )
        panels.append(_add_banner(patch_vis, title, subtitle))

    gap = 12
    row_h, col_w = panels[0].shape[:2]
    grid = np.full(
        (row_h * 2 + gap * 3, col_w * 2 + gap * 3, 3),
        fill_value=240,
        dtype=np.uint8,
    )
    positions = [
        (gap, gap),
        (gap, gap * 2 + col_w),
        (gap * 2 + row_h, gap),
        (gap * 2 + row_h, gap * 2 + col_w),
    ]
    for (gy, gx), panel in zip(positions, panels):
        grid[gy : gy + row_h, gx : gx + col_w, :] = panel
    return grid


def run_real_image_tests() -> list[TimingRecord]:
    """使用 DIV2K Validation 图像执行 2x/4x 实际插值测试并生成对比图。"""
    timings: list[TimingRecord] = []
    inputs_dir = RESULTS_ROOT / "inputs"
    comparisons_dir = RESULTS_ROOT / "comparisons"

    for filename in DEMO_IMAGES:
        src_path = DIV2K_VALID_DIR / filename
        if not src_path.is_file():
            raise FileNotFoundError(f"未找到 DIV2K Validation 测试图像: {src_path}")

        hr_rgb = _load_rgb_image(src_path)
        stem = src_path.stem

        for scale in SUPPORTED_SCALES:
            lr_rgb, hr_ref = _make_demo_lr(hr_rgb, scale)
            lr_h, lr_w = lr_rgb.shape[:2]
            _save_rgb_image(lr_rgb, inputs_dir / f"{stem}_lr_x{scale}.png")

            outputs: dict[str, np.ndarray] = {}
            for method in INTERPOLATION_METHODS:
                for _ in range(TIMING_WARMUP):
                    _ = upscale_image(lr_rgb, scale, method)

                samples_ms: list[float] = []
                out_rgb: np.ndarray | None = None
                for _ in range(TIMING_REPEATS):
                    t0 = time.perf_counter()
                    out_rgb = upscale_image(lr_rgb, scale, method)
                    t1 = time.perf_counter()
                    samples_ms.append((t1 - t0) * 1000.0)

                assert out_rgb is not None
                out_h, out_w = out_rgb.shape[:2]
                assert out_w == lr_w * scale and out_h == lr_h * scale, (
                    f"{filename} {method} x{scale} 尺寸不匹配: "
                    f"({out_w}, {out_h}) != ({lr_w * scale}, {lr_h * scale})"
                )
                assert out_rgb.shape[2] == 3

                timings.append(
                    TimingRecord(
                        image_name=filename,
                        method=method,
                        scale=scale,
                        input_size=(lr_w, lr_h),
                        output_size=(out_w, out_h),
                        mean_ms=sum(samples_ms) / len(samples_ms),
                        min_ms=min(samples_ms),
                        max_ms=max(samples_ms),
                    )
                )
                outputs[method] = out_rgb
                _save_rgb_image(
                    out_rgb,
                    RESULTS_ROOT / method / f"{stem}_x{scale}_{method}.png",
                )

            comp_grid = _build_comparison_grid(
                stem, scale, lr_rgb, outputs, hr_ref
            )
            _save_rgb_image(
                comp_grid, comparisons_dir / f"{stem}_x{scale}_comparison.png"
            )

    return timings


def main() -> int:
    print("=" * 72)
    print("阶段 3：传统图像插值算法测试与演示")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]} | OpenCV: {cv2.__version__} | NumPy: {np.__version__}")
    print("=" * 72)

    unit_records = run_unit_tests()
    passed_count = sum(1 for r in unit_records if r.passed)
    failed_count = len(unit_records) - passed_count

    print("\n[1/2] 功能与边界测试结果：")
    for rec in unit_records:
        status = "PASS" if rec.passed else "FAIL"
        print(f"  [{status}] {rec.name}: {rec.detail}")
    print(f"  汇总: 通过 {passed_count} / {len(unit_records)}，失败 {failed_count}")

    if failed_count > 0:
        return 1

    print(
        f"\n[2/2] DIV2K Validation 实际图像测试"
        f"（只读读取原图，预热 {TIMING_WARMUP} 次 + 重复 {TIMING_REPEATS} 次纯插值耗时）："
    )
    timings = run_real_image_tests()
    print(
        f"  {'图像':<10} {'算法':<10} {'倍率':<5} {'输入(WxH)':<12} "
        f"{'输出(WxH)':<12} {'均值(ms)':>10} {'最小(ms)':>10} {'最大(ms)':>10}"
    )
    print("  " + "-" * 86)
    for t in timings:
        in_str = f"{t.input_size[0]}x{t.input_size[1]}"
        out_str = f"{t.output_size[0]}x{t.output_size[1]}"
        print(
            f"  {t.image_name:<10} {t.method:<10} {t.scale}x   {in_str:<12} "
            f"{out_str:<12} {t.mean_ms:>10.3f} {t.min_ms:>10.3f} {t.max_ms:>10.3f}"
        )

    print(f"\n结果输出目录: {RESULTS_ROOT}")
    print("RESULT = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
