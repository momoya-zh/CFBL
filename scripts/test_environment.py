# -*- coding: utf-8 -*-
"""阶段 1 环境自检脚本（只读）。

运行方式（必须使用项目指定解释器的绝对路径，不要用系统默认 python 或 py launcher）::

    "D:\\Environment\\Python\\Python_AI\\python.exe" scripts\\test_environment.py

约束：
- 只做检查：不安装依赖、不修改任何系统/环境配置、不写文件。
- 仅执行一次小型 CUDA Tensor 矩阵运算，不做大型 GPU 测试。
- 输出中的耗时与显存数字只是本次执行的观测值，**不是性能基准**；
  训练速度与显存峰值的正式测量留待阶段 5。
- 检查失败时逐项列出失败项目并退出码置 1。
"""
from __future__ import annotations

import importlib
import platform
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

EXPECTED_INTERPRETER = r"D:\Environment\Python\Python_AI\python.exe"

# (import 名, 展示名)
REQUIRED_IMPORTS = [
    ("numpy", "NumPy"),
    ("cv2", "OpenCV / opencv-python"),
    ("PIL", "Pillow"),
    ("scipy", "SciPy"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
    ("yaml", "PyYAML"),
    ("skimage", "scikit-image"),
    ("gradio", "Gradio"),
]

_DIST_NAME = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "skimage": "scikit-image",
}

_failures: list[str] = []
_warnings: list[str] = []


def ok(msg: str) -> None:
    print("[OK]   " + msg)


def fail(msg: str) -> None:
    _failures.append(msg)
    print("[FAIL] " + msg)


def warn(msg: str) -> None:
    _warnings.append(msg)
    print("[WARN] " + msg)


def info(msg: str) -> None:
    print("[INFO] " + msg)


def section(title: str) -> None:
    print("")
    print("=" * 76)
    print(title)
    print("=" * 76)


def check_import(import_name: str, display_name: str) -> None:
    try:
        mod = importlib.import_module(import_name)
    except Exception as exc:
        fail("{} (import '{}') 导入失败：{}: {}".format(
            display_name, import_name, type(exc).__name__, exc))
        return
    version = getattr(mod, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as dist_version
            version = dist_version(_DIST_NAME.get(import_name, import_name))
        except Exception:
            version = "(包内无 __version__ 字段)"
    ok("{} = {} (import '{}')".format(display_name, version, import_name))


def finish() -> int:
    section("汇总 / SUMMARY")
    if _warnings:
        for w in _warnings:
            print("  WARN: " + w)
    else:
        info("无警告")
    if _failures:
        for f in _failures:
            print("  FAIL: " + f)
        print("RESULT = FAIL（{} 项失败，{} 项警告）".format(len(_failures), len(_warnings)))
        return 1
    print("RESULT = PASS（0 项失败，{} 项警告）".format(len(_warnings)))
    return 0


def main() -> int:
    section("阶段 1 环境自检 / test_environment.py")
    info("只做只读检查；不安装依赖、不修改配置。")
    info("耗时 / 显存仅为单次观测，不是性能基准；训练性能测量留待阶段 5。")

    section("1. Python 解释器")
    info("sys.executable   = {}".format(sys.executable))
    info("Python 版本      = {} / {}".format(sys.version.split()[0], platform.python_version()))
    info("实现 / 编译器    = {} / {}".format(platform.python_implementation(), platform.python_compiler()))
    info("平台             = {}".format(platform.platform()))
    info("sys.prefix       = {}".format(sys.prefix))
    if sys.executable.lower() == EXPECTED_INTERPRETER.lower():
        ok("解释器路径 = 项目指定的 Python_AI 绝对路径")
    else:
        fail("解释器路径不符：期望 {}，实际 {}".format(EXPECTED_INTERPRETER, sys.executable))

    section("2. PyTorch / torchvision / CUDA Runtime / cuDNN")
    try:
        import torch
    except Exception as exc:
        fail("torch 导入失败：{}: {}".format(type(exc).__name__, exc))
        return finish()
    info("torch.__version__    = {}".format(torch.__version__))
    info("torch.__file__       = {}".format(torch.__file__))
    try:
        import torchvision
        info("torchvision version = {}".format(torchvision.__version__))
    except Exception as exc:
        fail("torchvision 导入失败：{}: {}".format(type(exc).__name__, exc))
    info("torch.version.cuda   = {}".format(torch.version.cuda))
    cudnn_ver = torch.backends.cudnn.version()
    info("cuDNN version()      = {}{}".format(
        cudnn_ver, "  (None = 该构建无 cuDNN)" if cudnn_ver is None else ""))
    info("cudnn.enabled        = {}".format(torch.backends.cudnn.enabled))
    if torch.version.cuda is None:
        warn("torch 疑似 CPU-only 构建（torch.version.cuda 为 None），本项目需要 CUDA 版")

    section("3. 依赖导入检查（逐项输出版本）")
    for import_name, display_name in REQUIRED_IMPORTS:
        check_import(import_name, display_name)

    section("4. GPU / CUDA 状态")
    cuda_ok = torch.cuda.is_available()
    info("torch.cuda.is_available() = {}".format(cuda_ok))
    info("torch.cuda.device_count() = {}".format(torch.cuda.device_count()))
    if not cuda_ok:
        fail("CUDA 不可用：torch.cuda.is_available() == False；跳过第 5 节 GPU 运算")
        return finish()
    ok("CUDA 可用")
    props = torch.cuda.get_device_properties(0)
    info("GPU 0 名称           = {}".format(props.name))
    info("计算能力             = {}.{}".format(props.major, props.minor))
    info("显存总量             = {:.1f} MiB ({:.2f} GiB)".format(
        props.total_memory / 1048576, props.total_memory / 1073741824))
    info("SM 数 / L2 cache     = {} / {} bytes".format(props.multi_processor_count, props.L2_cache_size))
    free_b, total_b = torch.cuda.mem_get_info()
    info("当前空闲 / 总显存    = {:.1f} / {:.1f} MiB".format(
        free_b / 1048576, total_b / 1048576))

    section("5. 小型 CUDA Tensor 与矩阵运算（非性能基准）")
    a = b = c = None
    try:
        torch.cuda.reset_peak_memory_stats(0)
        t0 = time.perf_counter()
        a = torch.randn(512, 512, dtype=torch.float32, device="cuda")
        b = torch.randn(512, 512, dtype=torch.float32, device="cuda")
        t1 = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        ok("CUDA Tensor 创建成功：shape={}, dtype={}, device={}".format(tuple(a.shape), a.dtype, a.device))
        ok("矩阵乘法成功：c = a @ b, shape={}".format(tuple(c.shape)))
        ok("torch.cuda.synchronize() 已完成")
        all_finite = bool(torch.isfinite(a).all().item()
                          and torch.isfinite(b).all().item()
                          and torch.isfinite(c).all().item())
        info("结果均值 / 标准差   = {:.4f} / {:.4f}".format(c.mean().item(), c.std().item()))
        info("全部为有限值        = {}".format(all_finite))
        if all_finite:
            ok("运算结果全部为有限值")
        else:
            fail("运算结果包含 NaN / Inf")
        info("耗时（单次观测）     = 创建 {:.4f} s，乘加+同步 {:.4f} s，合计 {:.4f} s".format(
            t1 - t0, t2 - t1, t2 - t0))
        info("注意：以上耗时含首次 CUDA 上下文 / cuBLAS 初始化开销，不是性能基准；"
             "训练速度与显存峰值在阶段 5 测量。")
        info("memory_allocated    = {:.1f} MiB".format(torch.cuda.memory_allocated(0) / 1048576))
        info("memory_reserved     = {:.1f} MiB".format(torch.cuda.memory_reserved(0) / 1048576))
        info("max_memory_allocated= {:.1f} MiB".format(torch.cuda.max_memory_allocated(0) / 1048576))
        free_a, total_a = torch.cuda.mem_get_info()
        info("运算后空闲 / 总显存  = {:.1f} / {:.1f} MiB".format(
            free_a / 1048576, total_a / 1048576))
    except Exception as exc:
        fail("GPU 运算失败：{}: {}".format(type(exc).__name__, exc))
    finally:
        del a, b, c
        torch.cuda.empty_cache()
        info("已释放临时 GPU 张量并调用 torch.cuda.empty_cache()")

    return finish()


if __name__ == "__main__":
    sys.exit(main())
