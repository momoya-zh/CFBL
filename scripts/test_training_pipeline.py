# -*- coding: utf-8 -*-
"""阶段 1 GPU 训练链路冒烟测试（不是正式训练，也不是性能测试）。

运行方式::

    "D:\\Environment\\Python\\Python_AI\\python.exe" scripts\\test_training_pipeline.py

目的：只验证「前向传播 → L1 Loss → 反向传播 → Adam 单次参数更新」这条
计算链路可以跑通。**不代表 SRCNN / ESPCN 已实现，更不代表模型已训练完成。**

严格限制：
- 不读取真实数据集；不保存任何模型权重；不执行多轮训练；
- 使用很小的输入（N=2, 3×32×32），不做性能基准测试；
- 输出的耗时与显存只是本次执行的观测值，**不是训练速度或显存峰值的结论**；
  真正的训练性能与显存峰值测量留待阶段 5；
- 结束后释放所有 GPU 张量并调用 torch.cuda.empty_cache()；
- 不安装依赖、不修改任何系统/环境配置、不写文件。
"""
from __future__ import annotations

import platform
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

EXPECTED_INTERPRETER = r"D:\Environment\Python\Python_AI\python.exe"

BATCH, CHANNELS, SIZE = 2, 3, 32
LEARNING_RATE = 1e-3

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
    print("说明：PASS 只代表单次训练计算链路可执行，不代表模型已实现或训练完成，"
          "也不代表任何性能结论。")
    return 0


def main() -> int:
    section("阶段 1 GPU 训练链路冒烟测试 / test_training_pipeline.py")
    info("临时小型卷积网络，单次前向 / L1 Loss / 反向 / Adam step。")
    info("不读数据、不存权重、不多轮训练；耗时与显存为单次观测，不是性能基准。")
    info("正式训练性能与显存峰值测量留待阶段 5。")

    section("1. 运行环境")
    info("sys.executable = {}".format(sys.executable))
    info("Python         = {}".format(platform.python_version()))
    if sys.executable.lower() == EXPECTED_INTERPRETER.lower():
        ok("解释器路径 = 项目指定的 Python_AI 绝对路径")
    else:
        fail("解释器路径不符：期望 {}，实际 {}".format(EXPECTED_INTERPRETER, sys.executable))

    try:
        import torch
    except Exception as exc:
        fail("torch 导入失败：{}: {}".format(type(exc).__name__, exc))
        return finish()
    info("torch.__version__ = {}".format(torch.__version__))
    if not torch.cuda.is_available():
        fail("CUDA 不可用，无法执行 GPU 训练链路测试")
        return finish()
    device = torch.device("cuda:0")
    ok("CUDA 可用，device = cuda:0 ({})".format(torch.cuda.get_device_name(0)))

    # 一次性临时网络：仅用于验证计算链路，不是 SRCNN / ESPCN 的实现。
    class TinyConvNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = torch.nn.Sequential(
                torch.nn.Conv2d(CHANNELS, 8, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, CHANNELS, kernel_size=3, padding=1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.body(x)

    model = criterion = optimizer = None
    x = target = output = loss = None
    before = None
    try:
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats(0)
        free_b, total_b = torch.cuda.mem_get_info()

        model = TinyConvNet().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        ok("临时卷积网络已创建并移动到 CUDA；可训练参数量 = {}".format(n_params))
        info("网络结构 = Conv2d(3,8,3) -> ReLU -> Conv2d(8,3,3)（非 SRCNN / ESPCN）")

        x = torch.randn(BATCH, CHANNELS, SIZE, SIZE, dtype=torch.float32, device=device)
        target = torch.randn_like(x)
        ok("随机输入 / 目标张量创建成功：{}，device={}".format(tuple(x.shape), x.device))

        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}

        section("2. 单次训练链路")
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(x)                       # 前向
        loss = criterion(output, target)        # L1 Loss
        loss.backward()                         # 反向
        grads_ok = True
        grad_report = []
        for name, p in model.named_parameters():
            g = p.grad
            finite = (g is not None) and bool(torch.isfinite(g).all().item())
            grads_ok = grads_ok and finite
            grad_report.append("{}: grad={}".format(name, "finite" if finite else "MISSING/非有限"))
        optimizer.step()                        # Adam 更新一次
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        ok("前向传播完成：output shape = {}".format(tuple(output.shape)))
        ok("L1 Loss 计算完成")
        ok("反向传播完成")
        ok("Adam 优化器 step() 完成（仅 1 次）")
        ok("torch.cuda.synchronize() 已完成")

        section("3. 结果检查")
        out_finite = bool(torch.isfinite(output).all().item())
        loss_finite = bool(torch.isfinite(loss).item())
        info("L1 Loss 值            = {:.6f}".format(loss.item()))
        info("输出全部有限值        = {}".format(out_finite))
        info("损失为有限值          = {}".format(loss_finite))
        if out_finite and loss_finite:
            ok("损失与输出均为有限值")
        else:
            fail("损失或输出包含 NaN / Inf")

        for line in grad_report:
            info("  " + line)
        if grads_ok:
            ok("所有参数梯度存在且为有限值")
        else:
            fail("存在缺失或非有限的参数梯度")

        changed = 0
        max_abs_delta = 0.0
        for name, p in model.named_parameters():
            delta = (p.detach() - before[name]).abs().max().item()
            if not torch.equal(p.detach(), before[name]):
                changed += 1
            max_abs_delta = max(max_abs_delta, delta)
            info("  {} : 最大绝对变化 = {:.3e}".format(name, delta))
        total_params = len(before)
        info("发生变化的参数张量    = {} / {}".format(changed, total_params))
        if changed == total_params and max_abs_delta > 0.0:
            ok("参数已正常更新（全部 {} 个参数张量发生变化）".format(total_params))
        else:
            fail("参数未按预期更新（changed={}/{}, max_delta={}）".format(
                changed, total_params, max_abs_delta))

        section("4. 耗时与显存（单次观测，非性能结论）")
        info("本次链路耗时          = {:.4f} s（含首次 CUDA 上下文 / cuDNN / cuBLAS 初始化）".format(t1 - t0))
        info("显存空闲 / 总量（前） = {:.1f} / {:.1f} MiB".format(free_b / 1048576, total_b / 1048576))
        info("memory_allocated      = {:.1f} MiB".format(torch.cuda.memory_allocated(0) / 1048576))
        info("memory_reserved       = {:.1f} MiB".format(torch.cuda.memory_reserved(0) / 1048576))
        info("max_memory_allocated  = {:.1f} MiB".format(torch.cuda.max_memory_allocated(0) / 1048576))
        free_a, total_a = torch.cuda.mem_get_info()
        info("显存空闲 / 总量（后） = {:.1f} / {:.1f} MiB".format(free_a / 1048576, total_a / 1048576))
        info("注意：以上数字只用于证明本次计算确实执行完成并非空跑，"
             "不构成训练速度或显存峰值结论；性能测量在阶段 5 进行。")
    except Exception as exc:
        fail("训练链路执行失败：{}: {}".format(type(exc).__name__, exc))
    finally:
        del x, target, output, loss, before, model, optimizer, criterion
        torch.cuda.empty_cache()
        info("已释放临时 GPU 张量并调用 torch.cuda.empty_cache()；未保存任何权重")

    return finish()


if __name__ == "__main__":
    sys.exit(main())
