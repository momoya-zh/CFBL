# -*- coding: utf-8 -*-
"""阶段 4 深度学习超分辨率模型（SRCNN / ESPCN）测试脚本。

运行方式（CMD，使用项目指定 Python 解释器）::

    "D:\\Environment\\Python\\Python_AI\\python.exe" scripts\\test_models.py

测试内容：
1. 基础导入、模型工厂、四套 YAML 配置元数据一致性与参数量核验。
2. 前向传播尺寸（SRCNN 64x96/32x48 恒等；ESPCN 32x48->64x96、16x24->64x96）与张量合法性。
3. 统一推理接口（predict_tensor / predict_rgb_image）、倍率一致性、分支混用检测与 clamp。
4. 反向传播（L1Loss + backward）梯度完整性与有限性检查。
5. 模型实例独立性（参数存储不共享、单步更新互不影响）。
6. CUDA 前向/反向与单模型峰值显存统计（每套测试前重置显存统计）。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import predict_rgb_image, predict_tensor  # noqa: E402
from src.models import ESPCN, SRCNN, create_model  # noqa: E402
from src.processing import bicubic_upsample_pillow  # noqa: E402

MODEL_CONFIG_DIR = PROJECT_ROOT / "configs" / "models"

MODEL_SPECS = [
    ("srcnn", 2, (2, 3, 64, 96), (2, 3, 64, 96), 69251, "srcnn_x2.yaml"),
    ("srcnn", 4, (2, 3, 32, 48), (2, 3, 32, 48), 69251, "srcnn_x4.yaml"),
    ("espcn", 2, (2, 3, 32, 48), (2, 3, 64, 96), 26796, "espcn_x2.yaml"),
    ("espcn", 4, (2, 3, 16, 24), (2, 3, 64, 96), 37200, "espcn_x4.yaml"),
]


@dataclass
class TestRecord:
    """单条测试结果。"""

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


def _count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_cpu_tests() -> tuple[list[TestRecord], list[dict[str, object]]]:
    """执行全部 CPU 模型测试并统计参数量。"""
    records: list[TestRecord] = []
    param_stats: list[dict[str, object]] = []

    def check(name: str, fn) -> None:
        try:
            detail = fn() or "OK"
            records.append(TestRecord(name=name, passed=True, detail=detail))
        except Exception as exc:
            records.append(TestRecord(name=name, passed=False, detail=str(exc)))

    def test_factory_and_yaml_consistency() -> str:
        for model_name, scale, _, _, expected_params, yaml_name in MODEL_SPECS:
            model = create_model(model_name, scale)
            assert model.scale == scale
            if model_name == "srcnn":
                assert isinstance(model, SRCNN)
                assert model.upsample_inside is False
            else:
                assert isinstance(model, ESPCN)
                assert model.upsample_inside is True
                assert model.conv3.out_channels == 3 * (scale ** 2)

            total, trainable = _count_params(model)
            assert total == expected_params and trainable == expected_params, (
                f"{model_name} x{scale} 参数量不符: 实测={total}, 预期={expected_params}"
            )

            yaml_path = MODEL_CONFIG_DIR / yaml_name
            assert yaml_path.is_file(), f"缺少配置文件: {yaml_path}"
            with yaml_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            assert cfg["model_name"] == model_name
            assert cfg["scale"] == scale
            assert cfg["in_channels"] == 3 and cfg["out_channels"] == 3
            assert cfg["color_space"] == "RGB"
            assert cfg["downsample_kernel"] == "pillow_bicubic"
            assert cfg["expected_params"] == expected_params

            param_stats.append(
                {
                    "config": f"{model_name.upper()} x{scale}",
                    "total_params": total,
                    "trainable_params": trainable,
                }
            )

        # 非法参数校验
        _assert_raises(ValueError, create_model, "edsr", 2)
        _assert_raises(TypeError, create_model, 123, 2)
        _assert_raises(ValueError, create_model, "srcnn", 3)
        _assert_raises(TypeError, create_model, "espcn", True)
        _assert_raises(TypeError, SRCNN, 2.0)
        _assert_raises(ValueError, ESPCN, 8)
        return "4套模型工厂创建、scale/通道校验与 YAML 元数据完全一致"

    def test_forward_and_tensor_validity() -> str:
        torch.manual_seed(42)
        for model_name, scale, in_shape, out_shape, _, _ in MODEL_SPECS:
            model = create_model(model_name, scale)
            for p in model.parameters():
                assert torch.isfinite(p).all(), f"{model_name} x{scale} 初始参数含非有限值"

            x = torch.rand(in_shape, dtype=torch.float32)
            x_snapshot = x.clone()
            y = model(x)

            assert y.shape == out_shape, f"{model_name} x{scale} 输出形状={tuple(y.shape)} != {out_shape}"
            assert y.dtype == torch.float32
            assert y.shape[1] == 3
            assert torch.isfinite(y).all(), f"{model_name} x{scale} 输出含 NaN/Inf"
            assert torch.equal(x, x_snapshot), f"{model_name} x{scale} 修改了输入张量"

        # 异常输入拦截
        m = create_model("espcn", 2)
        _assert_raises(TypeError, m, np.zeros((1, 3, 8, 8), dtype=np.float32))
        _assert_raises(TypeError, m, torch.zeros((1, 3, 8, 8), dtype=torch.float16))
        _assert_raises(ValueError, m, torch.zeros((3, 8, 8), dtype=torch.float32))
        _assert_raises(ValueError, m, torch.zeros((1, 1, 8, 8), dtype=torch.float32))
        _assert_raises(ValueError, m, torch.zeros((1, 4, 8, 8), dtype=torch.float32))
        _assert_raises(ValueError, m, torch.zeros((1, 3, 0, 8), dtype=torch.float32))
        return "4套前向尺寸、float32/RGB/有限值、输入不可变及异常拦截全部通过"

    def test_unified_predictor_and_branch_isolation() -> str:
        torch.manual_seed(7)
        lr_x2 = torch.rand((2, 3, 32, 48), dtype=torch.float32)
        lr_x4 = torch.rand((2, 3, 16, 24), dtype=torch.float32)

        for model_name in ("srcnn", "espcn"):
            m2 = create_model(model_name, 2)
            out2 = predict_tensor(m2, lr_x2, scale=2, device="cpu", clamp=True)
            assert out2.shape == (2, 3, 64, 96)
            assert float(out2.min()) >= 0.0 and float(out2.max()) <= 1.0

            m4 = create_model(model_name, 4)
            out4 = predict_tensor(m4, lr_x4, scale=4, device="cpu", clamp=True)
            assert out4.shape == (2, 3, 64, 96)
            assert float(out4.min()) >= 0.0 and float(out4.max()) <= 1.0

        # 倍率不匹配必须报错
        m_srcnn2 = create_model("srcnn", 2)
        _assert_raises(ValueError, predict_tensor, m_srcnn2, lr_x2, 4)

        # 分支混用检测：把已上采样的 64x96 错喂给 ESPCN(2)，尺寸必不等于目标 64x96
        upsampled_for_srcnn = torch.from_numpy(
            bicubic_upsample_pillow(
                np.zeros((32, 48, 3), dtype=np.float32), 2
            ).transpose(2, 0, 1)
        ).unsqueeze(0)
        m_espcn2 = create_model("espcn", 2)
        wrong_out = m_espcn2(upsampled_for_srcnn)
        assert wrong_out.shape != (1, 3, 64, 96), "ESPCN 不应接收已上采样输入"

        # 图像入口验证
        rgb_lr = np.full((16, 24, 3), 128, dtype=np.uint8)
        rgb_sr = predict_rgb_image(m_srcnn2, rgb_lr, scale=2, device="cpu")
        assert rgb_sr.shape == (32, 48, 3) and rgb_sr.dtype == np.uint8
        return "predict_tensor/predict_rgb_image 尺寸正确、倍率校验与混用检测通过"

    def test_backward_and_gradients() -> str:
        torch.manual_seed(99)
        criterion = nn.L1Loss()
        for model_name, scale, in_shape, out_shape, _, _ in MODEL_SPECS:
            model = create_model(model_name, scale)
            model.train()
            x = torch.rand(in_shape, dtype=torch.float32)
            target = torch.rand(out_shape, dtype=torch.float32)

            out = model(x)
            loss = criterion(out, target)
            assert torch.isfinite(loss), f"{model_name} x{scale} loss 非有限"
            loss.backward()

            for pname, p in model.named_parameters():
                assert p.requires_grad
                assert p.grad is not None, f"{model_name} x{scale} {pname} 梯度为 None"
                assert torch.isfinite(p.grad).all(), (
                    f"{model_name} x{scale} {pname} 梯度含 NaN/Inf"
                )
        return "4套模型 L1Loss 反向传播全部可训练参数梯度非空且有限"

    def test_instance_independence() -> str:
        for model_name, scale, in_shape, out_shape, _, _ in MODEL_SPECS:
            m_a = create_model(model_name, scale)
            m_b = create_model(model_name, scale)
            assert m_a is not m_b

            for (_, pa), (_, pb) in zip(m_a.named_parameters(), m_b.named_parameters()):
                assert pa is not pb
                assert pa.data_ptr() != pb.data_ptr()

            b_before = [pb.detach().clone() for pb in m_b.parameters()]
            opt = torch.optim.SGD(m_a.parameters(), lr=0.1)
            loss = nn.L1Loss()(m_a(torch.rand(in_shape)), torch.rand(out_shape))
            opt.zero_grad()
            loss.backward()
            opt.step()

            for pb_old, pb_now in zip(b_before, m_b.parameters()):
                assert torch.equal(pb_old, pb_now), (
                    f"{model_name} x{scale} 更新实例 A 后实例 B 参数发生变化"
                )
        return "4套配置多实例参数存储完全独立，单步更新互不干扰"

    check("1-工厂创建/scale绑定/YAML元数据一致性", test_factory_and_yaml_consistency)
    check("2-前向传播尺寸/张量合法性/异常校验", test_forward_and_tensor_validity)
    check("3-统一推理接口/倍率校验/混用检测", test_unified_predictor_and_branch_isolation)
    check("4-反向传播梯度链路与有限性", test_backward_and_gradients)
    check("5-模型实例与参数独立性", test_instance_independence)
    return records, param_stats


def run_cuda_tests() -> tuple[bool, list[dict[str, object]], str]:
    """在 CUDA 可用时执行四套模型前向/反向测试与单模型峰值显存统计。"""
    if not torch.cuda.is_available():
        return False, [], "CUDA 不可用，跳过 GPU 测试"

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    criterion = nn.L1Loss()
    cuda_records: list[dict[str, object]] = []

    for model_name, scale, in_shape, out_shape, _, _ in MODEL_SPECS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        model = create_model(model_name, scale).to(device)
        model.train()
        x = torch.rand(in_shape, dtype=torch.float32, device=device)
        target = torch.rand(out_shape, dtype=torch.float32, device=device)

        out = model(x)
        assert out.shape == out_shape and out.device.type == "cuda"
        assert torch.isfinite(out).all()

        loss = criterion(out, target)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()

        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        cuda_records.append(
            {
                "config": f"{model_name.upper()} x{scale}",
                "in_shape": tuple(in_shape),
                "out_shape": tuple(out_shape),
                "peak_mem_mb": peak_mb,
            }
        )
        del model, x, target, out, loss
        torch.cuda.empty_cache()

    return True, cuda_records, device_name


def main() -> int:
    print("=" * 74)
    print("阶段 4：深度学习超分辨率模型测试（SRCNN / ESPCN）")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__}")
    print("=" * 74)

    cpu_records, param_stats = run_cpu_tests()
    passed = sum(1 for r in cpu_records if r.passed)
    failed = len(cpu_records) - passed

    print("\n[1/3] CPU 功能、推理、梯度与独立性测试：")
    for rec in cpu_records:
        tag = "PASS" if rec.passed else "FAIL"
        print(f"  [{tag}] {rec.name}: {rec.detail}")
    print(f"  汇总: 通过 {passed} / {len(cpu_records)}，失败 {failed}")
    if failed > 0:
        return 1

    print("\n[2/3] 四套模型参数量实测统计：")
    print(f"  {'模型配置':<14} {'总参数量':>12} {'可训练参数量':>14}")
    print("  " + "-" * 44)
    for row in param_stats:
        print(
            f"  {row['config']:<14} {row['total_params']:>12,d} "
            f"{row['trainable_params']:>14,d}"
        )

    print("\n[3/3] CUDA 前向/反向与单模型峰值显存测试：")
    cuda_ok, cuda_rows, cuda_info = run_cuda_tests()
    if cuda_ok:
        print(f"  设备: {cuda_info}")
        print(
            f"  {'模型配置':<14} {'输入形状':<18} {'输出形状':<18} {'峰值显存(MiB)':>14}"
        )
        print("  " + "-" * 68)
        for row in cuda_rows:
            print(
                f"  {row['config']:<14} {str(row['in_shape']):<18} "
                f"{str(row['out_shape']):<18} {row['peak_mem_mb']:>14.3f}"
            )
    else:
        print(f"  [SKIP] {cuda_info}")

    print("\nRESULT = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
