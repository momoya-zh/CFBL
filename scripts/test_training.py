# -*- coding: utf-8 -*-
"""阶段 5 训练链路测试脚本（T-1 ~ T-7 + T-9 变异探针）。

运行方式（CMD）::

    "D:\\Environment\\Python\\Python_AI\\python.exe" scripts\\test_training.py
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.training.losses as losses_mod  # noqa: E402
from src.models import create_model  # noqa: E402
from src.training import (  # noqa: E402
    build_loss,
    evaluate_pixel_weighted,
    load_training_checkpoint,
    save_training_checkpoint,
    set_deterministic_seed,
)


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


def run_training_tests() -> list[TestRecord]:
    records: list[TestRecord] = []

    def check(code: str, name: str, fn) -> None:
        try:
            detail = fn() or "OK"
            records.append(TestRecord(code=code, name=name, passed=True, detail=detail))
        except Exception as exc:
            records.append(TestRecord(code=code, name=name, passed=False, detail=str(exc)))

    def test_t1() -> str:
        set_deterministic_seed(42)
        model = create_model("espcn", 2)
        criterion = build_loss("l1")
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        lr_in = torch.rand((4, 3, 16, 16), dtype=torch.float32)
        hr_in = torch.rand((4, 3, 32, 32), dtype=torch.float32)
        opt.zero_grad()
        pred = model(lr_in)
        loss = criterion(pred, hr_in)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        return f"单步 L1 loss={float(loss.item()):.6f} 有限且参数已更新"

    def test_t2() -> str:
        def _run_20_steps() -> list[float]:
            set_deterministic_seed(42)
            model = create_model("espcn", 2)
            criterion = build_loss("l1")
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)
            losses: list[float] = []
            for _ in range(20):
                x = torch.rand((2, 3, 16, 16), dtype=torch.float32)
                y = torch.rand((2, 3, 32, 32), dtype=torch.float32)
                opt.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            return losses

        l_a = _run_20_steps()
        l_b = _run_20_steps()
        assert l_a == l_b, "同种子两次 20 step loss 未逐位相等"
        return "同种子两次训练前 20 step loss 逐位相等"

    def test_t3_and_t4() -> str:
        # 连续训练 5 epoch vs 训练 3 epoch -> 存 -> 重载 -> 续训 2 epoch
        def _fixed_batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
            g = torch.Generator().manual_seed(100)
            return [
                (
                    torch.rand((2, 3, 16, 16), generator=g, dtype=torch.float32),
                    torch.rand((2, 3, 32, 32), generator=g, dtype=torch.float32),
                )
                for _ in range(5)
            ]

        batches = _fixed_batches()
        criterion = build_loss("l1")

        # 基线：连续跑 5 个 epoch
        set_deterministic_seed(42)
        m_full = create_model("espcn", 2)
        opt_full = torch.optim.Adam(m_full.parameters(), lr=1e-4)
        full_losses: list[float] = []
        for ep in range(5):
            x, y = batches[ep]
            opt_full.zero_grad()
            loss = criterion(m_full(x), y)
            loss.backward()
            opt_full.step()
            full_losses.append(float(loss.item()))

        # 断点续训：跑 3 epoch -> 存 -> 重载 -> 跑后 2 epoch
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "latest.pth"
            set_deterministic_seed(42)
            m_part = create_model("espcn", 2)
            opt_part = torch.optim.Adam(m_part.parameters(), lr=1e-4)
            part_losses: list[float] = []
            for ep in range(3):
                x, y = batches[ep]
                opt_part.zero_grad()
                loss = criterion(m_part(x), y)
                loss.backward()
                opt_part.step()
                part_losses.append(float(loss.item()))

            save_training_checkpoint(
                ckpt_path,
                m_part,
                opt_part,
                epoch=3,
                global_step=3,
                best_val=part_losses[-1],
                seed=42,
                config={"batch_size": 2, "lr": 1e-4},
                history=[{"epoch": 1, "train_loss": part_losses[0]}],
            )

            m_resume = create_model("espcn", 2)
            opt_resume = torch.optim.Adam(m_resume.parameters(), lr=1e-4)
            ckpt = load_training_checkpoint(ckpt_path, m_resume, optimizer=opt_resume)
            required_keys = {
                "model_name",
                "scale",
                "state_dict",
                "optimizer",
                "epoch",
                "global_step",
                "best_val",
                "seed",
                "config",
                "env",
                "history",
            }
            missing = required_keys - set(ckpt.keys())
            assert not missing, f"checkpoint 缺少字段: {missing}"

            for ep in range(3, 5):
                x, y = batches[ep]
                opt_resume.zero_grad()
                loss = criterion(m_resume(x), y)
                loss.backward()
                opt_resume.step()
                part_losses.append(float(loss.item()))

        max_diff = max(abs(a - b) for a, b in zip(full_losses, part_losses))
        assert max_diff < 1e-6, f"续训损失与连续训练偏差过大: {max_diff}"
        return f"续训曲线连续（max_diff={max_diff:.2e} < 1e-6）且 checkpoint 字段齐全"

    def test_t5() -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "srcnn_x2.pth"
            m2 = create_model("srcnn", 2)
            opt = torch.optim.Adam(m2.parameters(), lr=1e-4)
            save_training_checkpoint(
                ckpt_path, m2, opt, epoch=1, global_step=1, best_val=0.1,
                seed=42, config={}, history=[]
            )
            m4 = create_model("srcnn", 4)
            _assert_raises(ValueError, load_training_checkpoint, ckpt_path, m4)
            e2 = create_model("espcn", 2)
            _assert_raises(ValueError, load_training_checkpoint, ckpt_path, e2)
        return "跨倍率与跨模型载入训练 checkpoint 均抛出 ValueError"

    def test_t6() -> str:
        model = create_model("srcnn", 2)
        model.train(True)
        before = [p.detach().clone() for p in model.parameters()]
        val_ds = TensorDataset(
            torch.rand((2, 3, 16, 16), dtype=torch.float32),
            torch.rand((2, 3, 8, 8), dtype=torch.float32),
            torch.rand((2, 3, 16, 16), dtype=torch.float32),
        )
        val_loader = DataLoader(val_ds, batch_size=1)
        _ = evaluate_pixel_weighted(
            model, val_loader, build_loss("l1"), torch.device("cpu"), use_preupsample=True
        )
        assert model.training is True, "验证不应破坏调用方 training 状态"
        for b, a in zip(before, model.parameters()):
            assert torch.equal(b, a), "验证路径修改了模型参数"
        return "验证路径不改变模型参数与 training 状态"

    def test_t7() -> str:
        # 构造恒等映射模型（权全 0，偏置全 0）以便手算 L1 期望值：
        # 图1：1x3x2x2（12 像素）全 0.2 -> L1=0.2；图2：1x3x4x4（48 像素）全 0.8 -> L1=0.8
        # 按像素加权期望 = (12*0.2 + 48*0.8) / (12 + 48) = 40.8 / 60 = 0.68
        class _ZeroModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros_like(x)

        hr1 = torch.full((1, 3, 2, 2), 0.2, dtype=torch.float32)
        hr2 = torch.full((1, 3, 4, 4), 0.8, dtype=torch.float32)
        val_loader = [(hr1, hr1), (hr2, hr2)]
        criterion = losses_mod.build_loss("l1")
        got = evaluate_pixel_weighted(
            _ZeroModel(), val_loader, criterion, torch.device("cpu"), use_preupsample=False
        )
        expected = 0.68
        err = abs(got - expected)
        assert err < 1e-6, f"按像素加权结果 {got:.6f} != 期望 {expected:.6f}"
        return f"按像素加权验证损失={got:.6f}（期望 0.680000，误差={err:.2e}）"

    check("T-1", "单步前向+L1+反向+Adam更新", test_t1)
    check("T-2", "同种子两次20步loss逐位相等", test_t2)
    check("T-3/T-4", "断点续训连续性与checkpoint字段齐全", test_t3_and_t4)
    check("T-5", "跨模型/跨倍率载入checkpoint报错", test_t5)
    check("T-6", "验证路径不改变模型参数", test_t6)
    check("T-7", "验证损失按像素加权计算", test_t7)
    return records


def run_training_mutation_probe() -> tuple[str, bool, str]:
    """变异 T-9：把 build_loss('l1') 临时篡改为 MSE(L2)，断言 T-7 必须变红 FAIL。"""
    orig_build = losses_mod.build_loss
    try:
        losses_mod.build_loss = lambda name="l1": nn.MSELoss(reduction="mean")  # type: ignore[assignment]
        recs = run_training_tests()
        failed_codes = [r.code for r in recs if not r.passed]
        caught = "T-7" in failed_codes
        return "T-9(L1改L2)", caught, f"触发 FAIL 用例: {failed_codes}"
    finally:
        losses_mod.build_loss = orig_build


def main() -> int:
    print("=" * 72)
    print("阶段 5：训练链路测试（T-1 ~ T-7 + T-9 变异探针）")
    print("=" * 72)

    records = run_training_tests()
    passed = sum(1 for r in records if r.passed)
    failed = len(records) - passed
    for r in records:
        tag = "PASS" if r.passed else "FAIL"
        print(f"  [{tag}] {r.code} {r.name}: {r.detail}")
    print(f"\n汇总: 通过 {passed} / {len(records)}，失败 {failed}")

    if failed > 0:
        return 1

    print("\n[变异守卫探针（预期改坏后必须触发 FAIL）]：")
    m_name, caught, detail = run_training_mutation_probe()
    tag = "CAUGHT" if caught else "MISSED"
    print(f"  [{tag}] {m_name} -> {detail}")
    if not caught:
        return 1

    print("\nRESULT = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
