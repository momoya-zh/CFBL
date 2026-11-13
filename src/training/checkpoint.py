# -*- coding: utf-8 -*-
"""训练态 checkpoint 持久化与断点续训校验模块。"""
from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.models.espcn import ESPCN
from src.models.srcnn import SRCNN


def _infer_model_name(model: nn.Module) -> str:
    if isinstance(model, SRCNN):
        return "srcnn"
    if isinstance(model, ESPCN):
        return "espcn"
    raise TypeError(f"不支持的模型类型: {type(model).__name__}")


def collect_env_snapshot() -> dict[str, str]:
    """采集当前 Python / PyTorch / CUDA / GPU 环境快照。"""
    gpu_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda or "none"),
        "gpu": gpu_name,
    }


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val: float,
    seed: int,
    config: dict[str, Any],
    history: list[dict[str, Any]],
) -> Path:
    """保存包含完整训练状态与元信息的 checkpoint。"""
    model_name = _infer_model_name(model)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_name": model_name,
        "scale": int(model.scale),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val": float(best_val),
        "seed": int(seed),
        "config": dict(config),
        "env": collect_env_snapshot(),
        "history": list(history),
    }
    torch.save(payload, target)
    return target


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """加载训练态 checkpoint 并校验 model_name 与 scale 一致性。"""
    model_name = _infer_model_name(model)
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("无效的训练 checkpoint 格式：缺少 state_dict")
    if ckpt.get("model_name") != model_name:
        raise ValueError(
            f"模型类型不匹配: checkpoint 为 {ckpt.get('model_name')!r}，目标模型为 {model_name!r}"
        )
    if ckpt.get("scale") != model.scale:
        raise ValueError(
            f"放大倍率不匹配: checkpoint 为 scale={ckpt.get('scale')}，目标模型为 scale={model.scale}"
        )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    if optimizer is not None:
        if "optimizer" not in ckpt:
            raise ValueError("checkpoint 缺少 optimizer 状态，无法恢复训练")
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
