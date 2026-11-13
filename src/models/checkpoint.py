# -*- coding: utf-8 -*-
"""模型权重保存与加载校验工具（绑定 model_name 与 scale）。"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.models.espcn import ESPCN
from src.models.srcnn import SRCNN


def _get_model_name(model: nn.Module) -> str:
    if isinstance(model, SRCNN):
        return "srcnn"
    if isinstance(model, ESPCN):
        return "espcn"
    raise TypeError(f"不支持的模型类型: {type(model).__name__}")


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    extra: dict[str, object] | None = None,
) -> Path:
    """保存包含 model_name 与 scale 元信息的模型权重。"""
    model_name = _get_model_name(model)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model_name": model_name,
        "scale": int(model.scale),
        "state_dict": model.state_dict(),
    }
    if extra:
        payload["extra"] = dict(extra)
    torch.save(payload, target)
    return target


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    """加载模型权重并强制校验 model_name 与 scale 一致性，防止跨倍率静默互载。"""
    model_name = _get_model_name(model)
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("无效的 checkpoint 格式：缺少 state_dict")
    ckpt_name = ckpt.get("model_name")
    ckpt_scale = ckpt.get("scale")
    if ckpt_name != model_name:
        raise ValueError(
            f"模型类型不匹配: checkpoint 为 {ckpt_name!r}，目标模型为 {model_name!r}"
        )
    if ckpt_scale != model.scale:
        raise ValueError(
            f"放大倍率不匹配: checkpoint 为 scale={ckpt_scale}，目标模型为 scale={model.scale}"
        )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return ckpt
