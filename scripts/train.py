# -*- coding: utf-8 -*-
"""阶段 5 训练入口脚本。

运行方式（CMD）::

    "D:\\Environment\\Python\\Python_AI\\python.exe" scripts\\train.py --config configs\\train\\espcn_x2.yaml --epochs 5
    "D:\\Environment\\Python\\Python_AI\\python.exe" scripts\\train.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import Trainer  # noqa: E402

ALL_CONFIGS = [
    "configs/train/srcnn_x2.yaml",
    "configs/train/srcnn_x4.yaml",
    "configs/train/espcn_x2.yaml",
    "configs/train/espcn_x4.yaml",
]


def _run_single(
    config_path: Path,
    epochs_override: int | None,
    resume: bool,
    startup_cmd: str,
) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    trainer = Trainer(cfg=cfg, project_root=PROJECT_ROOT, startup_cmd=startup_cmd)
    resume_path = None
    if resume:
        candidate = trainer.ckpt_dir / "latest.pth"
        if not candidate.is_file():
            raise FileNotFoundError(f"未找到断点文件: {candidate}")
        resume_path = candidate
    trainer.fit(max_epochs=epochs_override, resume_path=resume_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段 5 超分辨率模型训练入口")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="单个训练 YAML 配置文件路径")
    group.add_argument("--all", action="store_true", help="依次训练全部 4 套配置")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练 Epoch 数（用于冒烟测试）")
    parser.add_argument("--resume", action="store_true", help="从 latest.pth 断点续训")
    args = parser.parse_args()

    startup_cmd = " ".join(sys.argv)
    if args.all:
        for rel in ALL_CONFIGS:
            _run_single(PROJECT_ROOT / rel, args.epochs, args.resume, startup_cmd)
    else:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = PROJECT_ROOT / cfg_path
        _run_single(cfg_path, args.epochs, args.resume, startup_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
