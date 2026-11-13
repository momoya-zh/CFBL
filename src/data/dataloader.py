# -*- coding: utf-8 -*-
"""DataLoader 装配模块。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SRDataset, collect_image_paths
from src.data.sampler import SeededEpochSampler


def build_dataloader(
    cfg: dict[str, Any],
    training: bool,
    project_root: str | Path | None = None,
) -> tuple[DataLoader, SeededEpochSampler | None]:
    """根据训练配置字典装配训练或验证 DataLoader。"""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    data_cfg = cfg["data"]
    scale = int(cfg["scale"])
    patch_size = int(data_cfg.get("patch_size", 128))
    seed = int(cfg.get("seed", 42))
    use_preupsample = bool(cfg.get("use_preupsample", False))
    num_workers = int(data_cfg.get("num_workers", 4)) if training else int(data_cfg.get("val_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", True)) and torch.cuda.is_available()

    if training:
        train_dirs = [root / d for d in data_cfg["train_dirs"]]
        val_dirs = [root / d for d in data_cfg["val_dirs"]]
        train_paths = collect_image_paths(train_dirs)
        val_paths = collect_image_paths(val_dirs)
        overlap = {p.resolve() for p in train_paths} & {p.resolve() for p in val_paths}
        if overlap:
            raise RuntimeError(f"训练集与验证集路径存在交集: {len(overlap)} 个文件")

        default_cache = root / "datasets" / "_cache_npy"
        cache_dir_cfg = data_cfg.get("cache_dir")
        cache_dir = (
            (root / cache_dir_cfg)
            if cache_dir_cfg
            else (default_cache if default_cache.is_dir() else None)
        )

        dataset = SRDataset(
            hr_paths=train_paths,
            scale=scale,
            patch_size=patch_size,
            base_seed=seed,
            training=True,
            use_preupsample=use_preupsample,
            cache_dir=cache_dir,
        )
        sampler = SeededEpochSampler(len(dataset), base_seed=seed)
        batch_size = int(cfg["train"]["batch_size"])
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )
        return loader, sampler

    val_dirs = [root / d for d in data_cfg["val_dirs"]]
    val_paths = collect_image_paths(val_dirs)
    dataset = SRDataset(
        hr_paths=val_paths,
        scale=scale,
        patch_size=patch_size,
        base_seed=seed,
        training=False,
        use_preupsample=use_preupsample,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader, None
