# -*- coding: utf-8 -*-
"""BUSI 分割数据集加载器（支持 LR 退化、自训 SR 重建与课本 DIP 预处理分支）。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.degradation import downsample_bicubic
from src.inference.predictor import predict_rgb_image
from src.processing.dip_enhancement import clahe_enhancement, median_filter
from src.processing.resample import bicubic_upsample_pillow


class BUSISegDataset(Dataset):
    """BUSI 消融实验数据集，支持在内存缓存预处理后的 256x256 图像。"""

    def __init__(
        self,
        split_dir: Path,
        pipeline_mode: str = "baseline_bicubic",
        image_size: int = 256,
        sr_model: torch.nn.Module | None = None,
        sr_scale: int = 2,
        device: torch.device | str | None = None,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.img_dir = split_dir / "images"
        self.mask_dir = split_dir / "masks"
        self.image_paths = sorted(self.img_dir.glob("*.png"))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"未在 {self.img_dir} 找到 PNG 图像")

        self.image_size = image_size
        self.augment = augment
        self.rng = random.Random(seed)
        self.samples: list[tuple[np.ndarray, np.ndarray, str]] = []

        for img_path in self.image_paths:
            mask_path = self.mask_dir / img_path.name
            hr = np.asarray(
                Image.open(img_path)
                .convert("RGB")
                .resize((image_size, image_size), Image.Resampling.BICUBIC),
                dtype=np.uint8,
            )
            mask = np.asarray(
                Image.open(mask_path)
                .convert("L")
                .resize((image_size, image_size), Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
            mask_bin = (mask > 127).astype(np.uint8)

            # 统一构造 2x 退化低分辨率超声图 (128x128)
            lr = downsample_bicubic(hr, sr_scale)

            if pipeline_mode == "baseline_bicubic":
                proc = bicubic_upsample_pillow(lr, sr_scale)
            elif pipeline_mode == "dip_only":
                up = bicubic_upsample_pillow(lr, sr_scale)
                proc = clahe_enhancement(median_filter(up, ksize=3), clip_limit=2.0)
            elif pipeline_mode == "sr_only":
                if sr_model is None:
                    raise ValueError("pipeline_mode='sr_only' 需要提供 sr_model")
                proc = predict_rgb_image(sr_model, lr, sr_scale, device=device)
            elif pipeline_mode == "sr_plus_dip":
                if sr_model is None:
                    raise ValueError("pipeline_mode='sr_plus_dip' 需要提供 sr_model")
                up = predict_rgb_image(sr_model, lr, sr_scale, device=device)
                proc = clahe_enhancement(median_filter(up, ksize=3), clip_limit=2.0)
            else:
                raise ValueError(f"未知 pipeline_mode: {pipeline_mode}")

            self.samples.append((proc, mask_bin, img_path.name))

        if sum(int(m.sum()) for _, m, _ in self.samples) == 0:
            raise ValueError(f"数据集 {split_dir} 的前景掩膜像素总数为 0，请检查掩膜阈值或源文件")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, mask, _ = self.samples[idx]
        if self.augment and self.rng.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])

        img_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask[None, :, :], dtype=np.float32))
        return img_t, mask_t
