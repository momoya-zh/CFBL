# -*- coding: utf-8 -*-
"""超分辨率训练与验证 Dataset 实现。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.degradation import downsample_bicubic, make_lr_from_patch
from src.data.transforms import random_dihedral
from src.processing.resample import bicubic_upsample_pillow

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_image_paths(dirs: list[str | Path]) -> list[Path]:
    """按确定性字典序收集目录中的所有图像文件路径。"""
    paths: list[Path] = []
    for d in dirs:
        root = Path(d)
        if not root.is_dir():
            raise FileNotFoundError(f"数据集目录不存在: {root}")
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(p)
    return paths


def _hwc_uint8_to_chw_float32(img: np.ndarray) -> torch.Tensor:
    arr = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def _pad_hr_if_needed(hr_rgb: np.ndarray, patch_size: int) -> tuple[np.ndarray, int]:
    """当 HR 短边小于 patch_size 时使用 reflect 补边；无法补足时抛出 ValueError。"""
    h, w = hr_rgb.shape[:2]
    if h >= patch_size and w >= patch_size:
        return hr_rgb, 0
    if h < 2 or w < 2:
        raise ValueError(
            f"图像尺寸 ({h}, {w}) 过小，reflect 补边后仍不足 {patch_size}×{patch_size}"
        )
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h >= h or pad_w >= w:
        raise ValueError(
            f"图像尺寸 ({h}, {w}) 过小，单次 reflect 补边无法达到 {patch_size}×{patch_size}"
        )
    padded = np.pad(
        hr_rgb,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect",
    )
    if padded.shape[0] < patch_size or padded.shape[1] < patch_size:
        raise ValueError(
            f"补边后尺寸 {padded.shape[:2]} 仍不足 {patch_size}×{patch_size}"
        )
    added_pixels = (padded.shape[0] * padded.shape[1]) - (h * w)
    return padded, added_pixels


def _resolve_cache_path(hr_path: Path, cache_dir: Path | None) -> Path | None:
    """根据原图路径解析对应的 .npy 缓存路径（若存在）。"""
    if cache_dir is None:
        return None
    # 支持 DIV2K_train_HR/<stem>.npy 与 Flickr2K/HR -> Flickr2K_HR/<stem>.npy
    parent_name = hr_path.parent.name
    if parent_name == "HR" and hr_path.parent.parent.name == "Flickr2K":
        group_name = "Flickr2K_HR"
    else:
        group_name = parent_name
    candidate = cache_dir / group_name / f"{hr_path.stem}.npy"
    return candidate if candidate.is_file() else None


class SRDataset(Dataset):
    """支持 SRCNN（返回 3 元组）与 ESPCN（返回 2 元组）的超分辨率数据集。

    Args:
        hr_paths: 高分辨率图像路径列表。
        scale: 缩放倍率（``2`` 或 ``4``）。
        patch_size: 训练态 HR 随机裁剪尺寸（默认 ``128``）。
        base_seed: 随机种子基数（默认 ``42``）。
        training: ``True`` 为训练态（随机裁剪 + 二面体增强）；``False`` 为验证态（整图对齐）。
        use_preupsample: ``True`` 时（SRCNN）返回 ``(hr, lr, lr_up)``；否则返回 ``(hr, lr)``。
        cache_dir: 可选无压缩 ``.npy`` 缓存根目录，存在时训练态使用 ``mmap_mode='r'`` 切片读取。
        cache_val_in_memory: 验证态是否将退化后的 ``uint8`` 样本缓存在内存中以加速多轮验证。
    """

    def __init__(
        self,
        hr_paths: list[str | Path],
        scale: int,
        patch_size: int = 128,
        base_seed: int = 42,
        training: bool = True,
        use_preupsample: bool = False,
        cache_dir: str | Path | None = None,
        cache_val_in_memory: bool = True,
    ) -> None:
        if not hr_paths:
            raise ValueError("hr_paths 不能为空")
        if scale not in (2, 4):
            raise ValueError(f"不支持的倍率 scale={scale}，仅支持 2 或 4")
        if training and (patch_size <= 0 or patch_size % scale != 0):
            raise ValueError(
                f"patch_size={patch_size} 必须为正整数且能被 scale={scale} 整除"
            )

        self.hr_paths: list[Path] = [Path(p) for p in hr_paths]
        self.scale: int = scale
        self.patch_size: int = patch_size
        self.base_seed: int = base_seed
        self.training: bool = training
        self.use_preupsample: bool = use_preupsample
        self.cache_dir: Path | None = Path(cache_dir) if cache_dir is not None else None
        self.cache_paths: list[Path | None] = [
            _resolve_cache_path(p, self.cache_dir) for p in self.hr_paths
        ]
        self.cache_val_in_memory: bool = bool(cache_val_in_memory) and (not training)
        self._val_cache: dict[int, tuple[np.ndarray, ...]] = {}
        self.epoch: int = 0
        self.padded_count: int = 0
        self.padded_pixels_total: int = 0

    def set_epoch(self, epoch: int) -> None:
        """设置当前 Epoch，使每个 Epoch 的随机裁剪与增强可复现。"""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, index: int):
        if self.training:
            rng = np.random.default_rng(
                self.base_seed + self.epoch * 100003 + index
            )
            npy_path = self.cache_paths[index]
            if npy_path is not None:
                hr_mmap = np.load(npy_path, mmap_mode="r")
                h, w = hr_mmap.shape[:2]
                if h >= self.patch_size and w >= self.patch_size:
                    y0 = 0 if h == self.patch_size else int(rng.integers(0, h - self.patch_size + 1))
                    x0 = 0 if w == self.patch_size else int(rng.integers(0, w - self.patch_size + 1))
                    hr_patch = np.array(
                        hr_mmap[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size, :],
                        dtype=np.uint8,
                        copy=True,
                    )
                else:
                    hr_rgb = np.array(hr_mmap, dtype=np.uint8, copy=True)
                    hr_work, added_px = _pad_hr_if_needed(hr_rgb, self.patch_size)
                    if added_px > 0:
                        self.padded_count += 1
                        self.padded_pixels_total += added_px
                    hw, ww = hr_work.shape[:2]
                    y0 = 0 if hw == self.patch_size else int(rng.integers(0, hw - self.patch_size + 1))
                    x0 = 0 if ww == self.patch_size else int(rng.integers(0, ww - self.patch_size + 1))
                    hr_patch = hr_work[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size, :]
            else:
                path = self.hr_paths[index]
                with Image.open(path) as img:
                    hr_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
                hr_work, added_px = _pad_hr_if_needed(hr_rgb, self.patch_size)
                if added_px > 0:
                    self.padded_count += 1
                    self.padded_pixels_total += added_px
                h, w = hr_work.shape[:2]
                y0 = 0 if h == self.patch_size else int(rng.integers(0, h - self.patch_size + 1))
                x0 = 0 if w == self.patch_size else int(rng.integers(0, w - self.patch_size + 1))
                hr_patch = hr_work[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size, :]

            hr_aug = random_dihedral(hr_patch, rng)
            lr_patch = make_lr_from_patch(hr_aug, self.scale)

            hr_t = _hwc_uint8_to_chw_float32(hr_aug)
            lr_t = _hwc_uint8_to_chw_float32(lr_patch)
            if self.use_preupsample:
                # 在 uint8 域执行 Pillow BICUBIC 预上采样后再 /255.0，与推理域一致
                lr_up = bicubic_upsample_pillow(lr_patch, self.scale)
                lr_up_t = _hwc_uint8_to_chw_float32(lr_up)
                return hr_t, lr_t, lr_up_t
            return hr_t, lr_t

        # 验证态：整图路径，先按 scale 向零对齐裁掉右下角余数像素
        if self.cache_val_in_memory and index in self._val_cache:
            cached = self._val_cache[index]
            return tuple(_hwc_uint8_to_chw_float32(arr) for arr in cached)

        path = self.hr_paths[index]
        with Image.open(path) as img:
            hr_rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)

        h, w = hr_rgb.shape[:2]
        h_aligned = (h // self.scale) * self.scale
        w_aligned = (w // self.scale) * self.scale
        hr_aligned = hr_rgb[:h_aligned, :w_aligned, :]
        lr_img = downsample_bicubic(hr_aligned, self.scale)

        if self.use_preupsample:
            lr_up = bicubic_upsample_pillow(lr_img, self.scale)
            raw_tuple: tuple[np.ndarray, ...] = (hr_aligned, lr_img, lr_up)
        else:
            raw_tuple = (hr_aligned, lr_img)

        if self.cache_val_in_memory:
            self._val_cache[index] = raw_tuple

        return tuple(_hwc_uint8_to_chw_float32(arr) for arr in raw_tuple)
