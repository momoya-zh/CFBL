# -*- coding: utf-8 -*-
"""二面体群（D4）数据增强模块。

包含水平翻转、垂直翻转与 90° 旋转的组合，共 8 种等距对称变换。
"""
from __future__ import annotations

import numpy as np


def apply_dihedral(img: np.ndarray, op_id: int) -> np.ndarray:
    """对 (H, W, C) 图像施加指定编号（0..7）的二面体变换。"""
    if not (0 <= op_id < 8):
        raise ValueError(f"op_id 必须在 0..7 之间，实际为 {op_id}")
    k_rot = op_id % 4
    do_flip = (op_id // 4) == 1
    out = np.rot90(img, k=k_rot, axes=(0, 1))
    if do_flip:
        out = out[:, ::-1, :]
    return np.ascontiguousarray(out)


def random_dihedral(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """使用独立 numpy.random.Generator 随机施加 8 种二面体变换之一。"""
    op_id = int(rng.integers(0, 8))
    return apply_dihedral(img, op_id)
