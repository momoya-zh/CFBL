# -*- coding: utf-8 -*-
"""基于 Epoch 种子的可复现采样器。"""
from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import Sampler


class SeededEpochSampler(Sampler[int]):
    """每个 Epoch 使用 ``base_seed + epoch`` 生成确定性随机置换索引。"""

    def __init__(self, data_source_len: int, base_seed: int = 42) -> None:
        super().__init__()
        if data_source_len <= 0:
            raise ValueError(f"data_source_len 必须为正整数，实际为 {data_source_len}")
        self.data_source_len = int(data_source_len)
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)
        perm = torch.randperm(self.data_source_len, generator=g).tolist()
        return iter(perm)

    def __len__(self) -> int:
        return self.data_source_len
