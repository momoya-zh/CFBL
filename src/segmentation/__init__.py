# -*- coding: utf-8 -*-
"""医学超声图像分割模块（轻量化 U-Net 与受控消融训练器）。"""

from src.segmentation.dataset import BUSISegDataset
from src.segmentation.trainer import SegmentationResult, train_and_evaluate_segmentation
from src.segmentation.unet import UNet

__all__ = [
    "BUSISegDataset",
    "SegmentationResult",
    "UNet",
    "train_and_evaluate_segmentation",
]
