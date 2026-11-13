# -*- coding: utf-8 -*-
"""统一超分辨率推理模块。"""

from src.inference.predictor import predict_rgb_image, predict_tensor

__all__ = [
    "predict_rgb_image",
    "predict_tensor",
]
